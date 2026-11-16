"""
Compare saved FQE estimates across SOFA severity groups.

Joins raw official SOFA to features by (stay_id, time_bin). Missing SOFA
is reported separately. Uses the per-stay v0.npz arrays from completed
FQE fits without refitting the evaluator.

Writes <OUTPUT_BASE>/subgroup/final_subgroup.json.
"""
import json
import os

import numpy as np
import pandas as pd

from config.paths import OUTPUT_BASE, PROCESSED_DIR
from src.final_features import FINAL_FEATURES_DIR
from src.utils import ensure_dir

SEEDS = (42, 43, 44, 45, 46)
CQL_ALPHA = 2.0
# Conventional severity bands; the top band is where the treatment decision
# actually bites and where support is thinnest.
SOFA_BINS = [(-0.5, 5, "0-5"), (5, 10, "6-10"), (10, 100, "11+")]
OUT_DIR = os.path.join(OUTPUT_BASE, "subgroup")


def _stratum(raw: np.ndarray) -> np.ndarray:
    idx = np.full(len(raw), -1, dtype=int)
    for k, (lo, hi, _) in enumerate(SOFA_BINS):
        idx[(raw > lo) & (raw <= hi)] = k
    return idx


def _v0(path: str) -> pd.DataFrame | None:
    """Per-stay initial-state value from a completed FQE fit."""
    f = os.path.join(path, "v0.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f)
    return pd.DataFrame({"stay_id": z["stay_id"], "v0": z["v0"]})


def fqe_dir(kind: str, split: str, checkpoint: str) -> str:
    """
    Which FQE artefact directory holds this family's values.

    IQL's policy is its actor, so the Q-net checkpoint rule does not apply to
    it; its final variant is the actor re-extracted against the epoch-40
    critic and is named for that. Mapping it to the plain directory made the
    final subgroup table silently repeat the early one.
    """
    if kind == "iql":
        return f"fqe_{split}" if checkpoint == "best" else f"fqe_{split}_criticfinal"
    return f"fqe_{split}" + ("" if checkpoint == "best" else "_finalckpt")


def run(split: str = "test", checkpoint: str = "final") -> dict:

    proc = pd.read_parquet(os.path.join(PROCESSED_DIR, f"{split}.parquet"),
                           columns=["stay_id", "time_bin", "action",
                                    "icu_mortality"])
    feats = pd.read_parquet(
        os.path.join(FINAL_FEATURES_DIR, "features.parquet"),
        columns=["stay_id", "time_bin", "sofa_total", "sofa_observed"])
    df = proc.merge(feats, on=["stay_id", "time_bin"], how="left")
    # The feature table is read from disk by path, and it has been rebuilt more
    # than once during this project. A left merge against a table built for a
    # different dataset would quietly stratify on nothing. `sofa_observed` is
    # never null in a matching row, so a null here means the key was absent.
    unmatched = int(df["sofa_observed"].isna().sum())
    if unmatched:
        raise AssertionError(
            f"{unmatched:,} of {len(df):,} {split} rows have no row in "
            f"{FINAL_FEATURES_DIR}/features.parquet. That feature table was "
            f"built for a different dataset than {PROCESSED_DIR}.")

    # Initial state per stay defines the stratum a patient is scored in
    # head(1), not first(). Pandas `first()` takes the first non-null value of
    # each column independently, so it can pair `sofa_observed = 0` from the
    # initial row with a `sofa_total` recorded hours later. The initial state
    # has to be one actual row.
    init = (df.sort_values(["stay_id", "time_bin"])
              .groupby("stay_id", sort=False, as_index=False).head(1)
              .reset_index(drop=True))
    init["stratum"] = _stratum(init["sofa_total"].values)
    observed = init["sofa_observed"] == 1

    out: dict = {
        "split": split, "checkpoint": checkpoint,
        "sofa_source": "official mimiciv_3_1_derived.sofa, raw, joined on (stay_id, time_bin)",
        "stays_total": int(len(init)),
        "stays_sofa_observed": int(observed.sum()),
        "stays_sofa_missing": int((~observed).sum()),
        "pct_sofa_observed": round(100 * float(observed.mean()), 2),
        "strata": [b[2] for b in SOFA_BINS],
    }

    strat = init.loc[observed, ["stay_id", "stratum", "icu_mortality"]]
    out["stratum_sizes"] = {SOFA_BINS[k][2]: int((strat["stratum"] == k).sum())
                            for k in range(len(SOFA_BINS))}
    out["stratum_mortality_pct"] = {
        SOFA_BINS[k][2]: round(
            100 * float(strat.loc[strat["stratum"] == k, "icu_mortality"].mean()), 2)
        for k in range(len(SOFA_BINS))}

    # Policies to compare, each already scored under the same FQE protocol
    # "behaviour_est" is the MLP fitted to the logged actions, not the
    # logged clinician. The directory keeps its historical name.
    policies = {"behaviour_est": os.path.join(OUTPUT_BASE, "fqe_clinician", f"fqe_{split}"),
                "bc": os.path.join(OUTPUT_BASE, "fqe_bc", f"fqe_{split}"),
                "withhold": os.path.join(OUTPUT_BASE, "constant_action0", f"fqe_{split}")}
    for kind, pat in (("cql", f"cql_a{CQL_ALPHA}_seed"), ("d3qn", "d3qn_seed"),
                      ("iql", "iql_seed")):
        for s in SEEDS:
            policies[f"{kind}_seed{s}"] = os.path.join(
                OUTPUT_BASE, f"{pat}{s}", fqe_dir(kind, split, checkpoint))

    values: dict = {}
    for name, path in policies.items():
        v = _v0(path)
        if v is None:
            continue
        m = strat.merge(v, on="stay_id", how="left")
        per = {}
        for k, (_, _, lab) in enumerate(SOFA_BINS):
            sel = m.loc[m["stratum"] == k, "v0"].dropna()
            per[lab] = {"n": int(len(sel)),
                        "mean": round(float(sel.mean()), 4) if len(sel) else None}
        per["all"] = {"n": int(m["v0"].notna().sum()),
                      "mean": round(float(m["v0"].mean()), 4)}
        values[name] = per
    out["fqe_by_stratum"] = values

    # SOFA is missing at the initial state for roughly a third of stays,
    # because the derived grid starts at ICU admission rather than at onset.
    # Those stays are excluded from the severity bands above, so they are
    # reported here as their own group: not a fourth severity level, a
    # missingness group. Without it the ordering could differ on a third of the
    # cohort and nothing would say so.
    miss_ids = set(init.loc[~observed, "stay_id"])
    missing = {"n_stays": int(len(miss_ids)),
               "pct_of_stays": round(100 * len(miss_ids) / max(len(init), 1), 2),
               "mortality_90d_pct": round(
                   100 * float(init.loc[~observed, "icu_mortality"].mean()), 2)
               if len(miss_ids) else None,
               "note": ("stays whose initial decision point has no observed "
                        "SOFA. A missingness group, not a severity band."),
               "fqe_mean": {}}
    for name, path in policies.items():
        v = _v0(path)
        if v is None:
            continue
        sel = v.loc[v["stay_id"].isin(miss_ids), "v0"].dropna()
        missing["fqe_mean"][name] = (round(float(sel.mean()), 4)
                                     if len(sel) else None)
    # Between-seed spread for each learned method over the same stays
    import numpy as _np
    for kind in ("cql", "d3qn", "iql"):
        vals = [missing["fqe_mean"].get(f"{kind}_seed{s}") for s in SEEDS]
        vals = [x for x in vals if x is not None]
        if len(vals) > 1:
            missing["fqe_mean"][f"{kind}_mean"] = round(float(_np.mean(vals)), 4)
            missing["fqe_mean"][f"{kind}_sd"] = round(
                float(_np.std(vals, ddof=1)), 4)
    out["sofa_missing_at_initial_state"] = missing

    # Every core method must contribute all five seeds. Aggregating whatever
    # files happen to exist would let a missing seed shrink a subgroup mean
    # without saying so.
    missing = {kind: [s for s in SEEDS if f"{kind}_seed{s}" not in values]
               for kind in ("cql", "d3qn", "iql")}
    missing = {k: v for k, v in missing.items() if v}
    # Recorded now, raised at the end, so the partial table reaches disk for
    # inspection before the step reports its failure.
    out["seeds_expected"] = len(SEEDS)
    out["seeds_missing"] = missing
    out["complete"] = not missing

    # Seed-averaged view, which is what the thesis reports
    agg: dict = {}
    for kind in ("cql", "d3qn", "iql"):
        keys = [k for k in values if k.startswith(f"{kind}_seed")]
        if not keys:
            continue
        agg[kind] = {}
        for lab in [b[2] for b in SOFA_BINS] + ["all"]:
            vals = [values[k][lab]["mean"] for k in keys
                    if values[k][lab]["mean"] is not None]
            agg[kind][lab] = {"mean": round(float(np.mean(vals)), 4),
                              "sd": round(float(np.std(vals, ddof=1)), 4)
                              if len(vals) > 1 else None,
                              "n_seeds": len(vals)}
    for ref in ("behaviour_est", "bc", "withhold"):
        if ref in values:
            agg[ref] = {lab: {"mean": values[ref][lab]["mean"], "sd": None,
                              "n_seeds": 1}
                        for lab in [b[2] for b in SOFA_BINS] + ["all"]}
    out["fqe_by_stratum_seedmean"] = agg

    # Action agreement with the clinician, per stratum, at every decision point
    dfo = df[df["sofa_observed"] == 1].copy()
    dfo["stratum"] = _stratum(dfo["sofa_total"].values)
    out["decision_points_by_stratum"] = {
        SOFA_BINS[k][2]: int((dfo["stratum"] == k).sum())
        for k in range(len(SOFA_BINS))}
    out["clinician_zero_action_pct_by_stratum"] = {
        SOFA_BINS[k][2]: round(
            100 * float((dfo.loc[dfo["stratum"] == k, "action"] == 0).mean()), 2)
        for k in range(len(SOFA_BINS))}

    ensure_dir(OUT_DIR)
    p = os.path.join(OUT_DIR, f"final_subgroup_{split}_{checkpoint}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"  -> {p}")

    print(f"\nSOFA strata ({split}, {checkpoint} checkpoint, "
          f"{out['pct_sofa_observed']}% of stays have an observed SOFA)")
    hdr = f"{'policy':<14}" + "".join(f"{b[2]:>12}" for b in SOFA_BINS) + f"{'all':>12}"
    print(hdr)
    for name in ("behaviour_est", "withhold", "bc", "cql", "d3qn", "iql"):
        if name not in agg:
            continue
        row = f"{name:<14}"
        for lab in [b[2] for b in SOFA_BINS] + ["all"]:
            v = agg[name][lab]["mean"]
            row += f"{v:>12.3f}" if v is not None else f"{'-':>12}"
        print(row)

    m = out.get("sofa_missing_at_initial_state", {})
    if m.get("n_stays"):
        print(f"\nSOFA missing at the initial decision point: "
              f"{m['n_stays']:,} stays ({m['pct_of_stays']}%), "
              f"90-day mortality {m['mortality_90d_pct']}%. "
              f"A missingness group, not a severity band.")
        for kind in ("cql", "d3qn", "iql"):
            mu = m["fqe_mean"].get(f"{kind}_mean")
            sd = m["fqe_mean"].get(f"{kind}_sd")
            if mu is not None:
                print(f"  {kind:<6}{mu:>10.3f}" +
                      (f" ({sd:.3f})" if sd is not None else ""))
        for ref in ("behaviour_est", "withhold", "bc"):
            v = m["fqe_mean"].get(ref)
            if v is not None:
                print(f"  {ref:<6}{v:>10.3f}")

    if out.get("seeds_missing"):
        raise AssertionError(
            f"subgroup [{split}/{checkpoint}] is missing seeds "
            f"{out['seeds_missing']}. Refusing to report a stratified "
            f"comparison over an incomplete set of runs. The partial table is "
            f"at {p} for inspection.")
    return out


if __name__ == "__main__":
    import sys
    errs = []
    for ck in ("final", "best"):
        try:
            run("test", ck)
        except AssertionError as e:
            errs.append(str(e))
    if errs:
        print("\n" + "\n".join(errs), file=sys.stderr)
        sys.exit(1)
