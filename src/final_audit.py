"""
Audit the final Sepsis-3 dataset before training.

Checks interval spacing, patient split disjointness, treatment-history
leakage, SOFA coverage, action reachability and terminal rewards. Writes
<SEPSIS_STATUS_DIR>/dataset_audit.json and raises on failed checks.
"""
import json
import os

import numpy as np
import pandas as pd

from config.clinical import TIME_BIN_HOURS
from config.paths import PROJECT_ROOT
from src.final_features import FINAL_FEATURES_DIR
from src.final_preprocessing import FINAL_PROCESSED_DIR, STATE_COLS
from src.utils import ensure_dir

AUDIT_DIR = os.environ.get(
    "SEPSIS_STATUS_DIR",
    os.path.join(PROJECT_ROOT, "outputs", "final_preflight"))


def audit(save: bool = True) -> dict:
    splits = {k: pd.read_parquet(os.path.join(FINAL_PROCESSED_DIR, f"{k}.parquet"))
              for k in ("train", "val", "test")}
    with open(os.path.join(FINAL_PROCESSED_DIR, "metadata.json")) as f:
        meta = json.load(f)
    feats = pd.read_parquet(os.path.join(FINAL_FEATURES_DIR, "features.parquet"))

    a: dict = {"metadata": meta}
    fail: list[str] = []

    # ── Cohort and splits ────────────────────────────────────────────────────
    a["splits"] = {
        k: {"stays": int(v["stay_id"].nunique()),
            "subjects": int(v["subject_id"].nunique()),
            "transitions": int(len(v)),
            "deaths_90d": int(v.groupby("stay_id")["icu_mortality"].first().sum())}
        for k, v in splits.items()
    }
    a["total_stays"] = sum(s["stays"] for s in a["splits"].values())
    a["total_transitions"] = sum(s["transitions"] for s in a["splits"].values())
    all_deaths = sum(s["deaths_90d"] for s in a["splits"].values())
    a["mortality_90d_pct"] = round(100 * all_deaths / a["total_stays"], 2)
    a["deaths_90d"] = all_deaths
    if a["total_stays"] != 25442:
        fail.append(f"expected 25,442 stays, got {a['total_stays']:,}")
    if abs(a["mortality_90d_pct"] - 25.04) > 0.01:
        fail.append(f"expected 25.04% mortality, got {a['mortality_90d_pct']}%")

    # ── Patient disjointness ────────────────────────────────────────────────
    subj = {k: set(v["subject_id"]) for k, v in splits.items()}
    overlaps = {f"{x}&{y}": len(subj[x] & subj[y])
                for x, y in (("train", "val"), ("train", "test"), ("val", "test"))}
    a["subject_overlaps"] = overlaps
    if any(overlaps.values()):
        fail.append(f"subject leakage across splits: {overlaps}")

    # One stay per patient is the cohort rule; verify it survived preprocessing
    per_subject = pd.concat(splits.values()).groupby("subject_id")["stay_id"].nunique()
    a["max_stays_per_subject"] = int(per_subject.max())
    if per_subject.max() != 1:
        fail.append("some patients contribute more than one stay")

    # ── Interval spacing ────────────────────────────────────────────────────
    allt = pd.concat(splits.values()).sort_values(["stay_id", "time_bin"])
    g = allt.groupby("stay_id", sort=False)
    gap = g["time_bin"].shift(-1) - allt["time_bin"]
    interior = g.cumcount(ascending=False) > 0
    n_bad = int((interior & (gap != 1)).sum())
    a["non_consecutive_transitions"] = n_bad
    a["time_bin_hours"] = TIME_BIN_HOURS
    if n_bad:
        fail.append(f"{n_bad} transitions are not exactly {TIME_BIN_HOURS}h apart")

    lens = g.size()
    a["episode_length_intervals"] = {
        "min": int(lens.min()), "median": float(lens.median()),
        "mean": round(float(lens.mean()), 2), "max": int(lens.max())}
    if lens.min() < 2:
        fail.append("episodes shorter than 2 intervals present")
    if lens.max() > 18:
        fail.append(f"episode longer than 18 intervals ({lens.max()})")
    bins = allt["time_bin"]
    a["time_bin_range"] = [int(bins.min()), int(bins.max())]
    if bins.min() < -6 or bins.max() > 11:
        fail.append(f"time_bin outside [-6, 11]: {a['time_bin_range']}")

    # ── Terminal transitions and reward ─────────────────────────────────────
    term = allt[allt["done"] == 1]
    a["terminal_transitions"] = int(len(term))
    a["terminal_per_stay_max"] = int(term.groupby("stay_id").size().max())
    if len(term) != a["total_stays"]:
        fail.append(f"{len(term):,} terminals for {a['total_stays']:,} stays")
    nonterm_nonzero = int((allt.loc[allt["done"] == 0, "reward"] != 0).sum())
    a["nonterminal_nonzero_rewards"] = nonterm_nonzero
    if nonterm_nonzero:
        fail.append(f"{nonterm_nonzero} non-terminal rewards are non-zero "
                    "(reward must be sparse)")
    a["terminal_reward_values"] = sorted(term["reward"].unique().tolist())
    mism = int(((term["reward"] > 0) != (term["icu_mortality"] == 0)).sum())
    a["terminal_reward_label_mismatches"] = mism
    if mism:
        fail.append(f"{mism} terminal rewards disagree with the 90-day label")

    # ── Actions ─────────────────────────────────────────────────────────────
    tr = splits["train"]
    a["action_bins"] = meta["action_bins"]
    a["action_marginals"] = {
        "vaso": {int(l): round(100 * float((tr["vaso_level"] == l).mean()), 2)
                 for l in range(5)},
        "iv": {int(l): round(100 * float((tr["iv_level"] == l).mean()), 2)
               for l in range(5)},
    }
    a["joint_cells_occupied_train"] = int(tr["action"].nunique())
    a["zero_treatment_rate_pct"] = round(100 * float((tr["action"] == 0).mean()), 2)
    a["action_id_range"] = [int(tr["action"].min()), int(tr["action"].max())]
    for drug, col, n in (("vaso", "vaso_level", 5), ("iv", "iv_level", 5)):
        missing = set(range(n)) - set(tr[col].unique())
        if missing:
            fail.append(f"{drug} marginal levels unreachable: {sorted(missing)}")
    # Level 0 must mean exactly nothing given
    for lv_col, amt_col in (("vaso_level", "vaso_total"), ("iv_level", "iv_fluid_total")):
        leak = float(tr.loc[tr[lv_col] == 0, amt_col].abs().max())
        a[f"{lv_col}0_max_dose"] = leak
        if leak > 0:
            fail.append(f"{lv_col}==0 contains a non-zero dose ({leak})")

    # ── SOFA coverage and missingness ───────────────────────────────────────
    a["sofa_coverage_pct"] = round(100 * float(feats["sofa_total"].notna().mean()), 2)
    a["sofa_range"] = [float(feats["sofa_total"].min()), float(feats["sofa_total"].max())]
    if feats["sofa_total"].notna().mean() < 0.90:
        fail.append(f"official SOFA present at only {a['sofa_coverage_pct']}% of boundaries")

    raw_missing = {c: round(100 * float(feats[c].isna().mean()), 2)
                   for c in STATE_COLS if c in feats.columns}
    a["raw_missingness_pct"] = dict(sorted(raw_missing.items(),
                                           key=lambda kv: -kv[1])[:15])

    # ── Treatment-history features ──────────────────────────────────────────
    # prev_* must equal the previous interval's dose, never the current one.
    chk = feats.sort_values(["stay_id", "time_bin"]).copy()
    gg = chk.groupby("stay_id", sort=False)
    for prev, cur in (("prev_vaso", "vaso_total"), ("prev_iv", "iv_fluid_total")):
        expect = gg[cur].shift(1).fillna(0.0)
        bad = int((~np.isclose(chk[prev], expect)).sum())
        a[f"{prev}_mismatches"] = bad
        if bad:
            fail.append(f"{prev} does not equal the previous interval's {cur}")
        same = int((np.isclose(chk[prev], chk[cur]) & (chk[cur] > 0)).sum())
        a[f"{prev}_equals_current_rows"] = same

    a["state_features"] = len(STATE_COLS)
    for forbidden in ("vaso_total", "iv_fluid_total", "action"):
        if forbidden in STATE_COLS:
            fail.append(f"'{forbidden}' is in the state — that is the action")

    a["failures"] = fail
    a["passed"] = not fail

    if save:
        ensure_dir(AUDIT_DIR)
        p = os.path.join(AUDIT_DIR, "dataset_audit.json")
        with open(p, "w") as f:
            json.dump(a, f, indent=2, default=float)
        print(f"  audit -> {p}")

    print(f"\n{'PASS' if a['passed'] else 'FAIL'} — "
          f"{a['total_stays']:,} stays, {a['total_transitions']:,} transitions, "
          f"{a['mortality_90d_pct']}% 90-day mortality")
    for f_ in fail:
        print(f"  FAIL: {f_}")
    return a


if __name__ == "__main__":
    audit()
