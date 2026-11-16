"""
Collect FQE results for both checkpoints and all five training seeds.

Reports seed means, standard deviations and comparisons with the estimated
behaviour policy, behaviour cloning and always-withhold control. Checkpoint
definitions accompany the table. Missing runs cause the collector to fail
after writing partial results for inspection.

FQE values estimate discounted return. The behaviour reference is a fitted
policy, not the observed clinician return.
"""
import json
import os

import numpy as np
import pandas as pd

from config.paths import OUTPUT_BASE, PROCESSED_DIR
from src.utils import ensure_dir

SEEDS = (42, 43, 44, 45, 46)
CQL_ALPHA = 2.0
METHODS = {"cql": f"cql_a{CQL_ALPHA}_seed", "d3qn": "d3qn_seed", "iql": "iql_seed"}
# Keys are the REPORTED names; values are the directories. The reference in
# fqe_clinician is the estimated behaviour policy, an MLP fitted to the
# logged actions, not the logged clinician. Calling it "clinician" in a
# comparison column claims something the artefact does not support.
REFS = {"behaviour_est": "fqe_clinician", "bc": "fqe_bc",
        "withhold": "constant_action0"}


def _fqe(path: str) -> float | None:
    f = os.path.join(path, "fqe_result.json")
    if not os.path.exists(f):
        return None
    return float(json.load(open(f))["fqe_value"])


def _v0(path: str):
    f = os.path.join(path, "v0.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f)
    return pd.Series(z["v0"], index=z["stay_id"])


def collect(split: str = "test") -> pd.DataFrame:
    rows = []
    for kind, pat in METHODS.items():
        for seed in SEEDS:
            run = os.path.join(OUTPUT_BASE, f"{pat}{seed}")
            for ckpt, sfx in (("best", ""), ("final", "_finalckpt")):
                # IQL's policy is its actor; the Q-net rule does not apply, and
                # its final-critic variant lives under its own suffix
                if kind == "iql":
                    d = os.path.join(run, f"fqe_{split}" if ckpt == "best"
                                     else f"fqe_{split}_criticfinal")
                else:
                    d = os.path.join(run, f"fqe_{split}{sfx}")
                v = _fqe(d)
                if v is not None:
                    rows.append({"method": kind, "seed": seed,
                                 "checkpoint": ckpt, "split": split,
                                 "fqe": v})
    for name, sub in REFS.items():
        v = _fqe(os.path.join(OUTPUT_BASE, sub, f"fqe_{split}"))
        if v is not None:
            for ckpt in ("best", "final"):
                rows.append({"method": name, "seed": None, "checkpoint": ckpt,
                             "split": split, "fqe": v})
    return pd.DataFrame(rows)


def paired_vs_reference(split: str, ckpt: str, method: str,
                        reference: str) -> dict | None:
    """
    Seed-level paired comparison against a reference, on shared stays.

    Paired on stay_id rather than position, so two fits can be compared without
    assuming identical row order. The test is across seeds (n=5, df=4), which
    is the unit of replication here — the per-stay spread is dispersion of a
    single fit, not uncertainty about the method.
    """
    from scipy import stats
    ref_dir = os.path.join(OUTPUT_BASE, REFS[reference], f"fqe_{split}")
    ref = _v0(ref_dir)
    if ref is None:
        return None
    pat = METHODS[method]
    diffs = []
    for seed in SEEDS:
        run = os.path.join(OUTPUT_BASE, f"{pat}{seed}")
        if method == "iql":
            d = os.path.join(run, f"fqe_{split}" if ckpt == "best"
                             else f"fqe_{split}_criticfinal")
        else:
            d = os.path.join(run, f"fqe_{split}" + ("" if ckpt == "best"
                                                    else "_finalckpt"))
        v = _v0(d)
        if v is None:
            continue
        shared = v.index.intersection(ref.index)
        diffs.append(float(v[shared].mean() - ref[shared].mean()))
    if len(diffs) < 2:
        return None
    t, p = stats.ttest_1samp(diffs, 0.0)
    return {"method": method, "reference": reference, "checkpoint": ckpt,
            "split": split, "n_seeds": len(diffs),
            "mean_diff": round(float(np.mean(diffs)), 4),
            "sd_diff": round(float(np.std(diffs, ddof=1)), 4),
            "t": round(float(t), 3), "p": round(float(p), 4)}


def report(split: str = "test") -> dict:
    df = collect(split)
    if df.empty:
        print("no FQE results yet")
        return {}
    ensure_dir(OUTPUT_BASE)
    df.to_csv(os.path.join(OUTPUT_BASE, f"final_results_{split}.csv"), index=False)

    piv = (df.groupby(["method", "checkpoint"])["fqe"]
             .agg(["mean", "std", "count"]).reset_index())
    from src.final_campaign import CHECKPOINT_DEFINITIONS
    out = {"split": split, "table": piv.to_dict(orient="records"),
           "checkpoint_definition_by_method": CHECKPOINT_DEFINITIONS}

    # Completeness travels with the table. A mean over three seeds and a mean
    # over five look the same in print, and this is the collector whose output
    # becomes a thesis table.
    expected = {"cql": len(SEEDS), "d3qn": len(SEEDS), "iql": len(SEEDS),
                "behaviour_est": 1, "bc": 1, "withhold": 1}
    got = {m: {c: int(piv[(piv["method"] == m) & (piv["checkpoint"] == c)]
                      ["count"].sum()) for c in ("best", "final")}
           for m in expected}
    short = {m: g for m, g in got.items()
             if any(v < expected[m] for v in g.values())}
    out["seeds_expected"] = expected
    out["seeds_found"] = got
    out["complete"] = not short
    if short:
        print(f"\nINCOMPLETE: {short}. Every number below is a mean over "
              f"fewer runs than the protocol specifies.")

    # "clinician" is the estimated behaviour policy, an MLP classifier fitted to
    # the logged actions. It is not the logged clinician: the observed
    # discounted return is a separate quantity, reported as the empirical
    # clinician policy. Conflating the two is what makes an absolute-scale
    # discrepancy look like a calibration gap.
    order = ["cql", "iql", "d3qn", "withhold", "behaviour_est", "bc"]
    label = {"behaviour_est": "behaviour (est.)", "withhold": "withhold (all)",
             "bc": "behaviour cloning"}
    # Column headings for IQL name the critic, since its actor is
    # validation-selected in both arms. See CHECKPOINT_DEFINITIONS.
    col_note = {"iql": ("early = actor from the epoch-1 critic, "
                        "final = actor from the epoch-40 critic")}
    print(f"\nFQE, {split} split — seed mean (sd). Read every value against "
          f"the behaviour (est.) row of the same column.")
    print(f"{'policy':<18}{'early':>22}{'final':>22}")
    for m in order:
        line = f"{label.get(m, m):<18}"
        for ck in ("best", "final"):
            r = piv[(piv["method"] == m) & (piv["checkpoint"] == ck)]
            if len(r) == 0:
                line += f"{'-':>22}"
            else:
                mu, sd, n = r["mean"].iloc[0], r["std"].iloc[0], int(r["count"].iloc[0])
                line += (f"{mu:>13.3f} ({sd:.3f})" if n > 1 and not np.isnan(sd)
                         else f"{mu:>21.3f}")
        print(line)

    print("\n  CQL and D3QN: early is the epoch-1 Q checkpoint, after 5,000 "
          "updates; final is epoch 40.")
    print("  IQL: both arms use a validation-selected actor. Early is the "
          "actor from the\n       epoch-1 critic, final the actor from the "
          "epoch-40 critic. There is no\n       final-actor-epoch checkpoint, "
          "so the IQL columns vary the critic.")

    sig = []
    for ck in ("best", "final"):
        for m in ("cql", "iql", "d3qn"):
            for ref in ("behaviour_est", "withhold"):
                r = paired_vs_reference(split, ck, m, ref)
                if r:
                    sig.append(r)
    out["paired_tests"] = sig
    if sig:
        print(f"\nSeed-level paired differences (n=5, df=4)")
        print(f"{'method':<8}{'vs':<11}{'ckpt':<7}{'mean':>9}{'sd':>9}{'t':>8}{'p':>9}")
        for r in sig:
            print(f"{r['method']:<8}{r['reference']:<11}{r['checkpoint']:<7}"
                  f"{r['mean_diff']:>9.3f}{r['sd_diff']:>9.3f}"
                  f"{r['t']:>8.2f}{r['p']:>9.4f}")

    with open(os.path.join(OUTPUT_BASE, f"final_results_{split}.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    if short:
        # Written first so the partial table can be inspected. Returning
        # successfully would let the orchestrator mark the core results table
        # done while a checkpoint or a method was missing from it. stage_fqe
        # protects the 50-row matrix, but the final-critic IQL results come
        # from a separate analysis step that can fail while the chain goes on.
        raise AssertionError(
            f"results table for the {split} split is incomplete: {short} as "
            f"found against expected. Every mean in it is over fewer runs than "
            f"the protocol specifies.")
    return out


if __name__ == "__main__":
    # Both splits are assembled before either can stop the step, so a partial
    # state leaves both artefacts on disk rather than only the first.
    import sys
    errs = []
    for sp in ("test", "val"):
        try:
            report(sp)
        except AssertionError as e:
            errs.append(str(e))
    if errs:
        print("\n" + "\n".join(errs), file=sys.stderr)
        sys.exit(1)
