"""
Check whether extreme observations dominate training-set feature scales.

Compares each feature's full standard deviation with the standard deviation
of its central 99.8%. Ratios above 1.2 are flagged for review. This audit
does not change data; source-value filters are defined in final_features.

    python -m src.final_scaler_audit
"""
import json
import os

import numpy as np
import pandas as pd

from config.paths import PROJECT_ROOT
from src.final_preprocessing import STATE_COLS, compute_derived, split_by_subject
from src.final_features import FINAL_FEATURES_DIR
from src.utils import ensure_dir

# Above this, the fitted scale is being set by the tail rather than the data.
FLAG_RATIO = 1.2

# Features whose residual ratio has been examined and accepted, as a comma-
# separated list. A flag that is not listed here fails the audit, so a known
# exception has to be written down rather than noticed and forgotten. The
# threshold itself is never moved to make a flag disappear.
ACCEPTED = [c for c in os.environ.get("SEPSIS_SCALER_ACCEPT", "").split(",") if c]


def run(seed: int = 42) -> dict:
    feats = compute_derived(pd.read_parquet(
        os.path.join(FINAL_FEATURES_DIR, "features.parquet")))
    feats = feats.sort_values(["stay_id", "time_bin"]).reset_index(drop=True)
    train = split_by_subject(feats, seed=seed)["train"]

    rows = []
    for c in STATE_COLS:
        v = train[c].dropna()
        if not len(v):
            rows.append({"column": c, "n": 0})
            continue
        lo, med, hi = v.quantile([0.001, 0.5, 0.999])
        inner = v[(v >= lo) & (v <= hi)]
        sd, sd_inner = float(v.std()), float(inner.std())
        rows.append({
            "column": c, "n": int(len(v)),
            "min": float(v.min()), "p001": float(lo), "median": float(med),
            "p999": float(hi), "max": float(v.max()),
            "sd": sd, "sd_inner": sd_inner,
            "ratio": (sd / sd_inner) if sd_inner else None,
        })
    df = pd.DataFrame(rows).sort_values("ratio", ascending=False,
                                        na_position="last")
    flagged = df[df["ratio"] > FLAG_RATIO]["column"].tolist()
    unaccepted = [c for c in flagged if c not in ACCEPTED]
    out = {"flag_ratio": FLAG_RATIO, "n_features": len(df),
           "flagged": flagged, "accepted": ACCEPTED,
           "unaccepted": unaccepted, "passed": not unaccepted,
           "features": df.to_dict(orient="records")}

    d = os.environ.get("SEPSIS_STATUS_DIR",
                       os.path.join(PROJECT_ROOT, "outputs", "final_preflight"))
    ensure_dir(d)
    p = os.path.join(d, "scaler_audit.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2, default=float)

    print(f"\n{'column':<22}{'median':>10}{'max':>14}{'sd':>12}"
          f"{'sd inner':>11}{'ratio':>8}")
    for r in df.head(12).to_dict(orient="records"):
        if not r.get("n"):
            continue
        print(f"{r['column']:<22}{r['median']:>10,.3g}{r['max']:>14,.4g}"
              f"{r['sd']:>12,.4g}{r['sd_inner']:>11,.4g}{r['ratio']:>8.2f}")
    if flagged:
        print(f"\nflagged at ratio > {FLAG_RATIO}: " + ", ".join(flagged))
    if ACCEPTED:
        print("accepted by prior disposition: " + ", ".join(ACCEPTED))
    print(f"{'PASS' if out['passed'] else 'FAIL'}"
          + ("" if out["passed"] else
             " — not accepted: " + ", ".join(unaccepted)))
    print(f"saved -> {p}")
    return out


if __name__ == "__main__":
    import sys
    sys.exit(0 if run()["passed"] else 1)
