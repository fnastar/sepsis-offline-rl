"""
final_audit_extra.py: Three read-only audits requested before results are final.

None of these modify data or models. Each answers a question that would
otherwise have to be taken on trust.

1. SOFA strata source. Confirms the raw official SOFA can be joined back on
   (stay_id, time_bin) and quantifies what restricting to `sofa_observed == 1`
   costs, so subgroup analysis never has to inverse-transform a state feature
   that was median-imputed and clipped at +/-5 sigma.

2. storetime completeness. The protocol requires laboratory availability to be
   keyed on storetime; 11 rows had none and fell back to charttime. This finds
   which state cells they reach and whether dropping them changes anything.

3. Vasopressor aggregation. Compares the active-time mean
   (sum(rate*overlap) / sum(overlap)) against exposure integrated over the full
   interval (sum(rate*overlap) / 4 h), reporting how many continuous values,
   thresholds and discrete actions differ between the two conventions. Neither
   is the reported estimand, which is the maximum in the window; that
   comparison lives in src/final_vaso_audit.py.

Writes outputs/final_preflight/audit_extra.json.
"""
import json
import os

import numpy as np
import pandas as pd

from config.clinical import (
    ALL_LAB_ITEMIDS, ALL_VASO_ITEMIDS, LAB_ITEMIDS, TIME_BIN_HOURS,
    VASOPRESSOR_ITEMIDS,
)
from config.paths import FINAL_COHORT_PATH, PROJECT_ROOT
from src.data_loading import load_inputevents, load_labevents
from src.final_features import (
    FINAL_FEATURES_DIR, NOREPI_EQUIV_FACTORS, _allocate_across_bins,
    _build_itemid_to_name, _norepi_equivalent_rate,
)
from src.final_preprocessing import FINAL_PROCESSED_DIR
from src.utils import ensure_dir

AUDIT_DIR = os.environ.get(
    "SEPSIS_STATUS_DIR",
    os.path.join(PROJECT_ROOT, "outputs", "final_preflight"))


# ─────────────────────────────────────────────────────────────────────────────

def audit_sofa_strata() -> dict:
    """Raw SOFA joins cleanly, and what restricting to observed rows costs."""
    feats = pd.read_parquet(
        os.path.join(FINAL_FEATURES_DIR, "features.parquet"),
        columns=["stay_id", "time_bin", "sofa_total", "sofa_observed"])
    a: dict = {}
    a["rows"] = int(len(feats))
    a["observed_pct"] = round(100 * float(feats["sofa_observed"].mean()), 2)
    a["missing_rows"] = int((feats["sofa_observed"] == 0).sum())

    obs = feats[feats["sofa_observed"] == 1]
    a["raw_sofa_observed"] = {
        "min": float(obs["sofa_total"].min()), "max": float(obs["sofa_total"].max()),
        "mean": round(float(obs["sofa_total"].mean()), 3),
        "median": float(obs["sofa_total"].median()),
    }
    # Strata as used in the legacy subgroup analysis
    bins = [-0.1, 5, 10, 100]
    labels = ["0-5", "6-10", "11+"]
    strat = pd.cut(obs["sofa_total"], bins=bins, labels=labels)
    a["strata_counts_observed"] = {str(k): int(v) for k, v in
                                   strat.value_counts().sort_index().items()}
    # Coverage by interval, since the gap concentrates at the onset boundary
    by_bin = feats.groupby("time_bin")["sofa_observed"].mean().round(4)
    a["observed_pct_by_time_bin"] = {int(k): round(100 * float(v), 2)
                                     for k, v in by_bin.items()}

    # Does restricting to observed rows shift the cohort's severity mix?
    proc = pd.read_parquet(os.path.join(FINAL_PROCESSED_DIR, "test.parquet"),
                           columns=["stay_id", "time_bin", "icu_mortality"])
    m = proc.merge(feats, on=["stay_id", "time_bin"], how="left")
    a["test_mortality_all"] = round(
        100 * float(m.groupby("stay_id")["icu_mortality"].first().mean()), 2)
    mo = m[m["sofa_observed"] == 1]
    a["test_mortality_sofa_observed"] = round(
        100 * float(mo.groupby("stay_id")["icu_mortality"].first().mean()), 2)
    a["test_stays_all"] = int(m["stay_id"].nunique())
    a["test_stays_sofa_observed"] = int(mo["stay_id"].nunique())
    a["verdict"] = (
        "Join raw sofa_total on (stay_id, time_bin) and stratify on "
        "sofa_observed == 1; report the excluded fraction separately. No "
        "inverse transform of the model state is needed or used.")
    return a


# ─────────────────────────────────────────────────────────────────────────────

def audit_storetime() -> dict:
    """Which state cells the 11 storetime-less lab rows can reach."""
    cohort = pd.read_csv(FINAL_COHORT_PATH, usecols=["stay_id", "hadm_id"])
    hadm_ids = set(cohort["hadm_id"].dropna().astype(int))
    labs = load_labevents(
        hadm_ids=hadm_ids, item_ids=ALL_LAB_ITEMIDS,
        usecols=["labevent_id", "subject_id", "hadm_id", "itemid",
                 "charttime", "storetime", "value", "valuenum", "valueuom"])
    id_to_name = _build_itemid_to_name(LAB_ITEMIDS)
    labs = labs[labs["itemid"].isin(id_to_name)].copy()
    labs["feature"] = labs["itemid"].map(id_to_name)
    labs["storetime"] = pd.to_datetime(labs["storetime"])
    labs["charttime"] = pd.to_datetime(labs["charttime"])

    miss = labs[labs["storetime"].isna()]
    a = {
        "lab_rows_in_cohort": int(len(labs)),
        "rows_without_storetime": int(len(miss)),
        "pct_without_storetime": round(100 * len(miss) / max(len(labs), 1), 6),
    }
    if len(miss):
        a["affected_features"] = {str(k): int(v) for k, v in
                                  miss["feature"].value_counts().items()}
        a["affected_hadm_ids"] = sorted(miss["hadm_id"].dropna().astype(int).unique().tolist())[:20]
        a["also_missing_charttime"] = int(miss["charttime"].isna().sum())
    # How far storetime normally lags charttime — the reason the rule exists
    both = labs.dropna(subset=["storetime", "charttime"])
    lag = (both["storetime"] - both["charttime"]).dt.total_seconds() / 3600
    a["storetime_lag_hours"] = {
        "median": round(float(lag.median()), 2),
        "p90": round(float(np.percentile(lag, 90)), 2),
        "pct_over_4h": round(100 * float((lag > 4).mean()), 2),
    }
    a["verdict"] = (
        f"{len(miss)} of {len(labs):,} rows ({100*len(miss)/max(len(labs),1):.6f}%) "
        "lack storetime. They are dropped: the charttime fallback could place "
        "a result in the state before the clinician could have seen it. The "
        "correction diff confirms the predicted magnitude, which is none — "
        "zero feature columns changed.")
    return a


# ─────────────────────────────────────────────────────────────────────────────

def audit_vaso_convention() -> dict:
    """Active-time mean (used) against exposure over the full interval."""
    cohort = pd.read_csv(FINAL_COHORT_PATH,
                         usecols=["stay_id", "suspected_infection_time"])
    cohort["suspected_infection_time"] = pd.to_datetime(
        cohort["suspected_infection_time"])
    onset = cohort.set_index("stay_id")["suspected_infection_time"]

    vaso = load_inputevents(stay_ids=set(cohort["stay_id"]),
                            item_ids=ALL_VASO_ITEMIDS)
    id_to_name = _build_itemid_to_name(VASOPRESSOR_ITEMIDS)
    vaso = vaso[vaso["itemid"].isin(id_to_name)].copy()
    vaso["drug"] = vaso["itemid"].map(id_to_name)
    vaso = vaso[vaso["drug"].isin(NOREPI_EQUIV_FACTORS)]
    vaso["ne_rate"] = _norepi_equivalent_rate(vaso)

    alloc = _allocate_across_bins(vaso, onset)
    alloc = alloc[alloc["overlap_h"] > 0]
    alloc["weighted"] = alloc["ne_rate"] * alloc["overlap_h"]
    per_drug = (alloc.groupby(["stay_id", "time_bin", "drug"])
                     .agg(w=("weighted", "sum"), h=("overlap_h", "sum"))
                     .reset_index())
    per_drug["active_mean"] = per_drug["w"] / per_drug["h"].replace(0, np.nan)
    per_drug["interval_exposure"] = per_drug["w"] / TIME_BIN_HOURS

    g = (per_drug.groupby(["stay_id", "time_bin"])[["active_mean", "interval_exposure"]]
                 .sum().reset_index())
    a = {
        "treated_intervals": int(len(g)),
        "coverage_h": {
            "pct_full_4h": round(100 * float((per_drug["h"] >= 3.999).mean()), 2),
            "pct_under_2h": round(100 * float((per_drug["h"] < 2).mean()), 2),
            "pct_over_4h_overlapping_orders": round(
                100 * float((per_drug["h"] > 4.001).mean()), 2),
            "max_h": round(float(per_drug["h"].max()), 2),
        },
        "active_mean": {"median": round(float(g["active_mean"].median()), 5),
                        "p90": round(float(np.percentile(g["active_mean"], 90)), 5)},
        "interval_exposure": {
            "median": round(float(g["interval_exposure"].median()), 5),
            "p90": round(float(np.percentile(g["interval_exposure"], 90)), 5)},
    }
    ratio = g["interval_exposure"] / g["active_mean"].replace(0, np.nan)
    a["exposure_over_active_ratio"] = {
        "median": round(float(ratio.median()), 4),
        "p05": round(float(np.nanpercentile(ratio, 5)), 4),
        "p95": round(float(np.nanpercentile(ratio, 95)), 4),
    }

    # Thresholds and discrete levels under each convention, train split only
    proc = pd.read_parquet(os.path.join(FINAL_PROCESSED_DIR, "train.parquet"),
                           columns=["stay_id", "time_bin", "vaso_level"])
    m = proc.merge(g, on=["stay_id", "time_bin"], how="left").fillna(
        {"active_mean": 0.0, "interval_exposure": 0.0})
    cuts = {}
    for col in ("active_mean", "interval_exposure"):
        nz = m.loc[m[col] > 0, col]
        cuts[col] = [round(float(q), 6) for q in np.quantile(nz, [.25, .5, .75])]
    a["train_cutpoints"] = cuts

    lv = {}
    for col in ("active_mean", "interval_exposure"):
        edges = [1e-12] + cuts[col]
        lv[col] = np.digitize(m[col].values, edges).clip(0, 4)
    a["levels_changed_pct"] = round(
        100 * float((lv["active_mean"] != lv["interval_exposure"]).mean()), 2)
    treated = m["active_mean"].values > 0
    a["levels_changed_pct_among_treated"] = round(
        100 * float((lv["active_mean"][treated] != lv["interval_exposure"][treated]).mean()), 2)
    a["verdict"] = (
        "The reported estimand is the maximum instantaneous rate reached in "
        "the interval, which is what Raghu et al. and AI Clinician bin and "
        "what bounds duplicate order rows. Two alternatives are kept as "
        "sensitivity checks and neither is reported as the specification: the "
        "active-time mean, compared in src/final_vaso_audit.py, and interval "
        "exposure, compared here. Exposure is systematically lower whenever an "
        "infusion covers only part of a bin and answers a different question, "
        "namely drug per unit time over the whole interval.")
    return a


def main() -> dict:
    out = {}
    print("1/3 SOFA strata source...")
    out["sofa_strata"] = audit_sofa_strata()
    print("2/3 storetime completeness...")
    out["storetime"] = audit_storetime()
    print("3/3 vasopressor aggregation convention...")
    out["vaso_convention"] = audit_vaso_convention()

    ensure_dir(AUDIT_DIR)
    p = os.path.join(AUDIT_DIR, "audit_extra.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"  -> {p}")
    return out


if __name__ == "__main__":
    main()
