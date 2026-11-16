"""
Extract onset-anchored features for the final Sepsis-3 cohort.

Four-hour bins cover -24 to +48 hours from suspected-infection onset. Vitals
and laboratory results must be available strictly before the decision
boundary; laboratory availability uses storetime. Infusions are allocated
to the intervals they overlap.

SOFA comes from the official MIMIC-IV derived table, using the latest row
with endtime <= boundary. Previous-interval fluid and vasopressor exposure
are included as treatment-history features.

Writes <SEPSIS_FINAL_FEATURES_DIR>/features.parquet, one row per
(stay_id, time_bin).
"""
import os

import numpy as np
import pandas as pd

from config.clinical import (
    VITALS_ITEMIDS, LAB_ITEMIDS,
    VASOPRESSOR_ITEMIDS, IV_FLUID_ITEMIDS, URINE_OUTPUT_ITEMIDS,
    ALL_CHART_ITEMIDS, ALL_LAB_ITEMIDS,
    ALL_VASO_ITEMIDS, ALL_IV_ITEMIDS,
    TIME_BIN_HOURS,
)
from config.paths import COHORT_DIR, DATA_DIR, FINAL_COHORT_PATH
from src.data_loading import (
    load_chartevents, load_labevents, load_inputevents, load_outputevents,
)
from src.utils import ensure_dir

FINAL_FEATURES_DIR = os.environ.get(
    "SEPSIS_FINAL_FEATURES_DIR", os.path.join(DATA_DIR, "features_final"))
SOFA_OFFICIAL_PATH = os.environ.get(
    "SEPSIS_SOFA_OFFICIAL",
    os.path.join(COHORT_DIR, "sofa_official_v3_1.csv.gz"))

# Norepinephrine equivalences (Goradia et al. 2021), applied to a rate already
# normalised to per-minute units by `_norepi_equivalent_rate`. Vasopressin's
# factor is per unit/MIN; the legacy pipeline applied it to a units/HOUR figure,
# which is where its 60-fold inflation came from. That correction now lives in
# the unit conversion rather than in the factor, so rows genuinely recorded per
# minute are not divided twice.
NOREPI_EQUIV_FACTORS = {
    "norepinephrine": 1.0,
    "epinephrine": 1.0,
    "dopamine": 0.01,
    "phenylephrine": 0.1,
    "vasopressin": 5.0,
    # dobutamine and milrinone are inotropes, excluded
}


def _build_itemid_to_name(itemid_dict: dict) -> dict:
    """Invert {name: [itemids]} to {itemid: name}."""
    return {iid: name for name, ids in itemid_dict.items() for iid in ids}


# Physiological ceiling on the norepinephrine-equivalent rate, mcg/kg/min.
#
# MIMIC computes `rate` from the ordered amount and the recorded duration, so a
# duration documented as one minute turns a whole bag into an impossible rate:
# 32 mg of norepinephrine over one minute for an 89 kg patient is recorded as
# 359 mcg/kg/min. 165 of 387,474 rows exceed 5, 82 exceed 10, and the source
# distribution breaks cleanly — the 99.9th percentile is 2.37 while the 99.99th
# is 24.5. Sustained equivalents above 5 are not administered; salvage dosing in
# refractory shock reaches 2 to 3.
#
# Rates above the ceiling are clipped rather than dropped, so the patient stays
# on the drug and the interval keeps a top-level action. No action label moves:
# the level-4 cut-point is 0.26, and every clipped value is above it either way.
# Applied to source rows before aggregation and before binning, and blind to
# outcome.
NE_RATE_CEILING = float(os.environ.get("SEPSIS_NE_CEILING", 5.0))


def _norepi_equivalent_rate(vaso: pd.DataFrame) -> pd.Series:
    """
    Norepinephrine-equivalent rate in mcg/kg/min, converting each row by its
    own recorded unit.

    Applying one factor per drug is what produced the 60-fold vasopressin
    inflation in the legacy pipeline, and MIMIC-IV is not internally uniform:
    vasopressin is units/hour on 37,160 rows but units/min on 3, norepinephrine
    is mg/kg/min on 2 rows (a 1000-fold difference), and phenylephrine is
    mcg/min without the weight normalisation on 2. The counts are tiny, but a
    single row inflated 1000-fold moves the non-zero quartiles that define the
    entire action grid.
    """
    rate = vaso["rate"].fillna(0.0).astype(float)
    uom = vaso["rateuom"].fillna("").str.lower().str.strip()
    weight = vaso.get("patientweight")
    weight = (pd.Series(np.nan, index=vaso.index) if weight is None
              else weight.astype(float))
    # Median weight is a better fallback than dropping a real infusion
    weight = weight.where(weight > 0).fillna(weight[weight > 0].median() or 80.0)

    per_min = pd.Series(np.nan, index=vaso.index, dtype=float)
    per_min[uom == "mcg/kg/min"] = rate
    per_min[uom == "mg/kg/min"] = rate * 1000.0
    per_min[uom == "mcg/min"] = rate / weight
    per_min[uom == "mg/min"] = rate * 1000.0 / weight
    # Vasopressin is dosed in units, converted to an equivalent below
    per_min[uom == "units/hour"] = rate / 60.0
    per_min[uom == "units/min"] = rate
    unknown = per_min.isna() & (rate > 0)
    if unknown.any():
        print(f"  WARNING: {int(unknown.sum())} vasopressor rows with "
              f"unhandled rateuom {sorted(uom[unknown].unique())[:5]} — dropped")
    ne = per_min.fillna(0.0) * vaso["drug"].map(NOREPI_EQUIV_FACTORS)

    # Two phenylephrine rows carry a negative amount and rate, which is a
    # documentation reversal rather than an infusion. They are not doses.
    n_neg = int((ne < 0).sum())
    if n_neg:
        print(f"  {n_neg} vasopressor rows with a negative rate set to zero")
        ne = ne.clip(lower=0.0)

    n_hi = int((ne > NE_RATE_CEILING).sum())
    if n_hi:
        print(f"  {n_hi} vasopressor rows above {NE_RATE_CEILING} mcg/kg/min "
              f"equivalent clipped to the ceiling "
              f"(max was {ne.max():.1f}); see NE_RATE_CEILING")
        ne = ne.clip(upper=NE_RATE_CEILING)
    return ne


# ─────────────────────────────────────────────────────────────────────────────
# EPISODE GRID
# ─────────────────────────────────────────────────────────────────────────────

def build_boundaries(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (stay_id, time_bin) decision boundary.

    ``boundary`` is the instant the state is observed at and the action
    interval opens; the interval closes at ``boundary_end``. Bins run from
    ``first_full_time_bin`` to ``last_full_time_bin``, both computed in
    ``final_cohort.py`` after clipping the onset-centred window to the ICU
    stay, so partial intervals at either end are already excluded.
    """
    rows = []
    for r in cohort.itertuples(index=False):
        bins = np.arange(r.first_full_time_bin, r.last_full_time_bin + 1)
        rows.append(pd.DataFrame({
            "stay_id": r.stay_id,
            "hadm_id": r.hadm_id,
            "time_bin": bins,
            "boundary": r.suspected_infection_time
                        + pd.to_timedelta(bins * TIME_BIN_HOURS, unit="h"),
        }))
    b = pd.concat(rows, ignore_index=True)
    b["boundary_end"] = b["boundary"] + pd.Timedelta(hours=TIME_BIN_HOURS)
    return b.sort_values(["stay_id", "time_bin"]).reset_index(drop=True)


def _asof_state(boundaries: pd.DataFrame, events: pd.DataFrame,
                time_col: str, strictly_before: bool,
                value_col: str = "valuenum") -> pd.DataFrame:
    """
    For each boundary, the last observed value of each feature.

    ``strictly_before=True`` enforces the past-only rule: a value timestamped
    exactly at the boundary is treated as not yet available, since the action
    for that interval is taken at the boundary itself.
    """
    base = (boundaries[["stay_id", "time_bin", "boundary"]]
            .drop_duplicates(subset=["stay_id", "time_bin"])
            .sort_values("boundary")
            .reset_index(drop=True))

    # Each feature contributes exactly one column, assembled positionally
    # against a fixed row order. Merging feature-by-feature onto an
    # accumulating frame instead would let a single repeated key duplicate
    # either rows or columns, and neither shows up until much later.
    cols: dict[str, np.ndarray] = {}
    for feat, grp in events.groupby("feature", sort=False):
        right = (grp.loc[:, ["stay_id", time_col, value_col]]
                 .dropna(subset=[time_col, value_col])
                 .sort_values(time_col))
        right.columns = ["stay_id", time_col, "_v"]
        if right.empty:
            continue
        merged = pd.merge_asof(
            base[["stay_id", "time_bin", "boundary"]], right,
            left_on="boundary", right_on=time_col, by="stay_id",
            direction="backward",
            allow_exact_matches=not strictly_before,
        )
        assert len(merged) == len(base), (
            f"{feat}: merge_asof changed row count "
            f"({len(base)} -> {len(merged)})")
        cols[feat] = merged["_v"].to_numpy()

    out = base.copy()
    for feat, vals in cols.items():
        out[feat] = vals
    assert not out.columns.duplicated().any(), \
        f"duplicate columns: {out.columns[out.columns.duplicated()].tolist()}"
    return out


# ─────────────────────────────────────────────────────────────────────────────
# OBSERVATIONS (past-only)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# SOURCE PLAUSIBILITY
# ─────────────────────────────────────────────────────────────────────────────
#
# Values outside physiological possibility are data-entry errors, and they are
# far more damaging than their count suggests. The standardisation mean and
# scale are fitted on the raw training column, so a single mean arterial
# pressure recorded as 1,234,570 sets the scale every real blood pressure is
# then measured against: the fitted SD was 2,998 where the inner 99.8% of the
# column has an SD of 15, and a pressure of 40 against one of 80 separated by
# less than a tenth of a standard deviation. Clipping after standardisation
# cannot recover that.
#
# The bounds for the vitals and for FiO2 are not chosen here. They are the ones
# the official MIMIC-IV derived layer already applies, verified against the
# observed ranges of `mimiciv_3_1_derived.vitalsign` (heart rate 1 to 296, SBP
# 0.1 to 365, DBP 1 to 299, MAP 0.26 to 299, respiratory rate 0.13 to 69,
# temperature 15 to 43.1, SpO2 0.01 to 100) and of `mimiciv_3_1_derived.bg`
# (FiO2 21 to 100). This project already takes its cohort and its SOFA from
# that layer, so applying its plausibility rules to the variables it also
# publishes is consistency rather than a new choice.
#
# Base excess is the one entry with no derived-layer backing: `bg` publishes it
# unfiltered, its own range running from -414 to 188. The +/-50 here is a
# clinical judgement and is the only project-defined bound in this table. In
# this cohort it touches seven boundary values, all of them the same impossible
# source observation of -413, against an otherwise observed range of -22 to 44.
#
# Urine output is deliberately absent. Its tail was a mechanism rather than a
# distribution: GU irrigant instilled into the bladder was being summed as
# output. That is fixed at the source in extract_urine_final, and a bound on
# top of the fix would only hide whatever it failed to catch.
#
# Out-of-range values become missing rather than clipped: a MAP of 1.2 million
# is not a very high pressure, it is a wrong entry, and the imputation path
# already handles absent values. Applied at extraction, before the as-of
# selection, so an impossible reading cannot be chosen as the last observation
# before a boundary and displace a valid earlier one. Applied blind to outcome.
# (low, high, whether the endpoints themselves are kept). The derived layer
# writes its vital filters as strict inequalities and its FiO2 range as an
# inclusive one, so the flag carries that rather than flattening it.
PLAUSIBLE_RANGES = {
    # official mimiciv_3_1_derived.vitalsign filters
    "heart_rate": (0, 300, "neither"),
    "sbp": (0, 400, "neither"),
    "dbp": (0, 300, "neither"),
    "mbp": (0, 300, "neither"),
    "resp_rate": (0, 70, "neither"),
    "temperature": (10, 50, "neither"),
    "spo2": (0, 100, "right"),
    "glucose": (0, np.inf, "neither"),
    # official mimiciv_3_1_derived.bg range
    "fio2": (21, 100, "both"),
    # set here, no derived-layer convention exists
    "base_excess": (-50, 50, "both"),
}

# itemid 227488 is GU Irrigant Volume In: fluid instilled into the bladder, not
# urine produced. Summing it as output is what put 48,900 mL in a four-hour
# window. The official mimic-code urine_output concept negates it, and the
# corrected maximum is 9,800 mL.
GU_IRRIGANT_IN_ITEMID = 227488


# Written into the dataset manifest so the corrections travel with the data
# rather than living only in this file.
PLAUSIBILITY_PROVENANCE = {
    "ranges": {k: {"low": lo, "high": (None if hi == np.inf else hi),
                   "endpoints": inc}
               for k, (lo, hi, inc) in PLAUSIBLE_RANGES.items()},
    "source": {
        "vitals_and_glucose":
            "mimiciv_3_1_derived.vitalsign filters, verified against that "
            "table's observed ranges over 13,519,533 rows",
        "fio2":
            "mimiciv_3_1_derived.bg publishes FiO2 only in [21, 100], over "
            "697,418 rows",
        "base_excess":
            "project-defined. The derived bg table publishes base excess "
            "unfiltered, its own range running -414 to 188. In this cohort the "
            "bound touches seven boundary values, all the same source "
            "observation of -413, against an otherwise observed -22 to 44.",
    },
    "applied": "at extraction, before as-of selection, blind to outcome",
    "action_on_violation": "value set to missing, not clipped",
    "fio2_items": "223835 only; 227009 and 227010 are APACHE IV score fields, "
                  "not bedside measurements",
    "fio2_units": "values in [0.21, 1] read as a fraction and converted to "
                  "percent; [21, 100] kept; everything else missing",
    "urine_output": "itemid 227488 is GU irrigant instilled and is subtracted, "
                    "following the official concept; a negative interval net "
                    "is set to missing rather than zero",
    "pao2_fio2": "not a state feature. The last PaO2 and the last FiO2 at a "
                 "boundary are selected independently and keyed differently, "
                 "so their ratio can pair measurements from different times.",
}


def apply_plausible_ranges(df: pd.DataFrame, feature_col: str,
                           value_col: str) -> pd.DataFrame:
    """Drop rows whose value is outside physiological possibility."""
    # Label-based assignment below, so the labels have to be unique. The
    # chunked loaders concatenate with ignore_index, which guarantees it, but a
    # repeated label here would silently drop every row sharing it and that
    # exact failure has occurred in this pipeline before.
    assert df.index.is_unique, "apply_plausible_ranges needs a unique index"
    keep = pd.Series(True, index=df.index)
    dropped = {}
    for name, (lo, hi, inclusive) in PLAUSIBLE_RANGES.items():
        m = df[feature_col] == name
        if not m.any():
            continue
        v = df.loc[m, value_col]
        # A missing value is absent, not implausible, so it is left alone.
        bad = v.notna() & ~v.between(lo, hi, inclusive=inclusive)
        if bad.any():
            dropped[name] = int(bad.sum())
            keep.loc[bad[bad].index] = False
    if dropped:
        total = sum(dropped.values())
        detail = ", ".join(f"{k} {v:,}" for k, v in sorted(dropped.items()))
        print(f"  plausibility: dropped {total:,} readings ({detail})")
    return df[keep]


def extract_vitals_asof(boundaries: pd.DataFrame, stay_ids: set) -> pd.DataFrame:
    """Vitals from chartevents, carried forward from strictly before each boundary."""
    print("Extracting vitals (past-only)...")
    chart = load_chartevents(stay_ids=stay_ids, item_ids=ALL_CHART_ITEMIDS)
    if chart.empty:
        return boundaries[["stay_id", "time_bin"]].copy()

    id_to_name = _build_itemid_to_name(VITALS_ITEMIDS)
    chart = chart[chart["itemid"].isin(id_to_name)].copy()
    chart["feature"] = chart["itemid"].map(id_to_name)

    f_mask = (chart["feature"] == "temperature") & (chart["valueuom"] == "°F")
    if f_mask.any():
        chart.loc[f_mask, "valuenum"] = (chart.loc[f_mask, "valuenum"] - 32) * 5 / 9

    # FiO2 is charted both as a percentage and, on a few hundred rows, as a
    # fraction. The old rule treated everything above 1 as a percentage, so an
    # entry of 4 became 0.04 and a P/F ratio of 25 times PaO2. Read [0.21, 1]
    # as a fraction and convert; leave [21, 100] alone; everything else fails
    # the range check below, including zero, the 928 entries between 1 and 20,
    # and the twelve above 100. The bounds are the derived bg table's own.
    frac = (chart["feature"] == "fio2") & chart["valuenum"].between(0.21, 1.0)
    if frac.any():
        print(f"  fio2: {int(frac.sum()):,} rows recorded as a fraction, "
              f"converted to percent")
        chart.loc[frac, "valuenum"] = chart.loc[frac, "valuenum"] * 100.0

    chart = apply_plausible_ranges(chart, "feature", "valuenum")

    chart["charttime"] = pd.to_datetime(chart["charttime"])
    res = _asof_state(boundaries, chart, "charttime", strictly_before=True,
                      value_col="valuenum")
    # GCS is charted as three components; the state uses the total, summed only
    # where at least one component is present so an unrecorded GCS stays missing
    # rather than becoming a spurious 0.
    gcs = [c for c in ("gcs_eye", "gcs_verbal", "gcs_motor") if c in res.columns]
    if gcs:
        res["gcs_total"] = res[gcs].sum(axis=1, min_count=1)
    print(f"  vitals: {res.shape[1] - 3} features")
    return res.drop(columns=["boundary"])


def extract_labs_asof(boundaries: pd.DataFrame, hadm_ids: set,
                      hadm_to_stay: dict) -> pd.DataFrame:
    """
    Labs from labevents, keyed on ``storetime``.

    A specimen drawn at 03:00 and resulted at 07:40 is not information the
    clinician had at 04:00. Using storetime rather than charttime is what keeps
    the state to what was actually knowable at the boundary.
    """
    print("Extracting labs (past-only, storetime)...")
    labs = load_labevents(
        hadm_ids=hadm_ids, item_ids=ALL_LAB_ITEMIDS,
        usecols=["labevent_id", "subject_id", "hadm_id", "itemid",
                 "charttime", "storetime", "value", "valuenum", "valueuom"],
    )
    if labs.empty:
        return boundaries[["stay_id", "time_bin"]].copy()

    id_to_name = _build_itemid_to_name(LAB_ITEMIDS)
    labs = labs[labs["itemid"].isin(id_to_name)].copy()
    labs["feature"] = labs["itemid"].map(id_to_name)
    labs["stay_id"] = labs["hadm_id"].map(hadm_to_stay)
    labs = labs.dropna(subset=["stay_id"])
    labs["stay_id"] = labs["stay_id"].astype(int)

    labs = apply_plausible_ranges(labs, "feature", "valuenum")

    labs["storetime"] = pd.to_datetime(labs["storetime"])
    labs["charttime"] = pd.to_datetime(labs["charttime"])
    # The protocol keys availability strictly on storetime, so a result with no
    # storetime is dropped rather than back-filled from charttime. Falling back
    # would assert the result was available when it was drawn, which is the
    # exact assumption storetime exists to avoid. 11 rows of 8.1 million are
    # affected, all blood gases, so this cannot move an aggregate — but the
    # rule should hold without exception rather than almost.
    n_missing = int(labs["storetime"].isna().sum())
    if n_missing:
        print(f"  storetime missing on {n_missing:,} rows; dropped "
              f"({100 * n_missing / len(labs):.6f}% of lab rows)")
        labs = labs[labs["storetime"].notna()]

    res = _asof_state(boundaries, labs, "storetime", strictly_before=True,
                      value_col="valuenum")
    print(f"  labs: {res.shape[1] - 3} features")
    return res.drop(columns=["boundary"])


def attach_official_sofa(boundaries: pd.DataFrame) -> pd.DataFrame:
    """
    Official hourly SOFA at each boundary: latest row with ``endtime <= boundary``.

    ``endtime`` is when the scoring window closes, so a row whose endtime
    equals the boundary is available at it — exact matches are allowed here,
    unlike the raw observation features.
    """
    print("Attaching official SOFA...")
    sofa = pd.read_csv(SOFA_OFFICIAL_PATH)
    sofa["endtime"] = pd.to_datetime(sofa["endtime"])
    sofa = (sofa.dropna(subset=["sofa_24hours"])
                .sort_values("endtime")
                .rename(columns={"sofa_24hours": "sofa_total"}))

    left = boundaries[["stay_id", "time_bin", "boundary"]].sort_values("boundary")
    merged = pd.merge_asof(
        left, sofa[["stay_id", "endtime", "sofa_total"]],
        left_on="boundary", right_on="endtime", by="stay_id",
        direction="backward", allow_exact_matches=True,
    )
    # The derived SOFA grid starts at ICU admission, so a boundary that
    # coincides with intime has no preceding row. That is not random: it
    # concentrates on bin 0, which is precisely the initial state FQE
    # evaluates. Past-only matching is required by the protocol, so those
    # values get imputed downstream — the indicator keeps that visible to the
    # model and to the audit rather than hiding it in a median.
    merged["sofa_observed"] = merged["sofa_total"].notna().astype(float)
    cov = merged["sofa_total"].notna().mean()
    by_bin = merged.groupby("time_bin")["sofa_observed"].mean()
    print(f"  SOFA present at {cov:.1%} of boundaries "
          f"(bin 0: {by_bin.get(0, float('nan')):.1%})")
    return merged[["stay_id", "time_bin", "sofa_total", "sofa_observed"]]


# ─────────────────────────────────────────────────────────────────────────────
# TREATMENTS (interval, proportionally allocated)
# ─────────────────────────────────────────────────────────────────────────────

def _allocate_across_bins(ev: pd.DataFrame, cohort_onset: pd.Series) -> pd.DataFrame:
    """
    Split each administration across every interval it overlaps.

    Returns one row per (stay_id, time_bin, event) with ``overlap_h`` and the
    fraction of the event's duration falling in that interval. Instantaneous
    events (bolus, zero duration) land wholly in the interval containing them.
    """
    # The loaders concatenate chunks, so index labels repeat. `.loc[labels]`
    # would then return a cross product rather than one row per label, so the
    # expansion below is done positionally on a clean index.
    ev = ev.reset_index(drop=True).copy()
    ev["onset"] = ev["stay_id"].map(cohort_onset)
    ev = ev.dropna(subset=["onset"])
    ev["starttime"] = pd.to_datetime(ev["starttime"])
    ev["endtime"] = pd.to_datetime(ev["endtime"]).fillna(ev["starttime"])
    # Guard against reversed or missing intervals
    ev.loc[ev["endtime"] < ev["starttime"], "endtime"] = ev["starttime"]

    h = TIME_BIN_HOURS
    start_off = (ev["starttime"] - ev["onset"]).dt.total_seconds() / 3600
    end_off = (ev["endtime"] - ev["onset"]).dt.total_seconds() / 3600
    ev["_b0"] = np.floor(start_off / h).astype(int)
    # An event ending exactly on a boundary does not touch the next interval
    ev["_b1"] = np.ceil(end_off / h).astype(int) - 1
    ev["_b1"] = np.maximum(ev["_b1"], ev["_b0"])

    n_bins = (ev["_b1"] - ev["_b0"] + 1).values
    pos = np.repeat(np.arange(len(ev)), n_bins)
    offsets = (np.concatenate([np.arange(n) for n in n_bins])
               if len(ev) else np.array([], dtype=int))
    out = ev.iloc[pos].reset_index(drop=True).copy()
    out["time_bin"] = out["_b0"].values + offsets

    bin_start = out["onset"] + pd.to_timedelta(out["time_bin"] * h, unit="h")
    bin_end = bin_start + pd.Timedelta(hours=h)
    ov_start = np.maximum(out["starttime"].values, bin_start.values)
    ov_end = np.minimum(out["endtime"].values, bin_end.values)
    out["overlap_h"] = np.maximum(
        (ov_end - ov_start) / np.timedelta64(1, "h"), 0.0)
    # Kept so concurrent infusions can be resolved on a shared timeline rather
    # than by summing per-drug means, which would add drugs that never overlapped
    out["_ov_start"] = ov_start
    out["_ov_end"] = ov_end

    # Positional arithmetic throughout: a Series division here would align on
    # index labels, which is the bug this function already carries once.
    dur = ((out["endtime"] - out["starttime"]).dt.total_seconds() / 3600).values
    ov = out["overlap_h"].values
    out["frac"] = np.where(dur > 0, np.divide(ov, dur, out=np.ones_like(ov),
                                              where=dur > 0), 1.0)
    return out.drop(columns=["_b0", "_b1", "onset"])


VASO_AGG = os.environ.get("SEPSIS_VASO_AGG", "max")


def _intensity_on_shared_timeline(alloc: pd.DataFrame,
                                  mode: str | None = None) -> pd.DataFrame:
    """
    Norepinephrine-equivalent dose per interval, on a shared drug timeline.

    Two modes, selected by ``SEPSIS_VASO_AGG``:

    ``max`` (default) — the highest instantaneous rate reached in the
    interval. This is what Raghu et al. bin (`max VP in`, sec. 3.3) and what
    AI Clinician's `CREATE ACTIONS` uses (`max_dose_vaso`), so it is the
    choice that makes the action grid directly comparable to both. It is also
    idempotent under duplicate order rows, which the mean is not.

    ``intensity`` — mean rate over the time any pressor was running. A
    conditional dose: what the patient was on while supported. Retained for
    the sensitivity audit, not for the reported specification.

    In both modes the instantaneous rate sums the per-drug maxima: equivalents
    add across drugs, but overlapping rows for one drug are a rate update
    rather than a second infusion.
    """
    mode = mode or VASO_AGG
    alloc = alloc.copy()
    alloc["weighted"] = alloc["ne_rate"] * alloc["overlap_h"]
    # Route on row count, not drug count: two overlapping rows for one drug
    # inflate both the numerator and the denominator of a naive ratio, and
    # 0.38% of cells accumulate more than four hours of "coverage" inside a
    # four-hour interval that way, one reaching 13.5 h.
    n_rows = alloc.groupby(["stay_id", "time_bin"])["ne_rate"].transform("size")

    single = alloc[n_rows == 1]
    s = (single.groupby(["stay_id", "time_bin"])
               .agg(w=("weighted", "sum"), h=("overlap_h", "sum"))
               .reset_index())
    if mode == "max":
        s = (single.groupby(["stay_id", "time_bin"])["ne_rate"].max()
                   .reset_index().rename(columns={"ne_rate": "rate"}))
    else:
        s["rate"] = s["w"] / s["h"].replace(0, np.nan)
    parts = [s[["stay_id", "time_bin", "rate"]]]

    multi = alloc[n_rows >= 2]
    if len(multi):
        rows = []
        for (sid, tb), g in multi.groupby(["stay_id", "time_bin"], sort=False):
            starts = g["_ov_start"].values
            ends = g["_ov_end"].values
            rates = g["ne_rate"].values
            drugs = g["drug"].values
            pts = np.unique(np.concatenate([starts, ends]))
            total = 0.0
            covered = 0.0
            peak = 0.0
            for a, b in zip(pts[:-1], pts[1:]):
                dur = (b - a) / np.timedelta64(1, "h")
                if dur <= 0:
                    continue
                active = (starts <= a) & (ends >= b)
                if not active.any():
                    continue
                # Across drugs the equivalents add. Within one drug they must
                # not: overlapping rows for the same drug are rate updates
                # recorded before the previous order closed, not two infusions
                # running at once, which is not something anyone administers.
                # 96.7% of the 816 overlapping same-drug groups carry differing
                # rates, and summing them inflated one cell 19-fold.
                instant = 0.0
                for d in np.unique(drugs[active]):
                    instant += float(rates[active][drugs[active] == d].max())
                total += instant * dur
                covered += dur
                peak = max(peak, instant)
            if mode == "max":
                rows.append((sid, tb, peak if covered > 0 else np.nan))
            else:
                rows.append((sid, tb, total / covered if covered > 0 else np.nan))
        parts.append(pd.DataFrame(rows, columns=["stay_id", "time_bin", "rate"]))

    return pd.concat(parts, ignore_index=True)


def extract_vasopressors_final(boundaries: pd.DataFrame, stay_ids: set,
                               cohort_onset: pd.Series) -> pd.DataFrame:
    """Duration-weighted mean norepinephrine-equivalent rate per interval."""
    print(f"Extracting vasopressors (aggregation={VASO_AGG})...")
    vaso = load_inputevents(stay_ids=stay_ids, item_ids=ALL_VASO_ITEMIDS)
    if vaso.empty:
        return boundaries[["stay_id", "time_bin"]].assign(vaso_total=0.0)

    id_to_name = _build_itemid_to_name(VASOPRESSOR_ITEMIDS)
    vaso = vaso[vaso["itemid"].isin(id_to_name)].copy()
    vaso["drug"] = vaso["itemid"].map(id_to_name)
    vaso = vaso[vaso["drug"].isin(NOREPI_EQUIV_FACTORS)]
    vaso["ne_rate"] = _norepi_equivalent_rate(vaso)

    alloc = _allocate_across_bins(vaso, cohort_onset)
    alloc = alloc[alloc["overlap_h"] > 0]
    alloc["weighted"] = alloc["ne_rate"] * alloc["overlap_h"]

    # ESTIMAND: max-in-window — the highest instantaneous norepinephrine-
    # equivalent rate reached at any point during the interval. Zero means no
    # vasopressor ran at any point in it.
    #
    # Why. It is what the two precedent implementations bin: Raghu et al. use
    # `max VP in` (sec. 3.3) and AI Clinician `max_dose_vaso`, so the action
    # grid stays directly comparable to both. It matches the clinical reading
    # of the vasopressor axis as the highest level of support chosen in the
    # window, and it is unchanged by duplicate or stale order rows, which a
    # mean is not.
    #
    # It costs duration and shape: a brief peak sets the level for the whole
    # four hours, so 0.3 for ten minutes shares an action with 0.3 sustained.
    # `prev_vaso` and the next state carry part of that.
    #
    # Multi-drug. Equivalents are additive at a given instant, so the rate at
    # each timeline segment sums the per-drug maxima and the interval takes
    # the maximum across segments. Summing each drug's own maximum instead
    # would add drugs that never ran together, which is 4.85% of treated
    # intervals.
    #
    # The conditional-mean alternative is kept behind SEPSIS_VASO_AGG for the
    # sensitivity audit reported alongside the results.
    per_bin = _intensity_on_shared_timeline(alloc)
    out = per_bin.rename(columns={"rate": "vaso_total"})
    print(f"  vasopressor intervals: {len(out):,}")
    return out


def extract_iv_final(boundaries: pd.DataFrame, stay_ids: set,
                     cohort_onset: pd.Series) -> pd.DataFrame:
    """Actual fluid volume delivered per interval, apportioned by overlap."""
    print("Extracting IV fluids (proportional allocation)...")
    iv = load_inputevents(stay_ids=stay_ids, item_ids=ALL_IV_ITEMIDS)
    if iv.empty:
        return boundaries[["stay_id", "time_bin"]].assign(iv_fluid_total=0.0)

    id_to_name = _build_itemid_to_name(IV_FLUID_ITEMIDS)
    iv = iv[iv["itemid"].isin(id_to_name)].copy()
    iv["amount"] = iv["amount"].fillna(0.0)
    # 13 rows are recorded in litres. Treating them as millilitres understates
    # them 1000-fold, and since the action bins are quartiles of the non-zero
    # distribution, a handful of extreme values moves every cut-point.
    uom = iv["amountuom"].fillna("").str.lower().str.strip()
    iv.loc[uom == "l", "amount"] *= 1000.0
    bad = ~uom.isin(["ml", "l", ""])
    if bad.any():
        print(f"  WARNING: {int(bad.sum())} fluid rows with unhandled "
              f"amountuom {sorted(uom[bad].unique())[:5]} — treated as mL")

    alloc = _allocate_across_bins(iv, cohort_onset)
    alloc["vol"] = alloc["amount"] * alloc["frac"]
    out = (alloc.groupby(["stay_id", "time_bin"])["vol"].sum()
                .reset_index().rename(columns={"vol": "iv_fluid_total"}))
    print(f"  fluid intervals: {len(out):,}")
    return out


def extract_urine_final(boundaries: pd.DataFrame, stay_ids: set,
                        cohort_onset: pd.Series) -> pd.DataFrame:
    """
    Urine output over the interval *preceding* each boundary.

    Output is a window quantity, so pairing interval k's boundary with the sum
    over interval k-1 keeps it past-only.
    """
    print("Extracting urine output (previous interval)...")
    uo = load_outputevents(stay_ids=stay_ids, item_ids=set(URINE_OUTPUT_ITEMIDS))
    if uo.empty:
        return boundaries[["stay_id", "time_bin"]].assign(urine_output=np.nan)

    uo = uo.copy()
    uo["charttime"] = pd.to_datetime(uo["charttime"])
    uo["onset"] = uo["stay_id"].map(cohort_onset)
    uo = uo.dropna(subset=["onset"])
    off = (uo["charttime"] - uo["onset"]).dt.total_seconds() / 3600
    uo["time_bin"] = np.floor(off / TIME_BIN_HOURS).astype(int)
    val_col = "value" if "value" in uo.columns else "valuenum"
    # Signed net, as the official concept computes it: irrigant instilled is
    # subtracted rather than added. 1,198 rows across 714 intervals, and the
    # windows it touches move by a median of 6 litres.
    irr = uo["itemid"] == GU_IRRIGANT_IN_ITEMID
    n_irr = int((irr & (uo[val_col] > 0)).sum())
    if n_irr:
        print(f"  urine: {n_irr:,} GU-irrigant-in rows negated")
        uo.loc[irr & (uo[val_col] > 0), val_col] = \
            -uo.loc[irr & (uo[val_col] > 0), val_col]
    agg = (uo.groupby(["stay_id", "time_bin"])[val_col].sum()
             .reset_index().rename(columns={val_col: "urine_output"}))
    # A net below zero means more irrigant went in than came out over the
    # window. That is not a measurement of renal function, so it is missing
    # rather than zero, which would assert anuria.
    neg = int((agg["urine_output"] < 0).sum())
    if neg:
        print(f"  urine: {neg:,} intervals with a negative net set to missing")
        agg.loc[agg["urine_output"] < 0, "urine_output"] = np.nan
    # Shift forward one interval so the state at k sees interval k-1
    agg["time_bin"] = agg["time_bin"] + 1
    return agg


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Previous-interval treatment exposure.

    Without these the state cannot separate a normal MAP that was never treated
    from one achieved by treating, which is the confound raised in clinical
    review. Both are shifted by one interval within each stay, so they describe
    what was already given before the current decision.
    """
    df = df.sort_values(["stay_id", "time_bin"]).copy()
    g = df.groupby("stay_id", sort=False)
    df["prev_vaso"] = g["vaso_total"].shift(1).fillna(0.0)
    df["prev_iv"] = g["iv_fluid_total"].shift(1).fillna(0.0)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# DRIVER
# ─────────────────────────────────────────────────────────────────────────────

def build_final_features(save: bool = True) -> pd.DataFrame:
    """Assemble the onset-anchored feature table for the final Sepsis-3 cohort."""
    cohort = pd.read_csv(FINAL_COHORT_PATH)
    for c in ("suspected_infection_time", "intime", "outtime",
              "episode_start", "episode_end"):
        cohort[c] = pd.to_datetime(cohort[c])
    print(f"Final cohort: {len(cohort):,} stays")

    boundaries = build_boundaries(cohort)
    print(f"Decision boundaries: {len(boundaries):,}")

    stay_ids = set(cohort["stay_id"])
    hadm_ids = set(cohort["hadm_id"].dropna().astype(int))
    hadm_to_stay = dict(zip(cohort["hadm_id"], cohort["stay_id"]))
    onset = cohort.set_index("stay_id")["suspected_infection_time"]

    # Each source is cached: reading chartevents and labevents costs ~6 minutes,
    # and a failure in a later stage should not make that price be paid twice.
    ensure_dir(FINAL_FEATURES_DIR)

    def cached(name, fn):
        p = os.path.join(FINAL_FEATURES_DIR, f"_part_{name}.parquet")
        if os.path.exists(p):
            print(f"  [{name}] cached")
            return pd.read_parquet(p)
        df = fn()
        df.to_parquet(p, index=False)
        return df

    parts = [
        cached("vitals", lambda: extract_vitals_asof(boundaries, stay_ids)),
        cached("labs", lambda: extract_labs_asof(boundaries, hadm_ids, hadm_to_stay)),
        cached("sofa", lambda: attach_official_sofa(boundaries)),
        cached("vaso", lambda: extract_vasopressors_final(boundaries, stay_ids, onset)),
        cached("iv", lambda: extract_iv_final(boundaries, stay_ids, onset)),
        cached("urine", lambda: extract_urine_final(boundaries, stay_ids, onset)),
    ]

    feats = boundaries[["stay_id", "time_bin", "boundary", "boundary_end"]].copy()
    for p in parts:
        feats = feats.merge(p, on=["stay_id", "time_bin"], how="left")

    feats["vaso_total"] = feats["vaso_total"].fillna(0.0)
    feats["iv_fluid_total"] = feats["iv_fluid_total"].fillna(0.0)
    feats = add_history_features(feats)

    # Static covariates and the outcome label
    static = cohort[["stay_id", "hadm_id", "subject_id", "anchor_age", "gender",
                     "icu_mortality", "intime", "suspected_infection_time",
                     "first_careunit", "los_hours"]]
    feats = feats.merge(static, on="stay_id", how="left")
    feats["hours_from_onset"] = feats["time_bin"] * TIME_BIN_HOURS
    feats["hours_since_admission"] = (
        (feats["boundary"] - feats["intime"]).dt.total_seconds() / 3600)

    feats = feats.sort_values(["stay_id", "time_bin"]).reset_index(drop=True)
    print(f"Feature table: {feats.shape[0]:,} rows x {feats.shape[1]} cols")

    if save:
        ensure_dir(FINAL_FEATURES_DIR)
        out = os.path.join(FINAL_FEATURES_DIR, "features.parquet")
        feats.to_parquet(out, index=False)
        print(f"  saved -> {out}")
    return feats


if __name__ == "__main__":
    build_final_features()
