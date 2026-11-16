"""Build the final first-episode Sepsis-3 cohort for MIMIC-IV v3.1.

The Sepsis-3 classification and onset timestamps come exclusively from the
official PhysioNet/MIT-LCP derived table.  Demographics, ICU timing, and the
90-day mortality label are joined from the local MIMIC-IV v3.1 files.

Run from the repository root::

    python -m src.final_cohort

This writes ``data/cohort/sepsis3_final.csv.gz`` and a machine-readable audit
without modifying any historical cohort or run namespace.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from config.clinical import (
    EPISODE_END_HOURS_FROM_ONSET,
    EPISODE_START_HOURS_FROM_ONSET,
    MIN_AGE,
    MIN_COMPLETE_EPISODE_INTERVALS,
    MORTALITY_HORIZON_DAYS,
    TIME_BIN_HOURS,
)
from config.paths import (
    FINAL_COHORT_AUDIT_PATH,
    FINAL_COHORT_PATH,
    SEPSIS3_SOURCE_PATH,
)
from src.cohort_extraction import compute_mortality_label
from src.data_loading import load_admissions, load_icustays, load_patients
from src.utils import ensure_dir


REQUIRED_SEPSIS3_COLUMNS = {
    "subject_id",
    "stay_id",
    "suspected_infection_time",
    "sofa_time",
    "sofa_score",
    "sepsis3",
}

# Exact, independent BigQuery audit for the approved MIMIC-IV v3.1 protocol.
# These checks intentionally fail if a partial/wrong-version export is used.
EXPECTED_V3_1_COUNTS = {
    "official_sepsis3_stays": 32_899,
    "adult_stays": 32_899,
    "trajectory_eligible_stays": 32_690,
    "final_first_episode_stays": 25_442,
    "final_deaths_90d": 6_370,
}


def _sha256(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_official_sepsis3(path: str = SEPSIS3_SOURCE_PATH) -> pd.DataFrame:
    """Load and validate the official derived Sepsis-3 export."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Official Sepsis-3 export not found: {path}. "
            "Run sql/export_sepsis3_v3_1.sql in BigQuery and save its complete "
            "CSV result to this path."
        )

    df = pd.read_csv(path)
    missing = REQUIRED_SEPSIS3_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Sepsis-3 export is missing columns: {sorted(missing)}")

    for col in ["suspected_infection_time", "sofa_time"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # BigQuery CSV encodes booleans as true/false, but tolerate an already
    # parsed bool column while rejecting ambiguous values.
    if df["sepsis3"].dtype != bool:
        normalized = df["sepsis3"].astype(str).str.strip().str.lower()
        if not normalized.isin({"true", "false"}).all():
            bad = sorted(normalized[~normalized.isin({"true", "false"})].unique())
            raise ValueError(f"Unexpected sepsis3 values: {bad}")
        df["sepsis3"] = normalized.eq("true")

    if not df["sepsis3"].all():
        raise ValueError("Official export contains rows where sepsis3 is not TRUE")
    if df["stay_id"].isna().any() or df["stay_id"].duplicated().any():
        raise ValueError("Official export must contain one non-null row per stay_id")
    if df["subject_id"].isna().any():
        raise ValueError("Official export contains null subject_id")
    if df["suspected_infection_time"].isna().any():
        raise ValueError("Official export contains missing/invalid onset timestamps")

    return df


def select_final_cohort(
    official: pd.DataFrame,
    icustays: pd.DataFrame,
    patients: pd.DataFrame,
    admissions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Apply the approved filters and first-eligible-episode rule.

    The first episode is selected *after* age and trajectory-overlap
    eligibility. Thus an earlier episode with fewer than two complete decision
    intervals does not exclude the patient's first eligible Sepsis-3 episode.
    """
    official = official.copy()
    icustays = icustays.copy()
    patients = patients.copy()
    admissions = admissions.copy()

    for frame, cols in [
        (official, ["suspected_infection_time", "sofa_time"]),
        (icustays, ["intime", "outtime"]),
        (admissions, ["admittime", "dischtime", "deathtime"]),
    ]:
        for col in cols:
            if col in frame.columns:
                frame[col] = pd.to_datetime(frame[col], errors="coerce")

    joined = official.merge(
        icustays[
            ["subject_id", "hadm_id", "stay_id", "intime", "outtime",
             "first_careunit", "last_careunit"]
        ],
        on=["subject_id", "stay_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(official):
        missing = len(official) - len(joined)
        raise ValueError(f"{missing} official Sepsis-3 stays did not match local icustays")

    joined = joined.merge(
        patients[["subject_id", "gender", "anchor_age", "dod"]],
        on="subject_id",
        how="left",
        validate="many_to_one",
    )
    joined = joined.merge(
        admissions[
            ["hadm_id", "admittime", "dischtime", "deathtime",
             "hospital_expire_flag", "race"]
        ],
        on="hadm_id",
        how="left",
        validate="many_to_one",
    )

    joined["los_hours"] = (
        (joined["outtime"] - joined["intime"]).dt.total_seconds() / 3600
    )
    joined["onset_from_icu_hours"] = (
        (joined["suspected_infection_time"] - joined["intime"])
        .dt.total_seconds()
        / 3600
    )

    audit: dict[str, int | float | str | dict] = {
        "official_sepsis3_stays": int(len(joined)),
    }

    adult = joined[joined["anchor_age"] >= MIN_AGE].copy()
    audit["adult_stays"] = int(len(adult))

    # Define the onset-centred episode before first-episode selection. A global
    # ICU-LOS filter would exclude valid short/long stays even when the episode
    # itself contains enough sequential decisions.
    adult["episode_start"] = pd.concat(
        [
            adult["intime"],
            adult["suspected_infection_time"]
            + pd.Timedelta(hours=EPISODE_START_HOURS_FROM_ONSET),
        ],
        axis=1,
    ).max(axis=1)
    adult["episode_end"] = pd.concat(
        [
            adult["outtime"],
            adult["suspected_infection_time"]
            + pd.Timedelta(hours=EPISODE_END_HOURS_FROM_ONSET),
        ],
        axis=1,
    ).min(axis=1)

    start_offset = (
        (adult["intime"] - adult["suspected_infection_time"])
        .dt.total_seconds()
        / 3600
    )
    end_offset = (
        (adult["outtime"] - adult["suspected_infection_time"])
        .dt.total_seconds()
        / 3600
    )
    first_bin = np.ceil(start_offset / TIME_BIN_HOURS).astype(int).clip(
        lower=EPISODE_START_HOURS_FROM_ONSET // TIME_BIN_HOURS
    )
    last_bin = (np.floor(end_offset / TIME_BIN_HOURS).astype(int) - 1).clip(
        upper=EPISODE_END_HOURS_FROM_ONSET // TIME_BIN_HOURS - 1
    )
    adult["first_full_time_bin"] = first_bin
    adult["last_full_time_bin"] = last_bin
    adult["n_full_4h_intervals"] = (last_bin - first_bin + 1).clip(lower=0)

    eligible = adult[
        adult["n_full_4h_intervals"] >= MIN_COMPLETE_EPISODE_INTERVALS
    ].copy()
    audit["excluded_fewer_than_min_complete_intervals"] = int(
        len(adult) - len(eligible)
    )
    audit["minimum_complete_intervals"] = MIN_COMPLETE_EPISODE_INTERVALS
    audit["trajectory_eligible_stays"] = int(len(eligible))
    audit["trajectory_eligible_patients"] = int(eligible["subject_id"].nunique())
    audit["eligible_onset_before_icu_stays"] = int(
        (eligible["onset_from_icu_hours"] < 0).sum()
    )
    audit["eligible_onset_0_to_48h_stays"] = int(
        eligible["onset_from_icu_hours"].between(0, 48, inclusive="both").sum()
    )
    audit["eligible_onset_after_48h_stays"] = int(
        (eligible["onset_from_icu_hours"] > 48).sum()
    )

    ranked = eligible.sort_values(
        ["subject_id", "suspected_infection_time", "stay_id"]
    ).copy()
    ranked["eligible_episode_number"] = (
        ranked.groupby("subject_id", sort=False).cumcount() + 1
    )
    final = ranked[ranked["eligible_episode_number"] == 1].copy()

    audit["excluded_repeat_episode_stays"] = int(len(ranked) - len(final))
    audit["final_first_episode_stays"] = int(len(final))
    audit["final_unique_patients"] = int(final["subject_id"].nunique())

    final["icu_mortality"] = compute_mortality_label(
        final, horizon_days=MORTALITY_HORIZON_DAYS
    )
    audit["final_deaths_90d"] = int(final["icu_mortality"].sum())
    audit["final_mortality_90d_pct"] = round(
        100 * float(final["icu_mortality"].mean()), 2
    )

    audit["patients_with_at_least_2_full_intervals"] = int(
        (final["n_full_4h_intervals"] >= 2).sum()
    )
    audit["patients_with_fewer_than_2_full_intervals"] = int(
        (final["n_full_4h_intervals"] < 2).sum()
    )

    if final["subject_id"].duplicated().any():
        raise AssertionError("First-episode rule failed: duplicate subject_id remains")
    if final["stay_id"].duplicated().any():
        raise AssertionError("Duplicate stay_id remains in final cohort")
    if not final["icu_mortality"].isin([0, 1]).all():
        raise AssertionError("90-day mortality labels must be binary and complete")
    if not (final["episode_start"] <= final["episode_end"]).all():
        raise AssertionError("Invalid episode bounds after ICU clipping")

    final = final.sort_values(
        ["subject_id", "suspected_infection_time", "stay_id"]
    ).reset_index(drop=True)
    return final, audit


def _assert_expected_v3_1_counts(audit: dict) -> None:
    failures = {
        key: {"expected": expected, "observed": audit.get(key)}
        for key, expected in EXPECTED_V3_1_COUNTS.items()
        if audit.get(key) != expected
    }
    if failures:
        raise AssertionError(
            "Final cohort does not reproduce the independent BigQuery audit: "
            + json.dumps(failures, sort_keys=True)
        )


def build_final_cohort(
    source_path: str = SEPSIS3_SOURCE_PATH,
    output_path: str = FINAL_COHORT_PATH,
    audit_path: str = FINAL_COHORT_AUDIT_PATH,
    save: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Build, verify, and optionally save the approved final cohort."""
    official = load_official_sepsis3(source_path)
    final, audit = select_final_cohort(
        official,
        load_icustays(),
        load_patients(),
        load_admissions(),
    )
    _assert_expected_v3_1_counts(audit)

    audit.update(
        {
            "dataset": "MIMIC-IV v3.1",
            "cohort_source": "physionet-data.mimiciv_3_1_derived.sepsis3",
            "cohort_query": "sql/export_sepsis3_v3_1.sql",
            "audit_query": "sql/audit_final_cohort_v3_1.sql",
            "independent_bigquery_audit_verified": "2026-08-19",
            "source_path": str(source_path),
            "source_sha256": _sha256(source_path),
            "first_episode_rule": (
                "earliest eligible Sepsis-3 episode per subject_id after "
                "age and complete onset-centred interval filters"
            ),
            "mortality_horizon_days": MORTALITY_HORIZON_DAYS,
            "mortality_anchor": "ICU admission (intime)",
            "time_bin_hours": TIME_BIN_HOURS,
            "onset_inclusion_hours_from_icu": None,
            "episode_hours_from_onset": [
                EPISODE_START_HOURS_FROM_ONSET,
                EPISODE_END_HOURS_FROM_ONSET,
            ],
        }
    )

    if save:
        ensure_dir(str(Path(output_path).parent))
        final.to_csv(output_path, index=False, compression="gzip")
        ensure_dir(str(Path(audit_path).parent))
        with open(audit_path, "w") as handle:
            json.dump(audit, handle, indent=2, sort_keys=True)

    return final, audit


def main() -> None:
    final, audit = build_final_cohort()
    print("Final first-episode Sepsis-3 cohort built successfully")
    print(f"  stays/patients: {len(final):,}")
    print(
        f"  90-day deaths: {audit['final_deaths_90d']:,} "
        f"({audit['final_mortality_90d_pct']:.2f}%)"
    )
    print(f"  cohort: {FINAL_COHORT_PATH}")
    print(f"  audit:  {FINAL_COHORT_AUDIT_PATH}")


if __name__ == "__main__":
    main()
