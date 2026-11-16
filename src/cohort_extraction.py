"""
cohort_extraction.py: Extract sepsis cohort from MIMIC-IV.

Strategy:
1. Build base ICU cohort (age >= 18, LOS filters, mortality flags)
2. Identify sepsis via:
   a. Explicit sepsis ICD codes (A40, A41, R652, 99591, 99592, 78552)
   b. Suspected infection: ICD infection codes OR microbiology cultures
      OR antibiotic prescriptions within ±24h of ICU admission
3. Save cohort to data/cohort/sepsis_cohort.csv.gz
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from config.paths import COHORT_DIR
from config.clinical import (
    SEPSIS_ICD10_CODES, SEPSIS_ICD9_CODES,
    INFECTION_ICD10_PREFIXES, INFECTION_ICD9_PREFIXES,
    ANTIBIOTIC_KEYWORDS, ANTIBIOTIC_ROUTES,
    MIN_AGE, MIN_ICU_STAY_HOURS, MAX_ICU_STAY_DAYS,
)
from src.utils import ensure_dir
from src.data_loading import (
    load_admissions, load_patients, load_icustays,
    load_diagnoses_icd, load_prescriptions, load_microbiologyevents,
)


def compute_mortality_label(
    cohort: pd.DataFrame, horizon_days: int | None = None
) -> pd.Series:
    """
    Binary mortality outcome that drives the terminal RL reward.

    Parameters
    ----------
    cohort : DataFrame with `intime`, `outtime`, `deathtime`, `dod` columns.
    horizon_days : int or None
        - None  → in-ICU death: `deathtime` within [intime, outtime + 24h]
          (in-hospital deaths only). This is the original definition.
        - N     → N-day all-cause mortality from ICU admission: `dod` within
          [intime, intime + N days]. Uses `dod`, so post-discharge and
          out-of-hospital deaths are captured — the sepsis-RL convention
          (Komorowski et al. 2018 use 90-day). `dod` is date-only in MIMIC-IV,
          so the window is compared at day granularity (a same-day death = day 0).

    Returns a 0/1 int Series aligned to `cohort`.
    """
    intime = pd.to_datetime(cohort["intime"], errors="coerce")
    if horizon_days is None:
        deathtime = pd.to_datetime(cohort["deathtime"], errors="coerce")
        outtime = pd.to_datetime(cohort["outtime"], errors="coerce")
        died = (
            deathtime.notna()
            & (deathtime >= intime)
            & (deathtime <= outtime + pd.Timedelta(hours=24))
        )
    else:
        dod = pd.to_datetime(cohort["dod"], errors="coerce")
        days = (dod.dt.normalize() - intime.dt.normalize()).dt.days
        died = dod.notna() & (days >= 0) & (days <= horizon_days)
    return died.astype(int)


def build_base_icu_cohort(mortality_horizon_days: int | None = None) -> pd.DataFrame:
    """
    Build base ICU cohort with demographic and outcome information.

    Filters:
    - Age >= 18
    - ICU LOS >= MIN_ICU_STAY_HOURS
    - ICU LOS <= MAX_ICU_STAY_DAYS
    - First ICU stay per hospital admission

    `mortality_horizon_days` sets the outcome definition (see
    `compute_mortality_label`): None = in-ICU death (default, unchanged),
    an int = N-day mortality from ICU admission via `dod`.

    Returns DataFrame with columns:
        subject_id, hadm_id, stay_id, gender, anchor_age,
        intime, outtime, los_hours, admittime, dischtime, deathtime,
        hospital_expire_flag, icu_mortality, first_careunit
    """
    print("Loading base tables...")
    admissions = load_admissions()
    patients = load_patients()
    icustays = load_icustays()

    # Merge patients + admissions + icustays
    cohort = icustays.merge(
        patients[["subject_id", "gender", "anchor_age", "dod"]],
        on="subject_id", how="left",
    )
    cohort = cohort.merge(
        admissions[["hadm_id", "admittime", "dischtime", "deathtime",
                     "hospital_expire_flag", "race"]],
        on="hadm_id", how="left",
    )

    # Compute ICU length of stay in hours
    cohort["los_hours"] = (
        (cohort["outtime"] - cohort["intime"]).dt.total_seconds() / 3600
    )

    # Age filter
    cohort = cohort[cohort["anchor_age"] >= MIN_AGE].copy()
    print(f"  After age >= {MIN_AGE}: {len(cohort)} stays")

    # LOS filters
    cohort = cohort[cohort["los_hours"] >= MIN_ICU_STAY_HOURS].copy()
    print(f"  After LOS >= {MIN_ICU_STAY_HOURS}h: {len(cohort)} stays")

    cohort = cohort[cohort["los_hours"] <= MAX_ICU_STAY_DAYS * 24].copy()
    print(f"  After LOS <= {MAX_ICU_STAY_DAYS}d: {len(cohort)} stays")

    # Keep first ICU stay per hospital admission
    cohort = cohort.sort_values("intime").groupby("hadm_id").first().reset_index()
    print(f"  After first ICU stay per admission: {len(cohort)} stays")

    # Mortality flag (terminal-reward outcome). None = in-ICU (default);
    # an int = N-day mortality from ICU admission via dod.
    cohort["icu_mortality"] = compute_mortality_label(cohort, mortality_horizon_days)

    horizon_desc = ("in-ICU (+24h)" if mortality_horizon_days is None
                    else f"{mortality_horizon_days}-day")
    print(f"Base ICU cohort: {len(cohort)} stays, "
          f"{horizon_desc} mortality: {cohort['icu_mortality'].sum()} "
          f"({cohort['icu_mortality'].mean():.1%})")

    return cohort


def identify_sepsis_icd(cohort: pd.DataFrame) -> pd.Series:
    """
    Identify patients with explicit sepsis ICD codes.

    Returns a boolean Series indexed like cohort indicating
    whether each stay has a sepsis ICD code.
    """
    diagnoses = load_diagnoses_icd()
    hadm_ids = set(cohort["hadm_id"])
    diagnoses = diagnoses[diagnoses["hadm_id"].isin(hadm_ids)]

    # ICD-10 exact match
    icd10_mask = (
        (diagnoses["icd_version"] == 10)
        & diagnoses["icd_code"].isin(SEPSIS_ICD10_CODES)
    )
    # ICD-9 exact match
    icd9_mask = (
        (diagnoses["icd_version"] == 9)
        & diagnoses["icd_code"].isin(SEPSIS_ICD9_CODES)
    )

    sepsis_hadm_ids = set(diagnoses.loc[icd10_mask | icd9_mask, "hadm_id"])
    print(f"  Explicit sepsis ICD codes: {len(sepsis_hadm_ids)} admissions")

    return cohort["hadm_id"].isin(sepsis_hadm_ids)


def identify_suspected_infection(cohort: pd.DataFrame) -> pd.Series:
    """
    Identify patients with suspected infection based on:
    1. Infection-related ICD codes
    2. Microbiology cultures ordered
    3. Antibiotic prescriptions

    Returns a boolean Series indexed like cohort.
    """
    hadm_ids = set(cohort["hadm_id"])

    # 1. Infection ICD codes
    diagnoses = load_diagnoses_icd()
    diagnoses = diagnoses[diagnoses["hadm_id"].isin(hadm_ids)]

    infection_icd10 = diagnoses[
        (diagnoses["icd_version"] == 10)
        & diagnoses["icd_code"].apply(
            lambda x: any(str(x).startswith(p) for p in INFECTION_ICD10_PREFIXES)
        )
    ]["hadm_id"]

    infection_icd9 = diagnoses[
        (diagnoses["icd_version"] == 9)
        & diagnoses["icd_code"].apply(
            lambda x: any(str(x).startswith(p) for p in INFECTION_ICD9_PREFIXES)
        )
    ]["hadm_id"]

    infection_icd_hadms = set(infection_icd10) | set(infection_icd9)
    print(f"  Infection ICD codes: {len(infection_icd_hadms)} admissions")

    # 2. Microbiology cultures
    print("  Loading microbiology events...")
    micro = load_microbiologyevents(hadm_ids=hadm_ids)
    micro_hadms = set(micro["hadm_id"].dropna().astype(int))
    print(f"  Microbiology cultures: {len(micro_hadms)} admissions")

    # 3. Antibiotic prescriptions
    print("  Loading prescriptions...")
    prescriptions = load_prescriptions(hadm_ids=hadm_ids)
    abx_mask = prescriptions["drug"].str.lower().apply(
        lambda x: any(kw in str(x) for kw in ANTIBIOTIC_KEYWORDS)
        if pd.notna(x) else False
    )
    if "route" in prescriptions.columns:
        route_mask = prescriptions["route"].isin(ANTIBIOTIC_ROUTES)
        abx_mask = abx_mask & route_mask
    abx_hadms = set(prescriptions.loc[abx_mask, "hadm_id"])
    print(f"  Antibiotic prescriptions: {len(abx_hadms)} admissions")

    # Union of all suspected infection indicators
    suspected_hadms = infection_icd_hadms | micro_hadms | abx_hadms
    print(f"  Total suspected infection: {len(suspected_hadms)} admissions")

    return cohort["hadm_id"].isin(suspected_hadms)


def extract_sepsis_cohort(save: bool = True) -> pd.DataFrame:
    """
    Extract sepsis cohort using ICD codes + suspected infection.

    A patient is included if they have:
    - An explicit sepsis ICD code, OR
    - Suspected infection (infection ICD codes OR microbiology cultures
      OR antibiotic prescriptions)

    Parameters
    ----------
    save : bool
        If True, save to data/cohort/sepsis_cohort.csv.gz

    Returns
    -------
    pd.DataFrame
        Sepsis cohort with demographic and outcome information.
    """
    # Build base cohort
    cohort = build_base_icu_cohort()

    # Identify sepsis
    print("\nIdentifying sepsis patients...")
    sepsis_icd = identify_sepsis_icd(cohort)
    suspected_infection = identify_suspected_infection(cohort)

    # Combine: explicit sepsis OR suspected infection
    cohort["sepsis_icd"] = sepsis_icd.astype(int)
    cohort["suspected_infection"] = suspected_infection.astype(int)
    cohort["sepsis"] = ((sepsis_icd) | (suspected_infection)).astype(int)

    # Filter to sepsis patients only
    sepsis_cohort = cohort[cohort["sepsis"] == 1].copy()
    print(f"\nFinal sepsis cohort: {len(sepsis_cohort)} stays")
    print(f"  Explicit sepsis ICD: {sepsis_cohort['sepsis_icd'].sum()}")
    print(f"  Suspected infection: {sepsis_cohort['suspected_infection'].sum()}")
    print(f"  ICU mortality: {sepsis_cohort['icu_mortality'].sum()} "
          f"({sepsis_cohort['icu_mortality'].mean():.1%})")

    if save:
        ensure_dir(COHORT_DIR)
        out_path = os.path.join(COHORT_DIR, "sepsis_cohort.csv.gz")
        sepsis_cohort.to_csv(out_path, index=False, compression="gzip")
        print(f"  Saved to {out_path}")

    return sepsis_cohort
