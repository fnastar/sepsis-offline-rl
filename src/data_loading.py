"""
data_loading.py: Memory-efficient data loaders for MIMIC-IV v3.1 tables.

Small tables (admissions, patients, icustays, diagnoses) are read in full.
Large tables (chartevents ~3.3GB, labevents ~2.4GB) are read in chunks,
filtered by item_ids first (drops ~95% of rows), then by stay_ids/hadm_ids.
"""
import pandas as pd
from tqdm import tqdm

from config.paths import *
from config.clinical import ALL_CHART_ITEMIDS, ALL_LAB_ITEMIDS
from src.utils import dataframe_from_csv, chunked_read_csv


# ═══════════════════════════════════════════════════════════════════════════
# SMALL TABLE LOADERS
# ═══════════════════════════════════════════════════════════════════════════

def load_admissions() -> pd.DataFrame:
    """Load admissions table with parsed dates."""
    df = dataframe_from_csv(ADMISSIONS_PATH)
    for col in ["admittime", "dischtime", "deathtime", "edregtime", "edouttime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def load_patients() -> pd.DataFrame:
    """Load patients table."""
    return dataframe_from_csv(PATIENTS_PATH)


def load_icustays() -> pd.DataFrame:
    """Load ICU stays table with parsed dates."""
    df = dataframe_from_csv(ICUSTAYS_PATH)
    for col in ["intime", "outtime"]:
        df[col] = pd.to_datetime(df[col])
    return df


def load_diagnoses_icd() -> pd.DataFrame:
    """Load diagnoses_icd table."""
    return dataframe_from_csv(DIAGNOSES_ICD_PATH, dtype={"icd_code": str})


def load_d_icd_diagnoses() -> pd.DataFrame:
    """Load ICD diagnosis definitions."""
    return dataframe_from_csv(D_ICD_DIAGNOSES_PATH, dtype={"icd_code": str})


def load_d_items() -> pd.DataFrame:
    """Load d_items dictionary table."""
    return dataframe_from_csv(D_ITEMS_PATH)


def load_d_labitems() -> pd.DataFrame:
    """Load d_labitems dictionary table."""
    return dataframe_from_csv(D_LABITEMS_PATH)


# ═══════════════════════════════════════════════════════════════════════════
# LARGE TABLE LOADERS (CHUNKED)
# ═══════════════════════════════════════════════════════════════════════════

def load_chartevents(
    stay_ids: set | None = None,
    item_ids: set | None = None,
    usecols: list | None = None,
) -> pd.DataFrame:
    """
    Load chartevents with chunked reading. Filters by itemid first,
    then by stay_id.

    Parameters
    ----------
    stay_ids : set, optional
        If provided, keep only rows matching these stay_ids.
    item_ids : set, optional
        If provided, keep only rows matching these itemids.
        Defaults to ALL_CHART_ITEMIDS.
    usecols : list, optional
        Columns to read. Defaults to essential columns.
    """
    if item_ids is None:
        item_ids = ALL_CHART_ITEMIDS
    if usecols is None:
        usecols = ["stay_id", "charttime", "itemid", "value", "valuenum", "valueuom"]

    compression = "gzip" if CHARTEVENTS_PATH.endswith(".gz") else None
    chunks = []
    reader = pd.read_csv(
        CHARTEVENTS_PATH,
        compression=compression,
        chunksize=CHUNK_SIZE,
        usecols=usecols,
        low_memory=False,
    )
    for chunk in tqdm(reader, desc="Reading chartevents"):
        chunk = chunk[chunk["itemid"].isin(item_ids)]
        if stay_ids is not None:
            chunk = chunk[chunk["stay_id"].isin(stay_ids)]
        if len(chunk) > 0:
            chunks.append(chunk)

    if not chunks:
        return pd.DataFrame(columns=usecols)
    df = pd.concat(chunks, ignore_index=True)
    df["charttime"] = pd.to_datetime(df["charttime"])
    return df


def load_labevents(
    hadm_ids: set | None = None,
    item_ids: set | None = None,
    usecols: list | None = None,
) -> pd.DataFrame:
    """
    Load labevents with chunked reading. Filters by itemid first,
    then by hadm_id.
    """
    if item_ids is None:
        item_ids = ALL_LAB_ITEMIDS
    if usecols is None:
        usecols = ["labevent_id", "subject_id", "hadm_id", "itemid",
                    "charttime", "value", "valuenum", "valueuom"]

    compression = "gzip" if LABEVENTS_PATH.endswith(".gz") else None
    chunks = []
    reader = pd.read_csv(
        LABEVENTS_PATH,
        compression=compression,
        chunksize=CHUNK_SIZE,
        usecols=usecols,
        low_memory=False,
    )
    for chunk in tqdm(reader, desc="Reading labevents"):
        chunk = chunk[chunk["itemid"].isin(item_ids)]
        if hadm_ids is not None:
            chunk = chunk[chunk["hadm_id"].isin(hadm_ids)]
        if len(chunk) > 0:
            chunks.append(chunk)

    if not chunks:
        return pd.DataFrame(columns=usecols)
    df = pd.concat(chunks, ignore_index=True)
    df["charttime"] = pd.to_datetime(df["charttime"])
    return df


def load_inputevents(
    stay_ids: set | None = None,
    item_ids: set | None = None,
    usecols: list | None = None,
) -> pd.DataFrame:
    """Load inputevents, optionally filtered by stay_id and itemid."""
    if usecols is None:
        usecols = ["stay_id", "starttime", "endtime", "itemid",
                    "amount", "amountuom", "rate", "rateuom",
                    "ordercategoryname", "patientweight"]

    compression = "gzip" if INPUTEVENTS_PATH.endswith(".gz") else None
    chunks = []
    reader = pd.read_csv(
        INPUTEVENTS_PATH,
        compression=compression,
        chunksize=CHUNK_SIZE,
        usecols=usecols,
        low_memory=False,
    )
    for chunk in tqdm(reader, desc="Reading inputevents"):
        if item_ids is not None:
            chunk = chunk[chunk["itemid"].isin(item_ids)]
        if stay_ids is not None:
            chunk = chunk[chunk["stay_id"].isin(stay_ids)]
        if len(chunk) > 0:
            chunks.append(chunk)

    if not chunks:
        return pd.DataFrame(columns=usecols)
    df = pd.concat(chunks, ignore_index=True)
    for col in ["starttime", "endtime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def load_outputevents(
    stay_ids: set | None = None,
    item_ids: set | None = None,
    usecols: list | None = None,
) -> pd.DataFrame:
    """Load outputevents (primarily urine output)."""
    if usecols is None:
        usecols = ["stay_id", "charttime", "itemid", "value", "valueuom"]

    compression = "gzip" if OUTPUTEVENTS_PATH.endswith(".gz") else None
    chunks = []
    reader = pd.read_csv(
        OUTPUTEVENTS_PATH,
        compression=compression,
        chunksize=CHUNK_SIZE,
        usecols=usecols,
        low_memory=False,
    )
    for chunk in tqdm(reader, desc="Reading outputevents"):
        if item_ids is not None:
            chunk = chunk[chunk["itemid"].isin(item_ids)]
        if stay_ids is not None:
            chunk = chunk[chunk["stay_id"].isin(stay_ids)]
        if len(chunk) > 0:
            chunks.append(chunk)

    if not chunks:
        return pd.DataFrame(columns=usecols)
    df = pd.concat(chunks, ignore_index=True)
    df["charttime"] = pd.to_datetime(df["charttime"])
    return df


def load_prescriptions(hadm_ids: set | None = None) -> pd.DataFrame:
    """Load prescriptions table, optionally filtered by hadm_id."""
    df = dataframe_from_csv(PRESCRIPTIONS_PATH)
    if hadm_ids is not None:
        df = df[df["hadm_id"].isin(hadm_ids)]
    for col in ["starttime", "stoptime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_microbiologyevents(hadm_ids: set | None = None) -> pd.DataFrame:
    """Load microbiology events, optionally filtered by hadm_id."""
    df = dataframe_from_csv(MICROBIOLOGYEVENTS_PATH)
    if hadm_ids is not None:
        df = df[df["hadm_id"].isin(hadm_ids)]
    if "charttime" in df.columns:
        df["charttime"] = pd.to_datetime(df["charttime"])
    if "chartdate" in df.columns:
        df["chartdate"] = pd.to_datetime(df["chartdate"])
    return df
