"""
paths.py: File paths for MIMIC-IV v3.1 data and output directories.
"""
import os

# ── MIMIC-IV root ──────────────────────────────────────────────────────────
MIMIC_ROOT = os.environ.get("MIMIC_ROOT", os.path.abspath("data/mimic-iv-3.1"))
HOSP_DIR = os.path.join(MIMIC_ROOT, "hosp")
ICU_DIR = os.path.join(MIMIC_ROOT, "icu")

# ── Hospital tables ────────────────────────────────────────────────────────
ADMISSIONS_PATH = os.path.join(HOSP_DIR, "admissions.csv.gz")
PATIENTS_PATH = os.path.join(HOSP_DIR, "patients.csv.gz")
DIAGNOSES_ICD_PATH = os.path.join(HOSP_DIR, "diagnoses_icd.csv.gz")
D_ICD_DIAGNOSES_PATH = os.path.join(HOSP_DIR, "d_icd_diagnoses.csv.gz")
LABEVENTS_PATH = os.path.join(HOSP_DIR, "labevents.csv.gz")
MICROBIOLOGYEVENTS_PATH = os.path.join(HOSP_DIR, "microbiologyevents.csv.gz")
PRESCRIPTIONS_PATH = os.path.join(HOSP_DIR, "prescriptions.csv.gz")
D_LABITEMS_PATH = os.path.join(HOSP_DIR, "d_labitems.csv.gz")

# ── ICU tables ─────────────────────────────────────────────────────────────
ICUSTAYS_PATH = os.path.join(ICU_DIR, "icustays.csv.gz")
CHARTEVENTS_PATH = os.path.join(ICU_DIR, "chartevents.csv.gz")
D_ITEMS_PATH = os.path.join(ICU_DIR, "d_items.csv.gz")
INPUTEVENTS_PATH = os.path.join(ICU_DIR, "inputevents.csv.gz")
OUTPUTEVENTS_PATH = os.path.join(ICU_DIR, "outputevents.csv.gz")

# ── Output directories ────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
COHORT_DIR = os.path.join(DATA_DIR, "cohort")
FEATURES_DIR = os.environ.get(
    "SEPSIS_FEATURES_DIR", os.path.join(DATA_DIR, "features"))

# Isolated inputs/outputs for the final MIMIC-IV v3.1 Sepsis-3 cohort.  The
# official source is a small read-only export of
# physionet-data.mimiciv_3_1_derived.sepsis3; all large event tables continue
# to be read from the local MIMIC-IV files above.
SEPSIS3_SOURCE_PATH = os.environ.get(
    "SEPSIS3_SOURCE_PATH",
    os.path.join(COHORT_DIR, "sepsis3_official_v3_1.csv.gz"),
)
FINAL_COHORT_PATH = os.environ.get(
    "SEPSIS_FINAL_COHORT",
    os.path.join(COHORT_DIR, "sepsis3_final.csv.gz"),
)
FINAL_COHORT_AUDIT_PATH = os.environ.get(
    "SEPSIS_FINAL_COHORT_AUDIT",
    os.path.join(PROJECT_ROOT, "outputs", "final_preflight", "cohort_audit.json"),
)

# PROCESSED_DIR, OUTPUT_BASE and FIG_DIR are env-overridable so an alternate
# label version (e.g. 90-day mortality) can be built into a parallel namespace
# without touching the default 24h pipeline. With no env vars set, these resolve
# to the original paths, so existing runs are byte-for-byte unaffected.
PROCESSED_DIR = os.environ.get(
    "SEPSIS_PROCESSED_DIR", os.path.join(DATA_DIR, "processed"))
OUTPUT_BASE = os.environ.get("SEPSIS_OUTPUT_BASE", "outputs/runs")
FIG_DIR = os.environ.get("SEPSIS_FIG_DIR", "outputs/figures")

# Behaviour-policy weights must match the dataset's action bins. Use a
# separate output directory when rebuilding the action space.
BEHAVIOR_POLICY_DIR = os.environ.get(
    "SEPSIS_BP_DIR", "outputs/behavior_policy")

# ── Chunk size for large CSV reading ──────────────────────────────────────
CHUNK_SIZE = 5_000_000
