"""
clinical.py: Clinical parameters for sepsis cohort extraction, SOFA scoring,
and feature extraction from MIMIC-IV.
"""
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# SEPSIS ICD CODES
# ═══════════════════════════════════════════════════════════════════════════

# ICD-10 codes for explicit sepsis diagnosis
SEPSIS_ICD10_CODES = [
    "A40",   # Streptococcal sepsis
    "A400", "A401", "A403", "A408", "A409",
    "A41",   # Other sepsis
    "A410", "A4101", "A4102",
    "A411",
    "A412",
    "A413",
    "A414", "A4150", "A4151", "A4152", "A4153", "A4159",
    "A418", "A4181", "A4189",
    "A419",
    "R6520",  # Severe sepsis without septic shock
    "R6521",  # Severe sepsis with septic shock
]

# ICD-9 codes for explicit sepsis diagnosis
SEPSIS_ICD9_CODES = [
    "99591",  # Sepsis
    "99592",  # Severe sepsis
    "78552",  # Septic shock
]

# ICD-10 prefixes indicating infection (for suspected infection criterion)
INFECTION_ICD10_PREFIXES = [
    "A0", "A1", "A2", "A3", "A4",  # Certain infectious diseases
    "A5", "A6", "A7", "A8", "A9",
    "B0", "B1", "B2", "B3", "B4",  # Other infectious diseases
    "B5", "B6", "B7", "B8", "B9",
    "J0", "J1", "J2",              # Respiratory infections (pneumonia etc.)
    "J3", "J4",
    "J85", "J86",                   # Lung abscess, pyothorax
    "K35", "K57", "K61", "K63",    # Abdominal infections
    "K65",                          # Peritonitis
    "L0",                           # Skin infections
    "N10", "N11", "N12",           # Kidney/UTI infections
    "N136", "N151", "N30", "N34", "N39",
    "N70", "N71", "N73",           # Pelvic inflammatory disease
    "T814",                         # Surgical site infection
]

# ICD-9 prefixes indicating infection
INFECTION_ICD9_PREFIXES = [
    "001", "002", "003", "004", "005", "006", "007", "008", "009",
    "010", "011", "012", "013", "014", "015", "016", "017", "018",
    "020", "021", "022", "023", "024", "025", "026", "027",
    "030", "031", "032", "033", "034", "035", "036", "037", "038", "039",
    "040", "041",
    "480", "481", "482", "483", "484", "485", "486",  # Pneumonia
    "567",  # Peritonitis
    "590",  # Pyelonephritis
    "599",  # UTI
    "680", "681", "682",  # Skin infections
]

# Antibiotic drug categories for suspected infection detection
ANTIBIOTIC_ROUTES = ["IV", "ORAL", "IM"]
ANTIBIOTIC_KEYWORDS = [
    "cillin", "cef", "mycin", "cycline", "floxacin", "oxacin",
    "azole", "sulfa", "trim", "vancomycin", "meropenem", "imipenem",
    "ertapenem", "doripenem", "aztreonam", "linezolid", "daptomycin",
    "colistin", "polymyxin", "metronidazole", "clindamycin",
    "piperacillin", "tazobactam", "ampicillin", "sulbactam",
    "amoxicillin", "clavulanate", "levofloxacin", "ciprofloxacin",
    "moxifloxacin", "ceftriaxone", "cefepime", "ceftazidime",
    "gentamicin", "tobramycin", "amikacin",
]

# ═══════════════════════════════════════════════════════════════════════════
# MIMIC-IV ITEM IDS (d_items / d_labitems)
# ═══════════════════════════════════════════════════════════════════════════

# Vital signs (chartevents itemids)
VITALS_ITEMIDS = {
    "heart_rate": [220045],
    "sbp": [220050, 220179],          # Invasive & non-invasive
    "dbp": [220051, 220180],
    "mbp": [220052, 220181],
    "resp_rate": [220210, 224690],
    "spo2": [220277],
    "temperature": [223761, 223762],   # F and C
    "gcs_eye": [220739],
    "gcs_motor": [223901],
    "gcs_verbal": [223900],
    # 223835 only. 227009 and 227010 are FiO2_ApacheIV_old and FiO2_ApacheIV,
    # fields of a severity score rather than bedside measurements: different
    # meaning, and a score-window summary rather than a reading at a time. A
    # value-range filter cannot fix a semantic mismatch. Six rows in all of
    # MIMIC-IV carry them, so this costs nothing and removes an assumption.
    "fio2": [223835],
}

# Lab values (labevents itemids)
LAB_ITEMIDS = {
    "lactate": [50813],
    "wbc": [51301],
    "platelets": [51265],
    "creatinine": [50912],
    "bilirubin_total": [50885],
    "hemoglobin": [51222],
    "hematocrit": [51221],
    "bun": [51006],
    "sodium": [50983],
    "potassium": [50971],
    "chloride": [50902],
    "bicarbonate": [50882],
    "glucose": [50931],
    "albumin": [50862],
    "calcium": [50893],
    "magnesium": [50960],
    "phosphate": [50970],
    "inr": [51237],
    "pt": [51274],
    "ptt": [51275],
    "ph_blood": [50820],
    "pao2": [50821],
    "paco2": [50818],
    "base_excess": [50802],
    "bands": [51144],
    "crp": [50889],
}

# Vasopressor itemids (inputevents)
VASOPRESSOR_ITEMIDS = {
    "norepinephrine": [221906],
    "epinephrine": [221289],
    "dopamine": [221662],
    "dobutamine": [221653],
    "phenylephrine": [221749],
    "vasopressin": [222315],
    "milrinone": [221986],
}

# IV fluid itemids (inputevents)
IV_FLUID_ITEMIDS = {
    "normal_saline": [225158],
    "lactated_ringers": [225828],
    "dextrose_5": [225823],
    "albumin_5": [220862],
    "albumin_25": [220864],
    "fresh_frozen_plasma": [220970],
    "packed_rbc": [225168],
    "platelets_transfusion": [225170],
}

# Urine output itemids (outputevents)
URINE_OUTPUT_ITEMIDS = [
    226559, 226560, 226561, 226584, 226563, 226564,
    226565, 226567, 226557, 226558, 227488, 227489,
]

# Flat set of all chart itemids for filtering
ALL_CHART_ITEMIDS = set()
for ids in VITALS_ITEMIDS.values():
    ALL_CHART_ITEMIDS.update(ids)

ALL_LAB_ITEMIDS = set()
for ids in LAB_ITEMIDS.values():
    ALL_LAB_ITEMIDS.update(ids)

ALL_VASO_ITEMIDS = set()
for ids in VASOPRESSOR_ITEMIDS.values():
    ALL_VASO_ITEMIDS.update(ids)

ALL_IV_ITEMIDS = set()
for ids in IV_FLUID_ITEMIDS.values():
    ALL_IV_ITEMIDS.update(ids)

# ═══════════════════════════════════════════════════════════════════════════
# SOFA SCORING THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════

# Respiration: PaO2/FiO2 ratio thresholds → SOFA score
SOFA_RESP_THRESHOLDS = [
    (400, 0),
    (300, 1),
    (200, 2),   # 2 if not ventilated, 3 if ventilated
    (100, 3),   # 3 if ventilated
    (0, 4),     # 4 if ventilated
]

# Coagulation: Platelets (×10³/µL)
SOFA_COAG_THRESHOLDS = [
    (150, 0),
    (100, 1),
    (50, 2),
    (20, 3),
    (0, 4),
]

# Liver: Bilirubin (mg/dL)
SOFA_LIVER_THRESHOLDS = [
    (1.2, 0),
    (2.0, 1),
    (6.0, 2),
    (12.0, 3),
    (np.inf, 4),
]

# Cardiovascular: MAP or vasopressor dose
# Handled programmatically in sofa.py

# CNS: GCS score
SOFA_CNS_THRESHOLDS = [
    (15, 0),
    (13, 1),
    (10, 2),
    (6, 3),
    (0, 4),
]

# Renal: Creatinine (mg/dL) or urine output
SOFA_RENAL_THRESHOLDS = [
    (1.2, 0),
    (2.0, 1),
    (3.5, 2),
    (5.0, 3),
    (np.inf, 4),
]

# ═══════════════════════════════════════════════════════════════════════════
# TIME WINDOW PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

TIME_BIN_HOURS = 4         # Bin size for time-series aggregation
# Historical broad-cohort parameters retained only for reproducibility of the
# legacy pipeline. The final Sepsis-3 cohort does not apply either LOS filter.
MIN_ICU_STAY_HOURS = 24
MAX_ICU_STAY_DAYS = 30
MIN_AGE = 18               # Minimum patient age

# Final Sepsis-3 study protocol (MIMIC-IV v3.1). Construct each MDP trajectory
# from 24 h before through 48 h after suspected-infection onset, clipped to the
# observed ICU stay. Unlike the legacy cohort above, final-cohort eligibility
# has no global ICU-LOS or ICU-relative onset restriction: it requires enough
# overlap to form at least two complete decision intervals.
EPISODE_START_HOURS_FROM_ONSET = -24
EPISODE_END_HOURS_FROM_ONSET = 48
MIN_COMPLETE_EPISODE_INTERVALS = 2
MORTALITY_HORIZON_DAYS = 90
