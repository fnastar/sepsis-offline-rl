"""
final_preprocessing.py: MDP transitions for the final Sepsis-3 run.

Parallel to `preprocessing.py`; the legacy module is untouched so published
results stay reproducible. Writes to `data/processed_90d_final/`.

Differences from legacy:

* SPARSE REWARD. Terminal +15 / -15 on 90-day survival, nothing in between.
  The legacy shaping term was -c*dSOFA on a SOFA that could not reach its full
  range; and with episodes now capped at 18 intervals the discounted terminal
  signal reaches the first decision at 0.99^18 = 0.83, so there is little
  horizon left for shaping to bridge.
* ACTION BINS from the data. Level 0 is exactly no treatment; levels 1-4 are
  quartiles of the NON-ZERO dose distribution computed on the TRAINING SPLIT
  ONLY. The legacy cut-points were derived on the broad 66k-stay cohort and
  are miscalibrated for a population that actually receives these drugs.
* SPLIT BY SUBJECT. One episode per patient already holds by construction, but
  the split is keyed on subject_id so the guarantee does not depend on it.
"""
import json
import os

import numpy as np
import pandas as pd

from config.clinical import TIME_BIN_HOURS
from config.paths import DATA_DIR
from config.rl import N_VASO_LEVELS, N_IV_LEVELS, REWARD_SURVIVAL, REWARD_DEATH
from src.final_features import FINAL_FEATURES_DIR
from src.final_features import (NE_RATE_CEILING, PLAUSIBILITY_PROVENANCE, VASO_AGG)
from src.utils import ensure_dir, set_seed

FINAL_PROCESSED_DIR = os.environ.get(
    "SEPSIS_FINAL_PROCESSED", os.path.join(DATA_DIR, "processed_90d_final"))

STATE_COLS = [
    # Vitals
    "heart_rate", "sbp", "dbp", "mbp", "resp_rate", "spo2",
    "temperature", "fio2", "gcs_total",
    # Labs
    "lactate", "wbc", "platelets", "creatinine", "bilirubin_total",
    "hemoglobin", "bun", "sodium", "potassium",
    "chloride", "bicarbonate", "glucose", "albumin", "calcium",
    "magnesium", "phosphate", "inr", "ptt",
    "ph_blood", "pao2", "paco2", "base_excess", "bands", "crp",
    # Derived.
    #
    # `pao2_fio2` is deliberately absent. It divided the last PaO2 available at
    # the boundary by the last FiO2 available, and those two are selected
    # independently: PaO2 is keyed on lab storetime, FiO2 on chart charttime.
    # The pair need not describe the same oxygen setting, which produced
    # physiologically incompatible combinations such as PaO2 597 on FiO2 21%,
    # a ratio of 1,990. A valid P/F needs the FiO2 in effect at the specimen
    # time, which is what the official bg concept computes. Capping the ratio
    # would only hide that its numerator and denominator refer to different
    # moments. PaO2 and FiO2 are both retained separately, and the respiratory
    # component of the official SOFA already carries oxygenation in a
    # validated form. Pairing it correctly is a later sensitivity feature.
    "sofa_total", "sofa_observed", "urine_output",
    # Treatment history — lets the state distinguish an untreated normal MAP
    # from one achieved by treating
    "prev_vaso", "prev_iv",
    # Demographics and time
    "anchor_age", "gender_M", "hours_since_admission", "hours_from_onset",
]

CLIP_SIGMA = 5.0


def compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    """P/F ratio and the binary sex indicator."""
    df = df.copy()
    if "pao2_fio2" not in df.columns:
        fio2 = df["fio2"].replace(0, np.nan)
        # MIMIC records FiO2 as a percentage; convert to fraction when it looks like one
        fio2 = np.where(fio2 > 1.0, fio2 / 100.0, fio2)
        df["pao2_fio2"] = df["pao2"] / fio2
    df["gender_M"] = (df["gender"] == "M").astype(float)
    return df


def split_by_subject(df: pd.DataFrame, seed: int = 42,
                     fracs=(0.70, 0.15, 0.15)) -> dict:
    """Partition patients (not stays) into train/val/test."""
    set_seed(seed)
    subjects = df["subject_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    n = len(subjects)
    n_tr, n_va = int(n * fracs[0]), int(n * fracs[1])
    groups = {
        "train": set(subjects[:n_tr]),
        "val": set(subjects[n_tr:n_tr + n_va]),
        "test": set(subjects[n_tr + n_va:]),
    }
    out = {k: df[df["subject_id"].isin(v)].copy() for k, v in groups.items()}
    for k, v in out.items():
        print(f"  {k}: {v['stay_id'].nunique():,} stays, {len(v):,} transitions")
    return out


def learn_action_bins(train: pd.DataFrame) -> dict:
    """
    Cut-points for a true 5x5 grid, from the training split only.

    Level 0 is exactly zero. Levels 1-4 are quartiles of the strictly positive
    doses, so each non-zero level carries roughly equal mass by construction.
    """
    bins = {}
    for col, name in (("vaso_total", "vaso"), ("iv_fluid_total", "iv")):
        nz = train.loc[train[col] > 0, col]
        q = np.quantile(nz, [0.25, 0.50, 0.75]) if len(nz) else np.array([0., 0., 0.])
        # digitize cut-points: >0 opens level 1, then the three quartiles
        bins[name] = [0.0, 1e-12, float(q[0]), float(q[1]), float(q[2])]
        print(f"  {name}: n_nonzero={len(nz):,} cutpoints={bins[name][1:]}")
    return bins


def discretize(df: pd.DataFrame, bins: dict) -> pd.DataFrame:
    df = df.copy()
    df["vaso_level"] = np.digitize(df["vaso_total"].fillna(0), bins["vaso"][1:])
    df["iv_level"] = np.digitize(df["iv_fluid_total"].fillna(0), bins["iv"][1:])
    df["vaso_level"] = df["vaso_level"].clip(0, N_VASO_LEVELS - 1)
    df["iv_level"] = df["iv_level"].clip(0, N_IV_LEVELS - 1)
    df["action"] = df["vaso_level"] * N_IV_LEVELS + df["iv_level"]
    return df


def add_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Next state, terminal flag, and the sparse reward.

    The terminal transition is the episode's last interval. The reward is the
    90-day outcome and nothing else; there is no intermediate term.
    """
    df = df.sort_values(["stay_id", "time_bin"]).copy()
    g = df.groupby("stay_id", sort=False)

    # A shift(-1) is only a valid transition if the next row is the next
    # interval. The boundary grid is contiguous by construction, but a dropped
    # row anywhere upstream would silently splice a state to one 8 or 12 hours
    # later, so check rather than assume.
    gap = g["time_bin"].shift(-1) - df["time_bin"]
    interior = g.cumcount(ascending=False) > 0
    bad = int((interior & (gap != 1)).sum())
    if bad:
        raise AssertionError(
            f"{bad} non-consecutive transitions: interval grid has holes")

    for c in STATE_COLS:
        df[f"next_{c}"] = g[c].shift(-1)
    df["done"] = (g.cumcount(ascending=False) == 0).astype(float)

    df["reward"] = 0.0
    survived = (df["done"] == 1) & (df["icu_mortality"] == 0)
    died = (df["done"] == 1) & (df["icu_mortality"] == 1)
    df.loc[survived, "reward"] = REWARD_SURVIVAL
    df.loc[died, "reward"] = REWARD_DEATH

    # Absorbing-state convention: terminal rows keep their own state as next
    for c in STATE_COLS:
        m = df[f"next_{c}"].isna() & (df["done"] == 1)
        df.loc[m, f"next_{c}"] = df.loc[m, c]
    return df


def build(save: bool = True, seed: int = 42) -> dict:
    feats = pd.read_parquet(os.path.join(FINAL_FEATURES_DIR, "features.parquet"))
    print(f"Features: {len(feats):,} rows, {feats['stay_id'].nunique():,} stays")

    feats = compute_derived(feats)
    feats = feats.sort_values(["stay_id", "time_bin"]).reset_index(drop=True)

    splits = split_by_subject(feats, seed=seed)

    print("Learning action bins on train split...")
    bins = learn_action_bins(splits["train"])

    # Normalisation and imputation statistics come from train only
    train = splits["train"]
    med = train[STATE_COLS].median()
    mu = train[STATE_COLS].fillna(med).mean()
    sd = train[STATE_COLS].fillna(med).std().replace(0, 1.0)

    out = {}
    for name, part in splits.items():
        d = discretize(part, bins)
        d[STATE_COLS] = d[STATE_COLS].fillna(med)
        d[STATE_COLS] = ((d[STATE_COLS] - mu) / sd).clip(-CLIP_SIGMA, CLIP_SIGMA)
        d = add_transitions(d)
        keep = (STATE_COLS + [f"next_{c}" for c in STATE_COLS]
                + ["stay_id", "subject_id", "time_bin", "action", "vaso_level",
                   "iv_level", "reward", "done", "icu_mortality",
                   "vaso_total", "iv_fluid_total"])
        out[name] = d[keep].reset_index(drop=True)

    # No patient may appear in more than one split
    subj = {k: set(v["subject_id"]) for k, v in out.items()}
    assert not (subj["train"] & subj["val"]), "subject leakage train/val"
    assert not (subj["train"] & subj["test"]), "subject leakage train/test"
    assert not (subj["val"] & subj["test"]), "subject leakage val/test"

    # Every marginal level must be reachable, or the grid is not really 5x5.
    # Joint cells may legitimately be empty where a dose combination is not
    # given in practice, so those are reported rather than asserted.
    tr = out["train"]
    for col, n_lv in (("vaso_level", N_VASO_LEVELS), ("iv_level", N_IV_LEVELS)):
        present = sorted(tr[col].unique())
        assert set(present) == set(range(n_lv)), (
            f"{col} levels {present} — expected 0..{n_lv - 1}")
        print(f"  {col} marginal: "
              + " ".join(f"{lv}={100 * (tr[col] == lv).mean():.1f}%"
                         for lv in range(n_lv)))
    occ = tr["action"].value_counts()
    assert tr["action"].between(0, 24).all(), "action ids outside 0..24"
    print(f"Joint cells occupied in train: {len(occ)}/25 "
          f"(zero-treatment rate {100 * (tr['action'] == 0).mean():.1f}%)")

    if save:
        # Refuse to overwrite an existing dataset. The variable that selects
        # this directory is SEPSIS_FINAL_PROCESSED, not SEPSIS_PROCESSED_DIR,
        # and setting the wrong one once overwrote a dataset a running job was
        # reading. Overwriting is a deliberate act now.
        if (os.path.exists(os.path.join(FINAL_PROCESSED_DIR, "train.parquet"))
                and os.environ.get("SEPSIS_ALLOW_OVERWRITE") != "1"):
            raise SystemExit(
                f"{FINAL_PROCESSED_DIR} already holds a dataset.\n"
                f"Set SEPSIS_FINAL_PROCESSED to a new directory, or "
                f"SEPSIS_ALLOW_OVERWRITE=1 to replace it.")
        ensure_dir(FINAL_PROCESSED_DIR)
        for k, v in out.items():
            v.to_parquet(os.path.join(FINAL_PROCESSED_DIR, f"{k}.parquet"),
                         index=False)
        meta = {
            "cohort": "official MIMIC-IV v3.1 Sepsis-3, earliest eligible episode per patient",
            "state_cols": STATE_COLS,
            "n_state_features": len(STATE_COLS),
            "action_bins": bins,
            # Self-describing manifest. config.rl's bin-consistency guard
            # compares against its own static bin sets, which the final run
            # deliberately does not use — these thresholds are learned from
            # this cohort's training split. `verify_action_bins` below checks
            # the parquet against this manifest instead.
            "vaso_bins": bins["vaso"],
            "iv_bins": bins["iv"],
            "n_actions": int(N_VASO_LEVELS * N_IV_LEVELS),
            "action_bin_source": "train-split non-zero quartiles, this cohort",
            "reward": "sparse terminal +15/-15 on 90-day mortality",
            "mortality_horizon_days": 90,
            "time_bin_hours": TIME_BIN_HOURS,
            "episode_hours_from_onset": [-24, 48],
            "split": "by subject_id 70/15/15",
            "split_seed": seed,
            "sofa_source": "physionet-data.mimiciv_3_1_derived.sofa (sofa_24hours)",
            "state_timing": "past-only; labs keyed on storetime, strictly before boundary",
            "actions_occupied_train": int(len(occ)),
            "n_transitions": {k: int(len(v)) for k, v in out.items()},
            "vaso_aggregation": VASO_AGG,
            "ne_rate_ceiling": NE_RATE_CEILING,
            "state_source_plausibility": PLAUSIBILITY_PROVENANCE,
            # The imputation medians and the standardisation mean and scale are
            # learned parameters of the input pipeline, fitted on the training
            # split. Without them a saved dataset cannot be mapped back to
            # clinical units, and a policy cannot be applied to a new patient.
            "preprocessing_stats": {
                "fitted_on": "train split only",
                "order": "impute with median, then (x - mean) / sd, then clip",
                "clip_sigma": CLIP_SIGMA,
                "median": {c: float(med[c]) for c in STATE_COLS},
                "mean": {c: float(mu[c]) for c in STATE_COLS},
                "sd": {c: float(sd[c]) for c in STATE_COLS},
            },
        }
        with open(os.path.join(FINAL_PROCESSED_DIR, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  saved -> {FINAL_PROCESSED_DIR}")
    return out


def verify_action_bins(processed_dir: str | None = None) -> None:
    """
    Check the stored action ids against the stored thresholds.

    `config.rl` carries an equivalent guard, but it compares against its own
    static bin sets and the final run learns its thresholds from this cohort,
    so that guard has to be bypassed here. This replaces it: re-derive the
    levels from the raw doses in each split and confirm they reproduce the
    `action` column exactly. It catches the case the original guard exists for
    — data and thresholds drifting apart — without requiring the thresholds to
    be known ahead of time.
    """
    pdir = processed_dir or FINAL_PROCESSED_DIR
    with open(os.path.join(pdir, "metadata.json")) as f:
        meta = json.load(f)
    bins = {"vaso": meta["vaso_bins"], "iv": meta["iv_bins"]}
    for split in ("train", "val", "test"):
        d = pd.read_parquet(os.path.join(pdir, f"{split}.parquet"))
        redone = discretize(d[["vaso_total", "iv_fluid_total"]].copy(), bins)
        if not (redone["action"].values == d["action"].values).all():
            raise RuntimeError(
                f"{split}: stored actions do not match stored thresholds")
    print(f"  action bins verified against data in {pdir}")


if __name__ == "__main__":
    build()
    verify_action_bins()
