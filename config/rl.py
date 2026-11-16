"""
Shared action-space, reward and training defaults.

The final pipeline learns action bins from training data and sets its
training schedule in src.final_campaign. Static bin sets below support
earlier experiments.
"""
import os

import numpy as np

# Static action bins for earlier experiments. Each action combines one of
# five fluid levels with one of five vasopressor levels.
#
# legacy: historical thresholds with an unreachable top bin.
# minimal: the same thresholds with an explicit zero-dose level.
# quantile: thresholds estimated in an earlier training split.
#
# The final pipeline learns its own nonzero-dose quartiles from the training
# split. See final_preprocessing.py and docs/reproduction.md. Checkpoints
# from different bin definitions have matching shapes but different action
# meanings, so their data and behaviour models must not be mixed.

_LEGACY_VASO_BINS = [0, 0.08, 0.22, 0.45, np.inf]
_LEGACY_IV_BINS = [0, 50, 180, 530, np.inf]

# Same thresholds as legacy, with the dead `inf` edge moved to a zero test.
_MINIMAL_VASO_BINS = [0, 1e-9, 0.08, 0.22, 0.45]
_MINIMAL_IV_BINS = [0, 1e-9, 50, 180, 530]

# Earlier train-split non-zero quartiles, retained for shared-module compatibility.
_QUANTILE_VASO_BINS = [0, 1e-9, 0.046700, 0.088200, 0.190000]
_QUANTILE_IV_BINS = [0, 1e-9, 93.954500, 229.500000, 606.362100]

_BIN_SETS = {
    "legacy": (_LEGACY_VASO_BINS, _LEGACY_IV_BINS),
    "minimal": (_MINIMAL_VASO_BINS, _MINIMAL_IV_BINS),
    "quantile": (_QUANTILE_VASO_BINS, _QUANTILE_IV_BINS),
}

# SEPSIS_ACTION_FIX=1 is the older boolean shorthand; it now means "minimal".
_shorthand = os.environ.get("SEPSIS_ACTION_FIX", "0").lower() in ("1", "true", "yes")
ACTION_SPACE = os.environ.get(
    "SEPSIS_ACTION_SPACE", "minimal" if _shorthand else "legacy").lower()
if ACTION_SPACE not in _BIN_SETS:
    raise ValueError(
        f"SEPSIS_ACTION_SPACE={ACTION_SPACE!r}; expected one of {sorted(_BIN_SETS)}")

VASO_BINS, IV_BINS = _BIN_SETS[ACTION_SPACE]   # mcg/kg/min NE-eq. ; mL per 4h

# Compatibility flag used by earlier preprocessing code.
ACTION_FIX = ACTION_SPACE != "legacy"

N_VASO_LEVELS = len(VASO_BINS)  # 5 levels (including 0)
N_IV_LEVELS = len(IV_BINS)      # 5 levels (including 0)

# Total actions = N_VASO_LEVELS × N_IV_LEVELS = 25
N_ACTIONS = N_VASO_LEVELS * N_IV_LEVELS

# ── Reward parameters ─────────────────────────────────────────────────────
GAMMA = 0.99
REWARD_SURVIVAL = +15
REWARD_DEATH = -15
SOFA_REWARD_COEF = 0.1  # intermediate reward: r_t = -coef * ΔSOFA (non-terminal)

# ── Shared training defaults ─────────────────────────────────────
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
N_EPOCHS = 100
HIDDEN_DIMS = [256, 256]


# ── Action-space consistency check ────────────────────────────────────────
def _assert_bins_match_data():
    """Fail loudly if PROCESSED_DIR was built under a different action space.

    `preprocessing.py` records `vaso_bins`/`iv_bins`/`n_actions` in
    metadata.json, but nothing ever read them back — so a namespace built under
    one bin set could be trained/evaluated under another with no error, because
    N_ACTIONS is 25 in every case and every `load_state_dict` succeeds. This is
    the check that closes that gap.

    Lives here because 17 modules import config.rl directly and the runner
    scripts reach it transitively, so one insertion point covers every call site
    that touches actions. It cannot catch a stale *checkpoint* (shapes match
    regardless) — that needs namespace discipline, not an assert.

    Skipped when metadata.json is absent (fresh namespace, pre-preprocessing) or
    when SEPSIS_SKIP_BIN_CHECK is set (needed to re-preprocess over an existing
    namespace, where the old metadata still describes the old bins).
    """
    if os.environ.get("SEPSIS_SKIP_BIN_CHECK"):
        return
    from config.paths import PROCESSED_DIR   # paths never imports rl: no cycle

    meta_path = os.path.join(PROCESSED_DIR, "metadata.json")
    if not os.path.exists(meta_path):
        return
    import json

    with open(meta_path) as f:
        meta = json.load(f)
    # `inf` round-trips through json as float('inf'); 1e-9 round-trips exactly.
    want = ([float(b) for b in VASO_BINS], [float(b) for b in IV_BINS], int(N_ACTIONS))
    got = (meta.get("vaso_bins"), meta.get("iv_bins"), meta.get("n_actions"))
    if got != want:
        raise RuntimeError(
            f"Action-space mismatch.\n"
            f"  {meta_path} was built with:\n"
            f"    vaso={got[0]}\n    iv  ={got[1]}\n    n_actions={got[2]}\n"
            f"  but config.rl (SEPSIS_ACTION_SPACE={ACTION_SPACE!r}) has:\n"
            f"    vaso={want[0]}\n    iv  ={want[1]}\n    n_actions={want[2]}\n"
            f"  Fix SEPSIS_ACTION_SPACE or SEPSIS_PROCESSED_DIR, or set "
            f"SEPSIS_SKIP_BIN_CHECK=1 if you are re-preprocessing this dir.")


_assert_bins_match_data()
