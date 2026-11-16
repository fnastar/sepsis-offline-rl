"""
Load and align per-stay FQE estimates for policy comparisons.

The final campaign saves estimates and stay identifiers in v0.npz. A fallback
reconstructs values for older softmax-policy runs from saved networks;
verify_against_stored() checks those values against the recorded mean.
The final greedy-policy workflow uses its saved arrays.

The stored fqe_value_std is dispersion across stays, not a standard error.
"""
import json
import os

import numpy as np
import pandas as pd
import torch

from config.paths import PROCESSED_DIR, OUTPUT_BASE
from config.rl import N_ACTIONS
from src.models.fqe import FQENet
from src.models.d3qn import D3QN
from src.models.qr_d3qn import QRD3QN

HIDDEN_DIMS = [256, 256]
DEFAULT_TEMP = 0.5
CLINICIAN_DIR = os.path.join(OUTPUT_BASE, "fqe_clinician_validation")

# The final campaign fits the clinician reference through the same run_fqe path
# as every other policy, so it lands in fqe_clinician/fqe_{split}/v0.npz rather
# than the validation layout this module was written against. Both are
# accepted; the campaign layout wins when present.
CLINICIAN_DIR_CAMPAIGN = os.path.join(OUTPUT_BASE, "fqe_clinician")

# Which checkpoint's FQE artefacts to read. The campaign writes the
# lowest-validation-loss fit to fqe_{split} and the final epoch to
# fqe_{split}_finalckpt, and on this data the first of those is epoch 1, so the
# significance test has to be runnable on either.
FQE_SUFFIX = os.environ.get("SEPSIS_FQE_SUFFIX", "")


def _state_cols() -> list:
    with open(os.path.join(PROCESSED_DIR, "metadata.json")) as f:
        return json.load(f)["state_cols"]


def initial_states(split: str, state_cols: list) -> tuple:
    """(stay_ids, s0) for a split — the same derivation `train_fqe` uses.

    Sorting then grouping with sort=False means stay order follows sorted
    stay_id in both this module and fqe.py/validate_fqe.py. Pairing is still
    done by stay_id rather than by position, so a future change to either
    ordering cannot silently misalign two runs.
    """
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, f"{split}.parquet"))
    df = df.sort_values(["stay_id", "time_bin"])
    init = df.groupby("stay_id", sort=False).first().reset_index()
    return init["stay_id"].values, init[state_cols].values.astype(np.float32)


def _policy_net(run_dir: str, kind: str, state_dim: int):
    if kind == "iql":
        # Reconstructing IQL from its critic would evaluate softmax(Q), which
        # is a different policy from the extracted actor the run was scored
        # with. Better to fail than to return a plausible wrong number.
        raise ValueError(
            f"{run_dir}: IQL v0 cannot be reconstructed from the critic. "
            f"Its policy is the extracted actor; refit FQE if v0.npz is gone.")
    ckpt = os.path.join(run_dir, "d3qn_best.pt")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(run_dir, "d3qn_final.pt")
    net = (QRD3QN if kind == "qr" else D3QN)(state_dim, N_ACTIONS, HIDDEN_DIMS)
    net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    net.eval()
    return net


def fqe_subdir(split: str, kind: str = "d3qn") -> str:
    """FQE artefact directory for a run/split, per family.

    `src/fqe_qr.py` writes `fqe_mean_{split}` and `fqe_q25_{split}` (one per
    extracted policy), while `src/models/fqe.py` writes plain `fqe_{split}`.
    Assuming the latter for every family is what silently excluded QR from the
    significance test: `load_v0` raised FileNotFoundError and
    `fqe_significance._stack_differences` swallowed it as "no recoverable v0".

    `fqe_mean_*` is the right QR target: `load_v0` reconstructs with
    `policy.q_values()`, and `QRD3QN.q_values` is `forward().mean(dim=2)` —
    exactly the mean policy `fqe_mean_*` was fitted against, at the same
    temperature.
    """
    if kind == "qr":
        return f"fqe_mean_{split}{FQE_SUFFIX}"
    if kind == "iql":
        # IQL's policy is its actor, and the final-checkpoint variant is the
        # actor re-extracted against the epoch-40 critic, so its directory is
        # named for the critic rather than for the Q-checkpoint.
        return f"fqe_{split}_criticfinal" if FQE_SUFFIX else f"fqe_{split}"
    return f"fqe_{split}{FQE_SUFFIX}"


def load_v0(run: str, split: str, kind: str = "d3qn",
            temperature: float = DEFAULT_TEMP) -> tuple:
    """Per-stay V̂(s_0) for a policy run. Returns (stay_ids, v0).

    Prefers the persisted `v0.npz`; falls back to exact reconstruction from
    `fqe_net.pt` for runs that predate v0 persistence.
    """
    fqe_dir = os.path.join(OUTPUT_BASE, run, fqe_subdir(split, kind))
    cached = os.path.join(fqe_dir, "v0.npz")
    if os.path.exists(cached):
        d = np.load(cached)
        return d["stay_id"], d["v0"]

    state_cols = _state_cols()
    policy = _policy_net(os.path.join(OUTPUT_BASE, run), kind, len(state_cols))
    fqe_net = FQENet(len(state_cols), N_ACTIONS, HIDDEN_DIMS)
    fqe_net.load_state_dict(torch.load(os.path.join(fqe_dir, "fqe_net.pt"),
                                       map_location="cpu", weights_only=True))
    fqe_net.eval()

    stay_ids, s0 = initial_states(split, state_cols)
    parts = []
    with torch.no_grad():
        for i in range(0, len(s0), 4096):
            sb = torch.tensor(s0[i:i + 4096])
            q_pol = policy.q_values(sb) if kind == "qr" else policy(sb)
            pi_e = torch.softmax(q_pol / max(temperature, 1e-3), dim=1)
            parts.append((pi_e * fqe_net(sb)).sum(dim=1).numpy())
    return stay_ids, np.concatenate(parts)


def load_clinician_v0(split: str) -> tuple:
    """Per-stay V̂(s_0) under the behaviour policy π_β — the FQE reference.

    The final campaign persists this array while fitting its references.
    A legacy cached layout is also accepted when available.
    """
    campaign = os.path.join(CLINICIAN_DIR_CAMPAIGN, f"fqe_{split}", "v0.npz")
    if os.path.exists(campaign):
        d = np.load(campaign)
        return d["stay_id"], d["v0"]
    path = os.path.join(CLINICIAN_DIR, f"v0_{split}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Neither {campaign} nor {path} exists. The final campaign's "
            f"reference FQE must complete and persist v0.npz before the "
            f"significance analysis. See docs/reproduction.md; do not "
            f"substitute an evaluation from a different dataset or policy.")
    d = np.load(path)
    return d["stay_id"], d["v0"]


def pair(a_ids: np.ndarray, a_v0: np.ndarray,
         b_ids: np.ndarray, b_v0: np.ndarray) -> tuple:
    """Align two runs' per-stay values on stay_id. Returns (ids, a, b).

    Aligning by key rather than position: the two arms are produced by
    different scripts, and a positional assumption would fail silently.
    """
    order_a = np.argsort(a_ids)
    order_b = np.argsort(b_ids)
    ids_a, ids_b = a_ids[order_a], b_ids[order_b]
    if not np.array_equal(ids_a, ids_b):
        common = np.intersect1d(ids_a, ids_b)
        raise ValueError(
            f"stay_id sets differ: {len(ids_a)} vs {len(ids_b)}, "
            f"{len(common)} in common. Refusing to pair on a partial overlap.")
    return ids_a, a_v0[order_a], b_v0[order_b]


def verify_against_stored(run: str, split: str, kind: str = "d3qn") -> float:
    """Assert the loaded v0 reproduces the run's stored fqe_value. Returns |diff|."""
    _, v0 = load_v0(run, split, kind)
    stored = json.load(open(os.path.join(
        OUTPUT_BASE, run, fqe_subdir(split, kind), "fqe_result.json")))["fqe_value"]
    diff = abs(float(v0.mean()) - stored)
    if diff > 1e-5:
        raise AssertionError(
            f"{run}/{split}: recovered mean {v0.mean():.6f} != stored "
            f"{stored:.6f} (diff {diff:.2e}). v0 recovery is wrong.")
    return diff
