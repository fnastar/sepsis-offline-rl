"""
fqe.py: Fitted Q Evaluation (FQE) for offline RL.

Evaluates a fixed policy π_e without importance sampling by iterating the
Bellman evaluation operator:

    Q^{k+1}(s, a) ← r + γ (1-done) Σ_{a'} π_e(a'|s') Q^k(s', a')

At each outer iteration the current Q^k is frozen, next-state values are
precomputed for the entire training set, then Q^{k+1} is fitted as a
supervised regression problem.

This avoids the moving-target instability of doing both a forward pass and
a gradient update within the same loop, and is equivalent to the offline
FQE algorithm of Le Paine et al. 2020 / Voloshin et al. 2021.

References:
  Le Paine et al. 2020 "Hyperparameter Selection for Offline RL"
  Voloshin et al. 2021 "Empirical Study of Off-Policy Evaluation"
"""

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config.paths import PROCESSED_DIR
from config.rl import GAMMA, N_ACTIONS
from src.models.d3qn import D3QN


# ─────────────────────────────────────────────────────────────────────────────
# FQE NETWORK
# ─────────────────────────────────────────────────────────────────────────────

class FQENet(nn.Module):
    """Q^π_e network — same shared encoder as D3QN, non-dueling output."""
    def __init__(self, state_dim: int, n_actions: int,
                 hidden_dims: list[int] = [256, 256]):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU()]
            in_dim = h
        self.encoder = nn.Sequential(*layers)
        self.q_head = nn.Linear(in_dim, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q_head(self.encoder(x))


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _softmax_pi(q: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.softmax(q / max(temperature, 1e-3), dim=1)


@torch.no_grad()
def _precompute_v_next(
    fqe_net: FQENet,
    eval_net: D3QN,
    s_next: torch.Tensor,
    temperature: float,
    device: torch.device,
    chunk: int = 8192,
) -> torch.Tensor:
    """
    Precompute V^π_e(s') = Σ_a π_e(a|s') Q^k(s', a) for all training steps.

    π_e is derived from the D3QN Q-network (fixed).
    Q^k is the current FQE estimate (frozen outside this function by the caller).
    """
    fqe_net.eval()
    eval_net.eval()
    parts = []
    for i in range(0, len(s_next), chunk):
        sb = s_next[i:i + chunk].to(device)
        pi_e = _softmax_pi(eval_net(sb), temperature)   # (B, A) — fixed policy
        q_k  = fqe_net(sb)                              # (B, A) — current FQE
        v    = (pi_e * q_k).sum(dim=1)                  # (B,)
        parts.append(v.cpu())
    return torch.cat(parts)   # (N_train,)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN: TRAIN FQE AND EVALUATE
# ─────────────────────────────────────────────────────────────────────────────

def train_fqe(
    q_state_dict_path: str,
    state_dim: int,
    hidden_dims: list[int],
    split: str = "val",
    temperature: float = 0.5,
    gamma: float = GAMMA,
    n_iterations: int = 20,
    inner_epochs: int = 3,
    batch_size: int = 512,
    lr: float = 1e-3,
    output_dir: str | None = None,
    seed: int = 42,
    state_cols: list[str] | None = None,
    device: torch.device | None = None,
) -> dict:
    """
    Fit Q^π_e via FQE using training transitions; estimate policy value on
    `split` (val or test) initial states.

    Parameters
    ----------
    q_state_dict_path : path to D3QN d3qn_best.pt checkpoint (defines π_e)
    state_dim         : number of state features (39)
    hidden_dims       : hidden layer widths, e.g. [256, 256]
    split             : which split's initial states to evaluate on
    temperature       : softmax temperature for the evaluation policy
    gamma             : discount factor
    n_iterations      : number of FQE outer iterations
    inner_epochs      : SGD epochs per outer iteration
    output_dir        : if set, save fqe_net.pt + fqe_history.csv + fqe_result.json

    Returns
    -------
    dict with fqe_value (mean V̂(s_0) on split) and diagnostics
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or torch.device("cpu")

    if state_cols is None:
        with open(os.path.join(PROCESSED_DIR, "metadata.json")) as f:
            state_cols = json.load(f)["state_cols"]
    next_cols = [f"next_{c}" for c in state_cols]

    # ── Load training transitions ──────────────────────────────────────────────
    print(f"  Loading training transitions…")
    train_df = pd.read_parquet(os.path.join(PROCESSED_DIR, "train.parquet"))
    s_tr  = torch.tensor(train_df[state_cols].values, dtype=torch.float32)
    a_tr  = torch.tensor(train_df["action"].values,   dtype=torch.long)
    r_tr  = torch.tensor(train_df["reward"].values,   dtype=torch.float32)
    sn_tr = torch.tensor(train_df[next_cols].values,  dtype=torch.float32)
    dn_tr = torch.tensor(train_df["done"].values,     dtype=torch.float32)
    del train_df
    N = len(s_tr)
    print(f"  Training transitions: {N:,}")

    # ── Fixed evaluation policy: D3QN ─────────────────────────────────────────
    eval_net = D3QN(state_dim, N_ACTIONS, hidden_dims).to(device)
    eval_net.load_state_dict(torch.load(q_state_dict_path, map_location=device,
                                        weights_only=True))
    eval_net.eval()

    # ── FQE network and optimizer ──────────────────────────────────────────────
    fqe_net = FQENet(state_dim, N_ACTIONS, hidden_dims).to(device)
    opt = torch.optim.Adam(fqe_net.parameters(), lr=lr)

    history = []

    for iteration in range(n_iterations):
        # Step 1: precompute V^π_e(s') under frozen Q^k
        v_next = _precompute_v_next(fqe_net, eval_net, sn_tr, temperature,
                                    device)   # (N,) on CPU

        # Step 2: compute regression targets y = r + γ(1−done) V(s')
        y_tr = r_tr + gamma * (1.0 - dn_tr) * v_next   # (N,) CPU tensors

        # Step 3: supervised regression — fit Q(s, a) → y
        dataset = TensorDataset(s_tr, a_tr, y_tr)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                             num_workers=0)
        fqe_net.train()
        total_loss, n_batches = 0.0, 0
        for _epoch in range(inner_epochs):
            for sb, ab, yb in loader:
                sb = sb.to(device)
                ab = ab.to(device)
                yb = yb.to(device)
                # Q prediction for taken action
                q_pred = fqe_net(sb)[torch.arange(len(ab), device=device), ab]
                loss = F.smooth_l1_loss(q_pred, yb)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(fqe_net.parameters(), 10.0)
                opt.step()
                total_loss += loss.item()
                n_batches  += 1

        avg_loss = total_loss / max(n_batches, 1)
        history.append(avg_loss)
        if (iteration + 1) % 5 == 0 or iteration == 0:
            print(f"  FQE iter {iteration+1:2d}/{n_iterations}  "
                  f"loss={avg_loss:.4f}  "
                  f"v_next_mean={float(v_next.mean()):.3f}")

    # ── Evaluate: V̂(s_0) on initial states of `split` ─────────────────────────
    fqe_net.eval()
    eval_df = pd.read_parquet(os.path.join(PROCESSED_DIR, f"{split}.parquet"))
    eval_df = eval_df.sort_values(["stay_id", "time_bin"])
    init_df = eval_df.groupby("stay_id", sort=False).first().reset_index()
    s0 = torch.tensor(init_df[state_cols].values, dtype=torch.float32)

    with torch.no_grad():
        v0_parts = []
        for i in range(0, len(s0), 4096):
            sb = s0[i:i + 4096].to(device)
            pi_e = _softmax_pi(eval_net(sb), temperature)
            q_f  = fqe_net(sb)
            v    = (pi_e * q_f).sum(dim=1)
            v0_parts.append(v.cpu().numpy())
    v0 = np.concatenate(v0_parts)
    fqe_value = float(v0.mean())
    # NOTE: dispersion of V̂ across stays, NOT a standard error of the mean.
    # It is ~sqrt(n) times too large to use as an error bar. For inference use
    # the per-stay array persisted below (v0.npz), not this number.
    fqe_std   = float(v0.std())
    stay_ids  = init_df["stay_id"].values

    # Clinician baseline: mean discounted return on split
    traj_returns = []
    for _, g in eval_df.groupby("stay_id", sort=False):
        r_arr = g["reward"].values.astype(np.float64)
        traj_returns.append(float(np.dot(gamma ** np.arange(len(r_arr)), r_arr)))
    clinician_value = float(np.mean(traj_returns))

    result = {
        "split":           split,
        "fqe_value":       fqe_value,
        "fqe_value_std":   fqe_std,
        "clinician_value": clinician_value,
        "n_initial_states": int(len(v0)),
        "n_iterations":    n_iterations,
        "final_train_loss": history[-1] if history else None,
        "temperature":     temperature,
        "gamma":           gamma,
    }
    print(f"  FQE {split}: V̂(s_0)={fqe_value:.3f}±{fqe_std:.3f}  "
          f"clinician={clinician_value:.3f}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        torch.save(fqe_net.state_dict(),
                   os.path.join(output_dir, "fqe_net.pt"))
        pd.DataFrame({"iteration": range(len(history)), "loss": history}).to_csv(
            os.path.join(output_dir, "fqe_history.csv"), index=False)
        with open(os.path.join(output_dir, "fqe_result.json"), "w") as f:
            json.dump(result, f, indent=2)
        # Per-stay V̂(s_0), keyed by stay_id. Collapsing to the mean above
        # discards everything needed for inference; significance testing
        # resamples these. Keyed rather than positional so two runs can be
        # paired safely without assuming identical row order.
        np.savez_compressed(os.path.join(output_dir, "v0.npz"),
                            stay_id=stay_ids, v0=v0)
        print(f"  FQE artefacts saved → {output_dir}")

    return result
