"""
Fitted-Q evaluation with an explicit evaluation-policy function.

Supports greedy Q-network policies (D3QN/CQL), greedy actors (IQL), fitted
behaviour policies and constant-action controls. Each iteration computes
r + gamma * (1 - done) * V(s_next) under the fixed policy, then fits the
evaluation Q-network by supervised regression.

Use the same evaluator settings and seed for policy comparisons.
"""
import json
import os
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config.rl import GAMMA, N_ACTIONS
from src.models.d3qn import D3QN
from src.models.fqe import FQENet
from src.utils import set_seed

PolicyFn = Callable[[torch.Tensor], torch.Tensor]  # states (B,S) -> probs (B,A)


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION POLICIES
# ─────────────────────────────────────────────────────────────────────────────

def _one_hot(idx: torch.Tensor, n: int) -> torch.Tensor:
    return F.one_hot(idx, num_classes=n).float()


def greedy_q_policy(checkpoint: str, state_dim: int, hidden_dims: list,
                    device) -> PolicyFn:
    """Deterministic argmax over a D3QN/CQL Q-network."""
    net = D3QN(state_dim, N_ACTIONS, hidden_dims).to(device)
    net.load_state_dict(torch.load(checkpoint, map_location=device,
                                   weights_only=True))
    net.eval()

    def pi(states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return _one_hot(net(states).argmax(dim=1), N_ACTIONS)
    return pi


def actor_greedy_policy(run_dir: str, device) -> PolicyFn:
    """Deterministic argmax over IQL actor logits — the actual IQL policy."""
    from src.train_iql_actor import load_actor
    actor = load_actor(run_dir, device=device)

    def pi(states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return _one_hot(actor(states).argmax(dim=1), N_ACTIONS)
    return pi


def actor_greedy_policy_critic(run_dir: str, device, critic: str = "best") -> PolicyFn:
    """IQL actor policy, selecting which critic it was extracted against."""
    from src.train_iql_actor import load_actor
    actor = load_actor(run_dir, device=device, critic=critic)

    def pi(states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return _one_hot(actor(states).argmax(dim=1), N_ACTIONS)
    return pi


def marginal_policy(probs) -> PolicyFn:
    """
    A fixed action distribution, applied to every state.

    The always-withhold control asks whether a learned policy beats blanket
    restraint. This asks the sharper question: whether it beats *its own*
    prescribing habit applied blindly. Given the marginal distribution of a
    policy's recommendations, this reproduces how much treatment that policy
    gives while removing entirely which patient receives it. A learned policy
    that does not beat its own marginal has demonstrated a treatment rate, not
    personalisation.
    """
    import numpy as _np
    q = torch.tensor(_np.asarray(probs, dtype=_np.float32))
    q = q / q.sum()

    def pi(states: torch.Tensor) -> torch.Tensor:
        return q.to(states.device).expand(len(states), -1).clone()
    return pi


def constant_policy(action: int = 0) -> PolicyFn:
    """
    Always take one joint action, regardless of state.

    With action 0 this is the always-withhold control. Its purpose is to
    separate state-dependent selection from blanket restraint: a learned policy
    that cannot beat it has not demonstrated personalisation, only caution.
    """
    def pi(states: torch.Tensor) -> torch.Tensor:
        idx = torch.full((len(states),), action, dtype=torch.long,
                         device=states.device)
        return _one_hot(idx, N_ACTIONS)
    return pi


def mlp_classifier_policy(checkpoint: str, state_dim: int, device,
                          hidden: int = 128) -> PolicyFn:
    """
    Stochastic policy from an MLP action classifier.

    Serves both references: fitted on clinician actions it is the behaviour
    policy, which is what "the clinician" must mean when values are compared
    under FQE; the behaviour-cloning run uses the same architecture.
    """
    from src.models.behavior_policy import MLPBehaviorPolicyNet
    net = MLPBehaviorPolicyNet(state_dim, N_ACTIONS, hidden).to(device)
    net.load_state_dict(torch.load(checkpoint, map_location=device,
                                   weights_only=True))
    net.eval()

    def pi(states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.softmax(net(states), dim=1)
    return pi


def softmax_q_policy(checkpoint: str, state_dim: int, hidden_dims: list,
                     device, temperature: float = 0.5) -> PolicyFn:
    """Stochastic softmax over Q — kept for behaviour/BC references."""
    net = D3QN(state_dim, N_ACTIONS, hidden_dims).to(device)
    net.load_state_dict(torch.load(checkpoint, map_location=device,
                                   weights_only=True))
    net.eval()

    def pi(states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.softmax(net(states) / max(temperature, 1e-3), dim=1)
    return pi


# ─────────────────────────────────────────────────────────────────────────────
# FQE
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _precompute_v_next(fqe_net: FQENet, policy: PolicyFn, s_next: torch.Tensor,
                       device, chunk: int = 8192) -> torch.Tensor:
    fqe_net.eval()
    parts = []
    for i in range(0, len(s_next), chunk):
        sb = s_next[i:i + chunk].to(device)
        v = (policy(sb) * fqe_net(sb)).sum(dim=1)
        parts.append(v.cpu())
    return torch.cat(parts)


def _sha256(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_fqe(
    policy: PolicyFn,
    processed_dir: str,
    state_dim: int,
    hidden_dims: list,
    split: str = "test",
    gamma: float = GAMMA,
    n_iterations: int = 20,
    inner_epochs: int = 3,
    batch_size: int = 512,
    lr: float = 1e-3,
    output_dir: str | None = None,
    seed: int = 42,
    state_cols: list | None = None,
    device=None,
    label: str = "",
    reuse_from: str | None = None,
) -> dict:
    """
    Fit Q^pi on training transitions, evaluate V(s_0) on `split`.

    The fit uses training transitions only, so it does not depend on `split`;
    `split` selects the initial states the fitted network is then scored on.
    Fitting once per policy and checkpoint and scoring both splits from that
    one network is therefore exactly the same computation, and the val and test
    artefacts of an earlier run are byte-identical. `reuse_from` names a
    directory holding such a fit. The reused network's hash is checked against
    the source and recorded, so a reused result can be traced to the fit it
    came from.
    """
    set_seed(seed)
    device = device or torch.device("cpu")
    if state_cols is None:
        with open(os.path.join(processed_dir, "metadata.json")) as f:
            state_cols = json.load(f)["state_cols"]
    next_cols = [f"next_{c}" for c in state_cols]

    tr = pd.read_parquet(os.path.join(processed_dir, "train.parquet"))
    s_tr = torch.tensor(tr[state_cols].values, dtype=torch.float32)
    a_tr = torch.tensor(tr["action"].values, dtype=torch.long)
    r_tr = torch.tensor(tr["reward"].values, dtype=torch.float32)
    sn_tr = torch.tensor(tr[next_cols].values, dtype=torch.float32)
    dn_tr = torch.tensor(tr["done"].values, dtype=torch.float32)
    del tr

    fqe_net = FQENet(state_dim, N_ACTIONS, hidden_dims).to(device)
    reused_hash, reused_trace = None, []
    if reuse_from and os.path.exists(os.path.join(reuse_from, "fqe_net.pt")):
        src_net = os.path.join(reuse_from, "fqe_net.pt")
        fqe_net.load_state_dict(torch.load(src_net, map_location=device,
                                           weights_only=True))
        reused_hash = _sha256(src_net)
        src_hist = os.path.join(reuse_from, "fqe_history.csv")
        _h = pd.read_csv(src_hist) if os.path.exists(src_hist) else None
        history = _h["loss"].tolist() if _h is not None else []
        # The convergence trace belongs to the fit, so it comes along with it.
        reused_trace = (_h["v0_monitor_mean"].tolist()
                        if _h is not None and "v0_monitor_mean" in _h else [])
        n_iterations = 0
        print(f"  FQE[{label}] reusing the fit in {reuse_from}")

    opt = torch.optim.Adam(fqe_net.parameters(), lr=lr)
    if reused_hash is None:
        history = []

    # Convergence monitor. The stored training loss is measured against a
    # different Bellman target at every outer iteration, so a rise in it does
    # not by itself show that value iteration failed to converge. What does
    # speak to convergence is a quantity held fixed across iterations: the mean
    # initial-state value on a fixed monitor set, and how much it moves. Read
    # only, and never used to choose an iteration, since selecting the
    # estimator on the values it produces would couple it to what it estimates.
    monitor = pd.read_parquet(os.path.join(processed_dir, "val.parquet"))
    monitor = (monitor.sort_values(["stay_id", "time_bin"])
                      .groupby("stay_id", sort=False).first().reset_index())
    s_mon = torch.tensor(monitor[state_cols].values, dtype=torch.float32)
    del monitor
    v0_trace = list(reused_trace) if reused_hash else []

    def _v0_mean() -> float:
        fqe_net.eval()
        with torch.no_grad():
            tot, n = 0.0, 0
            for i in range(0, len(s_mon), 4096):
                sb = s_mon[i:i + 4096].to(device)
                tot += float((policy(sb) * fqe_net(sb)).sum(dim=1).sum())
                n += len(sb)
        return tot / max(n, 1)

    for it in range(n_iterations):
        v_next = _precompute_v_next(fqe_net, policy, sn_tr, device)
        y = r_tr + gamma * (1.0 - dn_tr) * v_next
        loader = DataLoader(TensorDataset(s_tr, a_tr, y),
                            batch_size=batch_size, shuffle=True)
        fqe_net.train()
        tot, nb = 0.0, 0
        for _ in range(inner_epochs):
            for sb, ab, yb in loader:
                sb, ab, yb = sb.to(device), ab.to(device), yb.to(device)
                q = fqe_net(sb)[torch.arange(len(ab), device=device), ab]
                loss = F.smooth_l1_loss(q, yb)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(fqe_net.parameters(), 10.0)
                opt.step()
                tot += loss.item(); nb += 1
        history.append(tot / max(nb, 1))
        v0_trace.append(_v0_mean())

    fqe_net.eval()
    ev = pd.read_parquet(os.path.join(processed_dir, f"{split}.parquet"))
    ev = ev.sort_values(["stay_id", "time_bin"])
    init = ev.groupby("stay_id", sort=False).first().reset_index()
    s0 = torch.tensor(init[state_cols].values, dtype=torch.float32)

    with torch.no_grad():
        parts = []
        for i in range(0, len(s0), 4096):
            sb = s0[i:i + 4096].to(device)
            parts.append(((policy(sb) * fqe_net(sb)).sum(dim=1)).cpu().numpy())
    v0 = np.concatenate(parts)

    # Empirical discounted return of the logged clinician policy, for reference.
    # Under a sparse reward this is essentially the discounted terminal outcome.
    returns = [float(np.dot(gamma ** np.arange(len(g)), g["reward"].values))
               for _, g in ev.groupby("stay_id", sort=False)]

    result = {
        "label": label,
        "split": split,
        "fqe_value": float(v0.mean()),
        "fqe_value_std": float(v0.std()),
        "empirical_clinician_return": float(np.mean(returns)),
        "n_initial_states": int(len(v0)),
        "n_iterations": len(history),
        "final_train_loss": history[-1] if history else None,
        "min_train_loss": min(history) if history else None,
        "argmin_train_loss": (int(np.argmin(history)) if history else None),
        # How much the estimate still moves between the last two iterations,
        # and the largest move over the final quarter. Small values are
        # consistent with convergence; large ones are not.
        "v0_last_delta": (abs(v0_trace[-1] - v0_trace[-2])
                          if len(v0_trace) > 1 else None),
        "v0_max_delta_last_quarter": (
            float(np.max(np.abs(np.diff(v0_trace[-max(2, len(v0_trace) // 4):]))))
            if len(v0_trace) > 2 else None),
        "gamma": gamma,
        "seed": seed,
        "reused_fit_from": reuse_from if reused_hash else None,
        "fit_sha256": reused_hash,
    }
    print(f"  FQE[{label}] {split}: {result['fqe_value']:+.4f}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        net_path = os.path.join(output_dir, "fqe_net.pt")
        torch.save(fqe_net.state_dict(), net_path)
        if reused_hash is not None:
            # A reused fit must be the fit it claims to be. The value here
            # differs from the source only in which patients it was scored on.
            assert _sha256(net_path) == reused_hash, (
                f"{net_path} does not match the fit it was reused from")
        else:
            result["fit_sha256"] = _sha256(net_path)
        hist = {"iteration": range(len(history)), "loss": history}
        if len(v0_trace) == len(history):
            hist["v0_monitor_mean"] = v0_trace
        pd.DataFrame(hist).to_csv(
            os.path.join(output_dir, "fqe_history.csv"), index=False)
        np.savez_compressed(os.path.join(output_dir, "v0.npz"),
                            stay_id=init["stay_id"].values, v0=v0)
        # `fqe_result.json` is written last, so its presence means every other
        # artefact is already there. Every caller treats that file as the
        # completion marker and skips a directory that has it, so a crash
        # between the per-stay values and the result would otherwise leave a
        # fit that reads as done but whose v0 a paired test cannot find.
        with open(os.path.join(output_dir, "fqe_result.json"), "w") as f:
            json.dump(result, f, indent=2)
    return result
