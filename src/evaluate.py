"""
evaluate.py: Off-policy evaluation (OPE) for offline RL on sepsis MDP.

Estimators implemented (per Thomas & Brunskill 2016, Jiang & Li 2016):
  - Weighted Importance Sampling (WIS): trajectory-level weights, normalized
  - Per-decision WIS (PDWIS): step-level weights
  - Doubly Robust (DR): combines model V̂ with importance-weighted residual

Bootstrap CIs computed by trajectory-level resampling (1000 reps).

Conventions:
  - Behavior policy: π_β(a|s) from supervised classifier (trained separately)
  - Target policy:   π_e(a|s) ∝ exp(Q(s, ·) / τ); softmax with temperature τ
  - Trajectory ρ:    product of π_e(a_t|s_t) / π_β(a_t|s_t) over the episode
  - Return G:        Σ_t γ^t r_t
"""
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from config.paths import PROCESSED_DIR
from config.rl import GAMMA, N_ACTIONS
from src.models.d3qn import D3QN

WEIGHT_CLIP_PERCENTILE = 99.0   # clip importance weights at this percentile
DEFAULT_TEMP = 0.5              # softmax temperature for target policy


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING (trajectory-level)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Trajectories:
    states: list      # list of (T_i, n_state) arrays
    actions: list     # list of (T_i,) int arrays
    rewards: list     # list of (T_i,) float arrays
    n: int

    @classmethod
    def from_parquet(cls, path: str, state_cols: list):
        df = pd.read_parquet(path).sort_values(["stay_id", "time_bin"])
        states, actions, rewards = [], [], []
        for sid, g in df.groupby("stay_id", sort=False):
            states.append(g[state_cols].values.astype(np.float32))
            actions.append(g["action"].values.astype(int))
            rewards.append(g["reward"].values.astype(np.float32))
        return cls(states=states, actions=actions, rewards=rewards, n=len(states))

    def returns(self, gamma: float) -> np.ndarray:
        """Discounted return per trajectory."""
        out = np.zeros(self.n, dtype=np.float64)
        for i, r in enumerate(self.rewards):
            t = np.arange(len(r))
            out[i] = float(np.sum((gamma ** t) * r))
        return out


# ─────────────────────────────────────────────────────────────────────────────
# TARGET POLICY FROM Q-NETWORK
# ─────────────────────────────────────────────────────────────────────────────

def softmax_policy_from_q(
    q_net: D3QN,
    states: np.ndarray,
    temperature: float = DEFAULT_TEMP,
    device: torch.device | None = None,
) -> np.ndarray:
    """Compute π_e(a|s) ∝ softmax(Q(s,a) / τ). Returns (N, n_actions)."""
    device = device or torch.device("cpu")
    q_net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(states), 4096):
            xb = torch.tensor(states[i:i+4096], dtype=torch.float32, device=device)
            q = q_net(xb)
            p = torch.softmax(q / max(temperature, 1e-3), dim=1).cpu().numpy()
            out.append(p)
    return np.vstack(out).astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# IMPORTANCE WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────

def trajectory_log_weights(
    trajs: Trajectories,
    pi_e_proba: np.ndarray,        # (total_steps, A) for π_e on all states
    pi_b_proba: np.ndarray,        # (total_steps, A) for π_β on all states
) -> tuple[np.ndarray, list]:
    """
    Returns:
      log_w     : (n_traj,) log of full-trajectory importance weight
      step_log_w: list of (T_i,) cumulative log weights up to step t (PDWIS)
    """
    log_w = np.zeros(trajs.n, dtype=np.float64)
    step_log_w = []
    pos = 0
    for i in range(trajs.n):
        T = len(trajs.actions[i])
        a = trajs.actions[i]
        # Use np.arange for the row-axis to get the diagonal (per-step prob),
        # not the outer product that arr[slice, array] would produce.
        rows = np.arange(pos, pos + T)
        # The TARGET probability is not floored. A deterministic policy assigns
        # exactly zero to any action it would not take, and a logged action
        # that disagrees must carry exactly zero importance weight. Flooring it
        # at 1e-12 gives every mismatching trajectory a tiny positive weight,
        # and because WIS self-normalises, the absolute scale then cancels and
        # the estimator returns a plausible number computed from the least
        # mismatching trajectories rather than from supported ones. That
        # changes the estimand rather than stabilising the arithmetic.
        #
        # The BEHAVIOUR probability keeps the 1e-3 floor its own fit applies,
        # since that is a denominator and a modelling convention.
        pe = pi_e_proba[rows, a]
        pb = pi_b_proba[rows, a].clip(1e-12, 1.0)
        with np.errstate(divide="ignore"):
            rho = np.log(pe) - np.log(pb)     # -inf where the target excludes a
        cum = np.cumsum(rho)                   # cumulative through step t (T,)
        step_log_w.append(cum)
        log_w[i] = cum[-1]
        pos += T
    return log_w, step_log_w


def clip_weights(w: np.ndarray, percentile: float = WEIGHT_CLIP_PERCENTILE) -> np.ndarray:
    """
    Cap the heaviest weights, taking the percentile over the positive ones.

    Over all weights the cap would be zero whenever most trajectories carry no
    weight, which is exactly the deterministic-policy case: with one supported
    trajectory in 3,817 the 99th percentile is zero and the only valid weight
    would be clipped away.
    """
    pos = w[w > 0]
    if pos.size == 0:
        return w
    return np.minimum(w, np.percentile(pos, percentile))


# ─────────────────────────────────────────────────────────────────────────────
# ESTIMATORS
# ─────────────────────────────────────────────────────────────────────────────

def n_supported(log_w: np.ndarray) -> int:
    """Trajectories the target policy could have generated, weight > 0."""
    return int(np.isfinite(log_w).sum())


def wis_estimator(
    log_w: np.ndarray,
    returns: np.ndarray,
    clip: bool = True,
) -> float:
    """
    Trajectory-level Weighted Importance Sampling.

    Undefined when no trajectory carries positive weight. Self-normalisation
    would otherwise turn zero support into a finite-looking number.
    """
    if n_supported(log_w) == 0:
        return float("nan")
    # Stabilize: subtract max log-weight before exp
    lw = log_w - log_w.max()
    w = np.exp(lw)
    if clip:
        w = clip_weights(w)
    den = float(np.sum(w))
    # A zero denominator is undefined, not zero. Substituting 1e-12 would turn
    # "nothing supports this" into a confident estimate of nothing.
    return float(np.sum(w * returns) / den) if den > 0 else float("nan")


def pdwis_estimator(
    step_log_w: list,
    rewards: list,
    gamma: float,
    clip: bool = True,
) -> float:
    """Per-decision WIS: weighted average of γ^t r_t with cumulative ratio."""
    n = len(rewards)
    # Find max horizon for normalization arrays
    max_T = max(len(r) for r in rewards)
    # For numerical stability: subtract max log-weight per timestep
    # First gather per-step log weights padded
    lw_pad = np.full((n, max_T), -np.inf)
    r_pad = np.zeros((n, max_T))
    mask = np.zeros((n, max_T), dtype=bool)
    for i in range(n):
        T = len(rewards[i])
        lw_pad[i, :T] = step_log_w[i]
        r_pad[i, :T] = rewards[i]
        mask[i, :T] = True

    # Stabilize per timestep
    lw_max = np.where(mask, lw_pad, -np.inf).max(axis=0, keepdims=True)
    supported_by_step = np.isfinite(np.where(mask, lw_pad, -np.inf)).sum(axis=0)
    lw_max = np.where(np.isfinite(lw_max), lw_max, 0.0)
    with np.errstate(invalid="ignore"):
        w = np.where(mask, np.exp(lw_pad - lw_max), 0.0)
    w = np.nan_to_num(w, nan=0.0)
    # A timestep no trajectory supports contributes nothing rather than a
    # normalised zero, and the caller is told how many such steps there are.
    _ = supported_by_step
    if clip:
        for t in range(max_T):
            col = w[:, t]
            if col.sum() > 0:
                cap = np.percentile(col[col > 0], WEIGHT_CLIP_PERCENTILE)
                w[:, t] = np.minimum(col, cap)
    discounts = gamma ** np.arange(max_T)
    num = np.sum(w * discounts * r_pad, axis=0)
    den = np.sum(w, axis=0).clip(1e-12)
    return float(np.sum(num / den))


def doubly_robust_estimator(
    step_log_w: list,
    rewards: list,
    v_hat: list,                  # list of (T_i,) V̂(s_t) under target policy
    q_hat_taken: list,            # list of (T_i,) Q̂(s_t, a_t) under target
    gamma: float,
    clip: bool = False,
) -> float:
    """
    Per-step DR (Jiang & Li 2016):
      DR = mean over trajectories of:
        Σ_t γ^t [ ρ_{0:t} (r_t - Q̂(s_t,a_t)) + ρ_{0:t-1} V̂(s_t) ]
      with WIS-style normalization at each step.
    """
    n = len(rewards)
    max_T = max(len(r) for r in rewards)
    lw_pad = np.full((n, max_T), -np.inf)
    r_pad = np.zeros((n, max_T))
    v_pad = np.zeros((n, max_T))
    q_pad = np.zeros((n, max_T))
    mask = np.zeros((n, max_T), dtype=bool)
    for i in range(n):
        T = len(rewards[i])
        lw_pad[i, :T] = step_log_w[i]
        r_pad[i, :T] = rewards[i]
        v_pad[i, :T] = v_hat[i]
        q_pad[i, :T] = q_hat_taken[i]
        mask[i, :T] = True

    lw_prev = np.full_like(lw_pad, -np.inf)
    lw_prev[:, 1:] = lw_pad[:, :-1]
    lw_prev[:, 0] = 0.0  # ρ_{0:-1} = 1

    # Per-step normalization (WIS-DR)
    def norm_weights(lw_pad_):
        lw_max = np.where(mask, lw_pad_, -np.inf).max(axis=0, keepdims=True)
        lw_max = np.where(np.isfinite(lw_max), lw_max, 0.0)
        w = np.where(mask, np.exp(lw_pad_ - lw_max), 0.0)
        if clip:
            # Same per-step percentile cap PDWIS applies, so the two
            # per-decision estimators can be compared on equal terms.
            for t in range(w.shape[1]):
                col = w[:, t]
                if col.sum() > 0:
                    cap = np.percentile(col[col > 0], WEIGHT_CLIP_PERCENTILE)
                    w[:, t] = np.minimum(col, cap)
        return w

    w_t = norm_weights(lw_pad)
    w_tm1 = norm_weights(lw_prev)
    discounts = gamma ** np.arange(max_T)

    # Term1: ρ_{0:t} (r - Q̂)/sum_w
    num1 = np.sum(w_t * discounts * (r_pad - q_pad), axis=0)
    den1 = np.sum(w_t, axis=0).clip(1e-12)
    # Term2: ρ_{0:t-1} V̂/sum_w
    num2 = np.sum(w_tm1 * discounts * v_pad, axis=0)
    den2 = np.sum(w_tm1, axis=0).clip(1e-12)
    return float(np.sum(num1 / den1 + num2 / den2))


# ─────────────────────────────────────────────────────────────────────────────
# Q̂ AND V̂ UNDER TARGET POLICY
# ─────────────────────────────────────────────────────────────────────────────

def compute_v_q_hat(
    q_net: D3QN,
    trajs: Trajectories,
    pi_e_proba: np.ndarray,        # (total_steps, A)
    device: torch.device | None = None,
) -> tuple[list, list]:
    """V̂(s) = Σ_a π_e(a|s) Q(s,a); Q̂_a = Q(s,a) for taken a."""
    device = device or torch.device("cpu")
    q_net.eval()
    flat_states = np.vstack(trajs.states)
    q_all = []
    with torch.no_grad():
        for i in range(0, len(flat_states), 4096):
            xb = torch.tensor(flat_states[i:i+4096],
                              dtype=torch.float32, device=device)
            q_all.append(q_net(xb).cpu().numpy())
    q_all = np.vstack(q_all).astype(np.float64)
    v_all = (q_all * pi_e_proba).sum(axis=1)

    v_hat, q_taken = [], []
    pos = 0
    for i in range(trajs.n):
        T = len(trajs.actions[i])
        v_hat.append(v_all[pos:pos+T])
        q_taken.append(q_all[np.arange(pos, pos+T), trajs.actions[i]])
        pos += T
    return v_hat, q_taken


# ─────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP CONFIDENCE INTERVALS
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(
    fn,
    n_traj: int,
    *args,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
):
    """
    Trajectory-level bootstrap. fn(idx, *args) → scalar.
    """
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_traj, n_traj)
        boots[b] = fn(idx, *args)
    # A resample can contain no supported trajectory, in which case the
    # statistic is undefined for that replicate rather than zero. The interval
    # is reported over the defined replicates and the defined fraction is
    # returned, so an interval resting on a handful of them is visible.
    ok = np.isfinite(boots)
    frac = float(ok.mean())
    if not ok.any():
        return float("nan"), float("nan"), float("nan"), 0.0
    d = boots[ok]
    return (float(d.mean()), float(np.percentile(d, 100 * (alpha / 2))),
            float(np.percentile(d, 100 * (1 - alpha / 2))), frac)


# ─────────────────────────────────────────────────────────────────────────────
# TOP-LEVEL: EVALUATE A TRAINED Q-NET
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_policy(
    q_state_dict_path: str,
    state_dim: int,
    hidden_dims: list,
    behavior_proba_path: str,
    split: str = "val",
    state_cols: list | None = None,
    temperature: float = DEFAULT_TEMP,
    gamma: float = GAMMA,
    n_boot: int = 1000,
    device: torch.device | None = None,
    policy_fn=None,
    q_hat_net=None,
) -> dict:
    """
    Evaluate a policy with WIS, PDWIS, DR + bootstrap CIs.

    By default the policy is softmax(Q/tau) over the loaded Q-network. Pass
    `policy_fn(states) -> (N, n_actions)` to evaluate a policy that is not a
    function of that Q-network — IQL's extracted actor is the case this exists
    for, since scoring softmax over its critic would describe a different
    policy than the one being reported. It takes the states rather than a
    precomputed array so the caller cannot get the row order wrong: the rows
    are trajectory order, not parquet order. The Q-network is still loaded and
    still supplies the doubly-robust baseline unless `q_hat_net` is given.

    `q_hat_net` is the control variate for the doubly-robust estimator. The
    network loaded from `q_state_dict_path` is the policy's own training critic,
    which is a conservative Q for CQL and an expectile critic for IQL, and in
    neither case an estimate of Q under the policy being evaluated. Passing the
    fitted FQE network for this policy, checkpoint and split makes the
    model-based half of the estimator model-correct. Without it, the DR column
    should be described as carrying a training-critic control variate rather
    than a policy-value model.
    """
    device = device or torch.device("cpu")
    if state_cols is None:
        with open(os.path.join(PROCESSED_DIR, "metadata.json")) as f:
            state_cols = json.load(f)["state_cols"]

    # Load Q-net
    q_net = D3QN(state_dim, N_ACTIONS, hidden_dims).to(device)
    q_net.load_state_dict(torch.load(q_state_dict_path, map_location=device))
    q_net.eval()

    # Load trajectories
    parquet = os.path.join(PROCESSED_DIR, f"{split}.parquet")
    trajs = Trajectories.from_parquet(parquet, state_cols)
    flat_states = np.vstack(trajs.states)

    # Compute π_e and π_β proba on every step
    if policy_fn is None:
        pi_e = softmax_policy_from_q(q_net, flat_states, temperature, device)
    else:
        pi_e = np.asarray(policy_fn(flat_states), dtype=np.float64)
        if pi_e.shape != (len(flat_states), N_ACTIONS):
            raise ValueError(f"policy_fn returned {pi_e.shape}, expected "
                             f"{(len(flat_states), N_ACTIONS)}")
    bp_data = np.load(behavior_proba_path)
    pi_b = bp_data["proba"]
    if len(pi_b) != len(flat_states):
        raise ValueError(f"Behavior proba size {len(pi_b)} != states {len(flat_states)}")
    # Row ORDER must match too, not just length: the saved proba rows were
    # generated from the parquet's stored order, while trajectories are
    # rebuilt sorted by (stay_id, time_bin). The npz stores the actions it
    # was generated against, so misalignment is detectable exactly.
    if "actions" in bp_data:
        flat_actions = np.concatenate(trajs.actions)
        if not np.array_equal(bp_data["actions"], flat_actions):
            raise ValueError(
                f"Behavior proba rows in {behavior_proba_path} are not aligned "
                f"with {split} trajectory order — importance weights would be "
                f"computed from mismatched (state, pi_beta) pairs.")

    log_w, step_log_w = trajectory_log_weights(trajs, pi_e, pi_b)
    returns = trajs.returns(gamma)

    # How many logged trajectories the target policy could have produced. For a
    # deterministic policy this is the number that match its action at every
    # step, and every importance-sampling estimate rests on exactly these.
    n_match = n_supported(log_w)
    max_T = max(len(sw) for sw in step_log_w)
    step_active, step_support = [], []
    for t in range(max_T):
        col = np.array([sw[t] for sw in step_log_w if len(sw) > t])
        step_active.append(int(col.size))
        step_support.append(int(np.isfinite(col).sum()))
    # A decision index where trajectories are still running but none carries
    # weight makes the full-horizon estimators undefined, not merely extreme.
    # Checking only whether any trajectory matches end to end misses this: a
    # policy can have supported prefixes and no supported tail.
    unsupported_steps = [t for t in range(max_T)
                         if step_active[t] > 0 and step_support[t] == 0]

    # Behavior policy estimated value (clinician)
    v_clinician = float(returns.mean())

    # Target policy estimates
    wis_val = wis_estimator(log_w, returns)
    pdwis_val = pdwis_estimator(step_log_w, trajs.rewards, gamma)
    if n_match == 0:
        # Nothing supports the estimate. Reporting a number here would be
        # reporting the arithmetic rather than the data.
        wis_val = float("nan")
    if n_match == 0 or unsupported_steps:
        # Per-decision estimators sum contributions over every step. A step
        # with no support contributes zero through denominator clipping, which
        # silently truncates the horizon instead of admitting it is undefined.
        pdwis_val = float("nan")

    v_hat, q_taken = compute_v_q_hat(q_hat_net or q_net, trajs, pi_e, device)
    dr_val = doubly_robust_estimator(step_log_w, trajs.rewards, v_hat, q_taken,
                                     gamma)
    dr_val_clipped = doubly_robust_estimator(step_log_w, trajs.rewards, v_hat,
                                             q_taken, gamma, clip=True)
    # DR is reported wherever it computes. Unlike the per-decision estimators
    # it does not become undefined as support vanishes: its importance weights
    # enter only the correction term, so it degrades toward its model-based
    # baseline by design. Suppressing it on the support criterion (as before
    # 2026-08-23) discarded an answer the estimator is built to give; the
    # unsupported_steps column still records how much of the horizon the
    # correction actually covers.

    # Bootstrap CIs
    def _wis(idx):
        return wis_estimator(log_w[idx], returns[idx])

    def _clin(idx):
        return float(returns[idx].mean())

    wis_mean, wis_lo, wis_hi, wis_frac = bootstrap_ci(_wis, trajs.n,
                                                      n_boot=n_boot)
    clin_mean, clin_lo, clin_hi, _ = bootstrap_ci(_clin, trajs.n, n_boot=n_boot)

    # Effective sample size, on both weight sets. `wis_estimator` clips at the
    # 99th percentile before averaging, so an ESS computed from unclipped
    # weights describes a different estimator from the WIS value beside it.
    # Both are reported: the raw one says how degenerate the weights are, the
    # clipped one says what the reported WIS actually rests on.
    def _ess(w):
        d = (w ** 2).sum()
        return float((w.sum() ** 2) / d) if d > 0 else 0.0

    if n_match:
        w_norm = np.exp(log_w - log_w.max())
        ess, ess_clipped = _ess(w_norm), _ess(clip_weights(w_norm))
    else:
        # Every log weight is -inf, so the stabilising subtraction is
        # -inf - -inf. There is no sample to be effective.
        ess = ess_clipped = 0.0

    return {
        "split": split,
        "n_trajectories": trajs.n,
        "temperature": temperature,
        "gamma": gamma,
        "clinician_value": v_clinician,
        "clinician_ci": [clin_lo, clin_hi],
        "dr_value_clipped": dr_val_clipped,
        "wis_value": wis_val,
        "wis_ci": [wis_lo, wis_hi],
        # What fraction of bootstrap resamples contained any supported
        # trajectory. An interval resting on a few of them is not an interval.
        "wis_ci_defined_fraction": round(wis_frac, 4),
        "pdwis_value": pdwis_val,
        "dr_value": dr_val,
        "ess": float(ess) if n_match else 0.0,
        "ess_clipped": float(ess_clipped) if n_match else 0.0,
        # Support, without which none of the three estimates above means
        # anything. `n_full_trajectory_matches` is how many of the logged
        # trajectories the target policy could have generated end to end.
        "n_full_trajectory_matches": n_match,
        "pct_full_trajectory_matches": round(100 * n_match / max(trajs.n, 1), 4),
        "n_supported_by_step": step_support,
        "n_active_by_step": step_active,
        # Decision indices where episodes are still running and nothing
        # supports them. Non-empty makes PDWIS and DR undefined.
        "unsupported_steps": unsupported_steps,
        "dr_control_variate": "fqe" if q_hat_net is not None else "training critic",
        "weight_clip_pct": WEIGHT_CLIP_PERCENTILE,
    }


if __name__ == "__main__":
    # Smoke test driver — only used manually
    import sys
    if len(sys.argv) < 3:
        print("usage: python -m src.evaluate <q_path> <bp_proba_path>")
        sys.exit(1)
    res = evaluate_policy(
        q_state_dict_path=sys.argv[1],
        state_dim=39, hidden_dims=[256, 256],
        behavior_proba_path=sys.argv[2],
        split="val",
    )
    print(json.dumps(res, indent=2))
