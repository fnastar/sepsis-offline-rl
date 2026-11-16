"""
final_ope.py: Importance-sampling diagnostics for the final Sepsis-3 run (RQ1).

FQE answers "which policy scores higher". RQ1 asks something prior to that:
whether the standard estimators can carry a claim on this data at all. That
needs WIS, per-decision WIS and doubly-robust run alongside it, with the
effective sample size reported, because ESS is what says whether an estimate
rests on the cohort or on a handful of trajectories.

Two policy conventions are scored, deliberately:

* GREEDY — the policy the protocol actually evaluates, and the one FQE scores.
  Importance weights for a deterministic policy are non-zero only where the
  clinician matched it at every step, so ESS collapse here is not a defect of
  the run, it is the finding.
* SOFTMAX(Q/0.5) — the convention the earlier campaign used. Scored for
  comparability, and because a stochastic policy is the standard mitigation
  when IS degenerates.

The behaviour policy is refit per namespace, and its probabilities must be
generated in trajectory order (sorted by stay_id, time_bin) or the weights
pair states with the wrong propensities. `evaluate_policy` asserts this by
comparing stored actions, so a misalignment fails loudly rather than silently.

Writes outputs/runs_90d_final/ope_diagnostics.csv and .json.
"""
import json
import os

import numpy as np
import pandas as pd
import torch

from config.paths import BEHAVIOR_POLICY_DIR, OUTPUT_BASE, PROCESSED_DIR
from config.rl import N_ACTIONS
from src.evaluate import (WEIGHT_CLIP_PERCENTILE, Trajectories,
                          evaluate_policy)
from src.models.behavior_policy import (EPSILON_FLOOR,
                                        MLPBehaviorPolicyNet)
from src.final_campaign import is_complete
from src.utils import ensure_dir


def _fqe_net_for(run_dir: str, kind: str, ckpt: str, split: str,
                 state_dim: int, device):
    """
    The fitted FQE network for this policy, checkpoint and split.

    This is the doubly-robust control variate. The policy's own training critic
    is not an estimate of Q under the evaluated policy: for CQL it is a
    conservative Q, and for IQL it is an expectile critic whose actor was
    extracted from it separately. Returns None when the fit is missing, so the
    caller can fall back and say so rather than silently substituting.
    """
    from src.models.fqe import FQENet
    if kind == "iql":
        sub = f"fqe_{split}" if ckpt == "best" else f"fqe_{split}_criticfinal"
    else:
        sub = f"fqe_{split}" + ("" if ckpt == "best" else "_finalckpt")
    path = os.path.join(run_dir, sub, "fqe_net.pt")
    if not os.path.exists(path):
        return None
    net = FQENet(state_dim, N_ACTIONS, [256, 256]).to(device)
    net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    net.eval()
    return net


def _one_hot(idx: np.ndarray) -> np.ndarray:
    """Deterministic policy as a distribution: all mass on the chosen action."""
    p = np.zeros((len(idx), N_ACTIONS), dtype=np.float64)
    p[np.arange(len(idx)), idx] = 1.0
    return p


def _greedy_q_fn(ckpt_path: str, state_dim: int, device):
    """
    Exact argmax over Q, as a one-hot distribution.

    This is the policy FQE scores. A softmax at a low temperature has the same
    modal action but different likelihood ratios, so its importance weights,
    effective sample size and every estimator built on them describe something
    else. `trajectory_log_weights` floors the selected probability at 1e-12, so
    an exact zero elsewhere is numerically safe; that floor is regularisation
    and the effective sample size of a deterministic policy should be expected
    to collapse.
    """
    from src.models.d3qn import D3QN
    net = D3QN(state_dim, N_ACTIONS, [256, 256]).to(device)
    net.load_state_dict(torch.load(ckpt_path, map_location=device,
                                   weights_only=True))
    net.eval()

    def fn(states):
        out = []
        with torch.no_grad():
            for i in range(0, len(states), 4096):
                xb = torch.tensor(states[i:i + 4096], dtype=torch.float32,
                                  device=device)
                out.append(net(xb).argmax(dim=1).cpu().numpy())
        return _one_hot(np.concatenate(out))
    return fn


def _actor_policy_fn(run_dir: str, ckpt: str, device):
    """
    softmax(actor logits / tau) for an IQL run, or None if it has no actor.

    `ckpt` names the critic the actor was extracted against: "best" is the
    lowest-validation-loss critic, "final" is epoch 40. Returns a factory so
    the actor is loaded once per run rather than once per temperature.
    """
    import torch
    from src.train_iql_actor import load_actor
    try:
        actor = load_actor(run_dir, device,
                           critic="best" if ckpt == "best" else "final")
    except FileNotFoundError:
        return None

    def factory(temperature):
        def fn(states):
            out = []
            with torch.no_grad():
                for i in range(0, len(states), 4096):
                    xb = torch.tensor(states[i:i + 4096], dtype=torch.float32,
                                      device=device)
                    logits = actor(xb)
                    if temperature == "greedy":
                        out.append(_one_hot(logits.argmax(dim=1).cpu().numpy()))
                    else:
                        out.append(torch.softmax(
                            logits / max(temperature, 1e-3), dim=1).cpu().numpy())
            return np.vstack(out)
        return fn
    return factory

SEEDS = (42, 43, 44, 45, 46)
CQL_ALPHA = 2.0


def build_behavior_proba(split: str, state_cols: list, device) -> str:
    """
    Behaviour-policy probabilities for one split, in trajectory order.

    Saved next to the split rather than reused from the fit, because the fit
    writes rows in parquet order while `Trajectories` rebuilds them sorted by
    (stay_id, time_bin). Storing the actions alongside lets the consumer prove
    the alignment instead of assuming it.
    """
    out = os.path.join(BEHAVIOR_POLICY_DIR, f"mlp_proba_{split}_ordered.npz")
    if os.path.exists(out):
        return out
    trajs = Trajectories.from_parquet(
        os.path.join(PROCESSED_DIR, f"{split}.parquet"), state_cols)
    states = np.vstack(trajs.states).astype(np.float32)
    actions = np.concatenate(trajs.actions)

    net = MLPBehaviorPolicyNet(len(state_cols), N_ACTIONS, 128).to(device)
    net.load_state_dict(torch.load(
        os.path.join(BEHAVIOR_POLICY_DIR, "mlp_behavior.pt"),
        map_location=device, weights_only=True))
    net.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(states), 8192):
            sb = torch.tensor(states[i:i + 8192], device=device)
            probs.append(torch.softmax(net(sb), dim=1).cpu().numpy())
    proba = np.vstack(probs)

    # The same floor-and-renormalise the fitted behaviour policy applies in
    # `predict_proba`. Without it these ordered propensities follow a different
    # convention from the fit they come from, and a raw softmax can put an
    # arbitrarily small number in the denominator of an importance weight.
    proba = np.clip(proba, EPSILON_FLOOR, None)
    proba = proba / proba.sum(axis=1, keepdims=True)

    np.savez_compressed(out, proba=proba, actions=actions,
                        epsilon_floor=EPSILON_FLOOR)
    print(f"  behaviour proba [{split}]: {proba.shape} -> {out} "
          f"(floor {EPSILON_FLOOR}, min {proba.min():.2e})")
    return out


def run(splits=("val", "test"), policies=("greedy", 0.5)) -> pd.DataFrame:
    """
    Score every trained policy under the importance-sampling family.

    Two policies per run. ``"greedy"`` is the exact argmax, as a one-hot
    distribution, which is the policy FQE evaluates. A low-temperature softmax
    was used here previously on the grounds that an exact one-hot would send
    log-weights to negative infinity wherever the clinician deviated. It does,
    and that is correct: a trajectory the policy could not have produced must
    carry zero weight. `trajectory_log_weights` no longer floors the target
    probability, so those weights are exactly zero, and WIS, PDWIS and DR are
    undefined rather than self-normalised out of nothing when support is
    absent. Expect the supported fraction to be small: on this cohort it runs
    from zero to a few per cent of trajectories.

    ``0.5`` is the stochastic sensitivity, and the ordering under it is the
    thing to read rather than the level.
    """
    device = torch.device("cpu")
    with open(os.path.join(PROCESSED_DIR, "metadata.json")) as f:
        meta = json.load(f)
    state_cols = meta["state_cols"]
    dim = meta["n_state_features"]

    bp = {s: build_behavior_proba(s, state_cols, device) for s in splits}

    runs = ([(f"cql_a{CQL_ALPHA}_seed{s}", "cql", s) for s in SEEDS]
            + [(f"d3qn_seed{s}", "d3qn", s) for s in SEEDS]
            + [(f"iql_seed{s}", "iql", s) for s in SEEDS])
    # Derived from the grid actually iterated below, so it cannot drift from it.
    expected = len(runs) * 2 * len(splits) * len(policies)

    rows = []
    for name, kind, seed in runs:
        d = os.path.join(OUTPUT_BASE, name)
        # A checkpoint exists after the first epoch, so its presence is not
        # evidence the run finished. Evaluating a half-trained policy would put
        # a row in the table that describes a network nobody trained.
        if not is_complete(d):
            print(f"  {name}: not a completed run, not evaluated")
            continue
        for ckpt in ("best", "final"):
            path = os.path.join(d, f"d3qn_{ckpt}.pt")
            if not os.path.exists(path):
                continue
            # IQL's policy is its extracted actor. Softmax over its critic is
            # a different policy, and reporting importance weights for it
            # under the name IQL is the defect this branch removes. The critic
            # still supplies the doubly-robust baseline.
            fn = None
            if kind == "iql":
                fn = _actor_policy_fn(d, ckpt, device)
                if fn is None:
                    print(f"  {name}/{ckpt}: no extracted actor, skipping")
                    continue
            for split in splits:
                for temp in policies:
                    if fn is not None:
                        pol_fn = fn(temp)
                    elif temp == "greedy":
                        pol_fn = _greedy_q_fn(path, dim, device)
                    else:
                        pol_fn = None
                    # The saved FQE net was fitted under the greedy policy,
                    # so it is Q^greedy. Using it for the softmax row would put
                    # the wrong policy's value function in the control variate,
                    # so that row keeps the training critic and says so.
                    qhat = (_fqe_net_for(d, kind, ckpt, split, dim, device)
                            if temp == "greedy" else None)
                    if temp == "greedy" and qhat is None:
                        print(f"  {name}/{ckpt}/{split}: no FQE net, "
                              f"DR falls back to the training critic")
                    try:
                        r = evaluate_policy(
                            q_state_dict_path=path, state_dim=dim,
                            hidden_dims=[256, 256],
                            behavior_proba_path=bp[split], split=split,
                            temperature=(1.0 if temp == "greedy" else temp),
                            n_boot=200, device=device, policy_fn=pol_fn,
                            q_hat_net=qhat)
                        # The control variate moves the doubly-robust value by
                        # a factor of three on this data, so reporting one
                        # number would hide that the estimator is not settled.
                        # Both are recorded and neither is preferred here.
                        r["dr_training_critic"] = (
                            r["dr_value"] if qhat is None else
                            evaluate_policy(
                                q_state_dict_path=path, state_dim=dim,
                                hidden_dims=[256, 256],
                                behavior_proba_path=bp[split], split=split,
                                temperature=(1.0 if temp == "greedy" else temp),
                                n_boot=1, device=device, policy_fn=pol_fn,
                                q_hat_net=None)["dr_value"])
                    except Exception as e:  # noqa: BLE001
                        print(f"  {name}/{ckpt}/{split}/T={temp}: FAILED {e}")
                        continue
                    rows.append({
                        "method": kind, "seed": seed, "checkpoint": ckpt,
                        "split": split,
                        "temperature": (None if temp == "greedy" else temp),
                        "policy": ("greedy (exact argmax)" if temp == "greedy"
                                   else f"softmax T={temp}"),
                        "wis": r["wis_value"], "pdwis": r["pdwis_value"],
                        "dr": r["dr_value"],
                        "dr_clipped": r["dr_value_clipped"], "ess": r["ess"],
                        "ess_clipped": r.get("ess_clipped"),
                        "dr_control_variate": r.get("dr_control_variate"),
                        "dr_training_critic": r.get("dr_training_critic"),
                        "n_trajectories": r["n_trajectories"],
                        "n_full_trajectory_matches":
                            r.get("n_full_trajectory_matches"),
                        "pct_full_trajectory_matches":
                            r.get("pct_full_trajectory_matches"),
                        "wis_ci_defined_fraction":
                            r.get("wis_ci_defined_fraction"),
                        "unsupported_steps":
                            json.dumps(r.get("unsupported_steps")),
                        "n_supported_by_step":
                            json.dumps(r.get("n_supported_by_step")),
                        "n_active_by_step":
                            json.dumps(r.get("n_active_by_step")),
                        "clinician_empirical": r["clinician_value"],
                    })
                    print(f"  {name:<22} {ckpt:<5} {split:<4} "
                          f"{str(temp):<7} "
                          f"WIS={r['wis_value']:+8.3f} PDWIS={r['pdwis_value']:+8.3f} "
                          f"DR={r['dr_value']:+8.3f} ESS={r['ess']:.2f}"
                          f" ({r.get('ess_clipped', float('nan')):.2f} clipped)")

    df = pd.DataFrame(rows)
    ensure_dir(OUTPUT_BASE)
    df.to_csv(os.path.join(OUTPUT_BASE, "ope_diagnostics.csv"), index=False)

    # The approved grid: three methods, five seeds, two checkpoints, two
    # splits, two policy conventions. Every per-policy exception above is
    # caught so one failure cannot cost the rest, which means a shrunken table
    # would otherwise look finished. A row whose estimates are NaN for want of
    # support still counts as present; an absent row does not.
    missing = expected - len(df)

    # The control variate must be the same one throughout each column, or the
    # DR values are not comparable under a single name.
    if len(df) and "dr_control_variate" in df:
        g = df[df["policy"].str.startswith("greedy")]["dr_control_variate"]
        sm = df[~df["policy"].str.startswith("greedy")]["dr_control_variate"]
        bad_cv = (set(g.dropna()) - {"fqe"}) | (set(sm.dropna()) - {"training critic"})
    else:
        bad_cv = set()

    summary = {}
    if len(df):
        for pol in df["policy"].unique():
            sub = df[(df["policy"] == pol) & (df["split"] == "test")]
            summary[pol] = {
                # Two ESS definitions. The raw one says how degenerate the
                # weights are, the clipped one matches the weights the WIS
                # value beside it was computed from.
                "ess_median": float(sub["ess"].median()),
                "ess_max": float(sub["ess"].max()),
                "ess_clipped_median": (float(sub["ess_clipped"].median())
                                       if "ess_clipped" in sub else None),
                "ess_over_n_pct": float(
                    100 * sub["ess"].median() / sub["n_trajectories"].iloc[0]),
                "weight_clip_pct": WEIGHT_CLIP_PERCENTILE,
                "wis_range": [float(sub["wis"].min()), float(sub["wis"].max())],
                "pdwis_range": [float(sub["pdwis"].min()), float(sub["pdwis"].max())],
                "dr_range": [float(sub["dr"].min()), float(sub["dr"].max())],
                # Which model the doubly-robust column used. "fqe" is the
                # policy-matched value function; "training critic" is not, and
                # a DR value carrying that label is a diagnostic rather than a
                # model-correct estimate.
                "dr_control_variate": sorted(
                    set(sub["dr_control_variate"].dropna())) or None,
            }
    out = {"rows_expected": expected, "rows_found": int(len(df)),
           "complete": missing == 0, "by_policy": summary}
    with open(os.path.join(OUTPUT_BASE, "ope_diagnostics.json"), "w") as f:
        json.dump(out, f, indent=2)
    if bad_cv:
        offenders = df[df["policy"].str.startswith("greedy")
                       & (df["dr_control_variate"] != "fqe")]
        where = sorted({f"{r.method}_seed{r.seed}/{r.checkpoint}/{r.split}"
                        for r in offenders.itertuples()})
        raise AssertionError(
            f"doubly-robust control variate is not consistent within a column: "
            f"saw {sorted(bad_cv)} on greedy rows. Every greedy row must use "
            f"the fitted FQE net and every softmax row the training critic, or "
            f"the DR column mixes two estimators under one name. A greedy row "
            f"falls back only when its FQE fit is missing, so this usually "
            f"means the fit has not been produced yet: {where[:12]}")
    if missing:
        raise AssertionError(
            f"OPE produced {len(df)} rows against {expected} expected. A "
            f"missing policy must not shrink the table quietly. Present: "
            f"{df.groupby(['method', 'checkpoint']).size().to_dict()}")
    print(json.dumps(summary, indent=2))
    return df


if __name__ == "__main__":
    run()
