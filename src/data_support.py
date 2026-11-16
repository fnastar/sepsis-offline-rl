"""
Measure action support on held-out decisions.

Reports two diagnostics:
  1. The estimated behaviour probability of the recommended action.
     Recommendations below 0.01 are counted as unsupported.
  2. The fraction of 50 nearest training decisions choosing that action,
     using 80,000 reference and 15,000 query decisions.

Evaluates greedy D3QN/CQL policies, IQL actors and observed clinician actions.
Optional QR-D3QN runs are skipped when their checkpoints are absent. These
diagnostics measure coverage; they do not establish treatment benefit.

SEPSIS_SUPPORT_CKPT selects best (early) or final checkpoints. Results are
written under <OUTPUT_BASE>/data_support as support_metrics.json or
support_metrics_final.json.

    python -m src.data_support
    SEPSIS_SUPPORT_CKPT=final python -m src.data_support
"""
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

from config.paths import PROCESSED_DIR, OUTPUT_BASE, FIG_DIR, BEHAVIOR_POLICY_DIR
from config.rl import N_ACTIONS, N_IV_LEVELS
from src.models.d3qn import D3QN
from src.models.qr_d3qn import QRD3QN
from src.models.behavior_policy import MLPBehaviorPolicy

# Which checkpoint rule this run describes. Both are primary outputs of the
# campaign, so a single support table would silently describe only one of them
# while the results and significance tables report both.
CKPT = os.environ.get("SEPSIS_SUPPORT_CKPT", "best")
_SFX = "" if CKPT == "best" else f"_{CKPT}"
OUTPUT_DIR = os.path.join(OUTPUT_BASE, "data_support")
# Behavior policy is label-independent, so it is shared across namespaces.
BP_WEIGHTS = os.path.join(BEHAVIOR_POLICY_DIR, "mlp_behavior.pt")
HIDDEN_DIMS = [256, 256]

# Headline checkpoints (seed 42) — one per method. Paths follow OUTPUT_BASE so the
# diagnostic reads the checkpoints from the active namespace (24h or 90d).
_Q = "d3qn_best.pt" if CKPT == "best" else "d3qn_final.pt"
_A = "actor_best.pt" if CKPT == "best" else "actor_criticfinal.pt"

# All five training seeds, so this is a method-level statement rather than an
# illustration from one run. The expensive part of the diagnostic is the k-NN
# fit, which is shared across policies, so the extra seeds cost forward passes
# only.
SEEDS = (42, 43, 44, 45, 46)
RUN_PATTERNS = {
    "CQL (alpha=2.0)": ("cql_a2.0_seed{seed}", "d3qn"),
    "QR-D3QN (mean)":  ("qr_d3qn_seed{seed}", "qr"),
    "IQL":             ("iql_seed{seed}", "actor"),
    "D3QN":            ("d3qn_seed{seed}", "d3qn"),
}


def _checkpoint(pattern: str, kind: str, seed: int) -> str:
    f = _A if kind == "actor" else _Q
    return os.path.join(OUTPUT_BASE, pattern.format(seed=seed), f)


# Seed 42 alone, kept because several call sites and the smoke tests refer to
# it. The reported metrics come from all five.
CHECKPOINTS = {
    "CQL (alpha=2.0)": (os.path.join(OUTPUT_BASE, f"cql_a2.0_seed42/{_Q}"), "d3qn"),
    "QR-D3QN (mean)":  (os.path.join(OUTPUT_BASE, f"qr_d3qn_seed42/{_Q}"),   "qr"),
    # IQL's policy is the extracted actor. Reading its critic here would
    # describe the data support of softmax(Q), which is not the policy the
    # thesis evaluates.
    "IQL":             (os.path.join(OUTPUT_BASE, f"iql_seed42/{_A}"),      "actor"),
    "D3QN":            (os.path.join(OUTPUT_BASE, f"d3qn_seed42/{_Q}"),      "d3qn"),
}

# k-NN diagnostic subsample sizes (kept tractable; seeded for reproducibility).
N_REF = 80_000     # clinician reference decision points (from train split)
N_QRY = 15_000     # query decision points (from test split)
K_NEIGHBORS = 50
SEED = 42

# A recommended action is "effectively unsupported" below this propensity.
UNSUPPORTED_THRESH = 0.01


# ─────────────────────────────────────────────────────────────────────────────
# POLICY RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _greedy_actions(kind: str, ckpt_path: str, states: np.ndarray) -> np.ndarray:
    """Greedy (argmax) action per state for a D3QN/CQL or QR-D3QN checkpoint."""
    device = torch.device("cpu")
    state_dim = states.shape[1]
    if kind == "actor":
        # The critic the actor was extracted against is named by the file, so
        # it has to be passed. Defaulting to "best" here made the final-
        # checkpoint support run silently identical to the early one.
        from src.train_iql_actor import load_actor
        critic = "final" if ckpt_path.endswith("actor_criticfinal.pt") else "best"
        actor = load_actor(os.path.dirname(ckpt_path), device, critic=critic)
        out = []
        for i in range(0, len(states), 8192):
            xb = torch.tensor(states[i:i + 8192], dtype=torch.float32)
            out.append(actor.greedy(xb).numpy())
        return np.concatenate(out)
    if kind == "qr":
        net = QRD3QN(state_dim, N_ACTIONS, HIDDEN_DIMS).to(device)
        net.load_state_dict(torch.load(ckpt_path, map_location=device,
                                       weights_only=True))
        net.eval()
        out = []
        for i in range(0, len(states), 8192):
            xb = torch.tensor(states[i:i + 8192], dtype=torch.float32)
            out.append(net.q_values(xb).argmax(dim=1).numpy())  # mean-Q policy
        return np.concatenate(out)
    # d3qn / cql share the dueling architecture
    net = D3QN(state_dim, N_ACTIONS, HIDDEN_DIMS).to(device)
    net.load_state_dict(torch.load(ckpt_path, map_location=device,
                                   weights_only=True))
    net.eval()
    out = []
    for i in range(0, len(states), 8192):
        xb = torch.tensor(states[i:i + 8192], dtype=torch.float32)
        out.append(net(xb).argmax(dim=1).numpy())
    return np.concatenate(out)


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def _propensity_stats(pi_beta: np.ndarray, a_star: np.ndarray) -> dict:
    """Metric A: pi_beta(a*|s) summary stats for one policy."""
    p = pi_beta[np.arange(len(a_star)), a_star]
    return {
        "mean_propensity": float(p.mean()),
        "median_propensity": float(np.median(p)),
        "pct_unsupported": float(100.0 * (p < UNSUPPORTED_THRESH).mean()),
    }


def _knn_support_stats(neigh_actions: np.ndarray, a_star_qry: np.ndarray) -> dict:
    """Metric B: empirical local support among k nearest clinician decisions."""
    support = (neigh_actions == a_star_qry[:, None]).mean(axis=1)  # (N_QRY,)
    return {
        "mean_local_support": float(support.mean()),
        "median_local_support": float(np.median(support)),
        "pct_zero_support": float(100.0 * (support == 0).mean()),
    }


def _vaso_rate(a_star: np.ndarray) -> float:
    """Fraction of decisions recommending any vasopressor (vaso_level > 0)."""
    return float((a_star // N_IV_LEVELS > 0).mean())


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run(split: str = "test") -> dict:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    with open(os.path.join(PROCESSED_DIR, "metadata.json")) as f:
        state_cols = json.load(f)["state_cols"]

    # ── Evaluation set: all decision points on the held-out split ─────────────
    eval_df = pd.read_parquet(os.path.join(PROCESSED_DIR, f"{split}.parquet"))
    s_eval = eval_df[state_cols].values.astype(np.float32)
    a_obs = eval_df["action"].values.astype(int)           # clinician's actual action
    print(f"[{split}] decision points: {len(s_eval):,}")

    # ── Recommended actions per policy (clinician = observed action) ──────────
    a_star = {"Clinician (observed)": [a_obs]}
    for name, (pattern, kind) in RUN_PATTERNS.items():
        per_seed = []
        for seed in SEEDS:
            ckpt = _checkpoint(pattern, kind, seed)
            if not os.path.exists(ckpt):
                continue
            per_seed.append(_greedy_actions(kind, ckpt, s_eval))
        if not per_seed:
            print(f"  [SKIP] {name}: no checkpoints found")
            continue
        a_star[name] = per_seed
        print(f"  {name}: recommendations computed for {len(per_seed)} seeds")

    # ── Metric A: behaviour-policy propensity pi_beta(a*|s) ───────────────────
    bp = MLPBehaviorPolicy(state_dim=len(state_cols))
    bp.net.load_state_dict(torch.load(BP_WEIGHTS, map_location=bp.device,
                                      weights_only=True))
    pi_beta = bp.predict_proba(s_eval)                     # (N, A)

    # ── Metric B: k-NN local support (subsampled, standardized state space) ───
    train_df = pd.read_parquet(os.path.join(PROCESSED_DIR, "train.parquet"))
    s_tr = train_df[state_cols].values.astype(np.float32)
    a_tr = train_df["action"].values.astype(int)
    ref_idx = rng.choice(len(s_tr), size=min(N_REF, len(s_tr)), replace=False)
    X_ref, a_ref = s_tr[ref_idx], a_tr[ref_idx]

    # z-score by the reference stats so Euclidean distance is scale-invariant
    mu, sigma = X_ref.mean(axis=0), X_ref.std(axis=0) + 1e-8
    X_ref_z = (X_ref - mu) / sigma

    qry_idx = rng.choice(len(s_eval), size=min(N_QRY, len(s_eval)), replace=False)
    X_qry_z = (s_eval[qry_idx] - mu) / sigma

    print(f"k-NN: {len(X_qry_z):,} queries vs {len(X_ref_z):,} clinician refs "
          f"(k={K_NEIGHBORS})...")
    nn = NearestNeighbors(n_neighbors=K_NEIGHBORS, algorithm="auto")
    nn.fit(X_ref_z)
    _, neigh = nn.kneighbors(X_qry_z)                      # (N_QRY, k) ref indices
    neigh_actions = a_ref[neigh]                           # (N_QRY, k) clinician actions

    # ── Assemble per-policy results ───────────────────────────────────────────
    results = {
        "split": split,
        "n_eval_points": int(len(s_eval)),
        "knn": {"n_ref": int(len(X_ref_z)), "n_query": int(len(X_qry_z)),
                "k": K_NEIGHBORS, "seed": SEED},
        "unsupported_threshold": UNSUPPORTED_THRESH,
        "policies": {},
    }
    results["seeds"] = list(SEEDS)
    results["per_seed"] = {}
    for name, pols in a_star.items():
        seed_rows = []
        for a_pol in pols:
            r = {"vaso_recommendation_rate": _vaso_rate(a_pol)}
            r.update(_propensity_stats(pi_beta, a_pol))
            r.update(_knn_support_stats(neigh_actions, a_pol[qry_idx]))
            seed_rows.append(r)
        keys = seed_rows[0].keys()
        row = {k: float(np.mean([r[k] for r in seed_rows])) for k in keys}
        if len(seed_rows) > 1:
            row.update({f"{k}_sd": float(np.std([r[k] for r in seed_rows], ddof=1))
                        for k in keys})
        row["n_seeds"] = len(seed_rows)
        results["per_seed"][name] = seed_rows
        results["policies"][name] = row
        print(f"\n{name}  (n={len(seed_rows)} seed"
              f"{'s' if len(seed_rows) > 1 else ''})")
        print(f"  vaso-rec rate      : {row['vaso_recommendation_rate']*100:5.1f}%")
        print(f"  mean pi_beta(a*|s) : {row['mean_propensity']:.4f}")
        print(f"  % unsupported(<{UNSUPPORTED_THRESH}): {row['pct_unsupported']:5.1f}%")
        print(f"  mean k-NN support  : {row['mean_local_support']:.4f}")
        print(f"  % zero k-NN support: {row['pct_zero_support']:5.1f}%")

    out_path = os.path.join(OUTPUT_DIR, f"support_metrics{_SFX}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")

    _plot(results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────────────────────────────────────

def _plot(results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")

    # Order: in-distribution ceiling -> ... -> most OOD
    order = ["Clinician (observed)", "CQL (alpha=2.0)", "QR-D3QN (mean)",
             "IQL", "D3QN"]
    color_by = {"Clinician (observed)": "#2a9d8f", "CQL (alpha=2.0)": "#457b9d",
                "QR-D3QN (mean)": "#e9c46a", "IQL": "#f4a261", "D3QN": "#e76f51"}
    names = [n for n in order if n in results["policies"]]
    labels = [n.replace("alpha", r"$\alpha$") for n in names]
    support = [results["policies"][n]["mean_local_support"] for n in names]
    unsup = [results["policies"][n]["pct_unsupported"] for n in names]
    palette = [color_by[n] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    b1 = ax1.bar(labels, support, color=palette, edgecolor="black", linewidth=0.6)
    ax1.set_ylabel("Mean local data support\n(fraction of $k$ nearest clinicians taking $a^*$)")
    ax1.set_title("(a) Empirical $k$-NN action support  (higher = better)")
    ax1.set_ylim(0, max(support) * 1.18)
    for r, v in zip(b1, support):
        ax1.text(r.get_x() + r.get_width() / 2, v, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

    b2 = ax2.bar(labels, unsup, color=palette, edgecolor="black", linewidth=0.6)
    ax2.set_ylabel(rf"% of decisions with $\pi_\beta(a^*|s) < {results['unsupported_threshold']:.2f}$")
    ax2.set_title("(b) Effectively unsupported recommendations  (lower = better)")
    ax2.set_ylim(0, max(max(unsup) * 1.18, 5))
    for r, v in zip(b2, unsup):
        ax2.text(r.get_x() + r.get_width() / 2, v, f"{v:.1f}%",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

    for ax in (ax1, ax2):
        ax.tick_params(axis="x", rotation=12)

    k = results["knn"]
    fig.suptitle(
        f"Data support of policy recommendations on the test split "
        f"(N = {results['n_eval_points']:,} decisions; "
        f"$k$-NN: {k['n_query']:,} queries vs {k['n_ref']:,} clinician refs, $k$={k['k']})",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"data_support{_SFX}.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG_DIR}/data_support{_SFX}.{{pdf,png}}")


if __name__ == "__main__":
    run("test")
