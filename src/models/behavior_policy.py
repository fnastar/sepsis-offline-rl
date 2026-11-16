"""
behavior_policy.py: Estimate the clinician (behavior) policy π_β(a|s) from
offline data. Required for off-policy evaluation (importance sampling).

Two estimators:
- LogisticBehaviorPolicy: multinomial logistic regression (calibrated, fast)
- MLPBehaviorPolicy:      small neural net (more flexible, smoothed)

Outputs P(a|s) over the full 25-action space. Unobserved actions get a small
floor probability (epsilon) to avoid division-by-zero in importance ratios.
"""
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

from config.paths import PROCESSED_DIR, BEHAVIOR_POLICY_DIR
from config.rl import N_ACTIONS

EPSILON_FLOOR = 1e-3  # minimum probability for any action


# ─────────────────────────────────────────────────────────────────────────────
# LOGISTIC REGRESSION (fast, well-calibrated)
# ─────────────────────────────────────────────────────────────────────────────

class LogisticBehaviorPolicy:
    """Multinomial logistic regression P(a|s)."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        # sklearn ≥1.5 removed multi_class arg; lbfgs uses multinomial by default.
        self.model = LogisticRegression(
            solver="lbfgs", C=C, max_iter=max_iter, n_jobs=-1,
        )
        self.classes_seen = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        self.classes_seen = self.model.classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(a|s) over full N_ACTIONS space, with epsilon floor."""
        p_seen = self.model.predict_proba(X)
        # Expand to full action space (unseen actions get 0 → floored to epsilon)
        p_full = np.full((len(X), N_ACTIONS), EPSILON_FLOOR)
        for i, a in enumerate(self.classes_seen):
            p_full[:, a] = p_seen[:, i]
        # Renormalize so each row sums to 1
        p_full = np.clip(p_full, EPSILON_FLOOR, None)
        p_full = p_full / p_full.sum(axis=1, keepdims=True)
        return p_full


# ─────────────────────────────────────────────────────────────────────────────
# MLP (more flexible, smoother)
# ─────────────────────────────────────────────────────────────────────────────

class MLPBehaviorPolicyNet(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class MLPBehaviorPolicy:
    def __init__(
        self,
        state_dim: int,
        n_actions: int = N_ACTIONS,
        hidden: int = 128,
        lr: float = 3e-4,
        epochs: int = 10,
        batch_size: int = 1024,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = MLPBehaviorPolicyNet(state_dim, n_actions, hidden).to(self.device)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False):
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.long, device=self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        n = len(X_t)
        for ep in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            total = 0.0
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                logits = self.net(X_t[idx])
                loss = loss_fn(logits, y_t[idx])
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item() * len(idx)
            if verbose:
                print(f"  bp epoch {ep+1}: loss={total/n:.4f}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 4096):
                xb = torch.tensor(X[i:i+4096], dtype=torch.float32, device=self.device)
                p = torch.softmax(self.net(xb), dim=1).cpu().numpy()
                out.append(p)
        p_full = np.vstack(out)
        p_full = np.clip(p_full, EPSILON_FLOOR, None)
        p_full = p_full / p_full.sum(axis=1, keepdims=True)
        return p_full


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE: train + save behavior policies on the processed splits
# ─────────────────────────────────────────────────────────────────────────────

def fit_behavior_policies(
    output_dir: str = BEHAVIOR_POLICY_DIR,
    use_mlp: bool = True,
):
    """Fit logistic and MLP behavior policies on training data."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(PROCESSED_DIR, "metadata.json")) as f:
        meta = json.load(f)
    state_cols = meta["state_cols"]

    train_df = pd.read_parquet(os.path.join(PROCESSED_DIR, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(PROCESSED_DIR, "val.parquet"))

    X_tr = train_df[state_cols].values.astype(np.float32)
    y_tr = train_df["action"].values.astype(int)
    X_va = val_df[state_cols].values.astype(np.float32)
    y_va = val_df["action"].values.astype(int)

    results = {}

    print("Fitting LogisticBehaviorPolicy...")
    log_bp = LogisticBehaviorPolicy().fit(X_tr, y_tr)
    p_va = log_bp.predict_proba(X_va)
    log_acc = (p_va.argmax(1) == y_va).mean()
    log_nll = -np.log(p_va[np.arange(len(y_va)), y_va] + 1e-12).mean()
    results["logistic"] = {"val_acc": float(log_acc), "val_nll": float(log_nll)}
    np.savez(os.path.join(output_dir, "logistic_proba_val.npz"),
             proba=p_va, actions=y_va)

    if use_mlp:
        print("Fitting MLPBehaviorPolicy...")
        mlp_bp = MLPBehaviorPolicy(state_dim=X_tr.shape[1])
        mlp_bp.fit(X_tr, y_tr, verbose=True)
        p_va_mlp = mlp_bp.predict_proba(X_va)
        mlp_acc = (p_va_mlp.argmax(1) == y_va).mean()
        mlp_nll = -np.log(p_va_mlp[np.arange(len(y_va)), y_va] + 1e-12).mean()
        results["mlp"] = {"val_acc": float(mlp_acc), "val_nll": float(mlp_nll)}
        np.savez(os.path.join(output_dir, "mlp_proba_val.npz"),
                 proba=p_va_mlp, actions=y_va)
        torch.save(mlp_bp.net.state_dict(),
                   os.path.join(output_dir, "mlp_behavior.pt"))

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Behavior policy metrics: {results}")
    return results


if __name__ == "__main__":
    fit_behavior_policies()
