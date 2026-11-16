"""
baselines.py: Reference baselines for OPE comparison.

- BehaviorCloning   : supervised classifier predicting clinician action.
                      Acts as a learned policy that mimics the clinician
                      → its OPE value should match clinician's value.
- FQI (Fitted Q-Iteration): Komorowski-2018-style tabular/MLP Q-iteration,
                            offline RL without target networks.
                            We use an MLP regressor for tractable scaling.
- RandomPolicy      : uniform over the observed action set (sanity floor).
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config.paths import PROCESSED_DIR
from config.rl import GAMMA, N_ACTIONS
from src.models.behavior_policy import MLPBehaviorPolicy
from src.utils import ensure_dir


# ─────────────────────────────────────────────────────────────────────────────
# BEHAVIOR CLONING
# ─────────────────────────────────────────────────────────────────────────────

def train_behavior_cloning(output_dir: str = "outputs/runs/bc"):
    """Train an MLP classifier on (state, action) — same as MLPBehaviorPolicy."""
    ensure_dir(output_dir)
    with open(os.path.join(PROCESSED_DIR, "metadata.json")) as f:
        meta = json.load(f)
    state_cols = meta["state_cols"]

    train_df = pd.read_parquet(os.path.join(PROCESSED_DIR, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(PROCESSED_DIR, "val.parquet"))

    X_tr = train_df[state_cols].values.astype(np.float32)
    y_tr = train_df["action"].values.astype(int)
    X_va = val_df[state_cols].values.astype(np.float32)
    y_va = val_df["action"].values.astype(int)

    bp = MLPBehaviorPolicy(state_dim=X_tr.shape[1], epochs=20)
    print("Training BC (MLP, 20 epochs)...")
    t0 = time.time()
    bp.fit(X_tr, y_tr, verbose=True)
    elapsed = time.time() - t0

    p_va = bp.predict_proba(X_va)
    val_acc = float((p_va.argmax(1) == y_va).mean())
    val_nll = float(-np.log(p_va[np.arange(len(y_va)), y_va] + 1e-12).mean())

    torch.save(bp.net.state_dict(), os.path.join(output_dir, "bc_final.pt"))
    np.savez(os.path.join(output_dir, "bc_proba_val.npz"),
             proba=p_va, actions=y_va)
    metrics = {"val_acc": val_acc, "val_nll": val_nll, "minutes": elapsed / 60}
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"BC done: acc={val_acc:.3f} nll={val_nll:.3f}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# FITTED Q-ITERATION (FQI) — neural-net regressor variant
# ─────────────────────────────────────────────────────────────────────────────

class FQINet(nn.Module):
    """Q(s,a) regressor — outputs all N_ACTIONS Q-values in one forward pass."""
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def train_fqi(
    output_dir: str = "outputs/runs/fqi",
    n_iterations: int = 30,
    inner_epochs: int = 3,
    batch_size: int = 1024,
    lr: float = 3e-4,
    gamma: float = GAMMA,
):
    """
    Fitted Q-Iteration: at each outer iteration, regress Q(s,a) toward
    r + γ · max_a' Q_old(s', a'). The "old" Q is held fixed during the
    inner regression — analogous to a target network.
    """
    ensure_dir(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(PROCESSED_DIR, "metadata.json")) as f:
        meta = json.load(f)
    state_cols = meta["state_cols"]
    next_cols = [f"next_{c}" for c in state_cols]

    train_df = pd.read_parquet(os.path.join(PROCESSED_DIR, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(PROCESSED_DIR, "val.parquet"))

    s = torch.tensor(train_df[state_cols].values, dtype=torch.float32, device=device)
    sp = torch.tensor(train_df[next_cols].values, dtype=torch.float32, device=device)
    a = torch.tensor(train_df["action"].values, dtype=torch.long, device=device)
    r = torch.tensor(train_df["reward"].values, dtype=torch.float32, device=device)
    d = torch.tensor(train_df["done"].values, dtype=torch.float32, device=device)
    s_va = torch.tensor(val_df[state_cols].values, dtype=torch.float32, device=device)
    sp_va = torch.tensor(val_df[next_cols].values, dtype=torch.float32, device=device)
    a_va = torch.tensor(val_df["action"].values, dtype=torch.long, device=device)
    r_va = torch.tensor(val_df["reward"].values, dtype=torch.float32, device=device)
    d_va = torch.tensor(val_df["done"].values, dtype=torch.float32, device=device)

    state_dim = len(state_cols)
    q_net = FQINet(state_dim, N_ACTIONS).to(device)
    optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)
    history = []
    n = len(s)

    print(f"FQI: {n_iterations} iters × {inner_epochs} epochs (batch={batch_size})")
    t0 = time.time()
    for it in range(1, n_iterations + 1):
        # Build target with frozen Q
        with torch.no_grad():
            q_next_max = q_net(sp).max(dim=1).values
            target = r + gamma * q_next_max * (1.0 - d)

        # Inner regression
        for _ in range(inner_epochs):
            perm = torch.randperm(n, device=device)
            running = 0.0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                q_pred = q_net(s[idx]).gather(1, a[idx].unsqueeze(1)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(q_pred, target[idx])
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                running += loss.item() * len(idx)

        # Validation
        with torch.no_grad():
            q_next_va = q_net(sp_va).max(dim=1).values
            tgt_va = r_va + gamma * q_next_va * (1.0 - d_va)
            q_pred_va = q_net(s_va).gather(1, a_va.unsqueeze(1)).squeeze(1)
            val_loss = nn.functional.smooth_l1_loss(q_pred_va, tgt_va).item()
            mean_q = q_net(s_va[:10000]).max(dim=1).values.mean().item()
        history.append({"iter": it, "train_loss": running / n,
                        "val_loss": val_loss, "mean_q": mean_q})
        print(f"  iter {it:2d}/{n_iterations} | tr={running/n:.4f} "
              f"val={val_loss:.4f} Q={mean_q:.3f}")

        if not np.isfinite(val_loss):
            print(f"FQI diverged at iter {it}")
            with open(os.path.join(output_dir, "DIVERGED"), "w") as f:
                f.write(f"iter={it}\n")
            break

    elapsed = time.time() - t0
    torch.save(q_net.state_dict(), os.path.join(output_dir, "fqi_final.pt"))
    pd.DataFrame(history).to_csv(
        os.path.join(output_dir, "training_history.csv"), index=False)

    hp = {
        "n_iterations": n_iterations, "inner_epochs": inner_epochs,
        "batch_size": batch_size, "lr": lr, "gamma": gamma,
        "n_state_features": state_dim, "minutes": elapsed / 60,
    }
    with open(os.path.join(output_dir, "hyperparams.json"), "w") as f:
        json.dump(hp, f, indent=2)
    print(f"FQI done in {elapsed/60:.1f} min")
    return q_net, history


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("bc", "all"):
        train_behavior_cloning()
    if which in ("fqi", "all"):
        train_fqi()
