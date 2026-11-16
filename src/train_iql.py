"""
train_iql.py: Offline training loop for IQL on sepsis MDP transitions.

Mirrors src/train.py conventions exactly (epochs, buffers, logging, artefact
names) so IQL runs are directly comparable with the D3QN/CQL matrix and reuse
the same downstream evaluation (FQE, data_support, subgroup_analysis). The
Q-network checkpoint is saved as `d3qn_best.pt` / `d3qn_final.pt` because it
IS a D3QN-class network — only the training losses differ (see models/iql.py).

Usage:
    python -m src.train_iql                    # seed 42, expectile 0.7
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config.paths import PROCESSED_DIR
from config.rl import (
    BATCH_SIZE, GAMMA, HIDDEN_DIMS, LEARNING_RATE, N_EPOCHS, N_ACTIONS,
)
from src.models.iql import build_iql, iql_losses
from src.train import ReplayBuffer
from src.utils import ensure_dir, set_seed

TARGET_UPDATE_FREQ = 100   # hard update every N gradient steps (as train.py)
GRAD_CLIP = 10.0
EXPECTILE = 0.7            # IQL paper default


def train_iql(
    output_dir: str = "outputs/runs/iql_seed42",
    seed: int = 42,
    expectile: float = EXPECTILE,
    n_epochs: int = N_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    gamma: float = GAMMA,
    hidden_dims: list = HIDDEN_DIMS,
    steps_per_epoch: int = 5000,
    quiet: bool = False,
):
    """Train discrete IQL; save checkpoint/history/hyperparams like train.py."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = (lambda *a, **k: None) if quiet else print

    with open(os.path.join(PROCESSED_DIR, "metadata.json")) as f:
        state_cols = json.load(f)["state_cols"]

    train_buf = ReplayBuffer(os.path.join(PROCESSED_DIR, "train.parquet"),
                             state_cols, device)
    val_buf = ReplayBuffer(os.path.join(PROCESSED_DIR, "val.parquet"),
                           state_cols, device)
    n_state = train_buf.n_features
    log(f"Device={device} | state_dim={n_state} | actions={N_ACTIONS} | "
        f"train={train_buf.size:,} val={val_buf.size:,}")

    q_online, q_target, v_net = build_iql(n_state, N_ACTIONS, hidden_dims)
    q_online, q_target, v_net = (q_online.to(device), q_target.to(device),
                                 v_net.to(device))
    opt_q = torch.optim.Adam(q_online.parameters(), lr=lr)
    opt_v = torch.optim.Adam(v_net.parameters(), lr=lr)

    ensure_dir(output_dir)
    hyperparams = {
        "seed": seed, "n_epochs": n_epochs, "batch_size": batch_size, "lr": lr,
        "gamma": gamma, "hidden_dims": hidden_dims,
        "steps_per_epoch": steps_per_epoch,
        "expectile": expectile,
        "n_state_features": n_state,
        "state_cols": train_buf.state_cols,
        "target_update_freq": TARGET_UPDATE_FREQ, "grad_clip": GRAD_CLIP,
        "model": "IQL",
    }
    with open(os.path.join(output_dir, "hyperparams.json"), "w") as f:
        json.dump(hyperparams, f, indent=2)

    best_val_loss = float("inf")
    history = []
    total_steps = 0
    log(f"Training IQL {n_epochs}×{steps_per_epoch} steps (batch={batch_size}, "
        f"γ={gamma}, lr={lr}, expectile={expectile})...")

    for epoch in range(1, n_epochs + 1):
        q_online.train(); v_net.train()
        t0 = time.time()
        epoch_q_loss = epoch_v_loss = 0.0

        for _ in range(steps_per_epoch):
            batch = train_buf.sample(batch_size)
            v_loss, q_loss = iql_losses(q_online, q_target, v_net, batch,
                                        gamma, expectile)
            opt_v.zero_grad()
            v_loss.backward()
            nn.utils.clip_grad_norm_(v_net.parameters(), GRAD_CLIP)
            opt_v.step()

            opt_q.zero_grad()
            q_loss.backward()
            nn.utils.clip_grad_norm_(q_online.parameters(), GRAD_CLIP)
            opt_q.step()

            epoch_q_loss += q_loss.item()
            epoch_v_loss += v_loss.item()
            total_steps += 1
            if total_steps % TARGET_UPDATE_FREQ == 0:
                q_target.load_state_dict(q_online.state_dict())

        # Validation (same diagnostics as train.py)
        q_online.eval(); v_net.eval()
        with torch.no_grad():
            vb = val_buf.sample(min(10_000, val_buf.size))
            val_v_loss, val_q_loss = iql_losses(q_online, q_target, v_net, vb,
                                                gamma, expectile)
            val_loss = (val_v_loss + val_q_loss).item()
            q_all = q_online(vb[0])
            q_max = q_all.max(dim=1).values
            mean_q = q_max.mean().item()
            max_q = q_max.max().item()
            actions = q_all.argmax(dim=1)
            vaso_frac = (actions >= 20).float().mean().item()
            action0_frac = (actions == 0).float().mean().item()

        avg_q = epoch_q_loss / steps_per_epoch
        avg_v = epoch_v_loss / steps_per_epoch
        elapsed = time.time() - t0
        history.append({
            "epoch": epoch, "train_loss": avg_q + avg_v,
            "train_q_loss": avg_q, "train_v_loss": avg_v,
            "val_loss": val_loss, "mean_q": mean_q,
        })
        log(f"  ep {epoch:3d}/{n_epochs} | q={avg_q:.4f} v={avg_v:.4f} "
            f"val={val_loss:.4f} Q={mean_q:.3f} maxQ={max_q:.1f} "
            f"a0%={action0_frac:.2f} vaso%={vaso_frac:.2f} {elapsed:.1f}s")

        if not np.isfinite(val_loss) or abs(mean_q) > 1e3:
            log(f"DIVERGED at epoch {epoch} (val={val_loss}, Q={mean_q})")
            with open(os.path.join(output_dir, "DIVERGED"), "w") as f:
                f.write(f"epoch={epoch} val_loss={val_loss} mean_q={mean_q}\n")
            break

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(q_online.state_dict(),
                       os.path.join(output_dir, "d3qn_best.pt"))
            torch.save(v_net.state_dict(),
                       os.path.join(output_dir, "v_net_best.pt"))

    torch.save(q_online.state_dict(), os.path.join(output_dir, "d3qn_final.pt"))
    torch.save(v_net.state_dict(), os.path.join(output_dir, "v_net_final.pt"))
    pd.DataFrame(history).to_csv(
        os.path.join(output_dir, "training_history.csv"), index=False)
    log(f"Done. best_val={best_val_loss:.4f} → {output_dir}")
    return q_online, pd.DataFrame(history)


if __name__ == "__main__":
    train_iql()
