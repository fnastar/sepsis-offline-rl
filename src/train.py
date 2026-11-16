"""
train.py: Offline training loop for D3QN on sepsis MDP transitions.

Double DQN update:
    a* = argmax_a Q_online(s', a)
    y  = r + γ · Q_target(s', a*) · (1 - done)
    L  = HuberLoss(Q_online(s, a), y)
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
try:
    import wandb
except ImportError:
    wandb = None

from config.paths import PROCESSED_DIR
from config.rl import (
    BATCH_SIZE, GAMMA, HIDDEN_DIMS, LEARNING_RATE, N_EPOCHS, N_ACTIONS,
)
from src.models.d3qn import build_d3qn
from src.utils import ensure_dir, set_seed

TARGET_UPDATE_FREQ = 100   # hard update every N gradient steps
GRAD_CLIP = 10.0


# ─────────────────────────────────────────────────────────────────────────────
# REPLAY BUFFER (offline — fixed dataset)
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Loads offline transitions into CPU tensors; samples random mini-batches."""

    def __init__(
        self,
        parquet_path: str,
        state_cols: list,
        device: torch.device,
        drop_cols: list | None = None,
        reward_scale_nonterminal: float = 1.0,
    ):
        df = pd.read_parquet(parquet_path)
        # Allow ablations that exclude certain features (e.g. demographics)
        cols = [c for c in state_cols if c not in (drop_cols or [])]
        next_cols = [f"next_{c}" for c in cols]

        # Reward scaling for SOFA-coef ablation: rescales non-terminal rewards
        # while leaving terminal ±15 intact. Default 1.0 = no change.
        rewards = df["reward"].values.astype("float32")
        dones = df["done"].values.astype("float32")
        if reward_scale_nonterminal != 1.0:
            rewards = rewards * (dones + (1.0 - dones) * reward_scale_nonterminal)

        self.state_cols  = cols
        self.states      = torch.tensor(df[cols].values,      dtype=torch.float32)
        self.next_states = torch.tensor(df[next_cols].values, dtype=torch.float32)
        self.actions     = torch.tensor(df["action"].values,  dtype=torch.long)
        self.rewards     = torch.tensor(rewards,              dtype=torch.float32)
        self.dones       = torch.tensor(dones,                dtype=torch.float32)
        self.device      = device
        self.size        = len(df)
        self.n_features  = len(cols)

    def sample(self, batch_size: int) -> tuple:
        idx = torch.randint(0, self.size, (batch_size,))
        return (
            self.states[idx].to(self.device),
            self.actions[idx].to(self.device),
            self.rewards[idx].to(self.device),
            self.next_states[idx].to(self.device),
            self.dones[idx].to(self.device),
        )


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def compute_loss(online, target, batch, gamma, cql_alpha: float = 0.0):
    """
    Double-DQN Bellman loss, optionally with the discrete CQL(H) regulariser.

    When ``cql_alpha > 0`` we add the Conservative Q-Learning penalty
    (Kumar et al. 2020, eq. 4, discrete form):

        L_CQL = logsumexp_a Q(s, a) − Q(s, a_dataset)

    The first term pushes down Q-values for *all* actions (especially the
    out-of-distribution ones the max operator would otherwise inflate); the
    second term pulls up the value of the action the clinician actually took.
    α scales how conservative the policy becomes.

    Returns
    -------
    (total_loss, bellman_loss, cql_penalty) — the latter two as floats for
    logging. When cql_alpha == 0 the penalty is 0.0 and total == bellman.
    """
    states, actions, rewards, next_states, dones = batch
    q_all = online(states)                                       # (B, A)
    q_values = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)
    with torch.no_grad():
        next_actions = online(next_states).argmax(dim=1)
        next_q = target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        targets = rewards + gamma * next_q * (1.0 - dones)
    bellman_loss = nn.functional.smooth_l1_loss(q_values, targets)

    if cql_alpha > 0.0:
        # logsumexp over actions − Q(s, a_dataset), averaged over the batch
        logsumexp_q = torch.logsumexp(q_all, dim=1)              # (B,)
        cql_penalty = (logsumexp_q - q_values).mean()
        total = bellman_loss + cql_alpha * cql_penalty
        return total, bellman_loss.item(), cql_penalty.item()
    return bellman_loss, bellman_loss.item(), 0.0


def train(
    output_dir: str = "outputs/checkpoints",
    seed: int = 42,
    n_epochs: int = N_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    gamma: float = GAMMA,
    hidden_dims: list = HIDDEN_DIMS,
    steps_per_epoch: int = 5000,
    drop_features: list | None = None,
    reward_scale_nonterminal: float = 1.0,
    cql_alpha: float = 0.0,
    train_path: str | None = None,
    val_path: str | None = None,
    quiet: bool = False,
    use_wandb: bool = False,
    run_name: str | None = None,
):
    """
    Train D3QN (or CQL when ``cql_alpha > 0``) on offline sepsis transitions.

    Saves model checkpoint, training history CSV, and hyperparameters JSON
    to ``output_dir``. Returns (online_net, history_df).
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = (lambda *a, **k: None) if quiet else print

    # --- Metadata + buffers ---
    with open(os.path.join(PROCESSED_DIR, "metadata.json")) as f:
        meta = json.load(f)
    state_cols = meta["state_cols"]

    train_buf = ReplayBuffer(
        train_path or os.path.join(PROCESSED_DIR, "train.parquet"),
        state_cols, device, drop_cols=drop_features,
        reward_scale_nonterminal=reward_scale_nonterminal,
    )
    val_buf = ReplayBuffer(
        val_path or os.path.join(PROCESSED_DIR, "val.parquet"),
        state_cols, device, drop_cols=drop_features,
        reward_scale_nonterminal=reward_scale_nonterminal,
    )
    n_state = train_buf.n_features
    log(f"Device={device} | state_dim={n_state} | actions={N_ACTIONS} | "
        f"train={train_buf.size:,} val={val_buf.size:,}")

    # --- Networks ---
    online, target = build_d3qn(n_state, N_ACTIONS, hidden_dims)
    online = online.to(device); target = target.to(device)
    optimizer = torch.optim.Adam(online.parameters(), lr=lr)

    # --- Output dir + hyperparam dump ---
    ensure_dir(output_dir)
    hyperparams = {
        "seed": seed, "n_epochs": n_epochs, "batch_size": batch_size, "lr": lr,
        "gamma": gamma, "hidden_dims": hidden_dims,
        "steps_per_epoch": steps_per_epoch,
        "drop_features": drop_features or [],
        "reward_scale_nonterminal": reward_scale_nonterminal,
        "cql_alpha": cql_alpha,
        "n_state_features": n_state,
        "state_cols": train_buf.state_cols,
        "target_update_freq": TARGET_UPDATE_FREQ, "grad_clip": GRAD_CLIP,
        "model": "CQL" if cql_alpha > 0 else "D3QN",
    }
    with open(os.path.join(output_dir, "hyperparams.json"), "w") as f:
        json.dump(hyperparams, f, indent=2)

    if use_wandb and wandb is None:
        raise ImportError("Install wandb to enable experiment tracking")
    if use_wandb:
        default_name = (f"cql_a{cql_alpha}_seed{seed}" if cql_alpha > 0
                        else f"d3qn_seed{seed}")
        wandb.init(
            project="sepsisrl",
            name=run_name or default_name,
            config=hyperparams,
            dir=output_dir,
        )

    # --- Training ---
    best_val_loss = float("inf")
    history = []
    total_steps = 0
    log(f"Training {n_epochs}×{steps_per_epoch} steps (batch={batch_size}, "
        f"γ={gamma}, lr={lr})...")

    for epoch in range(1, n_epochs + 1):
        online.train()
        t0 = time.time()
        epoch_loss = 0.0

        grad_norms = []
        epoch_cql = 0.0
        for _ in range(steps_per_epoch):
            batch = train_buf.sample(batch_size)
            loss, _bellman, cql_pen = compute_loss(
                online, target, batch, gamma, cql_alpha=cql_alpha)
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(online.parameters(), GRAD_CLIP)
            grad_norms.append(grad_norm.item())
            optimizer.step()
            epoch_loss += loss.item()
            epoch_cql += cql_pen
            total_steps += 1
            if total_steps % TARGET_UPDATE_FREQ == 0:
                target.load_state_dict(online.state_dict())

        # Validation
        online.eval()
        with torch.no_grad():
            vb = val_buf.sample(min(10_000, val_buf.size))
            val_loss = compute_loss(online, target, vb, gamma, cql_alpha=cql_alpha)[0].item()
            q_all = online(vb[0])                          # (B, A)
            q_max = q_all.max(dim=1).values
            mean_q  = q_max.mean().item()
            max_q   = q_max.max().item()
            std_q   = q_all.std().item()
            actions = q_all.argmax(dim=1)
            # vasopressor bin is action // 5; fluid bin is action % 5
            vaso_frac = (actions >= 20).float().mean().item()  # top vaso level
            action_counts = torch.bincount(actions, minlength=N_ACTIONS).float()
            action_entropy = -(action_counts / action_counts.sum() + 1e-8).log().sum().item()

        avg_train_loss = epoch_loss / steps_per_epoch
        avg_cql_pen    = epoch_cql / steps_per_epoch
        avg_grad_norm  = float(np.mean(grad_norms))
        elapsed = time.time() - t0
        history.append({
            "epoch": epoch, "train_loss": avg_train_loss,
            "val_loss": val_loss, "mean_q": mean_q,
            "cql_penalty": avg_cql_pen,
        })
        cql_str = f"cql={avg_cql_pen:.3f} " if cql_alpha > 0 else ""
        log(f"  ep {epoch:3d}/{n_epochs} | tr={avg_train_loss:.4f} "
            f"val={val_loss:.4f} Q={mean_q:.3f} maxQ={max_q:.1f} "
            f"{cql_str}grad={avg_grad_norm:.2f} vaso%={vaso_frac:.2f} {elapsed:.1f}s")
        if use_wandb:
            wandb.log({
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "mean_max_q": mean_q,
                "max_q": max_q,
                "std_q": std_q,
                "grad_norm": avg_grad_norm,
                "vaso_top_frac": vaso_frac,
                "action_entropy": action_entropy,
                "cql_penalty": avg_cql_pen,
            }, step=epoch)

        # Bail out if diverged
        if not np.isfinite(val_loss) or abs(mean_q) > 1e3:
            log(f"DIVERGED at epoch {epoch} (val={val_loss}, Q={mean_q})")
            with open(os.path.join(output_dir, "DIVERGED"), "w") as f:
                f.write(f"epoch={epoch} val_loss={val_loss} mean_q={mean_q}\n")
            break

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(online.state_dict(),
                       os.path.join(output_dir, "d3qn_best.pt"))

    torch.save(online.state_dict(), os.path.join(output_dir, "d3qn_final.pt"))
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(output_dir, "training_history.csv"), index=False)
    if use_wandb:
        wandb.log({"best_val_loss": best_val_loss})
        wandb.finish()
    log(f"Done. best_val={best_val_loss:.4f} → {output_dir}")
    return online, hist_df


if __name__ == "__main__":
    train()
