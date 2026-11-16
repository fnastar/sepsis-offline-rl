"""
train_iql_actor.py: Extract the IQL policy by advantage-weighted BC.

`train_iql.py` fits Q and V but neither defines a policy. Evaluating
softmax(Q/temperature) instead — as the earlier runs did — scores a method
that shares IQL's critic but is not IQL, so the negative result attributed to
IQL was really a result about Q-softmax extraction. This module trains the
actor Kostrikov et al. specify, saves it explicitly, and is the only thing
downstream evaluation should treat as the IQL policy.

Runs as a second stage over a finished IQL run: loads `d3qn_best.pt` and
`v_net_best.pt` from the run directory, trains the actor against the frozen
critic, and writes `actor_best.pt` alongside them.

    python -m src.train_iql_actor --run outputs/runs_90d_final/iql_seed42

Configuration is pre-specified, not swept: beta=3.0, weight clip 100.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from config.paths import PROCESSED_DIR
from config.rl import BATCH_SIZE, LEARNING_RATE, N_ACTIONS
from src.models.d3qn import D3QN
from src.models.iql import ValueNet, build_iql_actor, awbc_loss
from src.train import ReplayBuffer
from src.utils import set_seed

BETA = 3.0
WEIGHT_CLIP = 100.0


def train_actor(
    run_dir: str,
    seed: int = 42,
    n_epochs: int = 40,
    steps_per_epoch: int = 5000,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    beta: float = BETA,
    weight_clip: float = WEIGHT_CLIP,
    processed_dir: str | None = None,
    critic: str = "best",
    quiet: bool = False,
):
    """
    Train and save the IQL extraction policy for a finished IQL run.

    ``critic`` selects which saved critic the advantages come from. This is not
    cosmetic: `train_iql.py` keeps the epoch with the lowest validation loss,
    which on this data is epoch 1, an early checkpoint after 5,000 gradient
    updates. A critic that early makes Q(s,a) - V(s) close to zero across much
    of the state space. The AWBC weights then sit near exp(0) = 1 and the
    objective degenerates *toward* plain behavioural cloning, so the
    advantage-weighting contrast the method depends on is largely absent. The
    policy is still the output of the prespecified IQL extraction, and exact
    equality with behavioural cloning is not established. A mean weight near
    1.0 with nothing hitting the clip is the signature to watch.
    """
    set_seed(seed)
    log = (lambda *a, **k: None) if quiet else print
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pdir = processed_dir or PROCESSED_DIR

    with open(os.path.join(run_dir, "hyperparams.json")) as f:
        hp = json.load(f)
    state_cols = hp.get("state_cols")
    if state_cols is None:
        with open(os.path.join(pdir, "metadata.json")) as f:
            state_cols = json.load(f)["state_cols"]
    hidden = hp.get("hidden_dims", [256, 256])
    n_state = hp.get("n_state_features", len(state_cols))

    # Frozen critic from the completed IQL run
    q_target = D3QN(n_state, N_ACTIONS, hidden).to(device)
    suffix = "best" if critic == "best" else "final"
    q_target.load_state_dict(torch.load(
        os.path.join(run_dir, f"d3qn_{suffix}.pt"), map_location=device))
    q_target.eval().requires_grad_(False)

    v_net = ValueNet(n_state, hidden).to(device)
    v_net.load_state_dict(torch.load(
        os.path.join(run_dir, f"v_net_{suffix}.pt"), map_location=device))
    v_net.eval().requires_grad_(False)

    actor = build_iql_actor(n_state, N_ACTIONS, hidden).to(device)
    opt = torch.optim.Adam(actor.parameters(), lr=lr)

    train_buf = ReplayBuffer(os.path.join(pdir, "train.parquet"),
                             state_cols, device)
    val_buf = ReplayBuffer(os.path.join(pdir, "val.parquet"),
                           state_cols, device)
    log(f"Actor: state_dim={n_state} actions={N_ACTIONS} "
        f"train={train_buf.size:,} beta={beta} clip={weight_clip}")

    best_val = float("inf")
    history = []
    for epoch in range(1, n_epochs + 1):
        actor.train()
        t0 = time.time()
        tot, tot_w, tot_clip = 0.0, 0.0, 0.0
        for _ in range(steps_per_epoch):
            batch = train_buf.sample(batch_size)
            loss, mean_w, frac_clip = awbc_loss(
                actor, q_target, v_net, batch, beta=beta, weight_clip=weight_clip)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
            opt.step()
            tot += loss.item(); tot_w += mean_w; tot_clip += frac_clip

        actor.eval()
        with torch.no_grad():
            vb = val_buf.sample(min(10_000, val_buf.size))
            v_loss, _, _ = awbc_loss(actor, q_target, v_net, vb,
                                     beta=beta, weight_clip=weight_clip)
            v_loss = v_loss.item()
            probs = actor.probs(vb[0])
            greedy = probs.argmax(1)
            agree = (greedy == vb[1]).float().mean().item()
            ent = -(probs * probs.clamp_min(1e-12).log()).sum(1).mean().item()

        history.append({"epoch": epoch, "train_loss": tot / steps_per_epoch,
                        "val_loss": v_loss, "mean_weight": tot_w / steps_per_epoch,
                        "frac_clipped": tot_clip / steps_per_epoch,
                        "clinician_agreement": agree, "entropy": ent})
        log(f"  ep {epoch:3d}/{n_epochs} | tr={tot/steps_per_epoch:.4f} "
            f"val={v_loss:.4f} w̄={tot_w/steps_per_epoch:.2f} "
            f"clip={tot_clip/steps_per_epoch:.3f} agree={agree:.3f} "
            f"H={ent:.3f} {time.time()-t0:.1f}s")

        if v_loss < best_val:
            best_val = v_loss
            torch.save(actor.state_dict(), os.path.join(run_dir, _actor_file(critic)))

    tag = "" if critic == "best" else "_criticfinal"
    pd.DataFrame(history).to_csv(
        os.path.join(run_dir, f"actor_history{tag}.csv"), index=False)
    with open(os.path.join(run_dir, f"actor_hyperparams{tag}.json"), "w") as f:
        json.dump({"beta": beta, "weight_clip": weight_clip, "seed": seed,
                   "n_epochs": n_epochs, "steps_per_epoch": steps_per_epoch,
                   "batch_size": batch_size, "lr": lr,
                   "n_state_features": n_state, "hidden_dims": hidden,
                   "state_cols": state_cols,
                   "critic": critic,
                   "extraction": "advantage-weighted BC on Q(s,a_data) - V(s)"},
                  f, indent=2)
    # The displayed path has to be the one actually written. Hard-coding
    # actor_best.pt here made the final-critic extraction appear to overwrite
    # the early actor in the log, which it never did.
    log(f"Done. best_val={best_val:.4f} → "
        f"{os.path.join(run_dir, _actor_file(critic))}")
    return actor, pd.DataFrame(history)


def _actor_file(critic: str) -> str:
    return "actor_best.pt" if critic == "best" else "actor_criticfinal.pt"


def load_actor(run_dir: str, device=None, critic: str = "best"):
    """Load a saved IQL actor. Raises if the run has no extracted policy."""
    device = device or torch.device("cpu")
    path = os.path.join(run_dir, _actor_file(critic))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing — this IQL run has no extracted policy. "
            "Evaluating softmax(Q) here would not be IQL.")
    tag = "" if critic == "best" else "_criticfinal"
    with open(os.path.join(run_dir, f"actor_hyperparams{tag}.json")) as f:
        hp = json.load(f)
    actor = build_iql_actor(hp["n_state_features"], N_ACTIONS, hp["hidden_dims"])
    actor.load_state_dict(torch.load(path, map_location=device))
    return actor.to(device).eval()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--processed-dir", default=None)
    a = p.parse_args()
    train_actor(a.run, seed=a.seed, n_epochs=a.epochs,
                processed_dir=a.processed_dir)
