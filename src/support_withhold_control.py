"""
Measure data support for the always-withhold control (action 0).

Uses the same behaviour model, 80,000 training reference decisions, 15,000
test query decisions, 50 neighbours and sampling seed as data_support.
Samples are drawn in the same order to keep comparisons aligned.

Writes <OUTPUT_BASE>/data_support/support_metrics_withhold.json.
"""
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

from config.paths import PROCESSED_DIR, OUTPUT_BASE, BEHAVIOR_POLICY_DIR
from src.models.behavior_policy import MLPBehaviorPolicy
from src.data_support import (BP_WEIGHTS, N_REF, N_QRY, K_NEIGHBORS, SEED,
                              UNSUPPORTED_THRESH, _propensity_stats,
                              _knn_support_stats, _vaso_rate)

WITHHOLD_ACTION = 0
OUTPUT_DIR = os.path.join(OUTPUT_BASE, "data_support")


def run(split: str = "test") -> dict:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    with open(os.path.join(PROCESSED_DIR, "metadata.json")) as f:
        state_cols = json.load(f)["state_cols"]

    eval_df = pd.read_parquet(os.path.join(PROCESSED_DIR, f"{split}.parquet"))
    s_eval = eval_df[state_cols].values.astype(np.float32)
    a_obs = eval_df["action"].values.astype(int)
    a_star = np.full(len(s_eval), WITHHOLD_ACTION, dtype=int)

    bp = MLPBehaviorPolicy(state_dim=len(state_cols))
    bp.net.load_state_dict(torch.load(BP_WEIGHTS, map_location=bp.device,
                                      weights_only=True))
    pi_beta = bp.predict_proba(s_eval)

    # Same two draws, same order, as data_support.run, so the reference and
    # query sets match the ones behind support_metrics.json exactly.
    train_df = pd.read_parquet(os.path.join(PROCESSED_DIR, "train.parquet"))
    s_tr = train_df[state_cols].values.astype(np.float32)
    a_tr = train_df["action"].values.astype(int)
    ref_idx = rng.choice(len(s_tr), size=min(N_REF, len(s_tr)), replace=False)
    X_ref, a_ref = s_tr[ref_idx], a_tr[ref_idx]
    mu, sigma = X_ref.mean(axis=0), X_ref.std(axis=0) + 1e-8
    X_ref_z = (X_ref - mu) / sigma
    qry_idx = rng.choice(len(s_eval), size=min(N_QRY, len(s_eval)),
                         replace=False)
    X_qry_z = (s_eval[qry_idx] - mu) / sigma

    nn = NearestNeighbors(n_neighbors=K_NEIGHBORS, algorithm="auto")
    nn.fit(X_ref_z)
    _, neigh = nn.kneighbors(X_qry_z)
    neigh_actions = a_ref[neigh]

    row = {"vaso_recommendation_rate": _vaso_rate(a_star)}
    row.update(_propensity_stats(pi_beta, a_star))
    row.update(_knn_support_stats(neigh_actions, a_star[qry_idx]))
    row["n_seeds"] = 1

    out = {
        "split": split,
        "n_eval_points": int(len(s_eval)),
        "knn": {"n_ref": int(len(X_ref_z)), "n_query": int(len(X_qry_z)),
                "k": K_NEIGHBORS, "seed": SEED},
        "unsupported_threshold": UNSUPPORTED_THRESH,
        "clinician_untreated_rate": float((a_obs == WITHHOLD_ACTION).mean()),
        "policies": {"Withhold treatment": row},
    }
    path = os.path.join(OUTPUT_DIR, "support_metrics_withhold.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Withhold control (action {WITHHOLD_ACTION}), {split} split")
    print(f"  clinician untreated rate : "
          f"{out['clinician_untreated_rate']*100:5.2f}%")
    print(f"  mean pi_beta(a*|s)       : {row['mean_propensity']:.4f}")
    print(f"  % unsupported (<{UNSUPPORTED_THRESH})   : "
          f"{row['pct_unsupported']:5.2f}%")
    print(f"  mean k-NN local support  : {row['mean_local_support']:.4f}")
    print(f"  % zero k-NN support      : {row['pct_zero_support']:5.2f}%")
    print(f"Saved {path}")
    return out


if __name__ == "__main__":
    run()
