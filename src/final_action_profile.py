"""
Summarise treatment recommendations by method, seed and checkpoint.

Reports untreated decisions, fluid and vasopressor action distributions,
and agreement with observed clinician choices. Uses forward passes through
saved policies; no fitting is required.

    python -m src.final_action_profile
"""
import json
import os

import numpy as np
import pandas as pd
import torch

from config.paths import OUTPUT_BASE, PROCESSED_DIR
from config.rl import N_ACTIONS, N_IV_LEVELS
from src.final_campaign import actor_is_complete, is_complete
from src.utils import ensure_dir

SEEDS = (42, 43, 44, 45, 46)
CQL_ALPHA = 2.0
HIDDEN = [256, 256]
METHODS = {"cql": f"cql_a{CQL_ALPHA}_seed", "d3qn": "d3qn_seed", "iql": "iql_seed"}


def _greedy(policy, states: np.ndarray, device) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, len(states), 8192):
            sb = torch.tensor(states[i:i + 8192], dtype=torch.float32,
                              device=device)
            out.append(policy(sb).argmax(dim=1).cpu().numpy())
    return np.concatenate(out)


def _profile(actions: np.ndarray, observed: np.ndarray) -> dict:
    vaso = actions // N_IV_LEVELS
    iv = actions % N_IV_LEVELS
    n = len(actions)
    return {
        "n_decisions": int(n),
        "untreated_rate": round(float((actions == 0).mean()), 4),
        "any_vasopressor_rate": round(float((vaso > 0).mean()), 4),
        "any_fluid_rate": round(float((iv > 0).mean()), 4),
        "vaso_levels": {str(k): round(float((vaso == k).mean()), 4)
                        for k in range(5)},
        "iv_levels": {str(k): round(float((iv == k).mean()), 4)
                      for k in range(5)},
        "distinct_actions": int(len(np.unique(actions))),
        "agreement_with_clinician": round(float((actions == observed).mean()), 4),
        "vaso_level_agreement": round(
            float((vaso == observed // N_IV_LEVELS).mean()), 4),
    }


def run(split: str = "test") -> dict:
    device = torch.device("cpu")
    from src.final_fqe import (actor_greedy_policy, actor_greedy_policy_critic,
                               greedy_q_policy)
    with open(os.path.join(PROCESSED_DIR, "metadata.json")) as f:
        meta = json.load(f)
    cols, dim = meta["state_cols"], meta["n_state_features"]

    df = pd.read_parquet(os.path.join(PROCESSED_DIR, f"{split}.parquet"),
                         columns=cols + ["action"])
    states = df[cols].values.astype(np.float32)
    observed = df["action"].values.astype(int)
    del df

    out = {"split": split, "clinician": _profile(observed, observed),
           "policies": {}, "by_method": {}}

    for kind, pat in METHODS.items():
        for ckpt in ("best", "final"):
            rows = []
            for seed in SEEDS:
                run_dir = os.path.join(OUTPUT_BASE, f"{pat}{seed}")
                # A checkpoint appears after the first epoch, so its presence
                # is not evidence the run finished. Profiling a half-trained
                # policy would put a number in a table that describes nothing.
                if not is_complete(run_dir):
                    continue
                if kind == "iql" and not actor_is_complete(
                        run_dir, "best" if ckpt == "best" else "final"):
                    continue
                try:
                    if kind == "iql":
                        pol = (actor_greedy_policy(run_dir, device) if ckpt == "best"
                               else actor_greedy_policy_critic(run_dir, device,
                                                               "final"))
                    else:
                        ck = "d3qn_best.pt" if ckpt == "best" else "d3qn_final.pt"
                        if not os.path.exists(os.path.join(run_dir, ck)):
                            continue
                        pol = greedy_q_policy(os.path.join(run_dir, ck), dim,
                                              HIDDEN, device)
                except (FileNotFoundError, OSError):
                    continue
                p = _profile(_greedy(pol, states, device), observed)
                out["policies"][f"{kind}_seed{seed}_{ckpt}"] = p
                rows.append(p)
            if rows:
                keys = ("untreated_rate", "any_vasopressor_rate",
                        "any_fluid_rate", "agreement_with_clinician",
                        "vaso_level_agreement", "distinct_actions")
                agg = {k: round(float(np.mean([r[k] for r in rows])), 4)
                       for k in keys}
                agg.update({f"{k}_sd": round(float(np.std([r[k] for r in rows],
                                                          ddof=1)), 4)
                            for k in keys} if len(rows) > 1 else {})
                agg["n_seeds"] = len(rows)
                out["by_method"][f"{kind}_{ckpt}"] = agg

    # Completeness travels with the table and is enforced. A profile assembled
    # from three seeds looks exactly like one assembled from five.
    expected = {f"{k}_{c}": len(SEEDS) for k in METHODS for c in ("best", "final")}
    found = {k: out["by_method"].get(k, {}).get("n_seeds", 0) for k in expected}
    short = {k: (found[k], expected[k]) for k in expected if found[k] < expected[k]}
    out["seeds_expected"] = expected
    out["seeds_found"] = found
    out["complete"] = not short

    ensure_dir(OUTPUT_BASE)
    p = os.path.join(OUTPUT_BASE, f"action_profile_{split}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)

    c = out["clinician"]
    print(f"\nAction profiles, {split} split, {c['n_decisions']:,} decisions\n")
    print(f"{'policy':<16}{'untreated':>11}{'any vaso':>10}{'any fluid':>11}"
          f"{'agree':>8}{'seeds':>7}")
    print(f"{'clinician':<16}{c['untreated_rate']:>11.3f}"
          f"{c['any_vasopressor_rate']:>10.3f}{c['any_fluid_rate']:>11.3f}"
          f"{1.0:>8.3f}{'-':>7}")
    for name, a in out["by_method"].items():
        print(f"{name:<16}{a['untreated_rate']:>11.3f}"
              f"{a['any_vasopressor_rate']:>10.3f}{a['any_fluid_rate']:>11.3f}"
              f"{a['agreement_with_clinician']:>8.3f}{a['n_seeds']:>7}")
    print(f"\nsaved -> {p}")
    if short:
        # Written first so the partial table can be inspected, then raised so
        # the step is recorded as failed and the artefact is never mistaken
        # for the finished one.
        raise AssertionError(
            f"action profile is incomplete: {short} as found against expected. "
            f"A profile over fewer runs than the protocol specifies must not "
            f"read as final.")
    return out


if __name__ == "__main__":
    run("test")
