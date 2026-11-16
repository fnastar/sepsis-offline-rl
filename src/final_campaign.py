"""
Run the final Sepsis-3 training and evaluation campaign.

Stages, in order:
  pilot     CQL and D3QN seed 42, then reference-policy FQE
  iql       IQL seed 42, actor extraction and FQE
  seeds     remaining seeds 43-46 for all three methods
  fqe       evaluate the completed policy runs
  iqlfinal  extract and evaluate IQL actors using the final critic

The iql and seeds stages require a reviewed dataset, a SIGNOFF file and
SEPSIS_SEEDS_OK=1. Completed pilot runs are reused.
See docs/reproduction.md for configuration and commands.
"""
import argparse
import json
import os
import time

import pandas as pd
import torch

from config.paths import OUTPUT_BASE, PROCESSED_DIR, PROJECT_ROOT
from config.rl import N_ACTIONS
from src.final_fqe import (
    mlp_classifier_policy,
    actor_greedy_policy, constant_policy, greedy_q_policy, run_fqe,
)
from src.utils import ensure_dir

# What each column of the checkpoint comparison actually contains. The two
# arms are not the same comparison across methods: for the Q-network policies
# they vary the Q checkpoint, and for IQL they vary the critic the actor was
# extracted against while the actor itself is validation-selected in both. Any
# table putting them under one pair of headings implies a parallel that does
# not exist, so the definitions travel with every artefact that reports them.
CHECKPOINT_DEFINITIONS = {
    "cql_early": "greedy policy from the epoch-1 Q checkpoint, "
                 "after 5,000 gradient updates",
    "cql_final": "greedy policy from the epoch-40 Q checkpoint",
    "d3qn_early": "greedy policy from the epoch-1 Q checkpoint, "
                  "after 5,000 gradient updates",
    "d3qn_final": "greedy policy from the epoch-40 Q checkpoint",
    "iql_early": "validation-selected actor trained against the epoch-1 critic",
    "iql_final": "validation-selected actor trained against the epoch-40 critic",
    "note": "IQL has no final-actor-epoch checkpoint. Its actor is trained for "
            "the full 40 epochs and selected by actor validation loss in both "
            "arms, so the IQL columns vary the critic rather than the actor.",
}

N_EPOCHS_FINAL = 40
SEEDS = (42, 43, 44, 45, 46)
CQL_ALPHA = 2.0
IQL_EXPECTILE = 0.7
# Gate records are per-namespace. Two campaigns running side by side wrote
# the same gate_*.json and the first one's record was lost, so the directory
# follows SEPSIS_OUTPUT_BASE unless told otherwise.
STATUS_DIR = os.environ.get(
    "SEPSIS_STATUS_DIR",
    os.path.join(PROJECT_ROOT, "outputs", "final_preflight"))


def _meta():
    with open(os.path.join(PROCESSED_DIR, "metadata.json")) as f:
        return json.load(f)


def _status(name: str, payload: dict):
    """Durable per-gate record; the campaign is resumable from these."""
    ensure_dir(STATUS_DIR)
    p = os.path.join(STATUS_DIR, f"{name}.json")
    with open(p, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"  status -> {p}")


def is_complete(run_dir: str, n_epochs: int = N_EPOCHS_FINAL) -> bool:
    """
    Has this run actually finished, rather than merely started?

    `d3qn_best.pt` is written after the first epoch, so treating its presence
    as completion would let a run interrupted at epoch 1 resume as done and
    then be scored as a seed. Completion is the final checkpoint plus a history
    carrying a row for every epoch, and no divergence marker: `train.py` writes
    `DIVERGED` and stops, which can happen on the last epoch and would
    otherwise leave a full-length history behind.
    """
    if os.path.exists(os.path.join(run_dir, "DIVERGED")):
        return False
    final = os.path.join(run_dir, "d3qn_final.pt")
    hist = os.path.join(run_dir, "training_history.csv")
    if not (os.path.exists(final) and os.path.exists(hist)):
        return False
    try:
        return len(pd.read_csv(hist)) >= n_epochs
    except Exception:  # noqa: BLE001 — an unreadable history is not a finished run
        return False


def actor_is_complete(run_dir: str, critic: str = "best") -> bool:
    """
    Has the actor extraction finished, rather than merely started?

    `train_actor` saves the checkpoint inside its epoch loop and writes the
    hyperparameters and history only after it, so a crashed extraction leaves a
    checkpoint that `load_actor` then cannot load. The hyperparameters file is
    the completion marker, and it is also what `load_actor` reads.
    """
    tag = "" if critic == "best" else "_criticfinal"
    ckpt = "actor_best.pt" if critic == "best" else "actor_criticfinal.pt"
    return all(os.path.exists(os.path.join(run_dir, f)) for f in
               (ckpt, f"actor_hyperparams{tag}.json", f"actor_history{tag}.csv"))


def train_one(kind: str, seed: int) -> str:
    """Train a single policy; returns its run directory. Skips completed runs."""
    from src.train import train
    from src.train_iql import train_iql

    name = {"cql": f"cql_a{CQL_ALPHA}_seed{seed}",
            "d3qn": f"d3qn_seed{seed}",
            "iql": f"iql_seed{seed}"}[kind]
    out = os.path.join(OUTPUT_BASE, name)
    if is_complete(out):
        print(f"  {name}: already trained, skipping")
        return out
    if os.path.exists(os.path.join(out, "d3qn_best.pt")):
        print(f"  {name}: partial run found, retraining from scratch")
    stale = os.path.join(out, "DIVERGED")
    if os.path.exists(stale):
        # A marker from an earlier attempt would make the new run look diverged.
        os.rename(stale, stale + ".previous")
        print(f"  {name}: cleared a divergence marker from an earlier attempt")

    t0 = time.time()
    if kind == "iql":
        train_iql(output_dir=out, seed=seed, expectile=IQL_EXPECTILE,
                  n_epochs=N_EPOCHS_FINAL)
    else:
        train(output_dir=out, seed=seed, n_epochs=N_EPOCHS_FINAL,
              cql_alpha=CQL_ALPHA if kind == "cql" else 0.0)
    print(f"  {name}: {(time.time() - t0) / 60:.1f} min")
    return out


def policy_for(kind: str, run_dir: str, meta: dict, device):
    """The evaluation policy for a trained run, per the approved protocol."""
    dim, hid = meta["n_state_features"], [256, 256]
    if kind == "iql":
        # Must be the extracted actor. Scoring softmax(Q) here is what made the
        # earlier IQL result a result about Q-softmax rather than about IQL.
        return actor_greedy_policy(run_dir, device)
    return greedy_q_policy(os.path.join(run_dir, "d3qn_best.pt"), dim, hid, device)


def fqe_for(kind: str, run_dir: str, meta: dict, device, splits=("val", "test"),
            checkpoints=("best", "final")):
    """
    Score a run under both checkpoint-selection rules.

    `train.py` keeps the epoch with the lowest validation Bellman loss, but on
    this data that loss is minimised at epoch 1 and rises monotonically, so the
    "best" checkpoint is an early epoch-1 one. That is not an untrained network:
    epoch 1 follows 5,000 gradient updates, about 7.3 batch-sampled dataset
    equivalents. It inverts the comparison: D3QN
    diverges (Q reaches 33.7, max 114.5 against a +/-15 return bound) yet scores
    highest, because the rule freezes it before it can diverge. Training is
    shared between the two checkpoints, so both are evaluated here and the
    selection rule stays a reporting decision rather than a re-run.
    """
    out = {}
    for ckpt in checkpoints:
        # The FQE fit uses training transitions only, so it does not depend on
        # the split. The first split fits, the second scores that same network,
        # which halves this stage without changing a value.
        fitted = None
        for split in splits:
            suffix = "" if ckpt == "best" else "_finalckpt"
            d = os.path.join(run_dir, f"fqe_{split}{suffix}")
            key = f"{ckpt}_{split}"
            rj = os.path.join(d, "fqe_result.json")
            if os.path.exists(rj):
                out[key] = json.load(open(rj))
                fitted = fitted or d
                continue
            if kind == "iql":
                # The IQL policy is the extracted actor, which has its own
                # checkpoint; the Q-net rule does not apply to it.
                if ckpt != "best":
                    continue
                pol = policy_for(kind, run_dir, meta, device)
            else:
                ck = "d3qn_best.pt" if ckpt == "best" else "d3qn_final.pt"
                pol = greedy_q_policy(os.path.join(run_dir, ck),
                                      meta["n_state_features"], [256, 256], device)
            out[key] = run_fqe(
                pol, processed_dir=PROCESSED_DIR,
                state_dim=meta["n_state_features"], hidden_dims=[256, 256],
                split=split, output_dir=d, seed=42,
                state_cols=meta["state_cols"],
                label=f"{os.path.basename(run_dir)}:{ckpt}", device=device,
                reuse_from=fitted)
            fitted = fitted or d
    return out


def run_references(meta: dict, device) -> dict:
    """
    Always-withhold control, and the reference fits every value is read against.

    The constant policy exists to answer a question no learned-policy number
    can: whether an advantage over the clinician reflects state-dependent
    selection or simply recommending less treatment.
    """
    from config.paths import BEHAVIOR_POLICY_DIR
    from src.models.baselines import train_behavior_cloning
    from src.models.behavior_policy import fit_behavior_policies

    dim = meta["n_state_features"]

    # The behaviour policy is a 25-class classifier over actions, so the new
    # bins invalidate any earlier fit; it must be refit on this dataset.
    bp_ckpt = os.path.join(BEHAVIOR_POLICY_DIR, "mlp_behavior.pt")
    if not os.path.exists(bp_ckpt):
        print("Fitting behaviour policy...")
        fit_behavior_policies(output_dir=BEHAVIOR_POLICY_DIR)

    bc_dir = os.path.join(OUTPUT_BASE, "bc")
    bc_ckpt = os.path.join(bc_dir, "bc_final.pt")
    if not os.path.exists(bc_ckpt):
        print("Training behaviour cloning...")
        train_behavior_cloning(output_dir=bc_dir)

    specs = [
        ("withhold", constant_policy(0),
         os.path.join(OUTPUT_BASE, "constant_action0")),
        ("clinician", mlp_classifier_policy(bp_ckpt, dim, device),
         os.path.join(OUTPUT_BASE, "fqe_clinician")),
        ("bc", mlp_classifier_policy(bc_ckpt, dim, device),
         os.path.join(OUTPUT_BASE, "fqe_bc")),
    ]

    res = {}
    for name, pol, base in specs:
        fitted = None
        for split in ("val", "test"):
            d = os.path.join(base, f"fqe_{split}")
            f = os.path.join(d, "fqe_result.json")
            if os.path.exists(f):
                res[f"{name}_{split}"] = json.load(open(f))
                fitted = fitted or d
                continue
            res[f"{name}_{split}"] = run_fqe(
                pol, processed_dir=PROCESSED_DIR, state_dim=dim,
                hidden_dims=[256, 256], split=split, output_dir=d, seed=42,
                state_cols=meta["state_cols"], label=name, device=device,
                reuse_from=fitted)
            fitted = fitted or d
    return res


def _require_signoff(stage: str) -> None:
    """
    Explicit go/no-go before any stage that costs hours.

    The pilot may run on any dataset, since it exists to exercise the pipeline.
    Everything after it is the reported result, and a campaign started on a
    dataset that later turns out to need rebuilding costs more than the check.
    """
    # Two conditions, deliberately. The environment variable is set when the
    # campaign is launched; the file is written when the dataset actually
    # passes review. A chain already running can therefore be stopped by not
    # writing the file, which is the only way to halt it without killing it.
    signoff = os.path.join(STATUS_DIR, "SIGNOFF")
    if os.environ.get("SEPSIS_SEEDS_OK") != "1" or not os.path.exists(signoff):
        raise SystemExit(
            f"stage '{stage}' is gated. It needs SEPSIS_SEEDS_OK=1 and a "
            f"sign-off file at {signoff}, written once the dataset this "
            f"campaign reads has been audited.")


def empirical_return(results: dict, split: str):
    """
    The observed discounted return on a split, from any fit that carries it.

    `fqe_for` keys its results by "{checkpoint}_{split}", so there is no plain
    "test" key to read. Indexing one cost two pilots their gate record after
    every expensive step in them had already succeeded. The value is a property
    of the data rather than of a policy, so any fit on that split will do.
    """
    return next((r["empirical_clinician_return"]
                 for k, r in results.items()
                 if k.endswith(f"_{split}") and "empirical_clinician_return" in r),
                None)


def stage_pilot(device) -> dict:
    meta = _meta()
    runs = {k: train_one(k, 42) for k in ("cql", "d3qn")}
    fqe = {k: fqe_for(k, v, meta, device) for k, v in runs.items()}
    refs = run_references(meta, device)

    summary = {
        "stage": "pilot", "seeds": [42], "n_epochs": N_EPOCHS_FINAL,
        "processed_dir": PROCESSED_DIR, "output_base": OUTPUT_BASE,
        "runs": runs,
        "fqe": {k: {s: v[s]["fqe_value"] for s in v} for k, v in fqe.items()},
        "checkpoint_note": (
            "Both checkpoint rules are reported. 'best_*' is train.py's "
            "lowest-validation-Bellman-loss epoch, which on this data is epoch 1; "
            "'final_*' is epoch 40."),
        "references": {k: v["fqe_value"] for k, v in refs.items()},
        "empirical_clinician_return": {
            split: empirical_return(fqe["cql"], split)
            for split in ("val", "test")},
    }
    _status("gate_pilot", summary)
    return summary


def stage_iql(device) -> dict:
    _require_signoff("iql")
    from src.train_iql_actor import train_actor
    meta = _meta()
    run = train_one("iql", 42)
    if not os.path.exists(os.path.join(run, "actor_best.pt")):
        train_actor(run, seed=42, n_epochs=N_EPOCHS_FINAL,
                    processed_dir=PROCESSED_DIR)
    fqe = fqe_for("iql", run, meta, device)
    hist = pd.read_csv(os.path.join(run, "actor_history.csv"))
    # The evaluated policy is `actor_best.pt`, saved at the lowest-validation
    # epoch, which is not the last row of the history. Reporting the last row's
    # agreement and entropy described a network the FQE value does not come
    # from. Both are carried, each labelled with the checkpoint it belongs to.
    best = hist.loc[hist["val_loss"].idxmin()]
    last = hist.iloc[-1]
    summary = {
        "stage": "iql", "run": run,
        "fqe": {s: fqe[s]["fqe_value"] for s in fqe},
        "evaluated_policy": "actor_best.pt, the lowest-validation-loss epoch",
        "best_actor_epoch": int(best["epoch"]),
        "best_actor_val_loss": float(best["val_loss"]),
        "best_actor_agreement": float(best["clinician_agreement"]),
        "best_actor_entropy": float(best["entropy"]),
        "best_actor_mean_weight": float(best["mean_weight"]),
        "best_actor_frac_clipped": float(best["frac_clipped"]),
        "last_row_epoch": int(last["epoch"]),
        "actor_final_agreement": float(hist["clinician_agreement"].iloc[-1]),
        "actor_final_entropy": float(hist["entropy"].iloc[-1]),
        "actor_frac_clipped": float(hist["frac_clipped"].iloc[-1]),
    }
    _status("gate_iql", summary)
    return summary


def stage_seeds(device) -> dict:
    _require_signoff("seeds")
    from src.train_iql_actor import train_actor
    meta = _meta()
    out = {}
    for kind in ("cql", "d3qn", "iql"):
        for seed in SEEDS:
            if seed == 42 and kind in ("cql", "d3qn", "iql"):
                run = os.path.join(OUTPUT_BASE, {
                    "cql": f"cql_a{CQL_ALPHA}_seed42", "d3qn": "d3qn_seed42",
                    "iql": "iql_seed42"}[kind])
                # The pilot and the IQL gate already trained seed 42, so it
                # is reused rather than retrained. For IQL that needs the
                # extracted actor as well as the critic: the critic alone is a
                # complete run of a method the thesis does not evaluate.
                if is_complete(run) and (kind != "iql" or actor_is_complete(run)):
                    out[f"{kind}_{seed}"] = run
                    continue
            run = train_one(kind, seed)
            if not is_complete(run):
                raise AssertionError(
                    f"{run} did not complete. Refusing to continue with a "
                    f"partial seed, which downstream collectors would skip "
                    f"quietly and report as a smaller but successful table.")
            if kind == "iql":
                train_actor(run, seed=seed, n_epochs=N_EPOCHS_FINAL,
                            processed_dir=PROCESSED_DIR)
                if not actor_is_complete(run):
                    raise AssertionError(
                        f"{run} has no complete extracted actor. IQL without "
                        f"one is not IQL, and evaluating its critic would be a "
                        f"different method under the same name. The checkpoint "
                        f"alone is not enough: it is written inside the epoch "
                        f"loop, while the hyperparameters load_actor needs are "
                        f"written after it.")
            out[f"{kind}_{seed}"] = run
    _status("gate_seeds", {"stage": "seeds", "runs": out})
    return out


def stage_fqe(device) -> dict:
    meta = _meta()
    rows = []
    for kind, pat in (("cql", f"cql_a{CQL_ALPHA}_seed"), ("d3qn", "d3qn_seed"),
                      ("iql", "iql_seed")):
        for seed in SEEDS:
            run = os.path.join(OUTPUT_BASE, f"{pat}{seed}")
            if not is_complete(run):
                print(f"  {pat}{seed}: not a completed run, not scored")
                continue
            f = fqe_for(kind, run, meta, device)
            for key, r in f.items():
                ckpt, split = key.split("_", 1)
                rows.append({"method": kind, "seed": seed, "checkpoint": ckpt,
                             "split": split, "fqe_value": r["fqe_value"]})
    df = pd.DataFrame(rows)
    # 2 checkpoints x 2 splits for each of CQL and D3QN across five seeds, plus
    # val and test for the five IQL actors, which are scored once here and
    # again against the final critic in stage_iqlfinal.
    expected = len(SEEDS) * (2 * 2 + 2 * 2 + 2)
    if len(df) != expected:
        raise AssertionError(
            f"FQE produced {len(df)} rows, expected {expected}. A missing "
            f"policy must not become a thesis result by way of a tolerant "
            f"collector. Present: "
            f"{df.groupby(['method', 'checkpoint']).size().to_dict()}")
    p = os.path.join(OUTPUT_BASE, "final_fqe_results.csv")
    df.to_csv(p, index=False)
    print(df.groupby(["method", "checkpoint", "split"])["fqe_value"]
            .agg(["mean", "std", "count"]).to_string())
    _status("gate_fqe", {"stage": "fqe", "csv": p, "n_rows": len(df)})
    return {"csv": p, "n": len(df)}


def stage_iqlfinal(device) -> dict:
    """
    Re-extract the IQL policy against the final-epoch critic, and score it.

    Needed because the actor is trained on Q(s,a) - V(s) from a *saved* critic.
    With the lowest-validation-loss rule that critic is epoch 1, the advantages
    are close to zero over much of the state space, and advantage-weighted BC
    degenerates toward plain behavioural cloning, so the contrast the method
    depends on is largely absent under that rule. The policy is still the
    output of the prespecified extraction; the claim is about degeneracy, not
    identity. Scoring both
    keeps the checkpoint decision a reporting choice for every method rather
    than only for CQL and D3QN.
    """
    from src.train_iql_actor import train_actor
    from src.final_fqe import actor_greedy_policy_critic
    meta = _meta()
    out = {}
    for seed in SEEDS:
        run = os.path.join(OUTPUT_BASE, f"iql_seed{seed}")
        # The same completion test as everywhere else, rather than the
        # presence of one file. A skipped seed here is caught by the
        # cardinality assertion below, but consistency is cheaper than
        # relying on that.
        if not is_complete(run):
            print(f"  iql_seed{seed}: not a completed run, skipping")
            continue
        if not actor_is_complete(run, "final"):
            train_actor(run, seed=seed, n_epochs=N_EPOCHS_FINAL,
                        processed_dir=PROCESSED_DIR, critic="final")
            if not actor_is_complete(run, "final"):
                raise AssertionError(
                    f"{run}: final-critic actor extraction did not complete")
        fitted = None
        for split in ("val", "test"):
            d = os.path.join(run, f"fqe_{split}_criticfinal")
            rj = os.path.join(d, "fqe_result.json")
            if os.path.exists(rj):
                out[f"iql_{seed}_{split}"] = json.load(open(rj))["fqe_value"]
                fitted = fitted or d
                continue
            r = run_fqe(
                actor_greedy_policy_critic(run, device, critic="final"),
                processed_dir=PROCESSED_DIR, state_dim=meta["n_state_features"],
                hidden_dims=[256, 256], split=split, output_dir=d, seed=42,
                state_cols=meta["state_cols"],
                label=f"iql_seed{seed}:critic-final", device=device,
                reuse_from=fitted)
            fitted = fitted or d
            out[f"iql_{seed}_{split}"] = r["fqe_value"]
    expected = 2 * len(SEEDS)
    if len(out) != expected:
        raise AssertionError(
            f"final-critic IQL produced {len(out)} results, expected "
            f"{expected} (five seeds on val and test). Got {sorted(out)}")
    _status("gate_iqlfinal", {"stage": "iqlfinal", "fqe": out})
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True,
                   choices=["pilot", "iql", "seeds", "fqe", "iqlfinal"])
    a = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PROCESSED_DIR={PROCESSED_DIR}\nOUTPUT_BASE={OUTPUT_BASE}\n"
          f"device={dev} n_epochs={N_EPOCHS_FINAL}")
    {"pilot": stage_pilot, "iql": stage_iql, "seeds": stage_seeds,
     "fqe": stage_fqe, "iqlfinal": stage_iqlfinal}[a.stage](dev)
