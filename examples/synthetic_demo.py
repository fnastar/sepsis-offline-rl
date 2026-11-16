"""Exercise the research pipeline with wholly invented trajectories."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import torch


def main() -> None:
    # Set paths before importing research modules, which read configuration at import.
    with tempfile.TemporaryDirectory(prefix="sepsis-synthetic-") as temporary:
        root = Path(temporary)
        features = root / "features"
        processed = root / "processed"
        features.mkdir()
        for name, value in {
            "SEPSIS_FINAL_FEATURES_DIR": str(features),
            "SEPSIS_FINAL_PROCESSED": str(processed),
            "SEPSIS_PROCESSED_DIR": str(processed),
            "SEPSIS_SKIP_BIN_CHECK": "1",
        }.items():
            os.environ[name] = value

        from src.final_preprocessing import STATE_COLS, build, verify_action_bins
        from src.train import train
        from src.final_fqe import greedy_q_policy, run_fqe

        torch.set_num_threads(1)
        rng = np.random.default_rng(42)
        n_subjects, steps = 80, 6
        n = n_subjects * steps
        frame = pd.DataFrame(rng.normal(size=(n, len(STATE_COLS))), columns=STATE_COLS)
        frame["subject_id"] = np.repeat(np.arange(n_subjects), steps)
        frame["stay_id"] = frame["subject_id"] + 1000
        frame["time_bin"] = np.tile(np.arange(steps), n_subjects)
        frame["gender"] = np.where(frame["subject_id"] % 2, "M", "F")
        frame["icu_mortality"] = np.repeat(rng.integers(0, 2, n_subjects), steps)
        frame["fio2"] = 40.0
        frame["pao2"] = 90.0
        for column in ("vaso_total", "iv_fluid_total"):
            frame[column] = np.where(rng.random(n) < 0.35, 0.0, rng.uniform(0.01, 1, n))
        frame.to_parquet(features / "features.parquet", index=False)

        splits = build()
        verify_action_bins(str(processed))
        model_dir = root / "model"
        train(output_dir=str(model_dir), n_epochs=2, steps_per_epoch=3,
              batch_size=16, hidden_dims=[32, 32], quiet=True)
        device = torch.device("cpu")
        policy = greedy_q_policy(str(model_dir / "d3qn_best.pt"),
                                 len(STATE_COLS), [32, 32], device)
        result = run_fqe(policy, str(processed), len(STATE_COLS), [32, 32],
                         n_iterations=2, inner_epochs=1, batch_size=64,
                         state_cols=STATE_COLS, device=device,
                         output_dir=str(root / "evaluation"), label="SYNTHETIC")
        summary = {
            "data": "entirely synthetic; no clinical interpretation",
            "subjects": n_subjects,
            "transitions": {k: len(v) for k, v in splits.items()},
            "training_updates": 6,
            "demo_fqe_value": result["fqe_value"],
        }
        print(json.dumps(summary, indent=2))
        assert np.isfinite(result["fqe_value"]), "Synthetic evaluation must be finite"


if __name__ == "__main__":
    main()
