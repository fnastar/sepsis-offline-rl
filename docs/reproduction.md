# Research workflow

## Two levels of reproduction

To rebuild the figures and table from the saved real experiment summaries, install `requirements-figures.txt` and run `python -m scripts.plot_results`. This requires no clinical data.

The workflow below recomputes the research from MIMIC-IV records. It requires independently authorised **MIMIC-IV v3.1** access, its derived Sepsis-3/SOFA exports, and substantial memory, storage and compute. It has been checked for source dependencies and command order, but was **not run end to end for this release**. Exact numerical agreement across hardware and library versions is not guaranteed.

## Inputs and installation

Use Python 3.13 and install `requirements.txt`. The direct versions used for the local release checks are recorded in `requirements-tested.txt`; these are release-validation versions, not a recovered historical training environment.

Set `MIMIC_ROOT` to the directory containing `hosp/` and `icu/`. Obtain the dataset through [PhysioNet](https://physionet.org/content/mimiciv/3.1/). The final path reads the admissions, patients, icustays, chartevents, labevents, inputevents and outputevents tables. See `config/paths.py` for filenames.

Run the provided read-only BigQuery queries in your authorised data environment:

| Query | Save complete export as |
|---|---|
| `sql/export_sepsis3_v3_1.sql` | `data/cohort/sepsis3_official_v3_1.csv.gz` |
| `sql/export_sofa_v3_1.sql` | `data/cohort/sofa_official_v3_1.csv.gz` |

Preserve column names and timestamps. These are gzip-compressed CSV files, not plain CSV files renamed with a gzip extension. Do not commit or distribute the exports. Paths can be overridden with `SEPSIS3_SOURCE_PATH` and `SEPSIS_SOFA_OFFICIAL`.

## Build and audit

Run from the repository root, in the same shell. Use new data/output directories when changing a configuration; feature parts and fitted models are cached.

```bash
export MIMIC_ROOT=/path/to/mimic-iv-3.1
export SEPSIS_SKIP_BIN_CHECK=1
export SEPSIS_FINAL_FEATURES_DIR=data/features_final
export SEPSIS_FINAL_PROCESSED=data/processed_final
export SEPSIS_PROCESSED_DIR=data/processed_final
export SEPSIS_OUTPUT_BASE=outputs/runs_final
export SEPSIS_BP_DIR=outputs/behavior_final
export SEPSIS_STATUS_DIR=outputs/audit_final
export SEPSIS_FIG_DIR=outputs/figures_final
export SEPSIS_VASO_AGG=max
export SEPSIS_NE_CEILING=5.0

python -m src.final_cohort
python -m src.final_features
python -m src.final_preprocessing
python -m src.final_audit
python -m src.final_scaler_audit
python -m src.final_audit_extra
```

Stop and resolve failed audits before training. `SEPSIS_SKIP_BIN_CHECK=1` bypasses an old static-bin check; the final preprocessing learns thresholds from the training split and verifies the saved action bins itself.

The original scaler audit flagged `prev_vaso` and `wbc`. Their residual tail ratios were reviewed and accepted in that experiment. If the same flags occur, review the generated audit and document your reasoning before rerunning with `SEPSIS_SCALER_ACCEPT=prev_vaso,wbc`. This variable records accepted exceptions; it does not fix a bad dataset or waive new flags.

The reference cohort contains 25,442 first eligible adult episodes, split 70/15/15 by patient with seed 42: 176,012 training, 38,011 validation and 37,874 test transitions. The outcome is 90-day mortality, despite the inherited `icu_mortality` column name. The episode spans -24 to +48 hours from onset in four-hour bins. Different data releases, incomplete exports or stale caches can fail cohort checks.

Historical migration checks that compared earlier datasets are not required to build the final dataset from scratch. They are not included in this release.

## Train the campaign

```bash
python -m src.final_campaign --stage pilot
```

The pilot is **not a cheap demo**: it trains D3QN/CQL for seed 42 over the full schedule and fits the reference policies. Review its outputs and dataset audits. After accepting the dataset, write your review to `$SEPSIS_STATUS_DIR/SIGNOFF` and set `SEPSIS_SEEDS_OK=1`. The repository intentionally supplies no pre-approved signoff.

Then run, in order, stopping on errors:

```bash
export SEPSIS_SEEDS_OK=1
python -m src.final_campaign --stage iql
python -m src.final_campaign --stage seeds
python -m src.final_campaign --stage fqe
python -m src.final_campaign --stage iqlfinal
```

The `iql` and `seeds` training stages enforce the signoff gate. The later stages also require the preceding outputs. The campaign uses seeds 42–46, 40 training epochs, CQL alpha 2.0 and IQL expectile 0.7. It evaluates greedy policies. Early D3QN/CQL refers to the lowest-validation-Bellman-loss checkpoint, which was epoch 1 after 5,000 updates in the saved experiment. Final is epoch 40.

For IQL, `iqlfinal` trains the actor against the final critic; both early-critic and final-critic actors are selected by validation loss. It is necessary for the six-group results table. In the retained code, FQE is trained on training transitions and a fitted evaluator may be reused to score validation and test initial states. Neither evaluation split supplies FQE training transitions.

## Evaluate and assemble

```bash
python -m src.final_ope
SEPSIS_SUPPORT_CKPT=best python -m src.data_support
SEPSIS_SUPPORT_CKPT=final python -m src.data_support
python -m src.support_withhold_control
python -m src.final_results
SEPSIS_FQE_SUFFIX= python -m src.fqe_significance
SEPSIS_FQE_SUFFIX=_finalckpt python -m src.fqe_significance
python -m src.final_multiplicity
```

Run multiplicity correction **after** both significance runs and `final_results`; the declared family contains three methods × two checkpoints for each reference. Rerunning `final_results` afterwards replaces its corrected summary, so rerun multiplicity too.

`final_ope` also writes validation and softmax sensitivity results. The portfolio export selects only test/greedy rows and the corrected FQE-based DR control variate. `data_support` retains optional legacy QR support but skips it when no QR checkpoints exist; QR is not part of the reported three-method campaign.

The research commands create patient-level files locally, including processed tables and per-stay `v0.npz` arrays. Keep them in the ignored data/output directories. They are needed for full analysis but are not public release artifacts.

Optional descriptive analyses retained here are `python -m src.final_action_profile` and `python -m src.final_subgroup`. The public package does not claim to rerun every exploratory analysis in the thesis.

## Export aggregate findings

After confirming all five seeds and both checkpoint arms completed, export to a new filename so the bundled thesis snapshot is preserved. Replace the revision label below with the revision used for your run.

```bash
python -m scripts.export_results \
  --run-dir "$SEPSIS_OUTPUT_BASE" \
  --source-commit YOUR_RUN_REVISION \
  --out outputs/reproduced_results.json

python -m scripts.plot_results \
  --input outputs/reproduced_results.json \
  --out-dir outputs/reproduced_figures \
  --table outputs/reproduced_results.md
```

The exporter reads only six named summary files and selects experiment-level fields. It rejects incomplete or mixed-split results. Review any new export before publishing it; never publish the entire research output directory.

Some retained modules contain legacy entry points or fallback paths. Use the `final_*` workflow above and retain its per-stay evaluation outputs for the significance steps. Experiment tracking is optional; install `wandb` separately only if enabling it.
