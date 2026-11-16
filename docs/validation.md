# Validation

Checked on 2026-11-16 with Python 3.13 and the versions in `requirements-tested.txt`.

| Check | Result |
|---|---|
| Regression and result-integrity tests | 111 passed |
| Synthetic demo | Preprocessing, six training updates and FQE completed for 80 synthetic subjects / 480 transitions |
| Imports | All 36 non-package modules import without experiment tracking |
| Syntax and dependencies | 41 source/config/example/script modules compile; all internal imports resolve |
| Source integrity | Six aggregate input files match their pinned Git blob hashes |
| Thesis comparison | FQE rows match Table 3, early support matches Table 2, estimator counts match Section 5.1 |
| Figures and table | Regenerated from the aggregate JSON; both charts inspected |

The tests cover missing or duplicated seeds, mixed splits, checkpoint coverage, inconsistent FQE summaries, and the distinction between undefined estimates and zero.

The distribution contains 66 selected files: code, configuration, tests, SQL, documentation, aggregate results and figures. It excludes patient data, per-patient arrays, model checkpoints, caches, environments and Git history. Local-user paths, credential patterns and relative documentation links were checked.

The comment cleanup preserves executable code and research values. The direct dependency versions used for these checks are recorded in `requirements-tested.txt`; they are not a record of the original training environment.

## Reproduction limits

Clinical-data extraction and the full training campaign were not rerun for this release. The workflow was checked against the retained code and its dependencies. Regenerating the aggregate figures does not independently verify patient-level calculations or establish treatment benefit.

The GitHub Actions workflow runs the tests, figure generation and synthetic demo after a push. See [workflow runs](https://github.com/fnastar/sepsis-offline-rl/actions) for their status.
