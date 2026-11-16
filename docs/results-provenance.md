# Results provenance and interpretation

## Source and verification

The aggregate export comes from the final `runs_90d_final_v4` experiment outputs at source revision `0249561084302165c91eb0fac4f9ea33f43ac391`. It uses six named files:

- `final_results_test.csv`: FQE value per method, checkpoint and training seed, plus references.
- `final_results_test.json`: seed summaries, checkpoint definitions and saved paired comparisons after Holm correction.
- `data_support/support_metrics.json`: early-checkpoint support summaries.
- `data_support/support_metrics_final.json`: final-checkpoint support summaries.
- `data_support/support_metrics_withhold.json`: the always-withhold control on the same support samples.
- `ope_diagnostics.csv`: filtered to **test split, greedy (exact argmax)**. Validation and softmax sensitivity results are excluded.

The export does not use the superseded `ope_diagnostics_pre_drfix_2026-08-23` files. The selected DR estimates use the fitted evaluation critic as the control variate.

The FQE rows were checked against Appendix B, Table 3 of the [public thesis](https://github.com/fnastar/Thesis/blob/b41e6a5edb2bb6d8b21207fd748bc2f4f0f13d61/Thesis_v1.pdf); early support values match Table 2, and estimator counts match Section 5.1. The public thesis was checked at revision `b41e6a5edb2bb6d8b21207fd748bc2f4f0f13d61` during release preparation on 2026-11-16. Final support values come from the separate final-checkpoint summary, not the early table.

`results/thesis_results.json` records the source revision and SHA-256 of each input file. Its rows describe experiments or whole evaluation populations, never individual patients. The exporter selects fields explicitly, converts missing numeric values to JSON `null`, and rejects incomplete seed coverage, mixed splits and mismatched support samples. It excludes per-step count arrays as well as all patient-level arrays. The export includes 30 learned-policy FQE rows, 30 OPE diagnostic rows, six policy summaries, three FQE references and 12 saved paired comparisons.

## What is reproduced

`python -m scripts.plot_results` regenerates the displayed means, standard deviations, reference differences, diagnostic counts, two charts and detailed Markdown table from the saved aggregate export. The source-export command is included for researchers who rerun the experiments locally.

This is not an independent rerun of model training or clinical-data extraction. The saved paired tests are carried through from the final research summary; they cannot be independently recomputed at patient level without the restricted evaluation arrays. Five training seeds quantify only part of the uncertainty.

## Reading the figures

The value/support chart reports both checkpoint rules. Its left axis is FQE minus the **estimated behaviour policy**, a learned model of clinical actions. The right axis is mean local support: the fraction of 50 nearest training decisions that used the recommended action. Support uses 15,000 sampled test decisions and 80,000 training decisions, with sampling seed 42. Whiskers show one standard deviation across five training seeds, not a confidence interval. The single-fit withhold control has no seed SD.

The dashed support line describes **observed clinician actions**. It is a different reference from the fitted behaviour model used on the value axis. Neither FQE values nor their differences are survival probabilities or clinical effect sizes.

The estimator chart has one mark for each of three methods × two checkpoints × five seeds. Its categories reproduce the thesis figure's rule:

- WIS is undefined when its saved value is missing; otherwise it is defined. The displayed median effective sample size makes its limited support visible.
- PDWIS is defined when a saved value exists; otherwise it is undefined if no complete trajectory matched, or withheld by the per-step support criterion.
- DR is flagged when its value exceeds the possible return range of ±15.
- FQE is defined when its saved estimate exists. This does not label it unbiased or clinically validated.

## Limitations that matter

The always-withhold policy scores 12.713, above the estimated behaviour reference at 12.518. This is a negative control for interpretation, not a treatment recommendation. It demonstrates why a favourable score and local support do not establish better treatment.

The thesis reports an observed discounted return of about 7.04 on the test patients. That observed return and the FQE estimate for a fitted behaviour model are different quantities. The discrepancy prevents a clinical interpretation of the absolute FQE scale.

The study is retrospective and single-centre, with limited observed state information and possible unmeasured confounding. Patient-level random splits do not establish temporal or external generalisation. This package does not add evidence of clinical benefit.
