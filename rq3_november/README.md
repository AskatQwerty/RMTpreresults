# RQ3: SHAP stability on 1,000 applicants

The November analysis compares default-probability explanations for seven Level 4 models with 322 original features. All 50 explanation sets use the same 1,000 test IDs in the same order: 500 applicants with a bureau record and 500 without one. The common background contains 100 training applicants. Checks confirmed that all 1,000 IDs belong to the official 61,503-applicant test set. [Models and checksums](verification/BUNDLE.json) · [Input checksums](verification/input_checksums.json)

Each model was retrained five times on stratified samples of 18,000 out of 20,000 training applicants (seeds 1–5). The table averages the fraction of shared personal top-5 features over applicants and the ten pairs of retrainings.

| Model | Personal top-5 overlap across retrainings | Spearman correlation of global rankings |
|---|---:|---:|
| Logistic regression | 85.74% | 0.935 |
| Random forest | 80.82% | 0.974 |
| XGBoost | 72.13% | 0.911 |
| LightGBM | 74.57% | 0.894 |
| MLP | 58.90% | 0.578 |
| Soft voting | 80.20% | 0.919 |
| Stacking | 74.12% | 0.888 |

For soft voting, the 95% bootstrap interval for top-5 overlap is 79.66–80.73%, which includes the 80% threshold. [Stability and intervals](rq3/out/tables/rq3_stability.csv) · [Agreement with logistic regression](rq3/out/tables/rq3_consistency_with_LR.csv) · [Figures](rq3/out/figures/) · [Full results](RESULTS_RU.txt)

## Reproduction

The code uses the official [model bundle](../model_bundles/README.md), pinned packages and the input data supplied to the team. The [instructions](verification/REPRODUCE_RU.txt) describe directory layout, `SHAP_PROJECT` and execution commands. To run from this directory, copy the model bundle here as `active_models.zip`. The input directory `RMT_batch_11_15/data` must be alongside `rq3_november`.

Seeds: 42 for sampling/bootstrap, 1–5 for retraining, and 0, 11, 12 for permutations. Recalculation and checks on an Intel Core i5-10300H took about 33 minutes using the saved retrained models. [Versions](verification/environment.json) · [Hardware and timing scope](verification/run_environment.json) · [Explanation checks](verification/explanation_checks.csv) · [October experiment](../rq3/README_RQ3.md)
