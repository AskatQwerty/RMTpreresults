# RMT: Home Credit default risk

**Current results: `pilot_20k_20config_mlp_updated`.** The active set contains 28 fitted Pipelines: seven model types at four information levels. Four MLPs were replaced after a supplemental study; the other 24 models retain their final fits from the 20-configuration search.

- [RQ1/RQ2, conditional profit and fairness report](reports/pilot_20k_20config_mlp_updated/evaluation/results_report.html)
- [CSV / LaTeX tables](reports/pilot_20k_20config_mlp_updated/evaluation/tables) · [Figures](reports/pilot_20k_20config_mlp_updated/evaluation/figures)
- [All 28 models and loading instructions](model_bundles/README.md) · [Active version](docs/active_results.json)
- [November RQ3 / SHAP](rq3_november/README.md)

Data source: [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data). The unit of analysis is a current application; `TARGET=1` denotes default. The primary precision–recall metric is **average precision (AP)**. Raw data, prepared X, individual IDs/predictions and bootstrap samples are excluded from Git. The verified fitted-model archive contains no individual customer records.

## Data and training

The saved split contains **246,008 development** and **61,503 test** applicants. All models use the same **20,000 training applicants**: 18,385 non-defaults and 1,615 defaults, without class resampling. Test class counts are 56,538 / 4,965. The three saved stratified CV folds use seed 42.

| Level | Added information | Features before encoding |
|---|---|---:|
| 1 | application and deterministic application features | 124 |
| 2 | bureau, bureau_balance | 178 |
| 3 | POS_CASH_balance, credit_card_balance | 264 |
| 4 | installments_payments, previous_application | 322 |

Models: L2 Logistic Regression, Random Forest, XGBoost, LightGBM, MLP, soft voting and stacking. Both ensembles use LR, XGBoost and LightGBM. Preprocessing is fitted within each training partition, including stacking's OOF base Pipelines. TARGET, IDs, gender and direct age representations are excluded from X.

The original search completed **1,680 outer CV fits**: 28 pairs × 20 configurations × 3 folds, followed by 28 final fits on all 20,000 applicants. [Scores and selection evidence](reports/training/pilot_20k_20config_combined/) are preserved.

The supplemental MLP study added **48 outer CV evaluations** (four variants × four levels × three folds) and four replacement final fits. Epochs are selected by log-loss on an internal 15% validation split, followed by fresh preprocessing and network training on the entire available train. Selected MLPs clip standardized numeric values to ±5; alpha is 0.00001 at Level 1 and 0.1 at Levels 2–4. Final epoch counts are **27, 24, 16, 21**.

The MLP revision followed inspection of the previous test results. Selection used CV before the new test evaluation, but this remains a **post-hoc exploratory analysis**. The active MLPs are not unchanged winners of the original 20-configuration search. Repeated-training analyses are separate from bootstrap intervals.

## Evaluation

The active revision reused saved probabilities, with no new fit/predict during model adoption. Evaluation recomputed **1,000 common paired bootstrap replicates**, seed 42. The 95% percentile intervals are pointwise, unadjusted and exploratory, conditional on fixed fitted models.

Outputs cover quality, AP gains across levels, calibration, six Bonferroni-adjusted paired DeLong ROC-AUC comparisons against LR at Level 4, conditional profit and fairness. Fixed thresholds are **0.5, 1/6 and 1/11**. Approval means `p(default) < threshold`; profit uses observed TARGET with `m=1` and `L=1,5,10`. Fairness groups are F/M and age <25/≥25. Unknown groups remain in overall metrics and are excluded only from binary group gaps.

MLP Level 4: test log-loss **0.249034**, AP **0.235569**, ROC-AUC **0.754453**. See the current report for intervals and [the MLP revision report](reports/mlp_repair_20k_v1/results_report.html) for the before/after comparison.

## Archives and RQ3

- [Five-configuration preliminary results](reports/pilot_20k_5config/evaluation/results_report.html) and [archived README](docs/archive/README_pilot_20k_5config.md).
- [20-configuration results before MLP replacement](reports/pilot_20k_20config_final/evaluation/results_report.html).
- [November RQ3](rq3_november/README.md): seven final Level 4 models and five retrainings per model, explained on the same 1,000 test applicants. [Stability](rq3_november/rq3/out/tables/rq3_stability.csv) · [October RQ3](rq3/README_RQ3.md).

## Reproduction

Use Python **3.13.9** and the pinned versions in [requirements.txt](requirements.txt) and [software_versions.json](docs/software_versions.json); figures also require `requirements-reporting.txt`. Customer data are supplied separately. Loading a saved Pipeline does not require training.

- `scripts/import_pilot_cv_results.py`: verify and combine the 1,680 CV results.
- `scripts/run_final_pilot_background.py`: original 28 final fits.
- `scripts/run_mlp_repair.py`: configuration check by default; `--mode run` starts/resumes the supplemental MLP study.
- `scripts/promote_repaired_mlp.py`: adopt saved models into a new active revision; existing revisions are not overwritten.
- `python -m src.updated_pilot_evaluation`: evaluate saved local probabilities, without fitting.
- `scripts/validate_updated_pilot.py`: independent checks of metrics, lineage, profit/fairness and integrity.
- `scripts/build_active_model_bundle.py`: export saved models without customer records.

Notebooks 01–04 retain the original preparation and five-configuration pilot. Their outputs are historical; the current manuscript uses `reports/pilot_20k_20config_mlp_updated/evaluation/`. Do not rerun training merely to view the reports.

## License and release

Project code is available under the [MIT License](LICENSE). The archived version cited in the paper is [v1.0](https://github.com/AskatQwerty/RMTpreresults/releases/tag/v1.0). This English presentation update does not change saved numerical results or the release tag; see the [translation verification record](docs/english_presentation_audit.json).

AI tools assisted with coding, analysis, and editing. The authors are responsible for the reported results.
