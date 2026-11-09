# Archived RMT five-configuration pilot

This page documents the earlier **`pilot_20k_5config`** results. For the active models and November RQ3, see the [current repository README](../../README.md).

RQ1 compares credit-risk models; RQ2 measures the effect of adding credit history. The unit of analysis is one current application, `TARGET=1` denotes default, and the primary PR metric is average precision (AP), computed with `average_precision_score`. The [Home Credit Default Risk data](https://www.kaggle.com/competitions/home-credit-default-risk/data), individual records, predictions and the original fitted models were excluded from this pilot's Git publication. The resource pilot and unfinished full A2 experiment are separate.

## Models and information levels

Seven model types were used: L2 Logistic Regression, Random Forest, XGBoost, LightGBM, MLP, soft voting and stacking. Both ensembles use LR, XGBoost and LightGBM. Trainable preprocessing is part of every full Pipeline, including stacking's OOF base models.

| Level | Added sources | Raw prepared features |
|---|---|---:|
| 1 | `application` and deterministic application features | 124 |
| 2 | `bureau`, `bureau_balance` | 178 |
| 3 | `POS_CASH_balance`, `credit_card_balance` | 264 |
| 4 | `installments_payments`, `previous_application` | 322 |

Applicants are identical across levels. Historical records are aggregated before joins; absent history retains missing values and indicators. TARGET, IDs, gender and direct age features are excluded from X. Gender/age are stored separately for fairness. Imputation, encoding and scaling are fitted only on the current fold's train or final refit train.

## Completed pilot protocol

The 307,511 labelled applications were split once into **246,008 development** and **61,503 test** applicants, stratified with seed 42. The pilot used **20,000** development applicants: **18,385** class 0 and **1,615** class 1, with no class resampling. The other **226,008** development applicants were unused. Test class counts are **56,538 / 4,965**.

- Each of 28 model–level pairs used five random configurations with `RandomizedSearchCV(scoring="neg_log_loss", refit=True, random_state=42)`.
- Three common saved `StratifiedKFold` splits, `shuffle=True`, seed 42: train/validation sizes 13,333/6,667, 13,333/6,667 and 13,334/6,666.
- **28 full Pipelines, 420 outer CV fits and 28 refits**. Each pair was fitted once on all 20,000 applicants, without five repeated training samples. Including internal ensemble training, the count was 1,344 primitive fits.
- Sequential search used `n_jobs=1`, `pre_dispatch=1` and at most two model threads. Saved wall time including I/O and reporting was **10,640.14 s**, about 2 h 57 min. Refit time is already included in search time and is not added twice.
- Evaluation reused saved probabilities and **1,000 common paired bootstrap** samples of test applicants with `default_rng(42)`. The 95% percentile intervals are **pointwise, unadjusted and exploratory**, conditional on fixed models.
- Two-sided paired **DeLong ROC-AUC** tests compare six models with LR at Level 4, with **Bonferroni** correction within that family. This is not an AP test; no bootstrap p-values were calculated.

Sources: [actual protocol](../../reports/pilot_20k_5config/evaluation/actual_training_protocol.json), [software versions](../../docs/software_versions.json), [report](../../reports/pilot_20k_5config/evaluation/results_report.html).

## Preliminary findings

The values below come from the saved [RQ1](../../reports/pilot_20k_5config/evaluation/tables/rq1_level4.csv) and [RQ2](../../reports/pilot_20k_5config/evaluation/tables/rq2_baseline_gains.csv) tables, without recalculation.

- At Level 4, stacking has the highest point estimates of **AP 0.2603 [0.2493, 0.2727]** and **ROC-AUC 0.7707**. Soft voting has the lowest **log-loss 0.2426** and **Brier 0.0671**. These criteria do not establish one universally best model.
- The AP gain L4−L1 has an interval above zero for six models; the MLP interval includes zero. Adding Level 2 lowers MLP AP by **−0.0088 [−0.0145, −0.0031]**. The weak MLP result was retained without a causal diagnosis.
- Soft voting has the highest observed conditional profit in each specified `m=1`, `L=1,5,10` scenario. Profit uses observed TARGET, not probability substituted for outcome. Thresholds 0.5 and `m/(m+L)` were not optimized on test.
- At `t=1/11`, no model satisfies both observed gender and age approval-gap constraints of ≤5 percentage points. At `t=1/6`, only this archived MLP passes, which does not make it best in quality or profit. At `t=0.5`, RF approves everyone; zero fairness gaps do not imply useful default detection.

## Files and environment

`notebooks/` contains preparation, the A2 plan, pilot and evaluation; `src/` contains preprocessing, models and evaluation; `scripts/` contains execution/check/export tools; `tests/` checks preparation and evaluation logic. Aggregate results are under `reports/pilot_20k_5config/`, with article CSV/LaTeX in `evaluation/tables/` and PDF/SVG in `evaluation/figures/`. The original RQ3 handoff is documented in [SHAP_HANDOFF.md](../../docs/SHAP_HANDOFF.md). Local `data/`, `artifacts/`, `handoff/` and `tmp/` are excluded from Git.

Use **Python 3.13.9** from the repository root. Saved joblib models require compatible dependencies and an importable `src` package.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-reporting.txt
$env:HOME_CREDIT_DATA_DIR = 'D:/datasets/home-credit-default-risk'
```

Set `HOME_CREDIT_DATA_DIR` before running notebook 01; its original machine path is only a fallback. Notebooks resolve ROOT from the project root or `notebooks/`. Prepared data are expected at `data/processed/full` and pilot artifacts at `artifacts/training/pilot_20k_5config`. A Git clone supplies code and aggregate reports, not training data. The separate SHAP package provides the original fitted Pipelines and agreed inputs.

## Archived notebooks

1. `01_data_preparation.ipynb`: audits, preparation, four feature levels, fixed splits and additional quality checks. It need not be rerun to read results.
2. `02_model_training.ipynb`: original A2 plan and technical checks. The original full experiment was unfinished and was not the source of this pilot. Keep `RUN_FULL_EXPERIMENT` and `RUN_RESOURCE_PILOT` disabled when viewing reports.
3. `03_pilot_20k_5config.ipynb`: the completed pilot. Keep `RUN_PILOT=False` and run configuration plus the independent reporting cell to read existing results.
4. `04_pilot_results_analysis.ipynb`: saved-probability evaluation. Keep `RUN_ANALYSIS=False` when viewing tables/figures; configuration and final reporting do not retrain or bootstrap.

Do not rerun notebooks 01–03 for presentation changes. Publication checks found no individual customer records in their outputs. Aggregate results and scientific figures were preserved. Only 48 temporary widget outputs and associated metadata were removed from notebook 01. Local originals are in the excluded `local_private/publication_originals/2026-10-25`; checksums are in `docs/publication_audit.json`. Notebook 04 ran in a clean Python process after a stalled Jupyter-kernel attempt was stopped; this is not a claim that it executed in a Jupyter kernel. The actual method is recorded in metadata and the report.

Read the [HTML](../../reports/pilot_20k_5config/evaluation/results_report.html), [TXT](../../reports/pilot_20k_5config/evaluation/results_report.txt), [readiness](../../reports/pilot_20k_5config/evaluation/results_readiness.json), [tables](../../reports/pilot_20k_5config/evaluation/tables) and [figures](../../reports/pilot_20k_5config/evaluation/figures). Refreshing notebook reports requires local metadata; saved outputs and HTML are readable without it.

## Historical limitations and continuation

These were preliminary Assignment 3 results. The full A2 plan of 20 configurations and five repeated fits per pair was not complete at that stage: this pilot used 20k train, five configurations and one refit. Training variability was not estimated, and bootstrap intervals do not replace mean ± SD over five runs. Lack of a significant difference does not prove equivalence; the A2 AP non-inferiority selection rule was not confirmed.

Fairness correction and a fitted calibrator were deferred. Conditional profit is not a bank's monetary profit. Newly introduced analysis details were not presented as preregistered. Training CPU and total RAM were not logged, so current evaluation hardware is not substituted. LaTeX fragments require review in the article template ([usage notes](../../reports/pilot_20k_5config/evaluation/latex_usage.txt)).

The original next step was the separate SHAP/RQ3 handoff, containing all 28 Pipelines with one fit per pair. Publication and packaging do not run new training, search, bootstrap or SHAP. Maintainers use `python scripts/build_shap_handoff_package.py --prepare`, review and commit the publication files, then run the same script with `--finalize`. Preparation selects saved Parquet rows by ID; finalization links the package to an existing commit. Archives/intermediates stay in excluded `handoff/`; recipients do not rebuild them.
