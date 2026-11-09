# RQ1/RQ2 to SHAP/RQ3 handoff

> **November RQ3:** the [current report and code](../rq3_november/README.md) use seven Level 4 models from the [official bundle](../model_bundles/README.md). Every run uses the same 1,000 test applicants and 100 training-background applicants. [Model checksums](../rq3_november/verification/BUNDLE.json). The instructions below document the archived October five-configuration experiment.

## Scope of the archived handoff

Data preparation, `pilot_20k_5config` and preliminary RQ1/RQ2 evaluation were complete. Seven model types × four levels produced 28 full fitted Pipelines, each selected from five configurations on three common CV folds and automatically refitted once on all 20,000 pilot applicants. The common test contains 61,503 applicants. Metrics, paired bootstrap intervals, DeLong, profit, fairness, calibration and errors are in `reports/pilot_20k_5config/evaluation/`.

This archived handoff contains one final fit per pair. Cross-model explanations can be compared, but retraining stability was not measured at that stage. Repeating an explainer is not retraining a model. The older resource pilot on 221,407 applicants is separate. Packaging performs no new training, parameter search, bootstrap or SHAP.

## Obtaining the package

Obtain the matching Git revision and `handoff/pilot_20k_5config_shap.zip` from the team. The archive is excluded from Git and is not uploaded automatically. Extract it into a separate directory containing `src/`, `docs/`, `requirements.txt` and `package_manifest.json`.

`package_manifest.json` records relative paths, sizes, SHA-256 values and the source commit. Keep it with subsequent SHAP results. Load only the agreed project bundle, since `joblib` restores Python objects. Maintainers build it using `python scripts/build_shap_handoff_package.py --prepare`, commit the publication files, then run `python scripts/build_shap_handoff_package.py --finalize`. Recipients do not need to rebuild it.

## Environment

Use Python **3.13.9**, the versions in `requirements.txt` and `docs/software_versions.json`, plus `requirements-reporting.txt` to view reports and figures.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-reporting.txt
```

SHAP had not been selected or installed at this archived handoff stage. Record the explainer version, settings, masking method and output scale in the RQ3 environment. Do not silently replace saved sklearn/XGBoost/LightGBM versions to install SHAP. Verify saved-model loading and do not fit preprocessing again for explanations.

## Package paths

Let `P = artifacts/training/pilot_20k_5config` and `D = data/handoff/pilot_20k_5config`. Paths are relative to the extracted package root.

| File | Contents |
|---|---|
| `P/<model>/level_<N>/fit_checkpoint/pipeline.joblib` | Complete fitted Pipeline; all 28 pairs are included |
| `P/<model>/level_<N>/fit_checkpoint/best_params.json` | Selected parameters and CV log-loss |
| `P/<model>/level_<N>/fit_checkpoint/fit_diagnostics.json` | Fitted attributes and shapes; actual MLP `n_iter_`, `loss_curve_` and settings |
| `P/<model>/level_<N>/fit_checkpoint/warnings.json` | Search/refit warnings, including CV candidates |
| `P/<model>/level_<N>/fit_checkpoint/cv_results.parquet` | Saved results for five configurations |
| `P/<model>/level_<N>/probabilities.parquet` | `SK_ID_CURR`, `TARGET`, `probability_default`, scope/model/level/split |
| `P/cohort_ids.parquet`, `P/cohort_targets.parquet` | Original 20,000 pilot IDs and TARGET |
| `P/test_ids.parquet`, `P/cv_folds.parquet` | Original test IDs and three pilot CV folds |
| `P/experiment_manifest.json`, `P/cohort_contract.json` | Protocol and input checksums |
| `P/source_snapshot/` | Snapshot of the five core training modules |
| `D/X_train_level4.parquet`, `D/X_test_level4.parquet` | Prepared, untransformed Level 4 features; **SK_ID_CURR is a service column, not a model input** |
| `D/train_ids.parquet`, `D/test_ids.parquet` | Train/test row order matching the saved pilot IDs |
| `D/target_train.parquet`, `D/target_test.parquet` | Separate TARGET with `SK_ID_CURR` |
| `D/fairness_train.parquet`, `D/fairness_test.parquet` | Separate `CODE_GENDER`, `DAYS_BIRTH`, `AGE_YEARS`, `AGE_GROUP`, `GENDER_GROUP` and IDs |
| `D/features_level_1.json` … `D/features_level_4.json` | Exact input names and order for each level |
| `D/feature_levels.json`, `D/feature_schema.json` | Nested levels, data schema and column order |
| `D/model_schema.json` | Verified transformed feature names for every fitted Pipeline, separately for ensemble bases |
| `D/cv_folds.parquet` | Copy of the saved pilot CV folds |

Model names are `logistic_regression`, `random_forest`, `xgboost`, `lightgbm`, `mlp`, `soft_voting`, `stacking`; levels are 1–4. Level 4 contains all earlier-level features, so select the saved column list instead of storing separate large X files for Levels 1–3. Full development is not included; the 20,000 training applicants provide the train-only background pool.

## Read models and align IDs without fitting

The code below reads files, joins keys and inspects saved attributes. It does not call fit, predict or SHAP. Run it from the package root so `src.model_pipelines` is importable during `joblib.load`.

```python
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

ROOT = Path.cwd()
P = ROOT / "artifacts/training/pilot_20k_5config"
D = ROOT / "data/handoff/pilot_20k_5config"
model, level = "lightgbm", 4
pair = P / model / f"level_{level}"
pipeline = joblib.load(pair / "fit_checkpoint/pipeline.joblib")
features = json.loads((D / f"features_level_{level}.json").read_text(encoding="utf-8"))

test_ids = pd.read_parquet(D / "test_ids.parquet")
test_raw = pd.read_parquet(D / "X_test_level4.parquet")
target = pd.read_parquet(D / "target_test.parquet")
saved = pd.read_parquet(pair / "probabilities.parquet")
assert test_ids.SK_ID_CURR.is_unique and len(test_ids) == 61503
assert test_raw.SK_ID_CURR.is_unique and target.SK_ID_CURR.is_unique
assert set(test_raw.SK_ID_CURR) == set(target.SK_ID_CURR) == set(test_ids.SK_ID_CURR)
assert saved.SK_ID_CURR.is_unique and set(saved.SK_ID_CURR) == set(test_ids.SK_ID_CURR)
aligned = test_ids.merge(test_raw, on="SK_ID_CURR", how="left", validate="one_to_one", sort=False)
aligned_target = test_ids.merge(target, on="SK_ID_CURR", how="left", validate="one_to_one", sort=False)
aligned_prob = test_ids.merge(saved, on="SK_ID_CURR", how="left", validate="one_to_one", sort=False)
assert np.array_equal(aligned.SK_ID_CURR, test_ids.SK_ID_CURR)
assert aligned_target.TARGET.notna().all()
assert np.array_equal(aligned_target.TARGET, aligned_prob.TARGET)
assert aligned_prob.probability_default.between(0, 1).all()
X_test = aligned.loc[:, features]  # exact saved order; no ID/TARGET/fairness columns
y_test = aligned_target.TARGET
assert list(X_test.columns) == features
assert not {"SK_ID_CURR", "TARGET", "CODE_GENDER", "DAYS_BIRTH", "AGE_YEARS"} & set(features)
classes = np.asarray(pipeline.classes_)
positive_column = int(np.flatnonzero(classes == 1)[0])
print(model, level, X_test.shape, "TARGET=1 column:", positive_column)
```

Saved `probability_default` is **P(TARGET=1)**. An explainer's prediction callable must select the positive column through `classes_`. RQ1/RQ2 predicts default for `p_default >= t` and approves for `p_default < t`.

## Transformed features and ensembles

For the five standalone model types, fitted preprocessing is in `pipeline.named_steps["preprocess"]`:

```python
preprocess = pipeline.named_steps["preprocess"]
transformed_names = preprocess.get_feature_names_out().tolist()
assert list(preprocess.feature_names_in_) == features
# If needed, transform existing rows only; do not use fit_transform:
# transformed = preprocess.transform(X_test.loc[:, features])
```

`transformed_names` follows the saved ColumnTransformer: numeric values, original missingness indicators and encoded categories. Do not fit a new encoder or infer its category vocabulary from test. An entirely empty training column is retained with fallback 0 and an indicator; the fitted encoder handles unknown categories. LR/MLP use fitted scaling, while XGBoost receives dense float32 with explicit zeros. Use each specific model's own fitted transforms and names.

Package validation loaded all 28 Pipelines without fit, transform or predict and checked classes `[0, 1]`, exact raw feature order and transformed names. Levels 1–4 have 359/467/639/755 transformed features and 124/178/264/322 original features; full names are in `D/model_schema.json`.

Soft voting and stacking have no shared outer preprocessor. Each fitted base Pipeline has its own:

```python
ensemble = pipeline.named_steps["model"]
base_feature_names = {
    name: base.named_steps["preprocess"].get_feature_names_out().tolist()
    for name, base in ensemble.named_estimators_.items()
}
# Keys: lr, xgb, lgbm. These are fitted named_estimators_, not unfitted estimators templates.
```

An explanation of one base is not an explanation of the full ensemble. Stacking's meta-model is trained on three OOF base probabilities (`passthrough=False`), not on 322 raw features. Explaining those three predictions differs from explaining the entire stacking function in the original feature space. Document aggregation of OHE/indicator contributions back to original fields using actual names, not positions borrowed from another model.

## History availability and fairness

Level 4 indicators are `HAS_BUREAU`, `HAS_BB`, `HAS_POS`, `HAS_CC`, `HAS_INSTALLMENTS`, `HAS_PREVIOUS`. They correspond to `BUREAU_RECORD_COUNT`, `BB_RECORD_COUNT`, `POS_RECORD_COUNT`, `CC_RECORD_COUNT`, `INSTALLMENTS_RECORD_COUNT`, `PREVIOUS_RECORD_COUNT` and source aggregates. `APP_EMPLOYMENT_DATE_UNKNOWN` flags the processed employment-date sentinel. Missing history does not mean observed zero debt/payment; missing values and indicators remain separate.

Fairness columns were excluded from X. Join them by ID for group analysis without adding them to model inputs. The test has two applicants with unknown gender and none with unknown age. Unknown groups remain separate and contribute to overall metrics, but not to the F/M gap. Age exactly 25 belongs to `>=25`.

## Consistent SHAP protocol

1. Select and save the same test IDs once for all compared models/levels. Preserve order, join by ID, and record both the seed and sampling rule.
2. Select background from `D/train_ids.parquet` / `D/X_train_level4.parquet`, never test. Use the same saved background IDs and seed where explanations are comparable; document explainer-specific differences.
3. Specify the explained output: default probability or default log-odds, link/output, baseline and masking method. Do not silently compare probability SHAP with log-odds SHAP.
4. Save explainer type/version, settings, seed, explained/background IDs, feature names, expected value, model output and additivity check. Reuse fitted preprocessing; this explanation stage does not refit it.
5. Interpret results as cross-model explanations of fixed fits. Retraining stability requires a separate training protocol; repeated SHAP runs do not measure it.

## Archived MLP and study limitations

MLP settings and actual `n_iter_` / `loss_curve_` are in `P/mlp/level_<N>/fit_checkpoint/fit_diagnostics.json`; warnings are in `warnings.json`. The public summary is `reports/pilot_20k_5config/evaluation/mlp_actual_diagnostics.csv`. Actual epochs at Levels 1–4 were **300, 300, 300, 243**. Candidate CV warnings do not prove the selected refit reached max_iter, and training loss is not external log-loss. The archived weak MLP result alone did not establish overfitting or a preprocessing bug.

That preliminary RQ1/RQ2 analysis used 20k training applicants, five configurations and one refit instead of the full A2 protocol. Its exploratory intervals exclude training variability; fairness correction and a fitted calibrator were deferred. The full A2 model-selection rule was not established, and training CPU/RAM were not recorded. Article integration required RQ3, consistent SHAP scales and a LaTeX layout check. Preserve the original RQ1/RQ2 cohort and results when adding explanations.
