# RQ3 — SHAP consistency and stability (Assignment 3)

Code for the third research question, run on top of the team package `pilot_20k_5config`
(28 saved Pipelines, handoff data). Nothing in the package is refitted; retrainings build new
estimators with `src.model_pipelines.make_estimator` and the saved best parameters.

## Layout

```
rq3/rq3_shap.py          select applicants/background, explain (fixed fits or refits)
rq3/rq3_analyze.py       tables (CSV + LaTeX), figures (PDF), error analysis
rq3/run_all.sh           the exact sequence that produced rq3/out/runs/*
rq3/requirements-rq3.txt package versions used (training versions + shap)
rq3/out/selection.json   sampling rule and seeds
rq3/out/explained_ids.parquet, background_ids.parquet
rq3/out/runs/<run>/<model>.npy     SHAP values (1000 applicants x 322 raw features), probability scale
rq3/out/runs/<run>/<model>.json    explainer, additivity check, refit seed, fit time, machine
rq3/out/tables/, rq3/out/figures/
```

Runs: `fixed` = saved package fits; `fixed_noise` = same fits, second explainer seed (noise floor);
`r1..r5` = retrainings with seeds 1–5 on a 90% stratified subsample of the 20k training set.

## Reproduce

```
python -m venv .venv && .venv/bin/pip install -r requirements.txt -r rq3/requirements-rq3.txt
python rq3/rq3_shap.py select
bash rq3/run_all.sh          # or the individual `explain` commands inside it
python rq3/rq3_analyze.py
```

## Method notes

* One explained output for all models: P(TARGET = 1).
* Exact interventional Tree SHAP (shap.TreeExplainer) for random forest, XGBoost, LightGBM, on each model's
  own transformed feature space, then summed back to the 322 raw features.
* Permutation Shapley estimator (own implementation, `perm_shap`) for LR, MLP, soft voting, stacking:
  for every applicant and permutation, raw features are switched on one by one in the permuted order
  starting from every background row (forward and reverse pass); the averaged marginal changes of
  P(default) are the SHAP values. Masking happens in the transformed space of each fitted preprocessor
  (value column + missing indicator + one-hot columns of a raw feature are replaced together), which
  is exactly equivalent to masking the raw feature because the preprocessing is column-wise.
  `ModelFn` reproduces `pipeline.predict_proba` to < 1e-5 before any explanation is computed, and
  additivity (sum of SHAP values + base = model output) is checked and logged for every run.
* Measures: Spearman correlation of global rankings (mean |SHAP|), share of top-5 reasons kept /
  shared; 95% percentile bootstrap over applicants (1,000 resamples, seed 42); paired Bonferroni-adjusted
  bootstrap for differences vs LR (6 comparisons).
