"""Create a separate research-pilot notebook: 20k clients / 5 configurations.

The builder only writes the notebook; training requires a separate explicit run.
The existing notebooks/02_model_training.ipynb is unchanged.
"""
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(source))


def code(source, **metadata):
    cells.append(nbf.v4.new_code_cell(source, metadata=metadata))


md("""# 03. Research pilot: 20,000 applications, 5 configurations

This notebook describes a separate experiment with scope **`pilot_20k_5config`**. It uses the prepared features at four levels and one saved stratified development subsample. The original X, development/test split, five main training subsamples, and full-study CV folds remain unchanged.

Each of the 28 model–level pairs runs one RandomizedSearchCV: **5 configurations × 3 CV folds**, selected by `neg_log_loss`, followed by **one automatic refit** of the best Pipeline on the entire pilot training sample. Successful execution without retries produces 420 outer CV fits and 28 final refits. Internal voting/stacking fits are counted separately: 1,344 primitive fits in total.

The full Assignment 2 plan—28 searches with 20 configurations and 140 main fits—is retained separately in notebook 02. This pilot does not include five repeated fits per pair. It provides preliminary comparisons but does not measure stability across five training subsamples.

`RUN_PILOT=False` by default. Creating the notebook, initializing IDs/CV folds, and reading reports do not start training. To read existing results, run only the configuration and final reporting cells.""")

code("""from pathlib import Path
import json, sys
import pandas as pd
from IPython.display import display

ROOT=Path.cwd()
if ROOT.name=='notebooks':
    ROOT=ROOT.parent
sys.path.insert(0,str(ROOT))

from src.research_pilot import initialize_pilot, run_research_pilot
from src.pilot_reporting import export_pilot_reports
from src.model_pipelines import experiment_spec

PREPARED=ROOT/'data/processed/full'
PILOT_ROOT=ROOT/'artifacts/training/pilot_20k_5config'
REPORT_DIR=ROOT/'reports/pilot_20k_5config'
SCOPE='pilot_20k_5config'
SAMPLE_N=20000
THREADS=2
RUN_PILOT=False  # Set True only to explicitly start or resume this pilot.

print('Scope:',SCOPE)
print('Artifacts:',PILOT_ROOT)
print('Reports:',REPORT_DIR)
print('Training enabled:',RUN_PILOT)""", tags=['pilot_configuration'])

md("""## 1. One shared pilot sample and its own CV folds

`initialize_pilot` creates and saves a stratified sample of 20,000 development IDs shared by all models and levels. Its own `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)` is created once; assignments are saved in `cv_folds.parquet` inside the pilot directory. Resumed runs reuse the saved IDs and folds.

These are new pilot CV folds; assignments for the full development set are not overwritten. Test data are not used to select the pilot sample or create folds. This stage reads schemas and IDs and verifies split membership; it does not fit models.

The next cell displays the actual sample composition from `cohort_summary.json`. Values are not inferred from the planned sample size or replaced by expected counts.""")

code("""pilot_context=initialize_pilot(PREPARED,PILOT_ROOT,threads=THREADS,sample_n=SAMPLE_N)
cohort_path=PILOT_ROOT/'cohort_summary.json'
with cohort_path.open(encoding='utf-8') as stream:
    cohort_summary=json.load(stream)
print('Pilot sample size:',cohort_summary['sample_n'])
print('Stratified selection seed:',cohort_summary['random_state'])
print('Pilot TARGET balance:')
display(pd.DataFrame(cohort_summary['cohort_balance']))
print('Shared pilot CV folds:')
display(pd.DataFrame([{key:fold[key] for key in ['fold','train_rows','validation_rows']}
                      for fold in cohort_summary['cv']]))
print('Manifest:',PILOT_ROOT/'experiment_manifest.json')
print('Saved pilot CV folds:',PILOT_ROOT/'cv_folds.parquet')""", tags=['pilot_initialize_without_fit'])

md("""## 2. Execution order and budget

The four Logistic Regression levels run first, followed by the four levels of LightGBM, XGBoost, soft voting, stacking, MLP, and Random Forest. Pairs run sequentially in separate worker processes. Faster model families run first; the order is independent of test results.

Complete Pipelines and predefined hyperparameter spaces are saved in the manifest. Each search receives the entire unfitted Pipeline. `refit=True` fits the selected Pipeline once after CV; no separate manual `fit` is needed after the search.

The counts below describe successful execution. Retrying an interrupted search increases the actual number of attempts; saved events and reports must make this visible.""")

code("""MODEL_ORDER=['logistic_regression','lightgbm','xgboost','soft_voting','stacking','mlp','random_forest']
plan_rows=[]
for model in MODEL_ORDER:
    primitive_per_fit=experiment_spec(model)['primitive_fits_per_outer_fit']
    plan_rows.append({'model':model,'levels':4,'searches':4,'configurations_per_search':5,
        'cv_folds':3,'candidate_cv_fits':4*5*3,'final_refits':4,
        'primitive_fits_including_ensembles':4*(5*3+1)*primitive_per_fit})
pilot_plan=pd.DataFrame(plan_rows)
display(pilot_plan)
assert int(pilot_plan['searches'].sum())==28
assert int(pilot_plan['candidate_cv_fits'].sum())==420
assert int(pilot_plan['final_refits'].sum())==28
assert int(pilot_plan['primitive_fits_including_ensembles'].sum())==1344
print('Total: 28 searches, 420 CV fits, 28 automatic refits; 1 344 primitive fits.')""")

md("""## 3. Preprocessing and final evaluation

Numeric medians, handling of entirely missing training columns, missingness indicators, categorical imputation, OneHotEncoder, and required scaling are included in the Pipeline. During every CV fit, they are fitted only on the training fold. Validation and test data use the saved statistics; unknown categories are handled without refitting.

For stacking, every base model includes its own preprocessing during internal OOF fits as well. Parameters are selected solely by mean CV log-loss. TARGET and IDs are not input features; sex and age remain outside X, as specified in A2 preparation.

After search, the fitted `best_estimator_` is saved **before accessing test data for prediction**. Final probabilities and metrics use the existing test set; they are not used to select parameters, processing rules, or the classification threshold. Pilot comparisons must account for the single training sample and limited five-configuration search.

**RQ1 in A2 concerns model selection through joint assessment of accuracy, profit, and fairness.** The pilot provides descriptive model comparisons, Brier score, and calibration frequencies. Predefined profit and group-disparity scenarios are additionally computed for Level 4 from saved probabilities. These require no new training or `predict` calls.

Normalization: revenue from a repaid loan is `m=1`, and the default loss ratio `L/m` is **1, 5, or 10**. For each ratio, the fixed threshold **0.5** is compared with the break-even threshold **`1/(1+L/m)`**. An application is approved strictly when `p(default)<t`; equality results in rejection. Mean realized test profit is calculated across all applicants: `mean(approve * ((1-TARGET) - (L/m)*TARGET))`. These are conditional units under stated assumptions, not the lender's actual monetary returns. At `L/m=1`, both rules intentionally coincide.

Fairness groups come from the saved file: sex F/M and age **`<25` / `>=25`**, so exactly 25 years belongs to the older group. A2 does not separately specify equality at the boundary; the existing preparation rule is retained. Unknown groups are shown separately and excluded from binary gaps. Approval-rate gap and TPR gap are absolute differences in rates. The positive class is default (`TARGET=1`), so TPR is the fraction of actual group defaults detected. A metric with a zero denominator remains undefined. The calibration table uses 10 fixed probability intervals without fitting a calibrator.

All seven Level 4 models are shown without an accuracy eligibility filter. The full A2 rule first checks accuracy and then selects profit subject to fairness; this selection **is not performed** in the pilot. The final winner, significance tests, bootstrap CIs, DeLong, Bonferroni, and fairness corrections are deferred to the subsequent full analysis.

**RQ2 concerns the value of credit and payment history and the models' ability to use it.** It is addressed by changes in each model's metrics when history is added, primarily Levels 2–4 versus application-only Level 1. Adjacent levels provide additional descriptive comparisons. A gain from one pilot run does not establish significance or stability across repeated fits. SHAP and RQ3 are outside this pilot.""")

md("""## 4. Explicit pilot start / resume

To start from a clean kernel, set `RUN_PILOT=True` in the configuration cell, then execute the configuration and the cell below. The controller also initializes data if needed. This call does not start the full experiment in notebook 02.

Search and evaluation checkpoints are separate. The fitted Pipeline, selected parameters, CV results, and training diagnostics are saved first. Prediction and export follow. If prediction or export is interrupted, the saved model is loaded without another fit. If the search itself is interrupted before its checkpoint, the same search is repeated; completed pairs are skipped after saved-artifact validation.

Use the same `PILOT_ROOT` to resume. Incompatible data, parameters, or code must not be mixed in one directory. Existing results are preserved, and run status is determined from artifacts on disk.""")

code("""if RUN_PILOT:
    pilot_run_summary=run_research_pilot(PREPARED,PILOT_ROOT,REPORT_DIR,threads=THREADS)
    display(pilot_run_summary)
else:
    print('This cell did not start training: RUN_PILOT=False.')
    print('To start/resume: change the configuration flag and run this cell.')
print('Pilot artifacts:',PILOT_ROOT)""", tags=['research_pilot_explicit_opt_in'])

md("""## 5. Independent summary of saved results

The cell below uses only configuration paths and files on disk. It does not require `pilot_context`, `cohort_summary`, `pilot_plan`, or `pilot_run_summary` in memory and can run after configuration from a clean kernel. It does not load Pipelines for training or start any fit.

`metrics.csv` contains actual completed evaluations; `metrics.tex` is intended for the report. `report_metadata.json` records scope and completeness. The displays below provide a compact summary, preliminary RQ1 comparisons at Level 4, and RQ2 across four levels. Values and precomputed differences are read from CSV; metrics and predictions are not recomputed here. Missing pairs remain incomplete and appear as missing values.

For RQ1, AP, ROC-AUC, log-loss, and Brier describe predictive accuracy and probability quality only. Profit and fairness scenarios are saved separately in `decision_scenarios.csv` and `decision_groups.csv`; these results do not select a core model. For RQ2, lower log-loss and higher AP indicate better performance on this test set. One refit per pair does not support a significance claim. Completed MLP runs display actual `n_iter_`, `loss_curve_` length, and saved warnings.

Large model, probability, and checkpoint artifacts are stored in `artifacts/`, which is already excluded from Git. Small reporting tables are in `reports/pilot_20k_5config/`.""")

code("""if (PILOT_ROOT/'experiment_manifest.json').exists():
    pilot_disk_report=export_pilot_reports(PILOT_ROOT,REPORT_DIR)
    metadata_path=REPORT_DIR/'report_metadata.json'
    if metadata_path.exists():
        with metadata_path.open(encoding='utf-8') as stream:
            report_metadata=json.load(stream)
    else:
        report_metadata=pilot_disk_report
    status_fields={'scope':'Scope','status':'Status','completed_pairs':'Completed pairs',
                   'pending_pairs':'Pending','invalid_pairs':'Invalid'}
    display(pd.DataFrame([{title:report_metadata.get(key) for key,title in status_fields.items()}]))

    def read_report_csv(name):
        path=REPORT_DIR/name
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    comparison=read_report_csv('rq1_model_comparison.csv')
    saved_metrics=read_report_csv('metrics.csv')
    print('RQ1 — Level 4: preliminary performance comparison; no core-model selection.')
    if not comparison.empty:
        level4=comparison.loc[comparison['level']==4].copy()
        if not saved_metrics.empty and 'search_and_refit_seconds' in saved_metrics:
            runtime=saved_metrics.loc[saved_metrics['level']==4,['model','search_and_refit_seconds']]
            level4=level4.merge(runtime,on='model',how='left',validate='one_to_one')
        rq1_labels={'model_label':'Model','status':'Status','test_pr_auc_average_precision':'AP',
            'test_roc_auc':'ROC-AUC','test_log_loss':'Log-loss','test_brier_score':'Brier',
            'search_and_refit_seconds':'Search + refit, s'}
        rq1_table=level4[[name for name in rq1_labels if name in level4]].rename(columns=rq1_labels)
        formats={name:'{:.4f}' for name in ['AP','ROC-AUC','Log-loss','Brier'] if name in rq1_table}
        if 'Search + refit, s' in rq1_table:
            formats['Search + refit, s']='{:.1f}'
        display(rq1_table.style.format(formats,na_rep='—'))
    else:
        print('No saved RQ1 table yet.')

    history=read_report_csv('rq2_history_gains.csv')
    print('RQ2 — comparison of four levels; Δ L4−L1 differences are already saved in CSV.')
    if not history.empty:
        model_order=history['model_label'].drop_duplicates().tolist()
        for metric,delta,title in [
            ('test_log_loss','delta_log_loss_vs_level1','Log-loss: lower is better'),
            ('test_pr_auc_average_precision','delta_pr_auc_average_precision_vs_level1','AP: higher is better')]:
            table=history.pivot(index='model_label',columns='level',values=metric)
            table=table.reindex(index=model_order,columns=[1,2,3,4])
            table.columns=['Level 1','Level 2','Level 3','Level 4']
            table['Δ L4−L1']=history.loc[history['level']==4].set_index('model_label')[delta]
            table.index.name='Model'
            print(title)
            display(table.style.format('{:.4f}',na_rep='—'))
    else:
        print('No saved RQ2 table yet.')

    mlp_diagnostics=read_report_csv('mlp_diagnostics.csv')
    if not mlp_diagnostics.empty:
        print('MLP — actual epochs and saved training curve:')
        diagnostic_labels={'model':'Model','level':'Level','n_iter_':'Actual epochs',
            'max_iter':'Epoch limit','loss_curve_points':'loss_curve_ points',
            'loss_curve_length_matches_n_iter':'Curve length = n_iter_'}
        display(mlp_diagnostics[[name for name in diagnostic_labels if name in mlp_diagnostics]]
                .rename(columns=diagnostic_labels))
    else:
        print('No saved diagnostics for completed MLP runs yet.')
    saved_warnings=read_report_csv('warnings.csv')
    mlp_warnings=saved_warnings.loc[saved_warnings['model']=='mlp'] if not saved_warnings.empty else pd.DataFrame()
    if not mlp_warnings.empty:
        print('Saved MLP warnings (full text in warnings.csv):')
        with pd.option_context('display.max_colwidth',140):
            display(mlp_warnings[['level','stage','category','count','message']])
    elif not mlp_diagnostics.empty:
        print('No MLP warnings in the saved warnings.csv.')
    print('One refit per pair: no winner or statistical significance is established here.')
    print('Saved report files:')
    for report_path in sorted(REPORT_DIR.glob('*')):
        if report_path.is_file():
            print(report_path)
else:
    print('The pilot is not initialized yet; no report results are available.')""", tags=['pilot_disk_reports_independent'])

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3 (ipykernel)', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'},
    'research_scope': 'pilot_20k_5config',
})
destination = ROOT/'notebooks/03_pilot_20k_5config.ipynb'
nbf.write(notebook, destination)
print(f'Created {destination}; {len(cells)} cells; RUN_PILOT=False')
