"""Build the independent saved-prediction evaluation notebook; never run training."""
from pathlib import Path

import nbformat as nbf


ROOT=Path(__file__).resolve().parents[1]
cells=[]
def md(text): cells.append(nbf.v4.new_markdown_cell(text))
def code(text,**metadata): cells.append(nbf.v4.new_code_cell(text,metadata=metadata))


md("""# 04. Preliminary RQ1/RQ2 evaluation: saved 20k pilot

This notebook analyzes **only saved test probabilities** from the 28 Pipelines of `pilot_20k_5config`. It performs no new `fit`, `predict`, hyperparameter search, calibration fitting, or fairness correction. X, splits, parameters, models, and predictions remain unchanged. The resource pilot is excluded from these comparisons.

The pilot training sample contains 20,000 development clients, with five configurations and one refit per pair. These are preliminary Assignment 3 results, not a replacement for full A2 with 20 configurations and five repeated fits. The remaining development applications were not used to train these models.

`RUN_ANALYSIS=False` by default. Explicit analysis execution preserves completed bootstrap checkpoints: completed replicates are not recomputed. To view existing materials, run only the configuration and final reporting cells.""")

code("""from pathlib import Path
import json,sys
import pandas as pd
from IPython.display import display,SVG

ROOT=Path.cwd()
if ROOT.name=='notebooks':
    ROOT=ROOT.parent
sys.path.insert(0,str(ROOT))

from src.pilot_evaluation import run_evaluation
from src.pilot_evaluation_reporting import export_evaluation_report

PREPARED=ROOT/'data/processed/full'
PILOT_ROOT=ROOT/'artifacts/training/pilot_20k_5config'
PREVIOUS_REPORTS=ROOT/'reports/pilot_20k_5config'
EVALUATION_DIR=PREVIOUS_REPORTS/'evaluation'
N_BOOTSTRAP=1000
SEED=42
RUN_ANALYSIS=False

print('Evaluating saved probabilities; no new fit/predict calls.')
print('Pilot:',PILOT_ROOT)
print('New results:',EVALUATION_DIR)
print('Analysis enabled:',RUN_ANALYSIS)""",tags=['evaluation_configuration'])

md("""## 1. Test-ID alignment and verification of existing evaluations

Probabilities for all 28 combinations are explicitly aligned with saved test IDs and TARGET. Checks verify unique and identical sets of 61,503 clients, the positive class `TARGET=1`, no NaN/inf, and the [0,1] range. The primary PR metric throughout is **PR-AUC (average precision, AP)**, calculated using `average_precision_score`.

AP, ROC-AUC, log-loss, and Brier are recomputed and compared with the existing `rq1_model_comparison.csv`. Discrepancies are not hidden: details are in `existing_metrics_comparison.csv`, and ID checks are in `alignment_validation.json`. Input hashes are saved before and after analysis. Raw CSVs are not audited again.

Precision with a zero denominator remains undefined. For example, a model predicting no defaults does not have precision=0. Recall and F1 may be zero when actual defaults are present; such zeros are retained.""")

md("""## 2. Shared paired bootstrap and comparisons

The analysis performs 1,000 replicates with `default_rng(42)`. The resampling unit is a test client. Identical client weights in each replicate are applied to every model and level. Weights and RNG state are saved per replicate, allowing analysis to resume from a checkpoint after interruption.

95% percentile intervals use the 0.025/0.975 quantiles. They are labeled **pointwise, unadjusted, exploratory** and reflect only test-set uncertainty for fixed fitted models. They do not include retraining uncertainty and are not the mean ± SD of five training runs.

AP comparisons include six models versus LR at Level 4; within each model, L2/L3/L4 versus L1 and adjacent L2−L1, L3−L2, L4−L3 comparisons. All differences use the shared bootstrap results. Bootstrap p-values are not computed.

ROC-AUC is separately assessed by paired two-sided DeLong tests: six comparisons with LR at Level 4, with Bonferroni adjustment within this family. DeLong does not test AP. A nonsignificant difference does not establish equivalence, and the A2 rule 'not worse than the best by AP' is not considered formally satisfied. These analysis details are documented now but are not claimed to be preregistered.""")

md("""## 3. Profit, fairness, calibration, and errors

Level 4 considers m=1 and L=1,5,10. The threshold is fixed at 0.5 or m/(m+L). Both coincide for L=1, so that result is saved once: five scenarios per model. Approval means `p_default < t`; predicted default means `p_default >= t`.

Profit across all applicants is `mean(approve * ((1-TARGET)*m - TARGET*L))`. Rejected applications yield zero profit. The observed TARGET is used, not the expected outcome from predicted probabilities. Approve-all/reject-all provide benchmarks. These are conditional units, not actual bank profit; thresholds are not optimized on test data.

Fairness groups are joined by ID: sex and age `<25` / `>=25`. Unknown groups are shown separately; they are excluded from binary gaps but retained in overall metrics and profit. TPR refers to detection of actual defaults (`TARGET=1`). Both observed **approval-rate gaps ≤5 percentage points** are checked separately for sex and age; TPR gap does not replace this constraint. The limit is not relaxed after inspecting results.

Calibration curves use ten fixed equal-width probability intervals; the final right endpoint 1 is included. Each bin's observation count is saved. No calibrator is fitted. Errors are saved for all 28 combinations at t=0.5 and for Level 4 at the profit thresholds.

MLP remains in the tables. Actual `n_iter_`, settings, and warnings are read from saved logs. Warnings from the entire search+refit cannot be attributed solely to the selected refit. Training loss_ and external log-loss are not identical; weak performance alone does not prove overfitting or a preprocessing error.""")

md("""## 4. Explicit analysis start or resume — without training

To run the analysis, set `RUN_ANALYSIS=True` in configuration and execute the next cell. It reads saved probabilities and resumes incomplete bootstrap replicates; completed replicates are not recomputed. Input checks, point summaries, and intervals may be refreshed from saved data and checkpoints. New files are created only in `reports/pilot_20k_5config/evaluation/`.""")

code("""if RUN_ANALYSIS:
    evaluation_result=run_evaluation(PREPARED,PILOT_ROOT,PREVIOUS_REPORTS,EVALUATION_DIR,
                                     n_bootstrap=N_BOOTSTRAP,seed=SEED)
    print('Analysis status:',evaluation_result.get('status'))
    print('Completed bootstrap replicates:',evaluation_result.get('completed_bootstrap'))
    print('New fit/predict calls:',evaluation_result.get('new_fit_calls'),evaluation_result.get('new_predict_calls'))
else:
    print('This cell did not run calculations: RUN_ANALYSIS=False.')
    print('Existing reports are available in the next cell.')""",tags=['evaluation_explicit_opt_in'])

md("""## 5. Independent report and article materials

This cell depends only on configuration and saved files; it does not require `evaluation_result`. It updates CSV/LaTeX and readable HTML/TXT from computed results. It does not run bootstrap, fit, predict, or create new PDFs.

Main tables cover RQ1 Level 4 performance with CIs, four-level AP and paired gains for RQ2, and profit/fairness under fixed scenarios. Details remain in the full CSV and appendix. Operational Status/reason fields are omitted from article tables. Table and figure captions, notebook explanations, and the readable report are in English.

PDF/SVG figures are saved through a separate controlled export and visually checked. This cell only displays existing SVGs. `results_readiness.json` distinguishes completed materials from specific pending checks. Missing training hardware documentation is stated as a limitation and is not replaced with information about the current machine.""")

code("""readiness=export_evaluation_report(PILOT_ROOT,EVALUATION_DIR,previous_reports=PREVIOUS_REPORTS,
                                   create_figures=False)
print('Materials status:',readiness.get('status'))
if readiness.get('missing_tables'):
    print('Analysis tables not yet available:',', '.join(readiness['missing_tables']))
else:
    print('Bootstrap replicates:',readiness.get('completed_bootstrap'))
    print('Ready for the preliminary article section:',readiness.get('ready_for_article'))
    for message in readiness.get('blockers',[]):
        print('Remaining:',message)
    for name,title in [('rq1_level4','RQ1: Level 4 performance'),
                       ('rq2_ap_levels','RQ2: AP across four levels'),
                       ('rq2_baseline_gains','RQ2: paired AP gains versus L1'),
                       ('profit_fairness_L5','Fixed scenario L/m=5; other scenarios in the full report'),
                       ('errors_level4','Level 4 errors at t=0.5')]:
        path=EVALUATION_DIR/'tables'/f'{name}.csv'
        if path.exists():
            print(title)
            display(pd.read_csv(path).fillna('—'))
    mlp_path=EVALUATION_DIR/'mlp_actual_diagnostics.csv'
    if mlp_path.exists():
        print('MLP: actual epochs and warnings from the entire search+refit')
        mlp=pd.read_csv(mlp_path)
        display(mlp[['level','n_iter_','max_iter','loss_curve_points','warning_count_search_and_refit']])
    for name in ['ap_by_level','calibration_level4']:
        path=EVALUATION_DIR/'figures'/f'{name}.svg'
        if path.exists():
            display(SVG(filename=str(path)))
    print('Readable report:',EVALUATION_DIR/'results_report.html')
    print('Text report:',EVALUATION_DIR/'results_report.txt')
    print('Readiness and limitations:',EVALUATION_DIR/'results_readiness.json')
    print('CSV/LaTeX tables:',EVALUATION_DIR/'tables')
    print('PDF/SVG figures:',EVALUATION_DIR/'figures')
print('This notebook performs no new training or predict calls.')""",tags=['evaluation_disk_reports_independent'])

notebook=nbf.v4.new_notebook(cells=cells,metadata={
    'kernelspec':{'display_name':'Python 3 (ipykernel)','language':'python','name':'python3'},
    'language_info':{'name':'python'},'research_scope':'pilot_20k_5config_evaluation'})
destination=ROOT/'notebooks/04_pilot_results_analysis.ipynb'
nbf.write(notebook,destination)
print(f'Created {destination}; RUN_ANALYSIS=False; no calculations executed.')
