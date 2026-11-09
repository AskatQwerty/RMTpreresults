"""Create 02_model_training.ipynb: the initial execution is strictly technical."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
cells=[]
def md(source): cells.append(nbf.v4.new_markdown_cell(source))
def code(source,**metadata): cells.append(nbf.v4.new_code_cell(source,metadata=metadata))

md("""# 02. Model training: Pipeline and the A2 protocol

Use prepared `data/processed/full/`, four feature lists, development/test, five 90% training subsamples and saved CV folds. **X and splits are not recreated.** Gender and age remain in the external fairness file; ID serves only as the index. TARGET is passed separately.

The technical run, resource pilot and full experiment are separate. The full experiment is disabled by `RUN_FULL_EXPERIMENT=False`; the resource pilot has a separate `RUN_RESOURCE_PILOT` flag. Technical metrics neither estimate full-experiment quality nor determine its hyperparameters. Final summaries read saved artifacts and work after configuration in a clean kernel even when technical sections are skipped.

Implementation: `src/fold_preprocessing.py` handles numeric missingness; `src/model_pipelines.py` defines the seven complete Pipelines; `src/model_training.py` handles ID-based loading, search, repetitions, evaluation and persistence. Dependencies are pinned in `requirements.txt`.""")

code("""from pathlib import Path
import sys, json, time, importlib.metadata
import pandas as pd
import psutil
from IPython.display import display

ROOT=Path.cwd()
if ROOT.name=='notebooks': ROOT=ROOT.parent
sys.path.insert(0,str(ROOT))
from src.model_training import (PreparedData, full_experiment_plan, run_technical,
    run_full_experiment, atomic_json, sha256, PACKAGES)
from src.model_pipelines import MODEL_NAMES, make_estimator, parameter_space, experiment_spec
from src.data_quality import artifact_manifest
from src.training_reporting import write_full_report, write_technical_reports, write_resource_reports
from src.resource_pilot import run_resource_pilot

PREPARED=ROOT/'data/processed/full'
REPORTS=ROOT/'reports/training'
TECHNICAL_PARENT=ROOT/'artifacts/training/technical'
FULL_OUTPUT=ROOT/'artifacts/training/a2_full_v2'
RESOURCE_PARENT=ROOT/'artifacts/training/resource_pilot'
THREADS=2
TECHNICAL_TRAIN_N=1200
TECHNICAL_VALIDATION_N=400
RUN_FULL_EXPERIMENT=False  # Set True only to deliberately start the full long-running experiment.
RUN_RESOURCE_PILOT=False  # Separate 7 fits with full settings; not the main experiment.
REPORTS.mkdir(parents=True,exist_ok=True)
NOTEBOOK_STARTED=time.perf_counter()
print('Python:',sys.version.split()[0])
print('Versions:',{name:importlib.metadata.version(name) for name in PACKAGES})
print('Available RAM, GiB:',round(psutil.virtual_memory().available/1024**3,2))
print('Full experiment enabled:',RUN_FULL_EXPERIMENT)""")

md("""## 1. Data contract and existing splits

Loading checks IDs, TARGET order, uniqueness, disjoint development/test, repetition membership in development, CV coverage and nested feature lists. Numeric/categorical columns are identified **from the saved schema**, without learning categories from the full population.

`PreparedData.load()` scans Parquet in batches and returns only requested IDs. A physical file block may contain multiple splits; only required rows are retained before DataFrame/fit/score. Assertions prohibit test-ID requests in technical mode. Input SHA-256 hashes are recorded before work and compared afterward.""")
code("""data=PreparedData(PREPARED)
contract=data.contract_report()
frozen_before=artifact_manifest(PREPARED)
atomic_json(REPORTS/'input_contract.json',contract)
atomic_json(REPORTS/'prepared_sha256_before.json',frozen_before)
display(pd.DataFrame(contract['levels']))
display(pd.DataFrame(contract['cv']))
display(pd.DataFrame(contract['repeats']))
print('Development:',contract['development'],'test:',contract['test'])
print('Disjointness, ID/TARGET alignment, nesting and saved splits: verified.')""")

md("""## 2. Full preprocessing inside CV

**Numeric features.** The existing `make_numeric_pipeline` fits medians only on the current training fold/repetition; fully empty training columns remain present with a fixed 0 fallback. Every numeric column receives an original-missingness indicator, including new missingness in validation. Validation/test values are not transformed using refitted statistics.

**Categories.** Deterministically convert nonmissing categories to strings and None/pd.NA to NaN → `SimpleImputer(strategy='constant', fill_value='__RMT_MISSING__', keep_empty_features=True)` → `OneHotEncoder(handle_unknown='ignore')`. This also handles the actual Boolean `EMERGENCYSTATE_MODE` column in saved Parquet: Boolean values and a string missing marker are not mixed within an OHE column. Encoder categories are learned only from the current train. Unknown categories receive a zero vector for their OHE block; fully empty training columns do not break the schema.

**Scaling.** StandardScaler is included only in the numeric LR and MLP branches; it is fitted on the current train and does not scale indicators. Each ensemble scales its own LR base.

**Format and memory.** float32 is used; LR, RF, LightGBM and MLP receive CSR, while XGBoost receives dense float32 because it treats absent CSR entries as missing whereas numeric 0 must remain explicit. See the [official XGBoost explanation](https://xgboost.readthedocs.io/en/stable/faq.html#why-do-i-see-different-results-with-sparse-and-dense-data). The dense XGB matrix uses `n_rows × n_encoded_features × 4` bytes plus working copies. Search and ensembles run sequentially (`n_jobs=1`); libraries/BLAS are limited to two threads. RAM measurements for preselected full configurations are in the section 7 resource pilot and section 8 summary; they do not establish memory use for all candidates. RF search was predefined with depth at most 24 and at least 5 observations per leaf.

**Stacking.** Voting and stacking have fixed LR + XGBoost + LightGBM bases. Every base has its own full Pipeline. Stacking has no shared imputer/encoder before the ensemble: every internal OOF fit learns preprocessing afresh on its internal train. The meta-model is L2 LR on three OOF probabilities, with `passthrough=False` and internal stratified 3-fold CV. After OOF, base Pipelines are fitted separately on the full training partition of that outer fit. Saved outer validation/test data enter none of these fits.

No `fit_transform` is applied to full development before CV. The complete estimator Pipeline is passed to RandomizedSearchCV.""")
code("""specs=pd.DataFrame([experiment_spec(name) for name in MODEL_NAMES])
display(specs[['model','base_models','full_parameter_configurations',
               'base_full_fits_per_outer_fit','base_oof_fits_per_outer_fit',
               'meta_fits_per_outer_fit','primitive_fits_per_outer_fit']])
numeric,categorical=data.columns(4)
unfitted_example=make_estimator('stacking',numeric,categorical,seed=42,threads=THREADS)
print('Stacking: the outer Pipeline contains only',list(unfitted_example.named_steps))
print('Each base:',[(name,list(est.named_steps)) for name,est in unfitted_example.named_steps['model'].estimators])
print('All hyperparameter spaces were saved in advance; test is not used to modify them.')
atomic_json(REPORTS/'parameter_spaces.json',{name:parameter_space(name) for name in MODEL_NAMES})""")

md("""## 3. Agreed full experiment: specification

Seven models: Logistic Regression L2, Random Forest, XGBoost, LightGBM, MLP, soft voting and stacking. Each model–level pair receives **one** RandomizedSearchCV: `n_iter=20`, three saved CV folds, `scoring='neg_log_loss'`, `random_state=42`, `refit=False`, `error_score='raise'`. Maximizing negative log-loss minimizes log-loss. Finite spaces contain at least 20 configurations, sampled without replacement.

Search-selected parameters are then fixed. Five fresh Pipelines use seeds 42–46 and fit only their corresponding saved 90% training subsamples. Seed and training rows change, but hyperparameters remain fixed. Fitted medians/categories are not reused between repetitions. The primary protocol has no class weights, resampling, calibration, feature selection or test-based early stopping. MLP uses `early_stopping=False`; its iteration limit is predefined and convergence warnings are saved.

**Budget:** 28 searches × 20 × 3 = 1 680 outer CV fits; 7 × 4 × 5 = 140 main fits. `refit=False` avoids 28 intermediate full-development fits. One outer voting fit contains 3 base fits; stacking contains 3 full + 9 internal OOF base fits and 1 meta-model fit. The plan totals **5 200 base + 260 meta-model fits = 5 460 primitive fits**. This is an alternative accounting of the same work, not an addition to 1 820 outer calls. Restarted interrupted stages may increase actual calls.""")
code("""plan=full_experiment_plan()
atomic_json(REPORTS/'full_experiment_plan.json',plan)
display(pd.DataFrame(plan['model_specs'])[['model','searches','candidate_cv_outer_fits','main_outer_fits','primitive_fits']])
print('Searches:',plan['searches'],'CV-fit:',plan['candidate_cv_outer_fits'],
      'main fits:',plan['main_outer_fits'],'primitive fits:',plan['primitive_fits_including_ensembles'])""")

md("""### 3.1. MLP settings and actual diagnostics

Full working settings are not changed using technical metrics or test: `max_iter=300`, `tol=1e-4`, `n_iter_no_change=20`, `early_stopping=False`, `solver='adam'`. The fixed resource-pilot configuration uses `(64,32)` hidden units, `alpha=0.001`, `batch_size=256`, `learning_rate_init=0.001`, seed 42. Main search varies only predefined search-space parameters; these stopping limits remain fixed.

`max_iter` is an upper limit; actual completed epochs are read from `n_iter_`. After every saved repeat fit and resource-pilot fit, `fit_diagnostics.json` records `n_iter_`, the complete `loss_curve_` and final `loss_`; warnings, including ConvergenceWarning, are saved in stage JSON. Curve presence and length are checked; serialized Pipelines also retain these attributes. RandomizedSearchCV with `refit=False` does not retain candidate models: search warnings are aggregated, but individual CV candidate curves are not saved. Resource-pilot settings are not adjusted after measurement.""")
code("""mlp_numeric,mlp_categorical=PreparedData(PREPARED).columns(4)
mlp_full=make_estimator('mlp',mlp_numeric,mlp_categorical,seed=42,technical=False,threads=THREADS)
mlp_settings=mlp_full.named_steps['model'].get_params(deep=False)
mlp_display_keys=['hidden_layer_sizes','activation','solver','alpha','batch_size',
                  'learning_rate_init','max_iter','tol','n_iter_no_change','early_stopping','random_state']
display(pd.DataFrame([{'parameter':key,'full_mode_value':mlp_settings[key]} for key in mlp_display_keys]))
atomic_json(REPORTS/'mlp_full_settings.json',{key:mlp_settings[key] for key in mlp_display_keys})""")

md("""## 4. Technical run: development only

The check selects 1 200 clients from **saved** `train_seed_42` and 400 validation clients from development outside that train. Selection is TARGET-stratified with seed 2026. These are small technical subsets; official splits remain unchanged. Technical search reuses saved CV assignments for these 1 200 IDs rather than generating new folds.

All seven model types are checked at Level 4 (all sources). Each uses a reduced search of **2 configurations × 3 saved folds** and one technical fit with selected parameters; tree counts and MLP width/iterations are reduced. This gives 42 technical outer CV fits and 7 technical final fits, or 147 primitive fits including ensembles. Schemas of all four levels are also checked mechanically.

No test client enters fitting, selection or metrics in this check. Technical hyperparameters and metrics are not transferred to the full experiment. Files are written to a separate context-fingerprinted directory. Actual Pipelines, search results and ID-linked probabilities are saved; every model is reloaded from joblib and checked for identical probabilities.""")
code("""technical=run_technical(PREPARED,TECHNICAL_PARENT,
    train_n=TECHNICAL_TRAIN_N,validation_n=TECHNICAL_VALIDATION_N,threads=THREADS)
atomic_json(REPORTS/'technical_first_invocation.json',technical)
technical_root=Path(technical['root'])
technical_metrics=pd.read_parquet(technical_root/'technical_metrics.parquet')
technical_metrics.to_csv(REPORTS/'technical_metrics.csv',index=False)
display(technical_metrics[['model','train_rows','rows','pr_auc_average_precision','roc_auc',
                           'log_loss','brier_score','fit_seconds','warning_count']])
display(pd.DataFrame(technical['saved_cv_subset']))
print('Subsets read for training/evaluation:')
display(pd.DataFrame(technical['data_access']))
print('Test was not used:',not any(item['contains_test'] for item in technical['data_access']))
print('Technical-run results:',technical_root)
print('Time, seconds:',round(technical['elapsed_seconds_this_invocation'],2))""")

md("""## 5. Resume checks and recovery rules

`cv_results.parquet`, `best_params.json`, `pipeline.joblib`, `probabilities.parquet` (ID, TARGET, probability, model, level, seed), and `metrics.json` are saved atomically. `complete.json` is written **last** and contains SHA-256 for all stage artifacts. Metrics are log-loss, PR-AUC as average precision, ROC-AUC, Brier, precision/recall/F1, accuracy and the confusion matrix at fixed threshold 0.5. Profit thresholds, fairness and SHAP are later stages; test does not select thresholds or rules here.

Resume skips only verified completed searches and repetitions. An interrupted RandomizedSearchCV restarts the same 20 configurations; there is no within-candidate/fold recovery. Interrupted repetitions restart fitting. Errors are saved in `failure.json`, events in `events.jsonl`; warnings including ConvergenceWarning are retained separately in stage JSON and never hidden.

The manifest records SHA-256 of prepared data, splits and Python sources, search spaces, package versions and settings. A changed context rejects the existing full-run directory: retain the original code/environment or choose a new `FULL_OUTPUT`. The executable notebook is not hashed because its outputs change during execution. `run.lock` prevents concurrent writers. After a kernel crash, check the lock PID and remove only a stale lock whose process is no longer running.

After training diagnostics were added, the full directory became `a2_full_v2`. The interrupted `a2_full_v1` run and its sources remain separate; its manifest is not rewritten. Models, X, splits and the A2 budget remain unchanged.

The next cell actually repeats the technical command with the same context. All seven searches and seven fits must resume **without new training**, and model-file hashes must match.""")
code("""models_before={str(p.relative_to(technical_root)):sha256(p) for p in technical_root.rglob('pipeline.joblib')}
resume=run_technical(PREPARED,TECHNICAL_PARENT,
    train_n=TECHNICAL_TRAIN_N,validation_n=TECHNICAL_VALIDATION_N,threads=THREADS)
assert resume['run_counters']=={'new_searches':0,'resumed_searches':7,'new_fits':0,'resumed_fits':7}
models_after={str(p.relative_to(technical_root)):sha256(p) for p in technical_root.rglob('pipeline.joblib')}
assert models_before==models_after and len(models_after)==7
atomic_json(REPORTS/'technical_resume_check.json',{'counters':resume['run_counters'],
    'model_files_sha256_unchanged':True,'elapsed_seconds':resume['elapsed_seconds_this_invocation']})
print('Resume verified:',resume['run_counters'])
warning_rows=[]
for stage_name,records in [('search',technical['searches']),('fit',technical['fits'])]:
    for record in records:
        for warning in record['warnings']:
            warning_rows.append({'stage':stage_name,'model':record['model'],**warning})
warnings_frame=pd.DataFrame(warning_rows,columns=['stage','model','category','message','count'])
warnings_frame.to_csv(REPORTS/'technical_warnings.csv',index=False)
display(warnings_frame)
print('Convergence warnings require iteration monitoring in the full protocol; the technical run does not prove full-training convergence.')""")

md("""## 6. FULL LONG-RUNNING EXPERIMENT — DISABLED DURING THE INITIAL CHECK

To start, set **`RUN_FULL_EXPERIMENT=True`** in the configuration cell, run configuration, then run the cell below. Earlier stages need not all be rerun. Reuse `FULL_OUTPUT` to resume; current code and environment must match the manifest.

The run executes the agreed budget sequentially. Models and probabilities may require substantial disk space. Time, RAM and model-size measurements for fixed full configurations are saved by the resource pilot (sections 7–8); they cannot reliably predict all 20 candidates per pair or the total duration of 28 searches / 140 repetitions. Keep the application/kernel active. Successful stages are saved immediately. All input Parquet files remain unchanged.

During full execution, test is used only after each repeated Pipeline fit to obtain final probabilities/metrics. Results do not change the fixed search, parameters, preprocessing or 0.5 threshold.""")
code("""full_result=None
if RUN_FULL_EXPERIMENT:
    full_result=run_full_experiment(PREPARED,FULL_OUTPUT,threads=THREADS)
    display(full_result)
else:
    print('This cell did not run the full experiment: RUN_FULL_EXPERIMENT=False.')
    print('To start: RUN_FULL_EXPERIMENT=True; then run this cell.')
    print('Full-experiment directory:',FULL_OUTPUT)""",tags=['full_experiment_explicit_opt_in'])

md("""## 7. Separate resource pilot — Level 4 / train_seed_42

Enabled only by `RUN_RESOURCE_PILOT=True`. It uses all **221 407** saved `train_seed_42` IDs, Level 4 and seed 42. One `technical=False` factory configuration is fixed in advance for each of seven types, followed by exactly one outer fit. RandomizedSearchCV is not called, and validation/test are not read for evaluation. Voting/stacking retain their ordinary internal base/OOF fits; these are not additional configuration variants.

Selected settings are saved before fitting. Each type records row count, raw and transformed feature counts (by base and meta-model input for ensembles), duration, peak process RAM, saved model size, actual training diagnostics and warnings. The pilot is sequential and separate from the main experiment in `artifacts/training/resource_pilot/`. The report specifies the peak-memory method and sampling interval. One measured configuration does not guarantee memory requirements for every search candidate.

From a clean kernel, only configuration and the cell below are required. Parameter choice does not depend on technical metrics/test. This pilot does not consume the main 28-search/140-repeat budget. If available system RAM stays below 512 MiB for three seconds, the monitor stops only its own worker and saves a partial result. Training size and settings are not automatically reduced; failed attempts are counted separately from successful fits.

Saved pilots remain visible in section 8 with `RUN_RESOURCE_PILOT=False`. A repeated call with the same manifest verifies checkpoints and skips completed types; incomplete fits restart. Equivalent terminal command: `python -m src.resource_pilot --prepared data/processed/full --output-parent artifacts/training/resource_pilot`. To refresh report cells without fitting: `python scripts/execute_training_notebook.py --reports-only`.

**Actual execution:** all seven model types were saved. There were nine outer attempts: the first RF was stopped by the RAM guard; the first stacking fit completed but could not save its Pipeline because a service JSON file was temporarily locked on Windows. RF was retried after memory was freed; bounded JSON-write retries were added. Six ready Pipelines were reused byte-for-byte, and only stacking was repeated. Original manifests, source code and checkpoints are preserved; migration checks data, parameters and all code except the JSON-write function. The completed directory is `resource_de05e94a7cb7`; the original partial run is reported separately. Full attempt history and timing are in `reports/training/resource_pilot_attempts.csv`, with checks in `resource_pilot_validation.json`. The main A2 budget is unchanged.""")
code("""if RUN_RESOURCE_PILOT:
    resource_result=run_resource_pilot(PREPARED,RESOURCE_PARENT,threads=THREADS)
    atomic_json(REPORTS/'resource_pilot_summary.json',resource_result)
    resource_metrics_path=Path(resource_result['root'])/'resource_metrics.parquet'
    if resource_metrics_path.exists():
        pd.read_parquet(resource_metrics_path).to_csv(REPORTS/'resource_pilot_metrics.csv',index=False)
    display(resource_result)
else:
    print('This cell did not run the resource pilot: RUN_RESOURCE_PILOT=False.')
    print('To start, run configuration with RUN_RESOURCE_PILOT=True, then this cell.')
print('Separate resource-pilot directory:',RESOURCE_PARENT)""",tags=['resource_pilot_explicit_opt_in'])

md("""## 8. Separate summaries from saved artifacts

This cell depends **only on configuration paths**. It does not require `technical`, `technical_root`, `warnings_frame`, `frozen_before` or a previous fit result. It works after a clean kernel → configuration → section 6 → this section, producing the full report even without a technical run in this kernel.

The full report reads the experiment manifest, checks `complete.json` stage hashes and counts actual completed searches/repetitions. Status distinguishes no run, active/interrupted partial runs, corrupted artifacts and completion of the 28/140 plan. Progress from a run started before this kernel is visible; old artifacts are not removed. Input integrity is checked against `experiment_manifest.inputs`, not a `frozen_before` variable.

Technical summaries are saved separately in `reports/training/technical/`, and full summaries in `reports/training/full/`. Missing technical files or skipped technical cells do not prevent full reporting. Technical-sample metrics are not mixed with final test metrics. Resource results remain separate and do not select the quality winner.

Large artifacts are in `artifacts/` and excluded from Git; small aggregate summaries are in `reports/training/`. Load joblib only from this experiment's own trusted files.""")
code("""technical_disk_report=write_technical_reports(TECHNICAL_PARENT,REPORTS/'technical',prepared_dir=PREPARED)
full_disk_report=write_full_report(FULL_OUTPUT,REPORTS/'full',prepared_dir=PREPARED)
resource_disk_report=write_resource_reports(RESOURCE_PARENT,REPORTS/'resource')
atomic_json(REPORTS/'training_notebook_summary.json',{
    'schema_version':2,'reports_built_from_disk':True,
    'technical_report':str(REPORTS/'technical/technical_runs.json'),
    'full_report':str(REPORTS/'full/full_summary.json'),
    'resource_report':str(REPORTS/'resource/resource_runs.json'),
    'technical_runs_found':technical_disk_report['runs_found'],
    'full_status':full_disk_report['status'],
    'resource_runs_found':resource_disk_report['runs_found']})
print('Technical runs on disk:',technical_disk_report['runs_found'])
display(pd.DataFrame(technical_disk_report['runs']))
print('Separate full summary:')
display(pd.DataFrame([full_disk_report]))
print('Separate resource pilots on disk:',resource_disk_report['runs_found'])
resource_display=pd.DataFrame(resource_disk_report['model_metrics'])
if not resource_display.empty:
    resource_display['processed_features']=resource_display['processed_features_json'].map(
        lambda raw:'; '.join((item['path'].split('.bases.')[-1].replace('pipeline.model.meta','meta')+
            ': '+str(item['features'])) for item in json.loads(raw)) or 'no summary data')
    resource_display['model_MiB']=pd.to_numeric(resource_display['model_size_bytes'])/(1024**2)
    for resource_name,resource_rows in resource_display.groupby('run',sort=False):
        print(resource_name)
        display(resource_rows[['model','status','rows','raw_features','processed_features',
            'fit_seconds','peak_ram_gib','model_MiB','warning_count','mlp_n_iter_json']])
print('Resource metrics CSV by model:',resource_disk_report['model_metrics_csv'])
print('Resource-pilot warnings CSV:',resource_disk_report['warnings_csv'])
print('Summaries:',REPORTS/'technical/technical_runs.json',REPORTS/'full/full_summary.json')""",tags=['disk_reports_independent_of_technical'])

notebook=nbf.v4.new_notebook(cells=cells,metadata={
    'kernelspec':{'display_name':'Python 3 (ipykernel)','language':'python','name':'python3'},
    'language_info':{'name':'python'}})
nbf.write(notebook,ROOT/'notebooks/02_model_training.ipynb')
print('Created',len(cells),'cells; RUN_FULL_EXPERIMENT=False')
