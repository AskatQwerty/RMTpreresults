"""Create a readable notebook; execute it separately with execute_notebook.py."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

md("""# 01. Home Credit Default Risk data preparation

**The unit of analysis is one current application, `SK_ID_CURR`.** This notebook audits the data, runs a technical pilot and performs full deterministic preparation. No models are trained here.

Basis: the local `Aubakirov_Dagystanov_A2_Methods_v2.pdf`, Methods and Materials — Assignment 2, especially page 2. The accepted plan is **7 models × 4 levels × 5 repetitions = 140 main training runs**. Each of the 28 model–level pairs has a separate Randomized Search: 20 configurations × stratified 3-fold CV, selected by log-loss; the parameters are then fixed for five repetitions. This adds 1 680 candidate CV fits to the 140 main fits, excluding internal ensemble training, refit and calibration.

Data-fitted transformations—medians, scaling, one-hot or other learned encoding, and feature selection—**remain inside the next-stage Pipeline**. No `.fit()`, class resampling or TARGET-based feature selection is performed on Home Credit here. Future Pipeline behavior is checked only with synthetic data.

Implementation is in `src/data_audit.py` and `src/data_preparation.py`; the notebook explicitly runs each stage and saves its actual results. CSV inputs remain unchanged. Dependencies are pinned in `requirements.txt` (`python -m pip install -r requirements.txt`); from the project root, use `python scripts/execute_notebook.py` or Jupyter Run All.""")

code(r"""from pathlib import Path
import sys, os, time, json, gc, shutil
import pandas as pd
import psutil
from IPython.display import display

ROOT = Path.cwd()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent
assert (ROOT / 'src/data_preparation.py').is_file(), 'Run the notebook from the project root or notebooks/'
sys.path.insert(0, str(ROOT))
from src.data_audit import audit_sources
from src.data_preparation import (source_manifest, connect, run_preparation,
                                  compare_pilot_to_full, dump_json)

DATA_DIR = Path(os.environ.get('HOME_CREDIT_DATA_DIR', r'C:\Users\askat\Study\RMT\home-credit-default-risk'))
REPORTS = ROOT / 'reports'
PILOT_DIR = ROOT / 'data/processed/pilot'
FULL_DIR = ROOT / 'data/processed/full'
PILOT_SIZE = 3000
RUN_FULL = True
RUN_START = time.perf_counter()
REPORTS.mkdir(exist_ok=True)
print('Source CSV files:', DATA_DIR)
print('Total/available RAM, GiB:', round(psutil.virtual_memory().total/1024**3,2),
      round(psutil.virtual_memory().available/1024**3,2))
print('Free disk, GiB:', round(shutil.disk_usage(ROOT).free/1024**3,2))
print('DuckDB: memory_limit=2GB; threads=2; sequential sources, disk spill.')""")

md("""## 1. File availability and source integrity

All eight required files are checked. Execution stops and names any missing file. A SHA-256 is saved for each source and recomputed after preparation. `application_test.csv` and `sample_submission.csv` are outside the labelled study population and are not used.

Raw CSV, Parquet and temporary files are excluded through `.gitignore`. Only code, notebooks and small de-identified summaries belong in Git.""")
code("""manifest_before = source_manifest(DATA_DIR)
dump_json(REPORTS / 'source_sha256_before.json', manifest_before)
display(pd.DataFrame(manifest_before)[['file','bytes']])""")

md("""## 2. Full audit before feature construction

The audit reads **every row of every CSV**, even when the subsequent technical pilot uses 3 000 clients. DuckDB initially infers types from 100 000 rows, then reads the full contents strictly: `ignore_errors=false`, so incompatible types raise errors. Each column records its type, missing count/fraction and numeric min/max; date fields also record positive values, zeros and the `365243` sentinel.

Exact duplicate rows and candidate keys are checked. Repeated payment keys may represent partial payments and are not automatically removed. Unique application, bureau and previous_application keys are required for joins. Histories without a labelled current application are counted in the audit but excluded from X.

The dictionary is read as `cp1252`, with whitespace trimmed from column names. Documented `SK_BUREAU_ID` is matched to actual `SK_ID_BUREAU`. Fields described in the dictionary but absent from CSV are not invented. Relationship checks distinguish missing parents, incorrect contract owners and clients outside `application_train`.""")
code("""audit_connection = connect(ROOT / 'tmp/audit_spill')
try:
    audit = audit_sources(audit_connection, DATA_DIR, REPORTS / 'audit')
finally:
    audit_connection.close()
table_audit = pd.read_csv(REPORTS / 'audit/table_summary.csv')
display(table_audit)
display(pd.read_csv(REPORTS / 'audit/target_balance.csv'))
key_tables = table_audit[table_audit.table.isin(['application_train.csv','bureau.csv','previous_application.csv'])]
assert (key_tables.key_duplicate_surplus == 0).all()
assert (key_tables.rows_with_null_key == 0).all()
assert not audit['dictionary_unmatched_columns']
print('All actual columns match the dictionary; parent-table keys are unique.')""")
code("""column_audit = pd.read_csv(REPORTS / 'audit/column_audit.csv')
display(column_audit.sort_values('missing_fraction', ascending=False)[
    ['table','column','dtype','missing_count','missing_fraction']].head(25))
display(column_audit[column_audit.positive_count.notna()][
    ['table','column','min','max','positive_count','zero_count','placeholder_365243_count']])
relationships = pd.read_csv(REPORTS / 'audit/relationships.csv')
display(relationships)
print('Dictionary discrepancies:')
display(pd.read_csv(REPORTS / 'audit/dictionary_extra_rows.csv')[['Table','Row','Description']])""")

md("""## 3. Temporal availability and feature rules

The dataset has no calendar dates or complete change history. The rules below therefore provide reproducible filtering with available fields, but do not prove point-in-time availability of every attribute. Application fields, including `EXT_SOURCE_1/2/3`, are treated as a snapshot at application; the dictionary does not disclose external-score dates. Final states without a defensible availability date are excluded.

| Source | Rule and interpretation |
|---|---|
| application | `DAYS_EMPLOYED=365243` and other positive actual-date offsets become NULL; unknown employment dates receive an indicator. Age and gender are removed from X before ratios are created. |
| bureau | `DAYS_CREDIT < 0` and `DAYS_CREDIT_UPDATE < 0`: the decision and latest information must precede the application day. `DAYS_CREDIT_ENDDATE` is remaining contractual duration, so positive planned dates are retained. `DAYS_ENDDATE_FACT` is allowed only when <0. |
| bureau_balance | `MONTHS_BALANCE < 0`, linked through eligible bureau records → application. Month 0 is excluded because the exact snapshot day is unknown. |
| POS / credit card | `MONTHS_BALANCE < 0`, directly linked by `SK_ID_CURR`. The POS dictionary allows month 0 as an application-time snapshot, but the common strict rule excludes it. Future instalment counts represent scheduled obligations already known at the historical snapshot. |
| installments | Actual `DAYS_ENTRY_PAYMENT < 0`; unknown or future actual payments are excluded. Positive `DAYS_INSTALMENT` is not automatically removed: it may be the due date of a completed prepayment. `365243` is not a valid date. |
| previous | Only `DAYS_DECISION < 0`. `DAYS_FIRST_DUE` and `DAYS_LAST_DUE_1ST_VERSION` are contractual dates: positive values are retained and `365243` becomes NULL. `DAYS_FIRST_DRAWING`, `DAYS_LAST_DUE`, `DAYS_TERMINATION` are excluded from X because availability/update timing is ambiguous. |

The audit checks the `SK_ID_PREV` link to previous. Absence of a published previous application does not invalidate directly linked history with valid `SK_ID_CURR` and its own eligible date; Level 3 does not depend on the Level 4 table. All exclusions are counted in `temporal_filters.csv`; retained coverage is saved in `history_coverage.csv`.

Exact duplicates are diagnosed but not blindly removed without an event identifier. Partial `AMT_PAYMENT` values are first summed within `(SK_ID_CURR, SK_ID_PREV, NUM_INSTALMENT_VERSION, NUM_INSTALMENT_NUMBER)`. The scheduled amount is counted once. Obligations with conflicting due dates or amounts within the key are excluded from aggregates and counted separately. Late-payment fractions describe **observed paid instalments**; entirely absent unpaid obligations cannot be reconstructed from this table.

Instalment and partial-payment amounts are aggregated as **DECIMAL(24,3)**: the actual CSV was checked to contain at most three decimal places. This avoids false underpayment caused by floating-point summation order at equality. Excess precision is checked before conversion; final features use ordinary numeric types. Lateness based on the last payment date is an observed approximation: a late overpayment may mark an obligation fully paid on time as late. It is not the exact debt-settlement date. Zero or negative ratio denominators produce NULL rather than infinity. Historical mean/min/max describe typical values and ranges; monthly-table averages are weighted by observed contract-month counts. `HAS_*` indicators denote available eligible aggregates. With no history, they equal 0, while sums, fractions and counts remain NULL.""")

md("""## 4. Four nested levels and a technical pilot

1. **Level 1:** application and five ratios (credit/income, annuity/income, credit/goods, annuity/credit, income/family), plus an unknown-employment-date indicator.
2. **Level 2:** Level 1 + bureau and bureau_balance: credit counts, debt, arrears, durations and monthly-status fractions.
3. **Level 3:** Level 2 + POS and card: instalment and delinquency history, turnover, payments and credit-card limit utilization.
4. **Level 4:** Level 3 + installments and previous: payment/lateness of observed obligations, amounts and outcomes of previous applications.

Select 3 000 **clients** by stratified sampling with seed 2026 and read all their linked history, including bureau_balance through bureau. Do not independently take the first rows of each table. The technical dataset is saved separately and does not replace the full result. Each stage first saves aggregates with unique `SK_ID_CURR`, followed by left joins; assertions prohibit row multiplication.""")
code("""pilot = run_preparation(DATA_DIR, PILOT_DIR, REPORTS / 'pilot', sample_size=PILOT_SIZE)
display(pd.DataFrame(pilot['shapes']))
display(pd.DataFrame(pilot['history_coverage']))
print('Pilot: all checks passed:', pilot['checks'])
print('Pilot time, seconds:', pilot['elapsed_seconds'])""")

md("""## 5. Full preparation and saved splits

Full preparation starts only after the pilot and resource checks pass. DuckDB is limited to 2 GB and may spill intermediate operations to disk. Historical tables and four X matrices are not all held in memory simultaneously. At least 3 GiB available RAM and 10 GiB free disk are required; otherwise the notebook records the reason and remaining action.

The full population is split once with `train_test_split(test_size=0.20, stratify=TARGET, random_state=42)`. Development is used for later search; test is reserved for final evaluation. Five stratified 90% development subsamples use seeds 42…46. The remaining 10% of a repetition does not become a new test set. Repeated subsamples may overlap, as expected. A common 3-fold assignment is also saved for 28 future searches. Reuse these saved IDs rather than generating separate splits for each level/model.

Fairness attributes are stored separately: original `CODE_GENDER`, `DAYS_BIRTH`, `AGE_YEARS=-DAYS_BIRTH/365.25`, `<25`/`>=25` groups, and an unknown-gender group. Age in years is approximate because calendar dates are unavailable. These attributes, IDs and TARGET are excluded from X. Excluding direct protected attributes does not guarantee the absence of indirect proxies.""")
code("""available_gib = psutil.virtual_memory().available / 1024**3
disk_gib = shutil.disk_usage(ROOT).free / 1024**3
full = None
full_skip_reason = None
if not RUN_FULL:
    full_skip_reason = 'RUN_FULL=False: set True and run stages 5–7 to finish.'
elif available_gib < 3 or disk_gib < 10:
    full_skip_reason = f'Insufficient resources: RAM={available_gib:.2f} GiB, disk={disk_gib:.2f} GiB. Free resources and repeat stages 5–7.'
else:
    full = run_preparation(DATA_DIR, FULL_DIR, REPORTS / 'full', sample_size=None)
if full is None:
    print('FULL RUN NOT EXECUTED:', full_skip_reason)
else:
    print('FULL RUN COMPLETED:', full['n_applicants'], 'current applications')
    display(pd.DataFrame(full['shapes']))
    display(pd.DataFrame(full['splits']))
    display(pd.DataFrame(full['history_coverage']))
    print('Conflicting obligations excluded:', full['installment_conflicting_obligations_excluded'])""")

md("""## 6. Output checks and file formats

`X_level_N.parquet` contains **features only**, with no identifiers even in its index. All four files are ordered by `SK_ID_CURR`; their exact row mapping is saved in `row_ids.parquet`. `target.parquet` and `fairness.parquet` include IDs for safe alignment. JSON lists preserve feature order and nesting, while `feature_catalog.csv` documents each feature and its SQL formula.

Automatic checks cover unique applications and aggregate keys; identical clients and row counts at every level; equality of shared features; absence of TARGET/ID/gender/age from X; disjoint development/test; repetition IDs contained in development; no infinities; and NULL features for absent history. No client is dropped because history is missing.

An additional check compares pilot and full-data values for **the same clients**, reading only 30 columns at a time. It detects accidental partial-history selection or dependence of deterministic features on sample composition. Tolerance `1e-10` accounts for floating-point summation order during parallel aggregation; no parameters are learned.""")
code("""crosscheck = compare_pilot_to_full(PILOT_DIR, FULL_DIR) if full is not None else None
print('Pilot/full-data comparison:', crosscheck)
manifest_after = source_manifest(DATA_DIR)
assert manifest_after == manifest_before, 'A source CSV has changed!'
dump_json(REPORTS / 'source_sha256_after.json', manifest_after)
print('All eight source CSV files are unchanged: SHA-256 matches.')
dump_json(REPORTS / 'run_summary.json', {
    'processed_scope': 'full' if full is not None else 'technical_sample_only',
    'audit_seconds': audit['elapsed_seconds'], 'pilot': pilot, 'full': full,
    'full_skip_reason': full_skip_reason, 'pilot_full_comparison': crosscheck,
    'original_csv_sha256_unchanged': True,
    'total_elapsed_seconds': round(time.perf_counter()-RUN_START,3)
})
print('Checks:', (full or pilot)['checks'])""")

md("""## 7. Summary and handoff to modelling

The next cell reports the actual processed volume, dimensions, balance, time and file list. Large data are stored in `data/processed/full/`, intermediate aggregates in `aggregates/`, and split IDs in `splits/`. The pilot is in the adjacent `pilot/` directory. Small reports remain in `reports/`.

To reconstruct X by ID in the next stage:

```python
ids = pd.read_parquet(FULL_DIR / 'row_ids.parquet')['SK_ID_CURR']
X = pd.read_parquet(FULL_DIR / 'X_level_1.parquet')
X.index = pd.Index(ids, name='SK_ID_CURR')  # ID is an index, not an input column
y = pd.read_parquet(FULL_DIR / 'target.parquet').set_index('SK_ID_CURR')['TARGET']
dev_ids = pd.read_parquet(FULL_DIR / 'splits/development_ids.parquet')['SK_ID_CURR']
X_dev, y_dev = X.loc[dev_ids], y.loc[dev_ids]
```

Medians, missing indicators, categories and scaling are fitted **inside each CV training fold**. After search, they are fitted afresh on each 90% training subsample with fixed hyperparameters. Test is not used for these decisions; the fairness file is used only for the specified analysis/correction protocol. Every model at a level uses the same original feature list.""")
code("""result = full or pilot
print('Actually processed:', result['scope'], '—', result['n_applicants'], 'applications')
display(pd.DataFrame(result['shapes']).rename(columns={'features':'n_features','rows':'n_rows'}))
display(pd.DataFrame([{'TARGET=0': result['target_0'], 'TARGET=1': result['target_1'],
                       'TARGET=1 fraction': result['positive_rate']}]))
run_report = json.loads((REPORTS/'run_summary.json').read_text(encoding='utf-8'))
print('Time, seconds: audit=', audit['elapsed_seconds'], 'pilot=', pilot['elapsed_seconds'],
      'full=', full['elapsed_seconds'] if full else None,
      'whole notebook=', run_report['total_elapsed_seconds'])
generated = []
for directory in [ROOT/'data/processed', REPORTS, ROOT/'notebooks', ROOT/'src', ROOT/'scripts', ROOT/'tests']:
    for path in sorted(directory.rglob('*')):
        if path.is_file() and 'duckdb_tmp' not in str(path) and '__pycache__' not in str(path):
            generated.append({'file':str(path.relative_to(ROOT)), 'bytes':path.stat().st_size})
for name in ['.gitignore','requirements.txt']:
    path = ROOT/name
    generated.append({'file':name, 'bytes':path.stat().st_size})
inventory = pd.DataFrame(generated)
inventory.to_csv(REPORTS/'generated_files.csv', index=False)
with pd.option_context('display.max_rows', None, 'display.max_colwidth', 120):
    display(inventory)
if full_skip_reason:
    print(full_skip_reason)""")

from quality_notebook_cells import quality_cells
nb['cells'] = cells + quality_cells()
nb['metadata'] = {'kernelspec': {'display_name':'Python 3 (ipykernel)','language':'python','name':'python3'},
                  'language_info': {'name':'python','version':'3.12'}}
(ROOT/'notebooks').mkdir(exist_ok=True)
nbf.write(nb, ROOT/'notebooks/01_data_preparation.ipynb')
print('Notebook created:',len(cells),'cells')
