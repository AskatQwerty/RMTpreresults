"""Additional cells for 01: audit existing data only, without recreating them."""
import nbformat as nbf


def quality_cells():
    cells=[]
    def md(source):
        cells.append(nbf.v4.new_markdown_cell(source,metadata={'tags':['data_quality_v1']}))
    def code(source):
        cells.append(nbf.v4.new_code_cell(source,metadata={'tags':['data_quality_v1']}))

    md("""## 8. Additional data-quality checks before modeling

This section uses the **existing** `development_ids.parquet`, four X files, and feature lists. Splits are not recreated, and features are not overwritten. SHA-256 hashes of all prepared files are checked before and after. Distributions, percentiles, and semantic constraints are analyzed on **development** only; test data participate only in mechanical preservation checks and the requested breakdown of 2,905 rows with incomplete keys. Test data are not used for outlier decisions.

Results are written to `reports/quality/`. Counts of **raw history values**, **clients**, and **aggregate cells** are reported separately because these are different counting units.""")
    code(r"""from pathlib import Path
import sys, os, json, time
import pandas as pd
from IPython.display import display

QA_ROOT=Path.cwd()
if QA_ROOT.name=='notebooks':
    QA_ROOT=QA_ROOT.parent
sys.path.insert(0,str(QA_ROOT))
from src.data_quality import (artifact_manifest, development_profile, semantic_checks, sentinel_checks)
from src.payment_quality import audit_payment_keys
from src.fold_preprocessing import make_numeric_pipeline, numeric_pipeline_smoke_check
from src.data_preparation import dump_json, source_manifest

QA_DATA=Path(os.environ.get('HOME_CREDIT_DATA_DIR',r'C:\Users\askat\Study\RMT\home-credit-default-risk'))
QA_PREP=QA_ROOT/'data/processed/full'
QA_REPORT=QA_ROOT/'reports/quality'
QA_REPORT.mkdir(parents=True,exist_ok=True)
QA_STARTED=time.perf_counter()
qa_manifest_before=artifact_manifest(QA_PREP)
qa_sources_before=source_manifest(QA_DATA)
dump_json(QA_REPORT/'prepared_sha256_before.json',qa_manifest_before)
print('Existing prepared files protected by SHA-256:',len(qa_manifest_before))""")

    md("""### 8.1. Numeric distributions in development

Exact linearly interpolated P1, P25, median, P75, P99, and P99.9 are computed from nonmissing values; missingness, min/max, zeros, negative values, and infinities are counted separately. Computing a median for the **report** neither imputes data nor fits a transformation. Data are read in batches of 24 columns to avoid holding all levels in memory.

The full table is `numeric_development_summary.csv`; `first_level` and `levels` indicate feature membership. Shared features are profiled once. Entirely missing columns, a single unique **observed** value with missingness, and truly constant columns without missingness are distinguished. No columns are removed based on this global summary.""")
    code("""qa_profile=development_profile(QA_PREP,QA_REPORT)
display(pd.DataFrame(qa_profile['levels']))
numeric_summary=pd.read_csv(QA_REPORT/'numeric_development_summary.csv')
print('Full summary:',QA_REPORT/'numeric_development_summary.csv')
display(numeric_summary.sort_values('missing_fraction',ascending=False)[
    ['feature','first_level','missing','missing_fraction','min','p01','p25','median','p75','p99','p999','max','zeros','negatives']].head(25))
print('Empty and constant columns (all types):')
display(pd.read_csv(QA_REPORT/'empty_and_constant_columns.csv'))
print('Entirely missing numeric columns:',qa_profile['all_missing_numeric'])
print('Truly constant numeric columns without missingness:',qa_profile['constant_fully_numeric'])""")

    md("""### 8.2. Semantic constraints and proposed actions

Strict rules require positive current income/credit, nonnegative nominal amounts, integer family/child counts, and `*_SHARE` fractions in `[0,1]`. A missing value is neither negative nor zero. Absence of a sign violation does not establish that an extreme value is accurate.

Amount ratios **need not fall within `[0,1]`**: overpayments, exceeded credit limits, and differences between requested and approved credit are possible. The dictionary does not define comparable periods for income and annuity, so `annuity/income > 1` alone does not justify correction. Negative balances and cash movements can represent overpayments or reversals; no blanket rule removes all AMT < 0 values.

For each rule, `semantic_source_checks.csv` provides the SQL condition, rationale, action, status, and number of affected raw values/clients. The specific proposal for negative down payments is to replace them with NULL **before** aggregation after confirming the sign convention; the exact number of changed aggregates is computed, but the replacement is not applied now. Ambiguous limits/movements require source clarification rather than unsupported correction.

Rare large values are retained. P99/P99.9 describe the tail and are not removal, clipping, or winsorization thresholds.""")
    code("""qa_semantics=semantic_checks(QA_DATA,QA_PREP,QA_REPORT)
display(pd.read_csv(QA_REPORT/'semantic_source_checks.csv')[
    ['source','field','condition','affected_values','affected_clients','classification','proposed_action_ru','reason_ru']])
display(pd.read_csv(QA_REPORT/'ratio_checks_development.csv'))
focus=['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE','CNT_FAM_MEMBERS','CNT_CHILDREN',
       'APP_CREDIT_TO_INCOME','APP_ANNUITY_TO_INCOME','APP_CREDIT_TO_GOODS','APP_ANNUITY_TO_CREDIT','APP_INCOME_PER_MEMBER']
display(numeric_summary[numeric_summary.feature.isin(focus)][
    ['feature','missing','min','p01','p25','median','p75','p99','p999','max','zeros','negatives']])
print('Estimated impact of the proposed replacement; not applied:')
display(pd.DataFrame([qa_semantics['proposed_fix']]))
print('Income tail, retained unchanged:',qa_semantics['income_tail'])""")

    md("""### 8.3. Sentinel values: source → rule → saved result

For `DAYS_EMPLOYED=365243`, checks verify both absence of the number from X and agreement with the source for **every application**: the value becomes NULL, `APP_EMPLOYMENT_DATE_UNKNOWN=1`, and other valid dates remain unchanged. Other application dates describing actual events are checked similarly.

In previous applications, `365243` is replaced with NULL in the scheduled date fields used: `DAYS_FIRST_DUE` and `DAYS_LAST_DUE_1ST_VERSION`. Their AVG/MIN/MAX are recomputed from cleaned rows and compared with saved aggregates for all prepared applications. `DAYS_FIRST_DRAWING`, `DAYS_LAST_DUE`, and `DAYS_TERMINATION` are excluded from X entirely: 'after' means no feature, rather than a count of imputed values. Positive scheduled dates are retained.

Before/after NULL counts use the same set of **raw rows**, because comparing raw NULL counts with client-aggregate counts would be invalid. Aggregate verification is saved separately. `placeholder_before` always refers only to `365243`. The auxiliary `positive_before` counts all positive values, including the sentinel, for application dates describing actual events, but only positive nonsentinel values for scheduled previous/bureau dates: these fields have different semantic rules. Mechanical checks of final temporal features do not tune parameters on test data.""")
    code("""qa_sentinels=sentinel_checks(QA_DATA,QA_PREP,QA_REPORT)
display(pd.read_csv(QA_REPORT/'sentinels_before_after.csv')[
    ['source','field','scope','source_rows','placeholder_before','null_before','positive_before',
     'null_after_same_rows','placeholder_after_same_rows','saved_feature_status']])
display(pd.read_csv(QA_REPORT/'saved_date_aggregate_checks.csv'))
print('Saved sentinel checks:',qa_sentinels)""")

    md("""### 8.4. Incomplete keys in 2,905 installments_payments rows

The complete candidate key checked for a payment row is:
`(SK_ID_PREV, NUM_INSTALMENT_VERSION, NUM_INSTALMENT_NUMBER, DAYS_ENTRY_PAYMENT)`.
`SK_ID_CURR` additionally identifies the owner and links to the current application. This is **not a guaranteed unique payment-event ID**: several partial payments may share the same date and candidate key.

The installment key used in preparation is:
`(SK_ID_CURR, SK_ID_PREV, NUM_INSTALMENT_VERSION, NUM_INSTALMENT_NUMBER)`.
Actual payment date is not part of this key. Partial-payment amounts are summed exactly, while the scheduled amount is counted once. Full duplicates are diagnosed; repeated candidate keys do not trigger deletion.

The report shows missingness in all eight fields, breakdown by the existing splits, other payments for the same installment, and client preservation. A row without an actual payment date does not satisfy `<0`; an unknown amount is not filled with zero. Such a row establishes neither payment nor nonpayment reliably. Final checks verify preservation of the **number of raw payment rows** and the **exact payment total** in aggregated installments.""")
    code("""qa_payments=audit_payment_keys(QA_DATA,QA_PREP,QA_REPORT)
display(pd.read_csv(QA_REPORT/'installments_incomplete_key_fields.csv'))
display(pd.read_csv(QA_REPORT/'installments_incomplete_key_patterns.csv'))
display(pd.DataFrame(qa_payments['by_split']))
display(pd.DataFrame(qa_payments['obligation_recovery']))
display(pd.DataFrame(qa_payments['client_retention']))
display(pd.DataFrame([qa_payments['partial_payment_preservation']]).T)
print('Incomplete rows:',qa_payments['missing_key_rows'],'Missing fields:',qa_payments['null_fields'])""")

    md("""### 8.5. An entirely missing column within a training fold

Even if no column is entirely missing across development, this can occur in an individual training fold or 90% repeat. The next stage uses the prepared factory `src.fold_preprocessing.make_numeric_pipeline(scale=False)`, which returns an **unfitted** object.

Numeric branch: `SimpleImputer(strategy='median', keep_empty_features=True)`. If a column is entirely missing in the **current training set**, no median exists: the predefined technical fallback is **0**, and the column is retained. Observed validation/test values in that column remain unchanged; only missing values are filled. This is not an estimate from development/test. With no observed training values, the model has no basis for estimating the variable's effect.

In parallel, `MissingIndicator(features='all')` creates an original-missingness indicator for every numeric column, including missingness first seen in validation. Numeric output width is stable at **2 × p**, regardless of which columns were missing in training. With `scale=True`, StandardScaler is fitted only on the value branch within training; indicators are not scaled. After fallback for an entirely missing column, the scaler has mean=0 and scale=1.

Categorical encoding and optional feature selection remain in the full Pipeline at the next stage. The original feature lists for all four levels are preserved. The factory must be included in ColumnTransformer and Pipeline **before passing to CV**, rather than calling fit_transform on all development data beforehand. Only a check on three synthetic columns is run here; Home Credit data undergo no fit, imputation, scaling, or encoding.""")
    code("""unfitted_numeric_pipeline=make_numeric_pipeline(scale=False)
qa_pipeline=numeric_pipeline_smoke_check(QA_REPORT/'fold_pipeline_checks.json')
print('Demonstration on synthetic data only:')
display(pd.DataFrame([qa_pipeline]))""")

    md("""### 8.6. Audit summary and open questions

**Already corrected during preparation:** date sentinels are replaced with NULL or fields are excluded; unavailable actual future/undated payments are filtered out; division by a nonpositive denominator yields NULL; partial payments are combined without losing amounts or repeating the installment nominal amount. Decimal arithmetic prevents false underpayment at exact equality.

**Retained:** rare large incomes/amounts, ratios >1, positive contractual dates, zero amounts for rejected/cancelled previous applications, negative balances/movements pending clarification of their accounting meaning, and two features with a constant observed zero plus missingness. Data quality does not imply automatic distribution normalization.

**Kept inside the Pipeline:** training-fold medians, entirely missing training-column handling with fallback=0 and indicators, scaling when needed, learned encoding, and selection. None of these stages is fitted on the full Home Credit data.

**Open questions:** confirmation of the 117 million income value and income units/periods; sign conventions for negative down payments, withdrawals, limits, and balances; unavoidable incomplete information for installments missing payment date/amount; exact availability times of updated historical values. Where dictionary information is insufficient, ambiguous values are flagged for review rather than silently corrected.

The final cell confirms preservation of all X files, feature lists, target, fairness data, IDs, and aggregates; raw CSVs are also verified by SHA-256.""")
    code("""qa_manifest_after=artifact_manifest(QA_PREP)
assert qa_manifest_before==qa_manifest_after,'Prepared data or splits changed!'
assert qa_sources_before==source_manifest(QA_DATA),'Raw CSVs changed!'
dump_json(QA_REPORT/'prepared_sha256_after.json',qa_manifest_after)
qa_result={'scope':'development_quality_audit_existing_prepared_data',
           'profile':qa_profile,'semantics':qa_semantics,'sentinels':qa_sentinels,
           'payments':qa_payments,'pipeline':qa_pipeline,
           'prepared_files_sha256_unchanged':True,'source_csv_sha256_unchanged':True,
           'X_and_split_files_modified':0,'global_fit_on_home_credit':False,
           'elapsed_seconds':round(time.perf_counter()-QA_STARTED,3)}
dump_json(QA_REPORT/'quality_summary.json',qa_result)
print('Development:',qa_profile['development_rows'],'numeric features:',qa_profile['numeric_features'])
print('Strict violations:',qa_semantics['hard_rule_violations'],'ratio-rule violations:',qa_semantics['ratio_rule_violations'])
print('X, levels, splits, and sources: SHA-256 hashes unchanged. This audit did not modify values.')
print('Additional audit time, seconds:',qa_result['elapsed_seconds'])
quality_files=pd.DataFrame([{'file':p.name,'bytes':p.stat().st_size} for p in sorted(QA_REPORT.iterdir()) if p.is_file()])
display(quality_files)""")
    return cells
