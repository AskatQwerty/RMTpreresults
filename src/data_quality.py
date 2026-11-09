"""Additional development audit without modifying X or splits and without fit()."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .data_audit import register_sources
from .data_preparation import dump_json, qi, sql_path


def artifact_manifest(prepared_dir):
    rows = []
    for path in sorted(Path(prepared_dir).rglob('*')):
        if not path.is_file() or 'duckdb_tmp' in path.parts:
            continue
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024**2), b''):
                h.update(block)
        rows.append({'file': path.relative_to(prepared_dir).as_posix(), 'bytes':path.stat().st_size,
                     'sha256':h.hexdigest()})
    return rows


def development_context(prepared_dir):
    prepared_dir = Path(prepared_dir)
    ids = pd.read_parquet(prepared_dir/'row_ids.parquet').SK_ID_CURR.to_numpy()
    development = pd.read_parquet(prepared_dir/'splits/development_ids.parquet')
    assert len(np.unique(ids)) == len(ids) and (np.diff(ids)>0).all()
    positions = np.searchsorted(ids, development.SK_ID_CURR.to_numpy())
    assert np.array_equal(ids[positions],development.SK_ID_CURR.to_numpy())
    return development, positions


def _save(frame, report_dir, name):
    Path(report_dir).mkdir(parents=True,exist_ok=True)
    frame.to_csv(Path(report_dir)/name,index=False,encoding='utf-8-sig')
    return frame


def development_profile(prepared_dir, report_dir):
    """Exact linear percentiles; read 24 columns at a time, then retain development only."""
    prepared_dir, report_dir = Path(prepared_dir), Path(report_dir)
    development, positions = development_context(prepared_dir)
    levels = json.loads((prepared_dir/'feature_levels.json').read_text(encoding='utf-8'))
    schema = pq.read_schema(prepared_dir/'X_level_4.parquet')
    numeric = [field.name for field in schema if pa.types.is_integer(field.type)
               or pa.types.is_floating(field.type) or pa.types.is_decimal(field.type)]
    rows, diagnostics = [], []
    columns = levels['4']
    for start in range(0,len(columns),24):
        chosen = columns[start:start+24]
        frame = pd.read_parquet(prepared_dir/'X_level_4.parquet',columns=chosen).iloc[positions]
        for name in chosen:
            x = frame[name]
            observed = x.dropna()
            n_unique = int(observed.nunique())
            first = min(int(level) for level,cols in levels.items() if name in cols)
            common = {'feature':name,'first_level':first,'levels':','.join(map(str,range(first,5))),
                      'dtype':str(x.dtype),'n_development':len(x),'missing':int(x.isna().sum()),
                      'missing_fraction':float(x.isna().mean()),'n_unique_observed':n_unique,
                      'all_missing':n_unique==0,'constant_among_observed':n_unique==1,
                      'constant_including_missing':n_unique==1 and not x.isna().any(),
                      'constant_value':str(observed.iloc[0]) if n_unique==1 else None}
            diagnostics.append(common)
            if name not in numeric:
                continue
            values = observed.astype(float)
            percentiles = values.quantile([.01,.25,.5,.75,.99,.999],interpolation='linear')
            rows.append({**common,'min':values.min(),'p01':percentiles.loc[.01],
                         'p25':percentiles.loc[.25],'median':percentiles.loc[.5],
                         'p75':percentiles.loc[.75],'p99':percentiles.loc[.99],
                         'p999':percentiles.loc[.999],'max':values.max(),
                         'zeros':int(x.eq(0).sum()),'negatives':int(x.lt(0).sum()),
                         'infinite':int(np.isinf(values).sum()),
                         'placeholder_365243':int(x.eq(365243).sum()) if 'DAYS_' in name else None})
        del frame
    stats = _save(pd.DataFrame(rows),report_dir,'numeric_development_summary.csv')
    diagnostic = _save(pd.DataFrame(diagnostics),report_dir,'column_development_diagnostics.csv')
    special = diagnostic[diagnostic.all_missing | diagnostic.constant_among_observed]
    _save(special,report_dir,'empty_and_constant_columns.csv')
    # Record identical shared-feature statistics across levels without four copies of X.
    per_level = []
    for level, cols in levels.items():
        subset = stats[stats.feature.isin(cols)]
        per_level.append({'level':int(level),'rows':len(development),'total_features':len(cols),
                          'numeric_features':len(subset),'all_missing_numeric':int(subset.all_missing.sum()),
                          'constant_observed_numeric':int(subset.constant_among_observed.sum())})
    _save(pd.DataFrame(per_level),report_dir,'development_levels.csv')
    return {'development_rows':len(development),'numeric_features':len(stats),
            'all_missing_numeric':stats.loc[stats.all_missing,'feature'].tolist(),
            'constant_observed_numeric':stats.loc[stats.constant_among_observed,'feature'].tolist(),
            'constant_fully_numeric':stats.loc[stats.constant_including_missing,'feature'].tolist(),
            'infinite_values':int(stats.infinite.sum()),'levels':per_level}


def _connection(data_dir, prepared_dir, report_dir):
    con = duckdb.connect()
    con.execute("SET memory_limit='1GB'")
    con.execute('SET threads=1')
    temp = Path(report_dir)/'spill'
    temp.mkdir(parents=True,exist_ok=True)
    con.execute(f'SET temp_directory={sql_path(temp)}')
    register_sources(con,Path(data_dir))
    development,_ = development_context(prepared_dir)
    con.register('development_ids',development)
    con.register('prepared_ids',pd.read_parquet(Path(prepared_dir)/'row_ids.parquet'))
    for table in ['application','bureau','previous','cc','installments']:
        con.execute(f'CREATE VIEW dev_{table} AS SELECT h.* FROM raw_{table} h JOIN development_ids d USING(SK_ID_CURR)')
    return con


def semantic_checks(data_dir, prepared_dir, report_dir):
    """Distinguish hard constraints, allowed values and accounting questions."""
    prepared_dir,report_dir = Path(prepared_dir),Path(report_dir)
    con = _connection(data_dir,prepared_dir,report_dir)
    # condition, action and rationale are defined before examining tail magnitudes.
    specs = {
      'application':[
        ('AMT_INCOME_TOTAL','AMT_INCOME_TOTAL<=0','hard','NULL for income <=0; currently unnecessary','Positive income is required to interpret ratios; rare high income does not violate this rule.'),
        ('AMT_CREDIT','AMT_CREDIT<=0','hard','NULL for credit <=0; currently unnecessary','The current issued amount must be positive.'),
        ('AMT_ANNUITY','AMT_ANNUITY<0','hard','NULL for negative annuity; currently unnecessary','A negative regular obligation amount is inconsistent with the field description.'),
        ('AMT_GOODS_PRICE','AMT_GOODS_PRICE<0','hard','NULL for negative price; currently unnecessary','The price of purchased goods is nonnegative.'),
        ('CNT_FAM_MEMBERS','CNT_FAM_MEMBERS<1 OR CNT_FAM_MEMBERS<>floor(CNT_FAM_MEMBERS)','hard','NULL for family size <1 or noninteger; currently unnecessary','Family size includes the applicant and is a count.'),
        ('CNT_CHILDREN','CNT_CHILDREN<0 OR CNT_CHILDREN<>floor(CNT_CHILDREN)','hard','NULL for a negative/fractional count; currently unnecessary','The number of children is a nonnegative integer.'),
        ('CNT_FAM_MEMBERS/CNT_CHILDREN','CNT_FAM_MEMBERS<=CNT_CHILDREN','review','Review family composition; do not correct automatically','This condition assumes the applicant and children belong to the same family; other definitions require interpretation.'),
      ],
      'previous':[
        ('AMT_APPLICATION','AMT_APPLICATION<0','hard','NULL for a negative requested amount; currently unnecessary','The nominal requested amount is nonnegative.'),
        ('AMT_CREDIT','AMT_CREDIT<0','hard','NULL for a negative approved amount; currently unnecessary','The nominal credit amount is nonnegative.'),
        ('AMT_APPLICATION','AMT_APPLICATION=0','allowed','Retain','Canceled/rejected applications and other product types may have no requested amount.'),
        ('AMT_CREDIT','AMT_CREDIT=0','allowed','Retain','A zero disbursement is allowed for a loan that was not issued.'),
        ('AMT_DOWN_PAYMENT','AMT_DOWN_PAYMENT<0','review','Proposal: NULL before aggregation after confirming the sign semantics; retained for now','A negative down payment is suspicious but may be an accounting adjustment. Neither rounding nor replacement with 0 is justified.'),
      ],
      'bureau':[
        ('AMT_CREDIT_SUM','AMT_CREDIT_SUM<0','hard','NULL for a negative nominal amount; currently unnecessary','The nominal credit amount is nonnegative.'),
        ('AMT_CREDIT_SUM_DEBT','AMT_CREDIT_SUM_DEBT<0','review','Retain until overpayment accounting is clarified','Negative outstanding debt may indicate overpayment; a blanket AMT<0 rule is inappropriate.'),
        ('AMT_CREDIT_SUM_LIMIT','AMT_CREDIT_SUM_LIMIT<0','review','Retain; clarify the interpretation of the limit','The sign is unusual for a nominal limit, but the dictionary does not explain how adjustments are recorded.'),
      ],
      'cc':[
        ('AMT_BALANCE','AMT_BALANCE<0','allowed','Retain as a possible credit balance','A card balance may be negative after overpayment/refund.'),
        ('AMT_RECIVABLE','AMT_RECIVABLE<0','allowed','Retain as a possible balance adjustment','This is an accounting balance, not a nominal disbursement.'),
        ('AMT_DRAWINGS_ATM_CURRENT','AMT_DRAWINGS_ATM_CURRENT<0','review','Retain; check for a possible reversal','A negative cash withdrawal may correct a transaction; transaction details are unavailable.'),
        ('AMT_DRAWINGS_CURRENT','AMT_DRAWINGS_CURRENT<0','review','Retain; check for a possible reversal','The sign of a cash flow is not equivalent to a negative nominal loan amount.'),
        ('AMT_PAYMENT_CURRENT','AMT_PAYMENT_CURRENT<0','review','Retain; clarify payment refunds','A negative cash flow may have a valid accounting explanation.'),
        ('AMT_CREDIT_LIMIT_ACTUAL','AMT_CREDIT_LIMIT_ACTUAL=0','already_handled','Denominator 0 -> NULL utilization','A zero limit cannot be a divisor; the original value remains 0.'),
      ],
      'installments':[
        ('AMT_INSTALMENT','AMT_INSTALMENT<0','hard','NULL for a negative obligation; currently unnecessary','The scheduled installment amount is nonnegative.'),
        ('AMT_PAYMENT','AMT_PAYMENT<0','review','Check refunds; no negative values are currently present','A negative actual payment would require separate interpretation.'),
        ('AMT_INSTALMENT','AMT_INSTALMENT=0','allowed','Retain; ratio with a zero obligation amount -> NULL','Zero-valued schedule rows are not necessarily errors.'),
        ('AMT_PAYMENT','AMT_PAYMENT=0','allowed','Retain as an observed zero payment','Do not replace an absent payment with zero; this value was actually recorded.'),
      ]}
    filters={'application':'true','previous':'DAYS_DECISION<0',
             'bureau':'DAYS_CREDIT<0 AND DAYS_CREDIT_UPDATE<0','cc':'MONTHS_BALANCE<0',
             'installments':'DAYS_ENTRY_PAYMENT<0 AND DAYS_INSTALMENT IS NOT NULL AND DAYS_INSTALMENT<>365243'}
    rows=[]
    for table, checks in specs.items():
        expr=[]
        for i,(_,condition,_,_,_) in enumerate(checks):
            expr.extend([f'count(*) FILTER(WHERE {condition}) AS n{i}',
                         f'count(DISTINCT SK_ID_CURR) FILTER(WHERE {condition}) AS clients{i}'])
        result=con.execute(f"SELECT {','.join(expr)} FROM dev_{table} WHERE {filters[table]}").fetchone()
        for i,(field,condition,status,action,reason) in enumerate(checks):
            rows.append({'source':table,'field':field,'scope':'development, time-eligible raw rows',
                         'condition':condition,'affected_values':result[2*i],'affected_clients':result[2*i+1],
                         'classification':status,'proposed_action_ru':action,'reason_ru':reason,
                         'applied_now':False})
    raw_checks=_save(pd.DataFrame(rows),report_dir,'semantic_source_checks.csv')
    # Exact impact of the sole concrete proposal: negative down payment -> NULL.
    proposed=con.execute(f"""WITH alt AS (
      SELECT SK_ID_CURR,avg(CASE WHEN AMT_DOWN_PAYMENT>=0 THEN AMT_DOWN_PAYMENT END) a,
      min(CASE WHEN AMT_DOWN_PAYMENT>=0 THEN AMT_DOWN_PAYMENT END) n,
      max(CASE WHEN AMT_DOWN_PAYMENT>=0 THEN AMT_DOWN_PAYMENT END) x
      FROM dev_previous WHERE DAYS_DECISION<0 GROUP BY SK_ID_CURR)
      SELECT p.PREVIOUS_AMT_DOWN_PAYMENT_AVG AS old_avg,a AS new_avg,
      p.PREVIOUS_AMT_DOWN_PAYMENT_MIN AS old_min,n AS new_min,
      p.PREVIOUS_AMT_DOWN_PAYMENT_MAX AS old_max,x AS new_max
      FROM alt JOIN read_parquet({sql_path(prepared_dir/'aggregates/previous.parquet')}) p USING(SK_ID_CURR)""").df()
    differences={}
    for stat in ['avg','min','max']:
        differences[stat]=~np.isclose(proposed[f'old_{stat}'],proposed[f'new_{stat}'],rtol=1e-10,atol=1e-10,equal_nan=True)
    impact={'proposal':'previous AMT_DOWN_PAYMENT<0 -> NULL before aggregation',
            'raw_values':int(raw_checks.loc[(raw_checks.source=='previous')&(raw_checks.field=='AMT_DOWN_PAYMENT'),'affected_values'].iloc[0]),
            'changed_aggregate_cells':int(sum(mask.sum() for mask in differences.values())),
            'changed_clients':int(np.logical_or.reduce(list(differences.values())).sum()),
            'per_feature_changed_cells':{k:int(v.sum()) for k,v in differences.items()},
            'applied':False,'reason':'First confirm that -0.9 is not a valid accounting adjustment.'}
    dump_json(report_dir/'proposed_fix_impact.json',impact)
    # All computed ratios and shares are checked on development.
    dev,positions=development_context(prepared_dir)
    cols=json.loads((prepared_dir/'features_level_4.json').read_text(encoding='utf-8'))
    ratios=[s for s in cols if s.startswith('APP_') and s!='APP_EMPLOYMENT_DATE_UNKNOWN'
            or s.endswith('_SHARE') or 'UTILIZATION' in s or 'TO_' in s and s.startswith(('BUREAU_','PREVIOUS_','INSTALLMENTS_'))]
    frame=pd.read_parquet(prepared_dir/'X_level_4.parquet',columns=ratios).iloc[positions]
    ratio_rows=[]
    for name,x in frame.items():
        share=name.endswith('_SHARE')
        nonnegative=name.startswith('APP_') or name in ['PREVIOUS_CREDIT_TO_REQUEST_AVG','INSTALLMENTS_PAID_TO_DUE']
        invalid=int(((x<0)|(x>1)).sum()) if share else int((x<0).sum()) if nonnegative else 0
        explanation=('A share must lie in [0,1].' if share else
          'A positive amount ratio need not be <=1; income and annuity may refer to different periods.' if name.startswith('APP_') else
          'Utilization>1 indicates an exceeded limit; <0 is possible for a credit balance. Do not clip automatically.' if 'UTILIZATION' in name else
          'Negative debt is possible after overpayment; debt/credit need not lie in [0,1].' if name=='BUREAU_DEBT_TO_CREDIT' else
          'Payment/obligation>1 may indicate overpayment; approved credit may also exceed the original request.')
        ratio_rows.append({'feature':name,'missing':int(x.isna().sum()),'negative':int((x<0).sum()),
                           'above_one':int((x>1).sum()),'infinite':int(np.isinf(x).sum()),
                           'rule_violations':invalid,'rule_ru':explanation,'applied_correction':False})
    ratio_report=_save(pd.DataFrame(ratio_rows),report_dir,'ratio_checks_development.csv')
    # Describe extrema only; do not use them as removal thresholds.
    stats=pd.read_csv(report_dir/'numeric_development_summary.csv')
    focus=['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE','CNT_FAM_MEMBERS','CNT_CHILDREN']+ratios
    _save(stats[stats.feature.isin(focus)],report_dir,'financial_and_ratio_distribution.csv')
    tail=pd.read_parquet(prepared_dir/'X_level_4.parquet',columns=['AMT_INCOME_TOTAL']).iloc[positions].AMT_INCOME_TOTAL
    income_tail={'p99':float(tail.quantile(.99)),'p999':float(tail.quantile(.999)),
                 'max':float(tail.max()),'above_p999':int((tail>tail.quantile(.999)).sum()),
                 'removed':0,'winsorized':0,'note':'Quantiles were computed on development only and are not cleaning thresholds.'}
    dump_json(report_dir/'income_tail_review.json',income_tail)
    con.close()
    return {'hard_rule_violations':int(raw_checks.loc[raw_checks.classification=='hard','affected_values'].sum()),
            'ratio_rule_violations':int(ratio_report.rule_violations.sum()),
            'proposed_fix':impact,'income_tail':income_tail,
            'review_cases':raw_checks[(raw_checks.classification=='review')&(raw_checks.affected_values>0)].to_dict('records'),
            'values_modified':0}


def sentinel_checks(data_dir, prepared_dir, report_dir):
    """Check the source, rule and actually saved application/previous data."""
    prepared_dir, report_dir=Path(prepared_dir),Path(report_dir)
    con=_connection(data_dir,prepared_dir,report_dir)
    result=[]
    app_path=sql_path(prepared_dir/'aggregates/application.parquet')
    for scope,source in [('all_source','raw_application'),('development','dev_application')]:
        for col in ['DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH','DAYS_LAST_PHONE_CHANGE']:
            n,before,missing,positive,after,remaining,bad = con.execute(f"""SELECT count(*),
                count(*) FILTER(WHERE r.{col}=365243), count(*) FILTER(WHERE r.{col} IS NULL),
                count(*) FILTER(WHERE r.{col}>0), count(*) FILTER(WHERE a.{col} IS NULL),
                count(*) FILTER(WHERE a.{col}=365243),
                count(*) FILTER(WHERE a.{col} IS DISTINCT FROM (CASE WHEN r.{col}<=0 THEN r.{col} END))
                FROM {source} r JOIN read_parquet({app_path}) a USING(SK_ID_CURR)""").fetchone()
            assert remaining==0 and bad==0
            result.append({'source':'application','field':col,'scope':scope,'source_rows':n,
                           'placeholder_before':before,'null_before':missing,'positive_before':positive,
                           'null_after_same_rows':after,'placeholder_after_same_rows':remaining,
                           'saved_rule_mismatches':bad,'rule':'positive factual date -> NULL; <=0 retained',
                           'saved_feature_status':'included'})
    indicator_bad=con.execute(f"""SELECT count(*) FROM raw_application r JOIN read_parquet({app_path}) a USING(SK_ID_CURR)
      WHERE a.APP_EMPLOYMENT_DATE_UNKNOWN<>CAST(r.DAYS_EMPLOYED IS NULL OR r.DAYS_EMPLOYED>0 AS INTEGER)""").fetchone()[0]
    assert indicator_bad==0
    levels=json.loads((prepared_dir/'features_level_4.json').read_text(encoding='utf-8'))
    previous_dates=['DAYS_FIRST_DRAWING','DAYS_FIRST_DUE','DAYS_LAST_DUE_1ST_VERSION','DAYS_LAST_DUE','DAYS_TERMINATION']
    for scope,source in [('all_source','raw_previous'),('development','dev_previous')]:
        # Before/after rows for aggregates are restricted to the same eligible records.
        for col in previous_dates:
            n,sentinels,missing,planned_positive=con.execute(f"SELECT count(*),count(*) FILTER(WHERE {col}=365243),count(*) FILTER(WHERE {col} IS NULL),count(*) FILTER(WHERE {col}>0 AND {col}<>365243) FROM {source} WHERE DAYS_DECISION<0").fetchone()
            included=col in ['DAYS_FIRST_DUE','DAYS_LAST_DUE_1ST_VERSION']
            feature_names=[f'PREVIOUS_{col}_{stat}' for stat in ['AVG','MIN','MAX']]
            assert all(name in levels for name in feature_names) if included else not any(name in levels for name in feature_names)
            result.append({'source':'previous','field':col,'scope':scope,'source_rows':n,
                           'placeholder_before':sentinels,'null_before':missing,'positive_before':planned_positive,
                           'null_after_same_rows':missing+sentinels if included else None,
                           'placeholder_after_same_rows':0 if included else None,
                           'saved_rule_mismatches':0 if not included else None,
                           'rule':'365243 -> NULL; positive planned dates retained' if included else 'field excluded before feature construction',
                           'saved_feature_status':'included' if included else 'excluded'})
    for scope,source in [('all_source','raw_bureau'),('development','dev_bureau')]:
        n,before,missing,positive=con.execute(f"""SELECT count(*),
          count(*) FILTER(WHERE DAYS_CREDIT_ENDDATE=365243),
          count(*) FILTER(WHERE DAYS_CREDIT_ENDDATE IS NULL),
          count(*) FILTER(WHERE DAYS_CREDIT_ENDDATE>0 AND DAYS_CREDIT_ENDDATE<>365243)
          FROM {source} WHERE DAYS_CREDIT<0 AND DAYS_CREDIT_UPDATE<0""").fetchone()
        result.append({'source':'bureau','field':'DAYS_CREDIT_ENDDATE','scope':scope,
          'source_rows':n,'placeholder_before':before,'null_before':missing,
          'positive_before':positive,'null_after_same_rows':missing+before,
          'placeholder_after_same_rows':0,'saved_rule_mismatches':None,
          'rule':'365243 -> NULL; positive planned dates retained','saved_feature_status':'included'})
    # Recompute saved date aggregates for ALL prepared clients, including NULL equality.
    # This is a mechanical integrity check; no test distribution guides any decisions.
    checks=[]
    for col in ['DAYS_FIRST_DUE','DAYS_LAST_DUE_1ST_VERSION']:
        expected=con.execute(f"SELECT SK_ID_CURR,avg(NULLIF({col},365243)) AS avg,min(NULLIF({col},365243)) AS min,max(NULLIF({col},365243)) AS max FROM raw_previous JOIN prepared_ids USING(SK_ID_CURR) WHERE DAYS_DECISION<0 GROUP BY SK_ID_CURR ORDER BY SK_ID_CURR").df().set_index('SK_ID_CURR')
        actual=pd.read_parquet(prepared_dir/'aggregates/previous.parquet',columns=['SK_ID_CURR']+[f'PREVIOUS_{col}_{s}' for s in ['AVG','MIN','MAX']]).set_index('SK_ID_CURR').loc[expected.index]
        for stat in ['avg','min','max']:
            saved=actual[f'PREVIOUS_{col}_{stat.upper()}']
            mismatches=int((~np.isclose(saved,expected[stat],rtol=1e-10,atol=1e-10,equal_nan=True)).sum())
            assert mismatches==0
            checks.append({'feature':f'PREVIOUS_{col}_{stat.upper()}','prepared_clients_with_history':len(saved),
                           'mismatches_to_cleaned_history':mismatches,'positive_saved_values':int((saved>0).sum()),
                           'remaining_365243':int((saved==365243).sum())})
    numeric=pd.read_csv(report_dir/'numeric_development_summary.csv')
    date_stats=numeric[numeric.placeholder_365243.notna()]
    assert int(date_stats.placeholder_365243.sum())==0
    date_cols=[col for col in levels if 'DAYS_' in col]
    final_dates=pd.read_parquet(prepared_dir/'X_level_4.parquet',columns=date_cols)
    remaining_all=int(final_dates.eq(365243).sum().sum())
    assert remaining_all==0
    _save(pd.DataFrame(result),report_dir,'sentinels_before_after.csv')
    _save(pd.DataFrame(checks),report_dir,'saved_date_aggregate_checks.csv')
    con.close()
    return {'source_before_after_rows':len(result),'application_rule_mismatches':0,
            'employment_indicator_mismatches':indicator_bad,'saved_previous_aggregate_mismatches':0,
            'development_saved_date_placeholders':0,
            'all_prepared_saved_date_placeholders':remaining_all,
            'note':'Before/after NULL counts refer to original rows; aggregate checks are saved separately. After-processing counts for excluded fields are not applicable.'}
