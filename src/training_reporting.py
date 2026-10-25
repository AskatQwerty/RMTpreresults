"""Независимые отчёты из артефактов: не требуют состояния технического notebook.

Модели не загружаются и не обучаются. Сводки technical/full пишутся раздельно;
завершёнными считаются только этапы с корректным complete.json и SHA-256.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import psutil

from .data_quality import artifact_manifest


MODELS=('logistic_regression','random_forest','xgboost','lightgbm','mlp','soft_voting','stacking')
RESOURCE_METRIC_COLUMNS=('run','model','status','rows','raw_features','processed_features_json',
    'fit_seconds','worker_elapsed_seconds','peak_process_tree_rss_bytes','fit_peak_process_tree_rss_bytes',
    'windows_main_peak_wset_bytes','peak_ram_gib','model_size_bytes','warning_count','mlp_n_iter_json','resumed')


def _load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str,allow_nan=False),encoding='utf-8')
    os.replace(tmp,path)


def _hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda:source.read(8*1024**2),b''): digest.update(chunk)
    return digest.hexdigest()


def _check_stage(stage,fingerprint):
    """Вернуть ошибку проверки или None; никаких изменений в этапах эксперимента."""
    try:
        record=_load(stage/'complete.json')
        if record.get('fingerprint')!=fingerprint: return 'fingerprint не совпадает с manifest'
        hashes=record.get('artifact_sha256',{})
        if not hashes: return 'complete.json не содержит хэшей артефактов'
        for name,expected in hashes.items():
            path=stage/name
            if not path.is_file(): return f'Нет файла {name}'
            if _hash(path)!=expected: return f'SHA-256 не совпадает: {name}'
    except (OSError,ValueError,KeyError,TypeError) as exc:
        return f'{type(exc).__name__}: {exc}'
    return None


def _lock_info(root):
    path=root/'run.lock'
    if not path.exists(): return {'present':False,'owner_alive':False}
    try:
        record=_load(path)
        pid=record.get('pid')
        alive=bool(pid and psutil.pid_exists(int(pid)))
        return {'present':True,'pid':pid,'owner_alive':alive,
                'note':'Живой PID не доказывает прогресс fit; lock не удаляется отчётом.'}
    except (OSError,ValueError,TypeError):
        return {'present':True,'owner_alive':None,'note':'Lock не удалось прочитать; отчёт его не изменяет.'}


def _input_check(manifest,prepared_dir):
    if prepared_dir is None: return {'checked':False,'unchanged':None,'reason':'Не передан prepared_dir'}
    if 'inputs' not in manifest: return {'checked':False,'unchanged':None,'reason':'В manifest нет снимка inputs'}
    actual=artifact_manifest(Path(prepared_dir))
    return {'checked':True,'unchanged':actual==manifest['inputs'],
            'files_checked':len(actual),'baseline':'experiment_manifest.inputs'}


def _collect(root,expected_scope):
    manifest_path=root/'experiment_manifest.json'
    if not manifest_path.exists():
        return {'scope':expected_scope,'root':str(root.resolve()),'status':'not_started',
                'searches':[],'fits':[],'invalid_stages':[],'manifest':None,'lock':_lock_info(root)}
    manifest=_load(manifest_path)
    if manifest.get('mode')!=expected_scope:
        raise ValueError(f'Ожидался {expected_scope}, но manifest имеет mode={manifest.get("mode")}')
    searches=[]; fits=[]; invalid=[]
    for marker in sorted(root.glob('level_*/*/*/complete.json')):
        stage=marker.parent
        error=_check_stage(stage,manifest.get('fingerprint'))
        if error:
            invalid.append({'stage':str(stage.relative_to(root)),'error':error}); continue
        try:
            if stage.name=='search':
                record=_load(stage/'best_params.json')
                if record.get('scope')!=expected_scope: raise ValueError('scope поиска не совпадает')
                searches.append(record)
            elif stage.name.startswith('seed_'):
                record=_load(stage/'metrics.json')
                if record.get('scope')!=expected_scope: raise ValueError('scope fit не совпадает')
                fits.append(record)
        except (OSError,ValueError,KeyError,TypeError) as exc:
            invalid.append({'stage':str(stage.relative_to(root)),'error':str(exc)})
    return {'scope':expected_scope,'root':str(root.resolve()),'status':'partial',
            'searches':searches,'fits':fits,'invalid_stages':invalid,'manifest':manifest,'lock':_lock_info(root)}


def _tables(state,report_dir,prefix):
    report_dir=Path(report_dir); report_dir.mkdir(parents=True,exist_ok=True)
    fits=state['fits']; searches=state['searches']
    rows=[]; warning_rows=[]
    for record in fits:
        row={key:value for key,value in record.items()
             if key not in ['metrics','warnings','selected_params','fit_diagnostics','training_diagnostics']
             and isinstance(value,(str,int,float,bool,type(None)))}
        row.update(record.get('metrics',{}))
        row['warning_count']=sum(w.get('count',1) for w in record.get('warnings',[]))
        rows.append(row)
    for stage,records in [('search',searches),('fit',fits)]:
        for record in records:
            for warning in record.get('warnings',[]):
                warning_rows.append({'stage':stage,'model':record.get('model'),
                    'level':record.get('level'),'seed':record.get('seed'),**warning})
    metrics=pd.DataFrame(rows)
    warnings=pd.DataFrame(warning_rows,columns=['stage','model','level','seed','category','message','count'])
    metrics.to_csv(report_dir/f'{prefix}_metrics.csv',index=False)
    warnings.to_csv(report_dir/f'{prefix}_warnings.csv',index=False)
    if not metrics.empty:
        names=[name for name in ['pr_auc_average_precision','roc_auc','log_loss','brier_score','fit_seconds'] if name in metrics]
        aggregates=metrics.groupby(['level','model'])[names].agg(['mean','std','count'])
        aggregates.columns=['_'.join(item) for item in aggregates.columns]
        aggregates.reset_index().to_csv(report_dir/f'{prefix}_model_summary.csv',index=False)
    return {'warning_count':sum(w.get('count',1) for w in warning_rows),
            'warning_categories':sorted({str(w.get('category')) for w in warning_rows}),
            'search_seconds_completed':sum(float(s.get('elapsed_seconds',0)) for s in searches),
            'fit_seconds_completed':sum(float(f.get('fit_seconds',0)) for f in fits)}


def write_full_report(full_output,report_dir,*,prepared_dir=None):
    """Сформировать полный/частичный отчёт без technical и иных notebook-переменных."""
    root=Path(full_output); report_dir=Path(report_dir)
    state=_collect(root,'full')
    expected_search={(level,model) for level in range(1,5) for model in MODELS}
    expected_fit={(level,model,seed) for level in range(1,5) for model in MODELS for seed in range(42,47)}
    actual_search={(r.get('level'),r.get('model')) for r in state['searches']}
    actual_fit={(r.get('level'),r.get('model'),r.get('seed')) for r in state['fits']}
    invalid=state['invalid_stages']
    complete=actual_search==expected_search and actual_fit==expected_fit and not invalid
    if invalid: status='invalid_artifacts'
    elif complete: status='completed'
    elif state['manifest'] is None: status='not_started'
    elif state['lock'].get('owner_alive'): status='in_progress'
    else: status='partial_or_interrupted'
    input_state=_input_check(state['manifest'],prepared_dir) if state['manifest'] else {'checked':False,'unchanged':None}
    timing=_tables(state,report_dir,'full')
    summary={'scope':'full','root':str(root.resolve()),'status':status,
        'expected_searches':28,'expected_main_fits':140,'completed_searches':len(actual_search),
        'completed_main_fits':len(actual_fit),'completed_stage_hashes_checked':len(actual_search)+len(actual_fit),
        'full_plan_complete':complete,'invalid_stages':invalid,'lock':state['lock'],
        'input_integrity':input_state,'independent_of_technical_variables':True,
        'metrics_are_final_test_evaluations':True,'existing_experiment_summary':(root/'experiment_summary.json').exists(),
        **timing}
    _write(report_dir/'full_summary.json',summary)
    return summary


def write_technical_report(technical_root,report_dir,*,prepared_dir=None):
    """Отдельная сводка одного технического запуска; она не оценивает полный план."""
    root=Path(technical_root); report_dir=Path(report_dir)
    state=_collect(root,'technical')
    summary_path=root/'technical_summary.json'
    saved=_load(summary_path) if summary_path.exists() else {}
    invalid=state['invalid_stages']
    complete=len(state['searches'])==7 and len(state['fits'])==7 and not invalid
    status='invalid_artifacts' if invalid else 'completed' if complete else 'not_started' if state['manifest'] is None else 'partial'
    input_state=_input_check(state['manifest'],prepared_dir) if state['manifest'] else {'checked':False,'unchanged':None}
    timing=_tables(state,report_dir,'technical')
    summary={'scope':'technical','root':str(root.resolve()),'status':status,
        'completed_searches':len(state['searches']),'completed_fits':len(state['fits']),
        'training_level':saved.get('training_level',4),'train_rows':saved.get('train_rows'),
        'validation_rows':saved.get('validation_rows'),'test_used':False,
        'run_counters_last_invocation':saved.get('run_counters'),
        'invalid_stages':invalid,'input_integrity':input_state,**timing}
    _write(report_dir/'technical_summary.json',summary)
    return summary


def write_technical_reports(technical_parent,report_dir,*,prepared_dir=None):
    """Найти сохранённые технические запуски по конфигурационному пути."""
    parent=Path(technical_parent); report_dir=Path(report_dir)
    roots=sorted(p.parent for p in parent.glob('*/experiment_manifest.json')) if parent.exists() else []
    runs=[write_technical_report(root,report_dir/root.name,prepared_dir=prepared_dir) for root in roots]
    result={'scope':'technical_collection','runs_found':len(runs),'runs':runs,
            'status':'available' if runs else 'not_run','full_experiment_status_not_inferred':True}
    _write(report_dir/'technical_runs.json',result)
    return result


def _resource_rows(saved,run):
    """Строки из атомарного summary: Parquet может отставать после interruption."""
    rows=[]; warnings=[]
    for item in saved.get('results',[]):
        peak=item.get('peak_ram_bytes')
        rows.append({'run':run,'model':item['model'],'status':item['status'],
            'rows':item.get('rows'),'raw_features':item.get('raw_features'),
            'processed_features_json':json.dumps(item.get('processed_features',[]),ensure_ascii=False),
            'fit_seconds':item.get('fit_seconds'),'worker_elapsed_seconds':item.get('worker_elapsed_seconds'),
            'peak_process_tree_rss_bytes':item.get('peak_process_tree_rss_bytes'),
            'fit_peak_process_tree_rss_bytes':item.get('fit_peak_process_tree_rss_bytes'),
            'windows_main_peak_wset_bytes':item.get('windows_main_peak_wset_bytes'),
            'peak_ram_gib':peak/1024**3 if peak is not None else None,
            'model_size_bytes':item.get('model_size_bytes'),
            'warning_count':sum(w.get('count',1) for w in item.get('warnings',[])),
            'mlp_n_iter_json':json.dumps(item.get('actual_mlp_iterations',[])),
            'resumed':item.get('resumed',False)})
        for warning in item.get('warnings',[]):
            warnings.append({'run':run,'model':item['model'],'status':item['status'],**warning})
    return rows,warnings


def write_resource_reports(resource_parent,report_dir):
    """Сводка отдельных пилотов из summary.results, без повторного fit или test."""
    parent=Path(resource_parent); report_dir=Path(report_dir)
    summaries=sorted(parent.glob('*/resource_summary.json')) if parent.exists() else []
    runs=[]; all_rows=[]; all_warnings=[]
    for source in summaries:
        saved=_load(source)
        if saved.get('scope')!='resource_pilot':
            raise ValueError(f'В {source} ожидался scope=resource_pilot')
        root=source.parent
        destination=report_dir/root.name
        _write(destination/'resource_summary.json',saved)
        rows,warnings=_resource_rows(saved,root.name)
        all_rows.extend(rows); all_warnings.extend(warnings)
        pd.DataFrame(rows,columns=RESOURCE_METRIC_COLUMNS).to_csv(destination/'resource_metrics.csv',index=False)
        pd.DataFrame(warnings,columns=['run','model','status','category','message','count']).to_csv(
            destination/'resource_warnings.csv',index=False)
        runs.append({'root':str(root.resolve()),'status':saved.get('status'),
            'completed_models':saved.get('completed_models'),'attempted_models':saved.get('attempted_models'),
            'metrics_csv':str((destination/'resource_metrics.csv').resolve()),
            'warnings_csv':str((destination/'resource_warnings.csv').resolve()),
            'saved_summary':saved})
    report_dir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(all_rows,columns=RESOURCE_METRIC_COLUMNS).to_csv(report_dir/'resource_metrics.csv',index=False)
    pd.DataFrame(all_warnings,columns=['run','model','status','category','message','count']).to_csv(
        report_dir/'resource_warnings.csv',index=False)
    result={'scope':'resource_pilot_collection','runs_found':len(runs),'runs':runs,
            'status':'available' if runs else 'not_run',
            'independent_of_main_experiment':True,'test_metrics_not_used':True,
            'metrics_source':'resource_summary.results (atomic snapshot)',
            'model_metrics':all_rows,'model_metrics_csv':str((report_dir/'resource_metrics.csv').resolve()),
            'warnings_csv':str((report_dir/'resource_warnings.csv').resolve())}
    _write(report_dir/'resource_runs.json',result)
    return result
