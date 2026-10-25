"""Отдельный ресурсный пилот: один заранее заданный full fit на тип, без test/search.

Каждый тип обучается в новом дочернем процессе. Это освобождает память предыдущей
модели и позволяет измерять peak working set/RSS, включая реальную предобработку.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
import traceback
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from .data_quality import artifact_manifest
from .fit_diagnostics import extract_fit_diagnostics
from .model_pipelines import MODEL_NAMES, experiment_spec, make_estimator
from .model_training import (PreparedData, PACKAGES, atomic_json, read_json, sha256,
    initialize_run, run_lock, _complete, _mark_complete, _atomic_model,
    _atomic_parquet, _warnings_report, _json_text)


def _plain_config(value):
    if isinstance(value,np.generic): return _plain_config(value.item())
    if isinstance(value,float) and not math.isfinite(value): return str(value)
    if value is None or isinstance(value,(str,int,float,bool)): return value
    if isinstance(value,(tuple,list)): return [_plain_config(v) for v in value]
    if isinstance(value,dict): return {str(k):_plain_config(v) for k,v in value.items()}
    if hasattr(value,'get_n_splits'):
        return {'class':type(value).__name__,'n_splits':value.n_splits,
                'shuffle':value.shuffle,'random_state':value.random_state}
    raise TypeError(f'Неожиданная настройка модели: {type(value)}')


def describe_configuration(estimator):
    """Полные фиксированные настройки classifier и каждой базы, до fit."""
    if isinstance(estimator,Pipeline):
        return {'pipeline_steps':[name for name,_ in estimator.steps],
                'final':describe_configuration(estimator.steps[-1][1])}
    params=estimator.get_params(deep=False)
    description={'class':type(estimator).__name__,
                 'parameters':{k:_plain_config(v) for k,v in params.items()
                               if k not in {'estimators','final_estimator'}}}
    if 'estimators' in params:
        description['bases']={name:describe_configuration(base) for name,base in params['estimators']}
    if params.get('final_estimator') is not None:
        description['meta']=describe_configuration(params['final_estimator'])
    return description


def resource_context(data,threads):
    numeric,categorical=data.columns(4)
    selected={name:describe_configuration(make_estimator(name,numeric,categorical,
        seed=42,technical=False,threads=threads)) for name in MODEL_NAMES}
    root=Path(__file__).resolve().parent
    source_names=['resource_pilot.py','resource_monitor.py','model_training.py',
                  'model_pipelines.py','fold_preprocessing.py','fit_diagnostics.py']
    context={'mode':'resource_pilot','level':4,'seed':42,'train_split':'train_seed_42',
        'train_rows':len(data.repeats[42]),'raw_features':len(data.features['4']),
        'threads':threads,'technical':False,'randomized_search':False,'test_used':False,
        'outer_fits_planned':7,'primitive_fits_planned':21,
        'configuration_selection':'factory full defaults fixed before fits; no technical/test metrics used',
        'selected_configurations':selected,'inputs':artifact_manifest(data.path),
        'versions':{name:importlib.metadata.version(name) for name in PACKAGES},
        'python':platform.python_version(),
        'code_sha256':{name:sha256(root/name) for name in source_names},
        'memory_monitor':{'sample_interval_seconds':.1,'min_available_bytes':512*1024**2,
                          'low_memory_grace_seconds':3,'scope':'own worker and descendants only'},
        'main_experiment_budget_changed':False}
    context['fingerprint']=hashlib.sha256(_json_text(context).encode('utf-8')).hexdigest()
    return context


def _phase(stage,name,**extras):
    atomic_json(Path(stage)/'phase.json',{'phase':name,'pid':os.getpid(),
        'wall_time':time.time(),**extras})


def worker(prepared_dir,root,model):
    """Один fit; не вызываются RandomizedSearchCV, predict на test или выбор параметров."""
    root,prepared_dir=Path(root),Path(prepared_dir)
    stage=root/model
    context=read_json(root/'experiment_manifest.json')
    stage.mkdir(parents=True,exist_ok=True)
    caught=[]
    started=time.perf_counter()
    try:
        _phase(stage,'load')
        data=PreparedData(prepared_dir)
        ids=data.repeats[42]
        assert set(ids)<=set(data.development) and set(ids).isdisjoint(data.test)
        X,y=data.load(4,ids,purpose='technical_resource_train_seed_42')
        # Префикс technical в purpose включает запрет test, но рабочие настройки full.
        assert np.array_equal(X.index,ids) and X.index.equals(y.index)
        numeric,categorical=data.columns(4)
        estimator=make_estimator(model,numeric,categorical,seed=42,
            technical=False,threads=context['threads'])
        selected=describe_configuration(estimator)
        assert selected==context['selected_configurations'][model]
        atomic_json(stage/'selected_configuration.json',selected)
        baseline=psutil.Process().memory_info().rss
        _phase(stage,'fit',rows=len(X),raw_features=X.shape[1],rss_before_fit_bytes=baseline)
        print(f'{model}: full settings, rows={len(X)}, raw_features={X.shape[1]}; fit started',flush=True)
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=context['threads']):
            warnings.simplefilter('always')
            begin=time.perf_counter()
            estimator.fit(X,y)
            fit_seconds=time.perf_counter()-begin
        warning_list=_warnings_report(caught)
        _phase(stage,'diagnostics',fit_seconds=fit_seconds)
        diagnostics=extract_fit_diagnostics(estimator)
        atomic_json(stage/'fit_diagnostics.json',diagnostics)
        atomic_json(stage/'warnings.json',warning_list)
        rows,raw_features=len(X),X.shape[1]
        del X,y,data; gc.collect()
        _phase(stage,'serialize')
        begin=time.perf_counter()
        _atomic_model(estimator,stage/'pipeline.joblib')
        serialization_seconds=time.perf_counter()-begin
        # Проверяем, что фактические атрибуты MLP переживают joblib-сериализацию.
        # Удаляем оригинал до загрузки, чтобы не удваивать память большой RF.
        del estimator; gc.collect()
        restored=joblib.load(stage/'pipeline.joblib')
        assert extract_fit_diagnostics(restored)==diagnostics
        del restored; gc.collect()
        result={'model':model,'status':'completed','scope':'resource_pilot','level':4,'seed':42,
            'training_split':'train_seed_42','rows':rows,'raw_features':raw_features,
            'processed_features':diagnostics['processed_feature_counts'],
            'fit_seconds':fit_seconds,'serialization_seconds':serialization_seconds,
            'worker_elapsed_seconds':time.perf_counter()-started,
            'rss_before_fit_bytes':baseline,'model_size_bytes':(stage/'pipeline.joblib').stat().st_size,
            'warnings':warning_list,'actual_mlp_iterations':[r['fitted'].get('n_iter_') for r in diagnostics['mlp']],
            'diagnostics_file':'fit_diagnostics.json','selected_configuration_file':'selected_configuration.json',
            'serialization_diagnostics_equal':True,'test_used':False,'randomized_search_used':False,
            'outer_fit_count':1,'primitive_fit_count':experiment_spec(model)['primitive_fits_per_outer_fit']}
        atomic_json(stage/'worker_result.json',result)
        _phase(stage,'completed',fit_seconds=fit_seconds)
        print(f'{model}: fit completed {fit_seconds:.1f}s; model {(stage/"pipeline.joblib").stat().st_size} bytes',flush=True)
        return 0
    except BaseException as exc:
        atomic_json(stage/'warnings.json',_warnings_report(caught))
        atomic_json(stage/'failure.json',{'error_type':type(exc).__name__,'error':str(exc),
            'traceback':traceback.format_exc()[-16000:],'elapsed_seconds':time.perf_counter()-started})
        _phase(stage,'failed',error_type=type(exc).__name__,error=str(exc))
        traceback.print_exc()
        return 1


def _metrics_frame(results):
    records=[]
    for item in results:
        feature_counts=item.get('processed_features',[])
        records.append({'model':item['model'],'status':item['status'],'rows':item.get('rows'),
            'raw_features':item.get('raw_features'),'processed_features_json':json.dumps(feature_counts,ensure_ascii=False),
            'fit_seconds':item.get('fit_seconds'),'worker_elapsed_seconds':item.get('worker_elapsed_seconds'),
            'peak_process_tree_rss_bytes':item.get('peak_process_tree_rss_bytes'),
            'fit_peak_process_tree_rss_bytes':item.get('fit_peak_process_tree_rss_bytes'),
            'windows_main_peak_wset_bytes':item.get('windows_main_peak_wset_bytes'),
            'peak_ram_gib':item.get('peak_ram_bytes',0)/1024**3 if item.get('peak_ram_bytes') is not None else None,
            'model_size_bytes':item.get('model_size_bytes'),
            'warning_count':sum(w['count'] for w in item.get('warnings',[])),
            'mlp_n_iter_json':json.dumps(item.get('actual_mlp_iterations',[])),
            'resumed':item.get('resumed',False)})
    return pd.DataFrame(records)


def run_resource_pilot(prepared_dir,output_parent,*,threads=2):
    """Выполнить семь отдельных полных fit, сохраняя каждый независимо от остальных.

    Никакого поиска/оценки по test. При interruption завершённые worker-этапы
    проверяются и пропускаются. Неуспешная конфигурация не заменяется более лёгкой.
    """
    from .resource_monitor import monitor_subprocess
    data=PreparedData(prepared_dir)
    context=resource_context(data,threads)
    root=Path(output_parent)/f'resource_{context["fingerprint"][:12]}'
    root.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    results=[]
    with run_lock(root):
        initialize_run(root,context)
        atomic_json(root/'selected_configurations.json',context['selected_configurations'])
        for model in MODEL_NAMES:
            stage=root/model; stage.mkdir(exist_ok=True)
            if _complete(stage,context):
                result=read_json(stage/'resource_result.json'); result['resumed']=True
            else:
                atomic_json(stage/'selected_configuration.json',context['selected_configurations'][model])
                # Удаляется только устаревший промежуточный outcome этого незавершённого этапа.
                # Модель/diagnostics сохраняются до успешной атомарной перезаписи.
                (stage/'worker_result.json').unlink(missing_ok=True)
                _phase(stage,'launch_pending')
                print(f'Resource pilot: {model}; available RAM {psutil.virtual_memory().available/1024**3:.2f} GiB',flush=True)
                monitor=monitor_subprocess([sys.executable,'-m','src.resource_pilot','--worker',
                    '--prepared',str(Path(prepared_dir).resolve()),'--root',str(root.resolve()),'--model',model],
                    cwd=Path(__file__).resolve().parents[1],log_path=stage/'worker.log',
                    phase_path=stage/'phase.json',samples_path=stage/'memory_samples.csv')
                atomic_json(stage/'resource_monitor.json',monitor)
                if monitor['exit_code']==0 and (stage/'worker_result.json').exists():
                    result=read_json(stage/'worker_result.json')
                else:
                    result={'model':model,'status':monitor['status'] if monitor['status']!='completed' else 'failed','scope':'resource_pilot',
                        'rows':context['train_rows'],'raw_features':context['raw_features'],
                        'warnings':read_json(stage/'warnings.json') if (stage/'warnings.json').exists() else [],
                        'error':read_json(stage/'failure.json') if (stage/'failure.json').exists() else
                                {'reason':monitor['status'],'exit_code':monitor['exit_code']}}
                result.update({'peak_process_tree_rss_bytes':monitor['peak_tree_rss_bytes'],
                    'fit_peak_process_tree_rss_bytes':monitor['fit_peak_tree_rss_bytes'],
                    'windows_main_peak_wset_bytes':monitor['windows_main_peak_wset_bytes'],
                    'peak_ram_bytes':max(monitor['peak_tree_rss_bytes'] or 0,monitor['windows_main_peak_wset_bytes'] or 0),
                    'memory_measurement':'sampled100ms worker+children RSS; Windows main lifetime peak_wset also recorded. Includes runtime/data/preprocessing/fit/serialization.',
                    'min_available_system_bytes':monitor['min_available_bytes'],'resumed':False})
                atomic_json(stage/'resource_result.json',result)
                if result['status']=='completed':
                    _mark_complete(stage,context,['selected_configuration.json','pipeline.joblib',
                        'fit_diagnostics.json','warnings.json','worker_result.json','resource_monitor.json','resource_result.json'])
            results.append(result)
            summary={'scope':'resource_pilot','root':str(root.resolve()),
                'status':'completed' if len(results)==7 and all(r['status']=='completed' for r in results) else 'partial',
                'level':4,'training_split':'train_seed_42','rows':context['train_rows'],
                'raw_features':context['raw_features'],'completed_models':sum(r['status']=='completed' for r in results),
                'attempted_models':len(results),'planned_models':list(MODEL_NAMES),
                'outer_fits_planned':7,'primitive_fits_planned':21,'test_used':False,'randomized_search_used':False,
                'main_experiment_budget_changed':False,'selected_configurations_file':'selected_configurations.json',
                'results':results,'elapsed_seconds_this_invocation':time.perf_counter()-started}
            atomic_json(root/'resource_summary.json',summary)
            frame=_metrics_frame(results)
            _atomic_parquet(frame,root/'resource_metrics.parquet')
            frame.to_csv(root/'resource_metrics.csv',index=False)
            print(f'Resource pilot checkpoint: {model}: {result["status"]}',flush=True)
            if result['status'] in {'resource_guard','interrupted'}: break
        assert artifact_manifest(data.path)==context['inputs'],'Prepared X/splits changed during resource pilot'
        summary['input_sha256_unchanged']=True
        atomic_json(root/'resource_summary.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--prepared',required=True)
    parser.add_argument('--root')
    parser.add_argument('--model',choices=MODEL_NAMES)
    parser.add_argument('--output-parent')
    args=parser.parse_args()
    if args.worker:
        if not args.root or not args.model: parser.error('worker requires root and model')
        raise SystemExit(worker(args.prepared,args.root,args.model))
    if not args.output_parent: parser.error('controller requires output-parent')
    result=run_resource_pilot(args.prepared,args.output_parent)
    print(_json_text(result))


if __name__=='__main__': main()
