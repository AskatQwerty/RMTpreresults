"""RQ1/RQ2: отдельный pilot_20k_5config, 420 CV-fit + 28 автоматических refit.

Полные X/сплиты и код A2 не изменяются. Все fit принадлежат RandomizedSearchCV
с refit=True; ручного дополнительного fit здесь нет. Test загружается только
после сохранения fit-checkpoint. Одна пара выполняется в отдельном процессе.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits

from .data_quality import artifact_manifest
from .fit_diagnostics import extract_fit_diagnostics
from .model_pipelines import make_estimator, parameter_space, experiment_spec
from .model_training import (PreparedData, PACKAGES, atomic_json, read_json, sha256,
    _json_text, _atomic_parquet, _atomic_model, _complete, _mark_complete,
    _events, _failure, _warnings_report, initialize_run, run_lock,
    positive_probability, classification_metrics)

SCOPE='pilot_20k_5config'
MODEL_ORDER=('logistic_regression','lightgbm','xgboost','soft_voting','stacking','mlp','random_forest')
LEVELS=(1,2,3,4)
COHORT_FILES=('cohort_ids.parquet','cohort_targets.parquet','cv_folds.parquet','test_ids.parquet')
FIT_FILES=('pipeline.joblib','cv_results.parquet','best_params.json','fit_diagnostics.json',
           'warnings.json','fit_summary.json')
SOURCE_FILES=('research_pilot.py','model_training.py','model_pipelines.py',
              'fold_preprocessing.py','fit_diagnostics.py')


def pilot_plan(sample_n=20000):
    return {'scope':SCOPE,'models':list(MODEL_ORDER),'levels':list(LEVELS),
        'order':[{'model':model,'level':level} for model in MODEL_ORDER for level in LEVELS],
        'train_rows':sample_n,'cohort_seed':42,'stratified_selection':True,
        'class_resampling':False,'cv_folds':3,'cv_shuffle':True,'cv_random_state':42,
        'searches':28,'configurations_per_search':5,'search_random_state':42,
        'scoring':'neg_log_loss','refit':True,'technical':False,
        'candidate_cv_outer_fits':420,'final_refit_fits':28,'manual_additional_fits':0,
        'repeat_fits':0,'primitive_fits_including_ensembles':1344,
        'test_use':'final evaluation only; no parameter/rule/threshold selection',
        'shap_in_scope':False,
        'deviations_from_A2':['20,000 development clients instead of full development',
            '5 configurations per search instead of 20',
            'one automatic refit per pair instead of five 90-percent repeats']}


def _balance(labels):
    return [{'TARGET':value,'count':int((labels==value).sum()),
             'fraction':float((labels==value).mean()),'percent':float((labels==value).mean()*100)}
            for value in (0,1)]


def load_pilot_split(data,root):
    """Прочитать общие сохранённые ID и назначения фолдов без нового split."""
    root=Path(root)
    contract=read_json(root/'cohort_contract.json')
    for name,digest in contract['artifact_sha256'].items():
        assert sha256(root/name)==digest,f'Изменён пилотный split: {name}'
    ids=pd.read_parquet(root/'cohort_ids.parquet').SK_ID_CURR.to_numpy()
    target=pd.read_parquet(root/'cohort_targets.parquet')
    folds=pd.read_parquet(root/'cv_folds.parquet')
    test=pd.read_parquet(root/'test_ids.parquet').SK_ID_CURR.to_numpy()
    assert len(ids)==contract['sample_n'] and len(np.unique(ids))==len(ids)
    assert (np.diff(ids)>0).all() and set(ids)<=set(data.development)
    assert set(ids).isdisjoint(data.test) and np.array_equal(test,data.test)
    assert np.array_equal(target.SK_ID_CURR,ids) and np.array_equal(folds.SK_ID_CURR,ids)
    assert np.array_equal(target.TARGET,data.target.loc[ids])
    assert set(folds.validation_fold)=={0,1,2}
    splits=[]
    for fold in range(3):
        train=np.flatnonzero(folds.validation_fold.to_numpy()!=fold)
        validation=np.flatnonzero(folds.validation_fold.to_numpy()==fold)
        assert set(target.TARGET.iloc[train])=={0,1} and set(target.TARGET.iloc[validation])=={0,1}
        assert not(set(train)&set(validation))
        splits.append((train,validation))
    assert np.array_equal(np.sort(np.concatenate([v for _,v in splits])),np.arange(len(ids)))
    return ids,splits


def initialize_pilot(prepared_dir,output_root,*,threads=2,sample_n=20000):
    """Один раз выбрать cohort и собственные CV-folds; существующие не пересоздавать."""
    data=PreparedData(prepared_dir)
    root=Path(output_root); root.mkdir(parents=True,exist_ok=True)
    assert 6<=sample_n<len(data.development)
    threads=max(1,min(int(threads),2))
    if not (root/'cohort_contract.json').exists():
        assert not any((root/name).exists() for name in COHORT_FILES), \
            'Незавершённая инициализация cohort: сохраните каталог и используйте новый.'
        ids,_=train_test_split(data.development,train_size=sample_n,
            stratify=data.target.loc[data.development],random_state=42)
        ids=np.sort(ids)
        target=data.target.loc[ids]
        assigned=np.full(sample_n,-1,dtype=np.int8)
        splitter=StratifiedKFold(n_splits=3,shuffle=True,random_state=42)
        for fold,(_,validation) in enumerate(splitter.split(np.zeros(sample_n),target)):
            assigned[validation]=fold
        _atomic_parquet(pd.DataFrame({'SK_ID_CURR':ids}),root/'cohort_ids.parquet')
        _atomic_parquet(pd.DataFrame({'SK_ID_CURR':ids,'TARGET':target.to_numpy()}),root/'cohort_targets.parquet')
        _atomic_parquet(pd.DataFrame({'SK_ID_CURR':ids,'validation_fold':assigned}),root/'cv_folds.parquet')
        _atomic_parquet(pd.DataFrame({'SK_ID_CURR':data.test}),root/'test_ids.parquet')
        atomic_json(root/'cohort_contract.json',{'sample_n':sample_n,'selection_seed':42,
            'cv_seed':42,'artifact_sha256':{name:sha256(root/name) for name in COHORT_FILES}})
    contract=read_json(root/'cohort_contract.json')
    assert contract['sample_n']==sample_n,'Размер существующего cohort нельзя менять.'
    ids,splits=load_pilot_split(data,root)
    source=Path(__file__).resolve().parent
    context={'scope':SCOPE,'mode':SCOPE,'protocol':pilot_plan(sample_n),
        'train_rows':len(ids),'development_rows':len(data.development),'test_rows':len(data.test),
        'threads':threads,'search_n_jobs':1,'pre_dispatch':1,'error_score':'raise',
        'inputs':artifact_manifest(data.path),'cohort_contract':contract,
        'versions':{name:importlib.metadata.version(name) for name in PACKAGES},
        'python':platform.python_version(),
        'code_sha256':{name:sha256(source/name) for name in SOURCE_FILES},
        'parameter_spaces':{model:parameter_space(model,technical=False) for model in MODEL_ORDER}}
    context['fingerprint']=hashlib.sha256(_json_text(context).encode()).hexdigest()
    initialize_run(root,context)
    summary={'scope':SCOPE,'parent_split':'development','sample_n':len(ids),
        'random_state':42,'stratified':True,'class_resampling':False,
        'development_balance':_balance(data.target.loc[data.development]),
        'cohort_balance':_balance(data.target.loc[ids]),'test_rows':len(data.test),
        'test_ids_unchanged':True,'shared_ids_for_all_28_pairs':True,
        'cv':[{'fold':fold,'train_rows':len(train),'validation_rows':len(valid),
               'train_balance':_balance(data.target.loc[ids[train]]),
               'validation_balance':_balance(data.target.loc[ids[valid]])}
              for fold,(train,valid) in enumerate(splits)]}
    if (root/'cohort_summary.json').exists():
        assert read_json(root/'cohort_summary.json')==summary
    else: atomic_json(root/'cohort_summary.json',summary)
    snapshot=root/'source_snapshot'; snapshot.mkdir(exist_ok=True)
    for name,digest in context['code_sha256'].items():
        if (snapshot/name).exists(): assert sha256(snapshot/name)==digest
        else: shutil.copy2(source/name,snapshot/name)
    return context


def _cv_results(search):
    result=search.cv_results_
    assert len(result['params'])==5 and search.n_splits_==3
    assert np.isfinite(result['mean_test_score']).all()
    rows=[]
    for i,params in enumerate(result['params']):
        rows.append({'candidate':i,'params_json':_json_text(params),
            'mean_neg_log_loss':float(result['mean_test_score'][i]),
            'std_neg_log_loss':float(result['std_test_score'][i]),
            'rank':int(result['rank_test_score'][i]),
            'mean_fit_seconds':float(result['mean_fit_time'][i]),
            'std_fit_seconds':float(result['std_fit_time'][i]),
            'mean_score_seconds':float(result['mean_score_time'][i]),
            'std_score_seconds':float(result['std_score_time'][i]),
            **{f'fold_{fold}_neg_log_loss':float(result[f'split{fold}_test_score'][i]) for fold in range(3)}})
    return pd.DataFrame(rows)


def run_pair(prepared_dir,root,model,level):
    """Один search.fit с автоматическим refit; test только после fit-checkpoint."""
    assert model in MODEL_ORDER and level in LEVELS
    root=Path(root); context=read_json(root/'experiment_manifest.json')
    assert context['mode']==SCOPE and context['protocol']['refit'] is True
    for name,digest in context['code_sha256'].items():
        assert sha256(Path(__file__).resolve().parent/name)==digest,f'Источник изменился: {name}'
    stage=root/model/f'level_{level}'; stage.mkdir(parents=True,exist_ok=True)
    if _complete(stage,context): return read_json(stage/'metrics.json')
    started=time.perf_counter()
    data=PreparedData(prepared_dir)
    assert read_json(root/'cohort_contract.json')==context['cohort_contract'], 'Pilot cohort contract changed'
    ids,cv=load_pilot_split(data,root)
    fit=stage/'fit_checkpoint'; fit.mkdir(exist_ok=True)
    caught=[]
    try:
        if _complete(fit,context):
            fitted=joblib.load(fit/'pipeline.joblib')
            selected=read_json(fit/'best_params.json')
            fit_summary=read_json(fit/'fit_summary.json')
            diagnostics=read_json(fit/'fit_diagnostics.json')
            assert extract_fit_diagnostics(fitted)==diagnostics
            _events(root,'fit_checkpoint_resumed',model=model,level=level)
        else:
            X,y=data.load(level,ids,purpose='pilot_research_train')
            assert np.array_equal(X.index,ids) and X.index.equals(y.index)
            assert set(X.index).isdisjoint(data.test)
            numeric,categorical=data.columns(level)
            estimator=make_estimator(model,numeric,categorical,seed=42,technical=False,
                threads=context['threads'])
            search=RandomizedSearchCV(estimator,parameter_space(model,technical=False),
                n_iter=5,scoring='neg_log_loss',cv=cv,random_state=42,
                n_jobs=1,pre_dispatch=1,refit=True,error_score='raise',
                return_train_score=False,verbose=2)
            _events(root,'search_started',model=model,level=level,train_rows=len(ids),
                n_iter=5,cv_folds=3,refit=True,scope=SCOPE)
            print(f'{SCOPE}: {model} Level {level}; {len(ids)} train rows; 5 x 3 CV + automatic refit',flush=True)
            with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=context['threads']):
                warnings.simplefilter('always')
                begin=time.perf_counter()
                search.fit(X,y)
                search_seconds=time.perf_counter()-begin
            fitted=search.best_estimator_  # Это уже обученный refit, дополнительного fit нет.
            selected={'scope':SCOPE,'model':model,'level':level,'params':search.best_params_,
                'best_cv_log_loss':float(-search.best_score_),
                'best_cv_neg_log_loss':float(search.best_score_),
                'best_candidate_index':int(search.best_index_),'n_iter':5,'folds':3,
                'scoring':'neg_log_loss','refit':True,'train_rows':len(ids)}
            diagnostics=extract_fit_diagnostics(fitted)
            fit_summary={'scope':SCOPE,'model':model,'level':level,'train_rows':len(ids),
                'raw_features':len(data.features[str(level)]),
                'search_and_refit_seconds':search_seconds,'refit_seconds':float(search.refit_time_),
                'warnings':_warnings_report(caught),'fit_calls':{'candidate_cv':15,'automatic_refit':1,
                    'manual_extra':0,'primitive_including_ensembles':16*experiment_spec(model)['primitive_fits_per_outer_fit']},
                'fitted_on_all_saved_cohort_ids':True,'test_loaded_before_checkpoint':False}
            _atomic_model(fitted,fit/'pipeline.joblib')
            _atomic_parquet(_cv_results(search),fit/'cv_results.parquet')
            atomic_json(fit/'best_params.json',selected)
            atomic_json(fit/'fit_diagnostics.json',diagnostics)
            atomic_json(fit/'warnings.json',fit_summary['warnings'])
            atomic_json(fit/'fit_summary.json',fit_summary)
            _mark_complete(fit,context,FIT_FILES)
            _events(root,'search_and_refit_completed',model=model,level=level,
                search_and_refit_seconds=search_seconds,refit_seconds=fit_summary['refit_seconds'])
            del X,y,search,estimator; gc.collect()
            # Проверка реальной сериализации не вызывает fit.
            del fitted; gc.collect()
            fitted=joblib.load(fit/'pipeline.joblib')
            assert extract_fit_diagnostics(fitted)==diagnostics
        _events(root,'test_evaluation_started',model=model,level=level,test_rows=len(data.test))
        Xtest,ytest=data.load(level,data.test,purpose='pilot_final_test_evaluation')
        assert np.array_equal(Xtest.index,data.test) and Xtest.index.equals(ytest.index)
        with warnings.catch_warnings(record=True) as evaluation_warnings, threadpool_limits(limits=context['threads']):
            warnings.simplefilter('always')
            begin=time.perf_counter()
            probabilities=positive_probability(fitted,Xtest)
            predict_seconds=time.perf_counter()-begin
        predictions=pd.DataFrame({'SK_ID_CURR':Xtest.index.to_numpy(),'TARGET':ytest.to_numpy(),
            'probability_default':probabilities,'scope':SCOPE,'model':model,'level':level,
            'evaluation_split':'test'})
        assert predictions.SK_ID_CURR.is_unique and len(predictions)==context['test_rows']
        assert np.array_equal(predictions.TARGET,data.target.loc[predictions.SK_ID_CURR])
        prediction_warnings=_warnings_report(evaluation_warnings)
        combined=[dict(record,stage='search_and_refit') for record in fit_summary['warnings']]
        combined.extend(dict(record,stage='test_prediction') for record in prediction_warnings)
        metrics={'scope':SCOPE,'model':model,'level':level,'train_rows':len(ids),'test_rows':len(data.test),
            'raw_features':len(data.features[str(level)]),'selected_params':selected['params'],
            'best_cv_log_loss':selected['best_cv_log_loss'],
            'search_and_refit_seconds':fit_summary['search_and_refit_seconds'],
            'refit_seconds':fit_summary['refit_seconds'],'predict_seconds':predict_seconds,
            'total_pair_seconds':time.perf_counter()-started,'warnings':combined,
            'metrics':classification_metrics(ytest.to_numpy(),probabilities),
            'fit_diagnostics_file':'fit_checkpoint/fit_diagnostics.json',
            'pipeline_file':'fit_checkpoint/pipeline.joblib',
            'actual_mlp_iterations':[row['fitted']['n_iter_'] for row in diagnostics['mlp']],
            'fit_calls':fit_summary['fit_calls'],'threshold_selected_using_test':False,
            'preprocessing_refit_on_test':False,'data_access':data.loaded_scopes}
        _atomic_parquet(predictions,stage/'probabilities.parquet')
        atomic_json(stage/'prediction_warnings.json',prediction_warnings)
        atomic_json(stage/'metrics.json',metrics)
        files=['probabilities.parquet','prediction_warnings.json','metrics.json',
            'fit_checkpoint/complete.json']+[f'fit_checkpoint/{name}' for name in FIT_FILES]
        _mark_complete(stage,context,files)
        _events(root,'pair_completed',model=model,level=level,scope=SCOPE,
            search_and_refit_seconds=fit_summary['search_and_refit_seconds'],predict_seconds=predict_seconds)
        print(f'Completed {model} Level {level}; checkpoint and test probabilities saved.',flush=True)
        return metrics
    except BaseException as exc:
        if caught: atomic_json(stage/'failed_attempt_warnings.json',_warnings_report(caught))
        _failure(stage,root,'pilot_pair',exc)
        raise


def _completed_summary(root,context,started):
    records=[]; fitted_pairs=0
    for model in MODEL_ORDER:
        for level in LEVELS:
            stage=root/model/f'level_{level}'
            if _complete(stage/'fit_checkpoint',context): fitted_pairs+=1
            if _complete(stage,context): records.append(read_json(stage/'metrics.json'))
    return {'scope':SCOPE,'root':str(root.resolve()),'status':'completed' if len(records)==28 else 'partial',
        'planned_pairs':28,'completed_pairs':len(records),'completed_searches':fitted_pairs,
        'completed_final_refits':fitted_pairs,'candidate_cv_fits_completed':15*fitted_pairs,
        'train_rows':context['train_rows'],'test_rows':context['test_rows'],
        'primitive_fits_completed':sum(16*experiment_spec(model)['primitive_fits_per_outer_fit']
            for model in MODEL_ORDER for level in LEVELS if _complete(root/model/f'level_{level}'/'fit_checkpoint',context)),
        'elapsed_seconds_this_invocation':time.perf_counter()-started,
        'results':records,'original_full_experiment_modified':False,'shap_performed':False}


def run_research_pilot(prepared_dir,output_root,report_dir,*,threads=2):
    """Последовательные worker-процессы; экспорт сразу после каждой готовой пары."""
    from .pilot_reporting import export_pilot_reports
    root=Path(output_root)
    context=initialize_pilot(prepared_dir,root,threads=threads)
    started=time.perf_counter()
    with run_lock(root):
        _events(root,'pilot_invocation_started',scope=SCOPE,model_order=list(MODEL_ORDER))
        for model in MODEL_ORDER:
            for level in LEVELS:
                stage=root/model/f'level_{level}'
                if not _complete(stage,context):
                    stage.mkdir(parents=True,exist_ok=True)
                    logs=stage/'attempt_logs'; logs.mkdir(exist_ok=True)
                    attempt=len(list(logs.glob('attempt_*.log')))+1
                    log_path=logs/f'attempt_{attempt:03d}.log'
                    command=[sys.executable,'-u','-m','src.research_pilot','--worker',
                        '--prepared',str(Path(prepared_dir).resolve()),'--output-root',str(root.resolve()),
                        '--model',model,'--level',str(level)]
                    print(f'Starting {model} Level {level}; log={log_path}',flush=True)
                    with log_path.open('wb',buffering=0) as log:
                        process=subprocess.Popen(command,cwd=Path(__file__).resolve().parents[1],
                            stdout=log,stderr=subprocess.STDOUT,shell=False,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                        try:
                            last=time.perf_counter()
                            while process.poll() is None:
                                if time.perf_counter()-last>=60:
                                    try: rss=psutil.Process(process.pid).memory_info().rss/1024**3
                                    except psutil.NoSuchProcess: rss=0
                                    print(f'Running {model} Level {level}; worker PID={process.pid}; RSS={rss:.2f} GiB',flush=True)
                                    last=time.perf_counter()
                                time.sleep(1)
                        except BaseException:
                            process.terminate(); process.wait(timeout=15)
                            raise
                    if process.returncode!=0:
                        _events(root,'worker_failed',model=model,level=level,exit_code=process.returncode,log=str(log_path))
                        summary=_completed_summary(root,context,started)
                        atomic_json(root/'experiment_summary.json',summary)
                        export_pilot_reports(root,report_dir)
                        raise RuntimeError(f'{model} Level {level} failed; see {log_path}. Completed checkpoints are preserved.')
                    assert _complete(stage,context)
                summary=_completed_summary(root,context,started)
                atomic_json(root/'experiment_summary.json',summary)
                report=export_pilot_reports(root,report_dir)
                print(f'CHECKPOINT {summary["completed_pairs"]}/28; exported CSV/LaTeX: {report_dir}',flush=True)
        assert artifact_manifest(Path(prepared_dir))==context['inputs'],'Prepared data changed'
        summary=_completed_summary(root,context,started)
        summary['prepared_inputs_sha256_unchanged']=True
        atomic_json(root/'experiment_summary.json',summary)
        export_pilot_reports(root,report_dir)
        _events(root,'pilot_completed',completed_pairs=28,scope=SCOPE)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',default='data/processed/full')
    parser.add_argument('--output-root',default='artifacts/training/pilot_20k_5config')
    parser.add_argument('--report-dir',default='reports/pilot_20k_5config')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--model',choices=MODEL_ORDER)
    parser.add_argument('--level',type=int,choices=LEVELS)
    args=parser.parse_args()
    if args.worker:
        if args.model is None or args.level is None: parser.error('worker requires model and level')
        run_pair(args.prepared,args.output_root,args.model,args.level)
    else:
        summary=run_research_pilot(args.prepared,args.output_root,args.report_dir)
        print(_json_text({k:v for k,v in summary.items() if k!='results'}),flush=True)


if __name__=='__main__': main()
