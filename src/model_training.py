"""A2: сохранённые разбиения, train-only Pipeline, атомарные этапы и resume.

Полный эксперимент запускается только явным вызовом run_full_experiment.
Технический прогон использует исключительно development; его артефакты отделены.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import time
import traceback
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from threadpoolctl import threadpool_limits

from .data_quality import artifact_manifest
from .model_pipelines import MODEL_NAMES, experiment_spec, make_estimator, parameter_space
from .fit_diagnostics import extract_fit_diagnostics

SEEDS=(42,43,44,45,46)
LEVELS=(1,2,3,4)
PACKAGES=('numpy','pandas','pyarrow','scipy','scikit-learn','xgboost','lightgbm','joblib','threadpoolctl')


def _json_default(value):
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,Path): return str(value)
    raise TypeError(f'Не JSON-совместимое значение: {type(value)}')


def _json_text(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,default=_json_default,allow_nan=False)


def atomic_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(_json_text(value),encoding='utf-8')
    # Windows может кратко удерживать target открытым во время чтения монитором.
    deadline=time.monotonic()+2.0
    while True:
        try:
            os.replace(temp,path)
            return
        except PermissionError as exc:
            remaining=deadline-time.monotonic()
            if getattr(exc,'winerror',None) not in (5,32) or remaining<=0:
                raise
            time.sleep(min(0.05,remaining))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''): digest.update(block)
    return digest.hexdigest()


def _atomic_parquet(frame,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    frame.to_parquet(temp,index=False,compression='zstd')
    os.replace(temp,path)


def _atomic_model(estimator,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    joblib.dump(estimator,temp,compress=3)
    os.replace(temp,path)


class PreparedData:
    """ID является только индексом; X не изменяется и не содержит TARGET/ID."""
    def __init__(self,path):
        self.path=Path(path)
        self.ids=pd.read_parquet(self.path/'row_ids.parquet').SK_ID_CURR.to_numpy()
        target=pd.read_parquet(self.path/'target.parquet')
        assert np.array_equal(target.SK_ID_CURR,self.ids)
        assert len(np.unique(self.ids))==len(self.ids) and (np.diff(self.ids)>0).all()
        self.target=target.set_index('SK_ID_CURR').TARGET
        assert self.target.isin([0,1]).all()
        self.features=read_json(self.path/'feature_levels.json')
        self.development=self._split('development')
        self.test=self._split('test')
        self.repeats={seed:self._split(f'train_seed_{seed}') for seed in SEEDS}
        folds=pd.read_parquet(self.path/'splits/cv_folds.parquet')
        assert folds.SK_ID_CURR.is_unique
        self.folds=folds.set_index('SK_ID_CURR').validation_fold
        dev_set,test_set=set(self.development),set(self.test)
        assert dev_set.isdisjoint(test_set) and dev_set|test_set==set(self.ids)
        assert set(self.folds.index)==dev_set and set(self.folds)=={0,1,2}
        for seed,repeat in self.repeats.items():
            assert len(np.unique(repeat))==len(repeat)
            assert set(repeat)<=dev_set and set(repeat).isdisjoint(test_set)
            assert len(repeat)==int(.9*len(self.development))
        forbidden={'TARGET','CODE_GENDER','DAYS_BIRTH','AGE_YEARS','AGE_GROUP','GENDER_GROUP'}
        for level in LEVELS:
            names=self.features[str(level)]
            schema=pq.read_schema(self.path/f'X_level_{level}.parquet')
            assert schema.names==names and len(names)==len(set(names))
            assert not(set(names)&forbidden) and not any(n.startswith('SK_ID_') for n in names)
            assert pq.read_metadata(self.path/f'X_level_{level}.parquet').num_rows==len(self.ids)
            if level>1: assert names[:len(self.features[str(level-1)])]==self.features[str(level-1)]
        self.loaded_scopes=[]

    def _split(self,name):
        values=pd.read_parquet(self.path/'splits'/f'{name}_ids.parquet').SK_ID_CURR.to_numpy()
        assert len(np.unique(values))==len(values)
        return values

    def columns(self,level):
        schema=pq.read_schema(self.path/f'X_level_{level}.parquet')
        numeric=[f.name for f in schema if pa.types.is_integer(f.type) or pa.types.is_floating(f.type) or pa.types.is_decimal(f.type)]
        categorical=[f.name for f in schema if f.name not in numeric]
        assert all(pa.types.is_string(schema.field(n).type) or pa.types.is_large_string(schema.field(n).type)
                   or pa.types.is_boolean(schema.field(n).type) for n in categorical)
        return numeric,categorical

    def load(self,level,requested_ids,*,purpose):
        """Сканировать Parquet порциями и вернуть только запрошенные ID в их порядке.

        Физический блок Parquet может содержать строки разных split, но до создания
        DataFrame/fit/score выбираются исключительно requested_ids. Для technical
        принудительно запрещён запрос ID из test.
        """
        requested=np.asarray(requested_ids,dtype=self.ids.dtype)
        assert len(np.unique(requested))==len(requested)
        positions=np.searchsorted(self.ids,requested)
        assert (positions<len(self.ids)).all() and np.array_equal(self.ids[positions],requested)
        if purpose.startswith('technical'):
            assert set(requested)<=set(self.development) and set(requested).isdisjoint(self.test)
        sorted_order=np.argsort(positions)
        positions_sorted=positions[sorted_order]
        chunks=[]; offset=0
        reader=pq.ParquetFile(self.path/f'X_level_{level}.parquet')
        for batch in reader.iter_batches(batch_size=16384,columns=self.features[str(level)],use_threads=False):
            left=np.searchsorted(positions_sorted,offset)
            right=np.searchsorted(positions_sorted,offset+batch.num_rows)
            if right>left:
                chunks.append(batch.take(pa.array(positions_sorted[left:right]-offset)).to_pandas())
            offset+=batch.num_rows
        assert offset==len(self.ids)
        frame=pd.concat(chunks,ignore_index=True)
        frame.index=pd.Index(self.ids[positions_sorted],name='SK_ID_CURR')
        frame=frame.loc[requested]
        labels=self.target.loc[requested].copy()
        assert np.array_equal(frame.index,labels.index) and frame.columns.tolist()==self.features[str(level)]
        self.loaded_scopes.append({'level':level,'purpose':purpose,'rows':len(frame),
                                  'contains_test':bool(np.isin(requested,self.test).any())})
        return frame,labels

    def cv(self,ids):
        ids=np.asarray(ids)
        assigned=self.folds.reindex(ids)
        assert assigned.notna().all() and set(assigned)=={0,1,2}
        y=self.target.loc[ids]
        result=[]
        for fold in range(3):
            train=np.flatnonzero(assigned.to_numpy()!=fold)
            validation=np.flatnonzero(assigned.to_numpy()==fold)
            assert set(y.iloc[train])=={0,1} and set(y.iloc[validation])=={0,1}
            assert len(set(train)&set(validation))==0
            result.append((train,validation))
        assert np.array_equal(np.sort(np.concatenate([v for _,v in result])),np.arange(len(ids)))
        return result

    def contract_report(self):
        return {'applicants':len(self.ids),'development':len(self.development),'test':len(self.test),
                'levels':[{'level':level,'features':len(self.features[str(level)]),
                           'numeric':len(self.columns(level)[0]),'categorical':len(self.columns(level)[1])} for level in LEVELS],
                'cv':[{'fold':fold,'rows':int((self.folds==fold).sum()),
                       'positive':int(self.target.loc[self.folds.index[self.folds==fold]].sum())} for fold in range(3)],
                'repeats':[{'seed':seed,'rows':len(ids),'positive':int(self.target.loc[ids].sum())} for seed,ids in self.repeats.items()],
                'identifiers_are_index_only':True,'existing_splits_reused':True}


def full_experiment_plan():
    specs=[]
    for model in MODEL_NAMES:
        spec=experiment_spec(model)
        spec.update({'levels':4,'searches':4,'candidate_cv_outer_fits':4*20*3,
                     'main_outer_fits':4*5,'primitive_fits':4*(20*3+5)*spec['primitive_fits_per_outer_fit']})
        specs.append(spec)
    return {'models':list(MODEL_NAMES),'levels':list(LEVELS),'seeds':list(SEEDS),
            'searches':28,'configurations_per_search':20,'saved_cv_folds':3,
            'scoring':'neg_log_loss','refit':False,'candidate_cv_outer_fits':1680,
            'main_outer_fits':140,'primitive_fits_including_ensembles':sum(s['primitive_fits'] for s in specs),
            'stacking_meta_fits':260,'model_specs':specs,
            'note':'План успешного запуска без перезапусков прерванных этапов; внутренних ансамблевых fit больше, чем внешних вызовов.'}


def make_context(data,*,mode,threads=2,technical_train_n=1200,technical_validation_n=400):
    inputs=artifact_manifest(data.path)
    source_root=Path(__file__).resolve().parent
    context={'mode':mode,'protocol':full_experiment_plan(),
             'threads':threads,'search_n_jobs':1,'search_random_state':42,'error_score':'raise',
             'inputs':inputs,'versions':{name:importlib.metadata.version(name) for name in PACKAGES},
             'python':platform.python_version(),
             'code_sha256':{name:sha256(source_root/name) for name in ['model_training.py','model_pipelines.py','fold_preprocessing.py','fit_diagnostics.py']},
             'parameter_spaces':{name:parameter_space(name,technical=mode=='technical') for name in MODEL_NAMES}}
    if mode=='technical':
        context['technical']={'level':4,'train_n':technical_train_n,'validation_n':technical_validation_n,
           'selection_seed':2026,'train_parent':'train_seed_42','validation_parent':'development minus train_seed_42',
           'search_n_iter':2,'final_technical_fits':7,'test_access_for_fit_or_score':False}
    elif mode!='full': raise ValueError('mode должен быть technical или full')
    context['fingerprint']=hashlib.sha256(_json_text(context).encode('utf-8')).hexdigest()
    return context


def initialize_run(path,context):
    path=Path(path); path.mkdir(parents=True,exist_ok=True)
    manifest=path/'experiment_manifest.json'
    if manifest.exists():
        if read_json(manifest)['fingerprint']!=context['fingerprint']:
            raise RuntimeError('Контекст эксперимента изменён (данные/код/версии/параметры). Используйте новый каталог; результаты разных контекстов не смешиваются.')
    else: atomic_json(manifest,context)
    return path


@contextmanager
def run_lock(path):
    """Запретить два одновременных писателя. После kill проверить PID и убрать lock."""
    lock=Path(path)/'run.lock'
    try:
        handle=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f'Каталог занят: {lock}. Проверьте PID в lock; после аварийного завершения удалите только устаревший lock.') from exc
    try:
        os.write(handle,_json_text({'pid':os.getpid(),'utc':datetime.now(timezone.utc).isoformat()}).encode('utf-8'))
        os.close(handle)
        yield
    finally:
        lock.unlink(missing_ok=True)


def _complete(stage,context):
    marker=Path(stage)/'complete.json'
    if not marker.exists(): return False
    record=read_json(marker)
    if record['fingerprint']!=context['fingerprint']: raise RuntimeError(f'Несовместимый этап: {stage}')
    for name,digest in record['artifact_sha256'].items():
        if not (Path(stage)/name).is_file() or sha256(Path(stage)/name)!=digest:
            raise RuntimeError(f'Повреждён завершённый артефакт {stage}/{name}; восстановите его или используйте новый каталог.')
    return True


def _mark_complete(stage,context,names):
    atomic_json(Path(stage)/'complete.json',{'fingerprint':context['fingerprint'],
        'completed_utc':datetime.now(timezone.utc).isoformat(),
        'artifact_sha256':{name:sha256(Path(stage)/name) for name in names}})


def _events(root,event,**details):
    with (Path(root)/'events.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'event':event,**details},ensure_ascii=False,default=_json_default)+'\n')


def _warnings_report(records):
    counts={}
    for item in records:
        key=(item.category.__name__,str(item.message))
        counts[key]=counts.get(key,0)+1
    return [{'category':category,'message':message,'count':count} for (category,message),count in counts.items()]


def _failure(stage,root,kind,exc):
    atomic_json(Path(stage)/'failure.json',{'stage':kind,'error_type':type(exc).__name__,
        'error':str(exc),'traceback':traceback.format_exc()[-16000:],
        'utc':datetime.now(timezone.utc).isoformat()})
    _events(root,'failure',stage=str(stage),kind=kind,error=str(exc))


def search_pair(data,level,model,X,y,root,context):
    assert X.index.equals(y.index)
    assert np.array_equal(y.to_numpy(),data.target.loc[X.index].to_numpy())
    assert set(X.index)<=set(data.development) and set(X.index).isdisjoint(data.test)
    assert X.columns.tolist()==data.features[str(level)]
    if context['mode']=='full': assert np.array_equal(X.index,data.development)
    stage=Path(root)/f'level_{level}'/model/'search'
    stage.mkdir(parents=True,exist_ok=True)
    if _complete(stage,context): return read_json(stage/'best_params.json'),True
    technical=context['mode']=='technical'; n_iter=2 if technical else 20
    numeric,categorical=data.columns(level)
    estimator=make_estimator(model,numeric,categorical,seed=42,technical=technical,threads=context['threads'])
    splits=data.cv(X.index)
    search=RandomizedSearchCV(estimator=estimator,param_distributions=parameter_space(model,technical=technical),
        n_iter=n_iter,scoring='neg_log_loss',cv=splits,random_state=42,n_jobs=1,pre_dispatch=1,
        refit=False,error_score='raise',return_train_score=False)
    started=time.perf_counter(); _events(root,'search_started',level=level,model=model,n_iter=n_iter)
    try:
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=context['threads']):
            warnings.simplefilter('always')
            search.fit(X,y)
        elapsed=time.perf_counter()-started
        result=search.cv_results_
        assert len(result['params'])==n_iter and search.n_splits_==3
        assert np.isfinite(result['mean_test_score']).all()
        rows=[]
        for i,params in enumerate(result['params']):
            rows.append({'candidate':i,'params_json':_json_text(params),
                'mean_neg_log_loss':float(result['mean_test_score'][i]),
                'std_neg_log_loss':float(result['std_test_score'][i]),'rank':int(result['rank_test_score'][i]),
                'mean_fit_seconds':float(result['mean_fit_time'][i]),'mean_score_seconds':float(result['mean_score_time'][i]),
                **{f'fold_{fold}_neg_log_loss':float(result[f'split{fold}_test_score'][i]) for fold in range(3)}})
        selected={'level':level,'model':model,'params':search.best_params_,
            'best_cv_neg_log_loss':float(search.best_score_),'best_cv_log_loss':float(-search.best_score_),
            'best_candidate_index':int(search.best_index_),'n_iter':n_iter,'folds':3,
            'scoring':'neg_log_loss','refit':False,'fit_rows':len(X),'elapsed_seconds':elapsed,
            'warnings':_warnings_report(caught),'scope':context['mode'],
            'fit_call_accounting':{'outer_fits':n_iter*3,
                'primitive_fits':n_iter*3*experiment_spec(model)['primitive_fits_per_outer_fit']}}
        _atomic_parquet(pd.DataFrame(rows),stage/'cv_results.parquet')
        atomic_json(stage/'best_params.json',selected)
        _mark_complete(stage,context,['cv_results.parquet','best_params.json'])
        _events(root,'search_completed',level=level,model=model,elapsed_seconds=elapsed)
        print(f'{model} L{level}: search {n_iter} x 3, {elapsed:.1f}s, CV log-loss={-search.best_score_:.5f}',flush=True)
        del search; gc.collect()
        return selected,False
    except BaseException as exc:
        _failure(stage,root,'search',exc); raise


def positive_probability(estimator,X):
    classes=np.asarray(estimator.classes_)
    assert set(classes)=={0,1}
    probabilities=estimator.predict_proba(X)
    assert probabilities.shape==(len(X),2) and np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1),1,atol=1e-6)
    result=probabilities[:,int(np.flatnonzero(classes==1)[0])].astype(float)
    assert ((result>=0)&(result<=1)).all()
    return result


def classification_metrics(y,probabilities):
    predicted=(probabilities>=.5).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,predicted,labels=[0,1]).ravel()
    return {'rows':len(y),'positives':int(np.sum(y)),'positive_rate':float(np.mean(y)),
            'log_loss':float(log_loss(y,probabilities,labels=[0,1])),
            'pr_auc_average_precision':float(average_precision_score(y,probabilities)),
            'roc_auc':float(roc_auc_score(y,probabilities)),'brier_score':float(brier_score_loss(y,probabilities)),
            'threshold':.5,'accuracy':float(accuracy_score(y,predicted)),
            'precision':float(precision_score(y,predicted,zero_division=0)),
            'recall':float(recall_score(y,predicted,zero_division=0)),
            'f1':float(f1_score(y,predicted,zero_division=0)),
            'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),
            'approval_rate_at_0_5':float(np.mean(probabilities<.5))}


def fit_repeat(data,level,model,seed,X_train,y_train,X_evaluation,y_evaluation,root,context,selected):
    technical=context['mode']=='technical'
    stage=Path(root)/f'level_{level}'/model/f'seed_{seed}'
    stage.mkdir(parents=True,exist_ok=True)
    if _complete(stage,context): return read_json(stage/'metrics.json'),True
    assert X_train.index.equals(y_train.index) and X_evaluation.index.equals(y_evaluation.index)
    assert np.array_equal(y_train.to_numpy(),data.target.loc[X_train.index].to_numpy())
    assert np.array_equal(y_evaluation.to_numpy(),data.target.loc[X_evaluation.index].to_numpy())
    assert X_train.columns.tolist()==X_evaluation.columns.tolist()==data.features[str(level)]
    assert set(X_train.index).isdisjoint(X_evaluation.index)
    if technical:
        assert set(X_train.index)|set(X_evaluation.index)<=set(data.development)
        assert set(X_train.index).isdisjoint(data.test) and set(X_evaluation.index).isdisjoint(data.test)
    else:
        assert np.array_equal(X_train.index,data.repeats[seed])
        assert np.array_equal(X_evaluation.index,data.test)
    numeric,categorical=data.columns(level)
    # Всегда новый объект: гиперпараметры фиксированы, обученные медианы/категории — нет.
    estimator=make_estimator(model,numeric,categorical,seed=seed,technical=technical,threads=context['threads'])
    estimator.set_params(**selected['params'])
    _events(root,'repeat_started',level=level,model=model,seed=seed)
    try:
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=context['threads']):
            warnings.simplefilter('always')
            begin=time.perf_counter(); estimator.fit(X_train,y_train); fit_seconds=time.perf_counter()-begin
            begin=time.perf_counter(); probabilities=positive_probability(estimator,X_evaluation); predict_seconds=time.perf_counter()-begin
        evaluation_name='development_technical_validation' if technical else 'test'
        predictions=pd.DataFrame({'SK_ID_CURR':X_evaluation.index.to_numpy(),
            'TARGET':y_evaluation.to_numpy(),'probability_default':probabilities,
            'level':level,'model':model,'seed':seed,'evaluation_split':evaluation_name})
        assert predictions.SK_ID_CURR.is_unique
        assert np.array_equal(predictions.TARGET,data.target.loc[predictions.SK_ID_CURR].to_numpy())
        metrics={'level':level,'model':model,'seed':seed,'scope':context['mode'],
            'evaluation_split':evaluation_name,'train_rows':len(X_train),'raw_features':X_train.shape[1],
            'fit_seconds':fit_seconds,'predict_seconds':predict_seconds,
            'selected_params':selected['params'],'warnings':_warnings_report(caught),
            'metrics':classification_metrics(y_evaluation.to_numpy(),probabilities),
            'primitive_fit_count':experiment_spec(model)['primitive_fits_per_outer_fit']}
        diagnostics=extract_fit_diagnostics(estimator)
        atomic_json(stage/'fit_diagnostics.json',diagnostics)
        metrics['fit_diagnostics_file']='fit_diagnostics.json'
        metrics['actual_mlp_iterations']=[item['fitted'].get('n_iter_') for item in diagnostics['mlp']]
        _atomic_model(estimator,stage/'pipeline.joblib')
        # Проверяется реальная загрузка сохранённой Pipeline, а не только наличие файла.
        restored=joblib.load(stage/'pipeline.joblib')
        assert extract_fit_diagnostics(restored)==diagnostics
        with threadpool_limits(limits=context['threads']):
            np.testing.assert_allclose(positive_probability(restored,X_evaluation.iloc[:16]),probabilities[:16],rtol=1e-6,atol=1e-7)
        del restored
        metrics['serialization_prediction_check']=True
        _atomic_parquet(predictions,stage/'probabilities.parquet')
        atomic_json(stage/'metrics.json',metrics)
        _mark_complete(stage,context,['pipeline.joblib','probabilities.parquet','metrics.json','fit_diagnostics.json'])
        _events(root,'repeat_completed',level=level,model=model,seed=seed,fit_seconds=fit_seconds)
        print(f'{model} L{level} seed={seed}: fit {fit_seconds:.1f}s; {evaluation_name} log-loss={metrics["metrics"]["log_loss"]:.5f}',flush=True)
        del estimator; gc.collect()
        return metrics,False
    except BaseException as exc:
        _failure(stage,root,'repeat',exc); raise


def _flatten_metrics(records):
    return pd.DataFrame([{k:v for k,v in record.items() if k not in ['warnings','metrics','selected_params']}
                         |record['metrics']|{'warning_count':sum(w['count'] for w in record['warnings'])}
                         for record in records])


def run_technical(prepared_dir,output_parent,*,train_n=1200,validation_n=400,threads=2):
    """Семь моделей Level4, два кандидата, сохранённые CV-folds, test не используется."""
    started=time.perf_counter(); data=PreparedData(prepared_dir)
    context=make_context(data,mode='technical',threads=threads,technical_train_n=train_n,technical_validation_n=validation_n)
    root=Path(output_parent)/f'technical_{context["fingerprint"][:12]}'
    root.mkdir(parents=True,exist_ok=True)
    counters={'new_searches':0,'resumed_searches':0,'new_fits':0,'resumed_fits':0}
    with run_lock(root):
        initialize_run(root,context)
        parent_train=data.repeats[42]
        parent_validation=np.setdiff1d(data.development,parent_train)
        if train_n>=len(parent_train) or validation_n>=len(parent_validation):
            raise ValueError('Техническая выборка должна быть малой частью существующих development-подмножеств')
        train_ids,_=train_test_split(parent_train,train_size=train_n,stratify=data.target.loc[parent_train],random_state=2026)
        validation_ids,_=train_test_split(parent_validation,train_size=validation_n,stratify=data.target.loc[parent_validation],random_state=2026)
        train_ids,validation_ids=np.sort(train_ids),np.sort(validation_ids)
        for label,ids in [('technical_train',train_ids),('technical_validation',validation_ids)]:
            _atomic_parquet(pd.DataFrame({'SK_ID_CURR':ids}),root/f'{label}_ids.parquet')
        X_train,y_train=data.load(4,train_ids,purpose='technical_train')
        X_validation,y_validation=data.load(4,validation_ids,purpose='technical_validation')
        cv=data.cv(train_ids)
        records=[]; searches=[]
        for model in MODEL_NAMES:
            chosen,resumed=search_pair(data,4,model,X_train,y_train,root,context)
            counters['resumed_searches' if resumed else 'new_searches']+=1
            result,resumed=fit_repeat(data,4,model,42,X_train,y_train,X_validation,y_validation,root,context,chosen)
            counters['resumed_fits' if resumed else 'new_fits']+=1
            searches.append(chosen); records.append(result)
        after=artifact_manifest(data.path)
        assert after==context['inputs'],'Входные подготовленные данные изменились!'
        assert not any(item['contains_test'] for item in data.loaded_scopes)
        result={'scope':'technical_only','root':str(root.resolve()),'full_experiment_executed':False,
            'model_types_tested':list(MODEL_NAMES),'training_level':4,'train_rows':len(X_train),
            'validation_rows':len(X_validation),'train_positive':int(y_train.sum()),
            'validation_positive':int(y_validation.sum()),'saved_cv_subset':[
                {'fold':f,'train_rows':len(t),'validation_rows':len(v),
                 'validation_positive':int(y_train.iloc[v].sum())} for f,(t,v) in enumerate(cv)],
            'search_configs_per_model':2,'search_candidate_outer_fits':42,'technical_final_outer_fits':7,
            'planned_primitive_fits':sum(7*experiment_spec(name)['primitive_fits_per_outer_fit'] for name in MODEL_NAMES),
            'data_access':data.loaded_scopes,'input_sha256_unchanged':True,
            'run_counters':counters,'searches':searches,'fits':records,
            'elapsed_seconds_this_invocation':time.perf_counter()-started,
            'available_ram_gib_end':psutil.virtual_memory().available/1024**3}
        atomic_json(root/'technical_summary.json',result)
        _atomic_parquet(_flatten_metrics(records),root/'technical_metrics.parquet')
    return result


def run_full_experiment(prepared_dir,output_dir,*,threads=2):
    """ЯВНЫЙ длительный запуск A2: 28 поисков и 140 независимых повторов.

    Resume — завершённый поиск или отдельный повтор. Прерванный RandomizedSearchCV
    перезапускается целиком с теми же 20 кандидатами; mid-search resume не заявляется.
    Test не передаётся в поиск, раннюю остановку, калибровку или выбор параметров.
    """
    started=time.perf_counter(); data=PreparedData(prepared_dir)
    context=make_context(data,mode='full',threads=threads)
    root=Path(output_dir)
    root.mkdir(parents=True,exist_ok=True)
    records=[]; searches=[]
    counters={'new_searches':0,'resumed_searches':0,'new_fits':0,'resumed_fits':0}
    with run_lock(root):
        initialize_run(root,context)
        for level in LEVELS:
            X_dev,y_dev=data.load(level,data.development,purpose='search_development')
            X_test=y_test=None
            for model in MODEL_NAMES:
                chosen,resumed=search_pair(data,level,model,X_dev,y_dev,root,context)
                counters['resumed_searches' if resumed else 'new_searches']+=1
                searches.append(chosen)
                for seed in SEEDS:
                    # В памяти только текущий уровень и одна 90%-ная train-подвыборка.
                    stage=root/f'level_{level}'/model/f'seed_{seed}'
                    if _complete(stage,context):
                        result=read_json(stage/'metrics.json'); resumed=True
                    else:
                        if X_test is None: X_test,y_test=data.load(level,data.test,purpose='final_test_evaluation')
                        repeat=data.repeats[seed]
                        X_train,y_train=X_dev.loc[repeat],y_dev.loc[repeat]
                        result,resumed=fit_repeat(data,level,model,seed,X_train,y_train,X_test,y_test,root,context,chosen)
                        del X_train,y_train; gc.collect()
                    counters['resumed_fits' if resumed else 'new_fits']+=1
                    records.append(result)
                    _atomic_parquet(_flatten_metrics(records),root/'metrics_completed.parquet')
            del X_dev,y_dev,X_test,y_test; gc.collect()
        assert len(searches)==28 and len(records)==140
        assert all(search['n_iter']==20 and search['folds']==3 and not search['refit'] for search in searches)
        assert artifact_manifest(data.path)==context['inputs']
        result={'scope':'full','root':str(root.resolve()),'completed_searches':28,'completed_main_fits':140,
                'plan':full_experiment_plan(),'run_counters':counters,
                'input_sha256_unchanged':True,'elapsed_seconds_this_invocation':time.perf_counter()-started}
        atomic_json(root/'experiment_summary.json',result)
        frame=_flatten_metrics(records)
        for metric in ['pr_auc_average_precision','roc_auc','log_loss','brier_score','fit_seconds']:
            summary=frame.groupby(['level','model'])[metric].agg(['mean','std']).reset_index()
            _atomic_parquet(summary,root/f'summary_{metric}.parquet')
    return result
