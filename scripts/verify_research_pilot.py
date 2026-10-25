"""Read-only проверка фактических артефактов pilot_20k_5config.

Не загружает joblib, не вызывает fit/predict/transform и не создаёт разбиения/X.
По умолчанию печатает JSON; --output сохраняет этот JSON только внутри reports/.
При работающем эксперименте проверяет снимок уже завершённых этапов.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT=Path(__file__).resolve().parents[1]
SCOPE='pilot_20k_5config'
MODELS=('logistic_regression','lightgbm','xgboost','soft_voting','stacking','mlp','random_forest')
LEVELS=(1,2,3,4)
WIDTHS={1:124,2:178,3:264,4:322}
PRIMITIVE={'logistic_regression':1,'lightgbm':1,'xgboost':1,'soft_voting':3,
           'stacking':13,'mlp':1,'random_forest':1}
FIT_FILES={'pipeline.joblib','cv_results.parquet','best_params.json','fit_diagnostics.json',
           'warnings.json','fit_summary.json'}
PAIR_FILES={'probabilities.parquet','prediction_warnings.json','metrics.json',
            'fit_checkpoint/complete.json'}|{f'fit_checkpoint/{name}' for name in FIT_FILES}
COHORT_FILES={'cohort_ids.parquet','cohort_targets.parquet','cv_folds.parquet','test_ids.parquet'}
SOURCES={'research_pilot.py','model_training.py','model_pipelines.py',
         'fold_preprocessing.py','fit_diagnostics.py'}


def require(condition,message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def child_path(base,name):
    base=Path(base).resolve()
    path=(base/name).resolve()
    require(path.is_relative_to(base) and path!=base,f'Unsafe artifact path: {name}')
    return path


class HashReader:
    def __init__(self):
        self.cache={}
        self.bytes_read=0

    def digest(self,path):
        path=Path(path).resolve()
        stat=path.stat()
        signature=(stat.st_size,stat.st_mtime_ns)
        if path in self.cache:
            previous,digest=self.cache[path]
            require(previous==signature,f'File changed during verification: {path}')
            return digest
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024**2),b''):
                digest.update(block)
                self.bytes_read+=len(block)
        after=path.stat()
        require(signature==(after.st_size,after.st_mtime_ns),f'File changed while hashing: {path}')
        result=digest.hexdigest()
        self.cache[path]=(signature,result)
        return result

    def marker(self,stage,fingerprint,required):
        marker=read_json(stage/'complete.json')
        require(marker['fingerprint']==fingerprint,f'Fingerprint mismatch: {stage}')
        hashes=marker['artifact_sha256']
        require(set(hashes)==set(required),f'Incomplete/unexpected marker inventory: {stage}')
        for name,expected in hashes.items():
            require(self.digest(child_path(stage,name))==expected,f'SHA256 mismatch: {stage}/{name}')
        return marker


def equal_array(actual,expected,label):
    require(np.array_equal(np.asarray(actual),np.asarray(expected)),f'{label} differs')


def equal_number(actual,expected,label):
    require(np.isfinite(actual) and np.isclose(actual,expected,rtol=1e-10,atol=1e-12),
            f'{label}: {actual!r} != {expected!r}')


def check_global(prepared,root,reader,expected_rows,expected_test_rows,widths,input_count,source_dir):
    context=read_json(root/'experiment_manifest.json')
    payload={key:value for key,value in context.items() if key!='fingerprint'}
    text=json.dumps(payload,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)
    require(hashlib.sha256(text.encode()).hexdigest()==context['fingerprint'],'Manifest fingerprint is invalid')
    require(context['scope']==context['mode']==SCOPE,'Wrong experiment scope/mode')
    protocol=context['protocol']
    expected_protocol={'train_rows':expected_rows,'cohort_seed':42,'stratified_selection':True,
        'class_resampling':False,'cv_folds':3,'cv_shuffle':True,'cv_random_state':42,
        'searches':28,'configurations_per_search':5,'search_random_state':42,
        'scoring':'neg_log_loss','refit':True,'technical':False,'candidate_cv_outer_fits':420,
        'final_refit_fits':28,'manual_additional_fits':0,'repeat_fits':0,
        'primitive_fits_including_ensembles':1344}
    for name,value in expected_protocol.items():
        require(protocol.get(name)==value,f'Protocol mismatch: {name}')
    require(protocol['models']==list(MODELS) and protocol['levels']==list(LEVELS),'Wrong model/level plan')
    require(protocol['order']==[{'model':m,'level':level} for m in MODELS for level in LEVELS],
            'Wrong pair execution order')
    require(context['train_rows']==expected_rows and context['test_rows']==expected_test_rows,'Wrong train/test size')
    require(set(context['code_sha256'])==SOURCES,'Unexpected frozen source inventory')
    for name,digest in context['code_sha256'].items():
        require(reader.digest(source_dir/name)==digest,f'Frozen source changed: {name}')
        require(reader.digest(root/'source_snapshot'/name)==digest,f'Source snapshot changed: {name}')
    inputs=context['inputs']
    require(len(inputs)==input_count,'Wrong number of prepared input hashes')
    names=[item['file'] for item in inputs]
    require(len(set(names))==len(names),'Duplicate prepared input entry')
    actual_names={p.relative_to(prepared).as_posix() for p in prepared.rglob('*')
                  if p.is_file() and 'duckdb_tmp' not in p.parts}
    require(actual_names==set(names),'Prepared input inventory changed')
    for item in inputs:
        path=child_path(prepared,item['file'])
        require(path.stat().st_size==item['bytes'],f'Prepared input size changed: {item["file"]}')
        require(reader.digest(path)==item['sha256'],f'Prepared input SHA256 changed: {item["file"]}')
    row_ids=pd.read_parquet(prepared/'row_ids.parquet').SK_ID_CURR.to_numpy()
    targets=pd.read_parquet(prepared/'target.parquet')
    equal_array(targets.SK_ID_CURR,row_ids,'Prepared target IDs/order')
    require(len(np.unique(row_ids))==len(row_ids) and (np.diff(row_ids)>0).all(),'Prepared IDs not sorted/unique')
    require(targets.TARGET.isin([0,1]).all(),'Invalid TARGET values')
    target=targets.set_index('SK_ID_CURR').TARGET
    development=pd.read_parquet(prepared/'splits/development_ids.parquet').SK_ID_CURR.to_numpy()
    test=pd.read_parquet(prepared/'splits/test_ids.parquet').SK_ID_CURR.to_numpy()
    require(len(set(development))==len(development) and len(set(test))==len(test),'Duplicate split IDs')
    require(set(development).isdisjoint(test) and set(development)|set(test)==set(row_ids),'Invalid development/test partition')
    require(len(development)==context['development_rows'] and len(test)==expected_test_rows,'Split size mismatch')
    features=read_json(prepared/'feature_levels.json')
    for level,width in widths.items():
        columns=features[str(level)]
        require(len(columns)==width and len(set(columns))==width,f'Level {level} feature width/uniqueness mismatch')
        require(columns==read_json(prepared/f'features_level_{level}.json'),f'Level {level} feature lists differ')
        require(pq.read_schema(prepared/f'X_level_{level}.parquet').names==columns,f'Level {level} Parquet schema differs')
        require(pq.read_metadata(prepared/f'X_level_{level}.parquet').num_rows==len(row_ids),f'Level {level} row count differs')
        require(not set(columns)&{'TARGET','CODE_GENDER','DAYS_BIRTH','AGE_YEARS','AGE_GROUP','GENDER_GROUP'}
                and not any(name.startswith('SK_ID_') for name in columns),f'Forbidden feature in Level {level}')
        if level>1:
            require(columns[:len(features[str(level-1)])]==features[str(level-1)],'Levels are not nested')
    contract=read_json(root/'cohort_contract.json')
    require(contract==context['cohort_contract'],'Cohort contract differs from manifest')
    require(contract['sample_n']==expected_rows and contract['selection_seed']==contract['cv_seed']==42,'Wrong cohort setup')
    require(set(contract['artifact_sha256'])==COHORT_FILES,'Incomplete cohort hash inventory')
    for name,digest in contract['artifact_sha256'].items():
        require(reader.digest(root/name)==digest,f'Cohort SHA256 mismatch: {name}')
    ids=pd.read_parquet(root/'cohort_ids.parquet').SK_ID_CURR.to_numpy()
    cohort_target=pd.read_parquet(root/'cohort_targets.parquet')
    folds=pd.read_parquet(root/'cv_folds.parquet')
    equal_array(pd.read_parquet(root/'test_ids.parquet').SK_ID_CURR,test,'Pilot test IDs/order')
    require(len(ids)==expected_rows and len(set(ids))==expected_rows and (np.diff(ids)>0).all(),'Invalid cohort IDs')
    require(set(ids)<=set(development) and set(ids).isdisjoint(test),'Cohort contains non-development/test IDs')
    equal_array(cohort_target.SK_ID_CURR,ids,'Cohort target IDs/order')
    equal_array(cohort_target.TARGET,target.loc[ids],'Cohort TARGET')
    equal_array(folds.SK_ID_CURR,ids,'Cohort CV IDs/order')
    require(set(folds.validation_fold)=={0,1,2},'Wrong CV fold labels')
    require(abs(int(cohort_target.TARGET.sum())-expected_rows*target.loc[development].mean())<=1,
            'Cohort class proportions are inconsistent with stratified selection')
    fold_rows=[]
    for fold in range(3):
        labels=cohort_target.TARGET[folds.validation_fold==fold]
        require(set(labels)=={0,1},f'CV fold {fold} misses a class')
        fold_rows.append({'fold':fold,'train_rows':len(ids)-len(labels),'validation_rows':len(labels),
                          'validation_positives':int(labels.sum()),'validation_negatives':int((labels==0).sum())})
    for name in ('validation_rows','validation_positives','validation_negatives'):
        values=[row[name] for row in fold_rows]
        require(max(values)-min(values)<=1,f'CV is not stratified: {name}')
    return context,target,test,{'train_rows':len(ids),'test_rows':len(test),'train_positives':int(target.loc[ids].sum()),
        'test_positives':int(target.loc[test].sum()),'folds':fold_rows,'feature_widths':widths,
        'prepared_input_hashes_verified':len(inputs),'frozen_sources_verified':len(SOURCES),
        'shared_cohort_and_cv_verified':True,'train_test_disjoint':True}


def check_checkpoint(stage,model,level,context,reader,width):
    fit=stage/'fit_checkpoint'
    reader.marker(fit,context['fingerprint'],FIT_FILES)
    selected=read_json(fit/'best_params.json')
    summary=read_json(fit/'fit_summary.json')
    diagnostics=read_json(fit/'fit_diagnostics.json')
    for item in (selected,summary):
        require(item['scope']==SCOPE and item['model']==model and item['level']==level,'Checkpoint identity mismatch')
        require(item['train_rows']==context['train_rows'],'Checkpoint train rows mismatch')
    require(selected['n_iter']==5 and selected['folds']==3 and selected['scoring']=='neg_log_loss'
            and selected['refit'] is True,'Checkpoint search protocol differs')
    require(summary['raw_features']==width and diagnostics['raw_feature_count']==width,'Checkpoint raw width differs')
    require(summary['fitted_on_all_saved_cohort_ids'] is True and summary['test_loaded_before_checkpoint'] is False,
            'Checkpoint training scope differs')
    expected_budget={'candidate_cv':15,'automatic_refit':1,'manual_extra':0,
                     'primitive_including_ensembles':16*PRIMITIVE[model]}
    require(summary['fit_calls']==expected_budget,'Checkpoint fit budget differs')
    require(read_json(fit/'warnings.json')==summary['warnings'],'Checkpoint warnings differ')
    cv=pd.read_parquet(fit/'cv_results.parquet')
    require(len(cv)==5,'CV results must contain exactly 5 candidates')
    equal_array(cv.candidate,np.arange(5),'CV candidate indices')
    scores=cv[[f'fold_{fold}_neg_log_loss' for fold in range(3)]].to_numpy(dtype=float)
    require(np.isfinite(scores).all() and (scores<=1e-12).all(),'Invalid CV fold log-loss scores')
    means=scores.mean(axis=1)
    for i,row in cv.iterrows():
        equal_number(row.mean_neg_log_loss,means[i],f'CV candidate {i} mean')
        equal_number(row.std_neg_log_loss,scores[i].std(),f'CV candidate {i} std')
        require(int(row['rank'])==1+int((means>means[i]).sum()),f'CV candidate {i} rank differs')
        params=json.loads(row.params_json)
        require(set(params)==set(context['parameter_spaces'][model]),f'CV candidate {i} parameter keys differ')
        for name,value in params.items():
            require(value in context['parameter_spaces'][model][name],f'CV parameter outside frozen search space: {name}')
    best=int(np.argmax(means))
    require(selected['best_candidate_index']==best,'Wrong best CV candidate')
    require(selected['params']==json.loads(cv.iloc[best].params_json),'Selected parameters differ from best CV candidate')
    equal_number(selected['best_cv_neg_log_loss'],means[best],'Best negative CV log-loss')
    equal_number(selected['best_cv_log_loss'],-means[best],'Best CV log-loss')
    return selected,summary,diagnostics


def check_pair(stage,model,level,context,reader,target,test,width,checkpoint):
    reader.marker(stage,context['fingerprint'],PAIR_FILES)
    selected,fit_summary,diagnostics=checkpoint
    metrics=read_json(stage/'metrics.json')
    require(metrics['scope']==SCOPE and metrics['model']==model and metrics['level']==level,'Metrics identity mismatch')
    require(metrics['train_rows']==context['train_rows'] and metrics['test_rows']==len(test),'Metrics row counts differ')
    require(metrics['raw_features']==width and metrics['fit_calls']==fit_summary['fit_calls'],'Metrics width/budget differs')
    require(metrics['selected_params']==selected['params'],'Metrics selected parameters differ')
    equal_number(metrics['best_cv_log_loss'],selected['best_cv_log_loss'],'Metrics best CV log-loss')
    require(metrics['threshold_selected_using_test'] is False and metrics['preprocessing_refit_on_test'] is False,
            'Metrics indicate test-based fitting or threshold selection')
    require(metrics['actual_mlp_iterations']==[row['fitted']['n_iter_'] for row in diagnostics['mlp']],
            'MLP iteration summary differs from fit diagnostics')
    for access in metrics['data_access']:
        if access['contains_test']:
            require(access['purpose']=='pilot_final_test_evaluation' and access['rows']==len(test),'Unexpected test access')
        else:
            require(access['rows']==context['train_rows'],'Unexpected training access size')
    probabilities=pd.read_parquet(stage/'probabilities.parquet')
    require(len(probabilities)==len(test) and probabilities.SK_ID_CURR.is_unique,'Probability row count/uniqueness differs')
    equal_array(probabilities.SK_ID_CURR,test,'Prediction IDs/order')
    equal_array(probabilities.TARGET,target.loc[test],'Prediction TARGET')
    for column,value in [('scope',SCOPE),('model',model),('level',level),('evaluation_split','test')]:
        require(probabilities[column].eq(value).all(),f'Probability {column} differs')
    values=probabilities.probability_default.to_numpy()
    require(np.isfinite(values).all() and ((values>=0)&(values<=1)).all(),'Invalid probability values')
    require(metrics['metrics']['rows']==len(test) and metrics['metrics']['positives']==int(target.loc[test].sum()),
            'Metric TARGET totals differ')
    return {'probability_rows_verified':len(test),'best_cv_log_loss':selected['best_cv_log_loss'],
            'selected_params':selected['params'],'raw_features':width,'fit_calls':fit_summary['fit_calls']}


def read_log_snapshot(path):
    size=path.stat().st_size
    with path.open('rb') as stream:
        data=stream.read(size)
    return data.decode('utf-8',errors='replace')


def log_evidence(root):
    events=[]
    trailing_partial=False
    event_path=root/'events.jsonl'
    if event_path.exists():
        text=read_log_snapshot(event_path)
        lines=text.splitlines(keepends=True)
        for index,line in enumerate(lines):
            try: events.append(json.loads(line))
            except json.JSONDecodeError:
                if index==len(lines)-1 and not line.endswith('\n'):
                    trailing_partial=True
                else: raise ValueError(f'Invalid events.jsonl line {index+1}')
    counts=Counter(item.get('event','missing_event') for item in events)
    pairs={}
    for model in MODELS:
        for level in LEVELS:
            stage=root/model/f'level_{level}'
            attempts=[]
            for path in sorted((stage/'attempt_logs').glob('attempt_*.log')):
                text=read_log_snapshot(path)
                attempts.append({'file':path.relative_to(root).as_posix(),
                    'cv_end_lines':sum('[CV] END' in line for line in text.splitlines()),
                    'search_banner_lines':text.count('totalling 15 fits'),
                    'pair_completed_lines':text.count(f'Completed {model} Level {level};')})
            pair_events=Counter(item['event'] for item in events
                if item.get('model')==model and item.get('level')==level)
            pairs[(model,level)]={'attempt_log_count':len(attempts),'attempt_logs':attempts,
                'cv_end_lines':sum(item['cv_end_lines'] for item in attempts),'event_counts':dict(pair_events)}
    return {'event_counts':dict(counts),'search_started':counts['search_started'],
        'search_and_refit_completed':counts['search_and_refit_completed'],'pair_completed':counts['pair_completed'],
        'worker_failed':counts['worker_failed'],'failed':sum(count for name,count in counts.items()
            if name in {'failed','failure'}),'resumed':sum(count for name,count in counts.items() if 'resum' in name),
        'ignored_trailing_partial_event':trailing_partial,
        'attempt_log_count':sum(item['attempt_log_count'] for item in pairs.values()),
        'cv_end_lines':sum(item['cv_end_lines'] for item in pairs.values())},pairs


def verify_pilot(prepared_dir,root,*,require_complete=False,expected_rows=20000,expected_test_rows=61503,
                 expected_widths=None,expected_input_count=29,source_dir=None,require_attempt_logs=True):
    started=time.perf_counter()
    prepared,root=Path(prepared_dir).resolve(),Path(root).resolve()
    source=Path(source_dir).resolve() if source_dir else PROJECT_ROOT/'src'
    widths={int(key):int(value) for key,value in (expected_widths or WIDTHS).items()}
    reader=HashReader()
    result={'schema_version':1,'scope':SCOPE,'verified_at_utc':datetime.now(timezone.utc).isoformat(),
        'status':'failed','completeness':'partial','read_only':True,'fit_predict_transform_called':False,
        'joblib_loaded':False,'require_complete':require_complete,'training_root':str(root),
        'prepared_root':str(prepared),'errors':[],'warnings':[],'pairs':[],
        'summary':{'planned_pairs':28,'completed_pairs':0,'fit_checkpoints':0},
        'budget':{'planned_candidate_cv_fits':420,'planned_final_refits':28,
            'note':'Checkpoint budgets count saved successful work. Logs/events describe attempts; retries can add fits.'}}
    snapshots={(model,level):((root/model/f'level_{level}'/'complete.json').exists(),
                             (root/model/f'level_{level}'/'fit_checkpoint/complete.json').exists())
               for model in MODELS for level in LEVELS}
    try:
        context,target,test,global_summary=check_global(prepared,root,reader,expected_rows,
            expected_test_rows,widths,expected_input_count,source)
        result['fingerprint']=context['fingerprint']
        result['global_checks']=global_summary
        logs,pair_logs=log_evidence(root)
        result['logs']=logs
        for (model,level),(complete,fitted) in snapshots.items():
            row={'model':model,'level':level,'status':'complete' if complete else 'fit_checkpoint' if fitted else 'pending',
                 'verified':False,'logs':pair_logs[(model,level)]}
            if complete: result['summary']['completed_pairs']+=1
            if fitted: result['summary']['fit_checkpoints']+=1
            try:
                if complete or fitted:
                    checkpoint=check_checkpoint(root/model/f'level_{level}',model,level,context,reader,widths[level])
                    if complete:
                        row.update(check_pair(root/model/f'level_{level}',model,level,context,reader,
                                              target,test,widths[level],checkpoint))
                        attempts=row['logs']['attempt_logs']
                        require(not require_attempt_logs or bool(attempts),'Completed pair has no attempt logs')
                        if len(attempts)==1:
                            require(attempts[0]['cv_end_lines']==15,'Single attempt does not contain 15 CV completion lines')
                    row['verified']=True
            except Exception as exc:
                row['error']=f'{type(exc).__name__}: {exc}'
                result['errors'].append(f'{model}/level_{level}: {row["error"]}')
            result['pairs'].append(row)
        total=result['summary']['completed_pairs']
        fitted=result['summary']['fit_checkpoints']
        result['completeness']='complete' if total==28 else 'partial'
        result['budget'].update({'saved_candidate_cv_fits':15*fitted,'saved_automatic_refits':fitted,
            'saved_manual_extra_fits':0,'observed_search_started_events':logs['search_started'],
            'observed_search_refit_completed_events':logs['search_and_refit_completed'],
            'observed_cv_end_log_lines':logs['cv_end_lines']})
        retries=[f'{model}/level_{level}' for (model,level),item in pair_logs.items()
                 if item['attempt_log_count']>1 or item['event_counts'].get('search_started',0)>1]
        result['logs']['pairs_with_multiple_attempts']=retries
        single_pass=(total==28 and not retries and logs['search_started']==28
            and logs['search_and_refit_completed']==28 and logs['pair_completed']==28
            and logs['attempt_log_count']==28 and logs['cv_end_lines']==420
            and not logs['worker_failed'] and not logs['failed'])
        result['logs']['single_pass_420_cv_and_28_refits_supported']=single_pass
        if retries: result['warnings'].append('Retries detected; actual fit work can exceed the planned 420 + 28 budget.')
        if total==28 and not single_pass:
            result['warnings'].append('Complete checkpoints alone do not establish a single-pass actual fit total; inspect attempt/event counts.')
        if require_complete and total!=28:
            result['errors'].append(f'Incomplete experiment: {total}/28 pairs have completion markers')
    except Exception as exc:
        result['errors'].append(f'Global verification: {type(exc).__name__}: {exc}')
    result['status']='failed' if result['errors'] else 'passed'
    result['hashed_files']=len(reader.cache)
    result['hashed_bytes']=reader.bytes_read
    result['elapsed_seconds']=time.perf_counter()-started
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',default=str(PROJECT_ROOT/'data/processed/full'))
    parser.add_argument('--root','--output-root',dest='root',default=str(PROJECT_ROOT/'artifacts/training/pilot_20k_5config'))
    parser.add_argument('--require-complete',action='store_true')
    parser.add_argument('--output',type=Path,help='Optional JSON report inside this project reports/ directory')
    args=parser.parse_args(argv)
    if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    result=verify_pilot(args.prepared,args.root,require_complete=args.require_complete)
    if args.output:
        try:
            output=args.output.resolve()
            require(output.is_relative_to((PROJECT_ROOT/'reports').resolve()),'--output must be inside project reports/')
            require(not output.is_relative_to(Path(args.root).resolve())
                    and not output.is_relative_to(Path(args.prepared).resolve()),'Cannot write to training/prepared roots')
            output.parent.mkdir(parents=True,exist_ok=True)
            output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        except Exception as exc:
            result['status']='failed'
            result['errors'].append(f'Output: {type(exc).__name__}: {exc}')
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':
    raise SystemExit(main())
