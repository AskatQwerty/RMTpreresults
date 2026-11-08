from pathlib import Path
import gc,sys,os,time,json,warnings
import numpy as np, pandas as pd, joblib, pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
from sklearn.metrics import average_precision_score,roc_auc_score,log_loss,brier_score_loss
from sklearn.model_selection import StratifiedShuffleSplit
from src.mlp_repair import fit_mlp_variant
from src.model_pipelines import make_estimator
r=Path(__file__).resolve().parent
sys.path.insert(0,str(r/'rq3'))
import rq3_shap as rq
root=Path(os.environ['SHAP_PROJECT'])
old=root/'rq3/out'
d=root/'Get-FileHash pilot_20k_5config_shap/data/handoff/pilot_20k_5config'
data=r.parent/'RMT_batch_11_15/data'
out=Path(os.environ['RQ3_OUTPUT_DIR']) if os.environ.get('RQ3_OUTPUT_DIR') else r/'rq3/out'
out.mkdir(exist_ok=True)
exp=pd.read_parquet(old/'explained_ids.parquet')
bg=pd.read_parquet(old/'background_ids.parquet').SK_ID_CURR
assert len(exp)==exp.SK_ID_CURR.nunique()==1000 and len(bg)==bg.nunique()==100
X=pd.read_parquet(data/'X_level_4.parquet')
y=pd.read_parquet(data/'cohort_targets.parquet').set_index('SK_ID_CURR').TARGET
X.index=y.index
cols=json.loads((data/'columns.json').read_bytes())['4']
feats=list(X.columns)
assert json.loads((d/'features_level_4.json').read_bytes())==feats
settings=json.loads((r/'frozen_plan.json').read_bytes())['settings']
paramlist=json.loads((r/'params.json').read_bytes())
variant=json.loads((r/'frozen_selection.json').read_bytes())['selections'][3]['params']
truth=pd.read_parquet(d/'target_test.parquet').set_index('SK_ID_CURR').TARGET
test_ids=pd.read_parquet(d/'test_ids.parquet').SK_ID_CURR
assert test_ids.is_unique and len(test_ids)==61503
selected=[]
for batch in pq.ParquetFile(d/'X_test_level4.parquet').iter_batches(batch_size=2048):
    block=batch.to_pandas().set_index('SK_ID_CURR')
    selected.append(block.loc[block.index.isin(exp.SK_ID_CURR)])
E=pd.concat(selected).loc[exp.SK_ID_CURR]
assert np.array_equal(E.HAS_BUREAU,exp.HAS_BUREAU) and np.array_equal(truth.loc[E.index],exp.TARGET)
B=X.loc[bg]
w=r/'fits';w.mkdir(exist_ok=True)
def fit(model,run):
    key='fixed' if run.startswith('noise') else run
    file=w/(key+'_'+model+'.joblib')
    info=file.with_suffix('.json')
    if key=='fixed':
        import hashlib
        bundle=r/'active_bundle'
        source=bundle/'models'/model/'level_4/pipeline.joblib'
        spec=json.loads(source.with_name('metadata.json').read_bytes())
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        assert digest==spec['pipeline_sha256'] and spec['scope']=='pilot_20k_20config_mlp_updated'
        return joblib.load(source),dict(origin='Official RMT_active_28_models.zip verified by SHA-256',pipeline_sha256=digest,params=spec['selected_params'],training_rows=20000,training_seed=42)
    if file.exists() and info.exists():return joblib.load(file),json.loads(info.read_bytes())
    seed=42 if key=='fixed' else int(key[1:])
    rows=np.arange(len(y)) if key=='fixed' else next(StratifiedShuffleSplit(n_splits=1,train_size=.9,random_state=seed).split(X,y))[0]
    start=time.monotonic()
    if model=='mlp':
        est,diag=fit_mlp_variant(X.iloc[rows],y.iloc[rows],cols['numeric'],cols['categorical'],variant,**{**settings,'seed':seed})
    else:
        spec=next(z for z in paramlist if z['level']==4 and z['model']==model)
        est=make_estimator(model,cols['numeric'],cols['categorical'],seed=seed,threads=2)
        est.set_params(**{k:tuple(v) if isinstance(v,list) else v for k,v in spec['params'].items()})
        est.fit(X,y) if key=="fixed" else est.fit(X.iloc[rows],y.iloc[rows])
    meta={'training_seed':seed,'training_rows':len(rows),'fit_seconds':time.monotonic()-start,'params':variant if model=='mlp' else spec['params'],'origin':'November protocol locally reproduced' if key=='fixed' else '90% stratified retraining'}
    if model=='mlp':meta['epochs']=diag['selected_epochs']
    temp=file.with_suffix('.tmp');joblib.dump(est,temp);temp.replace(file)
    info.write_text(json.dumps(meta))
    return est,meta
def test(est,model,run,run_dir):
    ps=[]
    for batch in pq.ParquetFile(d/'X_test_level4.parquet').iter_batches(batch_size=2048):
        block=batch.to_pandas().set_index('SK_ID_CURR')
        ps.append(pd.Series(est.predict_proba(block[feats])[:,1],index=block.index))
    p=pd.concat(ps).loc[test_ids].astype(float);t=truth.loc[test_ids]
    scores=dict(ap=average_precision_score(t,p),roc_auc=roc_auc_score(t,p),log_loss=log_loss(t,p),brier=brier_score_loss(t,p))
    np.save(run_dir/(model+'_prob_test.npy'),p.to_numpy().astype('float32'))
    if run=='fixed':
        expected=pd.read_csv(r/'quality.csv').query('level==4 and model==@model').iloc[0]
        diff={k:abs(v-float(expected[k])) for k,v in scores.items()}
        print('Fixed metric differences',model,diff,flush=True)
        (run_dir/(model+'_verification.json')).write_text(json.dumps({'local':scores,'published':{k:float(expected[k]) for k in scores},'absolute_difference':diff},indent=2))
        if max(diff.values())>1e-4:raise RuntimeError('Fixed model requires review: '+model)
    return scores
def execute(run,model):
    dest=out/'runs'/run;dest.mkdir(parents=True,exist_ok=True)
    meta_file=dest/(model+'.json')
    if meta_file.exists():
        prior=json.loads(meta_file.read_bytes())
        if prior['n_explained']==1000 and np.array_equal(np.load(dest/(model+'_ids.npy')),exp.SK_ID_CURR.to_numpy()):
            return
    print('START',run,model,flush=True)
    est,meta=fit(model,run)
    if not run.startswith('noise'):meta['test_metrics']=test(est,model,run,dest)
    ensemble=model in ('soft_voting','stacking')
    e=E
    n=len(e)
    assert n==1000 and exp.HAS_BUREAU.value_counts().to_dict()=={0:500, 1:500}
    seed={'noise_a':11,'noise_b':12}.get(run,0)
    perms=2 if run=='fixed' and not ensemble else 1
    start=time.monotonic()
    fn=rq.ModelFn(model,est,feats,e,B,'probability')
    prob=est.predict_proba(e[feats])[:,1]
    if model in ('random_forest','lightgbm','xgboost'):
        v,base,_=rq.exact_shap(model,fn,'probability')
    elif ensemble or model=='logistic_regression':
        from path_shap import permutation
        v,base,meta['mask_validation_max_error']=permutation(fn,perms,seed)
    else:
        v,base=rq.perm_shap(fn,perms,seed)
    approx=model in ('mlp','soft_voting','stacking','logistic_regression')
    err=np.abs(v.sum(1)+base-prob)
    assert np.isfinite(v).all() and v.shape==(n,322) and np.isfinite(err).all()
    if approx and max(err)>1e-5:raise RuntimeError('Approximate additivity failure')
    for suffix,arr in [('',v),('_base',base),('_prob',prob),('_ids',np.asarray(e.index))]:np.save(dest/(model+suffix+'.npy'),arr)
    meta.update(run=run,model=model,n_explained=n,n_background=100,scale='probability',kind='approximate' if approx else 'exact',explainer='permutation' if approx else 'TreeExplainer',n_permutations=perms if approx else None,seed_explainer=seed,seconds=time.monotonic()-start,pipeline_check_max_err=fn.check_err,additivity_max_abs_err=float(max(err)),residuals_over_1e_3=int((err>.001).sum()))
    tmp=meta_file.with_suffix('.tmp');tmp.write_text(json.dumps(meta,indent=2));tmp.replace(meta_file)
    print('DONE',run,model,'seconds',round(meta['seconds']),'residual',max(err),flush=True)
    gc.collect()
if __name__=='__main__':
    with threadpool_limits(limits=2):
        models=sys.argv[1].split(',');runs=sys.argv[2].split(',')
        for run in runs:
            for model in models:
                if run.startswith('noise') and model in ('random_forest','lightgbm','xgboost'):continue
                execute(run,model)
