from pathlib import Path
import os,json,itertools,sys,zipfile,hashlib,importlib.metadata
import numpy as np,pandas as pd
from scipy.stats import spearmanr
r=Path(__file__).resolve().parent
sys.path.insert(0,str(r/'rq3'))
import rq3_analyze as ana
out=r/'rq3/out'; tab=out/'tables'; audit=r/'verification';audit.mkdir(exist_ok=True)
exp=pd.read_parquet(out/'explained_ids.parquet');bg=pd.read_parquet(out/'background_ids.parquet')
quality=pd.read_csv(r/'quality.csv')
assert len(exp)==exp.SK_ID_CURR.nunique()==1000 and exp.HAS_BUREAU.value_counts().to_dict()=={0:500,1:500}
records=[]
for run in ['fixed','r1','r2','r3','r4','r5','noise_a','noise_b']:
    for model in ana.MODELS:
        approx=model in ('mlp','logistic_regression','soft_voting','stacking')
        if run.startswith('noise') and not approx:continue
        folder=out/'runs'/run
        meta=json.loads((folder/(model+'.json')).read_bytes())
        n=1000
        v=np.load(folder/(model+'.npy'));base=np.load(folder/(model+'_base.npy'));p=np.load(folder/(model+'_prob.npy'))
        ids=np.load(folder/(model+'_ids.npy'))
        assert len(bg)==100 and meta['n_background']==100 and meta['scale']=='probability'
        assert np.array_equal(ids,exp.SK_ID_CURR.to_numpy()[np.arange(len(exp))]) and v.shape==(n,322)
        assert np.isfinite(v).all() and np.isfinite(base).all() and np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
        if run.startswith('r'):assert meta['training_seed']==int(run[1:]) and meta['training_rows']==18000
        else:assert meta['training_seed']==42 and meta['training_rows']==20000
        resid=np.abs(v.sum(1)+base-p)
        assert abs(resid.max()-meta['additivity_max_abs_err'])<1e-12
        if not run.startswith('r'):
            verified=json.loads((audit/'BUNDLE.json').read_bytes())['models']
            assert meta['pipeline_sha256']==verified['models/'+model+'/level_4/pipeline.joblib']
        if run=='fixed':
            item=json.loads((folder/(model+'_verification.json')).read_bytes())
            assert max(item['absolute_difference'].values())<1e-4
        records.append({'run':run,'model':model,'n':n,'background':100,'additivity_max_error':resid.max(),'count_over_0.001':int((resid>.001).sum()),'native_masked_check':meta.get('mask_validation_max_error',np.nan)})
records=pd.DataFrame(records)
assert len(records)==50 and records.n.sum()==50000
records.to_csv(audit/'explanation_checks.csv',index=False)
stab=pd.read_csv(tab/'rq3_stability.csv').query('group=="all"')
oldstab=pd.read_csv(r/'baseline/rq3_stability.csv').query('group=="all"')
assert len(stab)==7 and (stab.runs==5).all()
comp=oldstab[['model','top5_kept','spearman_global_mean']].merge(stab[['model','top5_kept','spearman_global_mean']],on='model',suffixes=('_oct','_nov'))
comp.to_csv(tab/'stability_october_vs_november.csv',index=False)
cons=pd.read_csv(tab/'rq3_consistency_with_LR.csv').query('group=="all"')
assert len(cons)==6
old=Path(os.environ['SHAP_PROJECT'])/'rq3/out/runs/fixed'
v=np.load(old/'mlp.npy')
assert v.shape==(1000,322) and np.allclose(v.sum(1)+np.load(old/'mlp_base.npy'),np.load(old/'mlp_prob.npy'),atol=1e-5)
new=np.load(out/'runs/fixed/mlp.npy')
ov=ana.boot_mean_ci(ana.overlap(v,new))
rho=ana.boot_global_spearman(np.abs(v),np.abs(new))
pd.DataFrame([['top5_overlap',*ov],['global_spearman',*rho]],columns=['metric','estimate','lower95','upper95']).to_csv(tab/'mlp_october_vs_november.csv',index=False)
features=json.loads((r/'data/handoff/pilot_20k_5config/features_level_4.json').read_bytes())
pd.DataFrame({'feature':features,'october_mean_abs_shap':np.abs(v).mean(0),'november_mean_abs_shap':np.abs(new).mean(0)}).sort_values('november_mean_abs_shap',ascending=False).to_csv(tab/'mlp_feature_importance_comparison.csv',index=False)
used={p:importlib.metadata.version(p) for p in ('numpy','pandas','scikit-learn','joblib','scipy','lightgbm','xgboost','shap','numba','threadpoolctl','pyarrow','matplotlib')}
(audit/'environment.json').write_text(json.dumps({'python':sys.version,'packages':used},indent=2))
assert len(pd.read_csv(tab/'rq3_refit_test_metrics.csv').query('metric=="log_loss"'))==7
assert len(pd.read_csv(tab/'rq3_stability_diff_vs_LR.csv'))==6
print('Verified',len(records),'model/run explanations;',sum(records.n),'explained rows. Residuals',records.additivity_max_error.max())
print(stab[['model','spearman_global_mean','top5_kept']].to_string(index=False))
print('MLP old/new',ov,rho)
