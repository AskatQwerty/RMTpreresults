import sys
from pathlib import Path
project = Path(__file__).resolve().parent
sys.path.insert(0, str(project))
sys.path.insert(0, str(project / 'rq3'))
from path_shap import permutation
import rq3_shap as sq
from types import SimpleNamespace
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
rng=np.random.default_rng(32)
x=rng.normal(size=(250,9)).astype('float32')
y=(x[:,1]+x[:,5]+rng.normal(size=250)>0).astype(int)
lr=LogisticRegression().fit(x,y)
xb=XGBClassifier(n_estimators=7,max_depth=3,n_jobs=2).fit(x,y)
lb=LGBMClassifier(n_estimators=8,num_leaves=4,n_jobs=2,verbosity=-1).fit(x,y)
fn=object.__new__(sq.ModelFn)
fn.scale='probability';fn.model='soft_voting';fn.ens=SimpleNamespace(weights=(2,1,1));fn.bases=[('lr',None,lr),('xgb',None,xb),('lgbm',None,lb)]
raw=np.array([0,1,2,3,4,5,5,6,6]);m=7
fn.maps=[SimpleNamespace(raw_index=raw,M=m,members=[np.flatnonzero(raw==i) for i in range(m)])]*3
fn.Te=[x[:4]]*3;fn.Tb=[x[20:30]]*3
v,b,e=permutation(fn,2,5)
a,c=sq.perm_shap(fn,2,5)
print(np.max(np.abs(v-a)),np.max(np.abs(b-c)),e)
assert np.max(np.abs(v-a))<1e-6
