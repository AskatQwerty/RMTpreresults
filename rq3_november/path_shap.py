"""Evaluate the same permutation game by reusing unaffected tree paths and LR score terms."""
import json
import numpy as np
from scipy.special import expit
from numba import njit

@njit
def tree_paths(e,bg,step,roots,feature,cut,left,right,leaf,n_features,is_xgb):
    # Each interval covers positions with exactly the same branch decisions.
    result=np.zeros((n_features+2,len(bg)))
    stack_node=np.empty(len(feature),np.int64)
    stack_lo=np.empty(len(feature),np.int64)
    stack_hi=np.empty(len(feature),np.int64)
    for i in range(len(bg)):
        for root in roots:
            size=1
            stack_node[0]=root;stack_lo[0]=0;stack_hi[0]=n_features
            while size:
                size-=1
                node=stack_node[size];lo=stack_lo[size];hi=stack_hi[size]
                j=feature[node]
                if j<0:
                    result[lo,i]+=leaf[node];result[hi+1,i]-=leaf[node]
                    continue
                old_left=bg[i,j]<cut[node] if is_xgb else bg[i,j]<=cut[node]
                new_left=e[j]<cut[node] if is_xgb else e[j]<=cut[node]
                old=left[node] if old_left else right[node]
                new=left[node] if new_left else right[node]
                switch=step[j]
                if old==new or hi<switch or lo>=switch:
                    stack_node[size]=new if lo>=switch else old
                    stack_lo[size]=lo;stack_hi[size]=hi;size+=1
                else:
                    stack_node[size]=old;stack_lo[size]=lo;stack_hi[size]=switch-1;size+=1
                    stack_node[size]=new;stack_lo[size]=switch;stack_hi[size]=hi;size+=1
    for j in range(1,n_features+1):
        result[j,:]+=result[j-1,:]
    return result[:-1]

def forest_arrays(model,is_xgb):
    roots=[]; feature=[];cut=[];left=[];right=[];leaf=[]
    def parse(node):
        idx=len(feature)
        feature.append(-1);cut.append(0.);left.append(-1);right.append(-1);leaf.append(0.)
        if 'leaf' in node or 'leaf_value' in node:
            leaf[idx]=node['leaf'] if is_xgb else node['leaf_value']
        else:
            if is_xgb:
                feature[idx]=int(node['split'].removeprefix('f'))
                cut[idx]=float(np.float32(node['split_condition']))
                children={c['nodeid']:c for c in node['children']}
                a,b=children[node['yes']],children[node['no']]
            else:
                assert node['decision_type']=='<=' and node['missing_type']!='Zero'
                feature[idx]=node['split_feature'];cut[idx]=node['threshold']
                a,b=node['left_child'],node['right_child']
            left[idx]=parse(a);right[idx]=parse(b)
        return idx
    trees=([json.loads(s) for s in model.get_booster().get_dump(dump_format='json')] if is_xgb else
           [t['tree_structure'] for t in model.booster_.dump_model()['tree_info']])
    for tree in trees:roots.append(parse(tree))
    return tuple(np.asarray(a) for a in (roots,feature,cut,left,right,leaf))

def permutation(fn,n_perm,seed):
    rng=np.random.default_rng(seed)
    raw=fn.maps[0].raw_index; m=fn.maps[0].M; n=len(fn.Te[0])
    engines=[]
    for (name,_,clf),tb in zip(fn.bases,fn.Tb):
        assert np.isfinite(tb).all()
        if name in ('self','lr'):
            engines.append(('lr',clf.coef_[0],clf.decision_function(tb)))
        else:
            xgb=name=='xgb'
            trees=forest_arrays(clf,xgb)
            native=clf.predict(tb,output_margin=True) if xgb else clf.predict(tb,raw_score=True)
            engines.append((name,trees,np.asarray(native,dtype=float)))
    v=np.zeros((n,m));base=np.zeros(n);max_error=0.
    for i in range(n):
        for p in range(n_perm):
            order=rng.permutation(m)
            for ordering in [order,order[::-1]]:
                switches=np.empty(m,dtype=int);switches[ordering]=np.arange(1,m+1)
                probs=[]
                for engine,te,tb in zip(engines,fn.Te,fn.Tb):
                    name,spec,initial=engine
                    if name=='lr':
                        delta=(te[i]-tb).astype(float)*spec
                        steps=np.zeros((m+1,len(tb)))
                        np.add.at(steps,switches[raw],delta.T)
                        probs.append(expit(initial+np.cumsum(steps,axis=0)))
                    else:
                        margin=tree_paths(te[i],tb,switches[raw],*spec,m,name=='xgb')
                        probs.append(expit(initial+(margin-margin[0])))
                if fn.ens is None:out=probs[0]
                elif fn.model=='soft_voting':out=np.average(probs,axis=0,weights=fn.ens.weights)
                else:
                    meta=fn.ens.final_estimator_
                    stack=np.stack(probs,axis=-1)
                    out=expit(stack@meta.coef_[0]+meta.intercept_[0])
                # Check path predictions directly against the native classifiers.
                # All positions on the first applicant, five positions per later applicant.
                positions=np.arange(m+1) if i==0 else [0,1+(i*17)%(m-1),m//2,m-1,m]
                for start in range(0,len(positions),8):
                    k=np.asarray(positions[start:start+8])
                    masked=[np.where((switches[raw][None,:]<=k[:,None])[:,None,:],te[i],tb).reshape(-1,tb.shape[1]) for te,tb in zip(fn.Te,fn.Tb)]
                    native=fn.prob(masked).reshape(len(k),-1)
                    err=float(np.max(np.abs(out[k]-native)))
                    max_error=max(max_error,err)
                    if err>=1e-5:raise ValueError(f'Native masked prediction mismatch: {err} (row {i})')
                g=out.mean(1)
                base[i]=g[0];v[i,ordering]+=np.diff(g)/(2*n_perm)
    return v,base,max_error
