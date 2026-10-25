"""Экспорт сохранённых bootstrap-массивов в CSV; без нового ресемплирования."""
from pathlib import Path
import csv
import json
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.pilot_evaluation import MODELS
from src.pilot_evaluation_decisions import THRESHOLDS,SCENARIOS,SCENARIO_NAMES,ATTRIBUTES
from src.pilot_evaluation_bootstrap import sha256,atomic_json,atomic_replace


def export_distributions(output):
    output=Path(output)
    manifest=json.loads((output/'evaluation_manifest.json').read_text(encoding='utf-8'))
    summary=json.loads((output/'evaluation_summary.json').read_text(encoding='utf-8'))
    assert summary['completed_bootstrap']==manifest['n_bootstrap']==1000
    schemas={
        'profit':['replicate','model','scenario','L','m','threshold','profit','approval_rate','approved_n','default_rate_among_approved'],
        'fairness':['replicate','model','threshold','attribute','approval_rate_gap','default_tpr_gap'],
        'threshold':['replicate','model','threshold','precision','recall','f1','TN','FP','FN','TP'],
    }
    streams={}; writers={}; paths={}; counts=dict.fromkeys(schemas,0)
    try:
        for name,columns in schemas.items():
            path=output/f'bootstrap_{name}_distribution.csv'
            temporary=path.with_name(path.name+'.tmp')
            stream=temporary.open('w',encoding='utf-8',newline='')
            streams[name]=stream; paths[name]=(temporary,path)
            writers[name]=csv.DictWriter(stream,fieldnames=columns)
            writers[name].writeheader()
        for rep in range(1000):
            path=output/'bootstrap'/f'replicate_{rep:04d}.npz'
            marker=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            assert sha256(path)==marker['sha256']
            with np.load(path,allow_pickle=False) as values:
                assert str(values['fingerprint'])==manifest['fingerprint']
                arrays={key:values[key] for key in values.files if key.startswith('decision_')}
            for m,model in enumerate(MODELS):
                for s,(loss,t) in enumerate(SCENARIOS):
                    writers['profit'].writerow({'replicate':rep,'model':model,'scenario':SCENARIO_NAMES[s],
                        'L':loss,'m':1,'threshold':THRESHOLDS[t],'profit':arrays['decision_profit'][m,s],
                        **{key:arrays[f'decision_{key}'][m,t] for key in ['approval_rate','approved_n','default_rate_among_approved']}})
                    counts['profit']+=1
                for t,threshold in enumerate(THRESHOLDS):
                    writers['threshold'].writerow({'replicate':rep,'model':model,'threshold':threshold,
                        **{key:arrays[f'decision_{key}'][m,t] for key in ['precision','recall','f1']},
                        **dict(zip(['TN','FP','FN','TP'],arrays['decision_confusion'][m,t]))})
                    counts['threshold']+=1
                    for a,attribute in enumerate(ATTRIBUTES):
                        writers['fairness'].writerow({'replicate':rep,'model':model,'threshold':threshold,
                            'attribute':attribute,'approval_rate_gap':arrays['decision_approval_gap'][m,t,a],
                            'default_tpr_gap':arrays['decision_tpr_gap'][m,t,a]})
                        counts['fairness']+=1
    finally:
        for stream in streams.values(): stream.close()
    for temporary,path in paths.values(): atomic_replace(temporary,path)
    result={'derived_from_completed_checkpoints':True,'new_resamples':0,'new_fit_calls':0,
            'files':{name:{'path':str(path.resolve()),'rows':counts[name],'sha256':sha256(path)}
                     for name,(_,path) in paths.items()}}
    atomic_json(output/'bootstrap_distribution_exports.json',result)
    return result


if __name__=='__main__':
    result=export_distributions(ROOT/'reports/pilot_20k_5config/evaluation')
    print(json.dumps(result,ensure_ascii=False))
