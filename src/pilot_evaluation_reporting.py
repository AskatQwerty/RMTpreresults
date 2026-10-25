"""Article tables, plots and Russian narrative from saved evaluation results.

No estimator is loaded; this module never fits a model or requests predictions.
Training protocol values come from the completed pilot's own saved artifacts.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_LABELS={
    'logistic_regression':'Logistic regression', 'random_forest':'Random forest',
    'xgboost':'XGBoost', 'lightgbm':'LightGBM', 'mlp':'MLP',
    'soft_voting':'Soft voting', 'stacking':'Stacking',
}
MODELS=list(MODEL_LABELS)
METRIC_LABELS={'ap':'AP','roc_auc':'ROC-AUC','log_loss':'Log-loss','brier':'Brier'}
REQUIRED_TABLES=('quality_point','quality_ci','rq2_deltas','ap_vs_lr','delong',
    'profit_point','baselines','fairness_groups','fairness_gaps','profit_ci','fairness_ci',
    'threshold_metrics','calibration_bins')
CI_NOTE='Pointwise, unadjusted, exploratory 95% percentile bootstrap CI; fixed trained models.'
LIMITATIONS=[
    'Пилот обучен на 20 000 development-заявках: остальные development-заявки в этом обучении не использованы.',
    'На пару проверены 5 конфигураций и выполнен один refit; основной A2 предусматривает 20 конфигураций и пять повторных обучений.',
    'Общие парные bootstrap-повторы отражают неопределённость состава test для фиксированных моделей, но не вариативность повторного обучения.',
    '95% percentile bootstrap-интервалы являются pointwise, unadjusted, exploratory; они не равны среднему ± SD пяти запусков.',
    'Парный DeLong и Bonferroni относятся только к семейству шести сравнений ROC-AUC с LR на Level 4; выводы не переносятся на AP.',
    'Отсутствие значимой разницы не доказывает эквивалентность. Формальное правило A2 «не хуже лучшего по AP» этим пилотом не подтверждается.',
    'Прибыль измеряется в условных единицах m=1 по заданной модели затрат; это не фактическая денежная прибыль банка.',
    'Percentile CI абсолютных fairness-разрывов не является тестом bias: неотрицательность и излом абсолютной величины в нуле требуют осторожной интерпретации, даже если нижняя граница выше нуля.',
    'Fairness-коррекции, обучение калибратора, SHAP и оценка устойчивости по пяти train-подвыборкам отложены.',
    'Методы текущего анализа зафиксированы для этого этапа, но не объявляются предварительно зарегистрированными.',
]


def _read_json(path,default=None):
    path=Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def _atomic_text(path,text):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(text,encoding='utf-8',newline='')
    os.replace(tmp,path)


def _json(path,value):
    _atomic_text(path,json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=str))


def _hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''): digest.update(chunk)
    return digest.hexdigest()


def _checked_pair_json(stage,name):
    marker=_read_json(stage/'complete.json')
    expected=(marker or {}).get('artifact_sha256',{}).get(name)
    path=(stage/name).resolve()
    if not path.is_relative_to(stage.resolve()) or not expected or not path.is_file() or _hash(path)!=expected:
        raise ValueError(f'Unchecked or damaged pilot metadata: {stage}/{name}')
    return _read_json(path)


def saved_protocol(pilot_root):
    """Read actual protocol and MLP diagnostics, without inferring past hardware."""
    root=Path(pilot_root)
    manifest=_read_json(root/'experiment_manifest.json')
    cohort=_read_json(root/'cohort_summary.json')
    summary=_read_json(root/'experiment_summary.json',{})
    if not manifest or manifest.get('scope')!='pilot_20k_5config' or not cohort:
        raise ValueError('Saved pilot manifest/cohort is missing or belongs to another scope')
    timings=[]; mlp=[]; warnings=[]; counts={}; source_paths=[root/'experiment_manifest.json',root/'cohort_summary.json']
    if (root/'experiment_summary.json').exists(): source_paths.append(root/'experiment_summary.json')
    test_balance=None
    for model in MODELS:
        for level in range(1,5):
            stage=root/model/f'level_{level}'
            record=_checked_pair_json(stage,'metrics.json'); source_paths.append(stage/'metrics.json')
            if record['scope']!='pilot_20k_5config' or record['model']!=model or record['level']!=level:
                raise ValueError(f'Wrong source model/level: {stage}')
            counts.setdefault(level,set()).add(int(record['raw_features']))
            timings.append({'model':model,'level':level,**{name:record.get(name) for name in
                ['search_and_refit_seconds','refit_seconds','predict_seconds','total_pair_seconds']}})
            n=int(record['metrics']['rows']); positive=int(record['metrics']['positives'])
            balance={'rows':n,'class_0':n-positive,'class_1':positive}
            if test_balance is not None and balance!=test_balance: raise ValueError('Inconsistent saved test counts')
            test_balance=balance
            for warning in record.get('warnings',[]): warnings.append({'model':model,'level':level,**warning})
            if model=='mlp':
                diagnostic_name=record.get('fit_diagnostics_file','fit_diagnostics.json')
                diagnostic=_checked_pair_json(stage,diagnostic_name); source_paths.append(stage/diagnostic_name)
                for item in diagnostic.get('mlp',[]):
                    actual=item.get('fitted',{}); config=item.get('config',{})
                    mlp.append({'model':model,'level':level,'n_iter_':actual.get('n_iter_'),
                        'max_iter':config.get('max_iter'),'tol':config.get('tol'),'n_iter_no_change':config.get('n_iter_no_change'),
                        'loss_curve_points':len(actual.get('loss_curve_',[])),'final_training_loss':actual.get('loss_'),
                        'best_cv_log_loss':record.get('best_cv_log_loss'),'test_log_loss':record['metrics'].get('log_loss'),
                        'test_ap':record['metrics'].get('pr_auc_average_precision'),
                        'warning_count_search_and_refit':sum(w.get('count',1) for w in record.get('warnings',[])),
                        'selected_params_json':json.dumps(record['selected_params'],ensure_ascii=False,sort_keys=True),
                        'fitted_config_json':json.dumps(config,ensure_ascii=False,sort_keys=True)})
    if any(len(values)!=1 for values in counts.values()): raise ValueError('Feature counts differ within a level')
    development=int(manifest['development_rows']); test=int(manifest['test_rows']); train=int(manifest['train_rows'])
    def class_counts(records): return {int(row['TARGET']):int(row['count']) for row in records}
    dev_counts=class_counts(cohort['development_balance']); pilot_counts=class_counts(cohort['cohort_balance'])
    balances=[{'sample':'Labelled data','rows':development+test,'class_0':dev_counts[0]+test_balance['class_0'],
               'class_1':dev_counts[1]+test_balance['class_1']},
        {'sample':'Development','rows':development,'class_0':dev_counts[0],'class_1':dev_counts[1]},
        {'sample':'Pilot train','rows':train,'class_0':pilot_counts[0],'class_1':pilot_counts[1]},
        {'sample':'Fixed test',**test_balance}]
    events=[]
    if (root/'events.jsonl').exists():
        events=[json.loads(line) for line in (root/'events.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        source_paths.append(root/'events.jsonl')
    plan=manifest.get('protocol',{})
    # CPU model and total RAM were not recorded by this pilot. A current system
    # probe would describe the evaluation environment, not historical training.
    recorded_hardware=manifest.get('hardware') or summary.get('hardware')
    protocol={'scope':'pilot_20k_5config','original_labelled_rows':development+test,
        'development_rows':development,'test_rows':test,'pilot_train_rows':train,
        'unused_development_rows':development-train,'class_balance':balances,
        'cv_folds':cohort['cv'],'feature_counts':{str(level):next(iter(values)) for level,values in counts.items()},
        'pilot_selection_seed':manifest.get('cohort_contract',{}).get('selection_seed'),
        'pilot_cv_seed':manifest.get('cohort_contract',{}).get('cv_seed'),
        'search_seed':plan.get('search_random_state'),'saved_plan':plan,
        'python_version_training':manifest.get('python'),'software_versions_training':manifest.get('versions',{}),
        'threads_configured':manifest.get('threads'),'search_n_jobs':manifest.get('search_n_jobs'),'pre_dispatch':manifest.get('pre_dispatch'),
        'hardware_recorded':recorded_hardware is not None,'hardware':recorded_hardware,
        'hardware_note':'Модель CPU и общий объём RAM не сохранены в журналах обучения пилота; оборудование текущей оценки не подставляется.'
                        if recorded_hardware is None else 'Прочитано из сохранённого manifest/summary обучения.',
        'completed_candidate_cv_fits':summary.get('candidate_cv_fits_completed'),
        'completed_refits':summary.get('completed_final_refits'),
        'completed_primitive_fits':summary.get('primitive_fits_completed'),
        'last_invocation_elapsed_seconds':summary.get('elapsed_seconds_this_invocation'),
        'training_first_event_utc':events[0].get('utc') if events else None,
        'training_last_event_utc':events[-1].get('utc') if events else None,
        'search_and_refit_seconds_sum':sum(r['search_and_refit_seconds'] for r in timings),
        'refit_seconds_sum':sum(r['refit_seconds'] for r in timings),
        'timing_note':'refit_seconds входит в search_and_refit_seconds; их нельзя суммировать. Время invocation также включает IO и отчёты.',
        'mlp_warning_note':'Предупреждения относятся ко всему search+refit, включая кандидатов CV; они не доказывают достижение max_iter выбранной моделью.',
        'sources':[{'path':str(p.resolve()),'sha256':_hash(p)} for p in source_paths]}
    return protocol,pd.DataFrame(timings),pd.DataFrame(mlp),pd.DataFrame(warnings)


def _escape(value):
    replacements={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$',
                  '#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    return ''.join(replacements.get(char,char) for char in str(value))


def _latex(frame,caption,label,*,long=False):
    env='longtable' if long else 'tabular'
    resized=not long and len(frame.columns)>=5
    if long:
        lines=[r'\begingroup',r'\scriptsize',r'\setlength{\tabcolsep}{3pt}',
               r'\begin{longtable}{'+'l'*len(frame.columns)+'}']
    else:
        lines=[r'\begin{table}[htbp]',r'\centering',r'\small',r'\caption{'+_escape(caption)+'}',r'\label{'+label+'}']
        if resized: lines.append(r'\resizebox{\linewidth}{!}{%')
        lines.append(r'\begin{tabular}{'+'l'*len(frame.columns)+'}')
    if long: lines.append(r'\caption{'+_escape(caption)+'}'+r'\label{'+label+'}'+r'\\')
    lines.extend([r'\toprule',' & '.join(_escape(c) for c in frame.columns)+r' \\',r'\midrule'])
    for _,row in frame.iterrows():
        cells=[]
        for value in row:
            text='--' if pd.isna(value) else str(value)
            parts=text.split('\n')
            # Braces prevent a CI beginning '[' after '\\' from being parsed
            # as TeX's optional row-spacing argument.
            cells.append(r'\shortstack{' + r' \\ '.join('{'+_escape(part)+'}' for part in parts)+'}' if len(parts)>1 else _escape(text))
        lines.append(' & '.join(cells)+r' \\')
    lines.extend([r'\bottomrule',r'\end{'+env+'}'])
    if long: lines.append(r'\endgroup')
    else:
        if resized: lines.append('}')
        lines.append(r'\end{table}')
    return '\n'.join(lines)+'\n'


def _ci(value,low=None,high=None,*,digits=4,scale=1):
    if value is None or pd.isna(value): return '--'
    point=f'{float(value)*scale:.{digits}f}'
    if low is None or high is None or pd.isna(low) or pd.isna(high): return point
    return point+f'\n[{float(low)*scale:.{digits}f}, {float(high)*scale:.{digits}f}]'


def _matching(frame,**keys):
    result=frame
    for key,value in keys.items():
        result=result.loc[np.isclose(result[key],value,atol=1e-12,rtol=0) if isinstance(value,float) else result[key]==value]
    if len(result)>1: raise ValueError(f'Duplicate reporting key: {keys}')
    return result.iloc[0] if len(result) else None


def _ci_row(frame,**keys):
    row=_matching(frame,**keys)
    return (row['estimate'],row['ci_low'],row['ci_high']) if row is not None else (None,None,None)


def _save_table(output,name,frame,caption,*,long=False):
    directory=Path(output)/'tables'
    csv=directory/f'{name}.csv'; tex=directory/f'{name}.tex'
    _atomic_text(csv,frame.to_csv(index=False)); _atomic_text(tex,_latex(frame,caption,'tab:'+name.replace('_','-'),long=long))
    return {'csv':str(csv.resolve()),'tex':str(tex.resolve()),'caption':caption}


def article_tables(tables):
    quality=tables['quality_point']; ci=tables['quality_ci']; deltas=tables['rq2_deltas']
    rq1=[]; levels=[]; baseline=[]; sequential=[]; errors=[]
    for model in MODELS:
        label=MODEL_LABELS[model]
        rq1.append({'Model':label,**{METRIC_LABELS[metric]:_ci(*_ci_row(ci,model=model,level=4,metric=metric))
                                    for metric in METRIC_LABELS}})
        levels.append({'Model':label,**{f'Level {level}':_ci(*_ci_row(ci,model=model,level=level,metric='ap'))
                                       for level in range(1,5)}})
        for kind,changes,destination in [('baseline',[(1,2),(1,3),(1,4)],baseline),
                                         ('sequential',[(1,2),(2,3),(3,4)],sequential)]:
            result={'Model':label}
            for start,end in changes:
                row=_matching(deltas,model=model,kind=kind,from_level=start,to_level=end)
                result[f'Delta AP L{end}-L{start}']=_ci(row['delta_ap'],row['ci_low'],row['ci_high']) if row is not None else '--'
            destination.append(result)
        point=_matching(quality,model=model,level=4)
        errors.append({'Model':label,'Missed defaults (FN)':int(point['fn']),'False rejections (FP)':int(point['fp']),
            'Precision':_ci(point['precision']),'Recall':_ci(point['recall']),'F1':_ci(point['f1'])})
    profit_rows=[]; tpr_rows=[]
    scenarios=['L1_t0.5','L5_t0.5','L5_t1/6','L10_t0.5','L10_t1/11']
    def yes(value):
        if value is None or pd.isna(value): return 'NA'
        return 'Yes' if value in (True,'True',1) else 'No'
    for model in MODELS:
        for scenario in scenarios:
            point=_matching(tables['profit_point'],model=model,scenario=scenario)
            if point is None: raise ValueError(f'Missing profit scenario {model}/{scenario}')
            threshold=float(point['threshold']); gap=_matching(tables['fairness_gaps'],model=model,threshold=threshold)
            row={'Model':MODEL_LABELS[model],'L/m':int(point['loss_to_margin']),'Threshold':f'{threshold:.4f}',
                'Profit (m)':_ci(*_ci_row(tables['profit_ci'],model=model,scenario=scenario,metric='realized_profit_per_applicant'),digits=3),
                'Approval (%)':_ci(*_ci_row(tables['profit_ci'],model=model,scenario=scenario,metric='approval_rate'),digits=1,scale=100)}
            for attribute,label in [('gender','Gender gap (pp)'),('age','Age gap (pp)')]:
                row[label]=_ci(*_ci_row(tables['fairness_ci'],model=model,threshold=threshold,attribute=attribute,metric='approval_rate_gap'),digits=2,scale=100)
            row['G/A gap <=5pp']=f'{yes(gap["gender_approval_gap_le_0_05"])}/{yes(gap["age_approval_gap_le_0_05"])}'
            profit_rows.append(row)
        for threshold in [0.5,1/6,1/11]:
            tpr_rows.append({'Model':MODEL_LABELS[model],'Threshold':f'{threshold:.4f}',
                'Gender TPR gap (pp)':_ci(*_ci_row(tables['fairness_ci'],model=model,threshold=threshold,attribute='gender',metric='default_tpr_gap'),digits=2,scale=100),
                'Age TPR gap (pp)':_ci(*_ci_row(tables['fairness_ci'],model=model,threshold=threshold,attribute='age',metric='default_tpr_gap'),digits=2,scale=100)})
    vs_lr=[]
    for model in MODELS:
        if model=='logistic_regression': continue
        row=_matching(tables['ap_vs_lr'],model=model)
        vs_lr.append({'Model':MODEL_LABELS[model],'Reference':'Logistic regression',
            'Delta AP':_ci(row['delta_ap'],row['ci_low'],row['ci_high'])})
    profit_frame=pd.DataFrame(profit_rows)
    output={
        'rq1_level4':(pd.DataFrame(rq1),'Level 4 quality. Primary PR metric: PR-AUC (average precision, AP). '+CI_NOTE,False),
        'rq2_ap_levels':(pd.DataFrame(levels),'PR-AUC (average precision, AP) across information levels. '+CI_NOTE,False),
        'rq2_baseline_gains':(pd.DataFrame(baseline),'Paired AP gains relative to application-only Level 1. '+CI_NOTE,False),
        'rq2_sequential_gains':(pd.DataFrame(sequential),'Paired AP gains between consecutive information levels. '+CI_NOTE,False),
        'ap_vs_lr_level4':(pd.DataFrame(vs_lr),'Paired Level 4 AP differences versus logistic regression. No bootstrap p-values. '+CI_NOTE,False),
        'errors_level4':(pd.DataFrame(errors),'Level 4 errors at fixed threshold 0.5. Positive class: default (TARGET=1). Undefined precision is shown as -- when no defaults are predicted; it is not zero.',False),
        'profit_fairness_all':(profit_frame,'All fixed scenarios. Profit per applicant in units m=1; rejected applications contribute zero. G/A: observed gender/age approval-rate gap <=5 percentage points. '+CI_NOTE,True),
        'tpr_gaps_level4':(pd.DataFrame(tpr_rows),'Absolute default-detection TPR gaps. These are not the 5 percentage point approval-gap constraint. '+CI_NOTE,True),
    }
    for loss in [1,5,10]:
        frame=profit_frame.loc[profit_frame['L/m']==loss].drop(columns='L/m').reset_index(drop=True)
        output[f'profit_fairness_L{loss}']=(frame,f'Fixed cost scenario L/m={loss}; both threshold rules coincide at L/m=1 and are listed once. G/A: gender/age approval-gap constraint. '+CI_NOTE,False)
    baseline=tables['baselines'].copy()
    baseline=baseline[['policy','loss_to_margin','realized_profit_per_applicant','approval_rate','approved_n','default_rate_among_approved']]
    baseline.columns=['Policy','L/m','Profit (m)','Approval rate','Approved n','Default rate among approved']
    output['profit_baselines']=(baseline.round(4),'Approve-all and reject-all reference policies; no threshold is optimized on test.',False)
    counts=tables['calibration_bins'].pivot(index='model',columns='bin',values='n').reindex(index=MODELS,columns=range(10)).astype(int)
    counts.index=counts.index.map(MODEL_LABELS); counts.index.name='Model'; counts.columns=[f'Bin {i+1}' for i in range(10)]
    output['calibration_bin_counts']=(counts.reset_index(),'Level 4 calibration-bin counts. Ten equal-width intervals: Bin 1 [0,0.1), ..., Bin 10 [0.9,1]. Empty bins have n=0.',False)
    return output


def _make_figures(tables,output,*,pdf_authorized):
    if not pdf_authorized:
        raise RuntimeError('PDF authoring requires the once-per-task artifact marker before this call.')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    directory=Path(output)/'figures'; directory.mkdir(parents=True,exist_ok=True)
    colors=['#1769aa','#c87912','#2a8d61','#8c4bb2','#c84343','#168b91','#555555']
    fig,ax=plt.subplots(figsize=(9.6,5.6))
    for model,color in zip(MODELS,colors):
        rows=tables['quality_ci'].query('model==@model and metric=="ap"').sort_values('level')
        x=rows['level'].to_numpy(); y=rows['estimate'].to_numpy()
        error=np.vstack([y-rows['ci_low'].to_numpy(),rows['ci_high'].to_numpy()-y])
        ax.errorbar(x,y,yerr=error,label=MODEL_LABELS[model],color=color,marker='o',markersize=4.5,
                    linewidth=1.6,capsize=2.5,elinewidth=.8,alpha=.95)
    ax.set(xticks=[1,2,3,4],xlabel='Information level',ylabel='PR-AUC (average precision, AP)',
           title='Discrimination across information levels')
    ax.grid(axis='y',color='#dce3e9',linewidth=.6)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.14),ncol=4,frameon=False,fontsize=9)
    fig.text(.5,.025,'95% paired percentile bootstrap CI (pointwise, unadjusted, exploratory); no retraining variability.',ha='center',fontsize=8.5)
    fig.subplots_adjust(left=.10,right=.98,top=.91,bottom=.25)
    files=[]
    for extension in ['pdf','svg']:
        path=directory/f'ap_by_level.{extension}'; fig.savefig(path,bbox_inches='tight'); files.append(str(path.resolve()))
    plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(12,6.6),sharex=True,sharey=True)
    for model,color,ax in zip(MODELS,colors,axes.flat):
        rows=tables['calibration_bins'].loc[tables['calibration_bins']['model']==model].sort_values('bin')
        nonempty=rows.loc[rows['n']>0]
        ax.plot([0,1],[0,1],linestyle='--',color='#9aa1a8',linewidth=.8)
        ax.plot(nonempty['mean_probability_default'],nonempty['observed_default_rate'],color=color,linewidth=1.3)
        sizes=12+100*np.sqrt(nonempty['n']/nonempty['n'].sum())
        ax.scatter(nonempty['mean_probability_default'],nonempty['observed_default_rate'],s=sizes,color=color,alpha=.85)
        ax.set(title=MODEL_LABELS[model],xlim=(-.02,1.02),ylim=(-.02,1.02),xticks=[0,.5,1],yticks=[0,.5,1])
        ax.grid(color='#e3e7eb',linewidth=.5)
    axes.flat[-1].axis('off')
    axes.flat[-1].text(.03,.88,'Level 4: fixed probabilities\n\n10 equal-width bins on [0, 1].\nIntervals [a, b); final bin includes 1.\nEmpty bins omitted from curves.\n\nMarker size increases with bin n.\nExact counts: calibration_bins.csv.\nSmall bins have high uncertainty.\n\nNo calibrator was fitted.',
                       transform=axes.flat[-1].transAxes,fontsize=9,va='top',linespacing=1.45)
    fig.supxlabel('Mean predicted default probability',y=.035)
    fig.supylabel('Observed default rate',x=.018)
    fig.suptitle('Calibration of saved Level 4 probabilities',y=.98,fontsize=13)
    fig.subplots_adjust(left=.075,right=.985,top=.91,bottom=.12,hspace=.30,wspace=.22)
    for extension in ['pdf','svg']:
        path=directory/f'calibration_level4.{extension}'; fig.savefig(path,bbox_inches='tight'); files.append(str(path.resolve()))
    plt.close(fig)
    return files


def _narrative(tables,protocol,mlp,evaluation_summary):
    quality=tables['quality_point']; l4=quality.loc[quality['level']==4]
    paragraphs=[f"Оценены 28 сохранённых сочетаний семи моделей и четырёх уровней на общем test из {protocol['test_rows']:,} клиентов. "
        'Основная PR-метрика: PR-AUC (average precision, AP). В этом этапе новые обучения, predict, подбор параметров, калибратор и fairness-коррекции не выполнялись.']
    best=[]
    for metric in ['ap','roc_auc','log_loss','brier']:
        row=l4.loc[l4[metric].idxmax() if metric in ['ap','roc_auc'] else l4[metric].idxmin()]
        best.append(f"{METRIC_LABELS[metric]}: {MODEL_LABELS[row['model']]} ({row[metric]:.4f})")
    paragraphs.append('Лучшие отдельные точечные значения на Level 4: '+'; '.join(best)+'. Это описательное сравнение, а не формальный выбор core-модели по A2.')
    vs=tables['ap_vs_lr']
    above=vs.loc[vs['ci_low']>0,'model'].map(MODEL_LABELS).tolist()
    overlap=vs.loc[(vs['ci_low']<=0)&(vs['ci_high']>=0),'model'].map(MODEL_LABELS).tolist()
    below=vs.loc[vs['ci_high']<0,'model'].map(MODEL_LABELS).tolist()
    paragraphs.append('Exploratory pointwise интервалы разности AP с LR полностью выше нуля для: '+(', '.join(above) or 'ни одной модели')+
        '; полностью ниже нуля для: '+(', '.join(below) or 'ни одной модели')+
        '. Ноль входит в интервал для: '+(', '.join(overlap) or 'ни одной модели')+'. Bootstrap p-values не вычислялись.')
    delong=tables['delong']; rejected=delong.loc[delong['reject_bonferroni_0_05']==True]
    rejected_text=[f"{MODEL_LABELS[row.model]} (ROC-AUC {'выше' if row.delta_roc_auc>0 else 'ниже'} LR; "
                   f"Δ={row.delta_roc_auc:+.4f}, p_adj={row.p_bonferroni:.3g})" for row in rejected.itertuples()]
    paragraphs.append('Парный DeLong с Bonferroni в семействе шести сравнений ROC-AUC с LR дал p_adj<0.05 для: '+
        ('; '.join(rejected_text) or 'ни одной модели')+'. Это вывод только о ROC-AUC; он не доказывает различия AP или эквивалентность остальных моделей.')
    gains=tables['rq2_deltas'].loc[(tables['rq2_deltas']['kind']=='baseline')&(tables['rq2_deltas']['to_level']==4)]
    gain_text=[]
    for model in MODELS:
        row=_matching(gains,model=model)
        gain_text.append(f"{MODEL_LABELS[model]}: {row['delta_ap']:+.4f} [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]")
    paragraphs.append('RQ2, прирост AP Level 4 относительно Level 1: '+'; '.join(gain_text)+'. Все интервалы парные; дополнительные baseline и последовательные сравнения приведены в таблицах.')
    negative=tables['rq2_deltas'].loc[(tables['rq2_deltas']['kind']=='sequential')&(tables['rq2_deltas']['delta_ap']<0)]
    if not negative.empty:
        pieces=[f"{MODEL_LABELS[row.model]} L{int(row.to_level)}-L{int(row.from_level)}: {row.delta_ap:+.4f} "
                f"[{row.ci_low:+.4f}, {row.ci_high:+.4f}]" for row in negative.itertuples()]
        paragraphs.append('Отрицательные точечные последовательные изменения AP: '+'; '.join(pieces)+
            '. Добавление истории не гарантирует улучшения каждой модели на каждом шаге; интервалы остаются exploratory.')
    gaps=tables['fairness_gaps']; pass_both=gaps.loc[gaps['both_approval_gaps_le_0_05']==True]
    paragraphs.append(f"Наблюдаемому ограничению approval-rate gap ≤5 п.п. одновременно по полу и возрасту соответствуют {len(pass_both)} "
        f"из {len(gaps)} уникальных сочетаний модели и порога. "+
        ('Подходящих вариантов нет; ограничение не ослабляется. ' if pass_both.empty else '')+
        'Половые и возрастные разрывы показаны отдельно; TPR-gap не подменяет approval-gap. Эти отметки не являются подтверждением общего правила отбора A2.')
    paragraphs.append('Интервалы абсолютных fairness-разрывов описательны. Из-за неотрицательности и излома абсолютной величины в нуле нижняя граница выше нуля сама по себе не является формальным тестом наличия bias.')
    threshold_pass=[]
    for threshold in [0.5,1/6,1/11]:
        rows=gaps.loc[np.isclose(gaps['threshold'],threshold,atol=1e-12,rtol=0)]
        names=rows.loc[rows['both_approval_gaps_le_0_05']==True,'model'].map(MODEL_LABELS).tolist()
        threshold_pass.append(f"t={threshold:.4f}: "+(', '.join(names) if names else 'подходящих вариантов нет'))
    paragraphs.append('По фиксированным порогам оба наблюдаемых approval-gap ограничения выполняют: '+'; '.join(threshold_pass)+
        '. Это проверка точечных разрывов, не гарантия ограничения в популяции; интервалы разрывов приведены отдельно.')
    profit=tables['profit_point']
    for loss in [1,5,10]:
        rows=profit.loc[profit['loss_to_margin']==loss]
        maximum=rows.loc[rows['realized_profit_per_applicant'].idxmax()]
        paragraphs.append(f"При L/m={loss} наблюдаемая прибыль семи моделей и фиксированных порогов лежит от "
            f"{rows['realized_profit_per_applicant'].min():.3f} до {rows['realized_profit_per_applicant'].max():.3f} m на заявителя; "
            f"доля одобрений — от {100*rows['approval_rate'].min():.1f}% до {100*rows['approval_rate'].max():.1f}%. "
            f"Наибольшее точечное значение среди этих фиксированных вариантов: {MODEL_LABELS[maximum['model']]}, t={maximum['threshold']:.4f}. "
            'Это описательный максимум, не статистически установленный победитель; порог по test не оптимизировался. Approve-all/reject-all приведены как ориентиры.')
    groups=tables['fairness_groups']
    unknown=groups.loc[groups['included_in_binary_gap']==False]
    details=[]
    for attribute in ['gender','age']:
        sample=unknown.loc[unknown['attribute']==attribute]
        by_key=sample.groupby(['model','threshold'])['n'].sum() if not sample.empty else pd.Series(dtype=float)
        values=sorted(set(int(x) for x in by_key))
        details.append(f'{attribute}: '+('/'.join(map(str,values)) if values else '0'))
    paragraphs.append('Неизвестные fairness-группы сохранены отдельно ('+'; '.join(details)+
        ' заявок на общей test-выборке), исключены из бинарных разрывов, но остаются в общих метриках и прибыли. Ровно 25 лет относится к группе >=25.')
    mlp_point=_matching(quality,model='mlp',level=4)
    lr=_matching(quality,model='logistic_regression',level=4)
    paragraphs.append(f"При t=0.5 LR пропускает {int(lr['fn'])} дефолтов и ошибочно отклоняет {int(lr['fp'])} заявок TARGET=0; "
        f"MLP — соответственно {int(mlp_point['fn'])} и {int(mlp_point['fp'])}. Ошибки всех моделей приведены рядом с precision/recall/F1. "
        'Изменение порога меняет этот баланс, поэтому дополнительные фиксированные пороги сохранены отдельно.')
    candidate_errors=[]
    for model in ['soft_voting','stacking']:
        row=_matching(quality,model=model,level=4)
        candidate_errors.append(f"{MODEL_LABELS[model]}: FN={int(row['fn'])}, FP={int(row['fp'])}")
    paragraphs.append('Для ансамблей при том же пороге: '+'; '.join(candidate_errors)+
        '. Разный баланс пропущенных дефолтов и ошибочных отказов не означает превосходства по всем метрикам. '
        'Random forest при t=0.5 одобряет всех: нулевые групповые разрывы сочетаются с нулевым обнаружением дефолтов и не являются свидетельством полезного решения кредитного риска.')
    undefined=l4.loc[l4['precision'].isna(),'model'].map(MODEL_LABELS).tolist()
    if undefined:
        paragraphs.append('Precision при t=0.5 не определена для: '+', '.join(undefined)+
            ', поскольку модель не предсказывает ни одного дефолта. Это пропуск с нулевым знаменателем, а не precision=0; соответствующий bootstrap CI может иметь 0 допустимых повторов. '
            'Нулевые recall/F1 при наличии фактических дефолтов сохраняются как нули.')
    epochs=', '.join(f"L{int(row.level)}={int(row.n_iter_)}" for row in mlp.itertuples())
    warning_counts=', '.join(f"L{int(row.level)}={int(row.warning_count_search_and_refit)}" for row in mlp.itertuples())
    paragraphs.append(f"MLP сохраняется во всех таблицах, несмотря на слабые значения AP и log-loss. Фактические n_iter_: {epochs}; "
        f"агрегированные ConvergenceWarning за search+refit: {warning_counts}. Предупреждения включают CV-кандидатов, поэтому не равны числу ошибок выбранного refit. "
        'Train loss_ содержит особенности оптимизации и регуляризацию и не тождественен внешней log-loss. Эти наблюдения не устанавливают переобучение, ошибку предобработки или лимит итераций как причину слабого качества.')
    completed=evaluation_summary.get('completed_bootstrap')
    paragraphs.append(f'Вычислено {completed} общих парных bootstrap-повторов test, seed=42. При каждом повторе одни и те же клиентские веса применены ко всем моделям и уровням. '
        'Интервалы описывают test-неопределённость фиксированных обученных моделей; новые train-повторы не выполнялись.')
    return paragraphs


def export_evaluation_report(pilot_root,evaluation_dir,*,previous_reports=None,tables=None,
                             create_figures=False,pdf_authorized=False):
    """Export saved analysis to English article material and a Russian report.

    create_figures=True must follow the root agent's successful one-time PDF
    artifact marker. A later call with False refreshes readiness after visual QA.
    """
    root=Path(pilot_root); output=Path(evaluation_dir)
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError('Evaluation output must not be inside frozen training artifacts')
    previous=Path(previous_reports) if previous_reports is not None else root.parents[2]/'reports'/root.name
    supplied={} if tables is None else dict(tables)
    missing=[]
    for name in REQUIRED_TABLES:
        if name not in supplied:
            path=output/f'{name}.csv'
            if path.exists(): supplied[name]=pd.read_csv(path)
            else: missing.append(name)
    if missing:
        return {'status':'waiting_for_analysis','missing_tables':missing,'new_fit_calls':0,'new_predict_calls':0,
                'output_dir':str(output.resolve())}
    quality=supplied['quality_point']
    if len(quality)!=28 or quality.duplicated(['model','level']).any():
        raise ValueError('Exactly 28 unique model/level quality rows are required')
    protocol,timing,mlp,warnings=saved_protocol(root)
    output.mkdir(parents=True,exist_ok=True)
    _json(output/'actual_training_protocol.json',protocol)
    _atomic_text(output/'latex_usage.txt',r'''Подключение таблиц в статью

Требуемые пакеты в преамбуле:
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{graphicx}

Пример включения готового фрагмента:
\input{tables/rq1_level4.tex}

Короткие таблицы с пятью и более столбцами используют
\resizebox{\linewidth}{!}{...}; подпись остаётся вне масштабирования.
В двухколоночной статье для широких основных таблиц замените окружение
table на table* (и соответствующий end). Тогда linewidth относится
к полной ширине двух колонок. Проверьте итоговую читаемость шрифта
в шаблоне журнала; не растягивайте таблицы вручную за границы полосы.

Таблицы longtable не масштабируются: они используют локальные
\scriptsize и \tabcolsep=3pt. Размещайте их в одноколоночном разделе
приложения; не вкладывайте longtable в table/table* или resizebox.

Точечная оценка находится на первой строке ячейки, 95% CI — на второй.
CI является pointwise, unadjusted, exploratory percentile bootstrap
интервалом при фиксированных обученных моделях. Он не учитывает
вариативность повторного обучения и не заменяет среднее ± SD пяти запусков.
Знак -- означает неопределённое значение, а не ноль.

Названия моделей и подписи английские. Для исходного текста статьи
используйте кодировку UTF-8 и стандартную настройку её шрифтов.
Эти фрагменты не компилировались в локальный TeX: проверьте окончательную
верстку в целевом шаблоне статьи. PDF/SVG рисунки проверяются отдельно.
''')
    for name,frame in [('training_timings',timing),('mlp_actual_diagnostics',mlp),('saved_training_warnings',warnings)]:
        _atomic_text(output/f'{name}.csv',frame.to_csv(index=False))
    prepared_tables=article_tables(supplied)
    d=supplied['delong'].copy()
    delong_paper=pd.DataFrame({'Model':d['model'].map(MODEL_LABELS),'Delta ROC-AUC':d['delta_roc_auc'].map(lambda v:f'{v:+.4f}'),
        'p (raw)':d['p_raw'].map(lambda v:f'{v:.3g}' if pd.notna(v) else '--'),
        'p (Bonferroni)':d['p_bonferroni'].map(lambda v:f'{v:.3g}' if pd.notna(v) else '--'),
        'Reject at 0.05':d['reject_bonferroni_0_05'].map({True:'Yes',False:'No'})})
    prepared_tables['delong_level4']=(delong_paper,'Paired two-sided DeLong ROC-AUC tests versus logistic regression. Bonferroni correction across six comparisons; not an AP test.',False)
    table_files={name:_save_table(output,name,frame,caption,long=long)
                 for name,(frame,caption,long) in prepared_tables.items()}
    balance=pd.DataFrame(protocol['class_balance'])
    table_files['sample_balance']=_save_table(output,'sample_balance',balance,'Actual saved sample sizes and class counts.')
    cv=pd.DataFrame([{key:fold[key] for key in ['fold','train_rows','validation_rows']} for fold in protocol['cv_folds']])
    table_files['cv_sizes']=_save_table(output,'cv_sizes',cv,'Three saved pilot CV folds, common to all models and levels.')
    summary=_read_json(output/'evaluation_summary.json',{})
    paragraphs=_narrative(supplied,protocol,mlp,summary)
    if create_figures: _make_figures(supplied,output,pdf_authorized=pdf_authorized)
    expected_figures=[output/'figures'/f'{name}.{extension}' for name in ['ap_by_level','calibration_level4'] for extension in ['pdf','svg']]
    figure_files=[str(path.resolve()) for path in expected_figures if path.exists()]
    graphics=_read_json(output/'graphics_validation.json',{})
    alignment=_read_json(output/'alignment_validation.json',{})
    integrity=_read_json(output/'integrity_after.json',{})
    statistical_path=output/'statistical_validation.json'
    statistical=_read_json(statistical_path)
    blockers=[]
    if summary.get('completed_bootstrap')!=1000: blockers.append('Не подтверждены все 1000 bootstrap-повторов.')
    main_ci=supplied['quality_ci'].loc[(supplied['quality_ci']['metric']=='ap')|
        ((supplied['quality_ci']['level']==4)&supplied['quality_ci']['metric'].isin(['roc_auc','log_loss','brier']))]
    if len(main_ci)!=49 or not (main_ci['bootstrap_n_valid']==1000).all():
        blockers.append('Для основных AP/ROC-AUC/log-loss/Brier интервалов не подтверждены 1000 допустимых повторов.')
    if summary.get('input_integrity_unchanged') is not True or integrity.get('unchanged') is not True:
        blockers.append('Не подтверждена неизменность входных артефактов после анализа.')
    if summary.get('alignment_verified') is not True: blockers.append('Не подтверждено согласование test-ID/TARGET.')
    if summary.get('existing_metrics_match') is not True: blockers.append('Не подтверждено согласие с прежними метриками.')
    if summary.get('new_fit_calls')!=0 or summary.get('new_predict_calls')!=0: blockers.append('Не подтверждено отсутствие новых fit/predict.')
    if len(figure_files)!=4: blockers.append('Не готовы обе пары PDF/SVG рисунков.')
    visual_ok=graphics.get('passed') is True or graphics.get('all_passed') is True
    if not visual_ok: blockers.append('Финальная визуальная проверка PDF ещё не зафиксирована.')
    if statistical is not None and statistical.get('status')!='passed':
        blockers.append('Независимая статистическая проверка сохранена, но не имеет статуса passed.')
    readiness={'scope':'pilot_20k_5config_evaluation','status':'ready' if not blockers else 'pending_checks',
        'ready_for_article':not blockers,'generated_utc':datetime.now(timezone.utc).isoformat(),
        'new_fit_calls':0,'new_predict_calls':0,'completed_bootstrap':summary.get('completed_bootstrap'),
        'training_protocol_file':str((output/'actual_training_protocol.json').resolve()),
        'table_files':table_files,'figure_files':figure_files,'graphics_validation':graphics,
        'latex_usage_file':str((output/'latex_usage.txt').resolve()),
        'statistical_validation':{'status':statistical.get('status') if statistical is not None else 'not_provided',
            'file':str(statistical_path.resolve()) if statistical is not None else None,
            'checks':len(statistical.get('checks',[])) if statistical is not None else 0},
        'input_integrity_unchanged':integrity.get('unchanged'),'alignment_validation':alignment,
        'blockers':blockers,'limitations':LIMITATIONS+[protocol['hardware_note']],
        'methods':{'primary_pr_metric':'PR-AUC (average precision, AP); sklearn average_precision_score',
            'bootstrap':'1000 common paired client resamples; default_rng(42); percentile quantiles 0.025/0.975',
            'ci_interpretation':CI_NOTE,'ap_comparisons':'Six Level 4 AP differences versus LR; within-model baseline and sequential level gains',
            'roc_auc_test':'Paired two-sided DeLong; six Level 4 comparisons versus LR; Bonferroni family size 6',
            'profit':'Observed TARGET; approve if p_default<threshold; m=1, L=1/5/10; fixed0.5 and m/(m+L); L=1 duplicate removed',
            'fairness_constraint':'Observed approval-rate gap <=0.05, separately gender and age; no relaxed limit and no A2 equivalence claim',
            'calibration':'10 fixed equal-width bins, [a,b), final includes1; exact counts in calibration_bins.csv; no fitting'},
        'sources':{'training':protocol['sources'],'evaluation_manifest':str((output/'evaluation_manifest.json').resolve()),
            'previous_reports':str(previous.resolve()),'evaluation_tables':{name:str((output/f'{name}.csv').resolve()) for name in REQUIRED_TABLES}},
        'a2_selection_formally_confirmed':False,'preregistered_claim':False}
    _json(output/'results_readiness.json',readiness)
    protocol_text=(f"Исходная размеченная выборка: {protocol['original_labelled_rows']:,}; development {protocol['development_rows']:,}, "
        f"test {protocol['test_rows']:,}; пилот {protocol['pilot_train_rows']:,}; не использовано в обучении {protocol['unused_development_rows']:,} development-заявок. "
        f"Число исходных подготовленных признаков по уровням: {protocol['feature_counts']}. "
        f"Завершено CV-fit {protocol['completed_candidate_cv_fits']}, refit {protocol['completed_refits']}; "
        f"последний invocation {protocol['last_invocation_elapsed_seconds']:.1f} с. {protocol['timing_note']} "
        f"Python {protocol['python_version_training']}; software versions: {protocol['software_versions_training']}. "
        f"Threads={protocol['threads_configured']}, search_n_jobs={protocol['search_n_jobs']}, pre_dispatch={protocol['pre_dispatch']}. {protocol['hardware_note']}")
    protocol_text+=' Seeds: выбор пилотных клиентов '+str(protocol['pilot_selection_seed'])+', CV '+str(protocol['pilot_cv_seed'])+', поиск '+str(protocol['search_seed'])+'. '
    protocol_text+='Размеры train/validation CV-фолдов: '+', '.join(f"fold {fold['fold']}: {fold['train_rows']}/{fold['validation_rows']}" for fold in protocol['cv_folds'])+'. '
    protocol_text+='Баланс классов (0/1): '+', '.join(f"{row['sample']}: {row['class_0']}/{row['class_1']}" for row in protocol['class_balance'])+'.'
    text='Результаты исследовательского пилота 20k / 5 конфигураций\n\n'+'\n\n'.join(paragraphs)+\
        '\n\nФактический протокол\n'+protocol_text+'\n\nОграничения и отклонения от A2\n'+'\n'.join('- '+item for item in LIMITATIONS)+\
        '\n\nГотовность материалов\n'+('Материалы готовы для предварительного раздела статьи.' if not blockers else '\n'.join('- '+b for b in blockers))+\
        '\n\nCSV и LaTeX: tables/. Рисунки PDF/SVG: figures/. Полные рассчитанные результаты: CSV в evaluation/.\n'
    _atomic_text(output/'results_report.txt',text)
    css='body{font-family:Segoe UI,Arial,sans-serif;max-width:1180px;margin:38px auto;color:#172b3a;line-height:1.55;padding:0 24px}h1,h2{color:#123e5b}h1{font-size:28px}h2{margin-top:35px;font-size:21px}p{max-width:1050px}table{border-collapse:collapse;width:100%;font-size:12px;margin:14px 0 8px}th{background:#edf3f7;text-align:left}td,th{border-bottom:1px solid #d7e0e7;padding:7px;white-space:pre-line;vertical-align:top}caption,small{color:#536979}figure{margin:20px 0}img{max-width:100%;height:auto}.note{background:#f2f6f8;padding:12px 16px;border-left:4px solid #6089a4}a{color:#14699c}.scroll{overflow-x:auto}'
    parts=['<!doctype html><html lang="ru"><meta charset="utf-8"><title>Пилот: оценка RQ1/RQ2</title><style>'+css+'</style><body>',
        '<h1>Предварительные результаты RQ1/RQ2</h1>',
        '<p class="note">20 000 train-клиентов, 5 конфигураций, один refit на пару. Только оценка сохранённых вероятностей; новых обучений нет.</p>']
    for paragraph in paragraphs: parts.append('<p>'+html.escape(paragraph)+'</p>')
    parts.append('<h2>Таблицы для статьи</h2>')
    main_names=['rq1_level4','rq2_ap_levels','rq2_baseline_gains','rq2_sequential_gains',
                'profit_fairness_L1','profit_fairness_L5','profit_fairness_L10','errors_level4','ap_vs_lr_level4','delong_level4']
    for name in main_names:
        frame,caption,_=prepared_tables[name]
        parts.extend(['<h3>'+html.escape(caption)+'</h3>','<div class="scroll">'+frame.to_html(index=False,border=0,escape=True)+'</div>',
            f'<small><a href="tables/{name}.csv">CSV</a> · <a href="tables/{name}.tex">LaTeX</a></small>'])
    parts.append('<h2>Рисунки</h2>')
    for name,title in [('ap_by_level','PR-AUC (average precision, AP) по уровням'),('calibration_level4','Калибровка Level 4')]:
        if (output/'figures'/f'{name}.svg').exists():
            parts.append(f'<figure><img src="figures/{name}.svg" alt="{html.escape(title)}"><figcaption>{html.escape(title)}. <a href="figures/{name}.pdf">PDF</a></figcaption></figure>')
    parts.extend(['<h2>Фактический протокол</h2>','<p>'+html.escape(protocol_text)+'</p>',balance.to_html(index=False,border=0),cv.to_html(index=False,border=0),
        '<h3>MLP: сохранённые атрибуты обучения</h3>',mlp.drop(columns=['selected_params_json','fitted_config_json']).to_html(index=False,border=0),
        '<h2>Ограничения и отклонения от A2</h2>','<ul>'+''.join('<li>'+html.escape(item)+'</li>' for item in LIMITATIONS)+'</ul>',
        '<h2>Готовность</h2>','<p>'+html.escape('Материалы готовы для предварительного раздела статьи.' if not blockers else ' '.join(blockers))+'</p>',
        '<p>Полные сценарии, TPR-разрывы и ориентиры прибыли находятся в приложении tables/; все числовые результаты и промежуточные bootstrap-файлы сохранены отдельно.</p>',
        '</body></html>'])
    _atomic_text(output/'results_report.html','\n'.join(parts))
    readiness['readable_reports']=[str((output/'results_report.html').resolve()),str((output/'results_report.txt').resolve())]
    notebook_execution=_read_json(output/'notebook_execution_validation.json',{})
    if notebook_execution:
        readiness['notebook_execution']=notebook_execution
        if notebook_execution.get('backend')=='python_namespace':
            note=('Все code-ячейки notebook исполнены в чистом Python-процессе; реальные outputs сохранены. '
                'Попытка запуска Jupyter-ядра не вернула outputs и была прервана. '
                'Это подтверждает исполнение Python-кода, но не проверку notebook в Jupyter kernel. '
                'Готовые bootstrap-повторы переиспользованы; новых fit/predict не было.')
            _atomic_text(output/'results_report.txt',text+'\nИсполнение notebook\n'+note+'\n')
            report_html='\n'.join(parts).replace('</body></html>',
                '<h2>Исполнение notebook</h2><p>'+html.escape(note)+'</p></body></html>')
            _atomic_text(output/'results_report.html',report_html)
    _json(output/'results_readiness.json',readiness)
    return readiness
