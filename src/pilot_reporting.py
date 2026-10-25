"""Отчёты исследовательского пилота из проверенных файлов, без fit и выбора по test.

export_pilot_reports вызывается после каждого complete.json. Таблицы отражают
только завершённые пары с верными SHA-256; остальные 28 строк остаются pending.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import time

import pandas as pd


SCOPE='pilot_20k_5config'
MODEL_LABELS={
    'logistic_regression':'Logistic Regression (L2)', 'random_forest':'Random Forest',
    'xgboost':'XGBoost', 'lightgbm':'LightGBM', 'mlp':'MLP',
    'soft_voting':'Soft voting', 'stacking':'Stacking',
}
QUALITY=('pr_auc_average_precision','roc_auc','log_loss','brier_score')
CLASSIFICATION=('rows','positives','positive_rate',*QUALITY,'threshold','accuracy',
                'precision','recall','f1','tn','fp','fn','tp','approval_rate_at_0_5')
TIMES=('search_and_refit_seconds','refit_seconds','predict_seconds','total_pair_seconds')
METRIC_COLUMNS=('model','model_label','level','train_rows','test_rows','raw_features',
                'selected_params_json','best_cv_log_loss',*TIMES,'warning_count',
                *[f'test_{name}' for name in CLASSIFICATION if name!='rows'])
STATUS_COLUMNS=('model','model_label','level','status','reason')
WARNING_COLUMNS=('model','level','stage','category','message','count')
MLP_COLUMNS=('model','level','estimator_path','n_iter_','max_iter','tol','n_iter_no_change',
             'loss_','loss_curve_points','loss_curve_length_matches_n_iter','source')


def _read(path):
    def invalid(value): raise ValueError(f'Недопустимое число JSON: {value}')
    return json.loads(Path(path).read_text(encoding='utf-8'),parse_constant=invalid)


def _atomic_text(path,text):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    # pandas.to_csv(None) уже возвращает CRLF на Windows: повторный перевод LF
    # через TextIO создал бы CRCRLF и пустые строки. Краткие Windows file locks
    # повторяем пять раз (1+2+4+8+16=31 s), затем PermissionError остаётся видимой.
    for attempt in range(6):
        try:
            temporary.write_text(text,encoding='utf-8',newline='')
            os.replace(temporary,path)
            return
        except PermissionError:
            if attempt==5: raise
            time.sleep(2**attempt)


def _write_json(path,value):
    _atomic_text(path,json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))


def _sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''): digest.update(block)
    return digest.hexdigest()


def _validated_pair(stage,model,level,fingerprint):
    """Не читать metrics/diagnostics до проверки всех файлов complete marker."""
    marker=_read(stage/'complete.json')
    if not fingerprint or marker.get('fingerprint')!=fingerprint:
        raise ValueError('fingerprint complete.json не совпадает с experiment_manifest.json')
    hashes=marker.get('artifact_sha256',{})
    if not isinstance(hashes,dict) or 'metrics.json' not in hashes:
        raise ValueError('В complete.json отсутствует SHA-256 metrics.json')
    for name,expected in hashes.items():
        file=(stage/name).resolve()
        if not file.is_relative_to(stage.resolve()):
            raise ValueError(f'Путь артефакта выходит из каталога пары: {name}')
        if not file.is_file() or _sha256(file)!=expected:
            raise ValueError(f'Не найден артефакт или не совпадает SHA-256: {name}')
    record=_read(stage/'metrics.json')
    if record.get('scope')!=SCOPE or record.get('model')!=model or record.get('level')!=level:
        raise ValueError('scope/model/level в metrics.json не соответствуют каталогу пилота')
    for field in ('train_rows','test_rows','raw_features','selected_params','best_cv_log_loss',*TIMES,'warnings','metrics'):
        if field not in record: raise ValueError(f'Нет обязательного поля metrics.json: {field}')
    for name in QUALITY:
        value=record['metrics'].get(name)
        if not isinstance(value,Real) or not math.isfinite(value):
            raise ValueError(f'Нет конечной числовой test-метрики: {name}')
    if record['metrics'].get('rows')!=record['test_rows']:
        raise ValueError('test_rows не совпадает с metrics.rows')
    diagnostic_file=record.get('fit_diagnostics_file','fit_diagnostics.json')
    if not isinstance(diagnostic_file,str): raise ValueError('fit_diagnostics_file должен быть строкой')
    diagnostic=_read(stage/diagnostic_file) if diagnostic_file in hashes else None
    return record,diagnostic


def _latex_escape(value):
    replacements={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$',
                  '#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    return ''.join(replacements.get(char,char) for char in str(value))


def _latex(frame,columns,caption,label):
    """Статическая booktabs-таблица; все текстовые значения экранируются."""
    lines=[r'\begin{table}[htbp]',r'\centering',r'\small',
        r'\caption{'+_latex_escape(caption)+'}',r'\label{'+_latex_escape(label)+'}',
        r'\begin{tabular}{'+('l'+'r'*(len(columns)-1))+'}',r'\toprule',
        ' & '.join(_latex_escape(title) for _,title in columns)+r' \\',r'\midrule']
    for _,row in frame.iterrows():
        values=[]
        for name,_ in columns:
            value=row.get(name)
            if value is None or (not isinstance(value,(list,dict)) and pd.isna(value)): text='--'
            elif isinstance(value,Real) and not isinstance(value,bool):
                text=str(int(value)) if name in ('level','train_rows','test_rows','raw_features') else f'{value:.4f}'
            else: text=_latex_escape(value)
            values.append(text)
        lines.append(' & '.join(values)+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}',''])
    return '\n'.join(lines)


def _decision_reports(root,report_dir,records):
    """Каждый checkpoint повторно пытается импортировать описательную постобработку."""
    probability_sources=[]
    for model in MODEL_LABELS:
        if (model,4) not in records: continue
        stage=root/model/'level_4'
        if 'probabilities.parquet' in _read(stage/'complete.json').get('artifact_sha256',{}):
            probability_sources.append(model)
    if not probability_sources:
        return {'status':'not_ready','reason':'Нет завершённых Level 4 с проверенным probabilities.parquet.',
                'additional_fit_calls':0,'test_threshold_optimization':False,'files':{}}
    try:
        from .pilot_decision_metrics import export_decision_metrics
    except ModuleNotFoundError as exc:
        if exc.name!=f'{__package__}.pilot_decision_metrics': raise
        return {'status':'pending_module','reason':'Модуль описательной постобработки ещё не доступен; импорт повторится при следующем checkpoint.',
                'additional_fit_calls':0,'test_threshold_optimization':False,'files':{}}
    return export_decision_metrics(root,report_dir)


def export_pilot_reports(root,report_dir):
    """Перезаписать CSV/LaTeX из hash-valid пар root/{model}/level_{1..4}.

    Возвращает JSON-safe summary с completed_pairs/pending_pairs/invalid_pairs,
    status и абсолютными путями files. Никакие исходные артефакты не изменяются.
    """
    root=Path(root); report_dir=Path(report_dir)
    manifest_path=root/'experiment_manifest.json'
    manifest=_read(manifest_path) if manifest_path.exists() else None
    if manifest is not None and manifest.get('mode')!=SCOPE:
        raise ValueError(f'Ожидался experiment_manifest.mode={SCOPE}')
    fingerprint=manifest.get('fingerprint') if manifest else None
    records={}; statuses=[]; warning_rows=[]; mlp_rows=[]; mlp_details=[]; metric_rows=[]
    for model,label in MODEL_LABELS.items():
        for level in range(1,5):
            stage=root/model/f'level_{level}'
            status={'model':model,'model_label':label,'level':level,'status':'pending','reason':'Нет complete.json'}
            if (stage/'complete.json').exists():
                try:
                    record,diagnostic=_validated_pair(stage,model,level,fingerprint)
                except (OSError,ValueError,KeyError,TypeError) as exc:
                    status.update(status='invalid',reason=str(exc))
                else:
                    status.update(status='completed',reason='Все SHA-256 complete.json проверены')
                    records[(model,level)]=record
                    row={name:record.get(name) for name in ('model','level','train_rows','test_rows','raw_features','best_cv_log_loss',*TIMES)}
                    row.update(model_label=label,selected_params_json=json.dumps(record['selected_params'],ensure_ascii=False,sort_keys=True),
                        warning_count=sum(w.get('count',1) for w in record['warnings']))
                    row.update({f'test_{name}':record['metrics'].get(name) for name in CLASSIFICATION if name!='rows'})
                    metric_rows.append(row)
                    for warning in record['warnings']:
                        warning_rows.append({'model':model,'level':level,'stage':warning.get('stage','search_and_refit'),
                            'category':warning.get('category'),'message':warning.get('message'),'count':warning.get('count',1)})
                    for item in (diagnostic or {}).get('mlp',[]):
                        fitted=item.get('fitted',{}); config=item.get('config',{}); curve=fitted.get('loss_curve_')
                        mlp_rows.append({'model':model,'level':level,'estimator_path':item.get('path'),
                            'n_iter_':fitted.get('n_iter_'),'max_iter':config.get('max_iter'),'tol':config.get('tol'),
                            'n_iter_no_change':config.get('n_iter_no_change'),'loss_':fitted.get('loss_'),
                            'loss_curve_points':len(curve) if curve is not None else None,
                            'loss_curve_length_matches_n_iter':len(curve)==fitted.get('n_iter_') if curve is not None else None,
                            'source':str((stage/record.get('fit_diagnostics_file','fit_diagnostics.json')).resolve())})
                        mlp_details.append({'model':model,'level':level,'diagnostic':item})
            statuses.append(status)
    aligned=[]; history_rows=[]
    for state in statuses:
        model,level=state['model'],state['level']
        current=records.get((model,level)); baseline=records.get((model,1))
        values={f'test_{name}':current['metrics'][name] if current else None for name in QUALITY}
        aligned.append({**state,**values,'train_rows':current.get('train_rows') if current else None,
            'test_rows':current.get('test_rows') if current else None,'raw_features':current.get('raw_features') if current else None})
        history_rows.append({**state,**values,'level1_available':baseline is not None,
            **{f'delta_{name}_vs_level1':current['metrics'][name]-baseline['metrics'][name]
               if current and baseline else None for name in QUALITY}})
    metrics=pd.DataFrame(metric_rows,columns=METRIC_COLUMNS)
    status_frame=pd.DataFrame(statuses,columns=STATUS_COLUMNS)
    history_frame=pd.DataFrame(history_rows)
    comparison_frame=pd.DataFrame(aligned).sort_values('level',kind='stable').reset_index(drop=True)
    files={}
    frames={'metrics':metrics,'pair_status':status_frame,'rq1_model_comparison':comparison_frame,
            'rq2_history_gains':history_frame,'warnings':pd.DataFrame(warning_rows,columns=WARNING_COLUMNS),
            'mlp_diagnostics':pd.DataFrame(mlp_rows,columns=MLP_COLUMNS)}
    for name,frame in frames.items():
        path=report_dir/f'{name}.csv'; _atomic_text(path,frame.to_csv(index=False)); files[f'{name}_csv']=str(path.resolve())
    primary_columns=[('model_label','Model'),('level','Level'),('status','Status'),
        ('test_pr_auc_average_precision','PR-AUC (AP)'),('test_roc_auc','ROC-AUC'),
        ('test_log_loss','Log-loss'),('test_brier_score','Brier')]
    for name,frame,columns,caption in [
        ('metrics',comparison_frame,primary_columns,'Исследовательский пилот: test-метрики одной выбранной через CV конфигурации на пару.'),
        ('rq2_history_gains',history_frame,[('model_label','Model'),('level','Level'),('status','Status'),
            ('delta_pr_auc_average_precision_vs_level1','Delta AP'),('delta_roc_auc_vs_level1','Delta ROC-AUC'),
            ('delta_log_loss_vs_level1','Delta log-loss'),('delta_brier_score_vs_level1','Delta Brier')],
            'RQ2: прирост от истории относительно Level 1 той же модели; разность current minus Level 1.'),
        ('rq1_model_comparison',comparison_frame,primary_columns,'RQ1: предварительное сравнение predictive accuracy и качества вероятностей; окончательная core-модель не выбирается.')]:
        path=report_dir/f'{name}.tex'; _atomic_text(path,_latex(frame,columns,caption,f'tab:pilot-{name.replace("_","-")}'))
        files[f'{name}_tex']=str(path.resolve())
    diagnostic_path=report_dir/'mlp_diagnostics.json'
    _write_json(diagnostic_path,mlp_details); files['mlp_diagnostics_json']=str(diagnostic_path.resolve())
    decision_reports=_decision_reports(root,report_dir,records)
    for name,path in decision_reports.get('files',{}).items(): files[f'decision_metrics_{name}']=path
    completed=len(records); invalid=sum(s['status']=='invalid' for s in statuses)
    summary={'scope':SCOPE,'root':str(root.resolve()),'report_dir':str(report_dir.resolve()),
        'generated_utc':datetime.now(timezone.utc).isoformat(),'expected_pairs':28,'completed_pairs':completed,
        'pending_pairs':28-completed-invalid,'invalid_pairs':invalid,
        'status':'invalid_artifacts' if invalid else 'completed' if completed==28 else 'partial' if completed else 'not_started',
        'source_fingerprint':fingerprint,'verified_stage_count':completed,'mlp_diagnostic_rows':len(mlp_rows),
        'warning_count':sum(w['count'] for w in warning_rows),'files':files,'decision_metrics':decision_reports,
        'plan':{'development_sample_rows':20000,'stratified_sample_seed':42,'models':7,'levels':4,
            'randomized_searches':28,'configurations_per_search':5,'cv_folds':3,'candidate_cv_fits':420,
            'refits':28,'repeat_trainings':0,'ensemble_internal_fits':'учитываются отдельно'},
        'deviations_from_a2':[
            'Это отдельный исследовательский пилот на 20 000 development-заявках, стратифицированных с seed=42.',
            'Для каждой пары выполняются 5 конфигураций × 3 CV-фолда и один автоматический refit; основной план A2 20 конфигураций и 5 повторов этим не заменяется.',
            'Используются полные рабочие настройки моделей (technical=False). SHAP в пилоте не вычисляется.'],
        'interpretation':[
            'Выбор параметров проводится по CV log-loss. Test используется для итоговой оценки, без изменения параметров, правил обработки или выбора конфигураций по качеству test.',
            'RQ1 в Assignment 2 рассматривает core-модель при совместном учёте accuracy, profit и fairness. Таблица rq1_model_comparison даёт только предварительное описательное сравнение predictive accuracy и качества вероятностей; окончательная core-модель по пилоту не выбирается.',
            'RQ2 показывает прирост от исторической информации: разности Level 2/3/4 − Level 1 внутри одной модели. Положительная разность AP/ROC-AUC и отрицательная разность log-loss/Brier означают улучшение на данном test; Level 1 имеет нулевую разность как baseline.',
            'Порядок моделей фиксирован; победитель, eligibility и пороги по test не выбираются.',
            'Один refit на пару не даёт оценки вариабельности по повторам: доверительные интервалы, p-values и утверждения о статистической значимости не формируются.',
            'Pending/invalid не имеют test-результатов в отчёте; отсутствующие метрики и baseline Level 1 остаются пустыми, а в LaTeX обозначены --.',
            'search_and_refit_seconds включает CV и автоматический refit; refit_seconds — его часть, поэтому эти столбцы нельзя суммировать. total_pair_seconds включает загрузку test, оценку и сохранение.',
            'Полные loss_curve_ и n_iter_ MLP читаются только из проверенного fit_diagnostics.json; CSV содержит компактную сводку, JSON — исходные атрибуты.',
            'LaTeX использует UTF-8 и booktabs: в документе необходим пакет booktabs и поддержка русского текста.'],
        'test_used_for_tuning':False,'statistical_significance_claimed':False,'shap_computed':False}
    metadata_path=report_dir/'report_metadata.json'; files['metadata_json']=str(metadata_path.resolve())
    _write_json(metadata_path,summary)
    return summary
