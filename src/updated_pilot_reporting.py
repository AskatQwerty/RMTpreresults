"""Article tables for the active 28-Pipeline set after four MLP replacements.

Reuses proven table formatting only. Protocol, narrative and readiness are
derived from the new final-fit artifacts; no legacy five-configuration labels.
"""
from __future__ import annotations

from datetime import datetime, timezone
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .pilot_evaluation_reporting import (MODEL_LABELS, MODELS, REQUIRED_TABLES,
    article_tables, _save_table, _atomic_text, _json, _hash, _checked_pair_json)

SCOPE = 'pilot_20k_20config_mlp_updated'
LIMITATIONS = [
    'Обучение выполнено на прежних 20 000 development-заявках; это расширенный пилот, а не обучение на полном development из A2.',
    '24 Pipeline шести остальных типов сохранены из поиска 20 конфигураций на трёх CV-фолдах. Четыре MLP заменены моделями отдельного дополнительного эксперимента: четыре заранее заданных варианта × четыре уровня × три прежних фолда = 48 внешних CV-оценок.',
    'MLP выбирается по среднему внешнему CV log-loss; число эпох — по внутреннему validation log-loss на 15% текущего train. Затем новая предобработка и сеть обучаются на всём train. Итоговые MLP используют 20 000 клиентов, clipping числовых z-значений до ±5 и, на уровнях 2–4, alpha=0.1.',
    'Изменение MLP выполнено после просмотра прежнего test. Хотя новый вариант и число эпох выбраны без использования test, это последующий исследовательский анализ, а не независимое подтверждение первоначального протокола A2.',
    'На каждую активную пару приходится один итоговый Pipeline; пять train-повторов A2 не выполнены. Исходные 1680 CV-результатов, 28 моделей до замены и пререзультаты поиска пяти конфигураций сохранены отдельно.',
    '1000 общих парных bootstrap-повторов отражают состав test при фиксированных моделях, а не вариативность повторного обучения.',
    '95% percentile CI являются pointwise, unadjusted, exploratory; SD пяти train-повторов здесь нет.',
    'Bonferroni применяется к шести парным DeLong-сравнениям ROC-AUC с LR на Level 4. Это не проверка различий AP или эквивалентности.',
    'Test совпадает с ранее использованным в пререзультатах; это обновлённый исследовательский анализ, не новая внешняя проверка.',
    'Прибыль дана в условных единицах m=1 по фиксированной модели затрат. Пороги по test не подбирались, калибратор не обучался.',
    'Fairness-разрывы описательны; отсутствие разницы не доказывает отсутствие bias, а CI абсолютного разрыва не является отдельным тестом bias.',
    'Для партии CV 11–15 задокументирована замена исполнителя потоков Random Forest при прежних n_jobs=2; времена разных компьютеров напрямую не сравниваются.',
    'SHAP, пять train-повторов и формальное правило отбора A2 по неухудшению AP этим анализом не выполняются.',
]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def saved_final_protocol(root):
    root = Path(root)
    manifest = read(root / 'experiment_manifest.json')
    if manifest.get('scope') != SCOPE:
        raise ValueError('Wrong final training scope')
    cohort = read(root / 'cohort_summary.json')
    timings, mlp, warnings = [], [], []
    feature_counts, sources = {}, []
    for model in MODELS:
        for level in range(1, 5):
            stage = root / model / f'level_{level}'
            record = _checked_pair_json(stage, 'metrics.json')
            if record['scope'] != SCOPE or record['model'] != model or record['level'] != level:
                raise ValueError('Wrong final metrics identity')
            feature_counts.setdefault(str(level), set()).add(int(record['raw_features']))
            timings.append(dict(model=model, level=level,
                **{k: record.get(k) for k in ['refit_seconds','predict_seconds','total_pair_seconds']}))
            warnings.extend(dict(model=model, level=level, **w) for w in record.get('warnings', []))
            sources.append({'path': str((stage/'metrics.json').resolve()), 'sha256': _hash(stage/'metrics.json')})
            if model == 'mlp':
                name = record.get('fit_diagnostics_file', 'fit_checkpoint/fit_diagnostics.json')
                diagnostics = _checked_pair_json(stage, name)
                fitted, config = diagnostics['final_optimizer'], diagnostics['stopping_rule']
                if fitted['epochs_completed'] != diagnostics['best_epoch'] or diagnostics['final_refit_rows'] != 20000:
                    raise ValueError('Repaired MLP must use selected epochs and all 20000 train rows')
                mlp.append(dict(model=model, level=level, selected_variant=diagnostics['variant']['name'],
                    epochs_completed=fitted['epochs_completed'], raw_n_iter_=fitted['raw_n_iter'],
                    max_epochs=config['max_epochs'], min_delta=config['min_delta'], patience=config['patience'],
                    alpha=diagnostics['variant']['alpha'], clip_z=diagnostics['variant']['clip_z'],
                    loss_curve_points=fitted['loss_curve_length'],
                    final_training_loss=fitted['final_training_objective'], train_log_loss=fitted['train_log_loss'],
                    best_cv_log_loss=record['best_cv_log_loss'], test_log_loss=record['metrics']['log_loss'],
                    test_ap=record['metrics']['pr_auc_average_precision'],
                    warning_count_final_fit=sum(w.get('count', 1) for w in record.get('warnings', [])),
                    selected_params_json=json.dumps(record['selected_params'], sort_keys=True),
                    fitted_config_json=json.dumps(config, sort_keys=True)))
    if len(mlp) != 4 or any(len(v) != 1 for v in feature_counts.values()):
        raise ValueError('Missing MLP actual diagnostics or inconsistent feature counts')
    summary_path = root / 'experiment_summary.json'
    summary = read(summary_path) if summary_path.exists() else {}
    protocol = dict(scope=SCOPE, train_rows=manifest['train_rows'], test_rows=manifest['test_rows'],
        development_rows=manifest.get('development_rows'), original_configurations_per_pair=20,
        original_completed_candidate_cv_fits=1680, original_models_retained=24,
        mlp_replacement_count=4, supplemental_mlp_outer_cv_evaluations=48,
        supplemental_mlp_variants_per_level=4, active_pipeline_count=28, final_fit_seed=42,
        repeats_per_pair=1, selection_metric='mean CV log_loss', selected_without_test=True,
        test_previously_seen=True, mlp_repair_is_posthoc=True, model_lineage=manifest['model_lineage'],
        cohort_summary=cohort, cv_folds=cohort.get('cv'),
        feature_counts={k: next(iter(v)) for k,v in feature_counts.items()},
        software_versions=manifest.get('versions'), python=manifest.get('python'),
        threads=manifest.get('threads'), hardware=manifest.get('hardware'),
        training_summary=summary, refit_seconds_sum=sum(float(r['refit_seconds']) for r in timings),
        predict_seconds_sum=sum(float(r['predict_seconds']) for r in timings),
        timing_note='Historical timings attached to active Pipelines, not the elapsed time of promotion. Each repaired MLP procedure includes internal epoch selection and fresh full-train refit. No fit or predict occurs during this evaluation.',
        sources=sources, original_preresults_preserved=True, original_20config_results_preserved=True)
    # root is artifacts/training/<scope>; resolve the project explicitly.
    audit_path = root.parents[2] / 'reports/training/pilot_20k_20config_combined/mlp_audit.json'
    if audit_path.exists():
        protocol['mlp_audit_before_final'] = {'path': str(audit_path.resolve()), 'sha256': _hash(audit_path),
            'status': read(audit_path)['status'], 'interpretation_ru': read(audit_path)['interpretation_ru']}
    return protocol, pd.DataFrame(timings), pd.DataFrame(mlp), pd.DataFrame(warnings)


def figures(tables, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                         'axes.spines.top':False, 'axes.spines.right':False, 'svg.fonttype':'none'})
    directory = Path(output) / 'figures'
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    def save(fig, name):
        for ext in ('png', 'svg'):
            path = directory / f'{name}.{ext}'
            fig.savefig(path, dpi=160, bbox_inches='tight')
            files.append(str(path.resolve()))
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 6))
    for model in MODELS:
        rows = tables['quality_ci'].query('model==@model and metric=="ap"').sort_values('level')
        line, = ax.plot(rows.level, rows.estimate, marker='o', label=MODEL_LABELS[model])
        ax.vlines(rows.level, rows.ci_low, rows.ci_high, color=line.get_color(), linewidth=1)
    ax.set(xticks=[1,2,3,4], xlabel='Information level', ylabel='PR-AUC (average precision)',
           title='Active 20k pilot after MLP replacement: AP by information level')
    ax.grid(axis='y', alpha=.2)
    ax.legend(loc='upper center', bbox_to_anchor=(.5,-.13), ncol=4, frameon=False, fontsize=9)
    fig.text(.5,.02,'95% paired percentile bootstrap CI; fixed trained models; 20,000 train / 61,503 test.', ha='center', fontsize=9)
    fig.subplots_adjust(bottom=.24)
    save(fig, 'ap_by_level')
    fig, axes = plt.subplots(2,4,figsize=(13,7), sharex=True, sharey=True)
    for model, ax in zip(MODELS, axes.flat):
        rows = tables['calibration_bins'].query('model==@model and n>0').sort_values('bin')
        ax.plot([0,1],[0,1], '--', color='#909090', linewidth=.8)
        ax.plot(rows.mean_probability_default, rows.observed_default_rate, color='#1769aa', linewidth=1.1)
        ax.scatter(rows.mean_probability_default, rows.observed_default_rate,
                   s=12+100*np.sqrt(rows.n/rows.n.sum()), color='#1769aa', alpha=.85)
        ax.set(title=MODEL_LABELS[model], xlim=(-.02,1.02), ylim=(-.02,1.02), xticks=[0,.5,1], yticks=[0,.5,1])
        ax.grid(alpha=.2)
    axes.flat[-1].axis('off')
    axes.flat[-1].text(.02,.9, 'Level 4: active models\nMLP replaced; six others retained\n\n10 fixed equal-width bins.\nEmpty bins omitted.\nMarker size reflects bin count.\n\nNo calibrator was fitted.\nSmall bins are uncertain.\nExact counts in CSV.',
                       transform=axes.flat[-1].transAxes, va='top', fontsize=10)
    fig.supxlabel('Mean predicted default probability')
    fig.supylabel('Observed default rate')
    fig.suptitle('Calibration of active Level 4 models after MLP replacement', fontsize=14)
    fig.subplots_adjust(top=.9,bottom=.1,left=.07,hspace=.3,wspace=.2)
    save(fig,'calibration_level4')
    fig, axes = plt.subplots(1,2,figsize=(12,5.5))
    for ax, metric, label in zip(axes, ['ap','roc_auc'], ['Average precision','ROC-AUC']):
        rows = tables['quality_ci'].query('level==4 and metric==@metric').sort_values('estimate')
        x = np.arange(len(rows))
        ax.hlines(x, rows.ci_low, rows.ci_high, color='#1769aa')
        ax.scatter(rows.estimate, x, color='#1769aa')
        ax.set(yticks=x, yticklabels=rows.model.map(MODEL_LABELS), xlabel=label)
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('Level 4: quality and 95% paired percentile bootstrap intervals')
    fig.tight_layout()
    save(fig,'rq1_quality_level4')
    return files


def export_final_report(pilot_root, evaluation_dir, *, previous_reports, create_figures=True):
    root, output, previous = map(Path, (pilot_root,evaluation_dir,previous_reports))
    if output.resolve().is_relative_to(previous.resolve()) or output.resolve().is_relative_to(root.resolve()):
        raise ValueError('Reports must remain separate from old preresults and training')
    tables = {name: pd.read_csv(output/f'{name}.csv') for name in REQUIRED_TABLES}
    protocol, timing, mlp, warnings = saved_final_protocol(root)
    _json(output/'actual_training_protocol.json', protocol)
    for name, frame in [('training_timings',timing), ('mlp_actual_diagnostics',mlp), ('saved_training_warnings',warnings)]:
        _atomic_text(output/f'{name}.csv', frame.to_csv(index=False))
    prepared = article_tables(tables)
    d = tables['delong']
    prepared['delong_level4'] = (pd.DataFrame({'Model':d.model.map(MODEL_LABELS),
        'Delta ROC-AUC':d.delta_roc_auc.map(lambda v:f'{v:+.5f}'),
        'p (raw)':d.p_raw.map(lambda v:f'{v:.5g}'), 'p (Bonferroni)':d.p_bonferroni.map(lambda v:f'{v:.5g}'),
        'Reject at 0.05':d.reject_bonferroni_0_05.map({True:'Yes',False:'No'})}),
        'Paired DeLong on Level 4 versus LR; six Bonferroni-adjusted ROC-AUC comparisons.',False)
    table_files = {name:_save_table(output,name,frame,caption,long=long)
                   for name,(frame,caption,long) in prepared.items()}
    if create_figures:
        figures(tables,output)
    figure_files = [str(p.resolve()) for p in sorted((output/'figures').glob('*')) if p.suffix in ('.png','.svg')]
    summary = read(output/'evaluation_summary.json')
    integrity = read(output/'integrity_after.json')
    l4 = tables['quality_point'].query('level==4')
    best_ap = l4.loc[l4.ap.idxmax()]
    best_loss = l4.loc[l4.log_loss.idxmin()]
    paragraphs = [
        'Актуальный набор содержит 28 Pipeline: 24 прежние модели и четыре новых MLP, заменивших неудачные исходные версии. '
        'Все обучены на тех же 20 000 клиентах и оценены на прежних 61 503 test-клиентах. '
        'После замены MLP пересчитаны RQ1/RQ2, прибыль, fairness, калибровка и парные сравнения. '
        'В этом этапе новых fit или predict не выполнялось: использованы сохранённые вероятности.',
        f'RQ1: максимальное точечное AP на Level 4 — {MODEL_LABELS[best_ap.model]} ({best_ap.ap:.6f}); '
        f'минимальный log-loss — {MODEL_LABELS[best_loss.model]} ({best_loss.log_loss:.6f}). '
        'Это описательные тестовые результаты, не доказательство превосходства по всем метрикам.',
    ]
    rejected = d[d.reject_bonferroni_0_05 == True]
    paragraphs.append('DeLong, ROC-AUC против LR, p Bonferroni <0.05: '+
        ('; '.join(f'{MODEL_LABELS[r.model]}: Δ={r.delta_roc_auc:+.6f}, p_adj={r.p_bonferroni:.5g}' for r in rejected.itertuples()) or 'нет')+
        '. Семейство включает шесть сравнений; AP этим тестом не проверяется.')
    gains = tables['rq2_deltas'].query('kind=="baseline" and to_level==4')
    paragraphs.append('RQ2, ΔAP Level 4 − Level 1 [95% CI]: '+ '; '.join(
        f'{MODEL_LABELS[r.model]} {r.delta_ap:+.5f} [{r.ci_low:+.5f}, {r.ci_high:+.5f}]' for r in gains.itertuples())+'.')
    paragraphs.append('Новые MLP, фактическое итоговое обучение: '+ '; '.join(
        f'L{r.level}: {r.epochs_completed} эпох, alpha={r.alpha:g}, clipping ±{r.clip_z:g}, '
        f'CV log-loss={r.best_cv_log_loss:.5f}, test log-loss={r.test_log_loss:.5f}' for r in mlp.itertuples())+
        '. Число эпох выбиралось внутри train по log-loss: максимум 120, patience=12, min_delta=0.0001. '
        'Счётчик sklearn n_iter_ при partial_fit равен 1; фактические эпохи подтверждены loss_curve_ и числом просмотренных обучающих строк.')
    paragraphs.append('Дополнительный поиск MLP включал 48 внешних CV-оценок; все четыре решения были зафиксированы до нового обращения к test. '
        'Предобработка внутренней проверки обучалась только на внутреннем train, затем новая сеть и предобработка обучались на всех 20 000 клиентах. '
        'Улучшение после изменения остановки, ограничения экстремальных z-значений и регуляризации согласуется с уменьшением переобучения, '
        'но не доказывает единственную причину прежней неудачи. Test уже просматривался, поэтому замена явно обозначена как последующий исследовательский анализ.')
    paragraphs.append('Исходные 28 моделей до замены и отчёты сохранены в pilot_20k_20config_final; пререзультаты пяти конфигураций также сохранены. '
        'comparison_to_preserved_20config_before_mlp.csv показывает изменения метрик: у всех 24 Pipeline остальных типов изменения равны нулю. '
        'Новые 1000 общих парных bootstrap-повторов используют тот же seed=42 и обновлённые вероятности, поэтому интервалы прибыли и fairness согласованы с активными MLP.')
    main_ci = tables['quality_ci'][(tables['quality_ci'].metric=='ap') |
        ((tables['quality_ci'].level==4)&tables['quality_ci'].metric.isin(['roc_auc','log_loss','brier']))]
    blockers = []
    if summary.get('completed_bootstrap') != 1000 or len(main_ci)!=49 or not main_ci.bootstrap_n_valid.eq(1000).all():
        blockers.append('Не подтверждены все 1000 повторов для основных интервалов.')
    if integrity.get('unchanged') is not True or integrity.get('old_preresults_unchanged') is not True:
        blockers.append('Не подтверждена сохранность исходных файлов и старых пререзультатов.')
    graphics_path = output/'graphics_validation.json'
    graphics = read(graphics_path) if graphics_path.exists() else {}
    if graphics.get('passed') is not True:
        blockers.append('Визуальная проверка новых рисунков ещё не зафиксирована.')
    if len(figure_files)!=6:
        blockers.append('Не готовы три пары рисунков PNG/SVG.')
    readiness = dict(scope=SCOPE, status='ready' if not blockers else 'pending_checks',
        generated_utc=datetime.now(timezone.utc).isoformat(), blockers=blockers,
        original_completed_cv_fits=1680, supplemental_mlp_outer_cv_evaluations=48,
        active_pipeline_count=28, original_models_retained=24, mlp_replacement_count=4,
        mlp_repair_is_posthoc=True, test_rows=61503, completed_bootstrap=1000,
        new_fit_calls_in_evaluation=0, new_predict_calls_in_evaluation=0,
        old_preresults_preserved=integrity.get('old_preresults_unchanged'), limitations=LIMITATIONS,
        table_files=table_files, figure_files=figure_files, graphics_validation=graphics,
        original_reports=str(previous.resolve()), metrics_are_new_final_predictions=True)
    _json(output/'results_readiness.json',readiness)
    _atomic_text(output/'latex_usage.txt',
        'Пакеты: booktabs, longtable, graphicx. Пример: \\input{tables/rq1_level4.tex}.\n'
        'В таблице точечная оценка и затем [95% CI]. Интервалы описывают фиксированные модели, не пять train-повторов.\n'
        'Фрагменты LaTeX не компилировались; итоговую верстку проверяйте в шаблоне статьи.\n')
    report_text = 'RQ1/RQ2, прибыль и fairness: актуальный пилот 20k после замены MLP\n\n'+'\n\n'.join(paragraphs)+\
        '\n\nОграничения\n'+'\n'.join('- '+x for x in LIMITATIONS)+\
        '\n\nГотовность\n'+('Числовые результаты и рисунки проверены.' if not blockers else '\n'.join(blockers))+'\n'
    _atomic_text(output/'results_report.txt',report_text)
    css='body{font:16px/1.55 Segoe UI,Arial,sans-serif;max-width:1180px;margin:35px auto;padding:0 22px;color:#173248}h1,h2{color:#123e5b}table{border-collapse:collapse;font-size:12px;width:100%}th,td{padding:7px;border-bottom:1px solid #d9e2e8;text-align:left;white-space:pre-line}th{background:#eef4f8}.scroll{overflow:auto}img{max-width:100%}small{color:#536979}'
    parts=['<!doctype html><html lang="ru"><meta charset="utf-8"><title>RMT: итоговые RQ1/RQ2</title><style>'+css+'</style><body>',
        '<h1>RQ1/RQ2, прибыль и fairness: пилот 20k после замены MLP</h1>']
    parts.extend('<p>'+html.escape(p)+'</p>' for p in paragraphs)
    for name in ['rq1_level4','rq2_ap_levels','rq2_baseline_gains','rq2_sequential_gains','ap_vs_lr_level4',
                 'delong_level4','profit_fairness_L1','profit_fairness_L5','profit_fairness_L10','errors_level4']:
        frame, caption, _ = prepared[name]
        parts.extend(['<h2>'+html.escape(caption)+'</h2>', '<div class="scroll">'+frame.to_html(index=False,border=0)+'</div>',
                      f'<small><a href="tables/{name}.csv">CSV</a> · <a href="tables/{name}.tex">LaTeX</a></small>'])
    for name in ('rq1_quality_level4','ap_by_level','calibration_level4'):
        parts.append(f'<figure><img src="figures/{name}.png" alt="{name}"><figcaption><a href="figures/{name}.svg">SVG</a></figcaption></figure>')
    parts.extend(['<h2>Фактическое обучение MLP</h2>',mlp.drop(columns=['selected_params_json','fitted_config_json']).to_html(index=False,border=0),
        '<h2>Ограничения</h2><ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in LIMITATIONS)+'</ul>',
        '<h2>Готовность</h2><p>'+html.escape('Проверено.' if not blockers else ' '.join(blockers))+'</p></body></html>'])
    _atomic_text(output/'results_report.html','\n'.join(parts))
    return readiness
