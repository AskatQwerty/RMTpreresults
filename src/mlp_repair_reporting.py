"""Supplementary MLP repair report from saved CV and final-fit artifacts only.

No estimator, prepared feature matrix or predictions are loaded here. The
original main experiment and preliminary reports are never modified.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCOPE = 'mlp_repair_20k_v1'
VARIANTS = ('stop_only', 'stop_regularized', 'stop_clipped', 'stop_clipped_regularized')
LIMITATIONS = [
    'Это отдельный последующий исследовательский эксперимент MLP. Основные 28 Pipeline, 1680 CV-результатов и прежние RQ1/RQ2 не заменяются.',
    'Использованы прежние 20 000 клиентов с естественным балансом классов и сохранённые внешние CV-фолды. Test ранее уже просматривался; он не является новой независимой проверкой.',
    'Вариант выбирается по среднему log-loss внешней CV. Внутри каждого train-фолда отдельные 15% используются только для выбора числа эпох по validation log-loss.',
    'Предобработка внутреннего этапа обучается только на его train. Затем новая предобработка и новая сеть обучаются на всём предоставленном train ровно выбранное число эпох.',
    'Ограничение z-значений до ±5 применяется только к масштабированным числовым значениям. Это отдельный вариант преобразования, а не удаление клиентов или изменение сохранённых X.',
    'Здесь показаны точечные оценки. Bootstrap, доверительные интервалы и проверки значимости различий не вычислялись; стандартное отклонение трёх CV-фолдов не является CI.',
    'В sklearn partial_fit значение n_iter_ относится к последнему вызову. Реальное число эпох записывается отдельно по циклу обучения, loss_curve_ и t_.',
    'Изменение правила остановки и исполнительного цикла partial_fit ограничивает строго причинное толкование сравнения с исходным MLP. Улучшение на уже использованном test остаётся исследовательским результатом.',
]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def checked_json(stage, name):
    """Respect checkpoint SHA when a marker is present; never load a model."""
    stage = Path(stage)
    path = (stage / name).resolve()
    require(path.is_relative_to(stage.resolve()), 'Report artifact escapes its stage')
    marker = stage / 'complete.json'
    if marker.exists():
        checksums = read(marker).get('artifact_sha256', {})
        require(name in checksums and digest(path) == checksums[name], f'Artifact checksum mismatch: {path}')
    return read(path)


def _metric(values, name):
    aliases = {'pr_auc_average_precision': ('pr_auc_average_precision', 'ap'),
               'brier_score': ('brier_score', 'brier')}
    for key in aliases.get(name, (name,)):
        if key in values:
            result = float(values[key])
            require(np.isfinite(result), 'Nonfinite saved metric: ' + name)
            return result
    raise KeyError(name)


def _curve(diagnostics):
    """Accept the estimator's explicit epoch history, not sklearn raw n_iter_."""
    inner = diagnostics.get('inner_selection', diagnostics.get('selection', {}))
    history = diagnostics.get('epoch_curve', inner.get('history', inner.get('curves', diagnostics.get('history', []))))
    if isinstance(history, dict):
        history = pd.DataFrame(history).to_dict('records')
    frame = pd.DataFrame(history)
    aliases = {'inner_train_log_loss': 'train_log_loss', 'val_log_loss': 'validation_log_loss',
               'inner_validation_log_loss': 'validation_log_loss'}
    frame = frame.rename(columns={k: v for k, v in aliases.items() if k in frame and v not in frame})
    require({'epoch', 'train_log_loss', 'validation_log_loss'} <= set(frame), 'Missing explicit epoch curves')
    require(frame.epoch.tolist() == list(range(1, len(frame) + 1)), 'Epoch curve order is invalid')
    require(np.isfinite(frame[['train_log_loss', 'validation_log_loss']].to_numpy()).all(), 'Nonfinite epoch curve')
    best = diagnostics.get('best_epoch', diagnostics.get('selected_epoch',
        inner.get('best_epoch', inner.get('selected_epoch'))))
    require(best is not None and 1 <= int(best) <= len(frame), 'Missing selected epoch')
    return frame, int(best)


def load_tables(root):
    root = Path(root)
    plan = read(root / 'plan.json')
    cv = pd.read_csv(root / 'cv_results.csv')
    candidates = pd.read_csv(root / 'candidate_summary.csv')
    selection = read(root / 'selection.json')
    entries = selection['selections']
    expected = {(level, variant, fold) for level in range(1, 5) for variant in VARIANTS for fold in range(3)}
    keys = list(zip(cv.level.astype(int), cv.variant, cv.fold.astype(int)))
    require(len(keys) == 48 and len(set(keys)) == 48 and set(keys) == expected, 'Expected all48 unique CV evaluations')
    require(np.isfinite(cv.log_loss).all(), 'Nonfinite CV loss')
    require(len(entries) == 4 and {int(e['level']) for e in entries} == {1, 2, 3, 4}, 'Expected four frozen selections')
    require(len(candidates) == 16 and not candidates.duplicated(['level', 'variant']).any(), 'Candidate table coverage')
    for r in candidates.itertuples():
        folds = cv[(cv.level == r.level) & (cv.variant == r.variant)]
        require(abs(folds.log_loss.mean() - r.mean_log_loss) < 1e-12, 'Candidate mean mismatch')
    comparisons, curves, sources = [], {}, []
    for e in sorted(entries, key=lambda v: v['level']):
        level, variant = int(e['level']), e['variant']
        require(variant in VARIANTS, 'Unknown selected variant')
        match = candidates[(candidates.level == level) & (candidates.variant == variant)]
        require(len(match) == 1 and abs(float(match.iloc[0].mean_log_loss) - float(e['mean_cv_log_loss'])) < 1e-12,
                'Selection does not match candidate summary')
        require(abs(float(e['mean_cv_log_loss']) - float(candidates.loc[candidates.level == level, 'mean_log_loss'].min())) < 1e-12,
                'Selected repair variant is not the CV minimum')
        eligible = bool(e['eligible_for_test'])
        row = dict(level=level, variant=variant, eligible_for_test=eligible,
            original_cv_log_loss=float(e['baseline_cv_log_loss']), repaired_cv_log_loss=float(e['mean_cv_log_loss']),
            delta_cv_log_loss=float(e['mean_cv_log_loss'])-float(e['baseline_cv_log_loss']),
            constant_prior_cv_log_loss=float(e['constant_prior_cv_log_loss']))
        if eligible:
            stage = root / 'final' / f'level_{level}'
            require((stage / 'complete.json').exists(), f'Missing final checkpoint: Level{level}')
            m = checked_json(stage, 'metrics.json')
            diagnostics = checked_json(stage, 'fit_diagnostics.json')
            require(int(m['level']) == level and m['variant'] == variant, 'Final model/selection mismatch')
            for metric in ('log_loss', 'pr_auc_average_precision', 'roc_auc', 'brier_score'):
                row['original_test_' + metric] = _metric(m['baseline_metrics'], metric)
                row['repaired_test_' + metric] = _metric(m['metrics'], metric)
                row['delta_test_' + metric] = row['repaired_test_' + metric] - row['original_test_' + metric]
            frame, best = _curve(diagnostics)
            curves[level] = (frame, best, variant)
            row.update(best_epoch=best, inner_epochs=len(frame), full_train_epochs=best,
                       fit_seconds=float(m['fit_seconds']), predict_seconds=float(m['predict_seconds']))
            require(int(diagnostics['final_optimizer']['epochs_completed']) == best,
                    'Final training epoch count differs from frozen selection')
            sources.extend([{'file': str((stage / name).resolve()), 'sha256': digest(stage / name)}
                            for name in ('metrics.json', 'fit_diagnostics.json')])
        else:
            require(not (root / 'final' / f'level_{level}' / 'complete.json').exists(),
                    'Ineligible level has a completed final evaluation')
        comparisons.append(row)
    for name in ('plan.json', 'cv_results.csv', 'candidate_summary.csv', 'selection.json'):
        sources.append({'file': str((root / name).resolve()), 'sha256': digest(root / name)})
    return plan, selection, cv, candidates, pd.DataFrame(comparisons), curves, sources


def figures(comparison, curves, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    folder = Path(output) / 'figures'
    folder.mkdir(exist_ok=True)
    files = []
    def save(fig, name):
        for ext in ('png', 'svg'):
            path = folder / f'{name}.{ext}'
            fig.savefig(path, dpi=160, bbox_inches='tight')
            files.append({'file': path.relative_to(output).as_posix(), 'sha256': digest(path)})
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for level, ax in zip(range(1, 5), axes.flat):
        if level not in curves:
            ax.axis('off')
            ax.text(.1, .5, f'Level {level}: no final fit\nCV eligibility criterion was not met.', transform=ax.transAxes)
            continue
        frame, best, variant = curves[level]
        ax.plot(frame.epoch, frame.train_log_loss, label='Inner train', color='#2563a6')
        ax.plot(frame.epoch, frame.validation_log_loss, label='Inner validation', color='#c47d16')
        ax.axvline(best, color='#51606c', linestyle='--', linewidth=1.2, label=f'Selected epoch {best}')
        ax.set(title=f'Level {level}: {variant}', xlabel='Epoch', ylabel='Unregularized log-loss')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('MLP epoch selection inside the 20,000-client training cohort\nNo outer-validation or test labels used in these curves', fontsize=13)
    save(fig, 'inner_learning_curves')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].plot(comparison.level, comparison.original_cv_log_loss, 'o-', label='Original MLP', color='#b34a39')
    axes[0].plot(comparison.level, comparison.repaired_cv_log_loss, 'o-', label='Selected repair variant', color='#2174a8')
    axes[0].plot(comparison.level, comparison.constant_prior_cv_log_loss, ':', label='Constant training prior', color='#64706c')
    axes[0].set(title='Saved outer-CV mean log-loss', ylabel='Log-loss (lower is better)')
    eligible = comparison[comparison.eligible_for_test]
    if len(eligible):
        axes[1].plot(eligible.level, eligible.original_test_log_loss, 'o-', label='Original MLP', color='#b34a39')
        axes[1].plot(eligible.level, eligible.repaired_test_log_loss, 'o-', label='Selected repair variant', color='#2174a8')
    else:
        axes[1].text(.1, .5, 'No variant passed the CV gate;\nno new test evaluation.', transform=axes[1].transAxes)
    axes[1].set(title='Previously used test: 61,503 clients', ylabel='Log-loss (lower is better)')
    for ax in axes:
        ax.set(xlabel='Information level', xticks=[1, 2, 3, 4])
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    fig.suptitle('Supplementary MLP repair: point estimates only\nVariants selected by CV; no bootstrap confidence intervals', fontsize=13)
    save(fig, 'mlp_before_after')
    return files


def export_mlp_repair_report(root, report):
    root, report = Path(root).resolve(), Path(report).resolve()
    require(root.name == SCOPE and report.name == SCOPE, 'Use the separate MLP-repair namespace')
    require(report != root and not report.is_relative_to(root) and not root.is_relative_to(report),
            'Report output must be separate from training artifacts')
    plan, selection, cv, candidates, comparison, curves, sources = load_tables(root)
    report.mkdir(parents=True, exist_ok=True)
    cv.to_csv(report / 'cv_results.csv', index=False, encoding='utf-8-sig')
    candidates.to_csv(report / 'candidate_summary.csv', index=False, encoding='utf-8-sig')
    comparison.to_csv(report / 'before_after.csv', index=False, encoding='utf-8-sig')
    write(report / 'frozen_plan.json', plan)
    write(report / 'frozen_selection.json', selection)
    curve_rows = []
    for level, (frame, best, variant) in curves.items():
        curve_rows.extend(frame.assign(level=level, variant=variant, selected_epoch=best).to_dict('records'))
    pd.DataFrame(curve_rows, columns=['epoch', 'train_log_loss', 'validation_log_loss',
        'optimizer_training_objective', 'epoch_seconds', 'level', 'variant', 'selected_epoch']).to_csv(
        report / 'inner_learning_curves.csv', index=False, encoding='utf-8-sig')
    figure_files = figures(comparison, curves, report)
    tested = comparison[comparison.eligible_for_test]
    narrative = [
        'Проверены четыре заранее заданных варианта MLP на четырёх уровнях и трёх сохранённых внешних CV-фолдах: 48 внешних CV-оценок.',
        'Каждая процедура включает внутренний выбор эпох по log-loss и отдельное обучение с нуля на всём доступном train. Поэтому 48 CV-оценок соответствуют 96 траекториям оптимизатора, а не 48 вызовам partial_fit.',
        f'По заранее записанному CV-критерию к итоговой оценке допущено {len(tested)} из четырёх уровней. Число итоговых процедур: {len(tested)}; они добавляют {2*len(tested)} траекторий оптимизатора.',
    ]
    for r in comparison.itertuples():
        line = f'Level {r.level}: выбран {r.variant}; CV log-loss {r.original_cv_log_loss:.6f} → {r.repaired_cv_log_loss:.6f}.'
        if r.eligible_for_test:
            line += (f' Test log-loss {r.original_test_log_loss:.6f} → {r.repaired_test_log_loss:.6f}; '
                     f'AP {r.original_test_pr_auc_average_precision:.6f} → {r.repaired_test_pr_auc_average_precision:.6f}; '
                     f'выбрано {int(r.best_epoch)} эпох.')
        else:
            line += ' Критерий допуска не выполнен; итоговое обучение и новая оценка на test отсутствуют.'
        narrative.append(line)
    narrative.append('Уменьшение CV/test log-loss описывает результат этого дополнительного эксперимента. Оно не доказывает, что найдена единственная причина исходной неудачи MLP.')
    text = 'Дополнительный эксперимент: попытка улучшить MLP\n\n' + '\n\n'.join(narrative) + '\n\nОграничения\n\n' + '\n'.join('- ' + x for x in LIMITATIONS)
    (report / 'results_report.txt').write_text(text + '\n', encoding='utf-8-sig')
    short = comparison[[c for c in ['level', 'variant', 'eligible_for_test', 'original_cv_log_loss', 'repaired_cv_log_loss',
        'original_test_log_loss', 'repaired_test_log_loss', 'original_test_pr_auc_average_precision',
        'repaired_test_pr_auc_average_precision', 'best_epoch'] if c in comparison]].copy()
    body = ['<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
        '<title>RMT: дополнительный эксперимент MLP</title><style>body{font:16px/1.55 system-ui,sans-serif;color:#173041;max-width:1200px;margin:32px auto;padding:0 20px}h1,h2{line-height:1.2}table{border-collapse:collapse;font-size:13px;min-width:650px}th,td{padding:7px 9px;border-bottom:1px solid #dce5ea;text-align:right}th{background:#edf3f7}td:first-child,th:first-child{text-align:left}.table{overflow-x:auto}img{display:block;width:100%;height:auto;margin:20px 0}.notice{border-left:4px solid #c47d16;padding:12px 16px;background:#fff8e9}a{color:#1769aa}</style></head><body>',
        '<h1>RMT: дополнительный эксперимент MLP</h1>',
        '<p class="notice">Результаты основного эксперимента сохранены отдельно. Этот анализ выполнен после просмотра прежнего test и имеет исследовательский характер.</p>']
    body.extend('<p>' + html.escape(s) + '</p>' for s in narrative)
    body.extend(['<h2>До и после</h2><div class="table">', short.to_html(index=False, border=0, float_format=lambda x: f'{x:.6f}', na_rep='—'), '</div>',
                 '<img src="figures/mlp_before_after.png" alt="Сравнение CV и test log-loss исходного и нового MLP">',
                 '<h2>Выбор числа эпох</h2><img src="figures/inner_learning_curves.png" alt="Кривые log-loss на внутреннем train и validation">',
                 '<h2>Все варианты CV</h2><div class="table">', candidates.to_html(index=False, border=0, float_format=lambda x: f'{x:.6f}'), '</div>',
                 '<h2>Ограничения</h2><ul>'])
    body.extend('<li>' + html.escape(s) + '</li>' for s in LIMITATIONS)
    body.append('</ul><h2>Файлы</h2><ul>')
    for name in ('before_after.csv', 'cv_results.csv', 'candidate_summary.csv', 'inner_learning_curves.csv',
                 'frozen_plan.json', 'frozen_selection.json', 'report_manifest.json', 'results_report.txt'):
        body.append(f'<li><a href="{name}">{name}</a></li>')
    body.append('</ul></body></html>')
    (report / 'results_report.html').write_text('\n'.join(body), encoding='utf-8')
    manifest = dict(scope=SCOPE, generated_utc=datetime.now(timezone.utc).isoformat(),
        status='generated_pending_visual_review', completed_cv_evaluations=48, optimizer_trajectories_cv=96,
        completed_final_procedures=len(tested), optimizer_trajectories_final=2*len(tested),
        new_fit_calls_in_reporting=0, new_predict_calls_in_reporting=0, bootstrap_calls=0,
        original_main_experiment_replaced=False, limitations=LIMITATIONS, sources=sources,
        figure_files=figure_files, source_report_sha256=digest(__file__))
    write(report / 'report_manifest.json', manifest)
    return manifest
