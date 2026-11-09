"""Separate article tables and figures for the 20-configuration final pilot.

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

SCOPE = 'pilot_20k_20config_final'
LIMITATIONS = [
    'Training used the same 20 000 development applications; this is an extended pilot, not training on the full development set specified by A2.',
    'For each of the 28 pairs, the configuration with the lowest mean log-loss among 20 configurations on the three saved CV folds was selected; one final fit was then performed, rather than the five A2 training repeats.',
    'The 1000 common paired bootstrap replicates reflect test composition with fixed models, not variation across repeated training.',
    'The 95% percentile CIs are pointwise, unadjusted, and exploratory; no SD across five training repeats is available here.',
    'Bonferroni correction is applied to six paired DeLong ROC-AUC comparisons with LR at Level 4. This does not test AP differences or equivalence.',
    'The test set is the same as in the preliminary results; this is an updated exploratory analysis, not a new external validation.',
    'Profit is expressed in relative units m=1 under a fixed cost model. Thresholds were not tuned on test, and no calibrator was fitted.',
    'Fairness gaps are descriptive; absence of a difference does not establish absence of bias, and an absolute-gap CI is not a separate test for bias.',
    'The CV batch 11–15 documents a change to the Random Forest thread executor while retaining n_jobs=2; timings from different computers are not directly comparable.',
    'This analysis does not perform SHAP, five training repeats, or the formal A2 selection rule based on AP noninferiority.',
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
                for item in diagnostics.get('mlp', []):
                    fitted, config = item['fitted'], item['config']
                    mlp.append(dict(model=model, level=level, n_iter_=fitted.get('n_iter_'),
                        max_iter=config.get('max_iter'), tol=config.get('tol'),
                        n_iter_no_change=config.get('n_iter_no_change'),
                        loss_curve_points=len(fitted.get('loss_curve_', [])), final_training_loss=fitted.get('loss_'),
                        best_cv_log_loss=record.get('best_cv_log_loss'), test_log_loss=record['metrics']['log_loss'],
                        test_ap=record['metrics']['pr_auc_average_precision'],
                        warning_count_final_fit=sum(w.get('count', 1) for w in record.get('warnings', [])),
                        selected_params_json=json.dumps(record['selected_params'], sort_keys=True),
                        fitted_config_json=json.dumps(config, sort_keys=True)))
    if len(mlp) != 4 or any(len(v) != 1 for v in feature_counts.values()):
        raise ValueError('Missing MLP actual diagnostics or inconsistent feature counts')
    summary_path = root / 'experiment_summary.json'
    summary = read(summary_path) if summary_path.exists() else {}
    protocol = dict(scope=SCOPE, train_rows=manifest['train_rows'], test_rows=manifest['test_rows'],
        development_rows=manifest.get('development_rows'), configurations_per_pair=20,
        completed_candidate_cv_fits=1680, completed_final_refits=28, final_fit_seed=42,
        repeats_per_pair=1, selection_metric='mean CV log_loss', selected_without_test=True,
        cohort_summary=cohort, cv_folds=cohort.get('cv'),
        feature_counts={k: next(iter(v)) for k,v in feature_counts.items()},
        software_versions=manifest.get('versions'), python=manifest.get('python'),
        threads=manifest.get('threads'), hardware=manifest.get('hardware'),
        training_summary=summary, refit_seconds_sum=sum(float(r['refit_seconds']) for r in timings),
        predict_seconds_sum=sum(float(r['predict_seconds']) for r in timings),
        timing_note='Final fit/predict times only; original 1680 CV work remains in separate batch logs. No double-counting old search/refit timings.',
        sources=sources, original_preresults_preserved=True)
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
           title='Final 20-configuration pilot: AP by information level')
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
    axes.flat[-1].text(.02,.9, 'Level 4: new final refits\n\n10 fixed equal-width bins.\nEmpty bins omitted.\nMarker size reflects bin count.\n\nNo calibrator was fitted.\nSmall bins are uncertain.\nExact counts in CSV.',
                       transform=axes.flat[-1].transAxes, va='top', fontsize=10)
    fig.supxlabel('Mean predicted default probability')
    fig.supylabel('Observed default rate')
    fig.suptitle('Calibration of final Level 4 models', fontsize=14)
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
        'Completed 28 final fits on the same 20,000 clients after parameter selection using all 1,680 CV results. '
        'Each model was evaluated on the same 61,503 test clients. Preprocessing was fitted only on train; parameters and thresholds were not changed using test.',
        f'RQ1: highest point estimate of AP at Level 4: {MODEL_LABELS[best_ap.model]} ({best_ap.ap:.6f}); '
        f'lowest log-loss: {MODEL_LABELS[best_loss.model]} ({best_loss.log_loss:.6f}). '
        'These are descriptive test results, not evidence of superiority across all metrics.',
    ]
    rejected = d[d.reject_bonferroni_0_05 == True]
    paragraphs.append('DeLong, ROC-AUC versus LR, Bonferroni p <0.05: '+
        ('; '.join(f'{MODEL_LABELS[r.model]}: Δ={r.delta_roc_auc:+.6f}, p_adj={r.p_bonferroni:.5g}' for r in rejected.itertuples()) or 'none')+
        '. The family contains six comparisons; this test does not assess AP.')
    gains = tables['rq2_deltas'].query('kind=="baseline" and to_level==4')
    paragraphs.append('RQ2, ΔAP Level 4 − Level 1 [95% CI]: '+ '; '.join(
        f'{MODEL_LABELS[r.model]} {r.delta_ap:+.5f} [{r.ci_low:+.5f}, {r.ci_high:+.5f}]' for r in gains.itertuples())+'.')
    paragraphs.append('MLP, actual final fits: '+ '; '.join(
        f'L{r.level}: n_iter={r.n_iter_}/{r.max_iter}, training loss={r.final_training_loss:.5f}, '
        f'CV log-loss={r.best_cv_log_loss:.5f}, test log-loss={r.test_log_loss:.5f}' for r in mlp.itertuples())+
        '. MLP settings were retained: max_iter=300, tol=0.0001, n_iter_no_change=20; the detailed audit accompanies the combined CV results.')
    paragraphs.append('Before final training, the MLP audit found no structural Pipeline error or leakage of statistics '
        'from validation: all checked matrices were finite. Selected configuration 13 reached max_iter=300 '
        'in all 12 saved CV fits. A low training objective with high CV log-loss is consistent '
        'with overfitting; rare extreme values after scaling may amplify instability. '
        'This is a diagnostic explanation, not proof of a single cause. Clipping, changes to regularization, '
        'and early stopping were not introduced. MLP loss_ includes regularization and is not a separate estimate of training log-loss.')
    paragraphs.append('The previous five-configuration search results are preserved separately. '
        'comparison_to_preserved_5config_preresults.csv shows descriptive changes in the new metrics; '
        'previous Pipelines were not substituted for new final models. This module only evaluated saved probabilities.')
    main_ci = tables['quality_ci'][(tables['quality_ci'].metric=='ap') |
        ((tables['quality_ci'].level==4)&tables['quality_ci'].metric.isin(['roc_auc','log_loss','brier']))]
    blockers = []
    if summary.get('completed_bootstrap') != 1000 or len(main_ci)!=49 or not main_ci.bootstrap_n_valid.eq(1000).all():
        blockers.append('Not all 1,000 replicates for the primary intervals have been confirmed.')
    if integrity.get('unchanged') is not True or integrity.get('old_preresults_unchanged') is not True:
        blockers.append('Preservation of source files and previous preliminary results has not been confirmed.')
    graphics_path = output/'graphics_validation.json'
    graphics = read(graphics_path) if graphics_path.exists() else {}
    if graphics.get('passed') is not True:
        blockers.append('Visual inspection of the new figures has not yet been recorded.')
    if len(figure_files)!=6:
        blockers.append('The three PNG/SVG figure pairs are not ready.')
    readiness = dict(scope=SCOPE, status='ready' if not blockers else 'pending_checks',
        generated_utc=datetime.now(timezone.utc).isoformat(), blockers=blockers,
        completed_cv_fits=1680, final_fit_count=28, test_rows=61503, completed_bootstrap=1000,
        new_fit_calls_in_evaluation=0, new_predict_calls_in_evaluation=0,
        old_preresults_preserved=integrity.get('old_preresults_unchanged'), limitations=LIMITATIONS,
        table_files=table_files, figure_files=figure_files, graphics_validation=graphics,
        original_reports=str(previous.resolve()), metrics_are_new_final_predictions=True)
    _json(output/'results_readiness.json',readiness)
    _atomic_text(output/'latex_usage.txt',
        'Packages: booktabs, longtable, graphicx. Example: \\input{tables/rq1_level4.tex}.\n'
        'The table gives the point estimate followed by [95% CI]. Intervals describe fixed models, not five training repeats.\n'
        'The LaTeX fragments were not compiled; check the final layout in the paper template.\n')
    report_text = 'RQ1/RQ2: final models of the 20k / 20-configuration pilot\n\n'+'\n\n'.join(paragraphs)+\
        '\n\nLimitations\n'+'\n'.join('- '+x for x in LIMITATIONS)+\
        '\n\nReadiness\n'+('Numeric results and figures have been checked.' if not blockers else '\n'.join(blockers))+'\n'
    _atomic_text(output/'results_report.txt',report_text)
    css='body{font:16px/1.55 Segoe UI,Arial,sans-serif;max-width:1180px;margin:35px auto;padding:0 22px;color:#173248}h1,h2{color:#123e5b}table{border-collapse:collapse;font-size:12px;width:100%}th,td{padding:7px;border-bottom:1px solid #d9e2e8;text-align:left;white-space:pre-line}th{background:#eef4f8}.scroll{overflow:auto}img{max-width:100%}small{color:#536979}'
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><title>RMT: final RQ1/RQ2 results</title><style>'+css+'</style><body>',
        '<h1>RQ1/RQ2: 20k pilot / 20 configurations</h1>']
    parts.extend('<p>'+html.escape(p)+'</p>' for p in paragraphs)
    for name in ['rq1_level4','rq2_ap_levels','rq2_baseline_gains','rq2_sequential_gains','ap_vs_lr_level4',
                 'delong_level4','profit_fairness_L1','profit_fairness_L5','profit_fairness_L10','errors_level4']:
        frame, caption, _ = prepared[name]
        parts.extend(['<h2>'+html.escape(caption)+'</h2>', '<div class="scroll">'+frame.to_html(index=False,border=0)+'</div>',
                      f'<small><a href="tables/{name}.csv">CSV</a> · <a href="tables/{name}.tex">LaTeX</a></small>'])
    for name in ('rq1_quality_level4','ap_by_level','calibration_level4'):
        parts.append(f'<figure><img src="figures/{name}.png" alt="{name}"><figcaption><a href="figures/{name}.svg">SVG</a></figcaption></figure>')
    parts.extend(['<h2>Actual MLP training</h2>',mlp.drop(columns=['selected_params_json','fitted_config_json']).to_html(index=False,border=0),
        '<h2>Limitations</h2><ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in LIMITATIONS)+'</ul>',
        '<h2>Readiness</h2><p>'+html.escape('Verified.' if not blockers else ' '.join(blockers))+'</p></body></html>'])
    _atomic_text(output/'results_report.html','\n'.join(parts))
    return readiness
