"""Evaluation Assignment3/RQ1/RQ2: только сохранённые вероятности пилота 20k.

Ни модели, ни X не загружаются. Нет fit, predict, оптимизации порога или
переопределения train/test. Все новые файлы находятся внутри evaluation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, roc_auc_score, log_loss,
    brier_score_loss, confusion_matrix, precision_score, recall_score, f1_score)
from threadpoolctl import threadpool_limits

from .pilot_evaluation_bootstrap import (sha256, atomic_json, atomic_replace,
    bootstrap_run, percentile_interval)
from .pilot_evaluation_decisions import (DecisionEvaluator, calibration_point_table,
    THRESHOLDS, SCENARIOS, SCENARIO_NAMES, ATTRIBUTES, LOSS_RATIOS, BASELINE_ORDER)
from .pilot_delong import paired_delong_table, DELONG_METADATA

MODELS = ('logistic_regression', 'lightgbm', 'xgboost', 'soft_voting',
          'stacking', 'mlp', 'random_forest')
LEVELS = (1, 2, 3, 4)
SCOPE = 'pilot_20k_5config'
CI_LABEL = '95% percentile; pointwise, unadjusted, exploratory; fixed fitted models'
SOURCE_FILES = ('pilot_evaluation.py', 'pilot_evaluation_bootstrap.py',
                'pilot_evaluation_decisions.py', 'pilot_delong.py')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def csv_write(frame, path):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        frame.to_csv(stream, index=False)
    atomic_replace(temporary, path)


def integrity_snapshot(prepared, root, previous):
    """Хеши готовых файлов; исходные CSV не открываются и не аудируются."""
    files = {}
    for prefix, folder in [('prepared', prepared), ('training', root), ('previous_reports', previous)]:
        for path in sorted(Path(folder).rglob('*')):
            if not path.is_file():
                continue
            relative = path.relative_to(folder)
            if prefix == 'previous_reports' and relative.parts[0] == 'evaluation':
                continue
            files[f'{prefix}/{relative.as_posix()}'] = sha256(path)
    return files


def _require_hash(path, expected):
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f'Input checksum mismatch: {path}')


def load_predictions(prepared, root):
    """Явные one-to-one merge по ID; общий канонический порядок sorted test ID."""
    prepared, root = Path(prepared), Path(root)
    manifest = read_json(root / 'experiment_manifest.json')
    if manifest['mode'] != SCOPE or manifest['train_rows'] != 20000 or manifest['test_rows'] != 61503:
        raise ValueError('Not the completed pilot_20k_5config cohort')
    original_hashes = {r['file']: r['sha256'] for r in manifest['inputs']}
    for name in ['target.parquet', 'fairness.parquet', 'splits/test_ids.parquet']:
        _require_hash(prepared / name, original_hashes[name])
    contract = read_json(root / 'cohort_contract.json')
    for name, digest in contract['artifact_sha256'].items():
        _require_hash(root / name, digest)
    test = pd.read_parquet(root / 'test_ids.parquet')
    original_test = pd.read_parquet(prepared / 'splits/test_ids.parquet')
    if not np.array_equal(test.SK_ID_CURR, original_test.SK_ID_CURR) or not test.SK_ID_CURR.is_unique:
        raise ValueError('Test ID changed')
    cohort = pd.read_parquet(root / 'cohort_ids.parquet').SK_ID_CURR
    if not set(cohort).isdisjoint(test.SK_ID_CURR):
        raise ValueError('Train/test overlap')
    target = pd.read_parquet(prepared / 'target.parquet')
    fairness = pd.read_parquet(prepared / 'fairness.parquet')
    if not target.SK_ID_CURR.is_unique or not fairness.SK_ID_CURR.is_unique:
        raise ValueError('Duplicate target/fairness ID')
    aligned = test.sort_values('SK_ID_CURR').merge(target, on='SK_ID_CURR',
        how='left', validate='one_to_one', indicator=True)
    if not aligned['_merge'].eq('both').all() or not aligned.TARGET.isin([0, 1]).all():
        raise ValueError('Missing TARGET after ID merge')
    aligned = aligned.drop(columns='_merge').merge(fairness, on='SK_ID_CURR',
        how='left', validate='one_to_one', indicator=True)
    if len(aligned) != 61503 or not aligned['_merge'].eq('both').all():
        raise ValueError('Missing fairness row after ID merge')
    aligned = aligned.drop(columns='_merge')
    gender = aligned.CODE_GENDER.where(aligned.CODE_GENDER.isin(['F', 'M']), 'unknown')
    ages = pd.to_numeric(aligned.AGE_YEARS, errors='coerce').to_numpy(dtype=float)
    expected_age = np.where(np.isfinite(ages), np.where(ages < 25, '<25', '>=25'), 'unknown')
    if not np.array_equal(gender, aligned.GENDER_GROUP) or not np.array_equal(expected_age, aligned.AGE_GROUP):
        raise ValueError('Saved fairness groups differ from their recorded definitions')
    groups = {'gender': gender.to_numpy(dtype=str), 'age': aligned.AGE_GROUP.to_numpy(dtype=str)}
    y = aligned.TARGET.to_numpy(dtype=np.int8)
    pairs, probabilities, checks = [], [], []
    for model in MODELS:
        for level in LEVELS:
            stage = root / model / f'level_{level}'
            marker = read_json(stage / 'complete.json')
            if marker['fingerprint'] != manifest['fingerprint']:
                raise ValueError(f'Wrong training fingerprint: {stage}')
            for name in ['probabilities.parquet', 'metrics.json']:
                _require_hash(stage / name, marker['artifact_sha256'][name])
            frame = pd.read_parquet(stage / 'probabilities.parquet')
            if len(frame) != len(test) or not frame.SK_ID_CURR.is_unique or set(frame.SK_ID_CURR) != set(test.SK_ID_CURR):
                raise ValueError(f'Different client set: {stage}')
            for field, value in [('scope', SCOPE), ('model', model), ('level', level), ('evaluation_split', 'test')]:
                if not frame[field].eq(value).all():
                    raise ValueError(f'Invalid prediction provenance: {stage}/{field}')
            joined = aligned[['SK_ID_CURR', 'TARGET']].merge(frame[['SK_ID_CURR', 'TARGET', 'probability_default']],
                on='SK_ID_CURR', how='left', suffixes=('_source', '_saved'), validate='one_to_one', indicator=True)
            if not joined['_merge'].eq('both').all() or not np.array_equal(joined.TARGET_source, joined.TARGET_saved):
                raise ValueError(f'TARGET mismatch after explicit ID merge: {stage}')
            p = joined.probability_default.to_numpy(dtype=np.float64)
            if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                raise ValueError(f'Invalid saved probability: {stage}')
            pairs.append({'model': model, 'level': level})
            probabilities.append(p)
            checks.append({'model': model, 'level': level, 'test_rows': len(p),
                           'same_ids': True, 'target_matches_after_merge': True,
                           'finite_probabilities_in_0_1': True})
    source = root / 'source_snapshot/model_training.py'
    _require_hash(source, manifest['code_sha256']['model_training.py'])
    source_text = source.read_text(encoding='utf-8')
    if 'probabilities[:,int(np.flatnonzero(classes==1)[0])]' not in source_text:
        raise ValueError('Positive class provenance needs manual review')
    validation = {'scope': SCOPE, 'test_rows': len(y), 'test_positives': int(y.sum()),
        'all_28_aligned': len(pairs) == 28, 'positive_class': 1, 'positive_class_evidence':
        'SHA-verified frozen positive_probability selects column where estimator.classes_ == 1; research_pilot saved this vector.',
        'canonical_order': 'ascending SK_ID_CURR; all joins explicit one-to-one',
        'unknown_group_counts': {k: int((v == 'unknown').sum()) for k, v in groups.items()},
        'unknown_policy': 'retained in quality/profit; reported separately, excluded only from binary group gaps',
        'age_definition': 'saved AGE_GROUP <25 / >=25; exactly25 belongs to >=25',
        'checks': checks, 'errors': []}
    return aligned, y, np.stack(probabilities), pairs, groups, validation


def quality_point(y, probabilities, pairs):
    rows = []
    for pair, p in zip(pairs, probabilities):
        predicted = p >= .5
        tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
        rows.append({**pair, 'ap': average_precision_score(y, p, pos_label=1),
            'roc_auc': roc_auc_score(y, p), 'log_loss': log_loss(y, p, labels=[0, 1]),
            'brier': brier_score_loss(y, p), 'threshold': .5,
            'precision': precision_score(y, predicted, zero_division=np.nan),
            'recall': recall_score(y, predicted, zero_division=np.nan),
            'f1': f1_score(y, predicted, zero_division=np.nan),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)})
    return pd.DataFrame(rows)


def compare_quality(point, previous):
    old = pd.read_csv(Path(previous) / 'rq1_model_comparison.csv')
    mapping = {'ap': 'test_pr_auc_average_precision', 'roc_auc': 'test_roc_auc',
               'log_loss': 'test_log_loss', 'brier': 'test_brier_score'}
    merged = point.merge(old, on=['model', 'level'], validate='one_to_one')
    if len(merged) != 28:
        raise ValueError('Existing report does not cover all28 pairs')
    rows = []
    for _, row in merged.iterrows():
        for metric, column in mapping.items():
            delta = abs(row[metric] - row[column])
            rows.append({'model': row.model, 'level': int(row.level), 'metric': metric,
                'recomputed': row[metric], 'previous': row[column], 'absolute_difference': delta,
                'within_tolerance': bool(np.isclose(row[metric], row[column], rtol=1e-10, atol=1e-12))})
    return pd.DataFrame(rows)


def summarize_bootstrap(arrays, point, pairs, decision_tables):
    rows, deltas, comparisons = [], [], []
    position = {(p['model'], p['level']): i for i, p in enumerate(pairs)}
    l4 = [i for i, p in enumerate(pairs) if p['level'] == 4]
    for i, pair in enumerate(pairs):
        low, high, n = percentile_interval(arrays['ap'][:, i])
        rows.append({**pair, 'metric': 'ap', 'estimate': point.iloc[i].ap,
                     'ci_low': low, 'ci_high': high, 'bootstrap_n_valid': n})
    for j, i in enumerate(l4):
        for metric, key in [('roc_auc', 'roc_auc_l4'), ('log_loss', 'log_loss_l4'), ('brier', 'brier_l4')]:
            low, high, n = percentile_interval(arrays[key][:, j])
            rows.append({**pairs[i], 'metric': metric, 'estimate': point.iloc[i][metric],
                         'ci_low': low, 'ci_high': high, 'bootstrap_n_valid': n})
        for metric in ['precision', 'recall', 'f1']:
            low, high, n = percentile_interval(arrays[f'decision_{metric}'][:, j, 0])
            rows.append({**pairs[i], 'metric': metric, 'estimate': point.iloc[i][metric],
                         'ci_low': low, 'ci_high': high, 'bootstrap_n_valid': n})
    reference = position[('logistic_regression', 4)]
    for model in MODELS:
        idx = position[(model, 4)]
        if model != 'logistic_regression':
            low, high, n = percentile_interval(arrays['ap'][:, idx] - arrays['ap'][:, reference])
            comparisons.append({'model': model, 'reference': 'logistic_regression',
                'delta_ap': point.iloc[idx].ap - point.iloc[reference].ap,
                'ci_low': low, 'ci_high': high, 'bootstrap_n_valid': n})
        for kind, edges in [('baseline', [(1, 2), (1, 3), (1, 4)]),
                            ('sequential', [(1, 2), (2, 3), (3, 4)])]:
            for start, stop in edges:
                a, b = position[(model, start)], position[(model, stop)]
                low, high, n = percentile_interval(arrays['ap'][:, b] - arrays['ap'][:, a])
                deltas.append({'model': model, 'comparison': f'L{stop}-L{start}',
                    'from_level': start, 'to_level': stop, 'kind': kind,
                    'delta_ap': point.iloc[b].ap - point.iloc[a].ap,
                    'ci_low': low, 'ci_high': high, 'bootstrap_n_valid': n})
    profit_ci, fairness_ci, baseline_ci, threshold_ci = [], [], [], []
    for m, model in enumerate(MODELS):
        for s, (loss, t) in enumerate(SCENARIOS):
            record = decision_tables['profit'].loc[(decision_tables['profit'].model == model)
                & (decision_tables['profit'].scenario == SCENARIO_NAMES[s])].iloc[0]
            for metric, values in [('realized_profit_per_applicant', arrays['decision_profit'][:, m, s]),
                ('approval_rate', arrays['decision_approval_rate'][:, m, t]),
                ('default_rate_among_approved', arrays['decision_default_rate_among_approved'][:, m, t])]:
                low, high, n = percentile_interval(values)
                profit_ci.append({'model': model, 'scenario': SCENARIO_NAMES[s], 'threshold': THRESHOLDS[t],
                    'L': loss, 'm': 1, 'metric': metric, 'estimate': record[metric],
                    'ci_low': low, 'ci_high': high, 'bootstrap_n_valid': n})
        for t, threshold in enumerate(THRESHOLDS):
            row = decision_tables['fairness_gaps'].loc[(decision_tables['fairness_gaps'].model == model) & (decision_tables['fairness_gaps'].threshold == threshold)].iloc[0]
            for a, attribute in enumerate(ATTRIBUTES):
                for metric, key, column in [('approval_rate_gap', 'approval_gap', f'{attribute}_approval_rate_gap'),
                                           ('default_tpr_gap', 'tpr_gap', f'{attribute}_tpr_gap')]:
                    low, high, n = percentile_interval(arrays[f'decision_{key}'][:, m, t, a])
                    fairness_ci.append({'model': model, 'threshold': threshold, 'attribute': attribute,
                        'metric': metric, 'estimate': row[column], 'ci_low': low, 'ci_high': high,
                        'bootstrap_n_valid': n})
            errors = decision_tables['threshold_metrics'].loc[(decision_tables['threshold_metrics'].model == model) & (decision_tables['threshold_metrics'].threshold == threshold)].iloc[0]
            for metric in ['precision', 'recall', 'f1', 'TN', 'FP', 'FN', 'TP']:
                values = arrays['decision_confusion'][:, m, t, ['TN','FP','FN','TP'].index(metric)] if metric.isupper() else arrays[f'decision_{metric}'][:, m, t]
                low, high, n = percentile_interval(values)
                threshold_ci.append({'model': model, 'threshold': threshold, 'metric': metric,
                    'estimate': errors[metric], 'ci_low': low, 'ci_high': high, 'bootstrap_n_valid': n})
    for l, loss in enumerate(LOSS_RATIOS):
        for b, policy in enumerate(BASELINE_ORDER):
            low, high, n = percentile_interval(arrays['decision_baselines'][:, l, b])
            row = decision_tables['baselines'].loc[(decision_tables['baselines'].policy == policy) & (decision_tables['baselines'].loss_to_margin == loss)].iloc[0]
            baseline_ci.append({'policy': policy, 'L': loss, 'm': 1, 'metric': 'realized_profit_per_applicant',
                'estimate': row.realized_profit_per_applicant, 'ci_low': low, 'ci_high': high, 'bootstrap_n_valid': n})
    return {name: pd.DataFrame(records) for name, records in [('quality_ci', rows), ('ap_vs_lr', comparisons),
        ('rq2_deltas', deltas), ('profit_ci', profit_ci), ('fairness_ci', fairness_ci),
        ('baselines_ci', baseline_ci), ('threshold_metrics_ci', threshold_ci)]}


def run_evaluation(prepared_dir, pilot_root, previous_reports, output_dir, *, n_bootstrap=1000, seed=42):
    started = time.perf_counter()
    prepared, root, previous, output = map(Path, (prepared_dir, pilot_root, previous_reports, output_dir))
    if output.resolve() != (previous / 'evaluation').resolve():
        raise ValueError('New outputs must be in reports/pilot_20k_5config/evaluation')
    if n_bootstrap != 1000 or seed != 42:
        raise ValueError('This research protocol requires1000 paired replicates and seed42')
    output.mkdir(parents=True, exist_ok=True)
    before = integrity_snapshot(prepared, root, previous)
    if (output / 'integrity_before.json').exists():
        if read_json(output / 'integrity_before.json')['sha256'] != before:
            raise ValueError('A source artifact changed since evaluation initialization')
    else:
        atomic_json(output / 'integrity_before.json', {'sha256': before})
    aligned, y, probabilities, pairs, groups, validation = load_predictions(prepared, root)
    atomic_json(output / 'alignment_validation.json', validation)
    point = quality_point(y, probabilities, pairs)
    comparison = compare_quality(point, previous)
    csv_write(comparison, output / 'existing_metrics_comparison.csv')
    if not comparison.within_tolerance.all():
        raise ValueError('Existing metrics differ materially; inspect comparison CSV before proceeding')
    csv_write(point, output / 'quality_point.csv')
    canonical = output / 'aligned_test_ids.parquet'
    if canonical.exists():
        if not np.array_equal(pd.read_parquet(canonical).SK_ID_CURR, aligned.SK_ID_CURR):
            raise ValueError('Saved bootstrap client order changed')
    else:
        aligned[['SK_ID_CURR', 'TARGET', 'GENDER_GROUP', 'AGE_GROUP']].to_parquet(canonical, index=False)
    probdict = {pair['model']: probabilities[i] for i, pair in enumerate(pairs) if pair['level'] == 4}
    evaluator = DecisionEvaluator(y, probdict, groups)
    decisions = evaluator.point_tables()
    for name, table in decisions.items():
        csv_write(table, output / f'{"profit_point" if name == "profit" else name}.csv')
    calibration = calibration_point_table(y, probdict)
    old_calibration = pd.read_csv(previous / 'calibration_bins.csv').sort_values(['model', 'bin']).reset_index(drop=True)
    expected = calibration.sort_values(['model', 'bin']).reset_index(drop=True)
    if not old_calibration[['model', 'bin', 'n', 'default_n']].equals(expected[['model', 'bin', 'n', 'default_n']]):
        raise ValueError('Existing calibration bin counts differ')
    for column in ['mean_probability_default', 'observed_default_rate']:
        if not np.allclose(old_calibration[column], expected[column], atol=1e-12, rtol=1e-10, equal_nan=True):
            raise ValueError('Existing calibration values differ')
    csv_write(old_calibration, output / 'calibration_bins.csv')  # verified reuse
    previous_profit = pd.read_csv(previous / 'decision_scenarios.csv')
    prior_checks = []
    for _, row in decisions['profit'].iterrows():
        matching = previous_profit.loc[(previous_profit.model == row.model) &
            (previous_profit.loss_to_margin == row.loss_to_margin) &
            np.isclose(previous_profit.threshold, row.threshold, atol=1e-14, rtol=0)]
        for metric in ['realized_profit_per_applicant', 'approval_rate', 'approved_n']:
            valid = len(matching) > 0 and np.allclose(matching[metric], row[metric], atol=1e-12, rtol=1e-10)
            prior_checks.append({'model': row.model, 'scenario': row.scenario, 'metric': metric, 'matches_previous': bool(valid)})
    csv_write(pd.DataFrame(prior_checks), output / 'existing_decision_comparison.csv')
    if not all(row['matches_previous'] for row in prior_checks):
        raise ValueError('Profit recalculation differs from existing report')
    delong = paired_delong_table(y, probdict)
    csv_write(delong, output / 'delong.csv')
    atomic_json(output / 'delong_methodology.json', DELONG_METADATA)
    context = {'scope': SCOPE, 'analysis': 'Assignment3 preliminary RQ1/RQ2',
        'training_fingerprint': read_json(root / 'experiment_manifest.json')['fingerprint'],
        'source_sha256': before, 'aligned_id_sha256': sha256(canonical), 'pairs': pairs,
        'analysis_code_sha256': {name: sha256(Path(__file__).parent / name) for name in SOURCE_FILES},
        'n_bootstrap': n_bootstrap, 'seed': seed, 'resampling_unit': 'client',
        'resampling': 'ordinary with replacement, not stratified; one shared vector of client counts per replicate',
        'rng': 'numpy.default_rng(42) / PCG64, checkpointed state', 'interval': CI_LABEL,
        'percentile_quantiles': [.025, .975], 'quantile_method': 'linear',
        'ap_definition': 'PR-AUC (average precision, AP), sklearn.metrics.average_precision_score(pos_label=1)',
        'weighted_replication': 'integer frequencies equivalent to duplicating sampled client IDs; AP uses official sklearn function every repetition',
        'thresholds': list(THRESHOLDS), 'decision_coordinates': evaluator.coordinates,
        'fairness_constraint': 'observed gender and age approval-rate gaps each <=0.05; TPR gaps descriptive without extra cutoff',
        'undefined_rates': 'NaN for zero denominator, never imputed; valid bootstrap count reported',
        'comparisons': {'ap_vs_lr_level4': [m for m in MODELS if m != 'logistic_regression'],
            'delong_roc_auc_vs_lr_level4': [m for m in MODELS if m != 'logistic_regression'],
            'bonferroni_family_size': 6, 'rq2_baseline': ['L2-L1', 'L3-L1', 'L4-L1'],
            'rq2_sequential': ['L2-L1', 'L3-L2', 'L4-L3']},
        'analysis_specification_time': 'specified for this post-training evaluation, not claimed preregistered',
        'new_fit_calls': 0, 'new_predict_calls': 0, 'threshold_optimization': False,
        'retraining_variability_estimated': False, 'formal_A2_AP_noninferiority_confirmed': False,
        'software': {name: importlib.metadata.version(name) for name in ['numpy','scipy','pandas','scikit-learn']},
        'method_sources': ['https://scikit-learn.org/1.7/modules/generated/sklearn.metrics.average_precision_score.html',
                           'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html']}
    context['fingerprint'] = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
    manifest_path = output / 'evaluation_manifest.json'
    if manifest_path.exists():
        if read_json(manifest_path)['fingerprint'] != context['fingerprint']:
            raise ValueError('Analysis protocol/code changed; do not silently reuse old bootstrap')
    else:
        atomic_json(manifest_path, context)
    print('Aligned28 pairs,61503 test clients; quality/profit/calibration match prior reports. No fit/predict.', flush=True)
    arrays, progress = bootstrap_run(y, probabilities, pairs, groups, output / 'bootstrap',
        fingerprint=context['fingerprint'], n_bootstrap=n_bootstrap, seed=seed)
    if progress['completed'] != n_bootstrap:
        raise RuntimeError('Bootstrap incomplete; checkpoints retained')
    tables = summarize_bootstrap(arrays, point, pairs, decisions)
    for name, table in tables.items():
        csv_write(table, output / f'{name}.csv')
    # Complete long CSV distributions for reproducibility; weights stay in npz.
    distribution_rows = []
    for replicate in range(n_bootstrap):
        for i, pair in enumerate(pairs):
            row = {'replicate': replicate, **pair, 'ap': arrays['ap'][replicate, i]}
            if pair['level'] == 4:
                m = MODELS.index(pair['model'])
                row.update({metric: arrays[key][replicate, m] for metric, key in
                            [('roc_auc','roc_auc_l4'),('log_loss','log_loss_l4'),('brier','brier_l4')]})
            distribution_rows.append(row)
    csv_write(pd.DataFrame(distribution_rows), output / 'bootstrap_quality_distribution.csv')
    after = integrity_snapshot(prepared, root, previous)
    differences = [key for key in set(before) | set(after) if before.get(key) != after.get(key)]
    atomic_json(output / 'integrity_after.json', {'unchanged': not differences,
        'checked_files': len(before), 'mismatches': sorted(differences),
        'before_manifest_sha256': sha256(output / 'integrity_before.json')})
    if differences:
        raise ValueError('Immutable source files changed during evaluation')
    summary = {'status': 'completed', 'completed_bootstrap': n_bootstrap, 'expected_bootstrap': n_bootstrap,
        'seed': seed, 'model_count': 7, 'pair_count': 28, 'test_rows': len(y),
        'input_integrity_unchanged': True, 'alignment_verified': True, 'existing_metrics_match': True,
        'calibration_reused_and_validated': True, 'new_fit_calls': 0, 'new_predict_calls': 0,
        'checkpoint_count': len(list((output / 'bootstrap').glob('replicate_*.npz'))),
        'evaluation_elapsed_seconds': time.perf_counter() - started,
        'bootstrap_new': progress['new'], 'bootstrap_reused': progress['reused'],
        'completed_utc': datetime.now(timezone.utc).isoformat(), 'intervals': CI_LABEL,
        'undefined_primary_ci_count': int(tables['quality_ci'].loc[tables['quality_ci'].metric.isin(['ap','roc_auc','log_loss','brier']), 'bootstrap_n_valid'].lt(n_bootstrap).sum())}
    atomic_json(output / 'evaluation_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', default='data/processed/full')
    parser.add_argument('--pilot-root', default='artifacts/training/pilot_20k_5config')
    parser.add_argument('--previous-reports', default='reports/pilot_20k_5config')
    parser.add_argument('--output', default='reports/pilot_20k_5config/evaluation')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run_evaluation(args.prepared, args.pilot_root, args.previous_reports, args.output)


if __name__ == '__main__':
    main()
