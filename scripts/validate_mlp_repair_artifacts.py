"""Independent, read-only audit of a completed supplemental MLP run.

No training module is imported, no fitted model is unpickled, and no fit,
prediction or bootstrap is performed. Metrics are recomputed from saved
probabilities. The optional --output writes only this new audit report.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
    brier_score_loss, confusion_matrix, f1_score, log_loss, precision_score,
    recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split


EXPECTED_FINGERPRINT = '2bb884673500177f5d369b1dcb98888290a69a56b1141bc6a1d838eb4a2d3bf5'
PROTECTED_ROOTS = (
    'artifacts/training/pilot_20k_5config', 'reports/pilot_20k_5config',
    'artifacts/training/pilot_20k_20config_final', 'reports/pilot_20k_20config_final',
    'reports/training/pilot_20k_20config_combined',
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, message):
    require(np.isfinite(actual) and np.isfinite(expected)
            and np.isclose(actual, expected, rtol=2e-11, atol=2e-12),
            f'{message}: saved={actual!r}, recomputed={expected!r}')


def aligned(actual, expected, message):
    require(np.array_equal(np.asarray(actual), np.asarray(expected)), message)


def validate_checkpoint(stage, fingerprint, required):
    marker = read_json(stage / 'complete.json')
    require(marker['fingerprint'] == fingerprint, f'{stage}: fingerprint mismatch')
    require(set(marker['artifact_sha256']) == set(required), f'{stage}: unexpected artifact set')
    for name, digest in marker['artifact_sha256'].items():
        path = stage / name
        require(path.is_file() and sha256(path) == digest, f'{path}: SHA256 mismatch')
    return marker


def recompute_metrics(labels, probabilities):
    prediction = probabilities >= .5
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    return dict(rows=len(labels), positives=int(np.sum(labels)), positive_rate=float(np.mean(labels)),
        log_loss=float(log_loss(labels, probabilities, labels=[0, 1])),
        pr_auc_average_precision=float(average_precision_score(labels, probabilities)),
        roc_auc=float(roc_auc_score(labels, probabilities)),
        brier_score=float(brier_score_loss(labels, probabilities)), threshold=.5,
        accuracy=float(accuracy_score(labels, prediction)),
        precision=float(precision_score(labels, prediction, zero_division=0)),
        recall=float(recall_score(labels, prediction, zero_division=0)),
        f1=float(f1_score(labels, prediction, zero_division=0)),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        approval_rate_at_0_5=float(np.mean(probabilities < .5)))


def validate_probabilities(path, expected_ids, expected_labels):
    frame = pd.read_parquet(path)
    require(frame.SK_ID_CURR.is_unique, f'{path}: duplicate IDs')
    aligned(frame.SK_ID_CURR, expected_ids, f'{path}: ID order')
    aligned(frame.TARGET, expected_labels, f'{path}: TARGET alignment')
    p = frame.probability_default.to_numpy(dtype=float)
    require(np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all(), f'{path}: invalid probability')
    return frame, recompute_metrics(np.asarray(expected_labels), p)


def validate_diagnostics(d, expected_ids, labels, columns, variant, settings):
    expected_ids, labels = np.asarray(expected_ids), np.asarray(labels)
    require(d['variant'] == variant and d['seed'] == settings['seed'], 'Diagnostics variant/seed')
    require(d['provided_train_rows'] == d['final_refit_rows'] == len(expected_ids), 'Train row count')
    require(d['provided_train_ids_sha256'] == d['final_refit_train_ids_sha256']
            == canonical_digest(expected_ids.tolist()), 'Train ID hash/order')
    require(d['raw_features'] == len(columns['numeric']) + len(columns['categorical']), 'Feature count')
    require(d['numeric_features'] == len(columns['numeric'])
            and d['categorical_features'] == len(columns['categorical']), 'Feature roles')
    inner, validation = train_test_split(np.arange(len(labels)), stratify=labels,
        test_size=settings['validation_fraction'], random_state=settings['seed'])
    split = d['internal_split']
    for name, positions in [('train', inner), ('validation', validation)]:
        actual_ids = expected_ids[positions]
        aligned(split[f'{name}_ids'], actual_ids, f'Internal {name} IDs/order')
        require(split[f'{name}_rows'] == len(positions), f'Internal {name} rows')
        require(split[f'{name}_ids_sha256'] == canonical_digest(actual_ids.tolist()), 'Internal ID hash')
        require(split[f'{name}_class_counts'] == {str(k): int((labels[positions] == k).sum())
            for k in (0, 1)}, 'Internal class balance')
    require(set(split['train_ids']).isdisjoint(split['validation_ids']), 'Internal split overlap')
    require(set(split['train_ids']) | set(split['validation_ids']) == set(expected_ids), 'Internal train coverage')
    require(split['disjoint'] is True and split['only_provided_training_rows'] is True, 'Internal policy')
    rule = d['stopping_rule']
    for name in ('max_epochs', 'patience', 'min_delta', 'validation_fraction'):
        close(rule[name], settings[name], f'Stopping setting {name}')
    curve = d['epoch_curve']
    require(1 <= len(curve) <= settings['max_epochs'], 'Epoch limit')
    require([r['epoch'] for r in curve] == list(range(1, len(curve) + 1)), 'Epoch sequence')
    for row in curve:
        require(all(np.isfinite(row[k]) and row[k] >= 0 for k in
            ('train_log_loss', 'validation_log_loss', 'optimizer_training_objective', 'epoch_seconds')),
            'Epoch curve has invalid value')
    best = min(range(len(curve)), key=lambda i: curve[i]['validation_log_loss']) + 1
    require(d['best_epoch'] == d['selected_epoch'] == d['selected_epochs'] == best,
            'Selected epoch is not earliest strict global minimum')
    close(d['best_internal_validation_log_loss'], curve[best - 1]['validation_log_loss'], 'Selected epoch loss')
    anchor, stagnant, stopping_epoch = float('inf'), 0, None
    for row in curve:
        if row['validation_log_loss'] < anchor - settings['min_delta']:
            anchor, stagnant = row['validation_log_loss'], 0
        else:
            stagnant += 1
        if stagnant >= settings['patience']:
            stopping_epoch = row['epoch']
            break
    if stopping_epoch is not None:
        require(stopping_epoch == len(curve)
                and d['stop_reason'] == 'internal_validation_log_loss_patience', 'Patience stopping rule')
    else:
        require(len(curve) == settings['max_epochs'] and d['stop_reason'] == 'max_epochs', 'Max epoch rule')
    for key, epochs, rows in [('inner_optimizer', len(curve), len(inner)),
                               ('final_optimizer', best, len(expected_ids))]:
        optimizer = d[key]
        require(optimizer['raw_n_iter'] == 1, 'Raw sklearn partial_fit n_iter changed')
        require(optimizer['epochs_completed'] == optimizer['loss_curve_length'] == epochs, 'Epoch accounting')
        require(optimizer['samples_seen'] == epochs * rows, 'Optimizer samples_seen accounting')
    require(len(d['final_optimizer']['loss_curve']) == best, 'Final loss curve length')
    close(d['final_optimizer']['loss_curve'][-1], d['final_optimizer']['final_training_objective'], 'Final loss objective')
    close(curve[-1]['optimizer_training_objective'], d['inner_optimizer']['final_training_objective'], 'Inner final objective')
    require(d['optimizer_fits'] == 2 and d['partial_fit_calls'] == len(curve) + best, 'Optimizer trajectory budget')
    for key, rows in [('inner_preprocessing', len(inner)), ('final_preprocessing', len(expected_ids))]:
        state = d[key].copy()
        digest = state.pop('sha256')
        require(canonical_digest(state) == digest, f'{key}: state hash')
        require(np.all(np.asarray(state['scaler_n_samples_seen']) == rows), f'{key}: fitted row count')
        for name in ('medians', 'scaler_mean', 'scaler_variance', 'scaler_scale'):
            require(len(state[name]) == len(columns['numeric']) and np.isfinite(state[name]).all(), f'{key}: {name}')
        require(len(state['categories']) == len(columns['categorical']), f'{key}: category feature count')
        require(state['encoded_features'] >= len(columns['numeric']), f'{key}: transformed feature count')
    policy = d['training_policy']
    for key in ('no_outer_validation_or_test_in_fit', 'preprocess_fit_inside_internal_train',
                'fresh_preprocessor_for_full_train', 'fresh_optimizer_for_full_train',
                'internal_selection_model_discarded', 'clipping_bounds_fixed_not_fitted',
                'indicators_and_categories_not_clipped'):
        require(policy[key] is True, f'Training policy {key}')
    require(policy['class_resampling'] is False and policy['threads'] == settings['threads']
            and policy['numeric_values_clip_z'] == variant['clip_z'], 'Training policy values')
    require(isinstance(d['warnings'], list) and all(w['count'] >= 1 for w in d['warnings']), 'Warnings structure')
    return dict(inner_epochs=len(curve), selected_epochs=best, partial_fit_calls=d['partial_fit_calls'],
                warning_count=sum(w['count'] for w in d['warnings']))


def validate(project, root, expected_fingerprint):
    begin = time.perf_counter()
    plan = read_json(root / 'plan.json')
    fingerprint = plan['fingerprint']
    require(fingerprint == expected_fingerprint, 'Unexpected predeclared plan fingerprint')
    payload = {k: v for k, v in plan.items() if k != 'fingerprint'}
    require(hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
        indent=2, allow_nan=False).encode()).hexdigest() == fingerprint, 'Plan content fingerprint')
    require(plan['levels'] == [1, 2, 3, 4] and len(plan['variants']) == 4, 'Experiment design')
    for name, digest in plan['code_sha256'].items():
        require(sha256(project / 'src' / name) == digest, f'Frozen source changed: {name}')
    data = Path(plan['data_root'])
    for relative, info in plan['inputs'].items():
        path = data.parent / relative
        require(path.stat().st_size == info['bytes'] and sha256(path) == info['sha256'], f'Frozen input changed: {relative}')
    for name in ('cohort_ids.parquet', 'cohort_targets.parquet', 'cv_folds.parquet'):
        require(sha256(data / name) == sha256(Path(plan['original_pilot']) / name), f'Original {name} differs')
    ids = pd.read_parquet(data / 'cohort_ids.parquet').SK_ID_CURR.to_numpy()
    targets = pd.read_parquet(data / 'cohort_targets.parquet')
    folds = pd.read_parquet(data / 'cv_folds.parquet')
    columns = read_json(data / 'columns.json')
    require(len(ids) == len(np.unique(ids)) == 20000, 'Cohort rows/uniqueness')
    aligned(ids, targets.SK_ID_CURR, 'Cohort TARGET IDs')
    aligned(ids, folds.SK_ID_CURR, 'Cohort CV IDs')
    labels = targets.TARGET.to_numpy()
    require(targets.TARGET.value_counts().to_dict() == {0: 18385, 1: 1615}, 'Natural class balance')
    prepared = Path(plan['prepared_dir'])
    prepared_inventory = {path.relative_to(prepared).as_posix(): path
        for path in prepared.rglob('*') if path.is_file() and 'duckdb_tmp' not in path.parts}
    require(set(prepared_inventory) == {r['file'] for r in plan['prepared_inputs']},
            'Prepared file inventory changed')
    for info in plan['prepared_inputs']:
        path = prepared_inventory[info['file']]
        require(path.stat().st_size == info['bytes'] and sha256(path) == info['sha256'],
                f'Prepared file changed: {info["file"]}')
    all_targets = pd.read_parquet(prepared / 'target.parquet').set_index('SK_ID_CURR').TARGET
    aligned(all_targets.loc[ids], labels, 'Prepared TARGET alignment')
    development = pd.read_parquet(prepared / 'splits/development_ids.parquet').SK_ID_CURR.to_numpy()
    test = pd.read_parquet(prepared / 'splits/test_ids.parquet').SK_ID_CURR.to_numpy()
    require(len(test) == len(np.unique(test)) == 61503 and set(ids) <= set(development), 'Original train/test sizes')
    require(set(ids).isdisjoint(test) and set(development).isdisjoint(test), 'Train/test overlap')
    aligned(test, pd.read_parquet(Path(plan['original_pilot']) / 'test_ids.parquet').SK_ID_CURR, 'Original pilot test IDs')
    cv = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=42).split(ids, labels))
    require(set(folds.validation_fold) == {0, 1, 2}, 'CV fold labels')
    for fold, (_, valid) in enumerate(cv):
        aligned(np.flatnonzero(folds.validation_fold.to_numpy() == fold), valid, 'Saved CV exact reproduction')
    expected_cv_paths = {root / 'cv' / f'level_{level}' / variant['name'] / f'fold_{fold}' / 'complete.json'
        for level in plan['levels'] for variant in plan['variants'] for fold in range(3)}
    require(set((root / 'cv').rglob('complete.json')) == expected_cv_paths, 'CV missing/extra checkpoints')
    records, diagnostics_summary, completed_times = [], [], []
    for level in plan['levels']:
        for variant in plan['variants']:
            for fold, (train, valid) in enumerate(cv):
                stage = root / 'cv' / f'level_{level}' / variant['name'] / f'fold_{fold}'
                marker = validate_checkpoint(stage, fingerprint,
                    ['result.json', 'fit_diagnostics.json', 'validation_probabilities.parquet'])
                completed_times.append(marker['completed_utc'])
                r = read_json(stage / 'result.json')
                require((r['level'], r['variant'], r['fold']) == (level, variant['name'], fold), 'CV identity')
                require(r['train_rows'] == len(train) and r['validation_rows'] == len(valid)
                        and r['test_used'] is False, 'CV row count/policy')
                _, recomputed = validate_probabilities(stage / 'validation_probabilities.parquet', ids[valid], labels[valid])
                for old, new in [('log_loss', 'log_loss'), ('ap', 'pr_auc_average_precision'),
                                 ('roc_auc', 'roc_auc'), ('brier', 'brier_score')]:
                    close(r[old], recomputed[new], f'CV {level}/{variant["name"]}/{fold}/{old}')
                d = read_json(stage / 'fit_diagnostics.json')
                ds = validate_diagnostics(d, ids[train], labels[train], columns[str(level)], variant, plan['settings'])
                require(r['best_epoch'] == r['full_train_epochs'] == ds['selected_epochs']
                        and r['inner_epochs'] == ds['inner_epochs'] and r['optimizer_trajectories'] == 2
                        and r['partial_fit_calls'] == ds['partial_fit_calls'], 'CV diagnostic summary')
                close(r['fit_seconds'], d['total_seconds'], 'CV fit duration')
                diagnostics_summary.append(dict(stage='cv', level=level, variant=variant['name'], fold=fold, **ds))
                records.append(r)
    frame = pd.DataFrame(records)
    csv = pd.read_csv(root / 'cv_results.csv')
    require(len(csv) == 48 and not csv.duplicated(['level', 'variant', 'fold']).any(), 'CV CSV uniqueness')
    pd.testing.assert_frame_equal(frame.sort_index(axis=1), csv.sort_index(axis=1),
        check_dtype=False, check_exact=False, rtol=2e-11, atol=2e-12)
    combined_path = project / 'reports/training/pilot_20k_20config_combined/all_1680_cv_folds.csv'
    require(sha256(combined_path) == plan['input_selection_sha256'], 'Baseline CV source changed')
    baseline = pd.read_csv(combined_path)
    baseline = baseline[(baseline.model == 'mlp') & (baseline.candidate_number == 13)]
    require(len(baseline) == 12 and not baseline.duplicated(['level', 'fold']).any(), 'Baseline MLP CV uniqueness')
    for r in plan['original_cv_scores']:
        row = baseline[(baseline.level == r['level']) & (baseline.fold == r['fold'])].iloc[0]
        close(r['log_loss'], row.log_loss, 'Original CV baseline')
    prior = float(np.mean([log_loss(labels[v], np.full(len(v), labels[t].mean()), labels=[0, 1]) for t, v in cv]))
    selection = read_json(root / 'selection.json')
    require(selection['plan_fingerprint'] == fingerprint and selection['selected_after_completed_cv'] == 48
        and selection['test_used_for_selection'] is False
        and selection['all_level_selections_frozen_before_test'] is True, 'Frozen selection policy')
    require([r['level'] for r in selection['selections']] == plan['levels'], 'Frozen selection level coverage')
    candidate_csv = pd.read_csv(root / 'candidate_summary.csv')
    require(len(candidate_csv) == 16 and not candidate_csv.duplicated(['level', 'variant']).any(), 'Candidate CSV uniqueness')
    eligible, chosen_summary = [], []
    for chosen in selection['selections']:
        level = chosen['level']
        means = []
        for order, variant in enumerate(plan['variants']):
            sub = frame[(frame.level == level) & (frame.variant == variant['name'])]
            mean = float(sub.log_loss.mean())
            means.append((mean, order))
            saved = candidate_csv[(candidate_csv.level == level) & (candidate_csv.variant == variant['name'])].iloc[0]
            require(saved.variant_order == order, 'Candidate tie order')
            for key, value in [('mean_log_loss', mean), ('std_log_loss', sub.log_loss.std(ddof=0)),
                               ('mean_ap', sub.ap.mean()), ('mean_best_epoch', sub.best_epoch.mean())]:
                close(saved[key], float(value), f'Candidate summary {key}')
        best_mean, order = min(means)
        variant = plan['variants'][order]
        require(chosen['variant'] == variant['name'] and chosen['params'] == variant, 'Selected candidate not strict ordered minimum')
        old = baseline[baseline.level == level].set_index('fold').log_loss
        sub = frame[(frame.level == level) & (frame.variant == variant['name'])]
        gate = (best_mean < old.mean() - 1e-6 and best_mean < prior - 1e-6
                and all(r.log_loss < old.loc[r.fold] for r in sub.itertuples()))
        require(chosen['eligible_for_test'] is bool(gate), 'Eligibility gate mismatch')
        for key, value in [('mean_cv_log_loss', best_mean), ('baseline_cv_log_loss', old.mean()),
                           ('constant_prior_cv_log_loss', prior)]:
            close(chosen[key], float(value), f'Selection {key}')
        chosen_summary.append(chosen)
        if gate:
            eligible.append(level)
    expected_final = {root / 'final' / f'level_{level}' / 'complete.json' for level in eligible}
    require(set((root / 'final').glob('level_*/complete.json')) == expected_final, 'Final eligible checkpoint coverage')
    final_metrics = []
    for level in eligible:
        stage = root / 'final' / f'level_{level}'
        validate_checkpoint(stage, fingerprint, ['metrics.json', 'probabilities.parquet', 'pipeline.joblib',
            'fit_diagnostics.json', 'fit_checkpoint/complete.json', 'fit_checkpoint/pipeline.joblib',
            'fit_checkpoint/fit_diagnostics.json'])
        validate_checkpoint(stage / 'fit_checkpoint', fingerprint, ['pipeline.joblib', 'fit_diagnostics.json'])
        for name in ('pipeline.joblib', 'fit_diagnostics.json'):
            require(sha256(stage / name) == sha256(stage / 'fit_checkpoint' / name), 'Final/fit copy mismatch')
        record = read_json(stage / 'metrics.json')
        chosen = next(r for r in selection['selections'] if r['level'] == level)
        require(record['cv_selection'] == chosen and record['train_rows'] == 20000 and record['test_rows'] == 61503
            and record['scope'] == plan['scope'] and record['variant'] == chosen['variant']
            and record['test_used_for_selection'] is False and record['original_result_replaced'] is False, 'Final scope/policy')
        prob, recomputed = validate_probabilities(stage / 'probabilities.parquet', test, all_targets.loc[test].to_numpy())
        require((prob.scope == plan['scope']).all() and (prob.model == 'mlp').all() and (prob.level == level).all(), 'Probability scope')
        require(set(record['metrics']) == set(recomputed), 'Final metric coverage')
        for name, value in recomputed.items():
            close(record['metrics'][name], value, f'Final L{level} {name}')
        old = read_json(Path(plan['original_final_root']) / 'mlp' / f'level_{level}' / 'metrics.json')
        require(record['baseline_metrics'] == old['metrics'], 'Original final metric comparison')
        d = read_json(stage / 'fit_diagnostics.json')
        ds = validate_diagnostics(d, ids, labels, columns[str(level)], chosen['params'], plan['settings'])
        require(record['best_epoch'] == ds['selected_epochs'], 'Final selected epoch')
        close(record['fit_seconds'], d['total_seconds'], 'Final duration')
        diagnostics_summary.append(dict(stage='final', level=level, variant=chosen['variant'], **ds))
        final_metrics.append(dict(level=level, variant=chosen['variant'], **recomputed))
    events = [json.loads(line) for line in (root / 'events.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    freezes = [e for e in events if e['event'] == 'all_level_selections_frozen']
    require(freezes and all(e['selection_sha256'] == sha256(root / 'selection.json') for e in freezes), 'Freeze event SHA')
    first_freeze = min(e['utc'] for e in freezes)
    require(all(t <= first_freeze for t in completed_times), 'Selection event precedes completed CV')
    require(all(e['utc'] >= first_freeze for e in events if e['event'].startswith('repair_final_')), 'Final stage before frozen selection')
    before = read_json(root / 'protected_before.json')
    actual_files = {path.relative_to(project).as_posix(): path for relative in PROTECTED_ROOTS
        for path in (project / relative).rglob('*') if path.is_file() and '__pycache__' not in path.parts}
    require(set(actual_files) == set(before), 'Protected original file inventory changed')
    for relative, info in before.items():
        path = actual_files[relative]
        require(path.stat().st_size == info['bytes'] and sha256(path) == info['sha256'], f'Protected original changed: {relative}')
    integrity = read_json(root / 'integrity_validation.json')
    require(integrity['passed'] is True and integrity['original_results_unchanged'] is True
            and integrity['differences'] == [] and integrity['checked_files'] == len(before), 'Original integrity report')
    summary = read_json(root / 'summary.json')
    require(summary['status'] == 'completed' and summary['completed_cv'] == 48
            and summary['outer_cv_evaluations'] == 48 and summary['cv_optimizer_trajectories'] == 96
            and summary['final_levels_completed'] == len(eligible) and summary['eligible_levels'] == eligible
            and summary['final_optimizer_trajectories'] == 2 * len(eligible)
            and summary['original_results_unchanged'] is True, 'Final summary budget')
    return dict(passed=True, scope=plan['scope'], plan_fingerprint=fingerprint,
        audit_utc=datetime.now(timezone.utc).isoformat(), read_only=True,
        no_fit_no_prediction_no_bootstrap=True, model_unpickling=False,
        completed_cv=48, optimizer_trajectories=96 + 2 * len(eligible),
        final_levels=eligible, protected_files_checked=len(before),
        prepared_files_checked=len(prepared_inventory),
        train_rows=len(ids), test_rows=len(test), class_counts=plan['class_counts'],
        selections=chosen_summary, final_metrics=final_metrics,
        total_partial_fit_calls=sum(d['partial_fit_calls'] for d in diagnostics_summary),
        diagnostics=diagnostics_summary, audit_seconds=time.perf_counter() - begin,
        limits=['Artifact consistency is verified; training is not replayed.',
                'Saved Pipeline bytes are hashed, not unpickled; saved probabilities are not regenerated.',
                'This remains a post-hoc supplemental experiment after earlier test access.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--root', type=Path, help='Defaults to PROJECT/artifacts/training/mlp_repair_20k_v1')
    parser.add_argument('--expected-fingerprint', default=EXPECTED_FINGERPRINT)
    parser.add_argument('--output', type=Path, help='Optional new JSON audit report; experiment files remain read-only')
    args = parser.parse_args(argv)
    project = args.project.resolve()
    root = args.root.resolve() if args.root else project / 'artifacts/training/mlp_repair_20k_v1'
    try:
        report = validate(project, root, args.expected_fingerprint)
    except Exception as exc:
        print(f'VALIDATION FAILED: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    if args.output:
        output = args.output.resolve()
        require(output not in {root / name for name in ('plan.json', 'selection.json', 'summary.json',
            'protected_before.json', 'integrity_validation.json', 'progress.json')}, 'Refuse experiment overwrite')
        require(not output.exists(), 'Audit output already exists; choose a new report path')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('diagnostics', 'final_metrics', 'selections')},
        ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
