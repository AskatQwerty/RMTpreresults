"""Evaluate 28 final refits selected from the completed 20-configuration pilot.

Reads saved test probabilities only. No model, prepared X, fit or predict is
loaded here. Bootstrap uses the original mathematical implementation with a
new immutable analysis fingerprint and separate resumable output directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .pilot_evaluation import (MODELS, LEVELS, CI_LABEL, quality_point,
                               summarize_bootstrap, csv_write)
from .pilot_evaluation_bootstrap import sha256, atomic_json, bootstrap_run
from .pilot_evaluation_decisions import (DecisionEvaluator, calibration_point_table,
                                        THRESHOLDS)
from .pilot_delong import paired_delong_table, DELONG_METADATA

SCOPE = 'pilot_20k_20config_final'
SOURCE_FILES = ('final_pilot_evaluation.py', 'final_pilot_reporting.py',
                'pilot_evaluation.py', 'pilot_evaluation_bootstrap.py',
                'pilot_evaluation_decisions.py', 'pilot_delong.py')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_file(base, relative, checksum):
    path = (Path(base) / relative).resolve()
    require(path.is_relative_to(Path(base).resolve()), f'Artifact outside root: {relative}')
    require(path.is_file() and sha256(path) == checksum, f'Artifact SHA mismatch: {path}')
    return path


def validate_output_path(output, previous, root):
    output, previous, root = map(lambda p: Path(p).resolve(), (output, previous, root))
    require(output != previous and not output.is_relative_to(previous)
            and not previous.is_relative_to(output), 'Keep all previous reports separate and unchanged')
    require(output != root and not output.is_relative_to(root), 'Evaluation must not overwrite training artifacts')
    require(output.name == 'evaluation' and output.parent.name == SCOPE,
            f'Use reports/{SCOPE}/evaluation for this analysis')


def align_prediction_frame(aligned, frame, model, level):
    """ID-based alignment, never positional reuse or silent TARGET mismatch."""
    require(frame.SK_ID_CURR.is_unique and len(frame) == len(aligned), 'Prediction ID count/uniqueness')
    require(set(frame.SK_ID_CURR) == set(aligned.SK_ID_CURR), 'Prediction client set differs')
    for field, value in [('scope', SCOPE), ('model', model), ('level', level), ('evaluation_split', 'test')]:
        require(frame[field].eq(value).all(), f'Wrong prediction provenance: {model}/L{level}/{field}')
    joined = aligned[['SK_ID_CURR', 'TARGET']].merge(
        frame[['SK_ID_CURR', 'TARGET', 'probability_default']], on='SK_ID_CURR',
        how='left', validate='one_to_one', suffixes=('_original', '_saved'), indicator=True)
    require(joined['_merge'].eq('both').all() and np.array_equal(joined.TARGET_original, joined.TARGET_saved),
            'Prediction TARGET does not match original target by ID')
    p = joined.probability_default.to_numpy(dtype=np.float64)
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'Invalid probability')
    return p


def immutable_snapshot(prepared, root, previous):
    """Hash immutable inputs, including ALL old reports, excluding live new logs."""
    files = {}
    for name in ('target.parquet', 'fairness.parquet', 'splits/test_ids.parquet'):
        files['prepared/' + name] = sha256(Path(prepared) / name)
    names = ['experiment_manifest.json', 'cohort_contract.json', 'cohort_summary.json',
             'cohort_ids.parquet', 'cohort_targets.parquet', 'cv_folds.parquet', 'test_ids.parquet']
    for name in names:
        files['training/' + name] = sha256(Path(root) / name)
    for model in MODELS:
        for level in LEVELS:
            stage = Path(root) / model / f'level_{level}'
            marker = read_json(stage / 'complete.json')
            files[f'training/{model}/level_{level}/complete.json'] = sha256(stage / 'complete.json')
            for name, digest in marker['artifact_sha256'].items():
                path = checked_file(stage, name, digest)
                files[f'training/{model}/level_{level}/{name}'] = sha256(path)
    for path in sorted(Path(previous).rglob('*')):
        if path.is_file():
            files['previous_reports/' + path.relative_to(previous).as_posix()] = sha256(path)
    return files


def load_final_predictions(prepared, root):
    prepared, root = Path(prepared), Path(root)
    manifest = read_json(root / 'experiment_manifest.json')
    require(manifest.get('scope') == SCOPE and manifest.get('train_rows') == 20000
            and manifest.get('test_rows') == 61503, 'Wrong final training manifest')
    require(manifest.get('positive_class') == 1, 'Positive-class provenance missing')
    source = checked_file(root, 'source_snapshot/model_training.py', manifest['code_sha256']['model_training.py'])
    require('probabilities[:,int(np.flatnonzero(classes==1)[0])]' in source.read_text(encoding='utf-8'),
            'Positive-class implementation requires manual review')
    hashes = {r['file']: r['sha256'] for r in manifest['inputs']}
    for name in ('target.parquet', 'fairness.parquet', 'splits/test_ids.parquet'):
        checked_file(prepared, name, hashes[name])
    contract = read_json(root / 'cohort_contract.json')
    for name, checksum in contract['artifact_sha256'].items():
        checked_file(root, name, checksum)
    test = pd.read_parquet(root / 'test_ids.parquet')
    original_test = pd.read_parquet(prepared / 'splits/test_ids.parquet')
    require(test.SK_ID_CURR.is_unique and np.array_equal(test.SK_ID_CURR, original_test.SK_ID_CURR),
            'Original test IDs or ordering changed')
    train = pd.read_parquet(root / 'cohort_ids.parquet')
    require(len(train) == train.SK_ID_CURR.nunique() == 20000
            and set(train.SK_ID_CURR).isdisjoint(test.SK_ID_CURR), 'Train/test overlap or cohort changed')
    target = pd.read_parquet(prepared / 'target.parquet')
    fairness = pd.read_parquet(prepared / 'fairness.parquet')
    require(target.SK_ID_CURR.is_unique and fairness.SK_ID_CURR.is_unique, 'Duplicate TARGET/fairness IDs')
    aligned = test.sort_values('SK_ID_CURR').merge(target, on='SK_ID_CURR', how='left',
        validate='one_to_one', indicator=True)
    require(aligned['_merge'].eq('both').all() and aligned.TARGET.isin([0, 1]).all(), 'Missing test target')
    aligned = aligned.drop(columns='_merge').merge(fairness, on='SK_ID_CURR', how='left',
        validate='one_to_one', indicator=True)
    require(len(aligned) == 61503 and aligned['_merge'].eq('both').all(), 'Missing fairness/test row')
    aligned = aligned.drop(columns='_merge')
    gender = aligned.CODE_GENDER.where(aligned.CODE_GENDER.isin(['F', 'M']), 'unknown')
    ages = pd.to_numeric(aligned.AGE_YEARS, errors='coerce').to_numpy(dtype=float)
    expected_age = np.where(np.isfinite(ages), np.where(ages < 25, '<25', '>=25'), 'unknown')
    require(np.array_equal(gender, aligned.GENDER_GROUP)
            and np.array_equal(expected_age, aligned.AGE_GROUP), 'Saved fairness definitions changed')
    groups = {'gender': gender.to_numpy(dtype=str), 'age': aligned.AGE_GROUP.to_numpy(dtype=str)}
    y = aligned.TARGET.to_numpy(dtype=np.int8)
    pairs, probabilities, validations, metric_records = [], [], [], []
    for model in MODELS:
        for level in LEVELS:
            stage = root / model / f'level_{level}'
            marker = read_json(stage / 'complete.json')
            require(marker['fingerprint'] == manifest['fingerprint'], 'Wrong final checkpoint fingerprint')
            for name in ('probabilities.parquet', 'metrics.json'):
                checked_file(stage, name, marker['artifact_sha256'][name])
            frame = pd.read_parquet(stage / 'probabilities.parquet')
            probabilities.append(align_prediction_frame(aligned, frame, model, level))
            pairs.append({'model': model, 'level': level})
            metric_records.append(read_json(stage / 'metrics.json'))
            validations.append({'model': model, 'level': level, 'same_ids': True,
                                'target_matches_after_join': True, 'probability_range_valid': True})
    validation = {'scope': SCOPE, 'all_28_aligned': len(pairs) == 28, 'test_rows': len(y),
        'test_positives': int(y.sum()), 'positive_class': 1,
        'positive_class_evidence': manifest.get('probability_column_selection'),
        'canonical_order': 'ascending SK_ID_CURR; explicit one-to-one merge',
        'unknown_policy': 'kept in all quality/profit; excluded only from binary group gaps',
        'unknown_group_counts': {k: int((v == 'unknown').sum()) for k, v in groups.items()},
        'checks': validations, 'errors': []}
    return aligned, y, np.stack(probabilities), pairs, groups, validation, metric_records


def verify_saved_metrics(point, records):
    mapping = {'ap': 'pr_auc_average_precision', 'roc_auc': 'roc_auc',
               'log_loss': 'log_loss', 'brier': 'brier_score'}
    rows = []
    for record in records:
        match = point[(point.model == record['model']) & (point.level == record['level'])]
        require(len(match) == 1 and record['scope'] == SCOPE, 'Unexpected saved final metrics')
        for metric, name in mapping.items():
            actual, saved = float(match.iloc[0][metric]), float(record['metrics'][name])
            rows.append(dict(model=record['model'], level=record['level'], metric=metric,
                             recomputed=actual, saved_final_fit=saved,
                             within_tolerance=bool(np.isclose(actual, saved, atol=1e-12, rtol=1e-10))))
    frame = pd.DataFrame(rows)
    require(len(frame) == 112 and frame.within_tolerance.all(), 'Final metrics disagree with saved probabilities')
    return frame


def export_distributions(arrays, pairs, output, n):
    """Export full quality/profit/threshold distributions without recomputation."""
    from .pilot_evaluation_decisions import SCENARIOS, SCENARIO_NAMES
    quality, profits, errors = [], [], []
    for r in range(n):
        for i, pair in enumerate(pairs):
            row = {'replicate': r, **pair, 'ap': arrays['ap'][r, i]}
            if pair['level'] == 4:
                m = MODELS.index(pair['model'])
                row.update({metric: arrays[key][r, m] for metric, key in
                            [('roc_auc','roc_auc_l4'),('log_loss','log_loss_l4'),('brier','brier_l4')]})
            quality.append(row)
        for m, model in enumerate(MODELS):
            for s, (_, t) in enumerate(SCENARIOS):
                profits.append(dict(replicate=r, model=model, scenario=SCENARIO_NAMES[s], threshold=THRESHOLDS[t],
                    realized_profit_per_applicant=arrays['decision_profit'][r, m, s],
                    approval_rate=arrays['decision_approval_rate'][r, m, t]))
            for t, threshold in enumerate(THRESHOLDS):
                errors.append(dict(replicate=r, model=model, threshold=threshold,
                    **{k: arrays['decision_confusion'][r, m, t, j] for j, k in enumerate(('TN','FP','FN','TP'))},
                    **{k: arrays['decision_' + k][r, m, t] for k in ('precision','recall','f1')}))
    for name, rows in [('quality', quality), ('profit', profits), ('threshold', errors)]:
        csv_write(pd.DataFrame(rows), output / f'bootstrap_{name}_distribution.csv')


def run_final_evaluation(prepared_dir, pilot_root, previous_reports, output_dir, *, n_bootstrap=1000, seed=42):
    started = time.perf_counter()
    prepared, root, previous, output = map(Path, (prepared_dir, pilot_root, previous_reports, output_dir))
    validate_output_path(output, previous, root)
    require(n_bootstrap == 1000 and seed == 42, 'Fixed evaluation: 1000 paired bootstrap, seed42')
    output.mkdir(parents=True, exist_ok=True)
    before = immutable_snapshot(prepared, root, previous)
    before_file = output / 'integrity_before.json'
    if before_file.exists():
        require(read_json(before_file)['sha256'] == before, 'Inputs changed since analysis initialization')
    else:
        atomic_json(before_file, {'sha256': before})
    aligned, y, probabilities, pairs, groups, validation, metrics = load_final_predictions(prepared, root)
    atomic_json(output / 'alignment_validation.json', validation)
    point = quality_point(y, probabilities, pairs)
    csv_write(verify_saved_metrics(point, metrics), output / 'saved_final_metrics_comparison.csv')
    csv_write(point, output / 'quality_point.csv')
    canonical = output / 'aligned_test_ids.parquet'
    if canonical.exists():
        require(np.array_equal(pd.read_parquet(canonical).SK_ID_CURR, aligned.SK_ID_CURR), 'Canonical IDs changed')
    else:
        aligned[['SK_ID_CURR','TARGET','GENDER_GROUP','AGE_GROUP']].to_parquet(canonical, index=False)
    probdict = {pair['model']: probabilities[i] for i, pair in enumerate(pairs) if pair['level'] == 4}
    evaluator = DecisionEvaluator(y, probdict, groups)
    decisions = evaluator.point_tables()
    for name, table in decisions.items():
        csv_write(table, output / f'{"profit_point" if name == "profit" else name}.csv')
    csv_write(calibration_point_table(y, probdict), output / 'calibration_bins.csv')
    csv_write(paired_delong_table(y, probdict), output / 'delong.csv')
    atomic_json(output / 'delong_methodology.json', DELONG_METADATA)
    context = dict(scope=SCOPE, analysis='Final 20-configuration pilot RQ1/RQ2',
        training_fingerprint=read_json(root / 'experiment_manifest.json')['fingerprint'], source_sha256=before,
        aligned_id_sha256=sha256(canonical), pairs=pairs,
        analysis_code_sha256={name: sha256(Path(__file__).parent / name) for name in SOURCE_FILES},
        n_bootstrap=n_bootstrap, seed=seed, resampling_unit='client',
        resampling='ordinary with replacement; common integer client multiplicities for all28 models/levels',
        rng='numpy.default_rng(42)/PCG64; saved state per replicate', interval=CI_LABEL,
        percentile_quantiles=[.025, .975], quantile_method='linear',
        ap_definition='sklearn average_precision_score(pos_label=1)', thresholds=list(THRESHOLDS),
        decision_coordinates=evaluator.coordinates, selection_used_test=False,
        selected_from_completed_cv_fits=1680, selected_configurations_per_pair=20,
        final_fit_count=28, repeats_per_pair=1, original_preresults_preserved_separately=True,
        bonferroni_family_size=6, new_fit_calls=0, new_predict_calls=0,
        threshold_optimization=False, retraining_variability_estimated=False,
        formal_A2_AP_noninferiority_confirmed=False,
        analysis_specification='Post-training evaluation; not claimed preregistered',
        software={name: importlib.metadata.version(name) for name in ['numpy','scipy','pandas','scikit-learn']})
    context['fingerprint'] = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
    manifest_path = output / 'evaluation_manifest.json'
    if manifest_path.exists():
        require(read_json(manifest_path)['fingerprint'] == context['fingerprint'],
                'Analysis changed; cannot reuse existing bootstrap')
    else:
        atomic_json(manifest_path, context)
    print('Final predictions aligned: 28 pairs, 61503 clients; separate 1000 paired bootstrap.', flush=True)
    arrays, progress = bootstrap_run(y, probabilities, pairs, groups, output / 'bootstrap',
        fingerprint=context['fingerprint'], n_bootstrap=n_bootstrap, seed=seed)
    require(progress['completed'] == n_bootstrap, 'Bootstrap incomplete; checkpoints retained')
    tables = summarize_bootstrap(arrays, point, pairs, decisions)
    for name, table in tables.items():
        csv_write(table, output / f'{name}.csv')
    export_distributions(arrays, pairs, output, n_bootstrap)
    # This comparison is descriptive: original and final models have different searches.
    old = pd.read_csv(previous / 'evaluation/quality_point.csv')
    compare = point.merge(old, on=['model','level'], validate='one_to_one', suffixes=('_20config', '_5config'))
    require(len(compare) == 28, 'Old preresults must be separately identifiable for all28 pairs')
    for name in ('ap','roc_auc','log_loss','brier'):
        compare['delta_' + name] = compare[name + '_20config'] - compare[name + '_5config']
    csv_write(compare, output / 'comparison_to_preserved_5config_preresults.csv')
    after = immutable_snapshot(prepared, root, previous)
    differences = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    atomic_json(output / 'integrity_after.json', dict(unchanged=not differences,
        checked_files=len(before), mismatches=differences, old_preresults_unchanged=not any(k.startswith('previous_reports/') for k in differences)))
    require(not differences, 'Immutable inputs or old preresults changed during analysis')
    summary = dict(status='completed', scope=SCOPE, completed_bootstrap=1000, expected_bootstrap=1000,
        seed=seed, model_count=7, pair_count=28, test_rows=len(y), input_integrity_unchanged=True,
        alignment_verified=True, saved_final_metrics_match=True, old_preresults_unchanged=True,
        calibration_recomputed_from_new_probabilities=True, new_fit_calls=0, new_predict_calls=0,
        evaluation_elapsed_seconds=time.perf_counter()-started, bootstrap_new=progress['new'],
        bootstrap_reused=progress['reused'], completed_utc=datetime.now(timezone.utc).isoformat(),
        intervals=CI_LABEL, checkpoint_count=len(list((output / 'bootstrap').glob('replicate_*.npz'))))
    atomic_json(output / 'evaluation_summary.json', summary)
    from .final_pilot_reporting import export_final_report
    export_final_report(root, output, previous_reports=previous)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', default='data/processed/full')
    parser.add_argument('--root', default='artifacts/training/' + SCOPE)
    parser.add_argument('--previous', default='reports/pilot_20k_5config')
    parser.add_argument('--output', default='reports/' + SCOPE + '/evaluation')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run_final_evaluation(args.prepared, args.root, args.previous, args.output)


if __name__ == '__main__':
    main()
