"""Predeclared supplemental MLP experiment. Original results are immutable.

48 outer CV evaluations, each with internal train-only log-loss epoch selection
and a fresh outer-train fit. Freeze all four level selections before test use.
No new cohort, outer split, class balancing, or changes to prepared X.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

from .data_quality import artifact_manifest
from .model_training import (PreparedData, PACKAGES, atomic_json, read_json, sha256, _json_text,
    _atomic_model, _atomic_parquet, _complete, _mark_complete, _events, _failure,
    _warnings_report, classification_metrics, positive_probability)
from .mlp_repair import fit_mlp_variant

SCOPE = 'mlp_repair_20k_v1'
VARIANTS = [
    {'name': 'stop_only', 'clip_z': None, 'alpha': 1e-5},
    {'name': 'stop_regularized', 'clip_z': None, 'alpha': .1},
    {'name': 'stop_clipped', 'clip_z': 5., 'alpha': 1e-5},
    {'name': 'stop_clipped_regularized', 'clip_z': 5., 'alpha': .1},
]
SETTINGS = dict(seed=42, max_epochs=120, patience=12, min_delta=1e-4,
                validation_fraction=.15, threads=2)
CODE_FILES = ('mlp_repair.py', 'mlp_repair_experiment.py', 'model_training.py',
              'model_pipelines.py', 'fold_preprocessing.py', 'data_quality.py',
              'data_preparation.py', 'data_audit.py', 'fit_diagnostics.py', 'resource_monitor.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def protected_snapshot(project):
    result = {}
    for relative in ('artifacts/training/pilot_20k_5config', 'reports/pilot_20k_5config',
                     'artifacts/training/pilot_20k_20config_final', 'reports/pilot_20k_20config_final',
                     'reports/training/pilot_20k_20config_combined'):
        for path in sorted((Path(project) / relative).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                result[path.relative_to(project).as_posix()] = {'bytes': path.stat().st_size, 'sha256': sha256(path)}
    return result


def initialize(project_root):
    project = Path(project_root).resolve()
    root = project / 'artifacts/training' / SCOPE
    data_root = project / 'handoff/RMT_batch_11_15/data'
    original_root = project / 'artifacts/training/pilot_20k_5config'
    final_root = project / 'artifacts/training/pilot_20k_20config_final'
    combined = project / 'reports/training/pilot_20k_20config_combined'
    original = read_json(original_root / 'experiment_manifest.json')
    require(read_json(final_root / 'experiment_summary.json')['completed_pairs'] == 28, 'Original final run incomplete')
    package = read_json(data_root.parent / 'package_manifest.json')
    input_files = {e['path']: e for e in package['files'] if e['path'].startswith('data/')}
    for relative, entry in input_files.items():
        path = data_root.parent / relative
        require(path.stat().st_size == entry['bytes'] and sha256(path) == entry['sha256'], 'Compact X changed')
    for name in ('cohort_ids.parquet', 'cohort_targets.parquet', 'cv_folds.parquet'):
        require(sha256(data_root / name) == sha256(original_root / name), 'Cohort/folds changed')
    prepared = project / 'data/processed/full'
    require(artifact_manifest(prepared) == original['inputs'], 'Prepared inputs changed')
    for name in ('model_pipelines.py', 'model_training.py', 'fold_preprocessing.py', 'fit_diagnostics.py'):
        require(sha256(Path(__file__).parent / name) == original['code_sha256'][name], 'Original core changed')
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    require(versions == original['versions'] and platform.python_version() == original['python'], 'Environment changed')
    baseline = pd.read_csv(combined / 'all_1680_cv_folds.csv')
    baseline = baseline[(baseline.model == 'mlp') & (baseline.candidate_number == 13)].copy()
    require(len(baseline) == 12, 'Original selected MLP CV scores missing')
    plan = dict(scope=SCOPE, project_root=str(project), data_root=str(data_root), prepared_dir=str(prepared),
        original_pilot=str(original_root), original_final_root=str(final_root),
        original_pilot_fingerprint=original['fingerprint'],
        original_final_fingerprint=read_json(final_root / 'experiment_manifest.json')['fingerprint'],
        variants=VARIANTS, settings=SETTINGS, levels=[1, 2, 3, 4], train_rows=20000,
        class_counts={'0': 18385, '1': 1615}, test_rows=61503, cv_folds=3,
        selection_metric='mean outer CV log_loss', tie_break='declared variant order',
        expected_outer_cv_evaluations=48, expected_cv_optimizer_trajectories=96,
        max_final_procedures=4, max_total_optimizer_trajectories=104,
        all_level_selections_frozen_before_test=True,
        test_gate='meanCV improves original MLP and constant train-prior by1e-6; all3folds beat original MLP',
        original_mlp_selected_candidate=13,
        original_cv_scores=baseline[['level', 'fold', 'log_loss']].to_dict('records'),
        inputs=input_files, prepared_inputs=original['inputs'], versions=versions,
        python=platform.python_version(),
        code_sha256={name: sha256(Path(__file__).parent / name) for name in CODE_FILES},
        input_selection_sha256=sha256(combined / 'all_1680_cv_folds.csv'),
        fixed_mlp_parameters={'hidden_layer_sizes': [32], 'batch_size': 256,
                              'learning_rate_init': .0003, 'activation': 'relu', 'solver': 'adam'},
        posthoc_supplemental=True, test_already_seen=True, original_results_replaced=False,
        notes=[
            'Inner preprocessing fits only innertrain, internalvalidation selects epochs bylogloss.',
            'Fresh preprocessing/model then fits all providedoutertrain or all20k for selectedepochs.',
            'Clip±5 applies only standardized numericvalues; no rows/rawfeatures are deleted.',
            'No learned bound or rule is chosen using test. Original X/cohort/outerfolds unchanged.',
            'Persistent RNG seeded42 for partial_fit; different shuffle trajectory from oldfit.',
            'Raw n_iter_=1 per partial_fit is preserved; actualepochs recorded separately.',
            'This is an exploratory supplemental experiment, not a replacement of1680CV/main28fits.',
        ])
    plan['fingerprint'] = hashlib.sha256(_json_text(plan).encode()).hexdigest()
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'plan.json').exists():
        require(read_json(root / 'plan.json') == plan, 'Frozen repair plan changed')
    else:
        atomic_json(root / 'plan.json', plan)
    before = root / 'protected_before.json'
    if not before.exists():
        atomic_json(before, protected_snapshot(project))
    return plan


def load_cohort(plan):
    data = Path(plan['data_root'])
    ids = pd.read_parquet(data / 'cohort_ids.parquet').SK_ID_CURR.to_numpy()
    targets = pd.read_parquet(data / 'cohort_targets.parquet')
    folds = pd.read_parquet(data / 'cv_folds.parquet')
    require(len(ids) == len(np.unique(ids)) == 20000, 'Cohort uniqueness')
    require(np.array_equal(ids, targets.SK_ID_CURR) and np.array_equal(ids, folds.SK_ID_CURR), 'Cohort alignment')
    y = pd.Series(targets.TARGET.to_numpy(), index=pd.Index(ids, name='SK_ID_CURR'), name='TARGET')
    require(y.value_counts().to_dict() == {0: 18385, 1: 1615}, 'Natural class balance changed')
    cv = []
    for fold in range(3):
        train = np.flatnonzero(folds.validation_fold.to_numpy() != fold)
        valid = np.flatnonzero(folds.validation_fold.to_numpy() == fold)
        require(len(valid) == (6666 if fold == 2 else 6667), 'Saved folds changed')
        cv.append((train, valid))
    return ids, y, cv, read_json(data / 'columns.json')


def phase(name, **detail):
    if os.environ.get('RMT_PHASE_PATH'):
        atomic_json(os.environ['RMT_PHASE_PATH'], {'phase': name, **detail})


def get_records(root, plan):
    result = []
    for level in plan['levels']:
        for variant in plan['variants']:
            for fold in range(3):
                stage = root / 'cv' / f'level_{level}' / variant['name'] / f'fold_{fold}'
                if _complete(stage, plan):
                    r = read_json(stage / 'result.json')
                    require((r['level'], r['variant'], r['fold']) == (level, variant['name'], fold), 'CV identity')
                    result.append(r)
    return result


def save_progress(root, plan, *, status, current=None):
    rows = get_records(root, plan)
    complete = sum(_complete(root / 'final' / f'level_{l}', plan) for l in plan['levels'])
    progress = dict(status=status, completed_cv=len(rows), expected_cv=48, final_levels_completed=complete,
                    current=current, scope=SCOPE)
    atomic_json(root / 'progress.json', progress)
    if rows:
        pd.DataFrame(rows).to_csv(root / 'cv_results.csv', index=False, encoding='utf-8-sig')
    return progress


def run(project_root):
    project = Path(project_root).resolve()
    root = project / 'artifacts/training' / SCOPE
    plan = read_json(root / 'plan.json')
    for name, digest in plan['code_sha256'].items():
        require(sha256(Path(__file__).parent / name) == digest, 'Frozen execution code changed: ' + name)
    for relative, entry in plan['inputs'].items():
        require(sha256(Path(plan['data_root']).parent / relative) == entry['sha256'], 'CV input changed')
    ids, y, cv, columns = load_cohort(plan)
    started = time.perf_counter()
    try:
        for level in plan['levels']:
            X = pd.read_parquet(Path(plan['data_root']) / f'X_level_{level}.parquet')
            X.index = pd.Index(ids, name='SK_ID_CURR')
            cols = columns[str(level)]
            for variant in plan['variants']:
                for fold, (train, valid) in enumerate(cv):
                    stage = root / 'cv' / f'level_{level}' / variant['name'] / f'fold_{fold}'
                    if _complete(stage, plan):
                        continue
                    stage.mkdir(parents=True, exist_ok=True)
                    current = dict(level=level, variant=variant['name'], fold=fold)
                    save_progress(root, plan, status='cv_fitting', current=current)
                    phase('fit', **current)
                    _events(root, 'repair_cv_started', **current)
                    print(f'REPAIR CV START L{level} {variant["name"]} fold{fold+1}', flush=True)
                    model, diagnostics = fit_mlp_variant(X.iloc[train], y.iloc[train],
                        cols['numeric'], cols['categorical'], variant, **plan['settings'])
                    phase('validation_prediction', **current)
                    begin = time.perf_counter()
                    with threadpool_limits(limits=2):
                        p = positive_probability(model, X.iloc[valid])
                    score_seconds = time.perf_counter() - begin
                    metrics = classification_metrics(y.iloc[valid].to_numpy(), p)
                    r = dict(**current, log_loss=metrics['log_loss'], ap=metrics['pr_auc_average_precision'],
                        roc_auc=metrics['roc_auc'], brier=metrics['brier_score'],
                        train_rows=len(train), validation_rows=len(valid),
                        fit_seconds=diagnostics['total_seconds'], score_seconds=score_seconds,
                        best_epoch=diagnostics['best_epoch'],
                        inner_epochs=diagnostics['inner_optimizer']['epochs_completed'],
                        full_train_epochs=diagnostics['final_optimizer']['epochs_completed'],
                        optimizer_trajectories=2, partial_fit_calls=diagnostics['partial_fit_calls'], test_used=False)
                    _atomic_parquet(pd.DataFrame({'SK_ID_CURR': ids[valid], 'TARGET': y.iloc[valid].to_numpy(),
                        'probability_default': p}), stage / 'validation_probabilities.parquet')
                    atomic_json(stage / 'result.json', r)
                    atomic_json(stage / 'fit_diagnostics.json', diagnostics)
                    _mark_complete(stage, plan, ['result.json', 'fit_diagnostics.json', 'validation_probabilities.parquet'])
                    _events(root, 'repair_cv_completed', **r)
                    print(f'REPAIR CV END L{level} {variant["name"]} fold{fold+1}: LL={r["log_loss"]:.6f}, epochs={r["best_epoch"]}', flush=True)
                    del model, diagnostics, p
                    gc.collect()
            del X
            gc.collect()
        rows = get_records(root, plan)
        require(len(rows) == 48, 'Incomplete CV; test forbidden')
        frame = pd.DataFrame(rows)
        frame.to_csv(root / 'cv_results.csv', index=False, encoding='utf-8-sig')
        candidates, selections = [], []
        prior = float(np.mean([log_loss(y.iloc[v], np.full(len(v), y.iloc[t].mean()), labels=[0, 1]) for t, v in cv]))
        for level in plan['levels']:
            level_candidates = []
            old = {r['fold']: r['log_loss'] for r in plan['original_cv_scores'] if r['level'] == level}
            for order, variant in enumerate(plan['variants']):
                sub = frame[(frame.level == level) & (frame.variant == variant['name'])].sort_values('fold')
                require(sub.fold.tolist() == [0, 1, 2], 'CV folds incomplete')
                record = dict(level=level, variant=variant['name'], variant_order=order,
                    mean_log_loss=float(sub.log_loss.mean()), std_log_loss=float(sub.log_loss.std(ddof=0)),
                    mean_ap=float(sub.ap.mean()), mean_best_epoch=float(sub.best_epoch.mean()))
                candidates.append(record); level_candidates.append(record)
            best = min(level_candidates, key=lambda r: (r['mean_log_loss'], r['variant_order']))
            variant = plan['variants'][best['variant_order']]
            fold_result = frame[(frame.level == level) & (frame.variant == best['variant'])]
            eligible = (best['mean_log_loss'] < np.mean(list(old.values())) - 1e-6
                        and best['mean_log_loss'] < prior - 1e-6
                        and all(r.log_loss < old[r.fold] for r in fold_result.itertuples()))
            selections.append(dict(level=level, variant=best['variant'], params=variant,
                mean_cv_log_loss=best['mean_log_loss'], baseline_cv_log_loss=float(np.mean(list(old.values()))),
                constant_prior_cv_log_loss=prior, eligible_for_test=bool(eligible)))
        pd.DataFrame(candidates).to_csv(root / 'candidate_summary.csv', index=False, encoding='utf-8-sig')
        selection = dict(plan_fingerprint=plan['fingerprint'], selections=selections,
            selected_after_completed_cv=48, test_used_for_selection=False,
            gate=plan['test_gate'], all_level_selections_frozen_before_test=True)
        if (root / 'selection.json').exists():
            require(read_json(root / 'selection.json') == selection, 'Frozen selections changed')
        else:
            atomic_json(root / 'selection.json', selection)
        _events(root, 'all_level_selections_frozen', selection_sha256=sha256(root / 'selection.json'))
        # Only now initialize test access; no test-dependent model/rule selection.
        data = PreparedData(plan['prepared_dir'])
        require(np.array_equal(data.test, pd.read_parquet(Path(plan['original_pilot']) / 'test_ids.parquet').SK_ID_CURR),
                'Test IDs changed')
        for chosen in selections:
            if not chosen['eligible_for_test']:
                continue
            level = chosen['level']
            stage = root / 'final' / f'level_{level}'
            if _complete(stage, plan):
                continue
            stage.mkdir(parents=True, exist_ok=True)
            fit = stage / 'fit_checkpoint'; fit.mkdir(exist_ok=True)
            save_progress(root, plan, status='final_fitting', current=chosen)
            if _complete(fit, plan):
                model = joblib.load(fit / 'pipeline.joblib')
                diagnostics = read_json(fit / 'fit_diagnostics.json')
            else:
                X = pd.read_parquet(Path(plan['data_root']) / f'X_level_{level}.parquet')
                X.index = pd.Index(ids, name='SK_ID_CURR')
                cols = columns[str(level)]
                phase('fit', level=level, variant=chosen['variant'])
                _events(root, 'repair_final_fit_started', level=level, variant=chosen['variant'])
                model, diagnostics = fit_mlp_variant(X, y, cols['numeric'], cols['categorical'],
                                                    chosen['params'], **plan['settings'])
                _atomic_model(model, fit / 'pipeline.joblib')
                atomic_json(fit / 'fit_diagnostics.json', diagnostics)
                _mark_complete(fit, plan, ['pipeline.joblib', 'fit_diagnostics.json'])
                del X, model
                gc.collect()
                model = joblib.load(fit / 'pipeline.joblib')
            phase('test_prediction', level=level)
            Xtest, ytest = data.load(level, data.test, purpose='supplemental_mlp_repair_final_test')
            begin = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=2):
                warnings.simplefilter('always')
                p = positive_probability(model, Xtest)
            predict_seconds = time.perf_counter() - begin
            metrics = classification_metrics(ytest.to_numpy(), p)
            baseline = read_json(Path(plan['original_final_root']) / 'mlp' / f'level_{level}' / 'metrics.json')
            record = dict(scope=SCOPE, level=level, variant=chosen['variant'], metrics=metrics,
                baseline_metrics=baseline['metrics'], fit_seconds=diagnostics['total_seconds'],
                predict_seconds=predict_seconds, best_epoch=diagnostics['best_epoch'],
                train_rows=20000, test_rows=len(ytest), cv_selection=chosen,
                test_used_for_selection=False, original_result_replaced=False,
                warnings=diagnostics['warnings'] + _warnings_report(caught))
            _atomic_parquet(pd.DataFrame({'SK_ID_CURR': Xtest.index, 'TARGET': ytest.to_numpy(),
                'probability_default': p, 'scope': SCOPE, 'model': 'mlp', 'level': level}),
                stage / 'probabilities.parquet')
            atomic_json(stage / 'metrics.json', record)
            # Standalone final files for straightforward handoff/report access.
            shutil.copyfile(fit / 'pipeline.joblib', stage / 'pipeline.joblib')
            shutil.copyfile(fit / 'fit_diagnostics.json', stage / 'fit_diagnostics.json')
            _mark_complete(stage, plan, ['metrics.json', 'probabilities.parquet', 'pipeline.joblib',
                'fit_diagnostics.json', 'fit_checkpoint/complete.json', 'fit_checkpoint/pipeline.joblib',
                'fit_checkpoint/fit_diagnostics.json'])
            _events(root, 'repair_final_completed', level=level, variant=chosen['variant'], metrics=metrics)
            print(f'REPAIR FINAL L{level}: LL {baseline["metrics"]["log_loss"]:.6f} -> {metrics["log_loss"]:.6f}', flush=True)
            del model, diagnostics, Xtest, ytest, p
            gc.collect()
        result = save_progress(root, plan, status='completed')
        before = read_json(root / 'protected_before.json')
        after = protected_snapshot(project)
        differences = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        atomic_json(root / 'integrity_validation.json', dict(passed=not differences,
            checked_files=len(before), differences=differences, original_results_unchanged=not differences))
        require(not differences, 'Original results changed')
        result.update(eligible_levels=[s['level'] for s in selections if s['eligible_for_test']],
                      elapsed_seconds=time.perf_counter()-started, original_results_unchanged=True,
                      outer_cv_evaluations=48, cv_optimizer_trajectories=96,
                      final_optimizer_trajectories=2*result['final_levels_completed'])
        atomic_json(root / 'summary.json', result)
        return result
    except BaseException as exc:
        _failure(root, root, 'mlp_repair', exc)
        save_progress(root, plan, status='failed')
        raise
