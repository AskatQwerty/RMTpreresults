"""28 final fits after the completed 20-configuration search; no new CV/search.

Original pilot artifacts are immutable. One fresh full Pipeline is trained on
the exact saved 20k cohort per pair; test X is loaded only after a verified fit
checkpoint. A failed prediction resumes from that checkpoint without refitting.
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
from threadpoolctl import threadpool_limits

from .data_quality import artifact_manifest
from .fit_diagnostics import extract_fit_diagnostics
from .model_pipelines import make_estimator, experiment_spec
from .model_training import (PreparedData, PACKAGES, atomic_json, read_json, sha256,
    _json_text, _atomic_parquet, _atomic_model, _complete, _mark_complete,
    _events, _failure, _warnings_report, initialize_run, positive_probability,
    classification_metrics)
from .research_pilot import MODEL_ORDER, LEVELS, COHORT_FILES, load_pilot_split

SCOPE = 'pilot_20k_20config_final'
FIT_FILES = ('pipeline.joblib', 'cv_results.parquet', 'best_params.json',
             'fit_diagnostics.json', 'warnings.json', 'fit_summary.json')
CORE_FILES = ('model_training.py', 'model_pipelines.py', 'fold_preprocessing.py',
              'fit_diagnostics.py', 'research_pilot.py')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def protected_snapshot(project):
    """Preserve evidence that the original preliminary outputs were not changed."""
    project = Path(project)
    result = {}
    for relative in ('artifacts/training/pilot_20k_5config', 'reports/pilot_20k_5config'):
        for path in sorted((project / relative).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                result[path.relative_to(project).as_posix()] = {
                    'bytes': path.stat().st_size, 'sha256': sha256(path)}
    return result


def initialize_final(prepared_dir, output_root):
    """No-fit validation and immutable input selection for the final 28 fits."""
    project = Path(os.environ.get('RMT_PROJECT_ROOT', Path(__file__).resolve().parents[1]))
    old = project / 'artifacts/training/pilot_20k_5config'
    combined = project / 'reports/training/pilot_20k_20config_combined'
    root = Path(output_root).resolve()
    prepared = Path(prepared_dir).resolve()
    _require(root != old.resolve() and old.resolve() not in root.parents, 'Use a separate final output')
    _require(root != prepared and prepared not in root.parents, 'Do not write into prepared data')
    audit = read_json(combined / 'mlp_audit.json')
    _require(audit.get('status') == 'passed_with_caveats' and audit.get('blocking_issues') == [],
             'MLP audit must pass before final training')
    imported = read_json(combined / 'import_audit.json')
    _require(imported['total_cv_fits'] == 1680 and imported['missing_cv_fits'] == 0
             and imported['duplicate_cv_fits'] == 0, 'Full CV evidence is incomplete')
    original = read_json(old / 'experiment_manifest.json')
    plan = read_json(project / 'artifacts/training/pilot_20k_20config_extension/configuration_plan.json')
    _require(imported['original_plan_fingerprint'] == plan['fingerprint'], 'Wrong selection plan')
    source = Path(__file__).resolve().parent
    for name in CORE_FILES:
        _require(sha256(source / name) == original['code_sha256'][name], 'Frozen core changed: ' + name)
    _require(platform.python_version() == original['python'], 'Python changed')
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    _require(versions == original['versions'], 'Package versions changed')
    inputs = artifact_manifest(prepared)
    _require(inputs == original['inputs'], 'Prepared data/features/splits changed')
    data = PreparedData(prepared)
    ids, _ = load_pilot_split(data, old)
    _require(len(ids) == 20000 and len(data.test) == 61503, 'Saved cohort/test sizes changed')
    selected = read_json(combined / 'selected_parameters.json')
    candidates = pd.read_csv(combined / 'all_560_candidates.csv')
    fold_scores = pd.read_csv(combined / 'all_1680_cv_folds.csv')
    _require(len(candidates) == 560 and len(fold_scores) == 1680, 'CV table coverage changed')
    expected = {(m, l) for m in MODEL_ORDER for l in LEVELS}
    _require(len(selected) == 28 and {(s['model'], s['level']) for s in selected} == expected,
             'Exactly one selection per model/level required')
    for s in selected:
        frame = candidates[(candidates.model == s['model']) & (candidates.level == s['level'])]
        _require(sorted(frame.candidate_number.tolist()) == list(range(1, 21)), 'Candidate coverage')
        best = frame.sort_values(['mean_log_loss', 'candidate_number']).iloc[0]
        _require(int(best.candidate_number) == s['candidate_number']
                 and abs(best.mean_log_loss - s['mean_cv_log_loss']) < 1e-12, 'Selection is not CV minimum')
        config = plan['configurations'][s['model']][s['candidate_number'] - 1]
        _require(s['params'] == config['params'], 'Selected parameters changed')
    root.mkdir(parents=True, exist_ok=True)
    for name in (*COHORT_FILES, 'cohort_contract.json', 'cohort_summary.json'):
        target = root / name
        if target.exists():
            _require(sha256(target) == sha256(old / name), 'Copied cohort file changed: ' + name)
        else:
            shutil.copy2(old / name, target)
    selection_dir = root / 'selection_evidence'
    selection_dir.mkdir(exist_ok=True)
    evidence = ('selected_parameters.json', 'all_1680_cv_folds.csv', 'all_560_candidates.csv',
                'import_audit.json', 'mlp_audit.json')
    for name in evidence:
        target = selection_dir / name
        if target.exists():
            _require(sha256(target) == sha256(combined / name), 'Frozen selection/audit changed: ' + name)
        else:
            shutil.copy2(combined / name, target)
    protocol = dict(scope=SCOPE, models=list(MODEL_ORDER), levels=list(LEVELS),
        order=[{'model': m, 'level': l} for m in MODEL_ORDER for l in LEVELS],
        train_rows=20000, cohort_seed=42, stratified_selection=True, class_resampling=False,
        cv_folds=3, cv_shuffle=True, cv_random_state=42, search_random_state=42,
        configurations_per_search=20, scoring='neg_log_loss', technical=False, estimator_seed=42,
        completed_candidate_cv_outer_fits=1680, new_candidate_cv_fits=0, new_searches=0,
        final_refit_fits=28, manual_additional_fits=28, automatic_refit_fits=0,
        primitive_fits_including_ensembles=84, repeat_fits=0,
        selection='minimum mean saved CV log-loss across20 fixed configurations; ties smallest candidate number',
        refit=True, test_use='final evaluation only; no selection/tuning using test', shap_in_scope=False,
        deviations_from_A2=['20,000 original pilot clients instead of full development',
                            'One final full-cohort fit per pair; five90%seed repeats assigned separately'])
    context = dict(scope=SCOPE, mode=SCOPE, protocol=protocol, train_rows=20000,
        development_rows=len(data.development), test_rows=len(data.test),
        project_root=str(project), prepared_dir=str(prepared), original_pilot=str(old),
        original_pilot_fingerprint=original['fingerprint'], original_plan_fingerprint=plan['fingerprint'],
        positive_class=1, probability_column_selection='classes_ == 1', threads=2, search_n_jobs=1,
        inputs=inputs, cohort_contract=read_json(root / 'cohort_contract.json'),
        versions=versions, python=platform.python_version(),
        code_sha256={name: sha256(source / name) for name in (*CORE_FILES, 'final_pilot_training.py')},
        selection_evidence_sha256={name: sha256(selection_dir / name) for name in evidence})
    context['fingerprint'] = hashlib.sha256(_json_text(context).encode()).hexdigest()
    initialize_run(root, context)
    snapshot = root / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for name in context['code_sha256']:
        path = snapshot / name
        if path.exists():
            _require(sha256(path) == context['code_sha256'][name], 'Source snapshot changed')
        else:
            shutil.copy2(source / name, path)
    before = root / 'old_preresults_integrity_before.json'
    if not before.exists():
        atomic_json(before, protected_snapshot(project))
    return context


def _validate_runtime(root, context):
    _require(context['scope'] == SCOPE and context['train_rows'] == 20000, 'Wrong final scope')
    for name, digest in context['code_sha256'].items():
        _require(sha256(Path(__file__).parent / name) == digest, 'Training source changed: ' + name)
    for name, digest in context['selection_evidence_sha256'].items():
        _require(sha256(Path(root) / 'selection_evidence' / name) == digest, 'Selection changed')


def _phase(name, model, level):
    path = os.environ.get('RMT_PHASE_PATH')
    if path:
        atomic_json(path, {'phase': name, 'model': model, 'level': level})


def _selected_pair(root, model, level):
    entries = read_json(Path(root) / 'selection_evidence/selected_parameters.json')
    matches = [s for s in entries if s['model'] == model and s['level'] == level]
    _require(len(matches) == 1, 'Missing/duplicate frozen selection')
    return matches[0]


def _saved_cv(root, model, level):
    frame = pd.read_csv(Path(root) / 'selection_evidence/all_560_candidates.csv')
    frame = frame[(frame.model == model) & (frame.level == level)].sort_values('candidate_number').copy()
    _require(frame.candidate_number.tolist() == list(range(1, 21)), 'CV evidence coverage')
    frame['candidate'] = frame.candidate_number - 1
    frame['mean_neg_log_loss'] = -frame.mean_log_loss
    frame['std_neg_log_loss'] = frame.std_log_loss
    frame['rank'] = frame.mean_log_loss.rank(method='min').astype(int)
    for fold in range(3):
        frame[f'fold_{fold}_neg_log_loss'] = -frame[f'fold_{fold}_log_loss']
    return frame


def run_final_pair(prepared_dir, output_root, model, level):
    _require(model in MODEL_ORDER and level in LEVELS, 'Unknown model/level')
    root = Path(output_root)
    context = read_json(root / 'experiment_manifest.json')
    _validate_runtime(root, context)
    stage = root / model / f'level_{level}'
    if _complete(stage, context):
        return read_json(stage / 'metrics.json')
    stage.mkdir(parents=True, exist_ok=True)
    fit = stage / 'fit_checkpoint'
    fit.mkdir(exist_ok=True)
    started = time.perf_counter()
    data = PreparedData(prepared_dir)
    ids, _ = load_pilot_split(data, root)
    chosen = _selected_pair(root, model, level)
    caught = []
    try:
        if _complete(fit, context):
            fitted = joblib.load(fit / 'pipeline.joblib')
            selected = read_json(fit / 'best_params.json')
            fit_summary = read_json(fit / 'fit_summary.json')
            diagnostics = read_json(fit / 'fit_diagnostics.json')
            _require(extract_fit_diagnostics(fitted) == diagnostics, 'Reload diagnostics changed')
            _events(root, 'final_fit_checkpoint_resumed', model=model, level=level)
        else:
            X, y = data.load(level, ids, purpose='final_selected_20k_train')
            _require(np.array_equal(X.index, ids) and X.index.equals(y.index), 'Training row alignment')
            _require(set(X.index).isdisjoint(data.test), 'Train/test overlap')
            numeric, categorical = data.columns(level)
            fitted = make_estimator(model, numeric, categorical, seed=42, technical=False, threads=2)
            fitted.set_params(**chosen['params'])
            _phase('final_fit', model, level)
            _events(root, 'final_fit_started', model=model, level=level, train_rows=len(ids),
                    candidate_number=chosen['candidate_number'], params=chosen['params'])
            print(f'FINAL FIT {model} L{level}: saved candidate{chosen["candidate_number"]}, {len(ids)} rows', flush=True)
            with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=2):
                warnings.simplefilter('always')
                begin = time.perf_counter()
                fitted.fit(X, y)
                fit_seconds = time.perf_counter() - begin
            diagnostics = extract_fit_diagnostics(fitted)
            selected = dict(scope=SCOPE, model=model, level=level, params=chosen['params'],
                best_cv_log_loss=chosen['mean_cv_log_loss'], best_cv_neg_log_loss=-chosen['mean_cv_log_loss'],
                best_candidate_index=chosen['candidate_number']-1, candidate_number=chosen['candidate_number'],
                n_iter=20, folds=3, scoring='neg_log_loss', refit=True, train_rows=len(ids),
                selection_source='existing1680CV; zero new search fits', fitted_seed=42)
            fit_summary = dict(scope=SCOPE, model=model, level=level, train_rows=len(ids),
                raw_features=len(data.features[str(level)]), refit_seconds=fit_seconds,
                search_seconds=0, warnings=_warnings_report(caught),
                fit_calls={'candidate_cv': 0, 'automatic_refit': 0, 'manual_final_refit': 1,
                           'primitive_including_ensembles': experiment_spec(model)['primitive_fits_per_outer_fit']},
                fitted_on_all_saved_cohort_ids=True, test_loaded_before_checkpoint=False)
            _atomic_model(fitted, fit / 'pipeline.joblib')
            _atomic_parquet(_saved_cv(root, model, level), fit / 'cv_results.parquet')
            atomic_json(fit / 'best_params.json', selected)
            atomic_json(fit / 'fit_diagnostics.json', diagnostics)
            atomic_json(fit / 'warnings.json', fit_summary['warnings'])
            fit_summary['pipeline_bytes'] = (fit / 'pipeline.joblib').stat().st_size
            atomic_json(fit / 'fit_summary.json', fit_summary)
            _mark_complete(fit, context, FIT_FILES)
            _events(root, 'final_fit_completed', model=model, level=level, refit_seconds=fit_seconds)
            del X, y, fitted
            gc.collect()
            fitted = joblib.load(fit / 'pipeline.joblib')
            _require(extract_fit_diagnostics(fitted) == diagnostics, 'Serialized Pipeline mismatch')
        _phase('test_prediction', model, level)
        _events(root, 'test_evaluation_started', model=model, level=level, test_rows=len(data.test))
        Xtest, ytest = data.load(level, data.test, purpose='final_selected_test_evaluation')
        _require(np.array_equal(Xtest.index, data.test) and Xtest.index.equals(ytest.index), 'Test row alignment')
        with warnings.catch_warnings(record=True) as evaluation_warnings, threadpool_limits(limits=2):
            warnings.simplefilter('always')
            begin = time.perf_counter()
            probabilities = positive_probability(fitted, Xtest)
            predict_seconds = time.perf_counter() - begin
        predictions = pd.DataFrame({'SK_ID_CURR': Xtest.index.to_numpy(), 'TARGET': ytest.to_numpy(),
            'probability_default': probabilities, 'scope': SCOPE, 'model': model, 'level': level,
            'evaluation_split': 'test'})
        _require(predictions.SK_ID_CURR.is_unique and len(predictions) == context['test_rows'], 'Test IDs')
        prediction_warnings = _warnings_report(evaluation_warnings)
        combined = [dict(w, stage='final_refit') for w in fit_summary['warnings']]
        combined.extend(dict(w, stage='test_prediction') for w in prediction_warnings)
        metrics = dict(scope=SCOPE, model=model, level=level, train_rows=len(ids), test_rows=len(data.test),
            raw_features=len(data.features[str(level)]), selected_params=selected['params'],
            selected_candidate_number=selected['candidate_number'], best_cv_log_loss=selected['best_cv_log_loss'],
            refit_seconds=fit_summary['refit_seconds'], predict_seconds=predict_seconds,
            total_pair_seconds=time.perf_counter()-started, warnings=combined,
            metrics=classification_metrics(ytest.to_numpy(), probabilities),
            fit_diagnostics_file='fit_checkpoint/fit_diagnostics.json', pipeline_file='fit_checkpoint/pipeline.joblib',
            actual_mlp_iterations=[r['fitted']['n_iter_'] for r in diagnostics['mlp']],
            fit_calls=fit_summary['fit_calls'], threshold_selected_using_test=False,
            preprocessing_refit_on_test=False, data_access=data.loaded_scopes)
        _atomic_parquet(predictions, stage / 'probabilities.parquet')
        atomic_json(stage / 'prediction_warnings.json', prediction_warnings)
        atomic_json(stage / 'metrics.json', metrics)
        _mark_complete(stage, context, ['probabilities.parquet', 'prediction_warnings.json', 'metrics.json',
            'fit_checkpoint/complete.json'] + ['fit_checkpoint/' + name for name in FIT_FILES])
        _events(root, 'pair_completed', model=model, level=level, scope=SCOPE,
                refit_seconds=fit_summary['refit_seconds'], predict_seconds=predict_seconds)
        print(f'FINAL COMPLETE {model} L{level}; test{len(predictions)}, AP={metrics["metrics"]["pr_auc_average_precision"]:.6f}', flush=True)
        return metrics
    except BaseException as exc:
        if caught:
            atomic_json(stage / 'failed_attempt_warnings.json', _warnings_report(caught))
        _failure(stage, root, 'final_pilot_pair', exc)
        raise


def write_summary(output_root):
    root = Path(output_root)
    context = read_json(root / 'experiment_manifest.json')
    metrics, fitted = [], 0
    for model in MODEL_ORDER:
        for level in LEVELS:
            stage = root / model / f'level_{level}'
            if _complete(stage / 'fit_checkpoint', context):
                fitted += 1
            if _complete(stage, context):
                metrics.append(read_json(stage / 'metrics.json'))
    summary = dict(scope=SCOPE, status='completed' if len(metrics) == 28 else 'partial',
        completed_pairs=len(metrics), completed_final_fits=fitted, expected_final_fits=28,
        train_rows=20000, test_rows=61503, completed_prior_cv_fits=1680, new_cv_fits=0,
        final_fit_seconds=sum(m['refit_seconds'] for m in metrics),
        test_predict_seconds=sum(m['predict_seconds'] for m in metrics),
        primitive_fit_count=sum(m['fit_calls']['primitive_including_ensembles'] for m in metrics),
        repeat_fits=0, test_selected_parameters=False)
    atomic_json(root / 'experiment_summary.json', summary)
    if metrics:
        rows = [dict(model=m['model'], level=m['level'], selected_candidate_number=m['selected_candidate_number'],
                     best_cv_log_loss=m['best_cv_log_loss'], refit_seconds=m['refit_seconds'],
                     predict_seconds=m['predict_seconds'], **m['metrics']) for m in metrics]
        pd.DataFrame(rows).to_csv(root / 'final_metrics.csv', index=False, encoding='utf-8-sig')
    return summary
