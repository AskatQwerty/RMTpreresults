"""Extend the fixed pilot with new CV fits and a checkpoint for each fold.

The first five configurations and their scores are read from pilot_20k_5config. The next
default batch covers configurations 6–10: 7 models × 4 levels × 5 × 3 = 420.
The shared list of 20 configurations is fixed before training. The same
20,000 IDs and saved pilot folds are used, without resampling clients.
Manual sequential iteration over fixed candidates uses the same
sklearn neg_log_loss scorer as the original RandomizedSearchCV. A full Pipeline
is recreated for every training fold, including the internal stacking bases.
This batch performs neither test evaluation nor final refits.
"""
from __future__ import annotations

import gc
import hashlib
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import get_scorer
from threadpoolctl import threadpool_limits

from .fit_diagnostics import extract_fit_diagnostics
from .model_pipelines import make_estimator, experiment_spec
from .model_training import (
    PreparedData, atomic_json, read_json, sha256, _json_text, _atomic_parquet,
    _complete, _mark_complete, _events, _failure, _warnings_report,
    initialize_run, run_lock,
)
from .research_pilot import MODEL_ORDER, LEVELS, load_pilot_split
from .pilot_extension_plan import ensure_extension_plan, build_extension_plan

SCOPE = 'pilot_20k_20config_extension'
DEFAULT_CANDIDATES = (6, 7, 8, 9, 10)


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _batch_numbers(numbers):
    values = tuple(int(value) for value in numbers)
    if len(values) != 5 or values != tuple(range(values[0], values[0] + 5)):
        raise ValueError('Each batch must contain five consecutive configuration numbers')
    if values[0] not in (6, 11, 16):
        raise ValueError('Allowed batches: 6–10, 11–15, 16–20')
    return values


def _batch_name(numbers):
    return f'configurations_{numbers[0]:02d}_{numbers[-1]:02d}'


def _fold_stage(root, model, level, number, fold):
    return Path(root) / model / f'level_{level}' / f'candidate_{number:02d}' / f'fold_{fold}'


def _context(plan, threads):
    result = {
        'scope': SCOPE, 'mode': 'pilot_extension_cv_only',
        'configuration_plan_fingerprint': plan['fingerprint'],
        'pilot_fingerprint': plan['pilot_fingerprint'],
        'pilot_root': plan['pilot_root'], 'prepared_dir': plan['prepared_dir'],
        'threads': threads, 'outer_jobs': 1, 'technical': False,
        'scoring': 'neg_log_loss', 'search_random_state': 42, 'estimator_seed': 42,
        'refit': False, 'test_evaluations': 0, 'repeat_fits': 0,
        'legacy_cv_fits': 420, 'planned_total_cv_fits': 1680,
        'batch_outer_cv_fits': 420, 'batch_primitive_fits_including_ensembles': 1260,
        'total_planned_primitive_cv_fits_including_ensembles': 5040,
        'execution': 'fixed ParameterSampler candidates, sequential train/validation folds',
        'methodology': 'ParameterSampler(n_iter=20, random_state=42) fixed before fit; '
                       'same sklearn neg_log_loss scorer and saved folds; '
                       'manual checkpointed continuation, not a new RandomizedSearchCV call',
        'resume_unit': 'one completed candidate/fold; no repeat of completed fits',
        'extension_source_sha256': {
            name: sha256(Path(__file__).resolve().parent / name)
            for name in ('pilot_extension.py', 'pilot_extension_plan.py')
        },
    }
    result['fingerprint'] = hashlib.sha256(_json_text(result).encode('utf-8')).hexdigest()
    return result


def _read_completed(root, context, plan):
    """Validate every completed fold; never turn corruption into another fit."""
    records = {}
    for model in MODEL_ORDER:
        for level in LEVELS:
            for number in range(6, 21):
                config = plan['configurations'][model][number - 1]
                for fold in range(3):
                    stage = _fold_stage(root, model, level, number, fold)
                    if not _complete(stage, context):
                        continue
                    record = read_json(stage / 'result.json')
                    identity = (model, level, number, fold)
                    if (record['model'], record['level'], record['candidate_number'], record['fold']) != identity:
                        raise RuntimeError(f'Checkpoint belongs to an incorrect context: {stage}')
                    if record['config_sha256'] != config['config_sha256']:
                        raise RuntimeError(f'Checkpoint parameters have changed: {stage}')
                    if not np.isfinite(record['neg_log_loss']):
                        raise RuntimeError(f'Nonnumeric score: {stage}')
                    records[identity] = record
    return records


def _progress(records, numbers, *, status, current=None, started_utc=None, error=None):
    batch_records = [record for key, record in records.items() if key[2] in numbers]
    pairs_complete = sum(
        all((model, level, number, fold) in records for number in numbers for fold in range(3))
        for model in MODEL_ORDER for level in LEVELS
    )
    result = {
        'scope': SCOPE, 'status': status, 'updated_utc': _utc(),
        'candidate_numbers': list(numbers), 'batch_expected_cv_fits': 420,
        'batch_completed_cv_fits': len(batch_records),
        'legacy_completed_cv_fits': 420, 'extension_completed_cv_fits': len(records),
        'overall_completed_cv_fits': 420 + len(records), 'overall_planned_cv_fits': 1680,
        'pairs_with_requested_batch_complete': pairs_complete, 'pair_count': 28,
        'refit': False, 'test_evaluations': 0, 'repeat_fits': 0,
        'current': current,
        'batch_completed_fit_seconds': sum(r['fit_seconds'] for r in batch_records),
        'batch_completed_score_seconds': sum(r['score_seconds'] for r in batch_records),
    }
    if started_utc is not None:
        result['invocation_started_utc'] = started_utc
    if error is not None:
        result['error'] = error
    return result


def _save_progress(root, records, numbers, **kwargs):
    result = _progress(records, numbers, **kwargs)
    atomic_json(Path(root) / 'progress.json', result)
    atomic_json(Path(root) / 'batches' / _batch_name(numbers) / 'progress.json', result)
    return result


def _fit_one_fold(model, numeric, categorical, params, X, y, train, validation, *, threads):
    """The only fitting function. Its estimator sees solely the current train."""
    estimator = make_estimator(model, numeric, categorical, seed=42,
                               technical=False, threads=threads)
    estimator.set_params(**params)
    with warnings.catch_warnings(record=True) as fitted_warnings, threadpool_limits(limits=threads):
        warnings.simplefilter('always')
        started = time.perf_counter()
        estimator.fit(X.iloc[train], y.iloc[train])
        fit_seconds = time.perf_counter() - started
    with warnings.catch_warnings(record=True) as score_warnings, threadpool_limits(limits=threads):
        warnings.simplefilter('always')
        started = time.perf_counter()
        score = float(get_scorer('neg_log_loss')(estimator, X.iloc[validation], y.iloc[validation]))
        score_seconds = time.perf_counter() - started
    if not np.isfinite(score):
        raise RuntimeError('neg_log_loss returned a non-finite score')
    diagnostics = extract_fit_diagnostics(estimator)
    records = [dict(item, stage='fit') for item in _warnings_report(fitted_warnings)]
    records += [dict(item, stage='validation_score') for item in _warnings_report(score_warnings)]
    del estimator
    return {'neg_log_loss': score, 'fit_seconds': fit_seconds,
            'score_seconds': score_seconds, 'warnings': records}, diagnostics


def _combined_pair(root, plan, model, level, records):
    """Combine saved legacy scores with complete new candidates; no model fit."""
    old_path = Path(plan['pilot_root']) / model / f'level_{level}' / 'fit_checkpoint' / 'cv_results.parquet'
    frame = pd.read_parquet(old_path).copy()
    if len(frame) != 5 or frame.candidate.tolist() != list(range(5)):
        raise RuntimeError(f'Incorrect original five candidates: {old_path}')
    frame['candidate_number'] = frame.candidate + 1
    frame['origin'] = 'legacy_pilot_20k_5config'
    new_rows = []
    for number in range(6, 21):
        keys = [(model, level, number, fold) for fold in range(3)]
        if not all(key in records for key in keys):
            continue
        fold_records = [records[key] for key in keys]
        scores = [r['neg_log_loss'] for r in fold_records]
        fits = [r['fit_seconds'] for r in fold_records]
        scoring = [r['score_seconds'] for r in fold_records]
        new_rows.append({
            'candidate': number - 1, 'candidate_number': number,
            'params_json': _json_text(plan['configurations'][model][number - 1]['params']),
            'mean_neg_log_loss': float(np.mean(scores)), 'std_neg_log_loss': float(np.std(scores)),
            'mean_fit_seconds': float(np.mean(fits)), 'std_fit_seconds': float(np.std(fits)),
            'mean_score_seconds': float(np.mean(scoring)), 'std_score_seconds': float(np.std(scoring)),
            **{f'fold_{fold}_neg_log_loss': value for fold, value in enumerate(scores)},
            'origin': SCOPE,
        })
    if new_rows:
        frame = pd.concat([frame, pd.DataFrame(new_rows)], ignore_index=True)
    frame = frame.sort_values('candidate').reset_index(drop=True)
    frame['rank'] = (-frame.mean_neg_log_loss).rank(method='min').astype(int)
    winner = frame.iloc[int(np.argmax(frame.mean_neg_log_loss.to_numpy()))]
    selected = {
        'scope': SCOPE, 'model': model, 'level': level,
        'completed_candidates': len(frame), 'planned_candidates': 20,
        'best_candidate_number': int(winner.candidate_number),
        'best_candidate_index': int(winner.candidate),
        'params': json.loads(winner.params_json),
        'best_cv_neg_log_loss': float(winner.mean_neg_log_loss),
        'best_cv_log_loss': float(-winner.mean_neg_log_loss),
        'best_candidate_origin': str(winner.origin), 'scoring': 'neg_log_loss',
        'refit': False, 'selected_parameters_are_not_fitted_pipeline': True,
        'train_rows': plan['cohort_contract']['sample_n'], 'cv_folds': 3,
    }
    pair_root = Path(root) / model / f'level_{level}'
    _atomic_parquet(frame, pair_root / 'combined_cv_results.parquet')
    atomic_json(pair_root / 'best_params_so_far.json', selected)
    return selected


def write_extension_report(output_root, report_dir=None):
    """Aggregate only existing checkpoints; safe after interruption (no fit)."""
    root = Path(output_root)
    plan = read_json(root / 'configuration_plan.json')
    context = read_json(root / 'experiment_manifest.json')
    if context['configuration_plan_fingerprint'] != plan['fingerprint']:
        raise RuntimeError('Manifest and configuration plan disagree')
    records = _read_completed(root, context, plan)
    current = read_json(root / 'progress.json') if (root / 'progress.json').exists() else {}
    numbers = _batch_numbers(current.get('candidate_numbers', DEFAULT_CANDIDATES))
    summary = _progress(records, numbers, status=current.get('status', 'partial'),
                        current=current.get('current'), started_utc=current.get('invocation_started_utc'),
                        error=current.get('error'))
    selections = [_combined_pair(root, plan, model, level, records)
                  for model in MODEL_ORDER for level in LEVELS]
    warnings_rows = [dict(model=key[0], level=key[1], candidate_number=key[2], fold=key[3], **warning)
                     for key, record in records.items() for warning in record['warnings']]
    rows = [{k: v for k, v in selected.items() if k != 'params'}
            | {'params_json': _json_text(selected['params'])} for selected in selections]
    destinations = [root]
    if report_dir is not None:
        destinations.append(Path(report_dir))
    for destination in destinations:
        destination.mkdir(parents=True, exist_ok=True)
        atomic_json(destination / 'summary.json', summary)
        atomic_json(destination / 'selected_parameters_so_far.json', selections)
        for name, frame in [
            ('combined_search_results.csv', pd.DataFrame(rows)),
            ('warnings.csv', pd.DataFrame(warnings_rows, columns=[
                'model', 'level', 'candidate_number', 'fold', 'stage', 'category', 'message', 'count'])),
        ]:
            temporary = destination / (name + '.tmp')
            frame.to_csv(temporary, index=False)
            temporary.replace(destination / name)
    return summary


def run_extension(prepared_dir, pilot_root, output_root, *, threads=2,
                  candidate_numbers=DEFAULT_CANDIDATES, report_dir=None):
    """Execute only the next explicitly requested five-candidate batch (420 fits)."""
    numbers = _batch_numbers(candidate_numbers)
    if int(threads) != 2:
        raise ValueError('The saved pilot uses threads=2; do not change its context')
    root = Path(output_root)
    if root.resolve() == Path(pilot_root).resolve():
        raise ValueError('Extension output must not overwrite the original pilot')
    plan = ensure_extension_plan(prepared_dir, pilot_root, root)
    context = _context(plan, 2)
    initialize_run(root, context)
    began = _utc()
    new_count = 0
    with run_lock(root):
        records = _read_completed(root, context, plan)
        # Later batches are allowed only after all preceding new candidates.
        if numbers[0] > 6 and not all(
            (model, level, number, fold) in records
            for model in MODEL_ORDER for level in LEVELS
            for number in range(6, numbers[0]) for fold in range(3)
        ):
            raise RuntimeError('The previous batch is incomplete; candidates cannot be skipped')
        batch_root = root / 'batches' / _batch_name(numbers)
        atomic_json(batch_root / 'batch_manifest.json', {
            'scope': SCOPE, 'fingerprint': context['fingerprint'],
            'candidate_numbers': list(numbers), 'expected_cv_fits': 420,
            'legacy_results_reused': 420, 'refit': False, 'test_evaluations': 0,
        })
        _save_progress(root, records, numbers, status='running', started_utc=began)
        _events(root, 'extension_batch_started', candidate_numbers=list(numbers),
                existing_extension_fits=len(records), expected_new_batch_fits=420)
        data = PreparedData(prepared_dir)
        ids, splits = load_pilot_split(data, pilot_root)
        assert len(ids) == plan['cohort_contract']['sample_n']
        assert set(ids).isdisjoint(data.test)
        try:
            for model in MODEL_ORDER:
                for level in LEVELS:
                    keys = [(model, level, number, fold) for number in numbers for fold in range(3)]
                    if all(key in records for key in keys):
                        continue
                    X, y = data.load(level, ids, purpose='pilot_extension_cv_train')
                    assert np.array_equal(X.index.to_numpy(), ids) and X.index.equals(y.index)
                    assert X.columns.tolist() == data.features[str(level)]
                    numeric, categorical = data.columns(level)
                    for number in numbers:
                        config = plan['configurations'][model][number - 1]
                        for fold, (train, validation) in enumerate(splits):
                            identity = (model, level, number, fold)
                            if identity in records:
                                continue
                            stage = _fold_stage(root, model, level, number, fold)
                            stage.mkdir(parents=True, exist_ok=True)
                            current = {'model': model, 'level': level, 'candidate_number': number,
                                       'fold': fold, 'train_rows': len(train),
                                       'validation_rows': len(validation), 'cohort_rows': len(ids),
                                       'params': config['params'], 'started_utc': _utc()}
                            _save_progress(root, records, numbers, status='fitting', current=current,
                                           started_utc=began)
                            _events(root, 'cv_fit_started', **current)
                            print(f'CV START {model} L{level} config={number}/20 fold={fold + 1}/3; '
                                  f'batch={sum(k[2] in numbers for k in records)}/420; '
                                  f'overall={420 + len(records)}/1680', flush=True)
                            try:
                                measured, diagnostics = _fit_one_fold(
                                    model, numeric, categorical, config['params'], X, y,
                                    train, validation, threads=2)
                                result = {**current, **measured, 'completed_utc': _utc(),
                                          'config_sha256': config['config_sha256'],
                                          'raw_features': len(X.columns), 'scoring': 'neg_log_loss',
                                          'refit': False, 'test_used': False,
                                          'primitive_fits': experiment_spec(model)['primitive_fits_per_outer_fit']}
                                atomic_json(stage / 'result.json', result)
                                atomic_json(stage / 'fit_diagnostics.json', diagnostics)
                                _mark_complete(stage, context, ['result.json', 'fit_diagnostics.json'])
                            except BaseException as exc:
                                _failure(stage, root, 'cv_fold', exc)
                                raise
                            records[identity] = result
                            new_count += 1
                            progress = _save_progress(root, records, numbers, status='running',
                                                      current=None, started_utc=began)
                            _events(root, 'cv_fit_completed', model=model, level=level,
                                    candidate_number=number, fold=fold,
                                    fit_seconds=result['fit_seconds'], neg_log_loss=result['neg_log_loss'],
                                    batch_completed_cv_fits=progress['batch_completed_cv_fits'],
                                    overall_completed_cv_fits=progress['overall_completed_cv_fits'])
                            print(f'CV END {model} L{level} config={number} fold={fold + 1}; '
                                  f'log_loss={-result["neg_log_loss"]:.6f}; '
                                  f'batch={progress["batch_completed_cv_fits"]}/420; '
                                  f'overall={progress["overall_completed_cv_fits"]}/1680', flush=True)
                            gc.collect()
                    write_extension_report(root, report_dir)
                    del X, y
                    gc.collect()
            if any(scope['contains_test'] for scope in data.loaded_scopes):
                raise RuntimeError('Unexpected test load')
            if build_extension_plan(prepared_dir, pilot_root)['fingerprint'] != plan['fingerprint']:
                raise RuntimeError('Source data or legacy pilot changed during batch')
            summary = _save_progress(root, records, numbers, status='completed', started_utc=began)
            assert summary['batch_completed_cv_fits'] == 420
            summary = write_extension_report(root, report_dir)
            summary['new_fits_this_invocation'] = new_count
            summary['no_test_X_loaded'] = True
            atomic_json(batch_root / 'completion.json', summary)
            _events(root, 'extension_batch_completed', candidate_numbers=list(numbers),
                    new_fits_this_invocation=new_count,
                    overall_completed_cv_fits=summary['overall_completed_cv_fits'])
            return summary
        except BaseException as exc:
            prior = read_json(root / 'progress.json')
            _save_progress(root, records, numbers, status='failed', current=prior.get('current'),
                           started_utc=began, error=f'{type(exc).__name__}: {exc}')
            raise
