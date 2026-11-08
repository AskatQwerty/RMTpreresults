"""Immutable 20-configuration plan extending the existing 20k research pilot.

Only reads existing data/checkpoints. No split generation, fit, transform,
prediction, or test evaluation is performed. Candidate numbers are 1-based;
candidate_index is the corresponding zero-based index used in CV results.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler

from .data_quality import artifact_manifest
from .model_pipelines import parameter_space
from .model_training import (PACKAGES, PreparedData, _complete, _json_text,
                             atomic_json, read_json, sha256)
from .research_pilot import (COHORT_FILES, LEVELS, MODEL_ORDER, SOURCE_FILES,
                             load_pilot_split)

PLAN_FILE = 'configuration_plan.json'
SCOPE = 'pilot_20k_20config_extension'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json_value(value):
    """Convert numpy scalar values to the same persistent JSON representation."""
    return json.loads(_json_text(value))


def _sample_configurations(spaces):
    configurations = {}
    for model in MODEL_ORDER:
        sampled = list(ParameterSampler(spaces[model], n_iter=20, random_state=42))
        _require(len(sampled) == 20, f'{model}: exactly 20 configurations required')
        entries = []
        for index, params in enumerate(sampled):
            params = _json_value(params)
            digest = hashlib.sha256(_json_text(params).encode('utf-8')).hexdigest()
            entries.append({'candidate_index': index, 'candidate_number': index + 1,
                            'params': params, 'config_sha256': digest})
        _require(len({e['config_sha256'] for e in entries}) == 20,
                 f'{model}: duplicate configurations in the fixed plan')
        configurations[model] = entries
    return configurations


def _validate_cv_frame(frame, entries, label):
    required = ['candidate', 'params_json', 'mean_neg_log_loss'] + [
        f'fold_{fold}_neg_log_loss' for fold in range(3)]
    _require(all(name in frame.columns for name in required),
             f'{label}: incomplete saved candidate/fold scores')
    frame = frame.sort_values('candidate').reset_index(drop=True)
    _require(frame.candidate.tolist() == list(range(5)),
             f'{label}: expected exactly saved candidate indices 0..4')
    scores = frame[[f'fold_{fold}_neg_log_loss' for fold in range(3)]].to_numpy(dtype=float)
    _require(np.isfinite(scores).all(), f'{label}: nonfinite legacy CV score')
    _require(np.allclose(scores.mean(axis=1), frame.mean_neg_log_loss.to_numpy(dtype=float),
                        atol=1e-12, rtol=0), f'{label}: inconsistent saved fold means')
    for index, row in frame.iterrows():
        _require(_json_text(json.loads(row.params_json)) == _json_text(entries[index]['params']),
                 f'{label}: old candidate {index + 1} differs from fixed first five')
    return frame


def build_extension_plan(prepared_dir, pilot_root):
    """Read-only full validation and deterministic plan (no fit or X loading).

    Rejects changes to source code, versions, prepared files, cohort, folds,
    or any of the 28 original fit checkpoints before reusing their CV scores.
    """
    prepared = Path(prepared_dir).resolve()
    pilot = Path(pilot_root).resolve()
    original = read_json(pilot / 'experiment_manifest.json')
    _require(original.get('scope') == 'pilot_20k_5config', 'Unexpected original pilot scope')
    _require(original.get('train_rows') == 20000, 'Keep exactly the existing 20,000 clients')
    _require(original['protocol'].get('configurations_per_search') == 5,
             'Expected original five-configuration pilot')
    _require(original['protocol'].get('search_random_state') == 42,
             'Unexpected original search seed')
    _require(original['protocol'].get('technical') is False,
             'Technical models cannot be reused as research results')
    _require(original['protocol'].get('scoring') == 'neg_log_loss', 'Unexpected scoring')
    _require(original.get('threads') == 2 and original.get('search_n_jobs') == 1,
             'Expected original sequential two-thread settings')

    source = Path(__file__).resolve().parent
    code = {name: sha256(source / name) for name in SOURCE_FILES}
    _require(code == original['code_sha256'], 'Original model/preprocessing code changed')
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    _require(versions == original['versions'], 'Package versions changed since the pilot')
    _require(platform.python_version() == original['python'], 'Python version changed')
    spaces = _json_value({model: parameter_space(model, technical=False) for model in MODEL_ORDER})
    _require(spaces == original['parameter_spaces'], 'Original parameter spaces changed')
    configurations = _sample_configurations(spaces)

    prepared_inputs = artifact_manifest(prepared)
    _require(prepared_inputs == original['inputs'], 'Prepared data/features/splits changed')
    data = PreparedData(prepared)
    contract = read_json(pilot / 'cohort_contract.json')
    _require(contract == original['cohort_contract'], 'Saved pilot cohort contract changed')
    _require(set(contract['artifact_sha256']) == set(COHORT_FILES), 'Unexpected cohort files')
    ids, splits = load_pilot_split(data, pilot)
    _require(len(ids) == 20000, 'Existing cohort must have 20,000 clients')
    labels = data.target.loc[ids]
    cohort_summary = {
        'train_rows': len(ids), 'development_rows': len(data.development),
        'test_rows': len(data.test), 'class_counts': {str(v): int((labels == v).sum()) for v in (0, 1)},
        'cv': [{'fold': fold, 'train_rows': len(train), 'validation_rows': len(valid)}
               for fold, (train, valid) in enumerate(splits)],
        'feature_counts': {str(level): len(data.features[str(level)]) for level in LEVELS},
        'saved_cohort_ids_reused': True, 'saved_cv_folds_reused': True,
        'new_split_calls': 0, 'test_X_loaded': False,
    }

    legacy_results = []
    for model in MODEL_ORDER:
        for level in LEVELS:
            fit = pilot / model / f'level_{level}' / 'fit_checkpoint'
            _require(_complete(fit, original), f'Incomplete original checkpoint: {fit}')
            selected = read_json(fit / 'best_params.json')
            summary = read_json(fit / 'fit_summary.json')
            _require(selected['n_iter'] == 5 and selected['folds'] == 3
                     and selected['train_rows'] == 20000
                     and selected['scoring'] == 'neg_log_loss',
                     f'{model} L{level}: incompatible original search')
            _require(summary['fit_calls']['candidate_cv'] == 15,
                     f'{model} L{level}: unexpected original fit budget')
            file = fit / 'cv_results.parquet'
            frame = _validate_cv_frame(pd.read_parquet(file), configurations[model],
                                       f'{model} L{level}')
            legacy_results.append({
                'model': model, 'level': level,
                'cv_results_file': file.relative_to(pilot).as_posix(),
                'cv_results_sha256': sha256(file),
                'complete_file': (fit / 'complete.json').relative_to(pilot).as_posix(),
                'complete_sha256': sha256(fit / 'complete.json'),
                'candidate_indices': frame.candidate.astype(int).tolist(),
                'candidate_numbers': list(range(1, 6)), 'candidate_cv_fits': 15,
                'first_five_exactly_match_fixed_plan': True,
            })
    _require(len(legacy_results) == 28, 'Expected all 28 model/level checkpoints')
    _require(not data.loaded_scopes, 'Plan validation must not load model X')

    plan = {
        'schema_version': 1, 'scope': SCOPE,
        'prepared_dir': str(prepared), 'pilot_root': str(pilot),
        'pilot_fingerprint': original['fingerprint'],
        'pilot_manifest_sha256': sha256(pilot / 'experiment_manifest.json'),
        'models': list(MODEL_ORDER), 'levels': list(LEVELS),
        'order': [{'model': model, 'level': level} for model in MODEL_ORDER for level in LEVELS],
        'train_rows': 20000, 'technical': False, 'scoring': 'neg_log_loss',
        'search_random_state': 42, 'estimator_seed': 42, 'threads': 2,
        'search_n_jobs': 1, 'pre_dispatch': 1, 'refit': False,
        'configurations_per_pair': 20, 'cv_folds': 3,
        'total_planned_cv_fits': 1680, 'legacy_completed_cv_fits': 420,
        'initial_extension_batch': {'candidate_indices': list(range(5, 10)),
                                    'candidate_numbers': list(range(6, 11)),
                                    'new_cv_fits': 420},
        'final_refits': 0, 'repeat_fits': 0, 'test_evaluations': 0,
        'parameter_spaces': spaces, 'configurations': configurations,
        'cohort_contract': contract, 'cohort_summary': cohort_summary,
        'prepared_inputs': prepared_inputs, 'code_sha256': code,
        'versions': versions, 'python': platform.python_version(),
        'legacy_results': legacy_results,
        'duplicate_policy': '20 unique parameter dictionaries per model, shared across four levels; '
                            'each candidate/fold identified separately; resume verified completed folds only',
        'deviations_from_A2': ['Existing 20,000-client pilot cohort instead of full development',
                               'This execution only adds candidate numbers 6..10; later batches require authorization',
                               'No final refits, five seed repeats, or test evaluation in this batch'],
    }
    plan['fingerprint'] = hashlib.sha256(_json_text(plan).encode('utf-8')).hexdigest()
    return plan


def ensure_extension_plan(prepared_dir, pilot_root, output_root):
    """Save the fixed plan once; never replace an existing incompatible plan."""
    plan = build_extension_plan(prepared_dir, pilot_root)
    destination = Path(output_root).resolve()
    pilot = Path(pilot_root).resolve()
    prepared = Path(prepared_dir).resolve()
    _require(destination != pilot and pilot not in destination.parents,
             'Extension output must be separate from the original pilot')
    _require(destination != prepared and prepared not in destination.parents,
             'Extension output must be separate from prepared data')
    file = destination / PLAN_FILE
    if file.exists():
        _require(read_json(file) == plan, 'Saved fixed plan changed or inputs no longer match; do not overwrite')
    else:
        destination.mkdir(parents=True, exist_ok=True)
        atomic_json(file, plan)
    return plan
