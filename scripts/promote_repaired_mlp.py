"""Assemble the active 28-model revision from verified saved models; never fit."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model_training import atomic_json, read_json, sha256
from src.mlp_repair_experiment import protected_snapshot

SCOPE = 'pilot_20k_20config_mlp_updated'
OLD = ROOT / 'artifacts/training/pilot_20k_20config_final'
REPAIR = ROOT / 'artifacts/training/mlp_repair_20k_v1'
OUT = ROOT / 'artifacts/training' / SCOPE
MODELS = ('logistic_regression', 'lightgbm', 'xgboost', 'soft_voting', 'stacking', 'mlp', 'random_forest')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_stage(path):
    marker = read_json(path / 'complete.json')
    for name, digest in marker['artifact_sha256'].items():
        file = (path / name).resolve()
        require(file.is_relative_to(path.resolve()) and sha256(file) == digest, f'Bad source: {file}')
    return marker


def protected():
    files = protected_snapshot(ROOT)
    # Include the supplemental experiment itself, now also an immutable source.
    for directory in (REPAIR, ROOT / 'reports/mlp_repair_20k_v1'):
        for path in directory.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                files[path.relative_to(ROOT).as_posix()] = {'bytes': path.stat().st_size, 'sha256': sha256(path)}
    return files


def mark(stage, fingerprint):
    names = [p.relative_to(stage).as_posix() for p in sorted(stage.rglob('*'))
             if p.is_file() and p != stage / 'complete.json']
    atomic_json(stage / 'complete.json', dict(fingerprint=fingerprint,
        action='adopt_existing_fitted_model_without_fit_or_predict',
        artifact_sha256={name: sha256(stage / name) for name in names}))


def main():
    require(not OUT.exists(), 'Output exists; refuse to overwrite an active revision')
    require(read_json(REPAIR / 'summary.json')['status'] == 'completed', 'Repair incomplete')
    require(read_json(ROOT / 'reports/mlp_repair_20k_v1/independent_validation.json')['passed'],
            'Independent repair validation required')
    old_manifest = read_json(OLD / 'experiment_manifest.json')
    repair_plan = read_json(REPAIR / 'plan.json')
    before = protected()
    lineage = []
    for model in MODELS:
        for level in range(1, 5):
            source = REPAIR / 'final' / f'level_{level}' if model == 'mlp' else OLD / model / f'level_{level}'
            checked_stage(source)
            lineage.append(dict(model=model, level=level, source_scope=REPAIR.name if model == 'mlp' else OLD.name,
                source_stage=str(source), source_complete_sha256=sha256(source / 'complete.json')))
    manifest = {**old_manifest, 'scope': SCOPE, 'mode': 'adopt_verified_saved_models',
        'supersedes_scope': OLD.name, 'model_lineage': lineage,
        'original_final_fingerprint': old_manifest['fingerprint'],
        'mlp_repair_plan_fingerprint': repair_plan['fingerprint'],
        'mlp_repair_selection_sha256': sha256(REPAIR / 'selection.json')}
    manifest['protocol'] = dict(scope=SCOPE, train_rows=20000, test_rows=61503,
        models=list(MODELS), levels=[1, 2, 3, 4], active_pipelines=28,
        unchanged_original_final_pipelines=24, replaced_mlp_pipelines=4,
        historical_original_candidate_cv_fits=1680, additional_mlp_outer_cv_evaluations=48,
        additional_mlp_optimizer_trajectories=104, configurations_per_non_mlp_pair=20,
        mlp_variants_per_level=4, cv_folds=3, cohort_seed=42, estimator_seed=42,
        new_fit_calls_in_promotion=0, new_predict_calls_in_promotion=0,
        class_resampling=False, repeats_per_pair=1, selection_metric='mean saved CV log_loss',
        non_mlp_selection='original minimum across20 configurations; identical saved final model',
        mlp_selection='minimum across4 predeclared supplemental variants; inner15%validation log_loss selects epochs; fresh20krefit',
        mlp_test_previously_seen=True, mlp_analysis_status='post-hoc exploratory replacement, user-authorized',
        thresholds_changed=False, test_use='same61503test; no new parameter or threshold selection',
        deviations_from_A2=['20k development cohort', 'one saved final model per pair rather than five repeats',
            'four active MLPs come from a supplemental repair protocol after the original test was inspected'],
        shap_in_scope=False)
    manifest.pop('fingerprint')
    manifest['fingerprint'] = hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False,
        separators=(',', ':')).encode('utf-8')).hexdigest()
    OUT.mkdir(parents=True)
    atomic_json(OUT / 'protected_before.json', before)
    atomic_json(OUT / 'experiment_manifest.json', manifest)
    for name in ('cohort_contract.json', 'cohort_summary.json', 'cohort_ids.parquet', 'cohort_targets.parquet',
                 'cv_folds.parquet', 'test_ids.parquet'):
        shutil.copy2(OLD / name, OUT / name)
    shutil.copytree(OLD / 'source_snapshot', OUT / 'source_snapshot')
    for name in ('mlp_repair.py', 'mlp_repair_experiment.py'):
        shutil.copy2(REPAIR / 'source_snapshot/src' / name, OUT / 'source_snapshot' / name)
    shutil.copytree(OLD / 'selection_evidence', OUT / 'selection_evidence/original20config')
    evidence = OUT / 'selection_evidence/mlp_repair'
    evidence.mkdir()
    for name in ('plan.json', 'selection.json', 'cv_results.csv', 'candidate_summary.csv', 'summary.json'):
        shutil.copy2(REPAIR / name, evidence / name)
    metric_rows = []
    for item in lineage:
        model, level = item['model'], item['level']
        source = Path(item['source_stage'])
        stage = OUT / model / f'level_{level}'
        if model != 'mlp':
            shutil.copytree(source, stage)
            record = read_json(source / 'metrics.json')
        else:
            stage.mkdir(parents=True)
            shutil.copytree(source / 'fit_checkpoint', stage / 'fit_checkpoint')
            diagnostics = read_json(source / 'fit_diagnostics.json')
            original = read_json(source / 'metrics.json')
            chosen = original['cv_selection']
            record = dict(model=model, level=level, scope=SCOPE, metrics=original['metrics'],
                train_rows=20000, test_rows=61503, raw_features=diagnostics['raw_features'],
                selected_candidate_number=None, selected_variant=chosen['variant'],
                selected_params=dict(model__alpha=chosen['params']['alpha'], model__hidden_layer_sizes=[32],
                    model__batch_size=256, model__learning_rate_init=0.0003,
                    numeric_fixed_clip_z=chosen['params']['clip_z'], selected_epochs=diagnostics['best_epoch']),
                best_cv_log_loss=chosen['mean_cv_log_loss'], warnings=original['warnings'],
                refit_seconds=original['fit_seconds'], predict_seconds=original['predict_seconds'],
                total_pair_seconds=original['fit_seconds'] + original['predict_seconds'],
                timing_note='historical repair time includes internal epoch selection and fresh full20krefit',
                actual_mlp_iterations=[diagnostics['best_epoch']], raw_sklearn_n_iter_=diagnostics['final_optimizer']['raw_n_iter'],
                preprocessing_refit_on_test=False, threshold_selected_using_test=False,
                pipeline_file='fit_checkpoint/pipeline.joblib', fit_diagnostics_file='fit_checkpoint/fit_diagnostics.json')
        record.update(scope=SCOPE, source_lineage=item, new_fit_calls_in_promotion=0)
        atomic_json(stage / 'metrics.json', record)
        atomic_json(stage / 'source_lineage.json', item)
        frame = pd.read_parquet(source / 'probabilities.parquet')
        frame['source_scope'] = item['source_scope']
        frame['scope'] = SCOPE
        frame['evaluation_split'] = 'test'
        require(frame.SK_ID_CURR.is_unique and len(frame) == 61503, 'Wrong test rows')
        frame.to_parquet(stage / 'probabilities.parquet', index=False)
        source_model = source / ('pipeline.joblib' if model == 'mlp' else 'fit_checkpoint/pipeline.joblib')
        require(sha256(stage / 'fit_checkpoint/pipeline.joblib') == sha256(source_model), 'Copied model changed')
        mark(stage / 'fit_checkpoint', manifest['fingerprint'])
        mark(stage, manifest['fingerprint'])
        metric_rows.append(dict(model=model, level=level, **record['metrics']))
    pd.DataFrame(metric_rows).to_csv(OUT / 'final_metrics.csv', index=False, encoding='utf-8-sig')
    after = protected()
    require(before == after, 'An archived input changed during promotion')
    atomic_json(OUT / 'integrity_validation.json', dict(passed=True, protected_files=len(before),
        new_fit_calls=0, new_predict_calls=0, archived_results_unchanged=True))
    summary = dict(scope=SCOPE, status='completed', completed_pairs=28, unchanged_models=24, replaced_mlp=4,
        new_fit_calls=0, new_predict_calls=0, train_rows=20000, test_rows=61503,
        completed_utc=datetime.now(timezone.utc).isoformat(), fingerprint=manifest['fingerprint'])
    atomic_json(OUT / 'experiment_summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
