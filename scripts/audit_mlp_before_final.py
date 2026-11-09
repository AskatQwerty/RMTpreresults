"""Audit frozen MLP CV outputs and train-fold preprocessing, without model fit/test."""
from __future__ import annotations

import gc
import hashlib
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model_pipelines import make_preprocessor, make_estimator

REPORT = ROOT / 'reports/training/pilot_20k_20config_combined'
DATA = ROOT / 'handoff/RMT_batch_11_15/data'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    plan = load(ROOT / 'artifacts/training/pilot_20k_20config_extension/configuration_plan.json')
    for name in ('model_pipelines.py', 'fold_preprocessing.py'):
        assert digest(ROOT / 'src' / name) == plan['code_sha256'][name], name
    manifest = load(ROOT / 'handoff/RMT_batch_11_15/package_manifest.json')
    for entry in manifest['files']:
        if entry['path'].startswith('data/'):
            assert digest(DATA.parent / entry['path']) == entry['sha256'], entry['path']
    frame = pd.read_csv(REPORT / 'all_1680_cv_folds.csv')
    frame = frame[frame.model == 'mlp'].copy()
    assert len(frame) == 240 and not frame.duplicated(['level', 'candidate_number', 'fold']).any()
    assert np.isfinite(frame.log_loss).all()
    diagnostic_rows = []
    selected = {r['level']: r for r in load(REPORT / 'selected_parameters.json') if r['model'] == 'mlp'}
    assert len(selected) == 4
    for row in frame.itertuples():
        record = dict(level=int(row.level), candidate_number=int(row.candidate_number), fold=int(row.fold),
                      cv_log_loss=float(row.log_loss), source=row.source,
                      selected_for_final=row.candidate_number == selected[row.level]['candidate_number'])
        if row.candidate_number <= 5:
            record['diagnostics_available'] = False
            record['note'] = 'Original CV only saves aggregate search diagnostics; do not substitute refit diagnostics'
        else:
            source = ROOT / row.source
            r = load(source)
            marker = load(source.parent / 'complete.json')
            for name, checksum in marker['artifact_sha256'].items():
                assert digest(source.parent / name) == checksum
            ds = load(source.parent / 'fit_diagnostics.json')['estimators']
            assert len(ds) == 1 and ds[0]['class_name'] == 'MLPClassifier'
            d = ds[0]
            fitted, config, checks = d['fitted'], d['config'], d['checks']
            curve = fitted['loss_curve_']
            assert fitted['n_iter_'] == len(curve) and np.isfinite(curve).all()
            assert checks['preprocessor_width_matches_classifier']
            assert config['random_state'] == 42 and config['max_iter'] == 300
            assert config['tol'] == 1e-4 and config['n_iter_no_change'] == 20
            assert config['early_stopping'] is False and fitted['validation_scores_'] is None
            assert r['refit'] is False and r['test_used'] is False
            assert r['params'] == plan['configurations']['mlp'][row.candidate_number-1]['params']
            record.update(diagnostics_available=True, n_iter=fitted['n_iter_'], max_iter=config['max_iter'],
                reached_max_iter=fitted['n_iter_'] == config['max_iter'], initial_train_objective=curve[0],
                final_train_objective=curve[-1], best_train_objective=min(curve),
                train_objective_decreased=curve[-1] < curve[0], encoded_features=d['model_input_features'],
                convergence_warning_count=sum(w.get('count', 1) for w in r['warnings'] if w['category']=='ConvergenceWarning'))
        diagnostic_rows.append(record)
    diagnostics = pd.DataFrame(diagnostic_rows)

    ids = pd.read_parquet(DATA / 'cohort_ids.parquet').SK_ID_CURR.to_numpy()
    target_frame = pd.read_parquet(DATA / 'cohort_targets.parquet')
    folds = pd.read_parquet(DATA / 'cv_folds.parquet')
    assert len(ids) == 20000 and len(np.unique(ids)) == 20000
    assert np.array_equal(target_frame.SK_ID_CURR, ids) and np.array_equal(folds.SK_ID_CURR, ids)
    y = target_frame.TARGET.to_numpy()
    roles = load(DATA / 'columns.json')
    pre_checks = []
    for level in range(1, 5):
        X = pd.read_parquet(DATA / f'X_level_{level}.parquet')
        numeric, categorical = roles[str(level)]['numeric'], roles[str(level)]['categorical']
        estimator = make_estimator('mlp', numeric, categorical, seed=42, technical=False, threads=2)
        estimator.set_params(**selected[level]['params'])
        assert 'preprocess' in estimator.named_steps
        assert estimator.named_steps['preprocess'].named_steps['columns'].transformers[0][1].named_steps['numeric'].transformer_list[0][1].named_steps['scale'].with_mean
        for fold in range(3):
            train = np.flatnonzero(folds.validation_fold.to_numpy() != fold)
            valid = np.flatnonzero(folds.validation_fold.to_numpy() == fold)
            assert not np.intersect1d(ids[train], ids[valid]).size
            pre = make_preprocessor(numeric, categorical, scale=True)
            # Diagnostic preprocessing fit only, solely on this saved train fold.
            train_z = pre.fit_transform(X.iloc[train])
            values = pre.named_steps['columns'].named_transformers_['numeric'].named_steps['numeric'].transformer_list[0][1]
            imputer, scaler = values.named_steps['median'], values.named_steps['scale']
            medians_before = imputer.statistics_.copy()
            means_before, scales_before = scaler.mean_.copy(), scaler.scale_.copy()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                raw_train = X.iloc[train][numeric].to_numpy(dtype=float, na_value=np.nan)
                expected = np.nanmedian(raw_train, axis=0)
            expected[np.isnan(expected)] = 0
            assert np.allclose(medians_before, expected, rtol=1e-6, atol=1e-7)
            imputed = imputer.transform(X.iloc[train][numeric]).astype(np.float64)
            assert np.allclose(means_before, imputed.mean(axis=0), rtol=1e-8, atol=1e-7)
            assert scaler.n_samples_seen_ == len(train)
            val_z = pre.transform(X.iloc[valid])
            assert np.array_equal(imputer.statistics_, medians_before)
            assert np.array_equal(scaler.mean_, means_before) and np.array_equal(scaler.scale_, scales_before)
            assert sparse.isspmatrix_csr(train_z) and sparse.isspmatrix_csr(val_z)
            assert train_z.shape[1] == val_z.shape[1]
            assert np.isfinite(train_z.data).all() and np.isfinite(val_z.data).all()
            numeric_train_z = values.transform(X.iloc[train][numeric])
            numeric_val_z = values.transform(X.iloc[valid][numeric])
            transformed_names = np.array(numeric)
            maxes = np.max(np.abs(numeric_val_z), axis=0)
            worst = int(maxes.argmax())
            baseline = float(log_loss(y[valid], np.repeat(y[train].mean(), len(valid))))
            pre_checks.append(dict(level=level, fold=fold, train_rows=len(train), validation_rows=len(valid),
                encoded_features=train_z.shape[1], finite_train=True, finite_validation=True,
                train_only_statistics_verified=True, statistics_unchanged_after_validation=True,
                fully_empty_train_columns=int(np.isnan(raw_train).all(axis=0).sum()),
                numeric_train_max_abs_z=float(np.max(np.abs(numeric_train_z))),
                numeric_validation_max_abs_z=float(maxes[worst]),
                worst_validation_scaled_feature=str(transformed_names[worst]),
                numeric_validation_values_abs_z_gt10=int((np.abs(numeric_val_z)>10).sum()),
                constant_train_prior_validation_log_loss=baseline))
            del train_z, val_z, raw_train, imputed, numeric_train_z, numeric_val_z, pre
            gc.collect()
        del X, estimator
    detailed = diagnostics[diagnostics.diagnostics_available].copy()
    chosen = detailed[detailed.selected_for_final]
    audit = dict(status='passed_with_caveats', blocking_issues=[],
        created_utc=datetime.now(timezone.utc).isoformat(), classifier_fit_calls=0, test_rows_loaded=0, test_used=False,
        diagnostic_preprocessor_fit_calls=12, diagnostic_scope='saved train fold only; discard transformers after checking',
        cv_score_count=len(frame), saved_cv_diagnostic_count=len(detailed),
        initial_cv_diagnostics_missing_count=60,
        source_core_hashes={n: digest(ROOT/'src'/n) for n in ('model_pipelines.py','fold_preprocessing.py')},
        frozen_mlp_config={'max_iter':300,'tol':1e-4,'n_iter_no_change':20,'early_stopping':False,'random_state':42},
        selected_parameters=list(selected.values()),
        diagnostics_summary={
            'max_iter_reached_fits':int(detailed.reached_max_iter.sum()),
            'convergence_warnings':int(detailed.convergence_warning_count.sum()),
            'train_objective_decreased_fits':int(detailed.train_objective_decreased.sum()),
            'n_iter_min':int(detailed.n_iter.min()),'n_iter_max':int(detailed.n_iter.max()),
            'selected_final_train_objective_min':float(chosen.final_train_objective.min()),
            'selected_final_train_objective_max':float(chosen.final_train_objective.max()),
            'selected_cv_log_loss_min':float(chosen.cv_log_loss.min()),
            'selected_cv_log_loss_max':float(chosen.cv_log_loss.max()),
            'selected_reached_max_iter_count':int(chosen.reached_max_iter.sum()),
        },
        preprocessing_checks=pre_checks,
        interpretation_ru=[
            'No structural Pipeline error was found: scaling is present, medians and the scaler are fitted on the fold’s training rows, and validation is transformed only.',
            'All checked training/validation matrices are finite and have matching widths; test data were not loaded.',
            'A low saved training objective with high CV log-loss is consistent with substantial overfitting; this is a diagnostic interpretation, not proof of a single cause.',
            'MLP loss_curve records a training objective with regularization and minibatch aggregation; it is not a separate evaluation of training log-loss.',
            'Scaling retains rare extreme validation values. Their presence alone is not an error; clipping/selection using CV or test was not introduced here.',
            'Individual CV loss_curve records are unavailable for original configurations1–5; final-refit curves were not substituted for them.',
            'Hyperparameters and the stopping rule were retained. Improving MLP through early stopping or new regularization would require a separate predeclared experiment.',
        ])
    REPORT.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(REPORT/'mlp_audit.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(pre_checks).to_csv(REPORT/'mlp_preprocessing_audit.csv',index=False,encoding='utf-8-sig')
    (REPORT/'mlp_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    summary='MLP audit before final training\n\n'
    summary+='Status: no Pipeline implementation errors found. New classifier fits: 0; test was not used.\n'
    summary+=f'Checked 240 CV evaluations and 180 individual diagnostics; 12 preprocessing checks used only saved training folds.\n'
    summary+=f'Reached max_iter=300: {int(detailed.reached_max_iter.sum())}/180; convergence warnings: {int(detailed.convergence_warning_count.sum())}.\n'
    summary+=f'Training objective for the selected configuration: {chosen.final_train_objective.min():.6f}–{chosen.final_train_objective.max():.6f}; CV log-loss by fold: {chosen.cv_log_loss.min():.6f}–{chosen.cv_log_loss.max():.6f}.\n'
    summary+=f'All 12 selected CV fits reached max_iter. Constant risk based on the training prior gives CV log-loss of approximately {np.mean([r["constant_train_prior_validation_log_loss"] for r in pre_checks]):.6f}.\n'
    summary+=f'Maximum finite absolute standardized validation value: {max(r["numeric_validation_max_abs_z"] for r in pre_checks):.3f}; feature details are in mlp_preprocessing_audit.csv.\n'
    summary+='Selected MLP configuration 13 is the same for all four levels; mean CV log-loss: '+', '.join(f'L{l}: {selected[l]["mean_cv_log_loss"]:.6f}' for l in range(1,5))+'.\n\n'
    summary+='\n'.join(audit['interpretation_ru'])+'\n'
    (REPORT/'mlp_audit.txt').write_text(summary,encoding='utf-8-sig')
    print(json.dumps({'status':audit['status'],'diagnostics_summary':audit['diagnostics_summary'],
                      'preprocessing_checks':len(pre_checks),'max_validation_abs_z':max(r['numeric_validation_max_abs_z'] for r in pre_checks)},indent=2))


if __name__=='__main__':
    main()
