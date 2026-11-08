"""Read-only validation of final28fits and preservation of the original pilot."""
from pathlib import Path
import hashlib
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.final_pilot_training import protected_snapshot, MODEL_ORDER, LEVELS
from src.model_training import _complete, read_json, atomic_json, sha256


def main():
    root = ROOT / 'artifacts/training/pilot_20k_20config_final'
    report = ROOT / 'reports/pilot_20k_20config_final/evaluation'
    context = read_json(root / 'experiment_manifest.json')
    selection = {(s['model'], s['level']): s for s in read_json(root / 'selection_evidence/selected_parameters.json')}
    expected = {(m, l) for m in MODEL_ORDER for l in LEVELS}
    events = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines() if line.strip()]
    for event in ('final_fit_started', 'final_fit_completed', 'pair_completed'):
        matching = [(x['model'], x['level']) for x in events if x['event'] == event]
        assert len(matching) == 28 and set(matching) == expected, event
    rows, fitted_parameters = [], []
    for model, level in sorted(expected):
        stage = root / model / f'level_{level}'
        assert _complete(stage, context) and _complete(stage / 'fit_checkpoint', context)
        chosen = selection[model, level]
        params = read_json(stage / 'fit_checkpoint/best_params.json')
        assert params['params'] == chosen['params'] and params['candidate_number'] == chosen['candidate_number']
        m = read_json(stage / 'metrics.json')
        assert m['train_rows'] == 20000 and m['test_rows'] == 61503
        assert m['fit_calls']['candidate_cv'] == 0 and m['fit_calls']['manual_final_refit'] == 1
        assert not m['preprocessing_refit_on_test'] and not m['threshold_selected_using_test']
        d = read_json(stage / 'fit_checkpoint/fit_diagnostics.json')
        for e in d['estimators']:
            assert e['checks'].get('preprocessor_width_matches_classifier', True)
            assert e['checks'].get('loss_curve_length_matches_n_iter', True)
        monitors = list((root / 'launches').glob(f'*/{model}_level_{level}/monitor_result.json'))
        assert len(monitors) == 1
        mon = read_json(monitors[0]); assert mon['status'] == 'completed' and mon['exit_code'] == 0
        rows.append(dict(model=model, level=level, selected_candidate=chosen['candidate_number'],
            train_rows=20000, test_rows=61503, raw_features=m['raw_features'],
            processed_feature_counts=json.dumps(d['processed_feature_counts']),
            fit_seconds=m['refit_seconds'], predict_seconds=m['predict_seconds'],
            peak_tree_rss_gib=mon['peak_tree_rss_gib'],
            pipeline_bytes=(stage / 'fit_checkpoint/pipeline.joblib').stat().st_size,
            warnings_count=sum(w.get('count', 1) for w in m['warnings'])))
        fitted_parameters.append(dict(model=model, level=level, candidate_number=chosen['candidate_number'],
            params=chosen['params'], fitted_final_pipeline=True, final_training_seed=42,
            pipeline_file=str(stage / 'fit_checkpoint/pipeline.joblib'),
            pipeline_sha256=sha256(stage / 'fit_checkpoint/pipeline.joblib')))
    before = read_json(root / 'old_preresults_integrity_before.json')
    after = protected_snapshot(ROOT)
    differences = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    atomic_json(root / 'old_preresults_integrity_after.json', after)
    validation = dict(passed=not differences, checked_files=len(before), mismatches=differences,
        old_models_and_reports_unchanged=not differences,
        before_manifest_sha256=sha256(root / 'old_preresults_integrity_before.json'),
        after_manifest_sha256=sha256(root / 'old_preresults_integrity_after.json'))
    atomic_json(root / 'old_preresults_integrity_validation.json', validation)
    assert not differences, 'Original preliminary outputs changed'
    report.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(report / 'final_training_resources.csv', index=False, encoding='utf-8-sig')
    atomic_json(report / 'fitted_selected_parameters.json', fitted_parameters)
    result = dict(passed=True, verified_pairs=28, unique_train_only_final_fits=28,
        saved_test_predictions=28, new_cv_fits=0, old_preservation=validation,
        total_pipeline_bytes=sum(r['pipeline_bytes'] for r in rows),
        total_final_fit_seconds=sum(r['fit_seconds'] for r in rows),
        total_predict_seconds=sum(r['predict_seconds'] for r in rows),
        peak_pair_rss_gib=max(r['peak_tree_rss_gib'] for r in rows))
    atomic_json(report / 'final_training_validation.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
