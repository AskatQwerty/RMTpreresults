"""Independent audit of the active 28-model set after the four MLP replacements.

Reads saved files only: no models are unpickled, no fit/predict is called, and
no bootstrap weights are generated. Confidence intervals are checked against
the already saved replicates; MLP decision statistics are also recomputed from
their saved client weights using direct formulas.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from threadpoolctl import threadpool_limits


SCOPE = 'pilot_20k_20config_mlp_updated'
OLD_SCOPE = 'pilot_20k_20config_final'
REPAIR_SCOPE = 'mlp_repair_20k_v1'
THRESHOLDS = (.5, 1 / 6, 1 / 11)
SCENARIOS = ((1, 0, 'L1_t0.5'), (5, 0, 'L5_t0.5'), (5, 1, 'L5_t1/6'),
             (10, 0, 'L10_t0.5'), (10, 2, 'L10_t1/11'))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, message):
    require(np.allclose(actual, expected, rtol=2e-10, atol=2e-12, equal_nan=True),
            f'{message}: actual={actual!r}, expected={expected!r}')


def one(frame, **keys):
    mask = np.ones(len(frame), dtype=bool)
    for key, value in keys.items():
        mask &= np.isclose(frame[key], value, rtol=0, atol=1e-14) if isinstance(value, float) else frame[key].eq(value)
    result = frame.loc[mask]
    require(len(result) == 1, f'Nonunique table lookup: {keys}')
    return result.iloc[0]


def pipeline_path(stage):
    choices = [stage / 'pipeline.joblib', stage / 'fit_checkpoint/pipeline.joblib']
    found = [p for p in choices if p.exists()]
    require(found, f'No Pipeline: {stage}')
    if len(found) == 2:
        require(sha(found[0]) == sha(found[1]), f'Divergent model copies: {stage}')
    return found[0]


def checkpoint(stage, fingerprint):
    marker = read(stage / 'complete.json')
    require(marker['fingerprint'] == fingerprint, f'Checkpoint scope mismatch: {stage}')
    for relative, expected in marker['artifact_sha256'].items():
        path = (stage / relative).resolve()
        require(path.is_relative_to(stage.resolve()), f'Escaping checkpoint: {path}')
        require(path.is_file() and sha(path) == expected, f'Checkpoint SHA mismatch: {path}')
    return marker


def quality(y, p):
    return dict(ap=average_precision_score(y, p), roc_auc=roc_auc_score(y, p),
                log_loss=log_loss(y, p, labels=[0, 1]), brier=brier_score_loss(y, p))


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else float('nan')


def decision_values(y, p, groups, weights):
    """Independent direct mask sums; positive means default/rejection."""
    weight = np.asarray(weights, dtype=np.float64)
    n = float(weight.sum())
    values = []
    for threshold in THRESHOLDS:
        approve = p < threshold
        tn = float(weight[approve & (y == 0)].sum())
        fn = float(weight[approve & (y == 1)].sum())
        fp = float(weight[~approve & (y == 0)].sum())
        tp = float(weight[~approve & (y == 1)].sum())
        row = dict(TN=tn, FP=fp, FN=fn, TP=tp, approved_n=tn + fn,
                   approval_rate=(tn + fn) / n,
                   default_rate_among_approved=ratio(fn, tn + fn),
                   precision=ratio(tp, tp + fp), recall=ratio(tp, tp + fn),
                   f1=ratio(2 * tp, 2 * tp + fp + fn), groups={}, gaps={})
        for attribute, labels in [('gender', ('F', 'M', 'unknown')),
                                  ('age', ('<25', '>=25', 'unknown'))]:
            group_values = []
            for label in labels:
                selected = groups[attribute] == label
                count = float(weight[selected].sum())
                defaults = float(weight[selected & (y == 1)].sum())
                approved = float(weight[selected & approve].sum())
                true_positive = float(weight[selected & ~approve & (y == 1)].sum())
                group_values.append(dict(n=count, default_n=defaults, approved_n=approved,
                    TP=true_positive, approval_rate=ratio(approved, count), TPR=ratio(true_positive, defaults)))
            row['groups'][attribute] = dict(zip(labels, group_values))
            row['gaps'][attribute] = dict(approval_rate_gap=abs(group_values[0]['approval_rate'] - group_values[1]['approval_rate']),
                                          default_tpr_gap=abs(group_values[0]['TPR'] - group_values[1]['TPR']))
        values.append(row)
    profits = [(values[t]['TN'] - loss * values[t]['FN']) / n for loss, t, _ in SCENARIOS]
    return values, profits


def verify_ci(row, values, message):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    interval = np.quantile(finite, [.025, .975], method='linear') if len(finite) else [np.nan, np.nan]
    close([row.ci_low, row.ci_high], interval, message)
    require(row.bootstrap_n_valid == len(finite), message + ': valid count')


def main(project):
    started = time.perf_counter()
    project = Path(project).resolve()
    root = project / 'artifacts/training' / SCOPE
    old = project / 'artifacts/training' / OLD_SCOPE
    repair = project / 'artifacts/training' / REPAIR_SCOPE
    report = project / 'reports' / SCOPE / 'evaluation'
    old_report = project / 'reports' / OLD_SCOPE / 'evaluation'
    prepared = project / 'data/processed/full'
    manifest = read(root / 'experiment_manifest.json')
    evaluation = read(report / 'evaluation_manifest.json')
    require(manifest['scope'] == evaluation['scope'] == SCOPE, 'Wrong active scope')
    require(manifest['train_rows'] == 20000 and manifest['test_rows'] == 61503, 'Wrong client counts')
    require(manifest['inputs'] == read(old / 'experiment_manifest.json')['inputs'], 'Prepared input contract changed')
    for record in manifest['inputs']:
        path = (prepared / record['file']).resolve()
        require(path.is_relative_to(prepared.resolve()) and path.is_file(), 'Invalid prepared input path')
        require(path.stat().st_size == record['bytes'] and sha(path) == record['sha256'],
                f'Prepared input changed: {record["file"]}')
    protected = read(root / 'protected_before.json')
    for relative, record in protected.items():
        path = (project / relative).resolve()
        require(path.is_relative_to(project), 'Protected path escapes project')
        require(path.is_file() and path.stat().st_size == record['bytes']
                and sha(path) == record['sha256'], f'Archived file changed: {relative}')
    require(read(root / 'integrity_validation.json')['passed'], 'Promotion integrity check failed')
    for key, expected in dict(original_completed_candidate_cv_fits=1680, original_models_retained=24,
        mlp_replacement_count=4, supplemental_mlp_outer_cv_evaluations=48,
        supplemental_mlp_variants_per_level=4, test_previously_seen=True,
        mlp_repair_is_posthoc=True, selection_used_test=False).items():
        require(evaluation[key] == expected, f'Incorrect analysis lineage: {key}')
    protocol = read(report / 'actual_training_protocol.json')
    for key in ('original_completed_candidate_cv_fits', 'original_models_retained',
                'mlp_replacement_count', 'supplemental_mlp_outer_cv_evaluations',
                'supplemental_mlp_variants_per_level', 'test_previously_seen', 'mlp_repair_is_posthoc'):
        require(protocol[key] == evaluation[key], f'Report/evaluation protocol mismatch: {key}')
    require(protocol['active_pipeline_count'] == 28, 'Reported active model count')
    require(evaluation['n_bootstrap'] == 1000 and evaluation['seed'] == 42, 'Bootstrap contract changed')
    for name in ('cohort_ids.parquet', 'cohort_targets.parquet', 'cv_folds.parquet', 'test_ids.parquet'):
        require(sha(root / name) == sha(old / name), f'Frozen cohort/split changed: {name}')
    ids = pd.read_parquet(root / 'test_ids.parquet').SK_ID_CURR.to_numpy()
    require(np.array_equal(ids, pd.read_parquet(prepared / 'splits/test_ids.parquet').SK_ID_CURR), 'Prepared test changed')
    target = pd.read_parquet(prepared / 'target.parquet').set_index('SK_ID_CURR')
    fairness = pd.read_parquet(prepared / 'fairness.parquet').set_index('SK_ID_CURR').loc[ids]
    y = target.loc[ids].TARGET.to_numpy()
    require(len(ids) == len(set(ids)) == 61503 and int(y.sum()) == 4965, 'Test balance/IDs changed')
    require(set(ids).isdisjoint(pd.read_parquet(root / 'cohort_ids.parquet').SK_ID_CURR), 'Train/test overlap')
    groups = {'gender': fairness.GENDER_GROUP.to_numpy(str), 'age': fairness.AGE_GROUP.to_numpy(str)}
    quality_frame = pd.read_csv(report / 'quality_point.csv')
    pairs = evaluation['pairs']
    require(len(pairs) == 28 and len({(p['model'], p['level']) for p in pairs}) == 28, 'Pair set')
    models = [p['model'] for p in pairs if p['level'] == 4]
    require(len(set(models)) == 7 and set(p['level'] for p in pairs) == {1, 2, 3, 4}, 'Model/level set')
    lineage = {(p['model'], p['level']): p for p in manifest['model_lineage']}
    records, mlp_probability = [], None
    for pair in pairs:
        model, level = pair['model'], pair['level']
        stage = root / model / f'level_{level}'
        source = repair / 'final' / f'level_{level}' if model == 'mlp' else old / model / f'level_{level}'
        checkpoint(stage, manifest['fingerprint'])
        require(sha(pipeline_path(stage)) == sha(pipeline_path(source)), f'Pipeline changed: {model}/L{level}')
        require(lineage[model, level]['source_scope'] == (REPAIR_SCOPE if model == 'mlp' else OLD_SCOPE), 'Model lineage scope')
        require(lineage[model, level]['source_complete_sha256'] == sha(source / 'complete.json'), 'Model lineage checkpoint SHA')
        active = pd.read_parquet(stage / 'probabilities.parquet')
        source_p = pd.read_parquet(source / 'probabilities.parquet')
        require(active.SK_ID_CURR.is_unique and np.array_equal(active.SK_ID_CURR, ids), f'Active ID order: {pair}')
        require(np.array_equal(source_p.SK_ID_CURR, ids), f'Source ID order: {pair}')
        require(np.array_equal(active.TARGET, y) and np.array_equal(source_p.TARGET, y), f'TARGET: {pair}')
        require(np.array_equal(active.probability_default, source_p.probability_default), f'Probability vector changed: {pair}')
        for name, expected in [('scope', SCOPE), ('model', model), ('level', level), ('evaluation_split', 'test')]:
            require(active[name].eq(expected).all(), f'Provenance: {pair}/{name}')
        p = active.probability_default.to_numpy(float)
        recomputed = quality(y, p)
        row = one(quality_frame, model=model, level=level)
        for name, value in recomputed.items():
            close(row[name], value, f'Quality {pair}/{name}')
        if model == 'mlp' and level == 4:
            mlp_probability = p
        records.append({**pair, 'pipeline_sha256_matches_source': True,
                        'probability_vector_identical_to_source': True, 'quality_recomputed': True})
    # Point estimates independently recomputed for all three operating thresholds.
    decisions, profits = decision_values(y, mlp_probability, groups, np.ones(len(y)))
    thresholds = pd.read_csv(report / 'threshold_metrics.csv')
    profit_table = pd.read_csv(report / 'profit_point.csv')
    group_table = pd.read_csv(report / 'fairness_groups.csv')
    gap_table = pd.read_csv(report / 'fairness_gaps.csv')
    for t, threshold in enumerate(THRESHOLDS):
        actual = one(thresholds, model='mlp', threshold=threshold)
        for key, value in decisions[t].items():
            if key not in ('groups', 'gaps'):
                close(actual[key], value, f'MLP threshold {threshold}/{key}')
        gap = one(gap_table, model='mlp', threshold=threshold)
        for attribute in ('gender', 'age'):
            for label, values in decisions[t]['groups'][attribute].items():
                actual_group = one(group_table, model='mlp', threshold=threshold, attribute=attribute, group=label)
                for key, value in values.items():
                    close(actual_group[key], value, f'MLP group {attribute}/{label}/{key}')
            close(gap[f'{attribute}_approval_rate_gap'], decisions[t]['gaps'][attribute]['approval_rate_gap'], 'MLP approval gap')
            close(gap[f'{attribute}_tpr_gap'], decisions[t]['gaps'][attribute]['default_tpr_gap'], 'MLP TPR gap')
    for index, (_, _, scenario) in enumerate(SCENARIOS):
        close(one(profit_table, model='mlp', scenario=scenario).realized_profit_per_applicant,
              profits[index], 'MLP profit ' + scenario)
    # Check immutable saved bootstrap vectors and independently reproduce MLP
    # decision statistics from existing weights. No resampling is performed.
    mlp_index = models.index('mlp')
    arrays = {}
    wanted = ['ap', 'roc_auc_l4', 'log_loss_l4', 'brier_l4', 'decision_profit',
              'decision_approval_rate', 'decision_default_rate_among_approved',
              'decision_precision', 'decision_recall', 'decision_f1', 'decision_confusion',
              'decision_approval_gap', 'decision_tpr_gap']
    bootstrap = report / 'bootstrap'
    require(len(list(bootstrap.glob('replicate_*.npz'))) == 1000, 'Bootstrap not complete')
    non_mlp = [i for i, model in enumerate(models) if model != 'mlp']
    non_mlp_pairs = [i for i, pair in enumerate(pairs) if pair['model'] != 'mlp']
    order = np.argsort(ids)
    for r in range(1000):
        path = bootstrap / f'replicate_{r:04d}.npz'
        marker = read(path.with_suffix('.json'))
        require(marker['replicate'] == r and marker['sha256'] == sha(path), f'Bootstrap checksum {r}')
        with np.load(path, allow_pickle=False) as data, np.load(old_report / 'bootstrap' / path.name, allow_pickle=False) as prior:
            require(str(data['fingerprint']) == evaluation['fingerprint'] and int(data['replicate']) == r, 'Bootstrap provenance')
            require(np.array_equal(data['weights'], prior['weights']), 'Paired bootstrap weights changed')
            require(int(data['weights'].sum()) == len(y), 'Bootstrap sample size')
            # Evaluation canonical order is ascending ID, so adapt only the
            # saved arrays to that ordering before direct weighted sums.
            values, profit = decision_values(y[order], mlp_probability[order],
                {k: v[order] for k, v in groups.items()}, data['weights'])
            close(data['decision_profit'][mlp_index], profit, f'MLP weighted profit {r}')
            for t in range(3):
                for key in ('approval_rate', 'default_rate_among_approved', 'precision', 'recall', 'f1'):
                    close(data['decision_' + key][mlp_index, t], values[t][key], f'MLP bootstrap {r}/{key}')
                close(data['decision_confusion'][mlp_index, t], [values[t][k] for k in ('TN','FP','FN','TP')], 'MLP bootstrap confusion')
                for a, attribute in enumerate(('gender', 'age')):
                    close(data['decision_approval_gap'][mlp_index, t, a], values[t]['gaps'][attribute]['approval_rate_gap'], 'MLP bootstrap approval gap')
                    close(data['decision_tpr_gap'][mlp_index, t, a], values[t]['gaps'][attribute]['default_tpr_gap'], 'MLP bootstrap TPR gap')
            for key in wanted:
                values_array = data[key]
                rows = non_mlp_pairs if key == 'ap' else non_mlp
                close(values_array[rows], prior[key][rows], f'Unchanged non-MLP bootstrap {r}/{key}')
                arrays.setdefault(key, []).append(values_array.copy())
    arrays = {key: np.stack(values) for key, values in arrays.items()}
    profit_ci = pd.read_csv(report / 'profit_ci.csv')
    fairness_ci = pd.read_csv(report / 'fairness_ci.csv')
    threshold_ci = pd.read_csv(report / 'threshold_metrics_ci.csv')
    quality_ci = pd.read_csv(report / 'quality_ci.csv')
    ci_checks = 0
    for row in profit_ci.loc[profit_ci.model == 'mlp'].itertuples():
        s = next(i for i, x in enumerate(SCENARIOS) if x[2] == row.scenario)
        t = SCENARIOS[s][1]
        key = 'decision_profit' if row.metric == 'realized_profit_per_applicant' else 'decision_' + row.metric
        verify_ci(row, arrays[key][:, mlp_index, s if key == 'decision_profit' else t], 'MLP profit CI')
        ci_checks += 1
    for row in fairness_ci.loc[fairness_ci.model == 'mlp'].itertuples():
        t = next(i for i, v in enumerate(THRESHOLDS) if np.isclose(v, row.threshold))
        a = ('gender', 'age').index(row.attribute)
        key = 'decision_approval_gap' if row.metric == 'approval_rate_gap' else 'decision_tpr_gap'
        verify_ci(row, arrays[key][:, mlp_index, t, a], 'MLP fairness CI')
        ci_checks += 1
    for row in threshold_ci.loc[threshold_ci.model == 'mlp'].itertuples():
        t = next(i for i, v in enumerate(THRESHOLDS) if np.isclose(v, row.threshold))
        values = arrays['decision_confusion'][:, mlp_index, t, ('TN','FP','FN','TP').index(row.metric)] if row.metric.isupper() else arrays['decision_' + row.metric][:, mlp_index, t]
        verify_ci(row, values, 'MLP threshold CI'); ci_checks += 1
    for row in quality_ci.loc[quality_ci.model == 'mlp'].itertuples():
        if row.metric == 'ap':
            position = pairs.index({'model': 'mlp', 'level': row.level})
            values = arrays['ap'][:, position]
        elif row.metric in ('roc_auc', 'log_loss', 'brier'):
            values = arrays[row.metric + '_l4'][:, mlp_index]
        else:
            values = arrays['decision_' + row.metric][:, mlp_index, 0]
        verify_ci(row, values, 'MLP quality CI'); ci_checks += 1
    rq2 = pd.read_csv(report / 'rq2_deltas.csv')
    for row in rq2.loc[rq2.model == 'mlp'].itertuples():
        a = pairs.index({'model': 'mlp', 'level': row.from_level})
        b = pairs.index({'model': 'mlp', 'level': row.to_level})
        verify_ci(row, arrays['ap'][:, b] - arrays['ap'][:, a], 'MLP RQ2 delta CI'); ci_checks += 1
    result = dict(passed=True, scope=SCOPE, verified_pairs=28, unchanged_original_models=24,
        repaired_mlp_models=4, aligned_test_rows=len(y), original_train_and_cv_unchanged=True,
        quality_metrics_recomputed=112, mlp_profit_scenarios_checked=5,
        protected_archived_files_unchanged=len(protected),
        prepared_input_files_unchanged=len(manifest['inputs']),
        mlp_fairness_thresholds_checked=3, bootstrap_replicates_read=1000,
        mlp_decision_replicates_independently_recomputed=1000,
        non_mlp_bootstrap_statistics_unchanged=True, mlp_ci_checks=ci_checks,
        new_fit_calls=0, new_predict_calls=0, new_bootstrap_resamples=0,
        posthoc_mlp_lineage_explicit=True, model_checks=records,
        elapsed_seconds=time.perf_counter()-started, completed_utc=datetime.now(timezone.utc).isoformat())
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        result = main(args.project)
    if args.output:
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
