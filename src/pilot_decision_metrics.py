"""Описательные profit/fairness/calibration-таблицы пилота из сохранённых прогнозов.

Нет загрузки моделей, fit, predict или выбора порогов по test. Числа являются
предварительными свидетельствами для RQ1, а не итоговым правилом выбора A2.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd


SCOPE = 'pilot_20k_5config'
MODELS = ('logistic_regression', 'lightgbm', 'xgboost', 'soft_voting',
          'stacking', 'mlp', 'random_forest')
RATIOS = (1, 5, 10)
SCENARIO_COLUMNS = ('model', 'level', 'loss_to_margin', 'threshold_rule', 'threshold',
    'rows', 'approved_n', 'approval_rate', 'realized_profit_per_applicant',
    'gender_approval_rate_gap', 'gender_default_tpr_gap', 'gender_unknown_n',
    'age_approval_rate_gap', 'age_default_tpr_gap', 'age_unknown_n')
GROUP_COLUMNS = ('model', 'level', 'loss_to_margin', 'threshold_rule', 'threshold',
    'attribute', 'group', 'included_in_binary_gap', 'n', 'default_n', 'repaid_n',
    'approved_n', 'predicted_default_n', 'true_positive_default_n',
    'approval_rate', 'default_tpr', 'realized_profit_per_applicant')
CALIBRATION_COLUMNS = ('model', 'level', 'bin', 'lower_bound', 'upper_bound',
    'upper_bound_inclusive', 'n', 'default_n', 'mean_probability_default',
    'observed_default_rate')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    # pandas.to_csv(None) уже использует системные окончания строк. Не переводим
    # LF повторно при записи, иначе Windows превращает CRLF в CRCRLF.
    temporary.write_text(value, encoding='utf-8', newline='')
    for attempt in range(6):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as exc:
            if getattr(exc, 'winerror', None) not in {5, 32, 33} or attempt == 5:
                raise
            time.sleep(.05 * (attempt + 1))


def _required_hash(path, expected, description):
    if not isinstance(expected, str) or not path.is_file() or _hash(path) != expected:
        raise ValueError(f'Не совпадает SHA-256: {description}')


def _load_fairness(root, prepared, manifest):
    inputs = {entry['file']: entry['sha256'] for entry in manifest.get('inputs', [])}
    for name in ('fairness.parquet', 'target.parquet', 'splits/test_ids.parquet'):
        _required_hash(prepared / name, inputs.get(name), name)
    contract = _read(root / 'cohort_contract.json')
    _required_hash(root / 'test_ids.parquet', contract['artifact_sha256'].get('test_ids.parquet'),
                   'pilot/test_ids.parquet')
    test_ids = pd.read_parquet(root / 'test_ids.parquet').SK_ID_CURR.to_numpy()
    original = pd.read_parquet(prepared / 'splits/test_ids.parquet').SK_ID_CURR.to_numpy()
    if not np.array_equal(test_ids, original) or len(np.unique(test_ids)) != len(test_ids):
        raise ValueError('Пилотные test ID отличаются от сохранённого test или повторяются')
    fairness = pd.read_parquet(prepared / 'fairness.parquet', columns=[
        'SK_ID_CURR', 'CODE_GENDER', 'GENDER_GROUP', 'AGE_YEARS', 'AGE_GROUP'])
    target = pd.read_parquet(prepared / 'target.parquet', columns=['SK_ID_CURR', 'TARGET'])
    if not fairness.SK_ID_CURR.is_unique or not target.SK_ID_CURR.is_unique:
        raise ValueError('ID в fairness/target не уникальны')
    frame = fairness.set_index('SK_ID_CURR').reindex(test_ids)
    labels = target.set_index('SK_ID_CURR').TARGET.reindex(test_ids)
    if not labels.isin([0, 1]).all() or not set(test_ids) <= set(fairness.SK_ID_CURR):
        raise ValueError('Test ID не полностью покрыты fairness/target')
    gender = frame.CODE_GENDER.where(frame.CODE_GENDER.isin(['F', 'M']), 'unknown')
    if not np.array_equal(gender.to_numpy(), frame.GENDER_GROUP.to_numpy()):
        raise ValueError('Сохранённый GENDER_GROUP не соответствует CODE_GENDER')
    age = pd.to_numeric(frame.AGE_YEARS, errors='coerce').to_numpy(dtype=float)
    expected_age = np.where(np.isfinite(age), np.where(age < 25, '<25', '>=25'), 'unknown')
    if not np.array_equal(frame.AGE_GROUP.to_numpy(), expected_age):
        raise ValueError('Сохранённый AGE_GROUP нарушает существующее правило <25 / >=25')
    # Группы берём из существующего fairness; равенство 25 явно относится к >=25.
    return test_ids, labels.to_numpy(dtype=int), {
        'CODE_GENDER': gender.to_numpy(), 'AGE_GROUP': frame.AGE_GROUP.to_numpy()}, {
        name: inputs[name] for name in ('fairness.parquet', 'target.parquet', 'splits/test_ids.parquet')}


def _validated_probabilities(stage, model, manifest, test_ids, target):
    marker = _read(stage / 'complete.json')
    if marker.get('fingerprint') != manifest.get('fingerprint'):
        raise ValueError('Fingerprint пары не совпадает с manifest')
    hashes = marker.get('artifact_sha256', {})
    for name in ('probabilities.parquet', 'metrics.json'):
        _required_hash(stage / name, hashes.get(name), f'{model}/level_4/{name}')
    record = _read(stage / 'metrics.json')
    if (record.get('scope'), record.get('model'), record.get('level')) != (SCOPE, model, 4):
        raise ValueError('Неверные scope/model/level в metrics.json')
    frame = pd.read_parquet(stage / 'probabilities.parquet')
    required = {'SK_ID_CURR', 'TARGET', 'probability_default', 'scope', 'model', 'level', 'evaluation_split'}
    if not required <= set(frame.columns):
        raise ValueError('Неполная схема probabilities.parquet')
    if not frame.SK_ID_CURR.is_unique or not np.array_equal(frame.SK_ID_CURR, test_ids):
        raise ValueError('ID/порядок прогнозов не совпадают с сохранённым test')
    if not np.array_equal(frame.TARGET, target):
        raise ValueError('TARGET прогнозов отличается от сохранённого target')
    for field, expected in [('scope', SCOPE), ('model', model), ('level', 4), ('evaluation_split', 'test')]:
        if not (frame[field] == expected).all():
            raise ValueError(f'Неверное поле {field} в прогнозах')
    probabilities = frame.probability_default.to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or not ((0 <= probabilities) & (probabilities <= 1)).all():
        raise ValueError('Вероятности должны быть конечными и лежать в [0,1]')
    if record.get('test_rows') != len(test_ids) or record.get('metrics', {}).get('rows') != len(test_ids):
        raise ValueError('Количество test-строк не совпадает с metrics.json')
    brier = float(np.mean((probabilities - target)**2))
    if not math.isclose(brier, record['metrics']['brier_score'], rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError('Brier в metrics.json не соответствует сохранённым прогнозам')
    return probabilities


def _rate(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _gap(rows, field):
    values = [row[field] for row in rows if row['included_in_binary_gap']]
    if len(values) != 2 or any(value is None for value in values):
        return None
    return float(abs(values[0] - values[1]))


def _decision_rows(model, target, probabilities, groups):
    scenarios, details = [], []
    for ratio in RATIOS:
        for rule, threshold in [('reference_0_5', .5), ('break_even', 1 / (1 + ratio))]:
            approve = probabilities < threshold
            predicted_default = ~approve
            realized = approve.astype(float) * ((1 - target) - ratio * target)
            common = {'model': model, 'level': 4, 'loss_to_margin': ratio,
                      'threshold_rule': rule, 'threshold': threshold}
            scenario = {**common, 'rows': len(target), 'approved_n': int(approve.sum()),
                        'approval_rate': float(approve.mean()),
                        'realized_profit_per_applicant': float(realized.mean())}
            for attribute, known, prefix in [('CODE_GENDER', ('F', 'M'), 'gender'),
                                              ('AGE_GROUP', ('<25', '>=25'), 'age')]:
                group_rows = []
                for group in (*known, 'unknown'):
                    mask = groups[attribute] == group
                    n = int(mask.sum())
                    defaults = int(target[mask].sum())
                    approved = int(approve[mask].sum())
                    true_positive = int((predicted_default[mask] & (target[mask] == 1)).sum())
                    row = {**common, 'attribute': attribute, 'group': group,
                        'included_in_binary_gap': group in known, 'n': n,
                        'default_n': defaults, 'repaid_n': n - defaults, 'approved_n': approved,
                        'predicted_default_n': int(predicted_default[mask].sum()),
                        'true_positive_default_n': true_positive,
                        'approval_rate': _rate(approved, n), 'default_tpr': _rate(true_positive, defaults),
                        'realized_profit_per_applicant': float(realized[mask].mean()) if n else None}
                    group_rows.append(row)
                    details.append(row)
                scenario[f'{prefix}_approval_rate_gap'] = _gap(group_rows, 'approval_rate')
                scenario[f'{prefix}_default_tpr_gap'] = _gap(group_rows, 'default_tpr')
                scenario[f'{prefix}_unknown_n'] = int((groups[attribute] == 'unknown').sum())
            scenarios.append(scenario)
    return scenarios, details


def _calibration_rows(model, target, probabilities):
    # Фиксированные интервалы [k/10,(k+1)/10); последнее включает p=1.
    bins = np.minimum(np.floor(probabilities * 10).astype(int), 9)
    rows = []
    for index in range(10):
        mask = bins == index
        n = int(mask.sum())
        defaults = int(target[mask].sum())
        rows.append({'model': model, 'level': 4, 'bin': index,
            'lower_bound': index / 10, 'upper_bound': (index + 1) / 10,
            'upper_bound_inclusive': index == 9, 'n': n, 'default_n': defaults,
            'mean_probability_default': float(probabilities[mask].mean()) if n else None,
            'observed_default_rate': _rate(defaults, n)})
    return rows


def _escape(value):
    mapping = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
               '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}'}
    return ''.join(mapping.get(char, char) for char in str(value))


def _latex(frame, columns, caption):
    lines = [r'\begin{longtable}{' + 'l' * len(columns) + '}',
             r'\caption{' + _escape(caption) + r'}\\', r'\toprule',
             ' & '.join(_escape(title) for _, title in columns) + r' \\',
             r'\midrule', r'\endhead']
    for _, row in frame.iterrows():
        values = []
        for name, _ in columns:
            value = row[name]
            if value is None or pd.isna(value):
                values.append('--')
            elif isinstance(value, (float, np.floating)):
                values.append(f'{value:.4f}')
            else:
                values.append(_escape(value))
        lines.append(' & '.join(values) + r' \\')
    return '\n'.join(lines + [r'\bottomrule', r'\end{longtable}', ''])


def export_decision_metrics(root, report_dir, prepared_dir=None):
    """Экспортировать фиксированные сценарии для завершённых Level4, без fit/predict.

    Возвращает JSON-safe metadata. Неполные пары pending; неверные SHA/ID/target
    отмечаются invalid и исключаются из числовых таблиц. Повреждённые общие
    fairness/target/test ID вызывают ошибку, а не создают правдоподобный отчёт.
    """
    root, report_dir = Path(root), Path(report_dir)
    prepared = Path(prepared_dir) if prepared_dir is not None else Path(__file__).resolve().parents[1] / 'data/processed/full'
    manifest = _read(root / 'experiment_manifest.json')
    if manifest.get('mode') != SCOPE:
        raise ValueError(f'Ожидался scope {SCOPE}')
    test_ids, target, groups, input_hashes = _load_fairness(root, prepared, manifest)
    completed, pending, invalid = [], [], []
    scenarios, details, calibration = [], [], []
    for model in MODELS:
        stage = root / model / 'level_4'
        if not (stage / 'complete.json').exists():
            pending.append(model)
            continue
        try:
            probabilities = _validated_probabilities(stage, model, manifest, test_ids, target)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            invalid.append({'model': model, 'reason': str(exc)})
            continue
        model_scenarios, model_details = _decision_rows(model, target, probabilities, groups)
        scenarios.extend(model_scenarios)
        details.extend(model_details)
        calibration.extend(_calibration_rows(model, target, probabilities))
        completed.append(model)
    scenario_frame = pd.DataFrame(scenarios, columns=SCENARIO_COLUMNS)
    group_frame = pd.DataFrame(details, columns=GROUP_COLUMNS)
    calibration_frame = pd.DataFrame(calibration, columns=CALIBRATION_COLUMNS)
    files = {name: str((report_dir / name).resolve()) for name in (
        'decision_scenarios.csv', 'decision_scenarios.tex', 'decision_groups.csv',
        'decision_groups.tex', 'calibration_bins.csv', 'decision_metadata.json')}
    for frame, name in [(scenario_frame, 'decision_scenarios.csv'),
                        (group_frame, 'decision_groups.csv'), (calibration_frame, 'calibration_bins.csv')]:
        _atomic_text(report_dir / name, frame.to_csv(index=False))
    _atomic_text(report_dir / 'decision_scenarios.tex', _latex(scenario_frame, [
        ('model', 'Model'), ('loss_to_margin', 'L/m'), ('threshold_rule', 'Rule'),
        ('threshold', 't'), ('realized_profit_per_applicant', 'Profit/applicant'),
        ('approval_rate', 'Approval'), ('gender_approval_rate_gap', 'Gender gap'),
        ('age_approval_rate_gap', 'Age gap')], 'Level 4: fixed normalized profit and approval scenarios; descriptive pilot'))
    _atomic_text(report_dir / 'decision_groups.tex', _latex(group_frame, [
        ('model', 'Model'), ('loss_to_margin', 'L/m'), ('threshold_rule', 'Rule'),
        ('attribute', 'Attribute'), ('group', 'Group'), ('n', 'N'), ('default_n', 'Defaults'),
        ('approval_rate', 'Approval'), ('default_tpr', 'Default TPR')],
        'Level 4: group counts and decision rates; unknown groups shown separately'))
    metadata = {'scope': SCOPE, 'status': 'completed' if len(completed) == 7 else 'partial',
        'completed_level4_models': completed, 'pending_models': pending, 'invalid_models': invalid,
        'level': 4, 'test_rows': len(test_ids), 'scenario_rows': len(scenarios),
        'group_rows': len(details), 'calibration_rows': len(calibration), 'files': files,
        'input_sha256': input_hashes, 'all_models_descriptive': True,
        'eligibility_filter_applied': False, 'winner_selected': False,
        'test_threshold_optimization': False, 'additional_fit_calls': 0, 'additional_predict_calls': 0,
        'fairness_correction_applied': False,
        'definitions': {
            'profit': 'Normalized m=1, L in {1,5,10}; realized test mean of I(p<t)*[(1-y)-L*y] over ALL applicants; rejection yields 0. Not observed currency profit.',
            'approval': 'p(default)<t; equality rejects. Fixed t=0.5 and t=1/(1+L/m), chosen before reading test predictions. At L/m=1 both rules coincide intentionally.',
            'default_tpr': 'P(p(default)>=t | TARGET=1, group): detected defaults / actual defaults; TARGET=1 is positive, not successful repayment.',
            'gaps': 'Absolute difference between F and M, or <25 and >=25; proportions, not percentage points. Unknown excluded from binary gaps and reported separately. Undefined denominators remain missing.',
            'age': 'Existing AGE_GROUP reused: <25 and >=25 based on AGE_YEARS=-DAYS_BIRTH/365.25. A2 says below/above25 without explicit equality; preserved preparation assigns age25 to >=25.',
            'gender': 'Existing CODE_GENDER F/M; other or missing values are unknown, checked against saved GENDER_GROUP.',
            'calibration': '10 fixed-width bins [k/10,(k+1)/10), last includes1; empty bins have n=0 and missing means. Descriptive curve only, no calibration fit.'},
        'limitations': [
            'All seven Level4 models shown descriptively; A2 applies profit/fairness to accuracy-eligible models. Eligibility and final joint selection are pending.',
            'Single20k cohort and5 search configurations; five-repeat uncertainty is not measured.',
            'Formal bootstrap confidence intervals, DeLong and Bonferroni tests are pending.',
            'No reweighting or group-specific threshold correction; no SHAP. Brier remains in main metrics.',
            'Profit ratios are assumptions; normalized realized averages are not actual lender revenue.'],
        'source': 'Assignment2 Methods and Materials, pp2-3; preserved fairness.parquet boundary convention'}
    _atomic_text(report_dir / 'decision_metadata.json', json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False))
    return metadata
