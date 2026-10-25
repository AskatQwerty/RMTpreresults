"""Чистые формулы evaluation: фиксированные решения и общий клиентский bootstrap.

Никаких файлов, моделей, fit или predict. TARGET=1 — дефолт; одобрение p<t,
поэтому p==t отклоняется. Неизвестные группы остаются в общих показателях.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


THRESHOLDS = (0.5, 1 / 6, 1 / 11)
ATTRIBUTES = ('gender', 'age')
GROUP_LABELS = (('F', 'M', 'unknown'), ('<25', '>=25', 'unknown'))
LOSS_RATIOS = (1, 5, 10)
# (L/m, индекс порога): общий L/m=1,t=.5 не дублируется.
SCENARIOS = ((1, 0), (5, 0), (5, 1), (10, 0), (10, 2))
SCENARIO_NAMES = ('L1_t0.5', 'L5_t0.5', 'L5_t1/6', 'L10_t0.5', 'L10_t1/11')
CONFUSION_ORDER = ('TN', 'FP', 'FN', 'TP')
BASELINE_ORDER = ('approve_all', 'reject_all')


def _validate_data(y, probabilities_by_model):
    target = np.asarray(y)
    if target.ndim != 1 or len(target) == 0 or not np.isin(target, [0, 1]).all():
        raise ValueError('y должен быть непустым одномерным массивом TARGET 0/1')
    if not isinstance(probabilities_by_model, dict) or not probabilities_by_model:
        raise ValueError('Нужен непустой dict model -> probability array')
    models = tuple(probabilities_by_model)
    values = []
    for model in models:
        probability = np.asarray(probabilities_by_model[model], dtype=np.float64)
        if probability.shape != target.shape:
            raise ValueError(f'Неверный размер вероятностей: {model}')
        if not np.isfinite(probability).all() or not ((0 <= probability) & (probability <= 1)).all():
            raise ValueError(f'Вероятности вне [0,1] или не конечны: {model}')
        values.append(probability)
    return target.astype(np.float64), models, np.stack(values)


def _divide(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    result = np.full(numerator.shape, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator != 0)
    return result


def _observed_approval_flag(gap):
    return bool(gap <= .05) if np.isfinite(gap) else pd.NA


class DecisionEvaluator:
    """Предвычислить маски один раз и применять общие веса клиентов много раз.

    weighted(weights) возвращает только numpy-массивы. Оси указаны в coordinates.
    Веса относятся ко всем исходным клиентам, одинаковы для каждой модели и
    группы. Группы НЕ ресемплируются отдельно. Нулевые знаменатели дают NaN.
    Параметры модели и вероятности не меняются между bootstrap-репликациями.
    """

    def __init__(self, y, probabilities_by_model, groups):
        self.y, self.models, self.probabilities = _validate_data(y, probabilities_by_model)
        self.n = len(self.y)
        self.shape = (len(self.models), len(THRESHOLDS))
        self.thresholds = np.asarray(THRESHOLDS, dtype=np.float64)
        self.scenarios = SCENARIOS
        self.group_values = {}
        self._group_cache = []
        # Кэшируются только наблюдаемые строки каждой группы. Это уменьшает
        # работу dot по сравнению с плотной матрицей с нулями для всех групп.
        approve = (self.probabilities[:, None, :] < self.thresholds[None, :, None])
        flat_approve = approve.reshape(-1, self.n)
        for attribute, labels in zip(ATTRIBUTES, GROUP_LABELS):
            if attribute not in groups:
                raise ValueError(f'Не переданы группы {attribute}')
            raw = np.asarray(groups[attribute], dtype=str)
            if raw.shape != (self.n,):
                raise ValueError(f'Неверный размер групп {attribute}')
            normalized = np.where(np.isin(raw, labels[:2]), raw, 'unknown')
            self.group_values[attribute] = normalized
            cached = []
            for label in labels:
                indices = np.flatnonzero(normalized == label)
                cached.append((indices, np.ascontiguousarray(flat_approve[:, indices], dtype=np.float64),
                               self.y[indices]))
            self._group_cache.append(cached)
        self.coordinates = {
            'models': list(self.models), 'thresholds': list(THRESHOLDS),
            'attributes': list(ATTRIBUTES), 'groups': [list(values) for values in GROUP_LABELS],
            'scenarios': [{'scenario': name, 'loss_to_margin': loss, 'threshold_index': index,
                           'threshold': THRESHOLDS[index]} for name, (loss, index) in zip(SCENARIO_NAMES, SCENARIOS)],
            'loss_ratios': list(LOSS_RATIOS), 'baselines': list(BASELINE_ORDER),
            'confusion_order': list(CONFUSION_ORDER),
            'array_axes': {
                'profit': ['model', 'scenario'], 'baselines': ['loss_ratio', 'baseline'],
                'approval_rate': ['model', 'threshold'], 'approved_n': ['model', 'threshold'],
                'default_rate_among_approved': ['model', 'threshold'],
                'precision': ['model', 'threshold'], 'recall': ['model', 'threshold'],
                'f1': ['model', 'threshold'], 'confusion': ['model', 'threshold', 'confusion_order'],
                'approval_gap': ['model', 'threshold', 'attribute'],
                'tpr_gap': ['model', 'threshold', 'attribute'],
                'group_approval_rate': ['model', 'threshold', 'attribute', 'group'],
                'group_tpr': ['model', 'threshold', 'attribute', 'group'],
                'group_tp': ['model', 'threshold', 'attribute', 'group'],
                'group_approved_n': ['model', 'threshold', 'attribute', 'group'],
                'group_n': ['attribute', 'group'], 'group_default_n': ['attribute', 'group'],
                'total_weight': []},
        }

    def weighted(self, weights):
        """Одни клиентские веса для всех моделей; нули разрешены, сумма положительна."""
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (self.n,) or not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError('Веса должны быть конечными, неотрицательными и соответствовать клиентам')
        total = float(weights.sum())
        if total <= 0:
            raise ValueError('Сумма весов должна быть положительна')
        group_n = np.zeros((2, 3), dtype=np.float64)
        group_default_n = np.zeros((2, 3), dtype=np.float64)
        group_approved = np.zeros((*self.shape, 2, 3), dtype=np.float64)
        group_approved_default = np.zeros_like(group_approved)
        for attribute_index, cached in enumerate(self._group_cache):
            for group_index, (indices, approval_matrix, group_y) in enumerate(cached):
                group_weights = weights[indices]
                weighted_default = group_weights * group_y
                group_n[attribute_index, group_index] = group_weights.sum()
                group_default_n[attribute_index, group_index] = weighted_default.sum()
                counts = approval_matrix @ np.column_stack((group_weights, weighted_default))
                group_approved[:, :, attribute_index, group_index] = counts[:, 0].reshape(self.shape)
                group_approved_default[:, :, attribute_index, group_index] = counts[:, 1].reshape(self.shape)
        # Все клиенты принадлежат одной gender-группе, включая unknown.
        approved = group_approved[:, :, 0, :].sum(axis=-1)
        approved_default = group_approved_default[:, :, 0, :].sum(axis=-1)
        total_default = group_default_n[0].sum()
        tn = approved - approved_default
        fn = approved_default
        tp = total_default - fn
        fp = total - total_default - tn
        group_tp = group_default_n[None, None, :, :] - group_approved_default
        group_approval_rate = _divide(group_approved, group_n[None, None, :, :])
        group_tpr = _divide(group_tp, group_default_n[None, None, :, :])
        profit = np.column_stack([(tn[:, index] - loss * fn[:, index]) / total
                                  for loss, index in SCENARIOS])
        baselines = np.column_stack(([(total - total_default - loss * total_default) / total
                                      for loss in LOSS_RATIOS], np.zeros(3)))
        return {
            'profit': profit, 'baselines': baselines, 'total_weight': np.asarray(total),
            'approved_n': approved, 'approval_rate': approved / total,
            'default_rate_among_approved': _divide(approved_default, approved),
            'precision': _divide(tp, tp + fp), 'recall': _divide(tp, tp + fn),
            'f1': _divide(2 * tp, 2 * tp + fp + fn),
            'confusion': np.stack((tn, fp, fn, tp), axis=-1),
            'group_n': group_n, 'group_default_n': group_default_n,
            'group_approved_n': group_approved, 'group_tp': group_tp,
            'group_approval_rate': group_approval_rate, 'group_tpr': group_tpr,
            'approval_gap': np.abs(group_approval_rate[:, :, :, 0] - group_approval_rate[:, :, :, 1]),
            'tpr_gap': np.abs(group_tpr[:, :, :, 0] - group_tpr[:, :, :, 1]),
        }

    def point_tables(self):
        values = self.weighted(np.ones(self.n, dtype=np.int64))
        profit_rows, group_rows, gap_rows, threshold_rows, baseline_rows = [], [], [], [], []
        for model_index, model in enumerate(self.models):
            for scenario_index, (loss, threshold_index) in enumerate(SCENARIOS):
                profit_rows.append({'model': model, 'level': 4, 'scenario': SCENARIO_NAMES[scenario_index],
                    'loss_to_margin': loss, 'margin': 1, 'threshold': THRESHOLDS[threshold_index],
                    'rows': self.n, 'approved_n': int(values['approved_n'][model_index, threshold_index]),
                    'approval_rate': values['approval_rate'][model_index, threshold_index],
                    'default_rate_among_approved': values['default_rate_among_approved'][model_index, threshold_index],
                    'realized_profit_per_applicant': values['profit'][model_index, scenario_index]})
            for threshold_index, threshold in enumerate(THRESHOLDS):
                common = {'model': model, 'level': 4, 'threshold': threshold}
                confusion = values['confusion'][model_index, threshold_index]
                threshold_rows.append({**common, 'rows': self.n,
                    **{name: int(value) for name, value in zip(CONFUSION_ORDER, confusion)},
                    **{name: values[name][model_index, threshold_index] for name in ('precision', 'recall', 'f1')},
                    'approved_n': int(values['approved_n'][model_index, threshold_index]),
                    'approval_rate': values['approval_rate'][model_index, threshold_index],
                    'default_rate_among_approved': values['default_rate_among_approved'][model_index, threshold_index]})
                gap_row = dict(common)
                for attribute_index, (attribute, labels) in enumerate(zip(ATTRIBUTES, GROUP_LABELS)):
                    for group_index, label in enumerate(labels):
                        index = (model_index, threshold_index, attribute_index, group_index)
                        group_rows.append({**common, 'attribute': attribute, 'group': label,
                            'included_in_binary_gap': group_index < 2,
                            'n': int(values['group_n'][attribute_index, group_index]),
                            'default_n': int(values['group_default_n'][attribute_index, group_index]),
                            'approved_n': int(values['group_approved_n'][index]), 'TP': int(values['group_tp'][index]),
                            'approval_rate': values['group_approval_rate'][index], 'TPR': values['group_tpr'][index]})
                    approval_gap = values['approval_gap'][model_index, threshold_index, attribute_index]
                    gap_row[f'{attribute}_approval_rate_gap'] = approval_gap
                    gap_row[f'{attribute}_tpr_gap'] = values['tpr_gap'][model_index, threshold_index, attribute_index]
                    gap_row[f'{attribute}_approval_gap_le_0_05'] = _observed_approval_flag(approval_gap)
                gaps = [gap_row[f'{attribute}_approval_rate_gap'] for attribute in ATTRIBUTES]
                gap_row['both_approval_gaps_le_0_05'] = (bool(all(value <= .05 for value in gaps))
                    if all(np.isfinite(value) for value in gaps) else pd.NA)
                gap_rows.append(gap_row)
        for loss_index, loss in enumerate(LOSS_RATIOS):
            for baseline_index, policy in enumerate(BASELINE_ORDER):
                approve_all = policy == 'approve_all'
                baseline_rows.append({'policy': policy, 'loss_to_margin': loss, 'margin': 1,
                    'rows': self.n, 'approved_n': self.n if approve_all else 0,
                    'approval_rate': float(approve_all),
                    'default_rate_among_approved': float(self.y.mean()) if approve_all else np.nan,
                    'realized_profit_per_applicant': values['baselines'][loss_index, baseline_index]})
        gap_frame = pd.DataFrame(gap_rows)
        for column in ('gender_approval_gap_le_0_05', 'age_approval_gap_le_0_05', 'both_approval_gaps_le_0_05'):
            gap_frame[column] = gap_frame[column].astype('boolean')
        return {'profit': pd.DataFrame(profit_rows), 'baselines': pd.DataFrame(baseline_rows),
                'fairness_groups': pd.DataFrame(group_rows), 'fairness_gaps': gap_frame,
                'threshold_metrics': pd.DataFrame(threshold_rows)}


def decision_point_tables(y, probabilities_by_model, groups):
    return DecisionEvaluator(y, probabilities_by_model, groups).point_tables()


def decision_bootstrap(y, probabilities_by_model, groups, weights):
    """Удобная одиночная оценка; для 1000 повторов переиспользуйте DecisionEvaluator."""
    return DecisionEvaluator(y, probabilities_by_model, groups).weighted(weights)


def calibration_point_table(y, probabilities_by_model):
    """Те же фиксированные 10 bins, что в старом отчёте; никаких калибровочных fit."""
    from .pilot_decision_metrics import _calibration_rows
    target, models, probabilities = _validate_data(y, probabilities_by_model)
    rows = []
    for model_index, model in enumerate(models):
        rows.extend(_calibration_rows(model, target, probabilities[model_index]))
    return pd.DataFrame(rows)
