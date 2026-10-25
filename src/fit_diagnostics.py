"""Сериализуемая диагностика уже обученных Pipeline без повторного fit/transform.

Функции только читают атрибуты estimator. Настройки не меняются, исходные данные
не требуются; предупреждения fit сохраняются вызывающим кодом отдельно.
"""
from __future__ import annotations

import math
from numbers import Number

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


CONFIG_FIELDS = (
    'max_iter', 'tol', 'n_iter_no_change', 'early_stopping',
    'hidden_layer_sizes', 'activation', 'solver', 'alpha', 'batch_size',
    'learning_rate', 'learning_rate_init', 'shuffle', 'validation_fraction',
    'random_state', 'n_estimators', 'max_depth', 'num_leaves', 'min_samples_leaf',
)
ACTUAL_FIELDS = (
    'n_iter_', 'loss_curve_', 'loss_', 'best_loss_', 't_', 'n_layers_',
    'n_outputs_', 'best_validation_score_', 'validation_scores_',
)


def _plain(value):
    """JSON-safe значения; нечисловой результат не маскируется нулём."""
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _safe_n_features(estimator):
    # LightGBM и XGBoost предоставляют это как fitted property, а sklearn —
    # обычно как атрибут. Не опираемся на внутренние объекты booster.
    value = getattr(estimator, 'n_features_in_', None)
    return int(value) if value is not None else None


def _preprocess_description(preprocessor):
    if preprocessor is None:
        return {'input_features': None, 'output_features': None,
                'representation': 'not_applicable'}
    names = preprocessor.get_feature_names_out()
    conversion = preprocessor.named_steps.get('float32') if isinstance(preprocessor, Pipeline) else None
    function_name = getattr(getattr(conversion, 'func', None), '__name__', None)
    representation = {
        'to_float32_dense': 'dense_float32',
        'to_float32_csr': 'CSR_float32',
    }.get(function_name, 'unknown')
    return {'input_features': _safe_n_features(preprocessor),
            'output_features': len(names), 'representation': representation}


def extract_fit_diagnostics(fitted_pipeline):
    """Вернуть actual итерации/кривые и размеры каждой обученной базы/метамодели.

    Схема результата:
    - root_class / root_model_class / raw_feature_count;
    - estimators: плоский список leaf-оценивателей с path, role, class_name,
      config, fitted, model_input_features, preprocessing, checks;
    - processed_feature_counts: компактная таблица баз и метамодели;
    - mlp: только записи MLP с actual n_iter_, loss_curve_, loss_, best_loss_.

    Для Voting/Stacking читаются fitted named_estimators_, а не исходные
    необученные estimators. Внутренние OOF-клоны StackingClassifier не хранятся
    sklearn после fit; здесь отражены сохранённые полные базы и метамодель.
    """
    check_is_fitted(fitted_pipeline)
    records = []

    def visit(estimator, path, role='model', inherited_preprocessor=None):
        if isinstance(estimator, Pipeline):
            preprocessor = estimator.named_steps.get('preprocess', inherited_preprocessor)
            final_name, final_estimator = estimator.steps[-1]
            visit(final_estimator, f'{path}.{final_name}', role, preprocessor)
            return
        if hasattr(estimator, 'named_estimators_'):
            for name, base in estimator.named_estimators_.items():
                if isinstance(base, str) and base == 'drop':
                    continue
                visit(base, f'{path}.bases.{name}', 'base')
            meta = getattr(estimator, 'final_estimator_', None)
            if meta is not None:
                visit(meta, f'{path}.meta', 'meta')
            return
        parameters = estimator.get_params(deep=False)
        config = {name: _plain(parameters[name]) for name in CONFIG_FIELDS if name in parameters}
        actual = {name: _plain(getattr(estimator, name)) for name in ACTUAL_FIELDS if hasattr(estimator, name)}
        preprocessing = _preprocess_description(inherited_preprocessor)
        model_input = _safe_n_features(estimator)
        if role == 'meta':
            preprocessing['representation'] = 'dense_base_probabilities'
        checks = {}
        if preprocessing['output_features'] is not None and model_input is not None:
            checks['preprocessor_width_matches_classifier'] = preprocessing['output_features'] == model_input
        n_iter = actual.get('n_iter_')
        max_iter = config.get('max_iter')
        if isinstance(n_iter, Number) and max_iter is not None:
            checks['reached_iteration_limit'] = n_iter >= max_iter
        if 'loss_curve_' in actual and actual['loss_curve_'] is not None:
            checks['loss_curve_length'] = len(actual['loss_curve_'])
            if isinstance(n_iter, Number):
                checks['loss_curve_length_matches_n_iter'] = len(actual['loss_curve_']) == n_iter
        records.append({'path': path, 'role': role, 'class_name': type(estimator).__name__,
                        'model_input_features': model_input,
                        'preprocessing': preprocessing, 'config': config,
                        'fitted': actual, 'checks': checks})

    visit(fitted_pipeline, 'pipeline')
    root_model = fitted_pipeline.steps[-1][1] if isinstance(fitted_pipeline, Pipeline) else fitted_pipeline
    return {
        'root_class': type(fitted_pipeline).__name__,
        'root_model_class': type(root_model).__name__,
        'raw_feature_count': _safe_n_features(fitted_pipeline),
        'estimators': records,
        'processed_feature_counts': [
            {'path': row['path'], 'role': row['role'], 'class_name': row['class_name'],
             'features': row['model_input_features'],
             'representation': row['preprocessing']['representation']}
            for row in records
        ],
        'mlp': [row for row in records if row['class_name'] == 'MLPClassifier'],
        'internal_oof_estimators_retained': False,
        'method': 'fitted_attributes_and_feature_names_only_no_transform_no_fit',
    }
