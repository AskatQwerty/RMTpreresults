"""Supplemental MLP experiment: train-only log-loss epoch selection and refit.

The original frozen Pipeline is not changed. Each call receives ONLY the current
outer training partition. It holds out an internal stratified validation split,
fits preprocessing on the internal training rows, chooses an epoch count by
internal validation log-loss, discards that model, then fits a fresh Pipeline on
all supplied rows for exactly that many epochs. Outer validation/test never enter
this function. Clipping, when enabled, bounds standardized numeric VALUES only.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import time
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from threadpoolctl import threadpool_limits

from .model_pipelines import make_preprocessor


def fixed_z_clip(values, *, limit):
    """Stateless fixed bound after train-only scaling; no rows/columns are removed."""
    limit = float(limit)
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError('A positive finite z bound is required')
    return np.clip(np.asarray(values), -limit, limit)


def make_repair_preprocessor(numeric, categorical, *, clip_z=None):
    """Reuse existing imputation/indicators/OHE; optionally clip only numeric values."""
    preprocessor = make_preprocessor(numeric, categorical, scale=True)
    if clip_z is not None:
        if not math.isfinite(float(clip_z)) or float(clip_z) <= 0:
            raise ValueError('clip_z must be None or a positive finite number')
        columns = preprocessor.named_steps['columns']
        numeric_pipeline = next(t for name, t, _ in columns.transformers if name == 'numeric')
        values = dict(numeric_pipeline.named_steps['numeric'].transformer_list)['values']
        values.steps.append(('fixed_z_clip', FunctionTransformer(fixed_z_clip,
            kw_args={'limit': float(clip_z)}, validate=False, feature_names_out='one-to-one')))
    return preprocessor


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(_plain(value), ensure_ascii=False,
        sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def _preprocessor_state(preprocessor):
    columns = preprocessor.named_steps['columns']
    numeric = columns.named_transformers_['numeric']
    values = dict(numeric.named_steps['numeric'].transformer_list)['values']
    imputer, scaler = values.named_steps['median'], values.named_steps['scale']
    encoder = columns.named_transformers_['categorical'].named_steps['onehot']
    state = dict(medians=_plain(imputer.statistics_), scaler_mean=_plain(scaler.mean_),
        scaler_variance=_plain(scaler.var_), scaler_scale=_plain(scaler.scale_),
        scaler_n_samples_seen=_plain(scaler.n_samples_seen_),
        categories=[_plain(x) for x in encoder.categories_],
        encoded_features=len(preprocessor.get_feature_names_out()))
    state['sha256'] = _digest(state)
    return state


def _check_matrix(matrix):
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    if not np.isfinite(values).all():
        raise ValueError('The preprocessed matrix contains nonfinite values')


def _probability(model, matrix):
    probabilities = model.predict_proba(matrix)
    if probabilities.shape != (matrix.shape[0], 2) or not np.isfinite(probabilities).all():
        raise ValueError('Invalid MLP probability shape/values')
    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6):
        raise ValueError('MLP probabilities do not sum to one')
    index = np.flatnonzero(model.classes_ == 1)
    if len(index) != 1:
        raise ValueError('Positive class1 is missing')
    result = probabilities[:, int(index[0])].astype(float)
    if not ((result >= 0) & (result <= 1)).all():
        raise ValueError('MLP probability outside [0,1]')
    return result


def _mlp(alpha, seed, max_epochs):
    # With an integer random_state sklearn1.7.2 reinitializes the shuffle RNG
    # inside each partial_fit. A fresh persistent RNG per optimizer stage makes
    # epoch shuffles reproducible without restarting their sequence each call.
    return MLPClassifier(hidden_layer_sizes=(32,), activation='relu', solver='adam',
        alpha=float(alpha), batch_size=256, learning_rate_init=3e-4,
        max_iter=int(max_epochs), tol=0, n_iter_no_change=int(max_epochs)+1,
        early_stopping=False, shuffle=True, random_state=np.random.RandomState(seed))


def _epoch(model, matrix, labels, *, first):
    if first:
        model.partial_fit(matrix, labels, classes=np.array([0, 1], dtype=labels.dtype))
    else:
        model.partial_fit(matrix, labels)


def _warning_summary(caught, stage):
    counts = Counter((w.category.__name__, str(w.message)) for w in caught)
    return [dict(stage=stage, category=k[0], message=k[1], count=v)
            for k, v in sorted(counts.items())]


def _select_epochs(model, train, y_train, validation, y_validation, *,
                   max_epochs, patience, min_delta):
    """Keep the strict minimum; use a separate meaningful-improvement patience anchor."""
    curve = []
    best_loss, best_epoch = float('inf'), None
    patience_anchor, stagnant = float('inf'), 0
    reason = 'max_epochs'
    for epoch in range(1, max_epochs + 1):
        begin = time.perf_counter()
        _epoch(model, train, y_train, first=epoch == 1)
        train_ll = float(log_loss(y_train, _probability(model, train), labels=[0, 1]))
        valid_ll = float(log_loss(y_validation, _probability(model, validation), labels=[0, 1]))
        if not math.isfinite(train_ll) or not math.isfinite(valid_ll):
            raise ValueError('Nonfinite epoch log-loss')
        curve.append(dict(epoch=epoch, train_log_loss=train_ll, validation_log_loss=valid_ll,
                          optimizer_training_objective=float(model.loss_),
                          epoch_seconds=time.perf_counter()-begin))
        # Tie policy: retain the earliest strict minimum. min_delta does not
        # cause selection of a higher-loss epoch; it controls only patience.
        if valid_ll < best_loss:
            best_loss, best_epoch = valid_ll, epoch
        if valid_ll < patience_anchor - min_delta:
            patience_anchor, stagnant = valid_ll, 0
        else:
            stagnant += 1
        if stagnant >= patience:
            reason = 'internal_validation_log_loss_patience'
            break
    return int(best_epoch), curve, reason


def fit_mlp_variant(X, y, numeric, categorical, variant, *, seed=42,
                    max_epochs=120, patience=12, min_delta=1e-4,
                    validation_fraction=.15, threads=2):
    """Return (fresh full-supplied-train fitted Pipeline, JSON-safe diagnostics).

    ``variant`` has keys name, alpha, clip_z. No outer validation/test argument
    exists. Two optimizer instances are trained: internal epoch selection and
    fresh all-training-row refit. The returned ordinary Pipeline is suitable for
    predict_proba and joblib persistence without this training function.
    """
    started = time.perf_counter()
    if not isinstance(X, pd.DataFrame) or not X.index.is_unique or not X.columns.is_unique:
        raise ValueError('X must be a DataFrame with unique row IDs and columns')
    if isinstance(y, pd.Series) and not X.index.equals(y.index):
        raise ValueError('TARGET must have exactly the same row ID order as X')
    labels = np.asarray(y)
    if labels.shape != (len(X),) or set(np.unique(labels)) != {0, 1}:
        raise ValueError('Both binary TARGET classes are required')
    labels = labels.astype(np.int8)
    if set(numeric) | set(categorical) != set(X.columns):
        raise ValueError('Column roles must cover exactly the provided X columns')
    if int(max_epochs) != max_epochs or max_epochs < 1 or int(patience) != patience or patience < 1:
        raise ValueError('Positive integer max_epochs and patience required')
    if not (0 < validation_fraction < 1) or not math.isfinite(min_delta) or min_delta < 0:
        raise ValueError('Invalid internal validation or stopping rule')
    alpha = float(variant['alpha'])
    clip_z = variant.get('clip_z')
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError('alpha must be finite and nonnegative')
    numeric, categorical = list(numeric), list(categorical)
    threads = max(1, min(int(threads), 2))
    inner_train, inner_validation = train_test_split(np.arange(len(X)),
        test_size=validation_fraction, stratify=labels, random_state=seed)
    if set(labels[inner_train]) != {0, 1} or set(labels[inner_validation]) != {0, 1}:
        raise ValueError('Both classes must occur in both internal partitions')

    with threadpool_limits(limits=threads):
        with warnings.catch_warnings(record=True) as selection_warnings:
            warnings.simplefilter('always')
            inner_preprocessor = make_repair_preprocessor(numeric, categorical, clip_z=clip_z)
            train = inner_preprocessor.fit_transform(X.iloc[inner_train], labels[inner_train])
            inner_state = _preprocessor_state(inner_preprocessor)
            validation = inner_preprocessor.transform(X.iloc[inner_validation])
            if _preprocessor_state(inner_preprocessor) != inner_state:
                raise ValueError('Validation transform changed fitted preprocessing statistics')
            _check_matrix(train)
            _check_matrix(validation)
            inner_model = _mlp(alpha, seed, max_epochs)
            best_epoch, curve, stop_reason = _select_epochs(inner_model, train, labels[inner_train],
                validation, labels[inner_validation], max_epochs=max_epochs,
                patience=patience, min_delta=min_delta)
        inner_optimizer = dict(raw_n_iter=int(inner_model.n_iter_),
            epochs_completed=len(curve), loss_curve_length=len(inner_model.loss_curve_),
            samples_seen=int(inner_model.t_), final_training_objective=float(inner_model.loss_))
        if len(inner_model.loss_curve_) != len(curve) or inner_model.t_ != len(inner_train)*len(curve):
            raise ValueError('Unexpected partial_fit epoch/sample accounting')
        del train, validation, inner_preprocessor, inner_model

        # Selected epoch count is now fixed. Internal validation is included in
        # fresh preprocessing and optimizer fit, as part of the supplied train.
        refit_started = time.perf_counter()
        with warnings.catch_warnings(record=True) as refit_warnings:
            warnings.simplefilter('always')
            final_preprocessor = make_repair_preprocessor(numeric, categorical, clip_z=clip_z)
            all_train = final_preprocessor.fit_transform(X, labels)
            _check_matrix(all_train)
            final_model = _mlp(alpha, seed, max_epochs)
            for epoch in range(1, best_epoch + 1):
                _epoch(final_model, all_train, labels, first=epoch == 1)
            train_ll = float(log_loss(labels, _probability(final_model, all_train), labels=[0, 1]))
        refit_seconds = time.perf_counter()-refit_started
        final_state = _preprocessor_state(final_preprocessor)
        if len(final_model.loss_curve_) != best_epoch or final_model.t_ != len(X)*best_epoch:
            raise ValueError('Fresh refit did not execute the selected epoch count')
        # Keep sklearn's raw n_iter_ honestly (1 in v1.7.2 partial_fit); explicit
        # cumulative epoch counts below are authoritative. Preserve reproducible
        # public constructor parameter for a future fresh fit, after training.
        final_model.set_params(random_state=seed)
        pipeline = Pipeline([('preprocess', final_preprocessor), ('model', final_model)])
        final_optimizer = dict(raw_n_iter=int(final_model.n_iter_), epochs_completed=best_epoch,
            loss_curve_length=len(final_model.loss_curve_), samples_seen=int(final_model.t_),
            loss_curve=_plain(final_model.loss_curve_), final_training_objective=float(final_model.loss_),
            train_log_loss=train_ll)

    class_counts = lambda inds: {str(k): int((labels[inds] == k).sum()) for k in (0, 1)}
    diagnostics = dict(schema_version=1, variant=_plain(variant), seed=int(seed),
        provided_train_rows=len(X), provided_train_ids_sha256=_digest(X.index.tolist()),
        raw_features=X.shape[1], numeric_features=len(numeric), categorical_features=len(categorical),
        best_epoch=best_epoch, selected_epoch=best_epoch, selected_epochs=best_epoch,
        best_internal_validation_log_loss=float(min(r['validation_log_loss'] for r in curve)),
        epoch_curve=curve, stop_reason=stop_reason,
        stopping_rule=dict(metric='unregularized internal validation log_loss',
            max_epochs=int(max_epochs), patience=int(patience), min_delta=float(min_delta),
            selected_epoch_policy='earliest strict minimum observed; patience anchor separate',
            validation_fraction=float(validation_fraction)),
        internal_split=dict(train_rows=len(inner_train), validation_rows=len(inner_validation),
            train_ids=_plain(X.index[inner_train].tolist()),
            validation_ids=_plain(X.index[inner_validation].tolist()),
            train_ids_sha256=_digest(X.index[inner_train].tolist()),
            validation_ids_sha256=_digest(X.index[inner_validation].tolist()),
            train_class_counts=class_counts(inner_train), validation_class_counts=class_counts(inner_validation),
            disjoint=True, only_provided_training_rows=True),
        inner_preprocessing=inner_state, final_preprocessing=final_state,
        inner_optimizer=inner_optimizer, final_optimizer=final_optimizer,
        final_refit_train_ids_sha256=_digest(X.index.tolist()), final_refit_rows=len(X),
        optimizer_fits=2, partial_fit_calls=len(curve)+best_epoch,
        final_refit_seconds=refit_seconds, total_seconds=time.perf_counter()-started,
        warnings=_warning_summary(selection_warnings, 'inner_epoch_selection')
                 + _warning_summary(refit_warnings, 'fresh_full_train_refit'),
        training_policy=dict(no_outer_validation_or_test_in_fit=True,
            preprocess_fit_inside_internal_train=True, fresh_preprocessor_for_full_train=True,
            fresh_optimizer_for_full_train=True, internal_selection_model_discarded=True,
            numeric_values_clip_z=clip_z, clipping_bounds_fixed_not_fitted=True,
            indicators_and_categories_not_clipped=True, class_resampling=False, threads=threads,
            random_state_policy='Fresh RandomState(seed) per optimizer stage, advanced across partial_fit calls; public parameter restored to seed before export',
            raw_n_iter_note='sklearn1.7.2 resets n_iter_ each partial_fit; cumulative epochs use loop/loss_curve_/t_'))
    # Guarantee exportability now, rather than failing after an expensive CV fit.
    json.dumps(diagnostics, ensure_ascii=False, allow_nan=False)
    return pipeline, diagnostics
