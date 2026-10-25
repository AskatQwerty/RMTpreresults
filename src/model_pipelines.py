"""Полные Pipeline семи моделей RMT; никакого fit при создании объектов.

Все обучаемые преобразования находятся внутри estimator, передаваемого в CV.
У ансамблей каждый базовый estimator содержит собственную предобработку;
особенно важно, что stacking не получает заранее преобразованный development.
"""
from __future__ import annotations

from math import prod

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from .fold_preprocessing import make_numeric_pipeline


MODEL_NAMES = (
    "logistic_regression", "random_forest", "xgboost", "lightgbm", "mlp",
    "soft_voting", "stacking",
)
ENSEMBLE_BASES = ("lr", "xgb", "lgbm")
INNER_STACKING_FOLDS = 3
MISSING_CATEGORY = "__RMT_MISSING__"


def normalize_category_missing(values):
    """Строковые категории (включая Boolean из Parquet), None/pd.NA -> np.nan.

    Детерминированное приведение типов не обучает словарь. Оно предотвращает
    смешение bool и строкового маркера пропуска внутри одного OHE-столбца.
    """
    if isinstance(values, pd.DataFrame):
        result = values.astype('string').astype(object)
        return result.where(pd.notna(result), np.nan)
    result = np.asarray(values, dtype=object).copy()
    missing = pd.isna(result)
    result[~missing] = [str(value) for value in result[~missing]]
    result[missing] = np.nan
    return result


def to_float32_csr(values):
    """Единый компактный формат результата; это детерминированное преобразование."""
    result = sparse.csr_matrix(values, dtype=np.float32)
    result.sort_indices()
    return result


def to_float32_dense(values):
    """Явно хранить нули для XGBoost: CSR-отсутствие иначе означает missing.

    Минимальный размер dense-массива n_rows*n_encoded_features*4 байт,
    без дополнительных рабочих копий бустинга. Не переключаемся незаметно на
    CSR при нехватке RAM: такая замена изменила бы семантику нулевых значений.
    """
    if sparse.issparse(values):
        return values.astype(np.float32, copy=False).toarray(order="C")
    return np.asarray(values, dtype=np.float32, order="C")


def make_preprocessor(numeric_columns, categorical_columns, *, scale=False,
                      dense_output=False):
    """Независимые train-only ветки, float32 CSR или явный dense для XGBoost.

    Медианы, параметры scaler и категории OneHotEncoder появляются только
    при fit текущего train. Неизвестная категория получает нулевой OHE-вектор.
    Числовые индикаторы идут для каждого исходного числового поля, включая
    столбцы, полностью пустые в train, как в fold_preprocessing.py. Для XGBoost
    dense_output=True сохраняет различие между числовым 0 и missing: tree booster
    трактует отсутствующие CSR-элементы как missing (официальный XGBoost FAQ).
    """
    numeric_columns, categorical_columns = list(numeric_columns), list(categorical_columns)
    if not numeric_columns and not categorical_columns:
        raise ValueError("Не заданы признаки")
    if set(numeric_columns) & set(categorical_columns):
        raise ValueError("Числовые и категориальные признаки пересекаются")
    all_columns = numeric_columns + categorical_columns
    if len(set(all_columns)) != len(all_columns):
        raise ValueError("Повторяющиеся имена входных признаков")
    forbidden = {"TARGET", "CODE_GENDER", "DAYS_BIRTH", "AGE_YEARS", "AGE_GROUP", "GENDER_GROUP"}
    if forbidden & set(all_columns) or any(name.startswith("SK_ID_") for name in all_columns):
        raise ValueError("TARGET, ID и прямые fairness-поля не могут быть признаками модели")
    categories = Pipeline([
        ("normalize_missing", FunctionTransformer(normalize_category_missing,
            validate=False, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY,
            keep_empty_features=True)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True,
            dtype=np.float32)),
    ])
    columns = ColumnTransformer([
        ("numeric", make_numeric_pipeline(scale=scale), numeric_columns),
        ("categorical", categories, categorical_columns),
    ], remainder="drop", sparse_threshold=1.0, n_jobs=1,
       verbose_feature_names_out=True)
    return Pipeline([
        ("columns", columns),
        ("float32", FunctionTransformer(to_float32_dense if dense_output else to_float32_csr, validate=False,
            accept_sparse=True, feature_names_out="one-to-one")),
    ])


def _base_classifier(name, seed, technical, threads):
    """Classifier без предобработки; используется только внутри полного Pipeline."""
    if name == "logistic_regression":
        return LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
            max_iter=500 if technical else 2000, tol=1e-4,
            class_weight=None, random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=40 if technical else 300,
            max_depth=8 if technical else 16, min_samples_leaf=2 if technical else 10,
            max_features="sqrt", class_weight=None, random_state=seed, n_jobs=threads)
    if name == "xgboost":
        # Ленивый импорт: сама проверка схем не требует установленного бустинга.
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=30 if technical else 300,
            max_depth=3 if technical else 5, learning_rate=0.08 if technical else 0.05,
            min_child_weight=1 if technical else 5, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0,
            objective="binary:logistic", eval_metric="logloss", tree_method="hist",
            random_state=seed, n_jobs=threads, verbosity=0)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=30 if technical else 300,
            num_leaves=15 if technical else 31, learning_rate=0.08 if technical else 0.05,
            min_child_samples=10 if technical else 50, max_depth=-1,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            reg_lambda=1.0, objective="binary", class_weight=None,
            random_state=seed, n_jobs=threads, verbosity=-1,
            deterministic=True, force_col_wise=True)
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(16,) if technical else (64,32),
            activation="relu", solver="adam", alpha=1e-3,
            batch_size=64 if technical else 256, learning_rate_init=1e-3,
            max_iter=60 if technical else 300, tol=1e-4,
            n_iter_no_change=10 if technical else 20, early_stopping=False,
            shuffle=True, random_state=seed)
    raise ValueError(f"Неизвестная базовая модель: {name}")


def make_estimator(name, numeric_columns, categorical_columns, *, seed=42,
                   technical=False, threads=2):
    """Создать полный необученный Pipeline для одной пары модель/уровень.

    Внешняя параллельность поиска и ансамблей равна 1; отдельное дерево/бустинг
    может использовать не более двух потоков. У stacking каждый внутренний
    OOF-fit клонирует базовый Pipeline вместе с imputer/scaler/encoder.
    Метамодель обучается на трёх OOF-вероятностях; исходные признаки ей
    не передаются. Внешний test никогда не нужен для fit этого объекта.
    """
    if name not in MODEL_NAMES:
        raise ValueError(f"Неизвестная модель {name!r}; допустимы {MODEL_NAMES}")
    threads = max(1, min(int(threads),2))
    numeric_columns, categorical_columns = list(numeric_columns), list(categorical_columns)
    if name in {"soft_voting", "stacking"}:
        bases = [
            ("lr", make_estimator("logistic_regression", numeric_columns, categorical_columns,
                                 seed=seed, technical=technical, threads=threads)),
            ("xgb", make_estimator("xgboost", numeric_columns, categorical_columns,
                                  seed=seed, technical=technical, threads=threads)),
            ("lgbm", make_estimator("lightgbm", numeric_columns, categorical_columns,
                                   seed=seed, technical=technical, threads=threads)),
        ]
        if name == "soft_voting":
            model = VotingClassifier(estimators=bases, voting="soft", weights=(1,1,1),
                                     n_jobs=1, flatten_transform=True)
        else:
            model = StackingClassifier(estimators=bases,
                final_estimator=LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                    max_iter=2000, class_weight=None, random_state=seed),
                cv=StratifiedKFold(n_splits=INNER_STACKING_FOLDS, shuffle=True, random_state=seed),
                stack_method="predict_proba", passthrough=False, n_jobs=1)
        # Принципиально нет общего preprocess перед ансамблем.
        return Pipeline([("model",model)])
    return Pipeline([
        ("preprocess", make_preprocessor(numeric_columns, categorical_columns,
                                        scale=name in {"logistic_regression", "mlp"},
                                        dense_output=name=="xgboost")),
        ("model",_base_classifier(name,seed,technical,threads)),
    ])


def parameter_space(name, *, technical=False):
    """Конечные заранее заданные пространства; RandomizedSearchCV берёт 20 из них.

    Все значения — обычные JSON-совместимые скаляры/списки/кортежи, без объектов
    scipy distribution с нестабильным repr. В техническом режиме ровно две
    конфигурации: это проверка механизма, не результат подбора для эксперимента.
    """
    if name not in MODEL_NAMES:
        raise ValueError(f"Неизвестная модель: {name}")
    if technical:
        return {
            "logistic_regression":{"model__C":[0.1,1.0]},
            "random_forest":{"model__max_depth":[4,8]},
            "xgboost":{"model__max_depth":[2,3]},
            "lightgbm":{"model__num_leaves":[7,15]},
            "mlp":{"model__alpha":[1e-3,1e-2]},
            "soft_voting":{"model__weights":[(1,1,1),(1,2,2)]},
            "stacking":{"model__final_estimator__C":[0.1,1.0]},
        }[name]
    return {
        "logistic_regression":{
            "model__C":np.logspace(-4,3,24).tolist(),
        },
        "random_forest":{
            "model__n_estimators":[150,300], "model__max_depth":[8,16,24],
            "model__min_samples_leaf":[5,10,30], "model__max_features":["sqrt",0.5],
        },
        "xgboost":{
            "model__n_estimators":[200,400], "model__max_depth":[3,5,7],
            "model__learning_rate":[0.03,0.08], "model__min_child_weight":[1,5],
            "model__subsample":[0.8,1.0], "model__colsample_bytree":[0.8,1.0],
            "model__reg_lambda":[1.0,5.0],
        },
        "lightgbm":{
            "model__n_estimators":[200,400], "model__num_leaves":[15,31,63],
            "model__learning_rate":[0.03,0.08], "model__min_child_samples":[20,50,100],
            "model__colsample_bytree":[0.8,1.0], "model__reg_lambda":[1.0,5.0],
        },
        "mlp":{
            "model__hidden_layer_sizes":[(32,),(64,),(64,32)],
            "model__alpha":[1e-5,1e-4,1e-3,1e-2],
            "model__learning_rate_init":[3e-4,1e-3], "model__batch_size":[128,256],
        },
        "soft_voting":{
            "model__lr__model__C":[0.01,0.1,1.0,10.0],
            "model__xgb__model__max_depth":[3,5],
            "model__lgbm__model__num_leaves":[15,31],
            "model__weights":[(1,1,1),(1,2,2),(2,1,1)],
        },
        "stacking":{
            "model__lr__model__C":[0.1,1.0,10.0],
            "model__xgb__model__max_depth":[3,5],
            "model__lgbm__model__num_leaves":[15,31],
            "model__final_estimator__C":[0.01,0.1,1.0,10.0],
        },
    }[name]


def experiment_spec(name):
    """Структура модели и честный подсчёт внутренних обучений на один outer fit."""
    if name not in MODEL_NAMES:
        raise ValueError(f"Неизвестная модель: {name}")
    is_ensemble = name in {"soft_voting","stacking"}
    base_full = 3 if is_ensemble else 1
    base_oof = 9 if name=="stacking" else 0
    meta = 1 if name=="stacking" else 0
    return {
        "model":name,
        "base_models":list(ENSEMBLE_BASES) if is_ensemble else [name],
        "scale_numeric":name in {"logistic_regression","mlp"},
        "ensemble_scaling_policy":"only LR base scales its own numeric branch" if is_ensemble else None,
        "base_full_fits_per_outer_fit":base_full,
        "base_oof_fits_per_outer_fit":base_oof,
        "meta_fits_per_outer_fit":meta,
        "primitive_fits_per_outer_fit":base_full+base_oof+meta,
        "internal_folds":INNER_STACKING_FOLDS if name=="stacking" else 0,
        "full_parameter_configurations":prod(len(values) for values in parameter_space(name).values()),
        "technical_parameter_configurations":2,
        "preprocessing_inside_each_base_pipeline":True,
        "class_resampling":False,
        "primary_class_weights":None,
        "encoded_format":"dense_float32" if name=="xgboost" else "per_base_XGB_dense_others_CSR" if is_ensemble else "CSR_float32",
        "xgboost_zero_semantics_reference":"https://xgboost.readthedocs.io/en/stable/faq.html#why-do-i-see-different-results-with-sparse-and-dense-data" if name=="xgboost" or is_ensemble else None,
    }
