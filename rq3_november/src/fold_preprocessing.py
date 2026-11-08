"""Заготовка числового Pipeline: обучение только внутри train конкретного фолда.

Никаких преобразований сохранённых Parquet этот модуль не выполняет. Фабрика
возвращает НЕ обученный объект для последующего включения в ColumnTransformer
и полный Pipeline с моделью. Состав исходных признаков четырёх уровней не меняется.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler


def make_numeric_pipeline(*, scale: bool = False) -> Pipeline:
    """Вернуть не обученный числовой Pipeline с постоянной шириной 2*p.

    Ветка values: медианы оцениваются только при fit на train текущего фолда.
    keep_empty_features=True сохраняет столбец, полностью пустой в этом train,
    и задаёт фиксированный fallback 0. Это техническое значение, а не оценка
    медианы по development/validation/test. Реальные значения этого столбца
    в validation/test сохраняются; заполняются только пропуски.

    Ветка missing: отдельный MissingIndicator(features='all') создаёт по одному
    индикатору исходного пропуска на каждый входной столбец. Поэтому размер и
    порядок результата не зависят от того, в каких столбцах train были пропуски.
    Новый пропуск в validation также получает 1. Индикаторы не стандартизуются.

    Если scale=True, StandardScaler стоит после imputer только в ветке values
    и тоже fit внутри того же train. Для полностью пустого train-столбца после
    fallback=0 scaler имеет mean=0 и scale=1. Категориальные признаки должны
    обрабатываться отдельной веткой ColumnTransformer внутри полного Pipeline.
    """
    value_steps = [
        ("median", SimpleImputer(strategy="median", keep_empty_features=True,
                                 add_indicator=False)),
    ]
    if scale:
        value_steps.append(("scale", StandardScaler()))
    union = FeatureUnion([
        ("values", Pipeline(value_steps)),
        ("missing", MissingIndicator(features="all", sparse=False,
                                     error_on_new=False)),
    ], n_jobs=1)
    return Pipeline([("numeric", union)])


def numeric_pipeline_smoke_check(report_path: str | Path | None = None) -> dict:
    """Проверить политику на маленьких искусственных данных, не на Home Credit.

    Эту демонстрацию можно исполнять в notebook качества: никакие параметры
    преобразования исследуемых данных здесь не оцениваются и не сохраняются.
    """
    train = pd.DataFrame({
        "empty_in_train": [np.nan, np.nan, np.nan],
        "partly_missing": [1.0, np.nan, 3.0],
        "complete_in_train": [10.0, 20.0, 30.0],
    })
    validation = pd.DataFrame({
        "empty_in_train": [7.0, np.nan],
        "partly_missing": [np.nan, 999.0],
        "complete_in_train": [np.nan, 40.0],
    })
    pipeline = make_numeric_pipeline()
    train_out = pipeline.fit_transform(train)
    validation_out = pipeline.transform(validation)
    expected = np.array([[7.0, 2.0, 20.0, 0.0, 1.0, 1.0],
                         [0.0, 999.0, 40.0, 1.0, 0.0, 0.0]])
    np.testing.assert_array_equal(validation_out, expected)
    assert train_out.shape == (3, 6)
    assert validation_out.shape == (2, 6)
    report = {
        "scope": "synthetic_only_no_fit_on_home_credit",
        "sklearn_version": sklearn.__version__,
        "all_checks_passed": True,
        "raw_numeric_columns": 3,
        "transformed_columns": 6,
        "empty_train_column_fallback": 0,
        "median_for_partly_missing_train_column": 2,
        "validation_observed_value_in_empty_train_column_preserved": True,
        "new_validation_missing_value_has_indicator": True,
        "width_is_twice_numeric_column_count": True,
        "feature_names": pipeline.get_feature_names_out().tolist(),
        "fit_policy_ru": "fit только на train каждого CV-фолда или 90%-ного повтора; test/validation не участвуют",
        "fallback_policy_ru": "Для полностью пустого train-столбца фиксированный 0 и индикатор пропуска; столбец не удаляется",
        "category_policy_ru": "Обучаемое кодирование категорий — отдельная ветка полного Pipeline, здесь не выполняется",
    }
    if report_path is not None:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
