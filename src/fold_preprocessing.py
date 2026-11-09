"""Numeric Pipeline factory: fit only on the training partition of the current fold.

This module does not transform saved Parquet files. The factory
returns an UNFITTED object for inclusion in a ColumnTransformer
and a complete model Pipeline. The original features at all four levels remain unchanged.
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
    """Return an unfitted numeric Pipeline with constant width 2*p.

    The values branch estimates medians only during fit on the current training fold.
    keep_empty_features=True retains a column that is entirely missing in this train
    and assigns a fixed fallback of 0. This is a technical value, not a median
    estimated from development/validation/test. Observed values of this column
    in validation/test are retained; only missing entries are imputed.

    The missing branch uses MissingIndicator(features='all') to create one
    original-missingness indicator for each input column. Thus output size and
    order do not depend on which train columns contained missing values.
    A new validation missing value also receives 1. Indicators are not standardized.

    If scale=True, StandardScaler follows the imputer in the values branch only
    and is fitted on the same train partition. For an entirely missing train column,
    the fallback=0 scaler has mean=0 and scale=1. Categorical features must
    be handled by a separate ColumnTransformer branch within the full Pipeline.
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
    """Check the policy on small synthetic data, not on Home Credit.

    This demonstration can run in the quality notebook: no preprocessing parameters
    are estimated or saved for the research dataset here.
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
        "fit_policy_ru": "fit only on train within each CV fold or 90% repeat; test/validation are excluded",
        "fallback_policy_ru": "An entirely missing train column receives fixed 0 and a missingness indicator; the column is retained",
        "category_policy_ru": "Learned categorical encoding belongs to a separate full-Pipeline branch and is not performed here",
    }
    if report_path is not None:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
