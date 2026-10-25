"""Синтетические проверки Pipeline и изоляции предобработки в CV/stacking.

Файлы Home Credit не читаются. Полный эксперимент не запускается.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterSampler, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.model_pipelines import MODEL_NAMES, MISSING_CATEGORY, make_estimator, parameter_space


NUMERIC = ['signal', 'empty_in_train']
CATEGORICAL = ['category', 'empty_category']


def synthetic_data(n=90):
    index = pd.Index(10000 + np.arange(n), name='SK_ID_CURR')
    signal = np.arange(n, dtype=float)
    signal[::11] = np.nan
    frame = pd.DataFrame({
        'signal': signal,
        'empty_in_train': np.full(n, np.nan),
        'category': np.where(np.arange(n) % 2 == 0, 'A', 'B').astype(object),
        'empty_category': pd.Series([None] * n, index=index, dtype=object),
    }, index=index)
    frame.loc[index[::13], 'category'] = None
    frame.loc[index[::17], 'category'] = np.nan
    target = pd.Series(np.arange(n) % 2, index=index, name='TARGET')
    return frame, target


def walk_transformers(estimator):
    yield estimator
    if isinstance(estimator, Pipeline):
        children = estimator.named_steps.values()
    elif isinstance(estimator, ColumnTransformer):
        children = [item[1] for item in getattr(estimator, 'transformers_', estimator.transformers)]
    elif isinstance(estimator, FeatureUnion):
        children = [item[1] for item in estimator.transformer_list]
    else:
        children = []
    for child in children:
        if not isinstance(child, str):
            yield from walk_transformers(child)


def record_numeric_imputer_fits(records):
    """Сохранять реально обученные медианы каждого клона, включая внутренние OOF."""
    original = SimpleImputer.fit

    def logged_fit(self, X, y=None, **kwargs):
        fitted = original(self, X, y=y, **kwargs)
        if self.strategy == 'median':
            records.append({
                'n_rows': len(X),
                'row_ids': tuple(X.index.tolist()) if isinstance(X, pd.DataFrame) else None,
                'statistics': self.statistics_.copy(),
            })
        return fitted

    return patch.object(SimpleImputer, 'fit', logged_fit)


class ModelPipelineTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y = synthetic_data()

    def factory(self, name, technical=True):
        return make_estimator(name, NUMERIC, CATEGORICAL, seed=42,
                              technical=technical, threads=1)

    def test_all_seven_factories_clone_and_unfitted(self):
        self.assertEqual(len(MODEL_NAMES), 7)
        self.assertEqual(len(set(MODEL_NAMES)), 7)
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                estimator = self.factory(name)
                self.assertIsInstance(estimator, Pipeline)
                cloned = clone(estimator)
                with self.assertRaises((NotFittedError, AttributeError)):
                    cloned.predict_proba(self.X.iloc[:3])

    def test_full_search_has_twenty_distinct_valid_configurations(self):
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                estimator = self.factory(name, technical=False)
                sampled = list(ParameterSampler(parameter_space(name, technical=False),
                                                n_iter=20, random_state=42))
                self.assertEqual(len(sampled), 20)
                keys = {json.dumps(p, sort_keys=True, default=str) for p in sampled}
                self.assertEqual(len(keys), 20)
                for parameters in sampled:
                    clone(estimator).set_params(**parameters)

    def test_numeric_and_categorical_train_only_statistics(self):
        estimator = self.factory('logistic_regression').fit(self.X, self.y)
        validation = pd.DataFrame({
            'signal': [np.nan, 1000000.0, 5.0],
            'empty_in_train': [7.0, np.nan, 4.0],
            'category': ['brand_new', None, 'A'],
            'empty_category': ['first_observed_outside_train', None, np.nan],
        }, index=pd.Index([900001, 900002, 900003], name='SK_ID_CURR'))
        imputers = [t for t in walk_transformers(estimator)
                    if isinstance(t, SimpleImputer) and t.strategy == 'median']
        self.assertEqual(len(imputers), 1)
        imputer = imputers[0]
        expected = np.array([self.X.signal.median(), 0.0])
        np.testing.assert_array_equal(imputer.statistics_, expected)
        encoders = [t for t in walk_transformers(estimator) if isinstance(t, OneHotEncoder)]
        self.assertEqual(len(encoders), 1)
        self.assertEqual(encoders[0].handle_unknown, 'ignore')
        learned_categories = [cat.copy() for cat in encoders[0].categories_]
        self.assertNotIn('brand_new', learned_categories[0])
        p = estimator.predict_proba(validation)
        self.assertEqual(p.shape, (3, 2))
        self.assertTrue(np.isfinite(p).all())
        np.testing.assert_array_equal(imputer.statistics_, expected)
        for before, after in zip(learned_categories, encoders[0].categories_):
            np.testing.assert_array_equal(before, after)
        order = [2, 0, 1]
        np.testing.assert_allclose(estimator.predict_proba(validation.iloc[order]), p[order])

    def test_nullable_boolean_category_accepts_true_false_and_missing(self):
        frame=self.X.copy()
        frame['bool_category']=pd.array([True,False,None]*30,dtype='boolean')
        original=frame.copy(deep=True)
        validation=frame.iloc[:3].copy()
        validation['bool_category']=pd.array([False,None,True],dtype='boolean')
        validation['category']=['unknown_category',None,'A']
        for name in ['logistic_regression','xgboost']:
            with self.subTest(model=name):
                estimator=make_estimator(name,NUMERIC,CATEGORICAL+['bool_category'],
                                         seed=42,technical=True,threads=1)
                estimator.fit(frame,self.y)
                encoders=[t for t in walk_transformers(estimator) if isinstance(t,OneHotEncoder)]
                self.assertEqual(len(encoders),1)
                self.assertEqual(set(encoders[0].categories_[-1]),{'True','False',MISSING_CATEGORY})
                probabilities=estimator.predict_proba(validation)
                self.assertEqual(probabilities.shape,(3,2))
                self.assertTrue(np.isfinite(probabilities).all())
                np.testing.assert_allclose(probabilities.sum(axis=1),1,atol=1e-6)
        pd.testing.assert_frame_equal(frame,original)

    def test_randomized_search_fits_preprocessing_inside_each_training_fold(self):
        X = self.X.copy()
        X.loc[X.index[-1], 'signal'] = 1000000
        cv = list(StratifiedKFold(3, shuffle=True, random_state=23).split(X, self.y))
        records = []
        estimator = self.factory('logistic_regression')
        search = RandomizedSearchCV(estimator, {'model__C': [0.1, 1.0]}, n_iter=2,
                                    scoring='neg_log_loss', cv=cv, refit=False,
                                    random_state=42, n_jobs=1, error_score='raise')
        with record_numeric_imputer_fits(records):
            search.fit(X, self.y)
        self.assertEqual(len(records), 6)
        self.assertTrue(np.isfinite(search.cv_results_['mean_test_score']).all())
        expected_ids = {tuple(X.iloc[train].index) for train, _ in cv}
        self.assertEqual({row['row_ids'] for row in records}, expected_ids)
        self.assertTrue(all(row['n_rows'] < len(X) for row in records))
        for row in records:
            np.testing.assert_array_equal(row['statistics'],
                                          [X.loc[list(row['row_ids']), 'signal'].median(), 0.0])

    def test_stacking_internal_oof_refits_each_base_preprocessor(self):
        estimator = self.factory('stacking')
        # Общая обучаемая предобработка перед StackingClassifier запрещена.
        self.assertEqual(list(estimator.named_steps), ['model'])
        stack = estimator.named_steps['model']
        self.assertFalse(stack.passthrough)
        self.assertEqual(stack.stack_method, 'predict_proba')
        for _, base in stack.estimators:
            self.assertIsInstance(base, Pipeline)
            self.assertIn('preprocess', base.named_steps)
        records = []
        with record_numeric_imputer_fits(records), warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            estimator.fit(self.X, self.y)
        expected_splits = list(stack.cv.split(self.X, self.y))
        expected_oof = {tuple(self.X.iloc[train].index) for train, _ in expected_splits}
        full = [record for record in records if record['n_rows'] == len(self.X)]
        oof = [record for record in records if record['n_rows'] < len(self.X)]
        self.assertEqual(len(full), len(stack.estimators))
        self.assertEqual(len(oof), len(stack.estimators) * 3)
        self.assertEqual({record['row_ids'] for record in oof}, expected_oof)
        for row in records:
            np.testing.assert_array_equal(row['statistics'],
                [self.X.loc[list(row['row_ids']), 'signal'].median(), 0.0])
        self.assertEqual(estimator.predict_proba(self.X.iloc[:3]).shape, (3, 2))

    def test_seven_models_predict_probabilities_and_serialize(self):
        validation = self.X.iloc[:5].copy()
        validation.loc[validation.index[0], 'category'] = 'unseen_validation_category'
        original = self.X.copy(deep=True)
        with tempfile.TemporaryDirectory(prefix='rmt_models_') as folder:
            for name in MODEL_NAMES:
                with self.subTest(model=name), warnings.catch_warnings():
                    warnings.simplefilter('ignore', ConvergenceWarning)
                    estimator = self.factory(name).fit(self.X, self.y)
                    probabilities = estimator.predict_proba(validation)
                    self.assertEqual(probabilities.shape, (len(validation), 2))
                    self.assertTrue(np.isfinite(probabilities).all())
                    self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
                    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
                    path = Path(folder) / f'{name}.joblib'
                    joblib.dump(estimator, path)
                    loaded = joblib.load(path)
                    np.testing.assert_allclose(loaded.predict_proba(validation), probabilities)
        pd.testing.assert_frame_equal(self.X, original)


if __name__ == '__main__':
    unittest.main(verbosity=2)
