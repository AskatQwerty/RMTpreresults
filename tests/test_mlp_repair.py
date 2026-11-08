"""Synthetic-only checks of supplemental MLP preprocessing and train-only refit."""
from __future__ import annotations

import copy
import io
import json
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.mlp_repair import (make_repair_preprocessor, fit_mlp_variant, _select_epochs,
                            _preprocessor_state)


class MlpRepairTests(unittest.TestCase):
    def fixture(self, n=80):
        rng = np.random.RandomState(4)
        X = pd.DataFrame({'value': rng.normal(size=n), 'constant': np.zeros(n),
                          'empty': np.full(n, np.nan),
                          'category': np.where(np.arange(n)%3 == 0, 'a', 'b')},
                         index=np.arange(10000, 10000+n))
        X.loc[X.index[::7], 'value'] = np.nan
        y = pd.Series((np.arange(n)%5 == 0).astype(np.int8), index=X.index)
        return X, y

    def test_clipping_only_numeric_values_and_train_statistics_unchanged(self):
        train = pd.DataFrame({'value': [1., 2., np.nan, 3.], 'empty': [np.nan]*4,
                              'cat': ['a', 'b', None, 'a']})
        valid = pd.DataFrame({'value': [1e6, np.nan], 'empty': [65250., np.nan],
                              'cat': ['unknown', None]})
        p = make_repair_preprocessor(['value', 'empty'], ['cat'], clip_z=5)
        p.fit_transform(train)
        before = copy.deepcopy(_preprocessor_state(p))
        out = p.transform(valid).toarray()
        self.assertEqual(before, _preprocessor_state(p))
        np.testing.assert_array_equal(out[:, :2], [[5, 5], [0, 0]])
        np.testing.assert_array_equal(out[:, 2:4], [[0, 0], [1, 1]])
        self.assertEqual(out[0, 4:].sum(), 0)  # unknown category
        self.assertEqual(out[1, 4:].sum(), 1)  # known missing category
        self.assertEqual(p.transform(train).shape[1], out.shape[1])

    def test_only_inner_train_fit_then_fresh_all_train_fit(self):
        X, y = self.fixture()
        train, valid = train_test_split(np.arange(len(X)), test_size=.15,
                                       stratify=y, random_state=42)
        X.loc[X.index[valid], 'constant'] = 10000
        seen = []
        original = make_repair_preprocessor
        def factory(*args, **kwargs):
            p = original(*args, **kwargs)
            original_fit = p.fit_transform
            def record(frame, labels=None, **options):
                seen.append(frame.index.tolist())
                return original_fit(frame, labels, **options)
            p.fit_transform = record
            return p
        with patch('src.mlp_repair.make_repair_preprocessor', side_effect=factory):
            pipeline, d = fit_mlp_variant(X, y, ['value', 'constant', 'empty'], ['category'],
                {'name': 'clip', 'alpha': .1, 'clip_z': 5}, max_epochs=3, patience=2)
        self.assertEqual(seen, [X.index[train].tolist(), X.index.tolist()])
        self.assertEqual(d['inner_preprocessing']['scaler_mean'][1], 0)
        self.assertEqual(d['final_preprocessing']['scaler_mean'][1], 1500)
        self.assertEqual(d['inner_preprocessing']['scaler_n_samples_seen'], len(train))
        self.assertEqual(d['final_preprocessing']['scaler_n_samples_seen'], len(X))
        self.assertEqual(set(d['internal_split']['train_ids']) | set(d['internal_split']['validation_ids']), set(X.index))
        self.assertFalse(set(d['internal_split']['train_ids']) & set(d['internal_split']['validation_ids']))
        self.assertEqual(d['final_optimizer']['samples_seen'], len(X)*d['best_epoch'])
        self.assertEqual(d['final_optimizer']['loss_curve_length'], d['best_epoch'])
        self.assertEqual(pipeline.named_steps['model'].random_state, 42)
        self.assertEqual(d['optimizer_fits'], 2)
        self.assertTrue(np.isfinite(pipeline.predict_proba(X)).all())
        json.dumps(d, allow_nan=False)
        # Remove the test-only local spy before exercising normal serialization.
        del pipeline.named_steps['preprocess'].fit_transform
        serialized = io.BytesIO()
        joblib.dump(pipeline, serialized)
        serialized.seek(0)
        restored = joblib.load(serialized)
        np.testing.assert_array_equal(pipeline.predict_proba(X), restored.predict_proba(X))
        self.assertEqual(restored.named_steps['model'].random_state, 42)

    def test_best_epoch_is_strict_minimum_not_patience_anchor(self):
        class Fake:
            loss_ = .1
        # Strict minima continue under min_delta; patience still expires.
        validation = iter([.4, .3999, .3998])
        with patch('src.mlp_repair._epoch'), patch('src.mlp_repair._probability', return_value=np.array([.5,.5])), \
             patch('src.mlp_repair.log_loss', side_effect=lambda y,p,labels: .2 if y[0] == 0 else next(validation)):
            epoch, curve, reason = _select_epochs(Fake(), np.zeros((2,1)), np.array([0,1]),
                np.zeros((2,1)), np.array([1,0]), max_epochs=10, patience=2, min_delta=.001)
        self.assertEqual(epoch, 3)
        self.assertEqual(len(curve), 3)
        self.assertEqual(reason, 'internal_validation_log_loss_patience')

    def test_deterministic_and_reject_misaligned_target(self):
        X, y = self.fixture()
        variant = {'name': 'standard', 'alpha': .1, 'clip_z': None}
        a, da = fit_mlp_variant(X,y,['value','constant','empty'],['category'],variant,max_epochs=3,patience=2)
        b, db = fit_mlp_variant(X,y,['value','constant','empty'],['category'],variant,max_epochs=3,patience=2)
        np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))
        self.assertEqual(da['best_epoch'], db['best_epoch'])
        self.assertEqual(da['internal_split'], db['internal_split'])
        with self.assertRaisesRegex(ValueError, 'row ID order'):
            fit_mlp_variant(X,y.iloc[::-1],['value','constant','empty'],['category'],variant,max_epochs=2)


if __name__ == '__main__':
    unittest.main()
