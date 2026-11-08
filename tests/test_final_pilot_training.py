"""Synthetic final-refit checks: train-only fit, prediction resume and SHA refusal."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from src import final_pilot_training as final


class RecordingTransform(BaseEstimator, TransformerMixin):
    fitted_ids = []

    def fit(self, X, y=None):
        type(self).fitted_ids.append(X.index.tolist())
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return X


class FinalPairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.ids = np.array([101, 102, 103, 104, 105, 106])
        self.test_ids = np.array([201, 202, 203, 204])
        self.context = {'fingerprint': 'fixture-context', 'scope': final.SCOPE,
                        'test_rows': 4, 'threads': 2}
        final.atomic_json(self.root / 'experiment_manifest.json', self.context)
        self.stage = self.root / 'logistic_regression/level_1'
        self.access = []
        RecordingTransform.fitted_ids = []
        owner = self

        class FakeData:
            test = owner.test_ids
            features = {'1': ['value']}

            def __init__(self, unused):
                self.loaded_scopes = []

            def columns(self, level):
                return ['value'], []

            def load(self, level, requested, *, purpose):
                is_test = np.array_equal(requested, owner.test_ids)
                if is_test:
                    if not final._complete(owner.stage / 'fit_checkpoint', owner.context):
                        raise AssertionError('Test X requested before a verified fit checkpoint')
                else:
                    np.testing.assert_array_equal(requested, owner.ids)
                owner.access.append(purpose)
                self.loaded_scopes.append({'purpose': purpose, 'rows': len(requested), 'contains_test': is_test})
                X = pd.DataFrame({'value': np.arange(len(requested), dtype=float)},
                                 index=pd.Index(requested, name='SK_ID_CURR'))
                y = pd.Series(np.arange(len(requested)) % 2, index=X.index, name='TARGET')
                return X, y

        def make(*args, **kwargs):
            self.assertFalse(kwargs['technical'])
            self.assertEqual(kwargs['seed'], 42)
            self.assertEqual(kwargs['threads'], 2)
            return Pipeline([('preprocess', RecordingTransform()),
                             ('model', DummyClassifier(strategy='prior'))])

        self.patches = [
            patch.object(final, '_validate_runtime'),
            patch.object(final, 'PreparedData', FakeData),
            patch.object(final, 'load_pilot_split', return_value=(self.ids, [])),
            patch.object(final, '_selected_pair', return_value={'candidate_number': 7,
                'params': {}, 'mean_cv_log_loss': .3}),
            patch.object(final, 'make_estimator', side_effect=make),
            patch.object(final, 'extract_fit_diagnostics', return_value={'mlp': []}),
            patch.object(final, '_saved_cv', return_value=pd.DataFrame({'candidate': range(20)})),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.temp.cleanup()

    def run_pair(self):
        return final.run_final_pair('synthetic-only', self.root, 'logistic_regression', 1)

    def test_train_only_once_and_completed_pair_skips_all_data(self):
        result = self.run_pair()
        self.assertEqual(RecordingTransform.fitted_ids, [self.ids.tolist()])
        self.assertEqual(self.access, ['final_selected_20k_train', 'final_selected_test_evaluation'])
        self.assertEqual(result['fit_calls']['candidate_cv'], 0)
        self.assertEqual(result['fit_calls']['manual_final_refit'], 1)
        saved = pd.read_parquet(self.stage / 'probabilities.parquet')
        self.assertEqual(saved.SK_ID_CURR.tolist(), self.test_ids.tolist())
        self.assertEqual(saved.TARGET.tolist(), [0, 1, 0, 1])
        self.assertEqual(self.run_pair(), result)
        self.assertEqual(RecordingTransform.fitted_ids, [self.ids.tolist()])
        self.assertEqual(len(self.access), 2)

    def test_prediction_failure_resumes_saved_fit_without_refitting(self):
        with patch.object(final, 'positive_probability', side_effect=RuntimeError('synthetic prediction failure')):
            with self.assertRaisesRegex(RuntimeError, 'synthetic prediction failure'):
                self.run_pair()
        self.assertTrue(final._complete(self.stage / 'fit_checkpoint', self.context))
        self.assertFalse((self.stage / 'complete.json').exists())
        self.run_pair()
        self.assertEqual(RecordingTransform.fitted_ids, [self.ids.tolist()])
        self.assertEqual(self.access.count('final_selected_20k_train'), 1)
        self.assertEqual(self.access.count('final_selected_test_evaluation'), 2)

    def test_corrupt_completed_pair_refuses_retraining(self):
        self.run_pair()
        (self.stage / 'metrics.json').write_bytes(b'corrupt')
        with self.assertRaises(RuntimeError):
            self.run_pair()
        self.assertEqual(RecordingTransform.fitted_ids, [self.ids.tolist()])


if __name__ == '__main__':
    unittest.main()
