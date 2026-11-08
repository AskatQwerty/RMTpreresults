"""Fake-only extension tests: no sklearn model fit, predict or prepared-X read."""
from contextlib import ExitStack
from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import pilot_extension as extension
from src import model_training as training


class FakeData:
    instances = []

    def __init__(self, path):
        self.path = Path(path)
        self.test = np.array([900, 901])
        self.development = np.arange(101, 107)
        self.features = {str(level): ['numeric'] for level in range(1, 5)}
        self.loaded_scopes = []
        self.instances.append(self)

    def load(self, level, ids, *, purpose):
        assert np.array_equal(ids, np.arange(101, 107))
        assert purpose == 'pilot_extension_cv_train'
        self.loaded_scopes.append({'contains_test': False, 'level': level})
        return pd.DataFrame({'numeric': np.arange(6.)}, index=ids), pd.Series([0, 1] * 3, index=ids)

    def columns(self, level):
        return ['numeric'], []


class PilotExtensionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='rmt_extension_')
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.output = self.base / 'output'
        self.pilot = self.base / 'legacy'
        self.prepared = self.base / 'prepared'
        self.calls = []
        self.fail_after = None
        FakeData.instances.clear()
        configs = {}
        for model in extension.MODEL_ORDER:
            configs[model] = []
            for number in range(1, 21):
                params = {'model__dummy': number}
                configs[model].append({'candidate_index': number - 1, 'candidate_number': number,
                                      'params': params,
                                      'config_sha256': hashlib.sha256(training._json_text(params).encode()).hexdigest()})
            for level in extension.LEVELS:
                path = self.pilot / model / f'level_{level}' / 'fit_checkpoint/cv_results.parquet'
                rows = []
                for candidate in range(5):
                    rows.append({'candidate': candidate,
                                 'params_json': training._json_text(configs[model][candidate]['params']),
                                 'mean_neg_log_loss': -.5 + candidate * .01,
                                 'std_neg_log_loss': 0., 'rank': 5 - candidate,
                                 'mean_fit_seconds': 1., 'std_fit_seconds': 0.,
                                 'mean_score_seconds': .1, 'std_score_seconds': 0.,
                                 **{f'fold_{fold}_neg_log_loss': -.5 + candidate * .01 for fold in range(3)}})
                training._atomic_parquet(pd.DataFrame(rows), path)
        self.plan = {'fingerprint': 'immutable-test-plan', 'pilot_fingerprint': 'legacy-test',
                     'pilot_root': str(self.pilot), 'prepared_dir': str(self.prepared),
                     'cohort_contract': {'sample_n': 6}, 'configurations': configs}
        self.splits = [(np.array([j for j in range(6) if j // 2 != fold]),
                        np.array([fold * 2, fold * 2 + 1])) for fold in range(3)]

    def ensure(self, prepared, pilot, root):
        path = Path(root) / 'configuration_plan.json'
        if path.exists():
            self.assertEqual(training.read_json(path), self.plan)
        else:
            training.atomic_json(path, self.plan)
        return self.plan

    def fake_fit(self, model, numeric, categorical, params, X, y, train, validation, *, threads):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError('simulated interruption')
        self.assertEqual(threads, 2)
        self.assertEqual(numeric, ['numeric'])
        self.assertFalse(categorical)
        self.assertEqual(len(train), 4)
        self.assertEqual(len(validation), 2)
        self.assertFalse(set(train) & set(validation))
        self.assertEqual(list(X.index), [101, 102, 103, 104, 105, 106])
        self.assertTrue(X.index.equals(y.index))
        number = params['model__dummy']
        self.assertTrue(6 <= number <= 10)
        self.calls.append((model, number, tuple(validation)))
        return {'neg_log_loss': -.4 + number * .001,
                'fit_seconds': 1., 'score_seconds': .1,
                'warnings': [{'stage': 'fit', 'category': 'FakeWarning', 'message': 'retained', 'count': 1}]}, {
                    'mlp': [{'fitted': {'n_iter_': 3, 'loss_curve_': [1., .8, .7]}}]
                }

    def run_batch(self):
        with ExitStack() as stack:
            stack.enter_context(patch.object(extension, 'ensure_extension_plan', side_effect=self.ensure))
            stack.enter_context(patch.object(extension, 'build_extension_plan', return_value=self.plan))
            stack.enter_context(patch.object(extension, 'PreparedData', FakeData))
            stack.enter_context(patch.object(extension, 'load_pilot_split', return_value=(np.arange(101, 107), self.splits)))
            stack.enter_context(patch.object(extension, '_fit_one_fold', side_effect=self.fake_fit))
            stack.enter_context(patch.object(extension.gc, 'collect', return_value=0))
            stack.enter_context(patch('builtins.print'))
            return extension.run_extension(self.prepared, self.pilot, self.output,
                                           report_dir=self.base / 'reports')

    def test_exact_420_new_fits_old_scores_preserved_and_no_test(self):
        result = self.run_batch()
        self.assertEqual(len(self.calls), 420)
        self.assertEqual(result['overall_completed_cv_fits'], 840)
        self.assertEqual(result['batch_completed_cv_fits'], 420)
        self.assertEqual(result['pairs_with_requested_batch_complete'], 28)
        self.assertEqual(result['new_fits_this_invocation'], 420)
        self.assertEqual(result['test_evaluations'], 0)
        self.assertFalse(result['refit'])
        self.assertEqual(len(FakeData.instances[0].loaded_scopes), 28)
        self.assertEqual(len(list(self.output.glob('*/level_*/candidate_*/fold_*/complete.json'))), 420)
        for model in extension.MODEL_ORDER:
            for level in extension.LEVELS:
                combined = pd.read_parquet(self.output / model / f'level_{level}/combined_cv_results.parquet')
                original = pd.read_parquet(self.pilot / model / f'level_{level}/fit_checkpoint/cv_results.parquet')
                self.assertEqual(combined.candidate_number.tolist(), list(range(1, 11)))
                for fold in range(3):
                    column = f'fold_{fold}_neg_log_loss'
                    self.assertEqual(combined.loc[:4, column].tolist(), original[column].tolist())
        self.assertFalse(list(self.output.rglob('pipeline.joblib')))

    def test_fold_resume_skips_completed_and_rejects_corruption(self):
        self.fail_after = 4
        with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
            self.run_batch()
        progress = training.read_json(self.output / 'progress.json')
        self.assertEqual(progress['batch_completed_cv_fits'], 4)
        self.assertEqual(progress['overall_completed_cv_fits'], 424)
        self.fail_after = None
        result = self.run_batch()
        self.assertEqual(len(self.calls), 420)
        self.assertEqual(result['new_fits_this_invocation'], 416)
        result = self.run_batch()
        self.assertEqual(len(self.calls), 420)
        self.assertEqual(result['new_fits_this_invocation'], 0)
        checkpoint = next(self.output.glob('*/level_*/candidate_*/fold_*/result.json'))
        checkpoint.write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError, 'артефакт'):
            self.run_batch()

    def test_fit_helper_uses_fresh_full_estimator_train_then_saved_scorer(self):
        ids = np.arange(101, 107)
        X = pd.DataFrame({'numeric': np.arange(6.)}, index=ids)
        y = pd.Series([0, 1] * 3, index=ids)
        train, validation = self.splits[0]
        observed = []

        class FakeEstimator:
            def set_params(self, **params):
                observed.append(('params', params))
                return self

            def fit(self, Xt, yt):
                observed.append(('fit', list(Xt.index), list(yt.index)))
                return self

        def scorer(estimator, Xv, yv):
            observed.append(('score', list(Xv.index), list(yv.index)))
            return -.25

        with patch.object(extension, 'make_estimator', return_value=FakeEstimator()) as maker, \
             patch.object(extension, 'get_scorer', return_value=scorer) as scorer_factory, \
             patch.object(extension, 'extract_fit_diagnostics', return_value={'mlp': []}):
            result, _ = extension._fit_one_fold('stacking', ['numeric'], [], {'model__dummy': 6},
                                              X, y, train, validation, threads=2)
        self.assertEqual(result['neg_log_loss'], -.25)
        maker.assert_called_once_with('stacking', ['numeric'], [], seed=42, technical=False, threads=2)
        scorer_factory.assert_called_once_with('neg_log_loss')
        self.assertEqual(observed[1][1], [103, 104, 105, 106])
        self.assertEqual(observed[2][1], [101, 102])


if __name__ == '__main__':
    unittest.main()
