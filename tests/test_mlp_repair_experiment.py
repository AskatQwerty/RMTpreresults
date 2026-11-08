"""Orchestration tests with fake estimators; no real Home Credit model fits."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import mlp_repair_experiment as experiment
from src.model_training import atomic_json, _complete


class FakeProbabilityModel:
    classes_ = np.array([0, 1])

    def __init__(self, correct_probability):
        self.correct_probability = correct_probability

    def predict_proba(self, X):
        p = np.where(X.signal.to_numpy() == 1, self.correct_probability,
                     1-self.correct_probability)
        return np.column_stack([1-p, p])


class RepairExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='rmt_mlp_repair_runner_')
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.root = self.project / 'artifacts/training' / experiment.SCOPE
        self.root.mkdir(parents=True)
        self.data = self.project / 'compact'
        self.data.mkdir()
        self.ids = np.arange(101, 113)
        self.y = pd.Series(np.tile([0, 1], 6), index=pd.Index(self.ids, name='SK_ID_CURR'), name='TARGET')
        self.test_ids = np.arange(201, 205)
        self.cv = [(np.flatnonzero(np.arange(12)//4 != fold), np.flatnonzero(np.arange(12)//4 == fold))
                   for fold in range(3)]
        self.columns = {str(level): {'numeric': ['signal'], 'categorical': ['category']} for level in range(1, 5)}
        X = pd.DataFrame({'signal': self.y.to_numpy(), 'category': ['a', 'b']*6})
        for level in range(1, 5):
            X.to_parquet(self.data / f'X_level_{level}.parquet', index=False)
        original = self.project / 'original'
        original.mkdir()
        pd.DataFrame({'SK_ID_CURR': self.test_ids}).to_parquet(original / 'test_ids.parquet', index=False)
        baseline = self.project / 'baseline'
        for level in range(1, 5):
            atomic_json(baseline / 'mlp' / f'level_{level}' / 'metrics.json', {'metrics': {'log_loss': .9}})
        self.plan = {'fingerprint': 'fake-test-context', 'levels': [1, 2, 3, 4],
            'variants': experiment.VARIANTS, 'settings': experiment.SETTINGS,
            'code_sha256': {}, 'inputs': {}, 'data_root': str(self.data),
            'prepared_dir': str(self.project / 'prepared'), 'original_pilot': str(original),
            'original_final_root': str(baseline), 'test_gate': 'test fixed gate',
            'original_cv_scores': [dict(level=l, fold=f, log_loss=.9) for l in range(1, 5) for f in range(3)]}
        atomic_json(self.root / 'plan.json', self.plan)
        atomic_json(self.root / 'protected_before.json', {})
        self.fit_ids, self.test_loads, self.test_failures = [], [], set()
        self.fail_test_level = None

    def fit(self, X, y, numeric, categorical, variant, **settings):
        self.assertTrue(X.index.equals(y.index))
        self.assertTrue(set(X.index) <= set(self.ids))
        self.assertTrue(set(X.index).isdisjoint(self.test_ids))
        self.assertEqual(settings, experiment.SETTINGS)
        self.fit_ids.append(tuple(X.index))
        order = [v['name'] for v in experiment.VARIANTS].index(variant['name'])
        return FakeProbabilityModel(.65+.05*order), {
            'total_seconds': .01, 'best_epoch': 3,
            'inner_optimizer': {'epochs_completed': 5},
            'final_optimizer': {'epochs_completed': 3},
            'partial_fit_calls': 8, 'warnings': []}

    def prepared_factory(self, _path):
        case = self
        # This is the first possible global test metadata access in the runner.
        self.assertEqual(len(experiment.get_records(self.root, self.plan)), 48)
        selection = json.loads((self.root / 'selection.json').read_text())
        self.assertEqual(len(selection['selections']), 4)
        self.assertFalse(selection['test_used_for_selection'])

        class FakePrepared:
            test = case.test_ids

            def load(self, level, requested, *, purpose):
                case.assertTrue(np.array_equal(requested, case.test_ids))
                case.assertTrue(_complete(case.root / 'final' / f'level_{level}' / 'fit_checkpoint', case.plan))
                case.test_loads.append(level)
                if level == case.fail_test_level and level not in case.test_failures:
                    case.test_failures.add(level)
                    raise RuntimeError('synthetic prediction interruption')
                y = pd.Series([0, 1, 0, 1], index=pd.Index(requested, name='SK_ID_CURR'), name='TARGET')
                X = pd.DataFrame({'signal': y.to_numpy(), 'category': ['a', 'b', 'a', 'b']}, index=y.index)
                return X, y

        return FakePrepared()

    def execute(self, fit=None):
        with patch.object(experiment, 'load_cohort', return_value=(self.ids, self.y, self.cv, self.columns)), \
             patch.object(experiment, 'fit_mlp_variant', side_effect=self.fit if fit is None else fit), \
             patch.object(experiment, 'PreparedData', side_effect=self.prepared_factory), \
             patch.object(experiment, 'protected_snapshot', return_value={}):
            return experiment.run(self.project)

    def test_exact48_cv_then4_refits_and_completed_resume(self):
        result = self.execute()
        self.assertEqual(result['completed_cv'], 48)
        self.assertEqual(result['final_levels_completed'], 4)
        self.assertEqual(result['cv_optimizer_trajectories'], 96)
        self.assertEqual(result['final_optimizer_trajectories'], 8)
        self.assertEqual([len(ids) for ids in self.fit_ids], [8]*48 + [12]*4)
        self.assertEqual(self.test_loads, [1, 2, 3, 4])
        for s in json.loads((self.root / 'selection.json').read_text())['selections']:
            self.assertEqual(s['variant'], 'stop_clipped_regularized')
            self.assertTrue(s['eligible_for_test'])
        self.execute(fit=lambda *a, **k: self.fail('Completed fits must not repeat'))
        self.assertEqual(len(self.fit_ids), 52)
        self.assertEqual(self.test_loads, [1, 2, 3, 4])

    def test_resume_failed_test_prediction_reuses_completed_fit(self):
        self.fail_test_level = 2
        with self.assertRaisesRegex(RuntimeError, 'synthetic prediction interruption'):
            self.execute()
        self.assertEqual(len(self.fit_ids), 50)  # 48CV + levels1,2; level2 fit is complete.
        self.assertTrue(_complete(self.root / 'final/level_2/fit_checkpoint', self.plan))
        self.assertFalse(_complete(self.root / 'final/level_2', self.plan))
        result = self.execute()
        self.assertEqual(result['final_levels_completed'], 4)
        self.assertEqual(len(self.fit_ids), 52)
        self.assertEqual(self.test_loads, [1, 2, 2, 3, 4])

    def test_selection_change_and_checkpoint_corruption_refuse_refit(self):
        self.execute()
        selection_path = self.root / 'selection.json'
        original = selection_path.read_bytes()
        selection = json.loads(original)
        selection['selections'][0]['variant'] = 'tampered'
        atomic_json(selection_path, selection)
        with self.assertRaisesRegex(ValueError, 'Frozen selections changed'):
            self.execute(fit=lambda *a, **k: self.fail('Selection tampering must not refit'))
        selection_path.write_bytes(original)
        bad = self.root / 'cv/level_1/stop_only/fold_0/result.json'
        bad.write_text(bad.read_text() + ' ', encoding='utf-8')
        with self.assertRaises(RuntimeError):
            self.execute(fit=lambda *a, **k: self.fail('Corrupt checkpoints must not refit'))
        self.assertEqual(len(self.fit_ids), 52)


if __name__ == '__main__':
    unittest.main()
