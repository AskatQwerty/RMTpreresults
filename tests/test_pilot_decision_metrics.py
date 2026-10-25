"""Проверки формул/границ и целостности отчётов; никакого обучения моделей."""
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.pilot_decision_metrics import export_decision_metrics, _hash, _calibration_rows


class PilotDecisionMetricsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='rmt_decision_metrics_')
        base = Path(self.temporary.name)
        self.root, self.prepared, self.reports = base/'pilot', base/'prepared', base/'reports'
        self.root.mkdir()
        (self.prepared/'splits').mkdir(parents=True)
        self.ids = np.arange(11, 17)
        self.target = np.array([0, 1, 1, 0, 0, 1])
        self.probabilities = np.array([.1, .2, .6, .8, 0., 1.])
        id_frame = pd.DataFrame({'SK_ID_CURR': self.ids})
        id_frame.to_parquet(self.root/'test_ids.parquet', index=False)
        id_frame.to_parquet(self.prepared/'splits/test_ids.parquet', index=False)
        pd.DataFrame({'SK_ID_CURR': self.ids, 'TARGET': self.target}).to_parquet(
            self.prepared/'target.parquet', index=False)
        pd.DataFrame({'SK_ID_CURR': self.ids, 'CODE_GENDER': ['F','F','M','M','XNA','M'],
            'GENDER_GROUP': ['F','F','M','M','unknown','M'],
            'AGE_YEARS': [24.,25.,24.,40.,np.nan,20.],
            'AGE_GROUP': ['<25','>=25','<25','>=25','unknown','<25']}).to_parquet(
            self.prepared/'fairness.parquet', index=False)
        self.manifest = {'mode': 'pilot_20k_5config', 'fingerprint': 'fixture-fingerprint',
            'inputs': [{'file': name, 'sha256': _hash(self.prepared/name)} for name in (
                'fairness.parquet','target.parquet','splits/test_ids.parquet')]}
        self.write_json(self.root/'experiment_manifest.json', self.manifest)
        self.write_json(self.root/'cohort_contract.json', {'artifact_sha256': {
            'test_ids.parquet': _hash(self.root/'test_ids.parquet')}})
        self.stage = self.root/'logistic_regression/level_4'
        self.stage.mkdir(parents=True)
        self.frame = pd.DataFrame({'SK_ID_CURR': self.ids, 'TARGET': self.target,
            'probability_default': self.probabilities, 'scope': 'pilot_20k_5config',
            'model': 'logistic_regression', 'level': 4, 'evaluation_split': 'test'})
        self.frame.to_parquet(self.stage/'probabilities.parquet', index=False)
        self.write_json(self.stage/'metrics.json', {'scope': 'pilot_20k_5config',
            'model': 'logistic_regression', 'level': 4, 'test_rows': 6,
            'metrics': {'rows': 6, 'brier_score': float(np.mean((self.probabilities-self.target)**2))}})
        self.refresh_marker()

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value), encoding='utf-8')

    def refresh_marker(self):
        self.write_json(self.stage/'complete.json', {'fingerprint': self.manifest['fingerprint'],
            'artifact_sha256': {name: _hash(self.stage/name) for name in ('probabilities.parquet','metrics.json')}})

    def test_profit_groups_unknowns_and_existing_age_boundary(self):
        metadata = export_decision_metrics(self.root, self.reports, self.prepared)
        scenarios = pd.read_csv(self.reports/'decision_scenarios.csv')
        groups = pd.read_csv(self.reports/'decision_groups.csv')
        self.assertEqual(len(scenarios), 6)
        self.assertEqual(len(groups), 36)
        breakeven = scenarios[(scenarios.loss_to_margin==5) & (scenarios.threshold_rule=='break_even')].iloc[0]
        self.assertAlmostEqual(breakeven.realized_profit_per_applicant, 2/6)
        reference = scenarios[(scenarios.loss_to_margin==5) & (scenarios.threshold_rule=='reference_0_5')].iloc[0]
        self.assertAlmostEqual(reference.realized_profit_per_applicant, (1-5+1)/6)
        self.assertAlmostEqual(reference.gender_approval_rate_gap, 1.)
        self.assertAlmostEqual(reference.gender_default_tpr_gap, 1.)
        self.assertEqual(reference.gender_unknown_n, 1)
        age = groups[(groups.loss_to_margin==5) & (groups.threshold_rule=='reference_0_5') &
                     (groups.attribute=='AGE_GROUP') & (groups.group=='>=25')].iloc[0]
        self.assertEqual(age.n, 2)  # age25 is in the preserved >=25 group.
        self.assertEqual(age.default_n, 1)
        self.assertEqual(age.default_tpr, 0)
        unknown = groups[(groups.attribute=='CODE_GENDER') & (groups.group=='unknown')].iloc[0]
        self.assertFalse(unknown.included_in_binary_gap)
        self.assertTrue(pd.isna(unknown.default_tpr))  # No defaults: undefined, not 0.
        self.assertEqual(metadata['additional_fit_calls'], 0)
        self.assertEqual(metadata['additional_predict_calls'], 0)
        self.assertFalse(metadata['winner_selected'])
        self.assertEqual(metadata['completed_level4_models'], ['logistic_regression'])
        self.assertIn('age25', metadata['definitions']['age'])
        self.assertTrue((self.reports/'decision_scenarios.tex').exists())
        self.assertTrue((self.reports/'decision_groups.tex').exists())
        for name in ('decision_scenarios.csv', 'decision_groups.csv', 'calibration_bins.csv'):
            self.assertNotIn(b'\r\r\n', (self.reports/name).read_bytes())

    def test_calibration_fixed_boundaries_empty_bins_and_one(self):
        values = np.array([0.,.1,.3,.9,1.])
        rows = _calibration_rows('fixture', np.array([0,1,0,1,1]), values)
        self.assertEqual(len(rows), 10)
        self.assertEqual(sum(r['n'] for r in rows), 5)
        self.assertEqual(rows[1]['n'], 1)
        self.assertEqual(rows[3]['n'], 1)
        self.assertEqual(rows[9]['n'], 2)
        self.assertTrue(rows[9]['upper_bound_inclusive'])
        self.assertEqual(rows[2]['n'], 0)
        self.assertIsNone(rows[2]['mean_probability_default'])
        self.assertIsNone(rows[2]['observed_default_rate'])

    def test_corrupt_prediction_hash_rejected_without_credible_metrics(self):
        with (self.stage/'probabilities.parquet').open('ab') as target:
            target.write(b'corrupt')
        metadata = export_decision_metrics(self.root, self.reports, self.prepared)
        self.assertEqual(metadata['completed_level4_models'], [])
        self.assertEqual(metadata['invalid_models'][0]['model'], 'logistic_regression')
        self.assertEqual(len(pd.read_csv(self.reports/'decision_scenarios.csv')), 0)

    def test_hash_valid_but_wrong_test_order_rejected(self):
        self.frame.iloc[::-1].to_parquet(self.stage/'probabilities.parquet', index=False)
        self.refresh_marker()
        metadata = export_decision_metrics(self.root, self.reports, self.prepared)
        self.assertEqual(metadata['completed_level4_models'], [])
        self.assertIn('ID/порядок', metadata['invalid_models'][0]['reason'])

    def test_hash_valid_but_wrong_targets_rejected(self):
        self.frame.loc[0, 'TARGET'] = 1
        self.frame.to_parquet(self.stage/'probabilities.parquet', index=False)
        self.refresh_marker()
        metadata = export_decision_metrics(self.root, self.reports, self.prepared)
        self.assertIn('TARGET', metadata['invalid_models'][0]['reason'])

    def test_changed_global_fairness_file_fails_loudly(self):
        with (self.prepared/'fairness.parquet').open('ab') as target:
            target.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'fairness.parquet'):
            export_decision_metrics(self.root, self.reports, self.prepared)


if __name__ == '__main__':
    unittest.main()
