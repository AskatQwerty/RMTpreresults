"""Фиксированные решения и общий bootstrap проверяются без моделей/файлов данных."""
import unittest

import numpy as np
import pandas as pd

from src.pilot_evaluation_decisions import (DecisionEvaluator, decision_point_tables,
    decision_bootstrap, calibration_point_table, THRESHOLDS, SCENARIOS)


class PilotEvaluationDecisionsTests(unittest.TestCase):
    def setUp(self):
        self.y = np.array([0, 1, 1, 0, 0, 1])
        self.p = np.array([.1, .2, .6, .8, 0., 1.])
        self.probabilities = {f'model_{i}': self.p.copy() for i in range(7)}
        self.groups = {'gender': np.array(['F','F','M','M','unknown','M']),
                       'age': np.array(['<25','>=25','<25','>=25','unknown','<25'])}

    def test_unique_scenarios_shapes_and_actual_profit(self):
        tables = decision_point_tables(self.y, self.probabilities, self.groups)
        self.assertEqual(len(tables['profit']), 35)
        self.assertEqual(len(tables['baselines']), 6)
        self.assertEqual(len(tables['fairness_groups']), 7*3*2*3)
        self.assertEqual(len(tables['fairness_gaps']), 21)
        self.assertEqual(len(tables['threshold_metrics']), 21)
        self.assertEqual(len(tables['profit'].query('loss_to_margin == 1')), 7)
        row = tables['profit'].query("model == 'model_0' and scenario == 'L5_t0.5'").iloc[0]
        self.assertAlmostEqual(row.realized_profit_per_applicant, (1-5+1)/6)
        self.assertEqual(row.approved_n, 3)
        self.assertAlmostEqual(row.default_rate_among_approved, 1/3)
        baseline = tables['baselines'].query("policy == 'approve_all' and loss_to_margin == 5").iloc[0]
        self.assertEqual(baseline.realized_profit_per_applicant, (3-5*3)/6)
        self.assertEqual(baseline.approved_n, 6)
        self.assertAlmostEqual(baseline.default_rate_among_approved, .5)
        reject = tables['baselines'].query("policy == 'reject_all'")
        self.assertTrue((reject.realized_profit_per_applicant == 0).all())
        self.assertTrue(reject.default_rate_among_approved.isna().all())

    def test_equality_at_every_threshold_rejects(self):
        for threshold in THRESHOLDS:
            evaluator = DecisionEvaluator([0, 1], {'model': np.array([threshold, threshold])},
                {'gender': ['F','M'], 'age': ['<25','>=25']})
            index = list(THRESHOLDS).index(threshold)
            values = evaluator.weighted([1, 1])
            self.assertEqual(values['approved_n'][0, index], 0)
            np.testing.assert_array_equal(values['confusion'][0, index], [0, 1, 0, 1])
            self.assertTrue(np.isnan(values['default_rate_among_approved'][0, index]))

    def test_tpr_is_positive_default_detection_and_unknown_not_dropped(self):
        tables = decision_point_tables(self.y, self.probabilities, self.groups)
        rows = tables['fairness_groups'].query("model == 'model_0' and threshold == 0.5 and attribute == 'gender'")
        female = rows[rows.group == 'F'].iloc[0]
        male = rows[rows.group == 'M'].iloc[0]
        unknown = rows[rows.group == 'unknown'].iloc[0]
        self.assertEqual(female.default_n, 1)
        self.assertEqual(female.TPR, 0)
        self.assertEqual(male.default_n, 2)
        self.assertEqual(male.TP, 2)
        self.assertEqual(male.TPR, 1)
        self.assertEqual(unknown.n, 1)
        self.assertEqual(unknown.approved_n, 1)
        self.assertTrue(np.isnan(unknown.TPR))
        self.assertFalse(unknown.included_in_binary_gap)
        self.assertEqual(rows.n.sum(), len(self.y))
        gap = tables['fairness_gaps'].query("model == 'model_0' and threshold == 0.5").iloc[0]
        self.assertEqual(gap.gender_approval_rate_gap, 1)
        self.assertEqual(gap.gender_tpr_gap, 1)
        self.assertFalse(gap.gender_approval_gap_le_0_05)
        self.assertFalse(any('tpr' in c and '0_05' in c for c in tables['fairness_gaps']))

    def test_weighted_bootstrap_matches_explicit_same_client_replication(self):
        weights = np.array([2, 0, 3, 1, 0, 2])
        evaluator = DecisionEvaluator(self.y, self.probabilities, self.groups)
        actual = evaluator.weighted(weights)
        selected = np.repeat(np.arange(len(self.y)), weights)
        copied = DecisionEvaluator(self.y[selected], {m: p[selected] for m, p in self.probabilities.items()},
            {name: group[selected] for name, group in self.groups.items()})
        expected = copied.weighted(np.ones(len(selected)))
        self.assertEqual(set(actual), set(expected))
        for name in actual:
            np.testing.assert_allclose(actual[name], expected[name], equal_nan=True, err_msg=name)
        self.assertEqual(actual['profit'].shape, (7, 5))
        self.assertEqual(actual['approval_gap'].shape, (7, 3, 2))
        self.assertEqual(actual['group_tpr'].shape, (7, 3, 2, 3))
        self.assertEqual(evaluator.coordinates['confusion_order'], ['TN','FP','FN','TP'])
        self.assertEqual(len(evaluator.coordinates['scenarios']), len(SCENARIOS))

    def test_missing_denominators_nan_and_nullable_gap_flags(self):
        tables = decision_point_tables([0, 0], {'m': np.array([0., 0.])},
            {'gender': ['F','unknown'], 'age': ['<25','unknown']})
        threshold = tables['threshold_metrics'].iloc[0]
        self.assertTrue(np.isnan(threshold.precision))
        self.assertTrue(np.isnan(threshold.recall))
        self.assertTrue(np.isnan(threshold.f1))
        self.assertTrue(tables['fairness_gaps'].gender_approval_gap_le_0_05.isna().all())
        self.assertTrue(tables['fairness_gaps'].both_approval_gaps_le_0_05.isna().all())

    def test_fixed_calibration_edges_match_old_definition(self):
        table = calibration_point_table([0, 1, 0, 1, 1], {'m': np.array([0., .1, .3, .9, 1.])})
        self.assertEqual(len(table), 10)
        self.assertEqual(table.n.sum(), 5)
        self.assertEqual(table.loc[table.bin==3, 'n'].iloc[0], 1)
        self.assertEqual(table.loc[table.bin==9, 'n'].iloc[0], 2)
        self.assertTrue(pd.isna(table.loc[table.bin==2, 'observed_default_rate'].iloc[0]))

    def test_reject_invalid_weights_and_data(self):
        evaluator = DecisionEvaluator(self.y, self.probabilities, self.groups)
        for bad in [np.zeros(6), np.ones(5), [-1,1,1,1,1,1], [np.nan,1,1,1,1,1]]:
            with self.assertRaises(ValueError):
                evaluator.weighted(bad)
        with self.assertRaises(ValueError):
            decision_bootstrap([2], {'m': np.array([.5])}, {'gender': ['F'], 'age': ['<25']}, [1])


if __name__ == '__main__':
    unittest.main()
