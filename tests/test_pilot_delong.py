"""DeLong: независимые O(mn) placements проверяют fast ranks и paired inference."""
from __future__ import annotations

import json
import math
import unittest

import numpy as np
from sklearn.metrics import roc_auc_score

from src.pilot_delong import delong_auc_covariance, paired_delong_table


def brute_force(y,scores):
    """Прямые сравнения каждой positive/negative пары, без ранжирования."""
    labels=np.asarray(y)
    matrix=np.atleast_2d(scores)
    m=int((labels==1).sum())
    n=int((labels==0).sum())
    positive=[]
    negative=[]
    auc=[]
    for values in matrix:
        comparison=np.zeros((m,n),dtype=float)
        for i,left in enumerate(values[labels==1]):
            for j,right in enumerate(values[labels==0]):
                comparison[i,j]=1.0 if left>right else 0.5 if left==right else 0.0
        positive.append(comparison.mean(axis=1))
        negative.append(comparison.mean(axis=0))
        auc.append(comparison.mean())
    covariance=np.atleast_2d(np.cov(positive,ddof=1))/m+np.atleast_2d(np.cov(negative,ddof=1))/n
    return np.asarray(auc),covariance


class PilotDeLongTests(unittest.TestCase):
    def setUp(self):
        self.y=np.array([0,1,0,1,1,0,0,1])
        self.reference=np.array([.1,.2,.3,.3,.8,.4,.6,.9])

    def seven_models(self,other=None):
        values=self.reference if other is None else other
        return {'logistic_regression':self.reference.copy(),
                **{f'model_{i}':np.array(values,copy=True) for i in range(6)}}

    def test_ties_match_independent_brute_force_auc_and_covariance(self):
        scores=np.array([self.reference,[.1,.4,.4,.4,.9,.2,.7,.8],
                         [.5,.5,.5,.5,.5,.5,.5,.5]])
        auc,covariance=delong_auc_covariance(self.y,scores)
        expected_auc,expected_covariance=brute_force(self.y,scores)
        np.testing.assert_allclose(auc,expected_auc,rtol=0,atol=1e-15)
        np.testing.assert_allclose(covariance,expected_covariance,rtol=0,atol=1e-15)
        np.testing.assert_allclose(auc,[roc_auc_score(self.y,row) for row in scores],atol=1e-15)
        self.assertNotEqual(covariance[0,1],0)
        self.assertEqual(auc[2],.5)

    def test_small_random_tied_examples_match_brute_force(self):
        random=np.random.default_rng(2026)
        for positives,negatives in [(2,3),(3,7),(5,4),(7,9)]:
            with self.subTest(positives=positives,negatives=negatives):
                y=random.permutation(np.r_[np.ones(positives),np.zeros(negatives)])
                scores=random.integers(0,4,size=(4,len(y)))
                auc,covariance=delong_auc_covariance(y,scores)
                expected_auc,expected_covariance=brute_force(y,scores)
                np.testing.assert_allclose(auc,expected_auc,rtol=0,atol=1e-15)
                np.testing.assert_allclose(covariance,expected_covariance,rtol=0,atol=1e-15)

    def test_identical_predictions_have_zero_difference_and_p_one(self):
        table=paired_delong_table(self.y,self.seven_models())
        self.assertEqual(len(table),6)
        for name in ('delta_roc_auc','se','z'):
            np.testing.assert_array_equal(table[name],np.zeros(6))
        np.testing.assert_array_equal(table.p_raw,np.ones(6))
        np.testing.assert_array_equal(table.p_bonferroni,np.ones(6))
        self.assertFalse(table.reject_bonferroni_0_05.any())
        self.assertTrue(table.inference_status.eq('zero_difference_zero_variance').all())

    def test_negated_ranks_preserve_paired_covariance_and_direction(self):
        auc,covariance=delong_auc_covariance(self.y,np.vstack([self.reference,-self.reference]))
        self.assertAlmostEqual(auc.sum(),1.0)
        self.assertAlmostEqual(covariance[0,0],covariance[1,1])
        self.assertAlmostEqual(covariance[0,1],-covariance[0,0])
        table=paired_delong_table(self.y,self.seven_models(1-self.reference))
        expected_auc,expected_covariance=brute_force(self.y,np.vstack([self.reference,1-self.reference]))
        delta=expected_auc[1]-expected_auc[0]
        se=np.sqrt(expected_covariance[0,0]+expected_covariance[1,1]-2*expected_covariance[0,1])
        self.assertLess(delta,0)
        np.testing.assert_allclose(table.delta_roc_auc,delta,atol=1e-15)
        np.testing.assert_allclose(table.se,se,atol=1e-15)
        np.testing.assert_allclose(table.z,delta/se,atol=1e-15)
        np.testing.assert_allclose(table.p_raw,math.erfc(abs(delta/se)/math.sqrt(2)),atol=1e-15)

    def test_bonferroni_family_fixed_at_six_and_metadata_serializable(self):
        table=paired_delong_table(self.y,self.seven_models(1-self.reference))
        np.testing.assert_allclose(table.p_bonferroni,np.minimum(1,6*table.p_raw))
        np.testing.assert_array_equal(table.reject_bonferroni_0_05,table.p_bonferroni<=.05)
        metadata=table.attrs['methodology']
        self.assertEqual(metadata['bonferroni_family_size'],6)
        self.assertEqual(metadata['n_positive'],4)
        self.assertEqual(metadata['n_negative'],4)
        self.assertFalse(metadata['ap_p_values_computed'])
        self.assertTrue(metadata['not_equivalence_test'])
        self.assertEqual(len(metadata['references']),2)
        json.dumps(metadata,allow_nan=False)

    def test_joint_row_permutation_and_monotone_transform_do_not_change_covariance(self):
        scores=np.vstack([self.reference,1-self.reference])
        original_scores=scores.copy()
        expected_auc,expected_covariance=delong_auc_covariance(self.y,scores)
        order=np.array([5,0,6,3,7,1,4,2])
        auc,covariance=delong_auc_covariance(self.y[order],np.exp(scores[:,order]))
        np.testing.assert_allclose(auc,expected_auc,atol=1e-15)
        np.testing.assert_allclose(covariance,expected_covariance,atol=1e-15)
        np.testing.assert_array_equal(scores,original_scores)

    def test_degenerate_nonzero_difference_is_marked_undefined(self):
        y=np.array([0,1,0,1,0,1])
        values={'logistic_regression':y.astype(float),
                **{f'model_{i}':1.0-y for i in range(6)}}
        table=paired_delong_table(y,values)
        self.assertTrue(table.p_raw.isna().all())
        self.assertTrue(table.p_bonferroni.isna().all())
        self.assertTrue(table.z.isna().all())
        self.assertFalse(table.reject_bonferroni_0_05.any())
        self.assertTrue(table.inference_status.eq('undefined_zero_variance_nonzero_difference').all())

    def test_invalid_labels_shapes_missing_values_and_family_are_rejected(self):
        for y,scores in [([0,0,0,0],np.zeros(4)),([0,0,0,1],np.zeros(4)),
                         ([0,1,2,1],np.zeros(4)),([0,1,0,np.nan],np.zeros(4)),
                         ([0,1,0,1],np.zeros((2,3))),([0,1,0,1],[0,1,np.inf,1])]:
            with self.subTest(y=y,scores=scores),self.assertRaises(ValueError):
                delong_auc_covariance(y,scores)
        models=self.seven_models()
        del models['model_0']
        with self.assertRaisesRegex(ValueError,'seven models'):
            paired_delong_table(self.y,models)
        with self.assertRaisesRegex(ValueError,'Reference model'):
            paired_delong_table(self.y,self.seven_models(),reference='absent')
        for invalid in [np.full(8,np.nan),np.full(8,1.01),np.zeros(7),np.zeros((8,1))]:
            models=self.seven_models()
            models['model_0']=invalid
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):
                paired_delong_table(self.y,models)


if __name__=='__main__':
    unittest.main(verbosity=2)
