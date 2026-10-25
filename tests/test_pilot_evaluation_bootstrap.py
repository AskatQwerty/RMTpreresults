"""Saved-probability bootstrap: independent duplicated-row metrics and resume."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from src.pilot_evaluation_bootstrap import QualityEvaluator, bootstrap_run, percentile_interval, sha256
from src.pilot_evaluation_decisions import DecisionEvaluator


class PilotEvaluationBootstrapTests(unittest.TestCase):
    def setUp(self):
        parent=Path(__file__).resolve().parents[1]/'tmp/test_temp'
        parent.mkdir(parents=True,exist_ok=True)
        self.temporary=tempfile.TemporaryDirectory(prefix='rmt_evaluation_bootstrap_',dir=parent)
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.y=np.array([0,1,1,0,1,0,0,1,0,1,0,1],dtype=np.int8)
        names=('logistic_regression','lightgbm','xgboost','soft_voting','stacking','mlp','random_forest')
        self.pairs=[{'model':name,'level':level} for name in names for level in (1,2,3,4)]
        base=np.array([0.,1.,.2,.2,.5,.5,.7,.1,.9,0.,1.,.4])
        self.p=np.stack([np.roll(base,index%len(base)) for index in range(28)])
        self.p[1]=self.p[0]  # Two identical pairs must have exactly zero paired delta.
        self.level4=[index for index,pair in enumerate(self.pairs) if pair['level']==4]
        self.groups={'gender':np.array(['F','M','unknown']*4),
                     'age':np.array(['<25','>=25','unknown']*4)}
        self.fingerprint='synthetic_28_pairs_same_clients'

    def run_bootstrap(self,directory,**kwargs):
        options={'n_bootstrap':4,'seed':42}
        options.update(kwargs)
        with redirect_stdout(io.StringIO()), \
             patch.object(Pipeline,'fit',side_effect=AssertionError('No model fitting')), \
             patch.object(Pipeline,'predict',side_effect=AssertionError('No model prediction')), \
             patch.object(Pipeline,'predict_proba',side_effect=AssertionError('Use saved probabilities')):
            return bootstrap_run(self.y,self.p,self.pairs,self.groups,directory,
                                 fingerprint=self.fingerprint,**options)

    def test_weighted_quality_matches_explicit_duplicate_rows_with_ties_and_zero_one(self):
        y_before=self.y.copy()
        p_before=self.p.copy()
        sampled=np.array([0,0,1,2,2,2,4,5,8,9,10,11])
        weights=np.bincount(sampled,minlength=len(self.y))
        weights_before=weights.copy()
        actual=QualityEvaluator(self.y,self.p,self.level4).weighted(weights)
        expected_ap=[average_precision_score(self.y[sampled],row[sampled]) for row in self.p]
        expected_auc=[roc_auc_score(self.y[sampled],self.p[index,sampled]) for index in self.level4]
        expected_log_loss=[log_loss(self.y[sampled],self.p[index,sampled],labels=[0,1])
                           for index in self.level4]
        expected_brier=[brier_score_loss(self.y[sampled],self.p[index,sampled],pos_label=1)
                        for index in self.level4]
        for name,expected in [('ap',expected_ap),('roc_auc_l4',expected_auc),
                              ('log_loss_l4',expected_log_loss),('brier_l4',expected_brier)]:
            np.testing.assert_allclose(actual[name],expected,rtol=1e-13,atol=1e-14,err_msg=name)
        self.assertEqual(actual['ap'].shape,(28,))
        self.assertEqual(actual['roc_auc_l4'].shape,(7,))
        np.testing.assert_array_equal(self.y,y_before)
        np.testing.assert_array_equal(self.p,p_before)
        np.testing.assert_array_equal(weights,weights_before)

    def test_pairing_shared_for_all_metrics_decisions_and_delta_intervals(self):
        quality_calls=[]
        decision_calls=[]
        quality_original=QualityEvaluator.weighted
        decision_original=DecisionEvaluator.weighted
        def quality_spy(evaluator,weights):
            quality_calls.append(np.asarray(weights).copy())
            return quality_original(evaluator,weights)
        def decision_spy(evaluator,weights):
            decision_calls.append(np.asarray(weights).copy())
            return decision_original(evaluator,weights)
        y_before=self.y.copy()
        p_before=self.p.copy()
        pairs_before=json.dumps(self.pairs,sort_keys=True)
        groups_before={name:values.copy() for name,values in self.groups.items()}
        with patch.object(QualityEvaluator,'weighted',new=quality_spy), \
             patch.object(DecisionEvaluator,'weighted',new=decision_spy):
            values,state=self.run_bootstrap(self.root/'paired')
        self.assertEqual(len(quality_calls),4)
        self.assertEqual(len(decision_calls),4)
        explicit_delta=[]
        for replicate,(quality_counts,decision_counts) in enumerate(zip(quality_calls,decision_calls)):
            np.testing.assert_array_equal(quality_counts,decision_counts)
            sampled=np.repeat(np.arange(len(self.y)),quality_counts.astype(int))
            expected_ap=np.array([average_precision_score(self.y[sampled],row[sampled]) for row in self.p])
            np.testing.assert_allclose(values['ap'][replicate],expected_ap,atol=1e-14)
            explicit_delta.append(expected_ap[2]-expected_ap[0])
        expected_limits=np.quantile(explicit_delta,[.025,.975],method='linear')
        low,high,valid=percentile_interval(values['ap'][:,2]-values['ap'][:,0])
        np.testing.assert_allclose([low,high],expected_limits,atol=1e-14)
        self.assertEqual(valid,4)
        self.assertEqual(percentile_interval(values['ap'][:,1]-values['ap'][:,0]),(0.0,0.0,4))
        self.assertEqual(values['decision_profit'].shape,(4,7,5))
        self.assertTrue(state['paired'] and state['same_counts_for_all_pairs'])
        self.assertFalse(state['stratified'])
        self.assertEqual(state['fit_calls'],0)
        self.assertEqual(state['predict_calls'],0)
        np.testing.assert_array_equal(self.y,y_before)
        np.testing.assert_array_equal(self.p,p_before)
        self.assertEqual(json.dumps(self.pairs,sort_keys=True),pairs_before)
        for name,before in groups_before.items():
            np.testing.assert_array_equal(self.groups[name],before)

    def test_resume_two_then_four_matches_uninterrupted_npz_rng_chain(self):
        directory=self.root/'resumed'
        partial,first_state=self.run_bootstrap(directory,max_new_replicates=2)
        self.assertEqual((first_state['completed'],first_state['new'],first_state['reused']),(2,2,0))
        self.assertEqual(partial['ap'].shape,(2,28))
        initial_hashes={path.name:sha256(path) for path in directory.glob('replicate_*')}
        resumed,state=self.run_bootstrap(directory)
        self.assertEqual((state['completed'],state['new'],state['reused']),(4,2,2))
        for name,digest in initial_hashes.items():
            self.assertEqual(sha256(directory/name),digest)
        uninterrupted,whole_state=self.run_bootstrap(self.root/'whole')
        self.assertEqual(whole_state['new'],4)
        for name in resumed:
            np.testing.assert_allclose(resumed[name],uninterrupted[name],equal_nan=True,err_msg=name)
        rng=np.random.default_rng(42)
        for replicate in range(4):
            before=json.dumps(rng.bit_generator.state,sort_keys=True)
            positions=rng.integers(0,len(self.y),size=len(self.y),dtype=np.int64)
            expected_counts=np.bincount(positions,minlength=len(self.y))
            after=json.dumps(rng.bit_generator.state,sort_keys=True)
            with np.load(directory/f'replicate_{replicate:04d}.npz',allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['weights'],expected_counts)
                self.assertEqual(str(saved['rng_before']),before)
                self.assertEqual(str(saved['rng_after']),after)
                self.assertEqual(int(saved['replicate']),replicate)
                self.assertEqual(str(saved['fingerprint']),self.fingerprint)
        progress=json.loads((directory/'progress.json').read_text(encoding='utf-8'))
        self.assertEqual(progress['completed'],4)

    def test_corrupt_completed_npz_is_rejected_before_recomputation(self):
        directory=self.root/'corrupt'
        self.run_bootstrap(directory,n_bootstrap=2)
        path=directory/'replicate_0000.npz'
        with path.open('ab') as stream:
            stream.write(b'corrupted checkpoint bytes')
        with patch.object(QualityEvaluator,'weighted',side_effect=AssertionError('Do not recompute corruption')):
            with self.assertRaisesRegex(ValueError,'Corrupt completed checkpoint'):
                self.run_bootstrap(directory,n_bootstrap=2)

    def test_different_seed_or_fingerprint_rejects_saved_rng_chain(self):
        directory=self.root/'wrong_context'
        self.run_bootstrap(directory,n_bootstrap=2)
        with self.assertRaisesRegex(ValueError,'RNG chain'):
            self.run_bootstrap(directory,n_bootstrap=2,seed=43)
        self.fingerprint='different_analysis'
        with self.assertRaisesRegex(ValueError,'another analysis'):
            self.run_bootstrap(directory,n_bootstrap=2)

    def test_missing_marker_recovers_complete_npz_without_weighted_calls(self):
        directory=self.root/'recover_marker'
        original,state=self.run_bootstrap(directory)
        self.assertEqual(state['completed'],4)
        path=directory/'replicate_0001.npz'
        original_hash=sha256(path)
        marker=path.with_suffix('.json')
        marker.unlink()
        with patch.object(QualityEvaluator,'weighted',side_effect=AssertionError('Recovery must not recompute quality')), \
             patch.object(DecisionEvaluator,'weighted',side_effect=AssertionError('Recovery must not recompute decisions')):
            recovered,recovered_state=self.run_bootstrap(directory)
        self.assertEqual((recovered_state['completed'],recovered_state['new'],recovered_state['reused']),(4,0,4))
        for name in original:
            np.testing.assert_allclose(recovered[name],original[name],equal_nan=True,err_msg=name)
        metadata=json.loads(marker.read_text(encoding='utf-8'))
        self.assertTrue(metadata['recovered_without_recomputation'])
        self.assertEqual(metadata['sha256'],original_hash)
        self.assertEqual(sha256(path),original_hash)

    def test_one_class_draw_has_undefined_ap_auc_but_finite_proper_losses(self):
        evaluator=QualityEvaluator(self.y,self.p,self.level4)
        for label in (0,1):
            with self.subTest(label=label):
                weights=(self.y==label).astype(int)
                values=evaluator.weighted(weights)
                self.assertTrue(np.isnan(values['ap']).all())
                self.assertTrue(np.isnan(values['roc_auc_l4']).all())
                self.assertTrue(np.isfinite(values['log_loss_l4']).all())
                self.assertTrue(np.isfinite(values['brier_l4']).all())
                low,high,valid=percentile_interval(values['ap'])
                self.assertTrue(np.isnan(low) and np.isnan(high))
                self.assertEqual(valid,0)
                self.assertFalse(any('p_value' in name or 'significant' in name for name in values))

    def test_percentile_interval_reports_only_finite_replicates(self):
        low,high,valid=percentile_interval([np.nan,1.,2.,3.,np.inf])
        np.testing.assert_allclose([low,high],np.quantile([1.,2.,3.],[.025,.975],method='linear'))
        self.assertEqual(valid,3)

    def test_invalid_weights_rejected(self):
        evaluator=QualityEvaluator(self.y,self.p,self.level4)
        for weights in [np.zeros(12),np.ones(11),np.r_[-1,np.ones(11)],
                        np.r_[np.nan,np.ones(11)],np.r_[np.inf,np.ones(11)]]:
            with self.subTest(weights=weights),self.assertRaises(ValueError):
                evaluator.weighted(weights)


if __name__=='__main__':
    unittest.main(verbosity=2)
