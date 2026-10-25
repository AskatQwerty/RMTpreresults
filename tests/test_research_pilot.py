"""Исследовательский пилот: синтетические ID/CV, бюджет fit и продолжение."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.model_training import PreparedData
from src.research_pilot import initialize_pilot, load_pilot_split, run_pair
from test_training_orchestration import make_prepared_fixture


class ResearchPilotTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='rmt_research_pilot_')
        self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name)
        self.prepared=self.base/'prepared'
        make_prepared_fixture(self.prepared)
        self.data=PreparedData(self.prepared)
        self.root=self.base/'pilot'
        self.context=initialize_pilot(self.prepared,self.root,threads=1,sample_n=60)
        self.ids,self.cv=load_pilot_split(self.data,self.root)
        self.stage=self.root/'logistic_regression/level_4'

    def fit_spy(self,records):
        original=LogisticRegression.fit
        def fit(estimator,X,y,*args,**kwargs):
            self.assertIsInstance(y,pd.Series)
            ids=np.asarray(y.index,dtype=np.int64)
            self.assertEqual(len(ids),X.shape[0])
            self.assertTrue(set(ids)<=set(self.ids))
            self.assertTrue(set(ids).isdisjoint(self.data.test))
            np.testing.assert_array_equal(y.to_numpy(),self.data.target.loc[ids].to_numpy())
            records.append({'ids':tuple(sorted(ids.tolist())),
                            'rows':len(y),'max_iter':estimator.max_iter})
            return original(estimator,X,y,*args,**kwargs)
        return fit

    def assert_full_lr_search_budget(self,records):
        self.assertEqual(len(records),16)
        self.assertEqual(Counter(record['rows'] for record in records),{40:15,60:1})
        self.assertTrue(all(record['max_iter']==2000 for record in records))
        expected=Counter({tuple(sorted(self.ids[train].tolist())):5 for train,_ in self.cv})
        actual=Counter(record['ids'] for record in records if record['rows']==40)
        self.assertEqual(actual,expected)
        self.assertEqual([record['ids'] for record in records if record['rows']==60],
                         [tuple(sorted(self.ids.tolist()))])

    def test_cohort_and_stratified_cv_are_shared_and_resume_without_recreation(self):
        self.assertEqual(len(self.ids),60)
        self.assertEqual(len(set(self.ids)),60)
        self.assertTrue(set(self.ids)<=set(self.data.development))
        self.assertTrue(set(self.ids).isdisjoint(self.data.test))
        self.assertEqual(int(self.data.target.loc[self.ids].sum()),30)
        self.assertEqual(len(self.cv),3)
        for train,validation in self.cv:
            self.assertEqual(len(train),40)
            self.assertEqual(len(validation),20)
            self.assertFalse(set(train)&set(validation))
            self.assertEqual(int(self.data.target.loc[self.ids[validation]].sum()),10)
        np.testing.assert_array_equal(np.sort(np.concatenate([v for _,v in self.cv])),np.arange(60))
        saved_folds=pd.read_parquet(self.root/'cv_folds.parquet').set_index('SK_ID_CURR')
        self.assertFalse(np.array_equal(saved_folds.loc[self.ids,'validation_fold'].to_numpy(),
                                        self.data.folds.loc[self.ids].to_numpy()))
        for level in (1,2,3,4):
            X,y=self.data.load(level,self.ids,purpose='technical_research_fixture')
            np.testing.assert_array_equal(X.index,self.ids)
            np.testing.assert_array_equal(y.index,self.ids)
            self.assertEqual(X.columns.tolist(),self.data.features[str(level)])
        paths=[self.root/name for name in ('cohort_ids.parquet','cv_folds.parquet',
                                           'cohort_targets.parquet','test_ids.parquet')]
        before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        with patch('src.research_pilot.train_test_split',side_effect=AssertionError('No new cohort')), \
             patch('sklearn.model_selection.StratifiedKFold.split',side_effect=AssertionError('No new folds')):
            resumed_context=initialize_pilot(self.prepared,self.root,threads=1,sample_n=60)
            resumed_ids,resumed_cv=load_pilot_split(self.data,self.root)
        self.assertEqual(resumed_context['fingerprint'],self.context['fingerprint'])
        self.assertEqual(before,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
        np.testing.assert_array_equal(resumed_ids,self.ids)
        for (train,validation),(resumed_train,resumed_validation) in zip(self.cv,resumed_cv):
            np.testing.assert_array_equal(train,resumed_train)
            np.testing.assert_array_equal(validation,resumed_validation)

    def test_search_fits_exactly_fifteen_candidates_and_one_refit_then_resumes(self):
        records=[]
        with patch.object(LogisticRegression,'fit',new=self.fit_spy(records)):
            metrics=run_pair(self.prepared,self.root,'logistic_regression',4)
        self.assert_full_lr_search_budget(records)
        self.assertTrue((self.stage/'fit_checkpoint/complete.json').is_file())
        self.assertTrue((self.stage/'complete.json').is_file())
        saved=json.loads((self.stage/'metrics.json').read_text(encoding='utf-8'))
        self.assertEqual(saved,metrics)
        with patch.object(LogisticRegression,'fit',side_effect=AssertionError('Completed pair must not refit')):
            resumed=run_pair(self.prepared,self.root,'logistic_regression',4)
        self.assertEqual(resumed,metrics)
        # Другая пара меняет признаки, но использует тот же cohort и те же CV ID.
        level_one_records=[]
        with patch.object(LogisticRegression,'fit',new=self.fit_spy(level_one_records)):
            run_pair(self.prepared,self.root,'logistic_regression',1)
        self.assert_full_lr_search_budget(level_one_records)
        self.assertEqual(Counter(record['ids'] for record in level_one_records),
                         Counter(record['ids'] for record in records))

    def test_evaluation_failure_resumes_fitted_checkpoint_without_any_refit(self):
        records=[]
        with patch.object(LogisticRegression,'fit',new=self.fit_spy(records)), \
             patch('src.research_pilot.positive_probability',side_effect=RuntimeError('Synthetic evaluation failure')):
            with self.assertRaisesRegex(RuntimeError,'Synthetic evaluation failure'):
                run_pair(self.prepared,self.root,'logistic_regression',4)
        self.assert_full_lr_search_budget(records)
        checkpoint=self.stage/'fit_checkpoint'
        self.assertTrue((checkpoint/'complete.json').is_file())
        self.assertTrue((checkpoint/'pipeline.joblib').is_file())
        self.assertFalse((self.stage/'complete.json').exists())
        self.assertTrue((self.stage/'failure.json').is_file())
        model_hash=hashlib.sha256((checkpoint/'pipeline.joblib').read_bytes()).hexdigest()
        with patch.object(LogisticRegression,'fit',side_effect=AssertionError('Checkpoint must not refit')):
            metrics=run_pair(self.prepared,self.root,'logistic_regression',4)
        self.assertEqual(model_hash,hashlib.sha256((checkpoint/'pipeline.joblib').read_bytes()).hexdigest())
        self.assertTrue((self.stage/'complete.json').is_file())
        self.assertEqual(metrics,json.loads((self.stage/'metrics.json').read_text(encoding='utf-8')))

    def test_test_id_contamination_in_cohort_is_rejected(self):
        path=self.root/'cohort_ids.parquet'
        cohort=pd.read_parquet(path)
        cohort.loc[0,'SK_ID_CURR']=self.data.test[0]
        cohort.to_parquet(path,index=False)
        with self.assertRaises((AssertionError,ValueError,RuntimeError)):
            load_pilot_split(self.data,self.root)

    def test_corrupt_context_fingerprint_is_rejected(self):
        path=self.root/'experiment_manifest.json'
        context=json.loads(path.read_text(encoding='utf-8'))
        context['fingerprint']='corrupted_fingerprint'
        path.write_text(json.dumps(context),encoding='utf-8')
        with self.assertRaises((AssertionError,ValueError,RuntimeError)):
            initialize_pilot(self.prepared,self.root,threads=1,sample_n=60)


if __name__=='__main__':
    unittest.main(verbosity=2)
