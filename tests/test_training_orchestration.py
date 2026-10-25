"""Небольшие синтетические Parquet: ID/CV, атомарные этапы и resume.

Полный эксперимент и настоящие подготовленные X не читаются и не обучаются.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import joblib
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from src.model_pipelines import make_estimator
from src.fit_diagnostics import extract_fit_diagnostics
from src.model_training import (PreparedData, _complete, fit_repeat, initialize_run,
                                make_context, search_pair)


def make_prepared_fixture(path):
    path=Path(path)
    (path/'splits').mkdir(parents=True,exist_ok=True)
    ids=np.arange(100001,100121,dtype=np.int64)
    target=pd.DataFrame({'SK_ID_CURR':ids,'TARGET':np.arange(120)%2})
    target[['SK_ID_CURR']].to_parquet(path/'row_ids.parquet',index=False)
    target.to_parquet(path/'target.parquet',index=False)
    frame=pd.DataFrame({'amount':np.arange(120,dtype=float),
                        'category':np.where(np.arange(120)%2==0,'A','B'),
                        'nullable':np.where(np.arange(120)%5==0,np.nan,np.arange(120)),
                        'empty':np.full(120,np.nan)})
    frame.loc[::9,'category']=None
    frame['bool_category']=pd.array([True,False,None]*40,dtype='boolean')
    features={'1':['amount','category'],
              '2':['amount','category','bool_category'],
              '3':['amount','category','bool_category','nullable'],
              '4':['amount','category','bool_category','nullable','empty']}
    (path/'feature_levels.json').write_text(json.dumps(features),encoding='utf-8')
    for level,names in features.items():
        frame[names].to_parquet(path/f'X_level_{level}.parquet',index=False)
        (path/f'features_level_{level}.json').write_text(json.dumps(names),encoding='utf-8')
    development=target.iloc[:96]
    test=target.iloc[96:]
    development[['SK_ID_CURR']].to_parquet(path/'splits/development_ids.parquet',index=False)
    test[['SK_ID_CURR']].to_parquet(path/'splits/test_ids.parquet',index=False)
    for seed in range(42,47):
        subset,_=train_test_split(development,train_size=86,
                                 stratify=development.TARGET,random_state=seed)
        subset[['SK_ID_CURR']].sort_values('SK_ID_CURR').to_parquet(
            path/f'splits/train_seed_{seed}_ids.parquet',index=False)
    folds=development[['SK_ID_CURR']].copy()
    folds['validation_fold']=np.repeat(np.arange(3),32)
    folds.to_parquet(path/'splits/cv_folds.parquet',index=False)
    return frame,target


class TrainingOrchestrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='rmt_training_contract_')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.base=Path(cls.temp.name)
        cls.prepared=cls.base/'prepared'
        cls.raw_frame,cls.raw_target=make_prepared_fixture(cls.prepared)

    def setUp(self):
        self.data=PreparedData(self.prepared)
        self.train_ids=self.data.repeats[42][:72]
        self.validation_ids=np.setdiff1d(self.data.development,self.data.repeats[42])[::-1]
        self.X,self.y=self.data.load(4,self.train_ids,purpose='technical_fixture_train')
        self.Xv,self.yv=self.data.load(4,self.validation_ids,purpose='technical_fixture_validation')
        self.context=make_context(self.data,mode='technical',threads=1,
                                  technical_train_n=len(self.X),technical_validation_n=len(self.Xv))
        self.root=initialize_run(self.base/self._testMethodName,self.context)

    def search(self):
        return search_pair(self.data,4,'logistic_regression',self.X,self.y,self.root,self.context)

    def fit(self,selected):
        return fit_repeat(self.data,4,'logistic_regression',42,self.X,self.y,
                          self.Xv,self.yv,self.root,self.context,selected)

    def test_loader_preserves_requested_id_and_target_order(self):
        requested=self.data.development[[75,2,33,1]]
        X,y=self.data.load(2,requested,purpose='technical_arbitrary_order')
        np.testing.assert_array_equal(X.index,requested)
        np.testing.assert_array_equal(y.index,requested)
        np.testing.assert_array_equal(X.amount.to_numpy(),requested-100001)
        np.testing.assert_array_equal(y.to_numpy(),(requested-100001)%2)
        self.assertEqual(X.columns.tolist(),self.data.features['2'])
        self.assertFalse(self.data.loaded_scopes[-1]['contains_test'])

    def test_arrow_boolean_is_a_categorical_feature(self):
        schema=pq.read_schema(self.prepared/'X_level_4.parquet')
        self.assertTrue(pa.types.is_boolean(schema.field('bool_category').type))
        numeric,categorical=self.data.columns(4)
        self.assertNotIn('bool_category',numeric)
        self.assertIn('bool_category',categorical)
        self.assertTrue(self.X.bool_category.isna().any())
        self.assertEqual(set(self.X.bool_category.dropna()),{True,False})

    def test_loader_forbids_test_in_technical_mode(self):
        with self.assertRaises(AssertionError):
            self.data.load(4,self.data.test[:3],purpose='technical_should_reject')
        with self.assertRaises(AssertionError):
            self.data.load(4,[self.data.development[0],self.data.test[0]],purpose='technical_mixed')

    def test_search_rejects_misaligned_or_incorrect_targets(self):
        with self.assertRaises(AssertionError):
            search_pair(self.data,4,'logistic_regression',self.X,self.y.iloc[::-1],
                        self.root,self.context)
        wrong=self.y.copy()
        wrong.iloc[0]=1-int(wrong.iloc[0])
        with self.assertRaises(AssertionError):
            search_pair(self.data,4,'logistic_regression',self.X,wrong,self.root,self.context)

    def test_repeat_rejects_wrong_train_or_evaluation_targets(self):
        selected={'params':{}}
        wrong_train=self.y.copy()
        wrong_train.iloc[0]=1-int(wrong_train.iloc[0])
        with self.assertRaises(AssertionError):
            fit_repeat(self.data,4,'logistic_regression',42,self.X,wrong_train,
                       self.Xv,self.yv,self.root,self.context,selected)
        wrong_validation=self.yv.copy()
        wrong_validation.iloc[0]=1-int(wrong_validation.iloc[0])
        with self.assertRaises(AssertionError):
            fit_repeat(self.data,4,'logistic_regression',42,self.X,self.y,
                       self.Xv,wrong_validation,self.root,self.context,selected)

    def test_saved_cv_is_mapped_by_id_even_for_reordered_frame(self):
        ids=self.data.development[::-1]
        folds=self.data.cv(ids)
        self.assertEqual(len(folds),3)
        for number,(train,validation) in enumerate(folds):
            expected=set(self.data.folds.index[self.data.folds==number])
            self.assertEqual(set(ids[validation]),expected)
            self.assertEqual(len(validation),32)
            self.assertEqual(set(ids[train]),set(ids)-expected)
            self.assertFalse(set(train)&set(validation))
        np.testing.assert_array_equal(np.sort(np.concatenate([v for _,v in folds])),np.arange(96))

    def test_completed_search_resumes_without_another_fit(self):
        selected,resumed=self.search()
        self.assertFalse(resumed)
        self.assertEqual(selected['scoring'],'neg_log_loss')
        self.assertEqual(selected['n_iter'],2)
        self.assertEqual(selected['folds'],3)
        self.assertFalse(selected['refit'])
        self.assertEqual(selected['fit_rows'],len(self.X))
        with patch.object(RandomizedSearchCV,'fit',side_effect=AssertionError('Resume must not fit')):
            loaded,resumed=self.search()
        self.assertTrue(resumed)
        self.assertEqual(loaded,selected)

    def test_completed_repeat_resumes_and_probabilities_retain_ids(self):
        selected,_=self.search()
        metrics,resumed=self.fit(selected)
        self.assertFalse(resumed)
        path=self.root/'level_4/logistic_regression/seed_42'
        probabilities=pd.read_parquet(path/'probabilities.parquet')
        np.testing.assert_array_equal(probabilities.SK_ID_CURR,self.validation_ids)
        np.testing.assert_array_equal(probabilities.TARGET,self.yv.to_numpy())
        self.assertTrue(probabilities.probability_default.between(0,1).all())
        self.assertEqual(metrics['evaluation_split'],'development_technical_validation')
        self.assertTrue(metrics['serialization_prediction_check'])
        with patch.object(Pipeline,'fit',side_effect=AssertionError('Resume must not fit')):
            loaded,resumed=self.fit(selected)
        self.assertTrue(resumed)
        self.assertEqual(loaded,metrics)

    def test_corrupt_completed_artifact_is_rejected(self):
        self.search()
        stage=self.root/'level_4/logistic_regression/search'
        (stage/'best_params.json').write_text('{"unexpected":"corruption"}',encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError,'Повреждён'):
            self.search()

    def test_mlp_repeat_persists_actual_diagnostics_warnings_and_reload(self):
        # Только синтетический тест: короткий fit гарантированно достигает лимита.
        selected={'params':{'model__max_iter':3,'model__tol':0.0,
                            'model__n_iter_no_change':100,'model__batch_size':12}}
        with patch.object(RandomizedSearchCV,'fit',side_effect=AssertionError('No search in repeat')):
            metrics,resumed=fit_repeat(self.data,4,'mlp',42,self.X,self.y,self.Xv,self.yv,
                                      self.root,self.context,selected)
        self.assertFalse(resumed)
        stage=self.root/'level_4/mlp/seed_42'
        saved_metrics=json.loads((stage/'metrics.json').read_text(encoding='utf-8'))
        diagnostics=json.loads((stage/'fit_diagnostics.json').read_text(encoding='utf-8'))
        complete=json.loads((stage/'complete.json').read_text(encoding='utf-8'))
        restored=joblib.load(stage/'pipeline.joblib')
        self.assertEqual(saved_metrics,metrics)
        self.assertEqual(saved_metrics['fit_diagnostics_file'],'fit_diagnostics.json')
        self.assertEqual(saved_metrics['actual_mlp_iterations'],[3])
        self.assertEqual(extract_fit_diagnostics(restored),diagnostics)
        self.assertEqual(len(diagnostics['mlp']),1)
        fitted=diagnostics['mlp'][0]['fitted']
        self.assertEqual(fitted['n_iter_'],3)
        self.assertEqual(len(fitted['loss_curve_']),fitted['n_iter_'])
        self.assertEqual(fitted['loss_curve_'],restored.named_steps['model'].loss_curve_)
        self.assertTrue(any(w['category']=='ConvergenceWarning' and w['count']>=1
                            for w in saved_metrics['warnings']))
        for filename in ['metrics.json','fit_diagnostics.json','pipeline.joblib']:
            self.assertEqual(complete['artifact_sha256'][filename],
                             hashlib.sha256((stage/filename).read_bytes()).hexdigest())
        self.assertTrue(_complete(stage,self.context))
        self.assertFalse(any(scope['contains_test'] for scope in self.data.loaded_scopes))

    def test_context_mismatch_is_rejected_for_run_and_stage(self):
        self.search()
        other=copy.deepcopy(self.context)
        other['fingerprint']='different_context_fingerprint'
        with self.assertRaisesRegex(RuntimeError,'Контекст'):
            initialize_run(self.root,other)
        with self.assertRaisesRegex(RuntimeError,'Несовместимый'):
            _complete(self.root/'level_4/logistic_regression/search',other)

    def test_unfinished_search_restarts_instead_of_trusting_partial_json(self):
        stage=self.root/'level_4/logistic_regression/search'
        stage.mkdir(parents=True)
        (stage/'best_params.json').write_text('{"incomplete":true}',encoding='utf-8')
        self.assertFalse(_complete(stage,self.context))
        selected,resumed=self.search()
        self.assertFalse(resumed)
        self.assertEqual(selected['n_iter'],2)
        self.assertTrue(_complete(stage,self.context))

    def test_xgboost_receives_explicit_dense_zero_while_lr_uses_csr(self):
        numeric,categorical=self.data.columns(4)
        frame=self.X.copy()
        frame.loc[frame.index[0],'amount']=0.0
        xgb=make_estimator('xgboost',numeric,categorical,technical=True,threads=1)
        dense=xgb.named_steps['preprocess'].fit_transform(frame)
        self.assertIsInstance(dense,np.ndarray)
        self.assertEqual(dense.dtype,np.float32)
        self.assertEqual(dense[0,0],0.0)
        self.assertTrue((dense==0).any())
        lr=make_estimator('logistic_regression',numeric,categorical,technical=True,threads=1)
        compact=lr.named_steps['preprocess'].fit_transform(self.X)
        self.assertTrue(sparse.isspmatrix_csr(compact))
        self.assertEqual(compact.dtype,np.float32)


if __name__=='__main__':
    unittest.main(verbosity=2)
