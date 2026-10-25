"""Проверки сохранения actual MLP-диагностики на искусственных данных."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from src.fit_diagnostics import extract_fit_diagnostics
from src.model_pipelines import make_estimator


class FitDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X = pd.DataFrame({'number': np.arange(36, dtype=float),
                             'category': ['A','B',None]*12,
                             'boolean': pd.array([True,False,None]*12,dtype='boolean')})
        cls.y = pd.Series(np.arange(36)%2)

    def test_unfitted_rejected(self):
        with self.assertRaises(NotFittedError):
            extract_fit_diagnostics(make_estimator('mlp',['number'],['category','boolean']))

    def test_full_mlp_settings_unchanged(self):
        model = make_estimator('mlp',['number'],['category','boolean'],technical=False).named_steps['model']
        self.assertEqual(model.max_iter,300)
        self.assertEqual(model.tol,1e-4)
        self.assertEqual(model.n_iter_no_change,20)
        self.assertEqual(model.hidden_layer_sizes,(64,32))
        self.assertFalse(model.early_stopping)

    def test_mlp_actual_attributes_survive_joblib_and_json(self):
        pipeline = make_estimator('mlp',['number'],['category','boolean'],technical=True,threads=1)
        pipeline.set_params(model__max_iter=5,model__n_iter_no_change=100,
                            model__tol=0.0,model__hidden_layer_sizes=(4,),model__batch_size=12)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            pipeline.fit(self.X,self.y)
        with patch.object(pipeline.named_steps['preprocess'],'transform',side_effect=AssertionError('No transform allowed')):
            report = extract_fit_diagnostics(pipeline)
        row = report['mlp'][0]
        self.assertEqual(row['fitted']['n_iter_'],5)
        self.assertEqual(len(row['fitted']['loss_curve_']),5)
        self.assertEqual(row['fitted']['loss_curve_'][-1],row['fitted']['loss_'])
        self.assertEqual(row['checks']['loss_curve_length_matches_n_iter'],True)
        self.assertEqual(row['preprocessing']['output_features'],row['model_input_features'])
        self.assertEqual(row['preprocessing']['representation'],'CSR_float32')
        self.assertTrue(any(w.category.__name__=='ConvergenceWarning' for w in caught))
        self.assertEqual(json.loads(json.dumps(report,allow_nan=False)),report)
        with tempfile.TemporaryDirectory(prefix='rmt_fit_diagnostics_') as temp:
            path=Path(temp)/'pipeline.joblib'
            joblib.dump(pipeline,path)
            restored=joblib.load(path)
            self.assertEqual(extract_fit_diagnostics(restored),report)

    def test_stacking_records_fitted_bases_and_meta_dimensions(self):
        pipeline = make_estimator('stacking',['number'],['category','boolean'],technical=True,threads=1)
        pipeline.fit(self.X,self.y)
        report=extract_fit_diagnostics(pipeline)
        self.assertEqual(len(report['estimators']),4)
        bases=[r for r in report['estimators'] if r['role']=='base']
        meta=[r for r in report['estimators'] if r['role']=='meta'][0]
        self.assertEqual(len(bases),3)
        self.assertEqual(meta['model_input_features'],3)
        self.assertTrue(all(r['checks']['preprocessor_width_matches_classifier'] for r in bases))
        xgb=[r for r in bases if r['class_name']=='XGBClassifier'][0]
        self.assertEqual(xgb['preprocessing']['representation'],'dense_float32')
        self.assertFalse(report['internal_oof_estimators_retained'])


if __name__=='__main__':
    unittest.main(verbosity=2)
