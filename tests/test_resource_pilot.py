"""Ресурсный пилот на маленьких синтетических Parquet, без реального test/search."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src.fit_diagnostics import extract_fit_diagnostics
from src.model_pipelines import MODEL_NAMES, make_estimator
from src.model_training import PreparedData, atomic_json, initialize_run, read_json
from src.resource_pilot import (describe_configuration, resource_context,
                                run_resource_pilot, worker)
from test_training_orchestration import make_prepared_fixture


class ResourcePilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='rmt_resource_pilot_')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.base=Path(cls.temp.name)
        cls.prepared=cls.base/'prepared'
        make_prepared_fixture(cls.prepared)

    def setUp(self):
        self.data=PreparedData(self.prepared)
        self.context=resource_context(self.data,threads=1)
        self.root=initialize_run(self.base/self._testMethodName,self.context)

    def test_all_full_configurations_are_serializable_and_untuned(self):
        numeric,categorical=self.data.columns(4)
        self.assertEqual(set(self.context['selected_configurations']),set(MODEL_NAMES))
        for model in MODEL_NAMES:
            described=describe_configuration(make_estimator(model,numeric,categorical,
                seed=42,technical=False,threads=1))
            self.assertEqual(json.loads(json.dumps(described,allow_nan=False)),described)
            self.assertEqual(described,self.context['selected_configurations'][model])
        mlp=self.context['selected_configurations']['mlp']['final']['parameters']
        self.assertEqual(mlp['max_iter'],300)
        self.assertEqual(mlp['tol'],1e-4)
        self.assertEqual(mlp['n_iter_no_change'],20)
        self.assertEqual(mlp['hidden_layer_sizes'],[64,32])
        self.assertFalse(mlp['early_stopping'])
        self.assertFalse(self.context['technical'])
        self.assertFalse(self.context['randomized_search'])
        self.assertFalse(self.context['test_used'])
        self.assertEqual(self.context['train_rows'],86)
        self.assertEqual(self.context['outer_fits_planned'],7)
        self.assertEqual(self.context['primitive_fits_planned'],21)

    def test_mlp_worker_uses_full_saved_train_and_preserves_actual_diagnostics(self):
        original_load=PreparedData.load
        requests=[]
        def checked_load(dataset,level,requested_ids,*,purpose):
            requests.append((level,np.array(requested_ids),purpose))
            np.testing.assert_array_equal(requested_ids,dataset.repeats[42])
            self.assertTrue(set(requested_ids).isdisjoint(dataset.test))
            return original_load(dataset,level,requested_ids,purpose=purpose)
        with patch.object(PreparedData,'load',autospec=True,side_effect=checked_load), \
             patch.object(RandomizedSearchCV,'fit',side_effect=AssertionError('No search in resource pilot')), \
             patch.object(Pipeline,'predict_proba',side_effect=AssertionError('No evaluation in resource worker')):
            exit_code=worker(self.prepared,self.root,'mlp')
        self.assertEqual(exit_code,0)
        self.assertEqual(len(requests),1)
        stage=self.root/'mlp'
        result=read_json(stage/'worker_result.json')
        diagnostics=read_json(stage/'fit_diagnostics.json')
        warnings=read_json(stage/'warnings.json')
        model=joblib.load(stage/'pipeline.joblib')
        self.assertEqual(diagnostics,extract_fit_diagnostics(model))
        actual=diagnostics['mlp'][0]
        self.assertEqual(actual['config']['max_iter'],300)
        self.assertEqual(actual['config']['n_iter_no_change'],20)
        self.assertEqual(actual['config']['tol'],1e-4)
        self.assertGreater(actual['fitted']['n_iter_'],0)
        self.assertLessEqual(actual['fitted']['n_iter_'],300)
        self.assertEqual(len(actual['fitted']['loss_curve_']),actual['fitted']['n_iter_'])
        self.assertEqual(actual['fitted']['loss_'],actual['fitted']['loss_curve_'][-1])
        self.assertEqual(result['warnings'],warnings)
        self.assertIsInstance(warnings,list)
        self.assertEqual(result['rows'],86)
        self.assertEqual(result['raw_features'],5)
        self.assertEqual(result['outer_fit_count'],1)
        self.assertFalse(result['test_used'])
        self.assertFalse(result['randomized_search_used'])
        self.assertEqual(result['model_size_bytes'],(stage/'pipeline.joblib').stat().st_size)
        self.assertGreater(result['fit_seconds'],0)
        self.assertTrue(result['serialization_diagnostics_equal'])
        self.assertEqual(result['processed_features'],diagnostics['processed_feature_counts'])

    def test_exit_zero_without_worker_outcome_is_not_marked_complete(self):
        # A dead/incomplete worker must not become a successful checkpoint merely
        # because its process reported exit_code=0. No classifier is fit here.
        monitor={'exit_code':0,'status':'completed','peak_tree_rss_bytes':1024,
                 'fit_peak_tree_rss_bytes':512,'windows_main_peak_wset_bytes':None,
                 'min_available_bytes':1024**3}
        parent=self.base/'missing_outcome'
        with patch('src.resource_monitor.monitor_subprocess',return_value=monitor) as mocked, \
             patch.object(Pipeline,'fit',side_effect=AssertionError('No real fit in mocked controller')):
            summary=run_resource_pilot(self.prepared,parent,threads=1)
        self.assertEqual(mocked.call_count,7)
        self.assertEqual(summary['completed_models'],0)
        self.assertEqual(summary['attempted_models'],7)
        self.assertEqual(summary['status'],'partial')
        self.assertTrue(summary['input_sha256_unchanged'])
        root=Path(summary['root'])
        for item in summary['results']:
            self.assertEqual(item['status'],'failed')
            self.assertFalse((root/item['model']/'complete.json').exists())
            self.assertEqual(item['peak_ram_bytes'],1024)


if __name__=='__main__':
    unittest.main(verbosity=2)
