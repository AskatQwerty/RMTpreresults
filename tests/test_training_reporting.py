"""Отчёт из чистого ядра/диска; массовое обучение здесь только подменяется fixture."""
from __future__ import annotations

import ast
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import pandas as pd

from src.data_quality import artifact_manifest
from src.training_reporting import (MODELS,write_full_report,write_technical_reports,
                                    write_resource_reports)

PROJECT=Path(__file__).resolve().parents[1]


def put_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def hash_file(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def add_stage(root,level,model,seed=None):
    path=Path(root)/f'level_{level}'/model/('search' if seed is None else f'seed_{seed}')
    name='best_params.json' if seed is None else 'metrics.json'
    record={'scope':'full','level':level,'model':model,'warnings':[]}
    if seed is None: record.update({'n_iter':20,'folds':3,'elapsed_seconds':1.0})
    else: record.update({'seed':seed,'fit_seconds':2.0,'metrics':{'log_loss':0.3,'roc_auc':0.7}})
    put_json(path/name,record)
    put_json(path/'complete.json',{'fingerprint':'fixture','artifact_sha256':{name:hash_file(path/name)}})


def complete_fixture(prepared,root):
    put_json(Path(root)/'experiment_manifest.json',{'mode':'full','fingerprint':'fixture','inputs':artifact_manifest(prepared)})
    for level in range(1,5):
        for model in MODELS:
            add_stage(root,level,model)
            for seed in range(42,47): add_stage(root,level,model,seed)
    return {'scope':'full','completed_searches':28,'completed_main_fits':140}


@contextmanager
def chdir(path):
    previous=Path.cwd()
    os.chdir(path)
    try: yield
    finally: os.chdir(previous)


def code_cells():
    """Прочитать строки code(...), не выполняя генератор или notebook."""
    tree=ast.parse((PROJECT/'scripts/build_training_notebook.py').read_text(encoding='utf-8'))
    found=[]
    for node in tree.body:
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='code':
            found.append((ast.literal_eval(node.value.args[0]),{k.arg:ast.literal_eval(k.value) for k in node.value.keywords}))
    return found


class TrainingReportingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='rmt_reports_')
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.prepared=self.root/'data/processed/full'
        self.prepared.mkdir(parents=True)
        (self.prepared/'immutable.txt').write_text('synthetic input',encoding='utf-8')

    def test_clean_configuration_full_section_and_report_without_technical_globals(self):
        cells=code_cells()
        config=cells[0][0]
        full=next(source for source,meta in cells if 'full_experiment_explicit_opt_in' in meta.get('tags',[]))
        report=next(source for source,meta in cells if 'disk_reports_independent_of_technical' in meta.get('tags',[]))
        fake_resource=types.ModuleType('src.resource_pilot')
        fake_resource.run_resource_pilot=lambda *args,**kwargs: self.fail('Resource pilot must not run in this test')
        namespace={}
        with chdir(self.root),patch.dict('sys.modules',{'src.resource_pilot':fake_resource}):
            exec(config,namespace)
            namespace['RUN_FULL_EXPERIMENT']=True
            namespace['display']=lambda *args,**kwargs:None
            namespace['run_full_experiment']=lambda prepared,output,threads:complete_fixture(prepared,output)
            exec(full,namespace)
            self.assertTrue(all(name not in namespace for name in ['technical','technical_root','warnings_frame','frozen_before']))
            exec(report,namespace)
        self.assertEqual(namespace['full_disk_report']['status'],'completed')
        self.assertEqual(namespace['full_disk_report']['completed_searches'],28)
        self.assertEqual(namespace['full_disk_report']['completed_main_fits'],140)
        self.assertTrue(namespace['full_disk_report']['input_integrity']['unchanged'])
        self.assertEqual(namespace['technical_disk_report']['runs_found'],0)

    def test_missing_run_is_not_started_and_report_writes_no_run_directory(self):
        absent=self.root/'missing_full'
        result=write_full_report(absent,self.root/'reports/full',prepared_dir=self.prepared)
        self.assertEqual(result['status'],'not_started')
        self.assertFalse(absent.exists())
        self.assertEqual(write_technical_reports(self.root/'missing_tech',self.root/'reports/technical')['runs_found'],0)

    def test_clean_configuration_disabled_full_and_report_do_not_require_fit_result(self):
        cells=code_cells()
        full=next(source for source,meta in cells if 'full_experiment_explicit_opt_in' in meta.get('tags',[]))
        report=next(source for source,meta in cells if 'disk_reports_independent_of_technical' in meta.get('tags',[]))
        fake_resource=types.ModuleType('src.resource_pilot')
        fake_resource.run_resource_pilot=lambda *args,**kwargs:self.fail('Resource pilot must remain disabled')
        namespace={}
        with chdir(self.root),patch.dict('sys.modules',{'src.resource_pilot':fake_resource}):
            exec(cells[0][0],namespace)
            namespace['display']=lambda *args,**kwargs:None
            namespace['run_full_experiment']=lambda *args,**kwargs:self.fail('Full experiment must remain disabled')
            exec(full,namespace)
            namespace.pop('full_result',None)
            exec(report,namespace)
        self.assertTrue(all(name not in namespace for name in [
            'technical','technical_root','warnings_frame','frozen_before','full_result','resource_result']))
        self.assertEqual(namespace['full_disk_report']['status'],'not_started')
        self.assertEqual(namespace['resource_disk_report']['status'],'not_run')
        self.assertTrue(pd.read_csv(namespace['resource_disk_report']['model_metrics_csv']).empty)
        self.assertFalse(namespace['FULL_OUTPUT'].exists())

    def test_partial_full_report_checks_hashes_and_preserves_artifacts(self):
        full=self.root/'full'
        put_json(full/'experiment_manifest.json',{'mode':'full','fingerprint':'fixture','inputs':artifact_manifest(self.prepared)})
        add_stage(full,1,'logistic_regression')
        add_stage(full,1,'logistic_regression',42)
        before={str(p.relative_to(full)):hash_file(p) for p in full.rglob('*') if p.is_file()}
        result=write_full_report(full,self.root/'reports/full',prepared_dir=self.prepared)
        self.assertEqual(result['status'],'partial_or_interrupted')
        self.assertEqual((result['completed_searches'],result['completed_main_fits']),(1,1))
        self.assertEqual(before,{str(p.relative_to(full)):hash_file(p) for p in full.rglob('*') if p.is_file()})
        metrics=full/'level_1/logistic_regression/seed_42/metrics.json'
        metrics.write_text('{}',encoding='utf-8')
        result=write_full_report(full,self.root/'reports/full',prepared_dir=self.prepared)
        self.assertEqual(result['status'],'invalid_artifacts')
        self.assertEqual(len(result['invalid_stages']),1)

    def test_resource_reports_are_separate_and_disk_only(self):
        parent=self.root/'resource'
        root=parent/'fixture'
        completed={'model':'mlp','status':'completed','scope':'resource_pilot','rows':221407,
            'raw_features':931,'processed_features':[{'path':'pipeline','n_features':1805}],
            'fit_seconds':1234.5,'worker_elapsed_seconds':1240.0,
            'peak_process_tree_rss_bytes':3*1024**3,'fit_peak_process_tree_rss_bytes':3*1024**3,
            'windows_main_peak_wset_bytes':4*1024**3,'peak_ram_bytes':4*1024**3,
            'model_size_bytes':123456,'warnings':[{'category':'ConvergenceWarning','message':'fixture warning','count':1}],
            'actual_mlp_iterations':[300],'resumed':False}
        partial={'model':'stacking','status':'resource_guard','scope':'resource_pilot','rows':221407,
            'raw_features':931,'peak_process_tree_rss_bytes':5*1024**3,'peak_ram_bytes':5*1024**3,
            'warnings':[]}
        put_json(root/'resource_summary.json',{'scope':'resource_pilot','status':'partial',
            'completed_models':1,'attempted_models':2,'results':[completed,partial],
            'test_used':False,'randomized_search_used':False})
        # Контроллер сохраняет summary раньше Parquet: имитируем interruption между записями.
        pd.DataFrame([{'model':'mlp','fit_seconds':-999}]).to_parquet(root/'resource_metrics.parquet')
        before={str(p.relative_to(root)):hash_file(p) for p in root.rglob('*') if p.is_file()}
        result=write_resource_reports(parent,self.root/'reports/resource')
        self.assertEqual(result['runs_found'],1)
        self.assertEqual(result['runs'][0]['status'],'partial')
        self.assertTrue(result['independent_of_main_experiment'])
        metrics=pd.read_csv(result['model_metrics_csv']).set_index('model')
        self.assertEqual(len(metrics),2)
        self.assertEqual(metrics.loc['mlp','fit_seconds'],1234.5)
        self.assertEqual(metrics.loc['mlp','peak_ram_gib'],4)
        self.assertEqual(metrics.loc['mlp','model_size_bytes'],123456)
        self.assertEqual(metrics.loc['mlp','warning_count'],1)
        self.assertEqual(json.loads(metrics.loc['mlp','mlp_n_iter_json']),[300])
        self.assertEqual(json.loads(metrics.loc['mlp','processed_features_json']),completed['processed_features'])
        self.assertTrue(pd.isna(metrics.loc['stacking','fit_seconds']))
        self.assertTrue(pd.isna(metrics.loc['stacking','model_size_bytes']))
        self.assertEqual(metrics.loc['stacking','status'],'resource_guard')
        self.assertEqual(len(pd.read_csv(result['warnings_csv'])),1)
        self.assertTrue(Path(result['runs'][0]['metrics_csv']).is_file())
        self.assertEqual(before,{str(p.relative_to(root)):hash_file(p) for p in root.rglob('*') if p.is_file()})

    def test_resource_report_does_not_require_parquet(self):
        parent=self.root/'resource'
        put_json(parent/'fixture/resource_summary.json',{'scope':'resource_pilot','status':'partial',
            'completed_models':0,'attempted_models':1,'results':[
                {'model':'mlp','status':'failed','rows':221407,'raw_features':931,'warnings':[]}]})
        result=write_resource_reports(parent,self.root/'reports/resource')
        self.assertEqual(result['model_metrics'][0]['status'],'failed')
        self.assertIsNone(result['model_metrics'][0]['peak_ram_gib'])
        self.assertTrue(Path(result['model_metrics_csv']).is_file())


if __name__=='__main__': unittest.main(verbosity=2)
