"""Синтетические checkpoints проверяют отчётность; модели здесь не обучаются."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import pandas as pd

from src.pilot_reporting import MODEL_LABELS, SCOPE, _atomic_text, _latex, export_pilot_reports


def put_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8')


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage(root,model,level,*,mlp=False):
    directory=root/model/f'level_{level}'
    record={'scope':SCOPE,'model':model,'level':level,'train_rows':20000,'test_rows':61503,
        'raw_features':100+level,'selected_params':{'model__alpha':0.001},'best_cv_log_loss':0.35,
        'search_and_refit_seconds':100.0,'refit_seconds':10.0,'predict_seconds':2.0,'total_pair_seconds':105.0,
        'warnings':[{'category':'ConvergenceWarning','message':'Синтетическое предупреждение','count':1}] if mlp else [],
        'metrics':{'rows':61503,'positives':4925,'positive_rate':4925/61503,
            'pr_auc_average_precision':0.2+level*.01,'roc_auc':0.7+level*.01,
            'log_loss':0.35-level*.01,'brier_score':0.08-level*.001,'threshold':0.5,
            'accuracy':0.91,'precision':0.6,'recall':0.3,'f1':0.4,
            'tn':55578,'fp':1000,'fn':3925,'tp':1000,'approval_rate_at_0_5':0.96}}
    put_json(directory/'metrics.json',record)
    (directory/'pipeline.joblib').write_bytes(b'synthetic artifact, not a serialized estimator')
    names=['metrics.json','pipeline.joblib']
    if mlp:
        put_json(directory/'fit_diagnostics.json',{'mlp':[{'path':'pipeline.model',
            'class_name':'MLPClassifier','config':{'max_iter':300,'tol':1e-4,'n_iter_no_change':20},
            'fitted':{'n_iter_':3,'loss_curve_':[0.5,0.4,0.3],'loss_':0.3}}]})
        names.append('fit_diagnostics.json')
    put_json(directory/'complete.json',{'fingerprint':'fixture',
        'artifact_sha256':{name:digest(directory/name) for name in names}})
    return directory


class PilotReportingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='rmt_pilot_reports_')
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'pilot'
        self.reports=Path(self.temp.name)/'reports'

    def manifest(self):
        put_json(self.root/'experiment_manifest.json',{'mode':SCOPE,'fingerprint':'fixture'})

    def test_not_started_has_28_pending_rows_and_does_not_create_source(self):
        result=export_pilot_reports(self.root,self.reports)
        self.assertEqual(result['status'],'not_started')
        self.assertEqual((result['completed_pairs'],result['pending_pairs']),(0,28))
        self.assertFalse(self.root.exists())
        self.assertTrue(pd.read_csv(result['files']['metrics_csv']).empty)
        self.assertEqual(len(pd.read_csv(result['files']['pair_status_csv'])),28)
        self.assertTrue(pd.read_csv(result['files']['warnings_csv']).empty)
        self.assertIn('\\toprule',Path(result['files']['metrics_tex']).read_text(encoding='utf-8'))

    def test_incremental_report_keeps_missing_baseline_empty_then_computes_rq2(self):
        self.manifest()
        stage(self.root,'random_forest',2)
        first=export_pilot_reports(self.root,self.reports)
        gains=pd.read_csv(first['files']['rq2_history_gains_csv'])
        level2=gains.query("model=='random_forest' and level==2").iloc[0]
        self.assertFalse(level2['level1_available'])
        self.assertTrue(pd.isna(level2['delta_log_loss_vs_level1']))
        stage(self.root,'random_forest',1)
        before={str(p.relative_to(self.root)):digest(p) for p in self.root.rglob('*') if p.is_file()}
        second=export_pilot_reports(self.root,self.reports)
        gains=pd.read_csv(second['files']['rq2_history_gains_csv'])
        level2=gains.query("model=='random_forest' and level==2").iloc[0]
        self.assertTrue(level2['level1_available'])
        self.assertAlmostEqual(level2['delta_pr_auc_average_precision_vs_level1'],0.01)
        self.assertAlmostEqual(level2['delta_log_loss_vs_level1'],-0.01)
        self.assertEqual(second['completed_pairs'],2)
        self.assertEqual(before,{str(p.relative_to(self.root)):digest(p) for p in self.root.rglob('*') if p.is_file()})
        # Без marker даже корректно выглядящий metrics.json остаётся pending.
        put_json(self.root/'random_forest/level_3/metrics.json',{'test_log_loss':0.0001})
        third=export_pilot_reports(self.root,self.reports)
        self.assertEqual(third['completed_pairs'],2)

    def test_invalid_hashed_auxiliary_artifact_excludes_metrics_and_warnings(self):
        self.manifest()
        directory=stage(self.root,'mlp',1,mlp=True)
        (directory/'pipeline.joblib').write_bytes(b'corrupted')
        result=export_pilot_reports(self.root,self.reports)
        self.assertEqual(result['status'],'invalid_artifacts')
        self.assertEqual(result['invalid_pairs'],1)
        self.assertEqual(result['completed_pairs'],0)
        self.assertTrue(pd.read_csv(result['files']['metrics_csv']).empty)
        self.assertTrue(pd.read_csv(result['files']['warnings_csv']).empty)
        self.assertTrue(pd.read_csv(result['files']['mlp_diagnostics_csv']).empty)

    def test_marker_must_hash_metrics_and_match_fingerprint(self):
        self.manifest()
        first=stage(self.root,'logistic_regression',1)
        marker=json.loads((first/'complete.json').read_text())
        del marker['artifact_sha256']['metrics.json']
        put_json(first/'complete.json',marker)
        second=stage(self.root,'logistic_regression',2)
        marker=json.loads((second/'complete.json').read_text()); marker['fingerprint']='other'
        put_json(second/'complete.json',marker)
        result=export_pilot_reports(self.root,self.reports)
        self.assertEqual(result['invalid_pairs'],2)
        self.assertEqual(result['completed_pairs'],0)

    def test_mlp_diagnostics_warnings_and_complete_report(self):
        self.manifest()
        for model in MODEL_LABELS:
            for level in range(1,5): stage(self.root,model,level,mlp=model=='mlp')
        result=export_pilot_reports(self.root,self.reports)
        self.assertEqual(result['status'],'completed')
        self.assertEqual(result['completed_pairs'],28)
        self.assertEqual(result['mlp_diagnostic_rows'],4)
        self.assertEqual(result['warning_count'],4)
        mlp=pd.read_csv(result['files']['mlp_diagnostics_csv'])
        self.assertTrue((mlp['n_iter_']==3).all())
        self.assertTrue((mlp['max_iter']==300).all())
        self.assertTrue(mlp['loss_curve_length_matches_n_iter'].all())
        curves=json.loads(Path(result['files']['mlp_diagnostics_json']).read_text(encoding='utf-8'))
        self.assertEqual(curves[0]['diagnostic']['fitted']['loss_curve_'],[0.5,0.4,0.3])
        comparison=pd.read_csv(result['files']['rq1_model_comparison_csv'])
        self.assertEqual(comparison.query('level==1')['model'].tolist(),list(MODEL_LABELS))
        self.assertFalse(any('rank' in c for c in comparison.columns))
        metadata=json.loads(Path(result['files']['metadata_json']).read_text(encoding='utf-8'))
        self.assertFalse(metadata['test_used_for_tuning'])
        self.assertFalse(metadata['statistical_significance_claimed'])
        self.assertFalse(metadata['shap_computed'])
        self.assertEqual(metadata['plan']['candidate_cv_fits'],420)
        self.assertEqual(metadata['plan']['refits'],28)
        self.assertEqual(metadata['plan']['repeat_trainings'],0)
        self.assertEqual(len(pd.read_csv(result['files']['metrics_csv'])),28)

    def test_nested_diagnostics_path_must_be_protected_by_marker(self):
        self.manifest()
        directory=stage(self.root,'mlp',1,mlp=True)
        diagnostic=json.loads((directory/'fit_diagnostics.json').read_text(encoding='utf-8'))
        nested='fit_checkpoint/fit_diagnostics.json'
        put_json(directory/nested,diagnostic)
        metrics=json.loads((directory/'metrics.json').read_text(encoding='utf-8'))
        metrics['fit_diagnostics_file']=nested
        put_json(directory/'metrics.json',metrics)
        marker=json.loads((directory/'complete.json').read_text(encoding='utf-8'))
        marker['artifact_sha256']['metrics.json']=digest(directory/'metrics.json')
        # Старый файл защищён, но явно указанный nested ещё нет: его не читаем.
        put_json(directory/'complete.json',marker)
        first=export_pilot_reports(self.root,self.reports)
        self.assertEqual(first['completed_pairs'],1)
        self.assertEqual(first['mlp_diagnostic_rows'],0)
        marker['artifact_sha256'][nested]=digest(directory/nested)
        put_json(directory/'complete.json',marker)
        second=export_pilot_reports(self.root,self.reports)
        self.assertEqual(second['mlp_diagnostic_rows'],1)
        rows=pd.read_csv(second['files']['mlp_diagnostics_csv'])
        self.assertEqual(Path(rows.iloc[0]['source']),directory/nested)

    def test_latex_escapes_text_and_formats_missing_as_dash(self):
        frame=pd.DataFrame([{'model':'name_with_underscore & 10%','metric':None}])
        rendered=_latex(frame,[('model','Model_name'),('metric','Log_loss')],'Русская подпись','tab:fixture')
        self.assertIn(r'name\_with\_underscore \& 10\%',rendered)
        self.assertIn(r'Log\_loss',rendered)
        self.assertIn('--',rendered)
        self.assertIn('Русская подпись',rendered)
        self.assertIn(r'\toprule',rendered)

    def test_csv_line_endings_are_preserved_without_windows_blank_rows(self):
        path=self.reports/'mixed_newlines.csv'
        source='название,значение\r\nпервая,1\r\nвторая,2\n'
        _atomic_text(path,source)
        self.assertEqual(path.read_bytes(),source.encode('utf-8'))
        result=export_pilot_reports(self.root,self.reports)
        content=Path(result['files']['pair_status_csv']).read_bytes()
        self.assertNotIn(b'\r\r\n',content)
        self.assertEqual(len(content.splitlines()),29)

    def test_atomic_write_retries_permission_error_five_times_then_raises(self):
        path=self.reports/'locked.csv'
        with patch('src.pilot_reporting.os.replace',side_effect=PermissionError('fixture lock')) as replace:
            with patch('src.pilot_reporting.time.sleep') as sleep:
                with self.assertRaises(PermissionError): _atomic_text(path,'x\r\n')
        self.assertEqual(replace.call_count,6)
        self.assertEqual([call.args[0] for call in sleep.call_args_list],[1,2,4,8,16])
        self.assertFalse(path.exists())

    def test_level4_checkpoint_invokes_lazy_decision_postprocessing_only_from_hashed_probabilities(self):
        self.manifest()
        directory=stage(self.root,'logistic_regression',4)
        module=types.ModuleType('src.pilot_decision_metrics')
        calls=[]
        def export(root,report_dir,prepared_dir=None):
            calls.append((root,report_dir,prepared_dir))
            return {'status':'available','completed_level4_models':['logistic_regression'],
                    'additional_fit_calls':0,'files':{'scenarios_csv':'fixture_decision_scenarios.csv'}}
        module.export_decision_metrics=export
        with patch.dict('sys.modules',{'src.pilot_decision_metrics':module}):
            initial=export_pilot_reports(self.root,self.reports)
            self.assertEqual(initial['decision_metrics']['status'],'not_ready')
            self.assertEqual(calls,[])
            (directory/'probabilities.parquet').write_bytes(b'synthetic; never read by mocked callback')
            marker=json.loads((directory/'complete.json').read_text(encoding='utf-8'))
            marker['artifact_sha256']['probabilities.parquet']=digest(directory/'probabilities.parquet')
            put_json(directory/'complete.json',marker)
            result=export_pilot_reports(self.root,self.reports)
        self.assertEqual(calls,[(self.root,self.reports,None)])
        self.assertEqual(result['decision_metrics']['additional_fit_calls'],0)
        self.assertEqual(result['files']['decision_metrics_scenarios_csv'],'fixture_decision_scenarios.csv')


if __name__=='__main__': unittest.main(verbosity=2)
