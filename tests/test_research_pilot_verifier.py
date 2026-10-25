"""Независимый verifier проверяется на маленьких сохранённых артефактах."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.research_pilot import initialize_pilot, run_pair
from test_training_orchestration import make_prepared_fixture


PROJECT=Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2),encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(root):
    return {path.relative_to(root).as_posix():digest(path)
            for path in Path(root).rglob('*') if path.is_file()}


class ResearchPilotVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary_parent=PROJECT/'tmp/test_temp'
        temporary_parent.mkdir(parents=True,exist_ok=True)
        cls.temp=tempfile.TemporaryDirectory(prefix='rmt_pilot_verifier_',dir=temporary_parent)
        cls.addClassCleanup(cls.temp.cleanup)
        cls.base=Path(cls.temp.name)
        cls.prepared=cls.base/'prepared'
        cls.baseline=cls.base/'baseline_pilot'
        make_prepared_fixture(cls.prepared)
        initialize_pilot(cls.prepared,cls.baseline,threads=1,sample_n=60)
        # Одна синтетическая пара (15 CV-fit + refit) создаёт файлы для всех тестов.
        with redirect_stdout(io.StringIO()):
            run_pair(cls.prepared,cls.baseline,'logistic_regression',1)
        manifest=read_json(cls.baseline/'experiment_manifest.json')
        cls.expected_input_count=len(manifest['inputs'])
        spec=importlib.util.spec_from_file_location('research_pilot_verifier_under_test',
                                                  PROJECT/'scripts/verify_research_pilot.py')
        cls.verifier=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=cls.verifier
        spec.loader.exec_module(cls.verifier)

    def setUp(self):
        self.root=self.base/self._testMethodName
        shutil.copytree(self.baseline,self.root)
        self.stage=self.root/'logistic_regression/level_1'

    def verify(self,*,require_complete=False,prepared=None,require_attempt_logs=False):
        # Чтение model.joblib для извлечения прогноза тоже не является проверкой файлов.
        with ExitStack() as stack:
            for model in (Pipeline,LogisticRegression):
                for method in ('fit','predict','predict_proba'):
                    stack.enter_context(patch.object(model,method,
                        side_effect=AssertionError('Verifier must not fit or predict')))
            stack.enter_context(patch.object(joblib,'load',
                side_effect=AssertionError('Verifier must not unpickle models')))
            result=self.verifier.verify_pilot(prepared or self.prepared,self.root,
                require_complete=require_complete,expected_rows=60,expected_test_rows=24,
                expected_widths={1:2,2:3,3:4,4:5},
                expected_input_count=self.expected_input_count,source_dir=PROJECT/'src',
                require_attempt_logs=require_attempt_logs)
        json.dumps(result,allow_nan=False)
        return result

    def refresh_stage_hashes(self):
        # Имитируем согласованные SHA, чтобы тест доходил до смысловой проверки.
        for directory in (self.stage/'fit_checkpoint',self.stage):
            path=directory/'complete.json'
            marker=read_json(path)
            marker['artifact_sha256']={name:digest(directory/name)
                                       for name in marker['artifact_sha256']}
            write_json(path,marker)

    def assert_failed(self,report):
        self.assertEqual(report['status'],'failed')
        self.assertTrue(report['errors'])

    def test_partial_verification_is_read_only_and_does_not_fit_or_predict(self):
        prepared_before=inventory(self.prepared)
        pilot_before=inventory(self.root)
        report=self.verify()
        self.assertEqual(report['status'],'passed',report['errors'])
        self.assertEqual(report['errors'],[])
        self.assertEqual(report['summary']['completed_pairs'],1)
        self.assertEqual(report['summary']['fit_checkpoints'],1)
        self.assertEqual(prepared_before,inventory(self.prepared))
        self.assertEqual(pilot_before,inventory(self.root))

    def test_require_complete_rejects_partial_pilot(self):
        report=self.verify(require_complete=True)
        self.assert_failed(report)
        self.assertEqual(report['summary']['completed_pairs'],1)

    def test_corrupted_probability_file_is_rejected_by_sha(self):
        path=self.stage/'probabilities.parquet'
        with path.open('ab') as stream:
            stream.write(b'corrupted')
        self.assert_failed(self.verify())

    def test_wrong_prediction_ids_rejected_even_when_artifact_sha_matches(self):
        path=self.stage/'probabilities.parquet'
        predictions=pd.read_parquet(path)
        cohort=pd.read_parquet(self.root/'cohort_ids.parquet')
        predictions.loc[0,'SK_ID_CURR']=cohort.SK_ID_CURR.iloc[0]
        predictions.to_parquet(path,index=False)
        self.refresh_stage_hashes()
        self.assert_failed(self.verify())

    def test_cv_mean_inconsistent_with_fold_scores_is_rejected(self):
        path=self.stage/'fit_checkpoint/cv_results.parquet'
        candidates=pd.read_parquet(path)
        candidates.loc[0,'mean_neg_log_loss']+=0.1
        candidates.to_parquet(path,index=False)
        self.refresh_stage_hashes()
        self.assert_failed(self.verify())

    def test_manual_extra_fit_rejected_even_when_records_and_sha_agree(self):
        for name in ('fit_checkpoint/fit_summary.json','metrics.json'):
            path=self.stage/name
            record=read_json(path)
            record['fit_calls']['manual_extra']=1
            write_json(path,record)
        self.refresh_stage_hashes()
        self.assert_failed(self.verify())

    def test_changed_prepared_input_digest_is_rejected(self):
        changed=self.base/'changed_prepared'
        shutil.copytree(self.prepared,changed)
        path=changed/'feature_levels.json'
        path.write_text(path.read_text(encoding='utf-8')+'\n',encoding='utf-8')
        self.assert_failed(self.verify(prepared=changed))

    def test_multiple_attempt_logs_report_extra_cv_work_without_single_pass_claim(self):
        logs=self.stage/'attempt_logs'
        logs.mkdir()
        banner='Fitting 3 folds for each of 5 candidates, totalling 15 fits\n'
        cv_line='[CV] END model__C=1.0; total time=0.0s\n'
        (logs/'attempt_001.log').write_text(banner+cv_line*4,encoding='utf-8')
        (logs/'attempt_002.log').write_text(
            banner+cv_line*15+'Completed logistic_regression Level 1; checkpoint saved.\n',
            encoding='utf-8')
        report=self.verify(require_attempt_logs=True)
        self.assertEqual(report['status'],'passed',report['errors'])
        self.assertEqual(report['completeness'],'partial')
        self.assertEqual(report['logs']['attempt_log_count'],2)
        self.assertEqual(report['logs']['cv_end_lines'],19)
        self.assertEqual(report['budget']['observed_cv_end_log_lines'],19)
        self.assertEqual(report['budget']['saved_candidate_cv_fits'],15)
        self.assertEqual(report['logs']['pairs_with_multiple_attempts'],['logistic_regression/level_1'])
        self.assertFalse(report['logs']['single_pass_420_cv_and_28_refits_supported'])
        self.assertTrue(any('retr' in message.lower() for message in report['warnings']))
        pair=next(row for row in report['pairs']
                  if row['model']=='logistic_regression' and row['level']==1)
        self.assertTrue(pair['verified'])
        self.assertEqual(pair['logs']['attempt_log_count'],2)
        self.assertEqual(pair['logs']['cv_end_lines'],19)

    def test_single_attempt_log_with_fourteen_cv_lines_is_rejected(self):
        logs=self.stage/'attempt_logs'
        logs.mkdir()
        (logs/'attempt_001.log').write_text(
            'Fitting 3 folds for each of 5 candidates, totalling 15 fits\n'
            +'[CV] END model__C=1.0; total time=0.0s\n'*14
            +'Completed logistic_regression Level 1; checkpoint saved.\n',encoding='utf-8')
        report=self.verify(require_attempt_logs=True)
        self.assert_failed(report)
        self.assertEqual(report['logs']['attempt_log_count'],1)
        self.assertEqual(report['budget']['observed_cv_end_log_lines'],14)
        self.assertTrue(any('15 CV completion lines' in error for error in report['errors']))


if __name__=='__main__':
    unittest.main(verbosity=2)
