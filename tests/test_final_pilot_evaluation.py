"""Regression checks for final-pilot alignment/isolation; no classifier fits."""
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.final_pilot_evaluation import (SCOPE, align_prediction_frame,
    validate_output_path, verify_saved_metrics)


class FinalPilotEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.aligned = pd.DataFrame({'SK_ID_CURR':[11,22,33], 'TARGET':[0,1,0]})
        self.frame = pd.DataFrame({'SK_ID_CURR':[33,11,22], 'TARGET':[0,0,1],
            'probability_default':[.3,.1,.8], 'scope':SCOPE,
            'model':'mlp', 'level':4, 'evaluation_split':'test'})

    def test_alignment_uses_ids_not_row_positions(self):
        p = align_prediction_frame(self.aligned, self.frame, 'mlp', 4)
        np.testing.assert_array_equal(p, [.1,.8,.3])

    def test_mismatch_target_duplicates_or_wrong_clients_rejected(self):
        for change in ('target','duplicate','wrong_id'):
            frame = self.frame.copy()
            if change == 'target': frame.loc[0,'TARGET']=1
            elif change == 'duplicate': frame.loc[0,'SK_ID_CURR']=11
            else: frame.loc[0,'SK_ID_CURR']=44
            with self.assertRaises(ValueError, msg=change):
                align_prediction_frame(self.aligned,frame,'mlp',4)

    def test_old_scope_wrong_level_and_invalid_probabilities_rejected(self):
        for field,value in [('scope','pilot_20k_5config'),('level',3),
                            ('probability_default',np.nan),('probability_default',1.01)]:
            frame = self.frame.copy()
            frame.loc[0,field]=value
            with self.assertRaises(ValueError):
                align_prediction_frame(self.aligned,frame,'mlp',4)

    def test_old_reports_never_output_destination(self):
        base=Path.cwd()/'tmp'/'evaluation_test_paths'
        previous=base/'reports/pilot_20k_5config'
        root=base/'artifacts/training'/SCOPE
        validate_output_path(base/'reports'/SCOPE/'evaluation', previous, root)
        for output in [previous/'evaluation', previous, root/'evaluation', base/'evaluation']:
            with self.assertRaises(ValueError):
                validate_output_path(output,previous,root)

    def test_metrics_checker_does_not_compare_new_models_to_old_metrics(self):
        point=[]; saved=[]
        for model in ['logistic_regression','lightgbm','xgboost','soft_voting','stacking','mlp','random_forest']:
            for level in range(1,5):
                point.append(dict(model=model,level=level,ap=.2,roc_auc=.7,log_loss=.25,brier=.07))
                saved.append(dict(model=model,level=level,scope=SCOPE,metrics={
                    'pr_auc_average_precision':.2,'roc_auc':.7,'log_loss':.25,'brier_score':.07}))
        checked=verify_saved_metrics(pd.DataFrame(point),saved)
        self.assertEqual(len(checked),112)
        self.assertTrue(checked.within_tolerance.all())
        saved[-1]['metrics']['log_loss']=.3
        with self.assertRaises(ValueError):
            verify_saved_metrics(pd.DataFrame(point),saved)

    def test_two_synthetic_bootstrap_replicates_and_report(self):
        from src.final_pilot_evaluation import (MODELS, LEVELS, quality_point,
            summarize_bootstrap, export_distributions)
        from src.pilot_evaluation_bootstrap import bootstrap_run, atomic_json
        from src.pilot_evaluation_decisions import DecisionEvaluator, calibration_point_table
        from src.pilot_delong import paired_delong_table
        from src.final_pilot_reporting import export_final_report
        y=np.tile([0,0,0,0,1],12).astype(np.int8)
        rng=np.random.default_rng(12)
        pairs=[dict(model=m,level=l) for m in MODELS for l in LEVELS]
        p=rng.uniform(.01,.7,size=(28,len(y)))
        groups={'gender':np.tile(['F','M'],30),'age':np.tile(['<25','>=25','>=25'],20)}
        probdict={pair['model']:p[i] for i,pair in enumerate(pairs) if pair['level']==4}
        point=quality_point(y,p,pairs)
        decisions=DecisionEvaluator(y,probdict,groups).point_tables()
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary)
            output=base/'reports'/SCOPE/'evaluation'
            output.mkdir(parents=True)
            arrays,progress=bootstrap_run(y,p,pairs,groups,output/'bootstrap',
                fingerprint='synthetic-test-no-real-clients',n_bootstrap=2,seed=42)
            self.assertEqual(progress['completed'],2)
            tables=summarize_bootstrap(arrays,point,pairs,decisions)
            tables.update(quality_point=point,calibration_bins=calibration_point_table(y,probdict),
                          delong=paired_delong_table(y,probdict))
            for name,frame in decisions.items(): tables['profit_point' if name=='profit' else name]=frame
            for name,frame in tables.items(): frame.to_csv(output/f'{name}.csv',index=False)
            export_distributions(arrays,pairs,output,2)
            self.assertEqual(len(pd.read_csv(output/'bootstrap_quality_distribution.csv')),56)
            atomic_json(output/'evaluation_summary.json',{'completed_bootstrap':2})
            atomic_json(output/'integrity_after.json',{'unchanged':True,'old_preresults_unchanged':True})
            mlp=pd.DataFrame([dict(model='mlp',level=l,n_iter_=300,max_iter=300,tol=.0001,
                n_iter_no_change=20,loss_curve_points=300,final_training_loss=.02,
                best_cv_log_loss=.5,test_log_loss=.6,test_ap=.2,warning_count_final_fit=1,
                selected_params_json='{}',fitted_config_json='{}') for l in LEVELS])
            with patch('src.final_pilot_reporting.saved_final_protocol',return_value=(
                {'scope':SCOPE},pd.DataFrame(),mlp,pd.DataFrame())):
                readiness=export_final_report(base/'artifacts/training'/SCOPE,output,
                    previous_reports=base/'reports/pilot_20k_5config')
            self.assertEqual(len(readiness['figure_files']),6)
            self.assertEqual(readiness['status'],'pending_checks')
            report=(output/'results_report.html').read_text(encoding='utf-8')
            self.assertIn('20 конфигураций',report)
            self.assertNotIn('20 000 train-клиентов, 5 конфигураций',report)
            self.assertTrue((output/'tables/rq1_level4.tex').exists())
            self.assertTrue((output/'tables/rq2_ap_levels.csv').exists())


if __name__=='__main__':
    unittest.main()
