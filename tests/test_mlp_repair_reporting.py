"""Synthetic report checks only; no model fitting or real data access."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest

import pandas as pd

from src.mlp_repair_reporting import export_mlp_repair_report, load_tables, VARIANTS


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class MLPRepairReportingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='mlp_repair_report_')
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.root = base / 'artifacts' / 'mlp_repair_20k_v1'
        self.report = base / 'reports' / 'mlp_repair_20k_v1'
        self.root.mkdir(parents=True)
        save(self.root / 'plan.json', {'scope': 'mlp_repair_20k_v1', 'test_used_for_selection': False})
        cv, candidates, selections = [], [], []
        for level in range(1, 5):
            for i, variant in enumerate(VARIANTS):
                losses = [.26 + .01*i + .001*fold for fold in range(3)]
                for fold, loss in enumerate(losses):
                    cv.append({'level': level, 'variant': variant, 'fold': fold, 'log_loss': loss})
                candidates.append({'level': level, 'variant': variant,
                                   'mean_log_loss': sum(losses)/3, 'std_log_loss': .00081649658})
            selections.append({'level': level, 'variant': 'stop_only', 'eligible_for_test': True,
                               'mean_cv_log_loss': .261, 'baseline_cv_log_loss': .4,
                               'constant_prior_cv_log_loss': .2806})
            stage = self.root / 'final' / f'level_{level}'
            metrics = {'log_loss': .25, 'pr_auc_average_precision': .23, 'roc_auc': .75, 'brier_score': .07}
            save(stage / 'metrics.json', {'level': level, 'variant': 'stop_only', 'metrics': metrics,
                'baseline_metrics': {**metrics, 'log_loss': .5}, 'fit_seconds': 2., 'predict_seconds': .2})
            save(stage / 'fit_diagnostics.json', {'best_epoch': 2, 'final_optimizer': {'epochs_completed': 2},
                'epoch_curve': [{'epoch': 1, 'train_log_loss': .3, 'validation_log_loss': .29},
                                {'epoch': 2, 'train_log_loss': .25, 'validation_log_loss': .26},
                                {'epoch': 3, 'train_log_loss': .2, 'validation_log_loss': .28}]})
            save(stage / 'complete.json', {'artifact_sha256': {name: hashlib.sha256((stage/name).read_bytes()).hexdigest()
                  for name in ('metrics.json', 'fit_diagnostics.json')}})
        pd.DataFrame(cv).to_csv(self.root / 'cv_results.csv', index=False)
        pd.DataFrame(candidates).to_csv(self.root / 'candidate_summary.csv', index=False)
        save(self.root / 'selection.json', {'selections': selections})

    def test_full_synthetic_report_and_sources_unchanged(self):
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        result = export_mlp_repair_report(self.root, self.report)
        self.assertEqual(result['completed_cv_evaluations'], 48)
        self.assertEqual(result['completed_final_procedures'], 4)
        self.assertEqual(result['bootstrap_calls'], 0)
        self.assertEqual(len(result['figure_files']), 4)
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()})
        text = (self.report / 'results_report.html').read_text(encoding='utf-8')
        self.assertIn('exploratory', text)
        self.assertIn('figures/inner_learning_curves.png', text)
        self.assertEqual(len(pd.read_csv(self.report / 'before_after.csv')), 4)

    def test_missing_outer_fold_is_rejected(self):
        p = self.root / 'cv_results.csv'
        frame = pd.read_csv(p)
        frame.iloc[:-1].to_csv(p, index=False)
        with self.assertRaisesRegex(ValueError, '48'):
            load_tables(self.root)

    def test_tampered_final_metric_is_rejected(self):
        p = self.root / 'final/level_1/metrics.json'
        p.write_text(p.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            load_tables(self.root)

    def test_original_report_namespace_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'namespace'):
            export_mlp_repair_report(self.root, self.report.with_name('pilot_20k_5config'))


if __name__ == '__main__':
    unittest.main()
