"""No-model tests of launcher checkpoint integrity and lock handling."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location('final_launcher',
    Path(__file__).resolve().parents[1] / 'scripts/run_final_pilot_background.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class FinalLauncherTests(unittest.TestCase):
    def test_verified_pair_requires_hashes_and_complete_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage = root / 'logistic_regression/level_1'
            (stage / 'fit_checkpoint').mkdir(parents=True)
            files = ['probabilities.parquet', 'metrics.json', 'fit_checkpoint/complete.json']
            for name in files:
                (stage / name).write_bytes(b'fixture')
            marker = {'fingerprint': 'fixture', 'artifact_sha256': {
                name: launcher.digest(stage / name) for name in files}}
            (stage / 'complete.json').write_text(json.dumps(marker))
            with patch.object(launcher, 'PAIRS', [('logistic_regression', 1)]):
                self.assertEqual(launcher.verified_pairs(root, {'fingerprint': 'fixture'}),
                                 [('logistic_regression', 1)])
                (stage / 'metrics.json').write_bytes(b'corrupt')
                with self.assertRaisesRegex(ValueError, 'Checkpoint SHA'):
                    launcher.verified_pairs(root, {'fingerprint': 'fixture'})

    def test_unknown_host_cannot_be_treated_as_stale(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'supervisor.lock'
            path.write_text(json.dumps({'pid': 123, 'create_time': 1, 'host': 'another-host'}))
            with patch.object(launcher.platform, 'node', return_value='this-host'):
                self.assertEqual(launcher.lock_state(path)['state'], 'unknown_host')

    def test_access_denied_cannot_be_treated_as_stale(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'supervisor.lock'
            path.write_text(json.dumps({'pid': 123, 'create_time': 1, 'host': 'this-host'}))
            with patch.object(launcher.platform, 'node', return_value='this-host'), \
                 patch.object(launcher.psutil, 'Process', side_effect=launcher.psutil.AccessDenied(123)):
                self.assertEqual(launcher.lock_state(path)['state'], 'unknown_access_denied')


if __name__ == '__main__':
    unittest.main()
