"""Небольшие собственные дочерние процессы; никаких обучений Home Credit."""
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

import psutil

from src.resource_monitor import _terminate_owned_tree, monitor_subprocess


class ResourceMonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='rmt_monitor_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_script(self, source, **kwargs):
        script = self.root/'worker.py'
        script.write_text(source, encoding='utf-8')
        return monitor_subprocess([sys.executable, '-u', str(script)], cwd=self.root,
            log_path=self.root/'worker.log', phase_path=self.root/'phase.json',
            samples_path=self.root/'samples.csv', sample_interval=.05, **kwargs)

    def test_allocation_phase_peak_and_stdout_are_observed(self):
        result = self.run_script("""from pathlib import Path
import json,os,time
print('worker started',flush=True)
block=bytearray(32*1024**2)
for i in range(0,len(block),4096): block[i]=1
Path('phase.tmp').write_text(json.dumps({'phase':'fit'}))
os.replace('phase.tmp','phase.json')
time.sleep(.7)
Path('phase.tmp').write_text(json.dumps({'phase':'done'}))
os.replace('phase.tmp','phase.json')
print('worker finished',flush=True)
""", min_available_bytes=0)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['exit_code'], 0)
        self.assertFalse(result['resource_guard_triggered'])
        self.assertGreater(result['sample_count'], 3)
        self.assertGreater(result['fit_sample_count'], 1)
        self.assertGreater(result['fit_peak_tree_rss_bytes'], 24*1024**2)
        self.assertGreaterEqual(result['peak_tree_rss_bytes'], result['fit_peak_tree_rss_bytes'])
        self.assertGreaterEqual(result['fit_peak_minus_baseline_bytes'], 0)
        if os.name == 'nt':
            self.assertGreater(result['windows_main_peak_wset_bytes'], 0)
            self.assertGreaterEqual(result['windows_tree_peak_wset_sum_bytes'], result['peak_tree_rss_bytes'])
            self.assertGreater(result['windows_main_peak_pagefile_bytes'], 0)
        self.assertIn('worker finished', (self.root/'worker.log').read_text())
        with (self.root/'samples.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), result['sample_count'])
        self.assertIn('fit', {row['phase'] for row in rows})
        json.dumps(result, allow_nan=False)

    def test_nonzero_child_exit_is_failure_and_old_phase_not_reused(self):
        phase = self.root/'phase.json'
        phase.write_text('{"phase":"fit"}')
        old = time.time()-60
        os.utime(phase, (old, old))
        result = self.run_script("print('intentional failure',flush=True)\nraise SystemExit(7)\n", min_available_bytes=0)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['exit_code'], 7)
        self.assertIsNone(result['fit_peak_tree_rss_bytes'])
        self.assertEqual(result['fit_sample_count'], 0)

    def test_guard_stops_only_own_worker_and_descendant(self):
        result = self.run_script("""import subprocess,sys,time,json
from pathlib import Path
child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])
Path('child.json').write_text(json.dumps({'pid':child.pid}))
Path('phase.json').write_text(json.dumps({'phase':'fit'}))
time.sleep(60)
""", min_available_bytes=psutil.virtual_memory().total+1, low_memory_grace_seconds=.5)
        self.assertEqual(result['status'], 'resource_guard')
        self.assertTrue(result['resource_guard_triggered'])
        self.assertIsNotNone(result['exit_code'])
        child = json.loads((self.root/'child.json').read_text())['pid']
        self.assertIn(child, result['termination_details']['terminated_pids'])
        self.assertIn(result['pid'], result['termination_details']['terminated_pids'])
        self.assertEqual(result['remaining_child_pids'], [])
        self.assertTrue(psutil.pid_exists(os.getpid()))

    def test_pid_identity_mismatch_does_not_terminate_current_process(self):
        own = psutil.Process()
        result = _terminate_owned_tree({own.pid:own.create_time()+1}, own.pid)
        self.assertEqual(result['terminated_pids'], [])
        self.assertIn(own.pid, result['identity_missing_or_changed'])
        self.assertTrue(own.is_running())


if __name__ == '__main__':
    unittest.main(verbosity=2)
