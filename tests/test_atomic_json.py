"""Атомарная JSON-запись при краткой Windows-блокировке читателем."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.model_training import atomic_json, read_json


class AtomicJsonTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='rmt_atomic_json_')
        self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'phase.json'
        atomic_json(self.path,{'phase':'fit'})
        self.original=self.path.read_bytes()

    @staticmethod
    def permission_error(winerror):
        error=PermissionError('Windows reader holds target')
        error.winerror=winerror
        return error

    def test_transient_windows_lock_retries_without_removing_old_target(self):
        from src.model_training import os
        replace=os.replace
        for code in (5,32):
            with self.subTest(winerror=code):
                atomic_json(self.path,{'phase':'fit'})
                calls=[]
                def replace_after_lock(source,target):
                    calls.append((source,target))
                    self.assertEqual(self.path.read_bytes(),self.original)
                    if len(calls)==1:
                        raise self.permission_error(code)
                    replace(source,target)
                with patch('src.model_training.os.replace',side_effect=replace_after_lock), \
                     patch('src.model_training.time.sleep') as sleep:
                    atomic_json(self.path,{'phase':'serialize'})
                self.assertEqual(len(calls),2)
                sleep.assert_called_once()
                self.assertEqual(read_json(self.path),{'phase':'serialize'})
                self.assertFalse(self.path.with_name('phase.json.tmp').exists())

    def test_persistent_windows_lock_stops_at_deadline_and_preserves_old_target(self):
        error=self.permission_error(5)
        with patch('src.model_training.os.replace',side_effect=error) as replace, \
             patch('src.model_training.time.monotonic',side_effect=[10.0,10.0,12.0]), \
             patch('src.model_training.time.sleep') as sleep:
            with self.assertRaises(PermissionError) as caught:
                atomic_json(self.path,{'phase':'serialize'})
        self.assertIs(caught.exception,error)
        self.assertEqual(replace.call_count,2)
        sleep.assert_called_once_with(0.05)
        self.assertEqual(self.path.read_bytes(),self.original)
        self.assertEqual(read_json(self.path.with_name('phase.json.tmp')),{'phase':'serialize'})

    def test_other_permission_error_is_not_retried(self):
        error=self.permission_error(87)
        with patch('src.model_training.os.replace',side_effect=error) as replace, \
             patch('src.model_training.time.sleep') as sleep:
            with self.assertRaises(PermissionError) as caught:
                atomic_json(self.path,{'phase':'serialize'})
        self.assertIs(caught.exception,error)
        replace.assert_called_once()
        sleep.assert_not_called()
        self.assertEqual(self.path.read_bytes(),self.original)


if __name__=='__main__':
    unittest.main(verbosity=2)
