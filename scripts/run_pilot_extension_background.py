"""Run the next five fixed candidates per pair on the unchanged pilot cohort.

Default check performs no fit. Explicit --mode monitor --confirm-next-420 starts
only candidates 6..10, 11..15 or 16..20 (the requested batch starts at 6).
No final refit, test evaluation or five-seed repeats are performed here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import uuid

import psutil

ROOT = Path(__file__).resolve().parents[1]


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.1)


def settings(args):
    prepared = (ROOT / 'data/processed/full').resolve()
    pilot = (ROOT / 'artifacts/training/pilot_20k_5config').resolve()
    output = (ROOT / 'artifacts/training/pilot_20k_20config_extension').resolve()
    candidates = tuple(range(args.candidate_start, args.candidate_start + 5))
    if candidates[0] not in (6, 11, 16) or candidates[-1] > 20:
        raise ValueError('Only consecutive five-candidate batches after the existing first five')
    return prepared, pilot, output, candidates


def check(args):
    prepared, pilot, output, candidates = settings(args)
    sys.path.insert(0, str(ROOT))
    from src.pilot_extension_plan import build_extension_plan
    plan = build_extension_plan(prepared, pilot)
    result = {'checked_utc': utc(), 'fit_started': False,
              'train_rows': 20000, 'candidate_numbers': candidates,
              'additional_cv_fits': 420, 'refit': False, 'test_evaluations': 0,
              'model_count': len(plan['configurations']),
              'planned_configurations_per_model': {
                  model: len(items) for model, items in plan['configurations'].items()},
              'output': str(output), 'available_ram_bytes': psutil.virtual_memory().available,
              'free_disk_bytes': shutil.disk_usage(ROOT).free,
              'pending_lock': (output / 'run.lock').exists() or (output / 'supervisor.lock').exists()}
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def freeze_source(output):
    snapshot = output / 'source_snapshot'
    marker = snapshot / 'snapshot_manifest.json'
    if marker.exists():
        record = json.loads(marker.read_text(encoding='utf-8'))
        for relative, checksum in record['files'].items():
            if digest(snapshot / relative) != checksum:
                raise RuntimeError(f'Frozen source changed: {relative}')
        if digest(Path(__file__)) != record['files']['launcher_source.py']:
            raise RuntimeError('Launcher changed since snapshot; review before resuming')
        return snapshot
    snapshot.mkdir(parents=True, exist_ok=True)
    files = {}
    for source in sorted((ROOT / 'src').glob('*.py')):
        target = snapshot / 'src' / source.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(source, target)
        files[target.relative_to(snapshot).as_posix()] = digest(target)
    shutil.copy2(Path(__file__), snapshot / 'launcher_source.py')
    files['launcher_source.py'] = digest(snapshot / 'launcher_source.py')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
    source_state = subprocess.check_output(
        ['git', 'status', '--porcelain', '--', 'src', 'scripts'], cwd=ROOT).decode()
    write_json(marker, {'created_utc': utc(), 'base_commit': revision, 'files': files,
                       'working_tree_at_launch': source_state,
                       'provenance': 'SHA256 identifies executed source, including new extension modules.'})
    return snapshot


def keep_awake(enable):
    if os.name != 'nt':
        return None
    import ctypes
    return bool(ctypes.windll.kernel32.SetThreadExecutionState(
        0x80000000 | (0x00000001 if enable else 0)))


def worker(args):
    prepared, pilot, output, candidates = settings(args)
    invocation = Path(args.invocation).resolve()
    if invocation.parent != output / 'launches':
        raise ValueError('Unexpected invocation directory')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = '2'
    runtime = invocation / 'runtime'
    runtime.mkdir(exist_ok=True)
    os.environ['TEMP'] = os.environ['TMP'] = str(runtime)
    sys.path.insert(0, str(output / 'source_snapshot'))
    from src.pilot_extension import run_extension, write_extension_report
    process = psutil.Process()
    awake = keep_awake(True)
    write_json(invocation / 'worker.json', {'pid': process.pid,
        'create_time': process.create_time(), 'started_utc': utc(),
        'candidate_numbers': candidates, 'train_rows': 20000,
        'threads': 2, 'sleep_request_accepted': awake})
    write_json(invocation / 'phase.json', {'phase': 'pilot_extension_cv', 'utc': utc()})
    try:
        result = run_extension(prepared, pilot, output, threads=2,
            candidate_numbers=candidates, report_dir=ROOT / 'reports/training/pilot_20k_extension')
        write_json(invocation / 'worker_result.json', {'status': 'completed',
            'ended_utc': utc(), 'result': result})
    except BaseException as exc:
        write_json(invocation / 'worker_result.json', {'status': 'failed',
            'ended_utc': utc(), 'error': f'{type(exc).__name__}: {exc}',
            'traceback': traceback.format_exc()})
        raise
    finally:
        keep_awake(False)
        try:
            write_extension_report(output, ROOT / 'reports/training/pilot_20k_extension')
        except Exception as exc:
            write_json(invocation / 'report_error.json', {'error': repr(exc)})


def monitor(args):
    if not args.confirm_next_420:
        raise ValueError('Explicit --confirm-next-420 required')
    prepared, pilot, output, candidates = settings(args)
    if (output / 'run.lock').exists():
        raise RuntimeError('Existing run.lock; inspect owner before resuming')
    output.mkdir(parents=True, exist_ok=True)
    lock = output / 'supervisor.lock'
    handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    process = psutil.Process()
    invocation = None
    try:
        os.write(handle, json.dumps({'pid': process.pid, 'create_time': process.create_time(),
                                    'started_utc': utc()}).encode('utf-8'))
        os.close(handle)
        invocation = output / 'launches' / (
            datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
        invocation.mkdir(parents=True)
        snapshot = freeze_source(output)
        shutil.copy2(Path(__file__), invocation / 'launcher_source.py')
        write_json(invocation / 'hardware.json', {'recorded_utc': utc(),
            'platform': platform.platform(), 'processor': platform.processor(),
            'python': sys.version, 'physical_cpus': psutil.cpu_count(logical=False),
            'logical_cpus': psutil.cpu_count(), 'ram_bytes': psutil.virtual_memory().total,
            'available_ram_bytes': psutil.virtual_memory().available,
            'free_disk_bytes': shutil.disk_usage(ROOT).free})
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--mode', 'worker',
                   '--candidate-start', str(candidates[0]), '--invocation', str(invocation)]
        start = {'status': 'running', 'started_utc': utc(),
            'supervisor_pid': process.pid, 'supervisor_create_time': process.create_time(),
            'invocation': str(invocation), 'output': str(output),
            'mode': 'pilot_20k_fixed_candidates_extension', 'train_rows': 20000,
            'candidate_numbers': candidates, 'new_cv_fits_authorized': 420,
            'existing_pilot_cv_fits': 420, 'refit': False, 'test_evaluations': 0,
            'threads': 2, 'command': command,
            'launcher_sha256': digest(snapshot / 'launcher_source.py')}
        write_json(invocation / 'launch.json', start)
        write_json(output / 'current_launch.json', start)
        sys.path.insert(0, str(snapshot))
        from src.resource_monitor import monitor_subprocess
        result = monitor_subprocess(command, cwd=ROOT,
            log_path=invocation / 'worker.log', phase_path=invocation / 'phase.json',
            samples_path=invocation / 'resource_samples.csv', sample_interval=1,
            min_available_bytes=512 * 1024**2, low_memory_grace_seconds=3)
        write_json(invocation / 'monitor_result.json', result)
        write_json(output / 'current_launch.json', {**start, 'status': result['status'],
            'ended_utc': utc(), 'worker_pid': result['pid'],
            'worker_create_time': result['process_create_time'],
            'exit_code': result['exit_code'], 'peak_rss_gib': result['peak_tree_rss_gib']})
        if result['status'] != 'completed':
            progress_file = output / 'progress.json'
            if progress_file.exists():
                progress = json.loads(progress_file.read_text(encoding='utf-8'))
                if progress.get('status') != 'completed':
                    progress.update(status='interrupted', stop_reason=result['status'],
                                    monitor_ended_utc=utc())
                    write_json(progress_file, progress)
                    write_json(output / 'batches' /
                               f'configurations_{candidates[0]:02d}_{candidates[-1]:02d}' /
                               'progress.json', progress)
            raise SystemExit(1)
    except BaseException as exc:
        if invocation is not None:
            status_file = output / 'current_launch.json'
            current = json.loads(status_file.read_text(encoding='utf-8')) if status_file.exists() else {}
            if current.get('status', 'running') == 'running':
                failed = {**current, 'status': 'launcher_failed', 'ended_utc': utc(),
                          'error': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc()}
                write_json(invocation / 'launcher_error.json', failed)
                write_json(status_file, failed)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['check', 'monitor', 'worker'], default='check')
    parser.add_argument('--candidate-start', type=int, choices=[6, 11, 16], default=6)
    parser.add_argument('--confirm-next-420', action='store_true')
    parser.add_argument('--invocation')
    args = parser.parse_args()
    {'check': check, 'monitor': monitor, 'worker': worker}[args.mode](args)
