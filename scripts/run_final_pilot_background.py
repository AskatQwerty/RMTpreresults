"""Guarded final 28 refits on the existing 20k cohort, followed by saved test predictions.

Default --check never fits. --run resumes verified pair checkpoints, starts one
fresh process per pair and limits all native thread pools to two. This launcher
does not start another search, bootstrap, SHAP or final five-seed repeats.
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
import sys
import time
import traceback
import uuid

import psutil

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('logistic_regression', 'lightgbm', 'xgboost', 'soft_voting', 'stacking', 'mlp', 'random_forest')
PAIRS = [(model, level) for model in MODELS for level in range(1, 5)]
THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    for attempt in range(30):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 29:
                raise
            time.sleep(.1)


def paths(args):
    prepared = Path(args.prepared).resolve()
    output = Path(args.output).resolve()
    if output == prepared or output in prepared.parents or prepared in output.parents:
        raise ValueError('Output must be separate from prepared data')
    old = ROOT / 'artifacts/training/pilot_20k_5config'
    if output == old or old in output.parents:
        raise ValueError('Old preliminary results must remain untouched')
    return prepared, output


def lock_state(path):
    if not Path(path).exists():
        return {'state': 'absent'}
    owner = read(path)
    if owner.get('host') != platform.node():
        return {'state': 'unknown_host', 'owner': owner}
    try:
        process = psutil.Process(owner['pid'])
        same = abs(process.create_time() - owner['create_time']) < 1e-6
        live = same and process.is_running() and process.status() != psutil.STATUS_ZOMBIE
        return {'state': 'alive' if live else 'stale', 'owner': owner}
    except psutil.NoSuchProcess:
        return {'state': 'stale', 'owner': owner}
    except psutil.AccessDenied:
        return {'state': 'unknown_access_denied', 'owner': owner}


def acquire(path):
    process = psutil.Process()
    owner = {'pid': process.pid, 'create_time': process.create_time(), 'host': platform.node(),
             'token': uuid.uuid4().hex, 'started_utc': utc()}
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(owner, stream)
    return owner


def release(path, owner):
    if Path(path).exists() and read(path).get('token') == owner['token']:
        Path(path).unlink()


def verified_pairs(output, context):
    done = []
    for model, level in PAIRS:
        stage = output / model / f'level_{level}'
        marker = stage / 'complete.json'
        if not marker.exists():
            continue
        record = read(marker)
        if record['fingerprint'] != context['fingerprint']:
            raise ValueError(f'Checkpoint fingerprint changed: {stage}')
        names = record['artifact_sha256']
        if not {'probabilities.parquet', 'metrics.json', 'fit_checkpoint/complete.json'} <= set(names):
            raise ValueError(f'Incomplete pair marker: {stage}')
        for name, checksum in names.items():
            path = (stage / name).resolve()
            if stage.resolve() not in path.parents or not path.is_file() or digest(path) != checksum:
                raise ValueError(f'Checkpoint SHA changed: {stage}/{name}')
        done.append((model, level))
    return done


def freeze(output):
    snapshot = output / 'source_snapshot'
    marker = snapshot / 'snapshot_manifest.json'
    if marker.exists():
        prior = read(marker)
        for name, checksum in prior['files'].items():
            if digest(snapshot / name) != checksum:
                raise ValueError(f'Frozen source changed: {name}')
            source = Path(__file__) if name == 'launcher_source.py' else ROOT / name
            if digest(source) != checksum:
                raise ValueError(f'Current source changed since launch: {name}; review before resume')
        return snapshot
    snapshot.mkdir(exist_ok=True)
    (snapshot / 'src').mkdir(exist_ok=True)
    # Evaluation is a separate process after all fits, and its adapter can be
    # completed while training runs. It is not a training dependency.
    sources = [p for p in sorted((ROOT / 'src').glob('*.py'))
               if p.name != 'final_pilot_evaluation.py']
    if not any(p.name == 'final_pilot_training.py' for p in sources):
        raise ValueError('Final training module is missing')
    files = {}
    for source in sources:
        relative = 'src/' + source.name
        shutil.copy2(source, snapshot / relative)
        files[relative] = digest(snapshot / relative)
    shutil.copy2(__file__, snapshot / 'launcher_source.py')
    files['launcher_source.py'] = digest(snapshot / 'launcher_source.py')
    write(marker, {'created_utc': utc(), 'files': files,
                   'scope': 'Frozen final-refit execution source; no imported teammate code is executed'})
    return snapshot


def keep_awake(enable):
    if os.name != 'nt':
        return None
    import ctypes
    return bool(ctypes.windll.kernel32.SetThreadExecutionState(
        0x80000000 | (0x00000001 if enable else 0)))


def progress(output, context, *, status_name, current=None, **extra):
    done = verified_pairs(output, context)
    value = {'status': status_name, 'updated_utc': utc(), 'completed_pairs': len(done),
             'expected_pairs': 28, 'remaining_pairs': 28-len(done),
             'completed': [{'model': m, 'level': l} for m, l in done],
             'current': current, 'new_cv_fits': 0, 'planned_final_refits': 28,
             'train_rows': 20000, 'test_rows': 61503, 'threads': 2, **extra}
    write(output / 'progress.json', value)
    return value


# No shell interpretation. Paths and identifiers travel as distinct argv values.
WORKER = r'''
import os,sys,json,platform
from pathlib import Path
from datetime import datetime, timezone
import psutil
snapshot, project, prepared, output, model, level, invocation = sys.argv[1:]
root=Path(output); inv=Path(invocation)
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='2'
runtime=inv/'runtime'; runtime.mkdir(exist_ok=True)
os.environ['TEMP']=os.environ['TMP']=str(runtime)
os.environ['RMT_PROJECT_ROOT']=project
os.environ['RMT_PHASE_PATH']=str(inv/'phase.json')
owner=json.loads((root/'supervisor.lock').read_text())
supervisor=psutil.Process(owner['pid'])
if owner['host']!=platform.node() or abs(supervisor.create_time()-owner['create_time'])>1e-6 or not supervisor.is_running():
    raise RuntimeError('Live matching supervisor required')
worker=psutil.Process()
(inv/'worker.json').write_text(json.dumps({'pid':worker.pid,'create_time':worker.create_time(),'started_utc':datetime.now(timezone.utc).isoformat(),'model':model,'level':int(level)}))
(inv/'phase.json').write_text(json.dumps({'phase':'pair_fit_and_test'}))
sys.path.insert(0,snapshot)
from src.final_pilot_training import run_final_pair
run_final_pair(prepared,output,model,int(level))
'''


def check(args):
    prepared, output = paths(args)
    sys.path.insert(0, str(ROOT))
    from src.final_pilot_training import initialize_final
    context = initialize_final(prepared, output)
    done = verified_pairs(output, context)
    print(json.dumps({'status': 'check_passed', 'fit_started': False, 'new_cv_fits': 0,
        'train_rows': 20000, 'test_rows': 61503, 'completed_pairs': len(done),
        'remaining_pairs': 28-len(done), 'output': str(output),
        'locks': {n: lock_state(output / n) for n in ('supervisor.lock', 'run.lock')},
        'available_ram_bytes': psutil.virtual_memory().available}, indent=2))


def run(args):
    prepared, output = paths(args)
    output.mkdir(parents=True, exist_ok=True)
    for name in ('supervisor.lock', 'run.lock'):
        state = lock_state(output / name)
        if state['state'] != 'absent':
            raise RuntimeError(f'{name}: {state["state"]}; inspect --status/--unlock-stale')
    owner = acquire(output / 'supervisor.lock')
    invocation = None
    awake = keep_awake(True)
    try:
        for name in THREAD_VARS:
            os.environ[name] = '2'
        os.environ['RMT_PROJECT_ROOT'] = str(ROOT)
        sys.path.insert(0, str(ROOT))
        from src.final_pilot_training import initialize_final, write_summary
        from src.resource_monitor import monitor_subprocess
        context = initialize_final(prepared, output)
        snapshot = freeze(output)
        done = verified_pairs(output, context)
        if len(done) == 28:
            progress(output, context, status_name='completed')
            write_summary(output)
            print('All 28 final pairs already complete; no fit performed.', flush=True)
            return
        invocation = output / 'launches' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        invocation.mkdir(parents=True)
        launch = {'status': 'running', 'started_utc': utc(), 'supervisor': owner,
                  'invocation': str(invocation), 'fingerprint': context['fingerprint'],
                  'threads': 2, 'sleep_request_accepted': awake, 'completed_before_launch': len(done)}
        write(output / 'current_launch.json', launch)
        write(invocation / 'hardware.json', {'platform': platform.platform(), 'host': platform.node(),
            'python': sys.version, 'physical_cpus': psutil.cpu_count(logical=False),
            'logical_cpus': psutil.cpu_count(), 'ram_bytes': psutil.virtual_memory().total,
            'available_ram_bytes': psutil.virtual_memory().available,
            'free_disk_bytes': shutil.disk_usage(ROOT).free})
        start = time.perf_counter()
        observations = []
        for model, level in PAIRS:
            if (model, level) in done:
                continue
            current = {'model': model, 'level': level, 'train_rows': 20000, 'test_rows': 61503}
            pair_invocation = invocation / f'{model}_level_{level}'
            pair_invocation.mkdir()
            progress(output, context, status_name='running', current=current,
                     invocation_started_utc=launch['started_utc'])
            write(output / 'current_launch.json', {**launch, 'current': current, 'pair_invocation': str(pair_invocation)})
            print(f'FINAL START {model} Level {level}; completed={len(done)}/28', flush=True)
            result = monitor_subprocess([sys.executable, '-u', '-c', WORKER, str(snapshot), str(ROOT),
                str(prepared), str(output), model, str(level), str(pair_invocation)], cwd=ROOT,
                log_path=pair_invocation / 'worker.log', phase_path=pair_invocation / 'phase.json',
                samples_path=pair_invocation / 'resource_samples.csv', sample_interval=1,
                min_available_bytes=512*1024**2, low_memory_grace_seconds=3)
            write(pair_invocation / 'monitor_result.json', result)
            observations.append({'model': model, 'level': level, **result})
            if result['status'] != 'completed':
                progress(output, context, status_name='interrupted', current=current, stop_reason=result['status'])
                write(output / 'current_launch.json', {**launch, 'status': result['status'], 'ended_utc': utc(),
                    'current': current, 'pair_invocation': str(pair_invocation), 'worker_pid': result['pid'],
                    'exit_code': result['exit_code'], 'elapsed_seconds': time.perf_counter()-start})
                raise RuntimeError(f'Pair stopped ({result["status"]}): {model} L{level}; see worker.log')
            done = verified_pairs(output, context)
            if (model, level) not in done:
                raise RuntimeError('Worker exited successfully without a verified pair checkpoint')
            write_summary(output)
            print(f'FINAL END {model} Level {level}; completed={len(done)}/28', flush=True)
        summary = progress(output, context, status_name='completed', elapsed_seconds=time.perf_counter()-start)
        write_summary(output)
        finish = {**launch, 'status': 'completed', 'ended_utc': utc(), 'completed_pairs': len(done),
                  'elapsed_seconds': time.perf_counter()-start, 'exit_code': 0}
        write(invocation / 'completion.json', {**finish, 'pair_observations': observations})
        write(output / 'current_launch.json', finish)
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException as exc:
        if invocation is not None:
            write(invocation / 'launcher_error.json', {'ended_utc': utc(), 'error': repr(exc),
                                                       'traceback': traceback.format_exc()})
            current = read(output / 'current_launch.json')
            if current.get('status') == 'running':
                write(output / 'current_launch.json', {**current, 'status': 'failed', 'ended_utc': utc(), 'error': repr(exc)})
        raise
    finally:
        keep_awake(False)
        release(output / 'supervisor.lock', owner)


def status(args):
    _, output = paths(args)
    context = read(output / 'experiment_manifest.json')
    done = verified_pairs(output, context)
    value = read(output / 'progress.json') if (output / 'progress.json').exists() else {}
    value.update(completed_pairs=len(done), expected_pairs=28, remaining_pairs=28-len(done),
                 locks={n: lock_state(output / n) for n in ('supervisor.lock', 'run.lock')})
    if len(done) == 28:
        value['status'] = 'completed'
    elif value.get('status') == 'running' and value['locks']['supervisor.lock']['state'] != 'alive':
        value['status'] = 'not_running_or_unverifiable'
    print(json.dumps(value, ensure_ascii=False, indent=2))


def unlock(args):
    _, output = paths(args)
    states = {name: lock_state(output / name) for name in ('supervisor.lock', 'run.lock')}
    if any(x['state'] not in ('absent', 'stale') for x in states.values()):
        raise RuntimeError('Cannot remove live or unverifiable locks')
    for name, state in states.items():
        if state['state'] == 'stale' and read(output / name) == state['owner']:
            (output / name).unlink()
    print(json.dumps(states, indent=2))


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    for option in ('check', 'run', 'status', 'unlock-stale'):
        modes.add_argument('--'+option, action='store_true')
    parser.add_argument('--prepared', default=str(ROOT / 'data/processed/full'))
    parser.add_argument('--output', default=str(ROOT / 'artifacts/training/pilot_20k_20config_final'))
    args = parser.parse_args()
    action = run if args.run else status if args.status else unlock if args.unlock_stale else check
    action(args)
