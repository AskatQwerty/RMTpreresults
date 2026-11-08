"""Run/resume the isolated, predeclared MLP repair experiment; default no-fit check."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'artifacts/training/mlp_repair_20k_v1'
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '2'
sys.path.insert(0, str(ROOT))
from scripts.run_final_pilot_background import lock_state, acquire, release, keep_awake
from src.model_training import atomic_json, read_json, sha256
from src.mlp_repair_experiment import initialize, CODE_FILES, get_records


def utc():
    return datetime.now(timezone.utc).isoformat()


def freeze(plan):
    snapshot = OUTPUT / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    (snapshot / 'src').mkdir(exist_ok=True)
    names = ['src/' + name for name in ('__init__.py', *CODE_FILES)]
    names += ['scripts/run_mlp_repair.py']
    files = {}
    for name in names:
        src = ROOT / name
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha256(target) != sha256(src):
                raise RuntimeError('Frozen source changed: ' + name)
        else:
            shutil.copy2(src, target)
        files[name] = sha256(target)
    record = dict(plan_fingerprint=plan['fingerprint'], files=files)
    marker = snapshot / 'manifest.json'
    if marker.exists() and read_json(marker) != record:
        raise RuntimeError('Frozen snapshot contract changed')
    atomic_json(marker, record)
    return snapshot


WORKER = r'''
import json,os,sys,platform
from pathlib import Path
import psutil
snapshot,project,invocation=sys.argv[1:]
inv=Path(invocation);out=Path(project)/'artifacts/training/mlp_repair_20k_v1'
owner=json.loads((out/'supervisor.lock').read_text())
p=psutil.Process(owner['pid'])
if owner['host']!=platform.node() or abs(p.create_time()-owner['create_time'])>1e-6 or not p.is_running():
    raise RuntimeError('Matching live supervisor required')
runtime=inv/'runtime';runtime.mkdir(exist_ok=True)
os.environ['TEMP']=os.environ['TMP']=str(runtime)
os.environ['RMT_PHASE_PATH']=str(inv/'phase.json')
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='2'
sys.path.insert(0,snapshot)
from src.model_training import atomic_json
atomic_json(inv/'worker.json',{'pid':os.getpid(),'create_time':psutil.Process().create_time()})
from src.mlp_repair_experiment import run
result=run(project)
atomic_json(inv/'worker_result.json',result)
'''


def main(mode):
    if mode == 'check':
        plan = initialize(ROOT)
        print(json.dumps(dict(status='check_passed', fit_calls=0, train_rows=plan['train_rows'],
            test_rows=plan['test_rows'], variants=plan['variants'], expected_outer_cv=48,
            max_final_procedures=4, fingerprint=plan['fingerprint']), indent=2))
        return
    if mode == 'status':
        plan = read_json(OUTPUT / 'plan.json')
        result = read_json(OUTPUT / 'progress.json') if (OUTPUT / 'progress.json').exists() else {}
        result.update(verified_cv=len(get_records(OUTPUT, plan)), lock=lock_state(OUTPUT / 'supervisor.lock'))
        print(json.dumps(result, indent=2))
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if lock_state(OUTPUT / 'supervisor.lock')['state'] != 'absent':
        raise RuntimeError('Existing supervisor lock; never start a second writer')
    owner = acquire(OUTPUT / 'supervisor.lock')
    keep_awake(True)
    launch = None
    try:
        plan = initialize(ROOT)
        snapshot = freeze(plan)
        invocation = OUTPUT / 'launches' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        invocation.mkdir(parents=True)
        atomic_json(invocation / 'phase.json', {'phase': 'initialization'})
        launch = dict(status='running', started_utc=utc(), supervisor=owner,
            invocation=str(invocation), plan_fingerprint=plan['fingerprint'], threads=2)
        atomic_json(OUTPUT / 'current_launch.json', launch)
        from src.resource_monitor import monitor_subprocess
        result = monitor_subprocess([sys.executable, '-u', '-c', WORKER, str(snapshot), str(ROOT), str(invocation)],
            cwd=ROOT, log_path=invocation / 'worker.log', phase_path=invocation / 'phase.json',
            samples_path=invocation / 'resource_samples.csv', sample_interval=1,
            min_available_bytes=512*1024**2, low_memory_grace_seconds=3)
        atomic_json(invocation / 'monitor_result.json', result)
        if result['status'] != 'completed':
            raise RuntimeError(f'Worker {result["status"]}; see {invocation / "worker.log"}')
        from src.mlp_repair_reporting import export_mlp_repair_report
        export_mlp_repair_report(OUTPUT, ROOT / 'reports/mlp_repair_20k_v1')
        atomic_json(OUTPUT / 'current_launch.json', {**launch, 'status': 'completed', 'ended_utc': utc(),
            'worker_pid': result['pid'], 'elapsed_seconds': result['elapsed_seconds'], 'exit_code': result['exit_code']})
        print(json.dumps(read_json(OUTPUT / 'summary.json'), indent=2))
    except BaseException as exc:
        if launch is not None:
            atomic_json(OUTPUT / 'current_launch.json', {**launch, 'status': 'failed', 'ended_utc': utc(),
                'error': repr(exc), 'traceback': traceback.format_exc()})
        raise
    finally:
        keep_awake(False)
        release(OUTPUT / 'supervisor.lock', owner)


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['check', 'run', 'status'], default='check')
    main(parser.parse_args().mode)
