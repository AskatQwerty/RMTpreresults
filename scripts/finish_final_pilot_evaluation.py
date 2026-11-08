"""Continue the authorized workflow once all 28 final pairs are complete.

No fitting. Waits for the existing training supervisor, validates its completed
checkpoint set, then computes the separate final RQ1/RQ2 analysis exactly once
(resuming existing bootstrap checkpoints if explicitly rerun after failure).
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / 'artifacts/training/pilot_20k_20config_final'
OUT = ROOT / 'reports/pilot_20k_20config_final/evaluation'
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '2'
sys.path.insert(0, str(ROOT))

from scripts.run_final_pilot_background import keep_awake, verified_pairs, lock_state
from src.model_training import atomic_json, read_json, sha256
from src.final_pilot_evaluation import run_final_evaluation, SOURCE_FILES


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / 'workflow.lock'
    # A second writer is never started automatically.
    with lock.open('x', encoding='utf-8') as stream:
        import psutil
        json.dump({'pid': os.getpid(), 'create_time': psutil.Process().create_time()}, stream)
    code = {name: sha256(ROOT / 'src' / name) for name in SOURCE_FILES}
    context = read_json(TRAIN / 'experiment_manifest.json')
    status = dict(status='waiting_for_training', started_utc=datetime.now(timezone.utc).isoformat(),
                  pid=os.getpid(), training_fingerprint=context['fingerprint'], analysis_sha256=code)
    atomic_json(OUT / 'workflow_status.json', status)
    keep_awake(True)
    try:
        while True:
            launch = read_json(TRAIN / 'current_launch.json')
            if launch['status'] == 'completed':
                if len(verified_pairs(TRAIN, context)) != 28:
                    raise RuntimeError('Final training completed without28validpaircheckpoints')
                break
            if launch['status'] != 'running' or lock_state(TRAIN / 'supervisor.lock')['state'] != 'alive':
                raise RuntimeError('Training stopped before completion; evaluation was not started')
            time.sleep(10)
        if code != {name: sha256(ROOT / 'src' / name) for name in SOURCE_FILES}:
            raise RuntimeError('Evaluation code changed while waiting; review required')
        atomic_json(OUT / 'workflow_status.json', {**status, 'status': 'evaluating',
                    'evaluation_started_utc': datetime.now(timezone.utc).isoformat()})
        run_final_evaluation(ROOT / 'data/processed/full', TRAIN,
                             ROOT / 'reports/pilot_20k_5config', OUT)
        atomic_json(OUT / 'workflow_status.json', {**status, 'status': 'completed_pending_visual_review',
                    'ended_utc': datetime.now(timezone.utc).isoformat()})
    except BaseException as exc:
        atomic_json(OUT / 'workflow_status.json', {**status, 'status': 'failed',
                    'ended_utc': datetime.now(timezone.utc).isoformat(), 'error': repr(exc),
                    'traceback': traceback.format_exc()})
        raise
    finally:
        keep_awake(False)
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
