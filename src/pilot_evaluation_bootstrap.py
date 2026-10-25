"""Парный client bootstrap сохранённых прогнозов; здесь нет fit или predict.

Каждый повтор хранит общие частоты выбранных клиентов, метрики и состояние RNG.
AP всегда вычисляется sklearn.average_precision_score; integer sample_weight
математически воспроизводит повторение выбранных ID, включая совпавшие scores.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def atomic_replace(temporary, destination):
    for attempt in range(6):
        try:
            os.replace(temporary, destination)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.1 * (attempt + 1))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8', newline='')
    atomic_replace(temporary, path)


class QualityEvaluator:
    """Предварительная сортировка не обучает статистики или преобразования X."""
    def __init__(self, y, probabilities, level4_indices):
        self.y = np.asarray(y, dtype=np.int8)
        self.p = np.asarray(probabilities, dtype=np.float64)
        self.level4 = np.asarray(level4_indices, dtype=int)
        if self.p.ndim != 2 or self.p.shape[1] != len(self.y):
            raise ValueError('Expected probabilities[pair, client]')
        if not np.isfinite(self.p).all() or np.any((self.p < 0) | (self.p > 1)):
            raise ValueError('Invalid probabilities')
        self.orders = np.argsort(self.p, axis=1, kind='stable')
        self.sorted_p = np.take_along_axis(self.p, self.orders, axis=1)
        self.sorted_y = self.y[self.orders]
        q = np.clip(self.p[self.level4], np.finfo(float).eps,
                    1 - np.finfo(float).eps)
        self.log_terms = -(self.y * np.log(q) + (1 - self.y) * np.log1p(-q))
        self.brier_terms = (self.p[self.level4] - self.y)**2

    def weighted(self, weights):
        weights = np.asarray(weights, dtype=np.float64)
        if (weights.shape != self.y.shape or not np.isfinite(weights).all()
                or np.any(weights < 0) or not weights.sum()):
            raise ValueError('Invalid client weights')
        positive = float(weights @ self.y)
        both = 0 < positive < weights.sum()
        ap = np.full(len(self.p), np.nan)
        auc = np.full(len(self.level4), np.nan)
        if both:
            for i in range(len(self.p)):
                ap[i] = average_precision_score(self.sorted_y[i], self.sorted_p[i],
                    pos_label=1, sample_weight=weights[self.orders[i]])
            for j, i in enumerate(self.level4):
                auc[j] = roc_auc_score(self.sorted_y[i], self.sorted_p[i],
                    sample_weight=weights[self.orders[i]])
        return {'ap': ap, 'roc_auc_l4': auc,
                'log_loss_l4': self.log_terms @ weights / weights.sum(),
                'brier_l4': self.brier_terms @ weights / weights.sum()}


def _load_checkpoint(path, fingerprint, replicate, n):
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    if str(arrays['fingerprint']) != fingerprint or int(arrays['replicate']) != replicate:
        raise ValueError(f'Checkpoint belongs to another analysis: {path}')
    weights = arrays['weights']
    if weights.shape != (n,) or int(weights.sum()) != n or np.any(weights < 0):
        raise ValueError(f'Invalid bootstrap counts: {path}')
    return arrays


def bootstrap_run(y, probabilities, pairs, groups, directory, *, fingerprint,
                  n_bootstrap=1000, seed=42, max_new_replicates=None):
    """1000 ordinary paired draws; save EACH completed repetition immediately.

    A completed npz without its marker (interruption between two writes) is
    recovered after checking its provenance/RNG state, without recomputing it.
    max_new_replicates is solely a resumability-test hook; production uses None.
    """
    from .pilot_evaluation_decisions import DecisionEvaluator

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y, dtype=np.int8)
    level4 = [i for i, pair in enumerate(pairs) if pair['level'] == 4]
    probdict = {pairs[i]['model']: probabilities[i] for i in level4}
    quality = QualityEvaluator(y, probabilities, level4)
    decisions = DecisionEvaluator(y, probdict, groups)
    rng = np.random.default_rng(seed)
    n, new, existing = len(y), 0, 0
    started = time.perf_counter()
    values = []
    with threadpool_limits(limits=2):
        for replicate in range(n_bootstrap):
            path = directory / f'replicate_{replicate:04d}.npz'
            marker = path.with_suffix('.json')
            before = json.dumps(rng.bit_generator.state, sort_keys=True)
            if path.exists():
                if marker.exists():
                    record = json.loads(marker.read_text(encoding='utf-8'))
                    if record['sha256'] != sha256(path):
                        raise ValueError(f'Corrupt completed checkpoint: {path}')
                arrays = _load_checkpoint(path, fingerprint, replicate, n)
                if str(arrays['rng_before']) != before:
                    raise ValueError(f'Bootstrap RNG chain changed: {path}')
                rng.bit_generator.state = json.loads(str(arrays['rng_after']))
                if not marker.exists():
                    atomic_json(marker, {'replicate': replicate, 'sha256': sha256(path),
                        'recovered_without_recomputation': True})
                existing += 1
            else:
                if marker.exists():
                    raise ValueError(f'Checkpoint data missing: {path}')
                if max_new_replicates is not None and new >= max_new_replicates:
                    break
                begin = time.perf_counter()
                selected_positions = rng.integers(0, n, size=n, dtype=np.int64)
                counts = np.bincount(selected_positions, minlength=n)
                if counts.max() > np.iinfo(np.uint16).max:
                    raise ValueError('Unexpected bootstrap multiplicity')
                arrays = quality.weighted(counts)
                arrays.update({f'decision_{key}': value
                               for key, value in decisions.weighted(counts).items()})
                arrays.update({'weights': counts.astype(np.uint16),
                    'replicate': np.array(replicate), 'fingerprint': np.array(fingerprint),
                    'rng_before': np.array(before),
                    'rng_after': np.array(json.dumps(rng.bit_generator.state, sort_keys=True)),
                    'elapsed_seconds': np.array(time.perf_counter() - begin)})
                temporary = path.with_name(path.name + '.tmp')
                with temporary.open('wb') as stream:
                    np.savez_compressed(stream, **arrays)
                atomic_replace(temporary, path)
                atomic_json(marker, {'replicate': replicate, 'sha256': sha256(path),
                                     'recovered_without_recomputation': False})
                new += 1
            values.append({key: value for key, value in arrays.items()
                           if key in {'ap', 'roc_auc_l4', 'log_loss_l4', 'brier_l4'}
                           or key.startswith('decision_')})
            if (replicate + 1) % 25 == 0 or replicate == n_bootstrap - 1:
                print(f'Paired bootstrap {replicate + 1}/{n_bootstrap}; '
                      f'new={new}, reused={existing}, elapsed={time.perf_counter()-started:.1f}s',
                      flush=True)
    stacked = {key: np.stack([value[key] for value in values]) for key in values[0]} if values else {}
    state = {'requested': n_bootstrap, 'completed': len(values), 'new': new,
             'reused': existing, 'elapsed_seconds_this_invocation': time.perf_counter()-started,
             'seed': seed, 'rng': 'numpy.default_rng / PCG64', 'unit': 'test client',
             'paired': True, 'stratified': False, 'same_counts_for_all_pairs': True,
             'checkpoint_per_replicate': True, 'fit_calls': 0, 'predict_calls': 0}
    atomic_json(directory / 'progress.json', state)
    return stacked, state


def percentile_interval(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return np.nan, np.nan, 0
    low, high = np.quantile(finite, [.025, .975], method='linear')
    return float(low), float(high), int(len(finite))
