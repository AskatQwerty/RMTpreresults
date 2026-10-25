"""Наблюдение за собственным worker-процессом ресурсного пилота.

RSS дерева — сумма рабочих наборов живых процессов; общие страницы могут быть
учтены несколько раз. Это не USS. Пик RSS измеряется с заданным интервалом и
является наблюдаемым пиком, а Windows peak_wset — отдельным high-water mark.
Функция не останавливает посторонние процессы и не задаёт таймаут обучения.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

import psutil


GIB = 1024**3


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _same_process(pid, create_time):
    """PID недостаточен: проверка времени создания защищает от его переиспользования."""
    try:
        process = psutil.Process(pid)
        if abs(process.create_time() - create_time) > 1e-6:
            return None
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return None
        return process
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def _terminate_owned_tree(owned, root_pid):
    """Завершить только процессы, чьи (PID, create_time) ранее наблюдал монитор."""
    details = {'terminated_pids': [], 'killed_pids': [], 'identity_missing_or_changed': [],
               'termination_errors': []}
    # Потомки сначала, корневой worker последним; чужие PID сюда не добавляются.
    order = [pid for pid in owned if pid != root_pid] + ([root_pid] if root_pid in owned else [])
    targets = []
    for pid in order:
        process = _same_process(pid, owned[pid])
        if process is None:
            details['identity_missing_or_changed'].append(pid)
            continue
        try:
            process.terminate()
            details['terminated_pids'].append(pid)
            targets.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
            details['termination_errors'].append({'pid': pid, 'error': type(exc).__name__})
    _, alive = psutil.wait_procs(targets, timeout=2)
    for candidate in alive:
        process = _same_process(candidate.pid, owned[candidate.pid])
        if process is None:
            continue
        try:
            process.kill()
            details['killed_pids'].append(process.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
            details['termination_errors'].append({'pid': candidate.pid, 'error': type(exc).__name__})
    psutil.wait_procs(alive, timeout=2)
    return details


def _phase(path, start_wall_ns, previous):
    try:
        # Старый phase.json от прежней попытки не должен пометить импорт как fit.
        if path.stat().st_mtime_ns < start_wall_ns:
            return previous
        value = json.loads(path.read_text(encoding='utf-8')).get('phase')
        return str(value) if value is not None else previous
    except (OSError, ValueError, TypeError):
        return previous


def monitor_subprocess(command, cwd, log_path, phase_path, samples_path,
                       sample_interval=.1, min_available_bytes=512*1024**2,
                       low_memory_grace_seconds=3):
    """Запустить и измерить свой worker; вернуть JSON-совместимую сводку.

    command — список аргументов, shell=False. Windows worker скрыт. stdout/stderr
    направляются в log_path. JSON phase_path с phase='fit' выделяет фазу обучения;
    CSV samples_path записывается по мере работы. При доступной системной RAM ниже
    порога непрерывно в течение grace останавливается только этот worker и ранее
    наблюдавшиеся потомки с совпавшим временем создания. Лимита времени fit нет.

    Завершение корневого worker завершает наблюдение. remaining_child_pids покажет
    неожиданных живых потомков; при штатном завершении они не убиваются молча.
    Измерения, которые не удалось наблюдать, имеют значение None, а не ноль.
    """
    if isinstance(command, (str, bytes)) or not command:
        raise TypeError('command должен быть непустым списком аргументов, без shell')
    if sample_interval <= 0 or low_memory_grace_seconds < 0 or min_available_bytes < 0:
        raise ValueError('Некорректные интервалы/порог памяти')
    log_path, phase_path, samples_path = map(Path, (log_path, phase_path, samples_path))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    started_utc = _utc()
    started_wall_ns = time.time_ns()
    started = time.perf_counter()
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    fields = ['utc', 'elapsed_seconds', 'phase', 'main_rss_bytes', 'tree_rss_bytes',
              'tree_vms_bytes', 'process_count', 'system_available_bytes',
              'main_peak_wset_bytes', 'tree_peak_wset_sum_bytes',
              'main_pagefile_bytes', 'main_peak_pagefile_bytes',
              'cpu_seconds_observed', 'low_memory_seconds']
    owned, cpu_by_identity = {}, {}
    peak_tree_rss = baseline_tree_rss = None
    fit_peak_tree_rss = fit_baseline_tree_rss = None
    minimum_available = None
    windows_peak_wset = windows_tree_peak_sum = windows_peak_pagefile = None
    sample_count = fit_sample_count = 0
    phase = 'starting'
    low_since = None
    last_progress = started
    termination_details = None
    monitor_error = None
    resource_guard_triggered = False
    status = None
    process_create_time = None
    fit_started_elapsed = fit_last_elapsed = None
    root_process = None
    observed_pid = None
    final_exit_code = None

    with log_path.open('wb', buffering=0) as log_stream, samples_path.open('w', newline='', encoding='utf-8') as sample_stream:
        writer = csv.DictWriter(sample_stream, fieldnames=fields)
        writer.writeheader()
        worker = subprocess.Popen([str(arg) for arg in command], cwd=str(cwd),
            stdout=log_stream, stderr=subprocess.STDOUT, shell=False,
            creationflags=creationflags, close_fds=True)
        observed_pid = worker.pid
        try:
            root_process = psutil.Process(worker.pid)
            process_create_time = root_process.create_time()
            owned[worker.pid] = process_create_time
        except psutil.NoSuchProcess:
            root_process = None
        try:
            while True:
                now = time.perf_counter()
                elapsed = now-started
                phase = _phase(phase_path, started_wall_ns, phase)
                current = []
                if root_process is not None and _same_process(worker.pid, process_create_time) is not None:
                    current.append(root_process)
                    try:
                        current.extend(root_process.children(recursive=True))
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                # Сохраняем известные PID, включая уже осиротевшего потомка этого worker.
                for process in current:
                    try:
                        created = process.create_time()
                        if process.pid not in owned or owned[process.pid] == created:
                            owned[process.pid] = created
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                active = []
                for pid, created in list(owned.items()):
                    process = _same_process(pid, created)
                    if process is not None:
                        active.append(process)
                main_rss = main_peak_wset = main_pagefile = main_peak_pagefile = None
                tree_rss = tree_vms = 0
                individual_highwaters = []
                measured_count = 0
                for process in active:
                    try:
                        memory = process.memory_info()
                        cpu = process.cpu_times()
                        cpu_by_identity[(process.pid, owned[process.pid])] = cpu.user+cpu.system
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                    measured_count += 1
                    tree_rss += memory.rss
                    tree_vms += memory.vms
                    highwater = getattr(memory, 'peak_wset', None)
                    if highwater is not None:
                        individual_highwaters.append(highwater)
                    if process.pid == worker.pid:
                        main_rss = memory.rss
                        main_peak_wset = highwater
                        main_pagefile = getattr(memory, 'pagefile', None)
                        main_peak_pagefile = getattr(memory, 'peak_pagefile', None)
                available = int(psutil.virtual_memory().available)
                minimum_available = available if minimum_available is None else min(minimum_available, available)
                tree_peak_sum = sum(individual_highwaters) if individual_highwaters else None
                cpu_seconds = sum(cpu_by_identity.values())
                if measured_count:
                    baseline_tree_rss = tree_rss if baseline_tree_rss is None else baseline_tree_rss
                    peak_tree_rss = tree_rss if peak_tree_rss is None else max(peak_tree_rss, tree_rss)
                    if phase == 'fit':
                        fit_sample_count += 1
                        fit_baseline_tree_rss = tree_rss if fit_baseline_tree_rss is None else fit_baseline_tree_rss
                        fit_peak_tree_rss = tree_rss if fit_peak_tree_rss is None else max(fit_peak_tree_rss, tree_rss)
                        fit_started_elapsed = elapsed if fit_started_elapsed is None else fit_started_elapsed
                        fit_last_elapsed = elapsed
                if main_peak_wset is not None:
                    windows_peak_wset = max(windows_peak_wset or 0, main_peak_wset)
                if tree_peak_sum is not None:
                    windows_tree_peak_sum = max(windows_tree_peak_sum or 0, tree_peak_sum)
                if main_peak_pagefile is not None:
                    windows_peak_pagefile = max(windows_peak_pagefile or 0, main_peak_pagefile)
                if available < min_available_bytes:
                    low_since = now if low_since is None else low_since
                else:
                    low_since = None
                low_seconds = 0 if low_since is None else now-low_since
                writer.writerow({'utc': _utc(), 'elapsed_seconds': round(elapsed, 6), 'phase': phase,
                    'main_rss_bytes': main_rss, 'tree_rss_bytes': tree_rss if measured_count else None,
                    'tree_vms_bytes': tree_vms if measured_count else None, 'process_count': measured_count,
                    'system_available_bytes': available, 'main_peak_wset_bytes': main_peak_wset,
                    'tree_peak_wset_sum_bytes': tree_peak_sum, 'main_pagefile_bytes': main_pagefile,
                    'main_peak_pagefile_bytes': main_peak_pagefile,
                    'cpu_seconds_observed': round(cpu_seconds, 6), 'low_memory_seconds': round(low_seconds, 6)})
                sample_stream.flush()
                sample_count += 1
                exit_code = worker.poll()
                if exit_code is not None:
                    final_exit_code = exit_code
                    status = 'completed' if exit_code == 0 else 'failed'
                    break
                if low_since is not None and low_seconds >= low_memory_grace_seconds:
                    resource_guard_triggered = True
                    print(f'[resource] low RAM guard: available={available/GIB:.3f} GiB; stopping own worker PID {worker.pid}', flush=True)
                    termination_details = _terminate_owned_tree(owned, worker.pid)
                    status = 'resource_guard'
                    break
                if now-last_progress >= 60:
                    peak_text = '?' if peak_tree_rss is None else f'{peak_tree_rss/GIB:.3f}'
                    print(f'[resource] phase={phase} elapsed={elapsed:.1f}s RSS={tree_rss/GIB:.3f} GiB peak={peak_text} GiB available={available/GIB:.3f} GiB CPU={cpu_seconds:.1f}s', flush=True)
                    last_progress = now
                # Интервал относится к началу соседних выборок, а не к паузе
                # после уже потраченного на psutil времени.
                time.sleep(max(0, sample_interval-(time.perf_counter()-now)))
        except KeyboardInterrupt:
            status = 'interrupted'
            termination_details = _terminate_owned_tree(owned, worker.pid)
        except Exception as exc:
            status = 'monitor_error'
            monitor_error = {'type': type(exc).__name__, 'message': str(exc)}
            termination_details = _terminate_owned_tree(owned, worker.pid)
        finally:
            try:
                final_exit_code = worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Только после ошибки/защитного завершения; не общий таймаут fit.
                if status in {'resource_guard', 'interrupted', 'monitor_error'}:
                    more = _terminate_owned_tree(owned, worker.pid)
                    termination_details = {'initial': termination_details, 'retry': more}
                    final_exit_code = worker.poll()

    remaining_children = [pid for pid, created in owned.items()
                          if pid != observed_pid and _same_process(pid, created) is not None]
    return {
        'status': status, 'exit_code': final_exit_code, 'pid': observed_pid,
        'process_create_time': process_create_time, 'started_utc': started_utc, 'ended_utc': _utc(),
        'elapsed_seconds': time.perf_counter()-started, 'sample_interval_seconds': sample_interval,
        'sample_count': sample_count, 'fit_sample_count': fit_sample_count,
        'baseline_tree_rss_bytes': baseline_tree_rss,
        'peak_tree_rss_bytes': peak_tree_rss,
        'peak_tree_rss_gib': None if peak_tree_rss is None else peak_tree_rss/GIB,
        'fit_baseline_tree_rss_bytes': fit_baseline_tree_rss,
        'fit_peak_tree_rss_bytes': fit_peak_tree_rss,
        'fit_peak_tree_rss_gib': None if fit_peak_tree_rss is None else fit_peak_tree_rss/GIB,
        'fit_peak_minus_baseline_bytes': None if fit_peak_tree_rss is None else fit_peak_tree_rss-fit_baseline_tree_rss,
        'fit_first_sample_elapsed_seconds': fit_started_elapsed,
        'fit_last_sample_elapsed_seconds': fit_last_elapsed,
        'min_available_bytes': minimum_available,
        'windows_main_peak_wset_bytes': windows_peak_wset,
        'windows_tree_peak_wset_sum_bytes': windows_tree_peak_sum,
        'windows_main_peak_pagefile_bytes': windows_peak_pagefile,
        'cpu_seconds_observed': sum(cpu_by_identity.values()),
        'resource_guard_triggered': resource_guard_triggered,
        'resource_guard_threshold_bytes': min_available_bytes,
        'resource_guard_grace_seconds': low_memory_grace_seconds,
        'termination_details': termination_details, 'monitor_error': monitor_error,
        'remaining_child_pids': remaining_children,
        'log_path': str(log_path.resolve()), 'samples_path': str(samples_path.resolve()),
        'memory_method_ru': 'RSS дерева = сумма рабочих наборов наблюдаемых живых процессов; разделяемые страницы могут считаться несколько раз. fit-пик — максимум выборок в фазе fit, baseline — первая выборка этой фазы.',
        'windows_highwater_note_ru': 'peak_wset относится ко всему времени жизни процесса; сумма индивидуальных high-water marks не является одновременным пиком дерева. peak_pagefile — commit, а не физическая RAM.',
    }
