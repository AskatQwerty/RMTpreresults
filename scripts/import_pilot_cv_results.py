"""Verify the returned 11--15 ZIP and consolidate saved CV scores; never fit.

Only JSON, CSV, Parquet and ZIP bytes are read. Imported Python/pickle files are
never executed. Original experiments, prepared X and splits remain untouched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / 'artifacts/training'
SENT = ROOT / 'handoff/RMT_batch_11_15'
OLD = TRAINING / 'pilot_20k_5config'
EXT = TRAINING / 'pilot_20k_20config_extension'
LAST = TRAINING / 'pilot_20k_config16_20'
REPORT = ROOT / 'reports/training/pilot_20k_20config_combined'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)


def write(path, value):
    Path(path).write_text(canonical(value) + '\n', encoding='utf-8')


def verified_zip(path):
    """Reject traversal, duplicate members and SHA/size mismatch before extraction."""
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        names = [i.filename for i in infos]
        require(len(names) == len(set(names)), 'Duplicate ZIP members')
        require(sum(i.file_size for i in infos) < 100_000_000, 'Unexpected archive size')
        for i in infos:
            p = PurePosixPath(i.filename)
            require(not p.is_absolute() and not any(x in ('..', '.') for x in p.parts)
                    and '\\' not in i.filename and ':' not in i.filename,
                    f'Unsafe archive path: {i.filename}')
            require(not i.is_dir() and not stat.S_ISLNK(i.external_attr >> 16),
                    'Only regular archive files are supported')
        require(z.testzip() is None, 'ZIP CRC check failed')
        blobs = {n: z.read(n) for n in names}
    manifest = json.loads(blobs['RETURN_MANIFEST.json'])
    entries = manifest['files']
    require(len(entries) == len({e['path'] for e in entries}), 'Duplicate manifest paths')
    require(set(blobs) == {'RETURN_MANIFEST.json'} | {e['path'] for e in entries},
            'Archive members differ from return manifest')
    for e in entries:
        b = blobs[e['path']]
        require(len(b) == e['bytes'] and sha(b) == e['sha256'], f'ZIP SHA mismatch: {e["path"]}')
    require(manifest['status'] == 'COMPLETE' and manifest['completed_cv_fits'] == 420,
            'Returned batch is incomplete')
    return blobs, manifest


def verify_stage(get_bytes, prefix, fingerprint):
    marker = json.loads(get_bytes(prefix + '/complete.json'))
    require(marker['fingerprint'] == fingerprint, f'Wrong checkpoint context: {prefix}')
    require(set(marker['artifact_sha256']) == {'result.json', 'fit_diagnostics.json'},
            f'Wrong checkpoint file list: {prefix}')
    for name, digest in marker['artifact_sha256'].items():
        require(sha(get_bytes(prefix + '/' + name)) == digest, f'Checkpoint SHA: {prefix}/{name}')
    return json.loads(get_bytes(prefix + '/result.json'))


def main(archive):
    blobs, returned = verified_zip(archive)
    plan = read(EXT / 'configuration_plan.json')
    plan_base = dict(plan)
    require(plan_base.pop('fingerprint') == sha(canonical(plan_base).encode()), 'Fixed plan fingerprint')
    portable = json.loads(blobs['job_plan.json'])
    for name in ('job_plan.json', 'package_manifest.json', 'data/cohort_ids.parquet',
                 'data/cohort_targets.parquet', 'data/cv_folds.parquet',
                 'data/feature_levels.json', 'data/columns.json'):
        require(blobs[name] == (SENT / name).read_bytes(), f'Sent package changed: {name}')
    require(portable['configurations'] == plan['configurations'], 'Frozen configurations changed')
    require(returned['original_plan_fingerprint'] == plan['fingerprint'], 'Wrong fixed plan')
    require(returned['original_pilot_fingerprint'] == plan['pilot_fingerprint'], 'Wrong original pilot')
    package = json.loads(blobs['package_manifest.json'])
    package_id = sha(json.dumps({'schema_version': 1, 'files': package['files']},
                               sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode())
    require(package_id == returned['package_id'] == package['package_id'], 'Wrong package ID')
    context = json.loads(blobs['results/experiment_context.json'])
    context_base = dict(context)
    require(context_base.pop('fingerprint') == sha(canonical(context_base).encode()), 'Returned context hash')
    require(context['fingerprint'] == returned['context_fingerprint'], 'Returned context mismatch')
    require(context['runner_sha256'] == sha((SENT / 'run_batch.py').read_bytes()), 'Runner changed')
    require(context['job_plan_sha256'] == sha(blobs['job_plan.json']), 'Job plan SHA')
    require(context['package_manifest_sha256'] == sha(blobs['package_manifest.json']), 'Package manifest SHA')
    for name, digest in portable['original_cohort_sha256'].items():
        require(sha(blobs['data/' + name]) == digest == sha((OLD / name).read_bytes()), 'Cohort mismatch')
    ids = pd.read_parquet(io.BytesIO(blobs['data/cohort_ids.parquet']))
    target = pd.read_parquet(io.BytesIO(blobs['data/cohort_targets.parquet']))
    folds = pd.read_parquet(io.BytesIO(blobs['data/cv_folds.parquet']))
    require(len(ids) == ids.SK_ID_CURR.nunique() == 20000, 'Cohort unique IDs')
    require(ids.SK_ID_CURR.equals(target.SK_ID_CURR) and ids.SK_ID_CURR.equals(folds.SK_ID_CURR), 'Row order')
    require(target.TARGET.value_counts().to_dict() == {0: 18385, 1: 1615}, 'Class balance')
    require(folds.validation_fold.value_counts().to_dict() == {0: 6667, 1: 6667, 2: 6666}, 'Saved folds')

    models, levels = plan['models'], plan['levels']
    records, warning_rows = {}, []
    source_checks = []
    def add(key, score, params, batch, source, **extra):
        require(key not in records, f'Duplicate CV identity: {key}')
        model, level, number, fold = key
        cfg = plan['configurations'][model][number - 1]
        require(params == cfg['params'] and math.isfinite(score), f'Invalid parameters/score: {key}')
        records[key] = dict(model=model, level=level, candidate_number=number, fold=fold,
                            neg_log_loss=score, log_loss=-score, batch=batch,
                            config_sha256=cfg['config_sha256'], params_json=json.dumps(params, sort_keys=True),
                            source=source, **extra)

    # The initial RandomizedSearchCV saved exact fold scores but only mean times.
    # Do not invent per-fold times or attribute search-level warnings to a fold.
    legacy_times = {}
    for entry in plan['legacy_results']:
        file, marker = OLD / entry['cv_results_file'], OLD / entry['complete_file']
        require(sha(file.read_bytes()) == entry['cv_results_sha256'], f'Legacy CSV SHA: {file}')
        require(sha(marker.read_bytes()) == entry['complete_sha256'], f'Legacy marker SHA: {marker}')
        complete = read(marker)
        require(complete['fingerprint'] == plan['pilot_fingerprint'], 'Legacy context')
        for name, digest in complete['artifact_sha256'].items():
            require(sha((marker.parent / name).read_bytes()) == digest, f'Legacy artifact SHA: {name}')
        frame = pd.read_parquet(file)
        require(sorted(frame.candidate.tolist()) == list(range(5)), 'Legacy five configurations')
        for r in frame.itertuples():
            values = [getattr(r, f'fold_{f}_neg_log_loss') for f in range(3)]
            require(abs(np.mean(values) - r.mean_neg_log_loss) < 1e-12, 'Legacy CV mean')
            key3 = (entry['model'], entry['level'], int(r.candidate) + 1)
            legacy_times[key3] = (r.mean_fit_seconds, r.mean_score_seconds)
            for f, score in enumerate(values):
                add((*key3, f), score, json.loads(r.params_json), '01_05',
                    file.relative_to(ROOT).as_posix(), fit_seconds=None, score_seconds=None,
                    timing_granularity='candidate_mean_only')
        for w in read(marker.parent / 'warnings.json'):
            warning_rows.append(dict(batch='01_05', model=entry['model'], level=entry['level'],
                                     attribution='search_and_original_refit', **w))
        source_checks.append({'path': file.relative_to(ROOT).as_posix(), 'sha256': sha(file.read_bytes())})

    ext_context = read(EXT / 'experiment_manifest.json')
    last_context = read(LAST / 'results/experiment_context.json')
    require(ext_context['configuration_plan_fingerprint'] == plan['fingerprint'], 'Extension context')
    require(last_context['original_plan_fingerprint'] == plan['fingerprint'], 'Final batch plan')
    require(read(LAST / 'job_plan.json')['configurations'] == plan['configurations'], 'Final batch configs')
    for name in portable['original_cohort_sha256']:
        require((LAST / 'data' / name).read_bytes() == blobs['data/' + name], 'Final batch cohort/folds')
    imported_sha = sha(Path(archive).read_bytes())
    import_root = TRAINING / 'pilot_20k_config11_15_imported' / imported_sha[:12]
    sources = [
        ('06_10', range(6, 11), EXT, ext_context['fingerprint'], ''),
        ('11_15', range(11, 16), None, context['fingerprint'], 'results/checkpoints/'),
        ('16_20', range(16, 21), LAST, last_context['fingerprint'], 'results/checkpoints/'),
    ]
    for batch, numbers, base, fingerprint, prefix in sources:
        expected = {f'{prefix}{m}/level_{l}/candidate_{n:02d}/fold_{f}/complete.json'
                    for m, l, n, f in itertools.product(models, levels, numbers, range(3))}
        if base is None:
            actual = {n for n in blobs if n.startswith(prefix) and n.endswith('/complete.json')}
            get_bytes = blobs.__getitem__
        else:
            actual = {p.relative_to(base).as_posix() for p in (base / prefix).glob('*/level_*/candidate_*/fold_*/complete.json')}
            get_bytes = lambda name, base=base: (base / name).read_bytes()
        require(actual == expected, f'Unexpected/missing checkpoints in {batch}')
        for marker in sorted(expected):
            r = verify_stage(get_bytes, marker.removesuffix('/complete.json'), fingerprint)
            key = (r['model'], r['level'], r['candidate_number'], r['fold'])
            expected_path = f'{prefix}{key[0]}/level_{key[1]}/candidate_{key[2]:02d}/fold_{key[3]}/complete.json'
            require(marker == expected_path, 'Checkpoint identity/path mismatch')
            cfg = plan['configurations'][key[0]][key[2]-1]
            require(r['config_sha256'] == cfg['config_sha256'], 'Configuration hash mismatch')
            require(r['cohort_rows'] == 20000 and r['train_rows'] == (13334 if key[3] == 2 else 13333)
                    and r['validation_rows'] == (6666 if key[3] == 2 else 6667), 'Fold row counts')
            require(r['scoring'] == 'neg_log_loss' and r['refit'] is False and r['test_used'] is False,
                    'Unexpected scorer/refit/test use')
            if base != EXT:
                require(r['original_plan_fingerprint'] == plan['fingerprint']
                        and r['original_pilot_fingerprint'] == plan['pilot_fingerprint']
                        and r['technical'] is False, 'Checkpoint provenance')
            relative = marker.replace('/complete.json', '/result.json')
            source = (import_root / relative if base is None else base / relative).relative_to(ROOT).as_posix()
            add(key, r['neg_log_loss'], r['params'], batch, source,
                fit_seconds=r['fit_seconds'], score_seconds=r['score_seconds'], timing_granularity='per_fold')
            for w in r['warnings']:
                warning_rows.append(dict(batch=batch, model=key[0], level=key[1], candidate_number=key[2],
                                         fold=key[3], attribution='fold', **w))
    all_expected = set(itertools.product(models, levels, range(1, 21), range(3)))
    require(set(records) == all_expected and len(records) == 1680, 'Full 1680 coverage failed')
    fold_frame = pd.DataFrame(records.values()).sort_values(['model', 'level', 'candidate_number', 'fold'])
    candidate_rows = []
    for key, group in fold_frame.groupby(['model', 'level', 'candidate_number'], sort=True):
        require(group.fold.tolist() == [0, 1, 2], 'Exactly three folds per candidate')
        mean_fit, mean_score = legacy_times.get(key, (group.fit_seconds.mean(), group.score_seconds.mean()))
        candidate_rows.append(dict(model=key[0], level=key[1], candidate_number=key[2],
            mean_log_loss=float(group.log_loss.mean()), std_log_loss=float(group.log_loss.std(ddof=0)),
            **{f'fold_{f}_log_loss': float(group.loc[group.fold == f, 'log_loss'].iloc[0]) for f in range(3)},
            mean_fit_seconds=mean_fit, mean_score_seconds=mean_score,
            batch=group.batch.iloc[0], config_sha256=group.config_sha256.iloc[0], params_json=group.params_json.iloc[0]))
    candidates = pd.DataFrame(candidate_rows)
    require(len(candidates) == 560, 'Expected 560 candidate/model/level combinations')
    selected = candidates.sort_values(['mean_log_loss', 'candidate_number']).groupby(['model', 'level'], sort=True).head(1)
    selected = selected.sort_values(['model', 'level']).reset_index(drop=True)
    require(len(selected) == 28, 'Expected 28 CV selections')
    returned_candidates = candidates[candidates.batch == '11_15'].sort_values('mean_log_loss')
    # Recompute returned summary means from checkpoints, do not trust presentation CSVs.
    returned_csv = pd.read_csv(io.BytesIO(blobs['results/candidate_results.csv']))
    for r in returned_candidates.itertuples():
        match = returned_csv[(returned_csv.model == r.model) & (returned_csv.level == r.level)
                             & (returned_csv.candidate_number == r.candidate_number)]
        require(len(match) == 1 and abs(-float(match.iloc[0].mean_neg_log_loss) - r.mean_log_loss) < 1e-12,
                'Returned candidate summary disagrees with checkpoints')

    # Preserve original archive and its complete read-only evidence in a separate root.
    import_root.mkdir(parents=True, exist_ok=True)
    for name, data in blobs.items():
        destination = import_root / name
        require(destination.resolve().is_relative_to(import_root.resolve()), 'Extraction outside root')
        if destination.exists():
            require(destination.read_bytes() == data, f'Existing imported evidence differs: {name}')
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
    archive_copy = import_root / Path(archive).name
    if archive_copy.exists():
        require(sha(archive_copy.read_bytes()) == imported_sha, 'Existing archive differs')
    else:
        shutil.copyfile(archive, archive_copy)
    report = REPORT
    report.mkdir(parents=True, exist_ok=True)
    fold_frame.to_csv(report / 'all_1680_cv_folds.csv', index=False, encoding='utf-8-sig')
    candidates.to_csv(report / 'all_560_candidates.csv', index=False, encoding='utf-8-sig')
    selected.to_csv(report / 'best_28_by_cv.csv', index=False, encoding='utf-8-sig')
    selected.pivot(index='model', columns='level', values='mean_log_loss').to_csv(
        report / 'best_cv_log_loss_by_level.csv', encoding='utf-8-sig')
    pd.DataFrame(warning_rows).to_csv(report / 'warnings.csv', index=False, encoding='utf-8-sig')
    write(report / 'selected_parameters.json', [dict(model=r.model, level=r.level,
        candidate_number=r.candidate_number, mean_cv_log_loss=r.mean_log_loss,
        params=json.loads(r.params_json), fitted_final_pipeline=False) for r in selected.itertuples()])
    shim = next(n for n in blobs if n.endswith('/forest_thread_executor.py.txt'))
    audit = dict(status='validated_with_documented_execution_deviation',
        created_utc=datetime.now(timezone.utc).isoformat(), archive_sha256=imported_sha,
        archive_bytes=Path(archive).stat().st_size, archive_members=len(blobs),
        verified_archive_payload_files=len(returned['files']), returned_cv_fits=420,
        total_cv_fits=1680, expected_cv_fits=1680, missing_cv_fits=0, duplicate_cv_fits=0,
        candidates_per_pair=20, folds=3, pairs=28, train_rows=20000,
        class_counts={'0': 18385, '1': 1615}, saved_cohort_and_folds_unchanged=True,
        original_plan_fingerprint=plan['fingerprint'], original_pilot_fingerprint=plan['pilot_fingerprint'],
        imported_evidence=import_root.relative_to(ROOT).as_posix(),
        counts_by_batch=fold_frame.groupby('batch').size().to_dict(),
        best_in_returned_batch=returned_candidates.iloc[0].to_dict(),
        best_overall_cv=candidates.sort_values('mean_log_loss').iloc[0].to_dict(),
        execution_deviation={'scope': 'all 60 RandomForest CV fits in batch11_15',
            'file': (import_root / shim).relative_to(ROOT).as_posix(), 'sha256': sha(blobs[shim]),
            'description': 'joblib threading backend replaced by ThreadPoolExecutor; original n_jobs=2 retained',
            'model_hyperparameters_changed': False, 'included_in_original_package_fingerprint': False,
            'bitwise_equivalence_test_available': False,
            'interpretation': 'Retained as documented execution adaptation; raw cross-host timings are not directly comparable'},
        source_cv_files=source_checks,
        no_fit_or_predict_called=True, new_test_evaluations=0, new_bootstrap_calls=0,
        limitations=[
            'This completes the 1680 CV budget on the existing 20000-client pilot, not full development.',
            'Final selected Pipelines have not been refitted or evaluated on test by this consolidation.',
            'Original 28 Pipelines/test results remain tied to the original five-configuration search.',
            'Five training repeats per selected pair are not performed here.',
            'Initial batch stores mean candidate times and search-level warnings, not exact per-fold timing/warnings.',
            'Three-fold standard deviation is descriptive dispersion, not a confidence interval.',
            'ZIP checksums establish file integrity/consistency, not independent proof of runtime execution.',
        ])
    write(report / 'import_audit.json', audit)
    best = audit['best_overall_cv']
    text = f'''Проверка и объединение сохранённых CV-результатов RMT\n\nГотово: 1680 / 1680 CV-обучений (7 моделей × 4 уровня × 20 конфигураций × 3 фолда).\nПо каждой партии 01–05, 06–10, 11–15, 16–20: 420 результатов; пропусков и повторов ключей нет.\nTrain: прежние 20 000 клиентов, классы 18 385 / 1 615. Сохранённые ID и CV-фолды совпадают.\nЭто расширенный пилот на 20 000, а не эксперимент на всём development из A2.\n\nЛучший средний CV log-loss: {best['mean_log_loss']:.9f}, {best['model']}, Level {best['level']}, конфигурация {best['candidate_number']}.\nЭто результат выбора по CV, не итоговая тестовая метрика.\n\nФайлы:\nall_1680_cv_folds.csv — точные оценки каждого сохранённого фолда и источник.\nall_560_candidates.csv — среднее и стандартное отклонение трёх фолдов по каждому кандидату.\nbest_28_by_cv.csv / selected_parameters.json — лучшие параметры для каждой модели и уровня.\nbest_cv_log_loss_by_level.csv — таблица 7 × 4.\nwarnings.csv — сохранённые предупреждения с указанием доступной точности атрибуции.\nimport_audit.json — контрольные суммы, проверки, источники и ограничения.\n\nОсобенность партии 11–15: для 60 RF-fit после WinError 5 заменён механизм joblib threading на ThreadPoolExecutor с прежними n_jobs=2.\nКод моделей, параметры и разбиения не менялись. Адаптация не входила в исходный fingerprint пакета; её текст и SHA сохранены отдельно.\nПобитовая эквивалентность механизмов не проверялась. Время разных компьютеров нельзя непосредственно сравнивать как скорость моделей.\nОстановленные незавершённые попытки не считаются дополнительными завершёнными CV-fit.\n\nСейчас только прочитаны и объединены существующие результаты: новых fit/predict/bootstrap нет.\nДля итоговой оценки победителей по 20 конфигурациям потребуется отдельное обучение выбранных Pipeline на train и оценка на неизменном test.\nСтарые 28 Pipeline и их тестовые отчёты относятся к поиску по 5 конфигурациям; их нельзя выдавать за новые результаты.\nПять повторов из A2 здесь не выполнялись. Исходные результаты, X, test и разбиения сохранены без изменений.\n'''
    (report / 'README_RU.txt').write_text(text, encoding='utf-8-sig')
    print(canonical({k: audit[k] for k in ['status', 'returned_cv_fits', 'total_cv_fits',
                                          'counts_by_batch', 'best_in_returned_batch', 'best_overall_cv']}))
    print('Report:', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    main(args.archive)
