"""Собрать локальный пакет SHAP без fit/predict и без повторных вычислений.

--prepare выбирает готовые строки X по сохранённым ID и копирует пилот.
--finalize включает код из текущего Git commit, проверяет SHA-256 и создаёт ZIP.
Архив и рабочая папка остаются в игнорируемом handoff/; публикации нет.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCOPE = 'pilot_20k_5config'
STAGING = ROOT / 'handoff' / f'{SCOPE}_shap'
ARTIFACT = Path('artifacts/training') / SCOPE
DATA = Path('data/handoff') / SCOPE
PREPARED = ROOT / 'data/processed/full'
ID = 'SK_ID_CURR'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def copy_checked(source, relative):
    destination = STAGING / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    before = sha256(source)
    shutil.copy2(source, destination)
    assert sha256(destination) == before and sha256(source) == before
    return {'path': Path(relative).as_posix(), 'bytes': destination.stat().st_size,
            'sha256': before}


def inspect_saved_pipelines():
    """Проверить загрузку и имена столбцов; не вызывать fit/transform/predict."""
    import joblib
    records = []
    for path in sorted((STAGING / ARTIFACT).rglob('pipeline.joblib')):
        pipeline = joblib.load(path)
        model = path.parents[2].name
        level = int(path.parents[1].name.split('_')[1])
        expected = json.loads((STAGING / DATA / f'features_level_{level}.json').read_text(encoding='utf-8'))
        assert list(pipeline.feature_names_in_) == expected
        assert list(pipeline.classes_) == [0, 1]
        if 'preprocess' in pipeline.named_steps:
            preprocessors = {'preprocess': pipeline.named_steps['preprocess']}
        else:
            preprocessors = {name: base.named_steps['preprocess']
                for name, base in pipeline.named_steps['model'].named_estimators_.items()}
        names = {name: preprocessor.get_feature_names_out().tolist()
                 for name, preprocessor in preprocessors.items()}
        records.append({'model': model, 'level': level,
            'pipeline_path': path.relative_to(STAGING).as_posix(), 'classes_': [0, 1],
            'raw_feature_count': len(expected), 'raw_feature_order_verified': True,
            'transformed_feature_counts': {name: len(columns) for name, columns in names.items()},
            'transformed_feature_names': names})
        del pipeline, preprocessors
        gc.collect()
    assert len(records) == 28
    return {'load_only_check_passed': True, 'new_fit_calls': 0,
        'new_transform_calls': 0, 'new_predict_calls': 0, 'models': records}


def prepare():
    STAGING.mkdir(parents=True, exist_ok=True)
    (STAGING / DATA).mkdir(parents=True, exist_ok=True)
    pilot = ROOT / ARTIFACT
    experiment = json.loads((pilot / 'experiment_manifest.json').read_text(encoding='utf-8'))
    assert experiment['mode'] == SCOPE
    assert experiment['train_rows'] == 20000 and experiment['test_rows'] == 61503
    sources = {}
    # Read only prepared data and pilot artifacts, never raw CSVs.
    needed = ['X_level_4.parquet', 'row_ids.parquet', 'target.parquet', 'fairness.parquet',
              'feature_levels.json', *[f'features_level_{level}.json' for level in range(1, 5)]]
    expected = {row['file']: row['sha256'] for row in experiment['inputs']}
    for name in needed:
        path = PREPARED / name
        sources[path.relative_to(ROOT).as_posix()] = sha256(path)
        assert sources[path.relative_to(ROOT).as_posix()] == expected[name], name
    levels = json.loads((PREPARED / 'feature_levels.json').read_text(encoding='utf-8'))
    features = levels['4']
    for level in range(1, 5):
        names = json.loads((PREPARED / f'features_level_{level}.json').read_text(encoding='utf-8'))
        assert names == levels[str(level)] and set(names).issubset(features)
    cohorts = {'train': pd.read_parquet(pilot / 'cohort_ids.parquet'),
               'test': pd.read_parquet(pilot / 'test_ids.parquet')}
    assert [len(cohorts[s]) for s in ['train', 'test']] == [20000, 61503]
    assert set(cohorts['train'][ID]).isdisjoint(cohorts['test'][ID])
    row_ids = pd.read_parquet(PREPARED / 'row_ids.parquet')[ID]
    assert row_ids.is_unique and all(c[ID].is_unique for c in cohorts.values())
    parquet = pq.ParquetFile(PREPARED / 'X_level_4.parquet')
    assert parquet.metadata.num_rows == len(row_ids) and parquet.schema.names == features
    assert not set([ID, 'TARGET', 'DAYS_BIRTH', 'CODE_GENDER']).intersection(features)
    selected = {name: [] for name in cohorts}
    offset = 0
    for batch in parquet.iter_batches(batch_size=16384):
        ids = row_ids.iloc[offset:offset + batch.num_rows].to_numpy()
        table = pa.Table.from_batches([batch])
        for name, cohort in cohorts.items():
            mask = np.isin(ids, cohort[ID].to_numpy())
            if mask.any():
                part = table.filter(pa.array(mask))
                part = part.add_column(0, ID, pa.array(ids[mask]))
                selected[name].append(part)
        offset += batch.num_rows
    assert offset == len(row_ids)
    target = pd.read_parquet(PREPARED / 'target.parquet')
    fairness = pd.read_parquet(PREPARED / 'fairness.parquet')
    assert target[ID].is_unique and fairness[ID].is_unique
    counts = {}
    for name, cohort in cohorts.items():
        table = pa.concat_tables(selected[name])
        positions = pd.Index(table[ID].to_numpy()).get_indexer(cohort[ID])
        assert (positions >= 0).all() and len(table) == len(cohort)
        table = table.take(pa.array(positions))
        assert np.array_equal(table[ID].to_numpy(), cohort[ID].to_numpy())
        path = STAGING / DATA / f'X_{name}_level4.parquet'
        pq.write_table(table, path, compression='zstd')
        assert pq.read_table(path).equals(table, check_metadata=False)
        cohort.to_parquet(STAGING / DATA / f'{name}_ids.parquet', index=False)
        for label, frame in [('target', target), ('fairness', fairness)]:
            joined = cohort.merge(frame, on=ID, how='left', sort=False, validate='one_to_one', indicator=True)
            assert joined['_merge'].eq('both').all()
            joined = joined.drop(columns='_merge')
            assert joined[ID].tolist() == cohort[ID].tolist()
            joined.to_parquet(STAGING / DATA / f'{label}_{name}.parquet', index=False)
            if label == 'target':
                assert joined.TARGET.isin([0, 1]).all()
                counts[name] = {str(k): int(v) for k, v in joined.TARGET.value_counts().sort_index().items()}
        del table
    assert counts['train'] == {'0': 18385, '1': 1615}
    assert counts['test'] == {'0': 56538, '1': 4965}
    for name in ['feature_levels.json', *[f'features_level_{i}.json' for i in range(1, 5)]]:
        copy_checked(PREPARED / name, DATA / name)
    copy_checked(pilot / 'cv_folds.parquet', DATA / 'cv_folds.parquet')
    history = [name for name in features if name.startswith('HAS_')]
    schema = {'scope': SCOPE, 'id_column': ID, 'target_column': 'TARGET',
        'positive_class': 1, 'probability_meaning': 'P(TARGET=1): default',
        'feature_count': len(features), 'raw_feature_order': features,
        'feature_counts': {level: len(names) for level, names in levels.items()},
        'arrow_types': {field.name: str(field.type) for field in parquet.schema_arrow},
        'history_indicators': history, 'x_layout': 'SK_ID_CURR followed by raw Level4 features; select feature list before Pipeline use',
        'row_order': 'Exact existing cohort_ids.parquet and test_ids.parquet order; explicit SK_ID_CURR column',
        'fairness_columns': fairness.columns.tolist(), 'target_not_in_X': True,
        'sensitive_direct_features_not_in_X': True, 'new_fit_calls': 0, 'new_predict_calls': 0,
        'selection_only': True, 'rows': {name: len(ids) for name, ids in cohorts.items()}, 'class_counts': counts}
    json_write(STAGING / DATA / 'feature_schema.json', schema)
    artifact_paths = sorted(p for p in pilot.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    pipelines = [p for p in artifact_paths if p.name == 'pipeline.joblib']
    assert len(pipelines) == 28
    # All 28 are small; the resource-pilot directory is never traversed.
    assert sum(p.stat().st_size for p in pipelines) < 500 * 1024**2
    for path in artifact_paths:
        rel = path.relative_to(ROOT)
        record = copy_checked(path, rel)
        sources[rel.as_posix()] = record['sha256']
    json_write(STAGING / DATA / 'model_schema.json', inspect_saved_pipelines())
    # Prepared source files must still be byte-identical after row selection.
    for rel, digest in sources.items():
        assert sha256(ROOT / rel) == digest, rel
    preparation = {'scope': SCOPE, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'models': 28, 'model_bytes': sum(p.stat().st_size for p in pipelines),
        'rows': schema['rows'], 'class_counts': counts, 'new_fit_calls': 0,
        'new_predict_calls': 0, 'source_sha256': sources,
        'prepared_file_sha256': {p.relative_to(STAGING).as_posix(): sha256(p)
            for p in sorted((STAGING / DATA).rglob('*')) if p.is_file()}}
    json_write(STAGING / 'preparation_validation.json', preparation)
    print(json.dumps({key: preparation[key] for key in ['models', 'model_bytes', 'rows', 'class_counts']}, ensure_ascii=False))
    return preparation


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, check=True, capture_output=True).stdout


def finalize():
    preparation = json.loads((STAGING / 'preparation_validation.json').read_text(encoding='utf-8'))
    commit = git('rev-parse', 'HEAD').decode().strip()
    branch = git('branch', '--show-current').decode().strip()
    tracked = git('ls-tree', '-r', '--name-only', 'HEAD').decode().splitlines()
    code = [name for name in tracked if name == 'README.md' or name.startswith('requirements')
        or name.startswith('src/') and name.endswith('.py') or name.startswith('docs/')
        or name == 'scripts/build_shap_handoff_package.py']
    assert 'README.md' in code and 'docs/SHAP_HANDOFF.md' in code
    assert not git('diff', '--name-only', 'HEAD').strip(), 'Commit tracked changes before packaging'
    code_hashes = {}
    repository_hashes = {}
    # The reviewed public commit is small. Include its complete snapshot so
    # notebook/report links also work without a separate repository checkout.
    for name in tracked:
        content = git('show', f'{commit}:{name}')
        destination = STAGING / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        repository_hashes[name] = sha256(destination)
        if name in code:
            code_hashes[name] = repository_hashes[name]
    for rel, digest in preparation['source_sha256'].items():
        assert sha256(ROOT / rel) == digest, f'Source changed: {rel}'
        if rel.startswith('artifacts/'):
            assert sha256(STAGING / rel) == digest
    for rel, digest in preparation['prepared_file_sha256'].items():
        assert sha256(STAGING / rel) == digest
    instructions = (
        'Локальный пакет pilot_20k_5config для участника SHAP/RQ3\n\n'
        'Распакуйте в отдельный каталог и работайте из его корня.\n'
        'Инструкция и пример загрузки без обучения: docs/SHAP_HANDOFF.md.\n'
        'Данные клиентов и модели не предназначены для публичного GitHub.\n'
        'Python 3.13.9; установите requirements.txt; для отчётов также requirements-reporting.txt.\n'
        'Включены 28 Pipeline: 7 моделей x 4 уровня, один refit на сочетание.\n'
        'Также включён полный проверенный снимок публичного commit: код, notebooks, отчёты и документация.\n'
        'X_train_level4 и X_test_level4 содержат SK_ID_CURR; перед передачей в Pipeline\n'
        'выберите столбцы строго по features_level_N.json. Уровни вложены.\n'
        'Новых fit, predict, bootstrap и SHAP при сборке не выполнялось.\n'
        f'Версия кода: {commit}; рабочая ветка: {branch}.\n'
        'package_manifest.json содержит размеры и SHA-256 всех файлов, кроме самого manifest.\n')
    (STAGING / 'PACKAGE_README.txt').write_text(instructions, encoding='utf-8')
    included = sorted(p for p in STAGING.rglob('*') if p.is_file() and p.name != 'package_manifest.json')
    manifest = {'scope': SCOPE, 'package_kind': 'private_team_shap_handoff',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'git_commit': commit,
        'git_branch': branch, 'code_sha256': code_hashes,
        'repository_snapshot_sha256': repository_hashes,
        'models': 28, 'rows': preparation['rows'], 'class_counts': preparation['class_counts'],
        'feature_schema': (DATA / 'feature_schema.json').as_posix(),
        'source_unchanged': True, 'new_fit_calls': 0, 'new_predict_calls': 0,
        'files': [{'path': p.relative_to(STAGING).as_posix(), 'bytes': p.stat().st_size,
                   'sha256': sha256(p)} for p in included]}
    json_write(STAGING / 'package_manifest.json', manifest)
    archive = ROOT / 'handoff' / f'{SCOPE}_shap.zip'
    temporary = archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6,
                         allowZip64=True) as zipped:
        for path in sorted(p for p in STAGING.rglob('*') if p.is_file()):
            zipped.write(path, arcname=path.relative_to(STAGING).as_posix())
    with zipfile.ZipFile(temporary) as zipped:
        assert zipped.testzip() is None
        for record in manifest['files']:
            digest = hashlib.sha256(zipped.read(record['path'])).hexdigest()
            assert digest == record['sha256'], record['path']
    temporary.replace(archive)
    digest = sha256(archive)
    archive.with_suffix('.zip.sha256').write_text(f'{digest}  {archive.name}\n', encoding='ascii')
    summary = {'archive': str(archive), 'bytes': archive.stat().st_size,
        'sha256': digest, 'git_commit': commit, 'git_branch': branch,
        'files': len(manifest['files']) + 1, 'uncompressed_bytes': sum(x['bytes'] for x in manifest['files']),
        'models': 28, 'rows': preparation['rows'], 'archive_crc_and_sha256_verified': True,
        'new_fit_calls': 0, 'new_predict_calls': 0}
    json_write(ROOT / 'handoff' / f'{SCOPE}_package_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--finalize', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        finalize()
