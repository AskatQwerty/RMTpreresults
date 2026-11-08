"""Publish the active 28 saved Pipelines without client records or new fitting."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SCOPE = 'pilot_20k_20config_mlp_updated'
MODELS = ('logistic_regression', 'random_forest', 'xgboost', 'lightgbm',
          'mlp', 'soft_voting', 'stacking')
BUNDLE_NAME = 'RMT_active_28_models.zip'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(value).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked(stage, marker, name):
    path = (stage / name).resolve()
    require(path.is_relative_to(stage.resolve()), 'Checkpoint path escapes its stage')
    content = path.read_bytes()
    require(digest(content) == marker['artifact_sha256'][name], 'Checksum mismatch: ' + str(path))
    return content


README = '''# Active RMT models: four updated MLPs

This bundle contains all **28 fitted Pipelines** for `pilot_20k_20config_mlp_updated`:
seven model types and four information levels. The training cohort remains the same
20,000 applicants; the evaluation test remains the same 61,503 applicants.

Twenty-four non-MLP Pipelines are byte-for-byte copies of the final models selected
from 20 configurations per pair. The four MLPs replace the old MLPs with models
selected in a separate, post-hoc experiment: four predeclared variants per level,
three unchanged CV folds, internal validation log-loss for epoch selection,
and a fresh fit on all 20,000 training applicants. The selected MLPs clip scaled
numeric values to ±5; levels 2–4 use alpha=0.1, level 1 uses alpha=0.00001.
Their selected epoch counts are 27, 24, 16 and 21.

The test had already been inspected before that supplemental experiment. These
MLP results must be labelled exploratory. Previous model and report versions
remain preserved in the project. Existing RQ3/SHAP results describe the earlier
`pilot_20k_5config` models; SHAP has not been recomputed for these replacements.

## Load a saved model

Extract the ZIP into a new directory and run Python from that directory. Use
Python 3.13.9 and the included pinned requirements for compatible deserialization.
Only load this trusted project bundle: joblib is not a safe format for unknown files.

```python
from pathlib import Path
import json
import joblib

root = Path('.')
model = joblib.load(root / 'models/mlp/level_4/pipeline.joblib')
features = json.loads((root / 'feature_levels.json').read_text(encoding='utf-8'))['4']
# X contains the existing prepared, untransformed Level 4 columns:
# probabilities = model.predict_proba(X.loc[:, features])[:, 1]
```

The `src/` directory must be importable. From another working directory, add the
absolute extraction directory to `sys.path` before calling `joblib.load`.
Preprocessing is already fitted inside each Pipeline. Do not fit it again before
prediction. A direct call to `Pipeline.fit` does not reproduce the supplemental
MLP epoch-selection procedure; use the research code for any new training.

## Contents and verification

- `models/<model>/level_<N>/pipeline.joblib`: all 28 fitted Pipelines.
- The adjacent `metadata.json`: selected parameters, epoch counts and lineage.
- `src/`: the four exact import modules needed by the saved Pipelines.
- `feature_levels.json` and `column_roles.json`: feature names and roles only.
- `MANIFEST.json`: source fingerprints and SHA256 of every other ZIP member.
- `requirements.txt`: pinned package versions used by the project.

No prepared X, individual IDs, TARGET records, fairness records, individual
probabilities or raw fit diagnostics are included. Building this bundle performs
no training, prediction, SHAP or bootstrap. All 28 Pipelines are deserialized in
an isolated directory as a compatibility check; their bytes are not altered.
'''


LOAD_CHECK = r'''
import json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(root))
import joblib
import src.model_pipelines,src.fold_preprocessing,src.mlp_repair
for module in (src.model_pipelines,src.fold_preprocessing,src.mlp_repair):
    assert Path(module.__file__).resolve().is_relative_to(root)
manifest=json.loads((root/'MANIFEST.json').read_text(encoding='utf-8'))
loaded=[]
for item in manifest['models']:
    path=root/item['pipeline_file']
    model=joblib.load(path)
    assert hasattr(model,'predict_proba')
    assert list(model.classes_)==[0,1]
    loaded.append({'model':item['model'],'level':item['level'],'class':type(model).__name__})
    del model
assert len(loaded)==28
print(json.dumps({'loaded':loaded,'fit_calls':0,'predict_calls':0,'bundled_imports_only':True}))
'''


def build(project=ROOT):
    project = Path(project).resolve()
    active = project / 'artifacts/training' / SCOPE
    summary = read(active / 'experiment_summary.json')
    require(summary['status'] == 'completed' and summary['completed_pairs'] == 28,
            'Wait until all 28 active pairs are complete')
    manifest = read(active / 'experiment_manifest.json')
    require(manifest['scope'] == SCOPE and manifest['fingerprint'] == summary['fingerprint'],
            'Active manifest does not match the completed revision')
    require(read(active / 'integrity_validation.json')['passed'], 'Promotion integrity audit failed')
    files, records = {}, []
    for model in MODELS:
        for level in range(1, 5):
            stage = active / model / f'level_{level}'
            marker = read(stage / 'complete.json')
            require(marker['fingerprint'] == manifest['fingerprint'], 'Mixed checkpoint revisions')
            record = json.loads(checked(stage, marker, 'metrics.json'))
            require(record['scope'] == SCOPE and record['model'] == model and record['level'] == level,
                    'Wrong pair provenance')
            name = f'models/{model}/level_{level}/pipeline.joblib'
            content = checked(stage, marker, 'fit_checkpoint/pipeline.joblib')
            files[name] = content
            lineage = record['source_lineage']
            # Whitelist fields rather than publishing full diagnostics or absolute paths.
            metadata = dict(model=model, level=level, scope=SCOPE, train_rows=20000,
                test_rows=61503, raw_features=record['raw_features'],
                selected_candidate_number=record.get('selected_candidate_number'),
                selected_variant=record.get('selected_variant'), selected_params=record['selected_params'],
                actual_mlp_epochs=record.get('actual_mlp_iterations', []),
                best_cv_log_loss=record['best_cv_log_loss'],
                source_lineage={key: lineage[key] for key in
                                ('model', 'level', 'source_scope', 'source_complete_sha256')},
                pipeline_sha256=digest(content), active_revision_fingerprint=manifest['fingerprint'])
            files[f'models/{model}/level_{level}/metadata.json'] = encoded(metadata)
            records.append(dict(model=model, level=level, pipeline_file=name,
                pipeline_bytes=len(content), pipeline_sha256=digest(content),
                source_scope=lineage['source_scope']))

    snapshot = active / 'source_snapshot'
    code = {}
    for name in ('__init__.py', 'fold_preprocessing.py', 'model_pipelines.py', 'mlp_repair.py'):
        frozen = snapshot / ('src/' + name if name == '__init__.py' else name)
        content = frozen.read_bytes()
        require(content == (project / 'src' / name).read_bytes(),
                'Current code differs from the fitted source snapshot: ' + name)
        files['src/' + name] = content
        code[name] = digest(content)

    features = read(project / 'data/processed/full/feature_levels.json')
    roles = read(project / 'handoff/RMT_batch_11_15/data/columns.json')
    prepared_digest = digest((project / 'data/processed/full/feature_levels.json').read_bytes())
    expected_digest = next(item['sha256'] for item in manifest['inputs'] if item['file'] == 'feature_levels.json')
    require(prepared_digest == expected_digest, 'Prepared feature list changed')
    for level, expected_count in zip(range(1, 5), (124, 178, 264, 322)):
        fields = features[str(level)]
        types = roles[str(level)]
        require(len(fields) == expected_count and len(set(fields)) == expected_count,
                'Unexpected feature list')
        require(set(types['numeric']).isdisjoint(types['categorical'])
                and set(types['numeric'] + types['categorical']) == set(fields), 'Column roles do not match X')
        require(not set(fields) & {'SK_ID_CURR', 'TARGET', 'CODE_GENDER', 'DAYS_BIRTH', 'AGE_YEARS'},
                'Excluded identifier/target/fairness field in model inputs')
    files['feature_levels.json'] = encoded(features)
    files['column_roles.json'] = encoded(roles)
    files['requirements.txt'] = (project / 'requirements.txt').read_bytes()
    files['README.md'] = README.encode('utf-8')
    public = dict(scope=SCOPE, active_revision_fingerprint=manifest['fingerprint'],
        original_final_fingerprint=manifest['original_final_fingerprint'],
        mlp_repair_plan_fingerprint=manifest['mlp_repair_plan_fingerprint'],
        mlp_repair_selection_sha256=manifest['mlp_repair_selection_sha256'],
        train_rows=20000, test_rows=61503, model_count=28,
        unchanged_non_mlp_models=24, replacement_mlp_models=4,
        models=records, source_module_sha256=code, feature_list_sha256=prepared_digest,
        no_individual_records=True, fit_calls_in_export=0, predict_calls_in_export=0,
        files_sha256={name: digest(content) for name, content in sorted(files.items())})
    files['MANIFEST.json'] = encoded(public)

    output = project / 'model_bundles'
    output.mkdir(exist_ok=True)
    archive = output / BUNDLE_NAME
    if archive.exists():
        with zipfile.ZipFile(archive) as existing:
            require(json.loads(existing.read('MANIFEST.json')) == public,
                    'Refuse to replace a bundle from another active revision')
    temporary = output / (BUNDLE_NAME + '.tmp')
    require(not temporary.exists(), 'Another bundle writer may be active')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for name, content in sorted(files.items()):
            zipped.writestr(name, content)
    with zipfile.ZipFile(temporary) as zipped:
        require(zipped.testzip() is None, 'ZIP CRC error')
        require(set(zipped.namelist()) == set(files), 'ZIP file inventory differs')
        require(sum(name.endswith('/pipeline.joblib') for name in zipped.namelist()) == 28,
                'Wrong bundled Pipeline count')
        for name, expected in public['files_sha256'].items():
            require(digest(zipped.read(name)) == expected, 'Bundled SHA mismatch: ' + name)

    (project / 'tmp').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='active_model_bundle_', dir=project / 'tmp') as temp:
        extraction = Path(temp)
        with zipfile.ZipFile(temporary) as zipped:
            for name in zipped.namelist():
                require((extraction / name).resolve().is_relative_to(extraction.resolve()), 'Unsafe ZIP member')
            zipped.extractall(extraction)
        env = dict(os.environ)
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            env[name] = '2'
        checked_load = subprocess.run([sys.executable, '-I', '-c', LOAD_CHECK, str(extraction)],
            cwd=extraction, env=env, capture_output=True, text=True, encoding='utf-8', check=True)
        load_result = json.loads(checked_load.stdout)
    temporary.replace(archive)
    result = dict(scope=SCOPE, bundle=BUNDLE_NAME, bytes=archive.stat().st_size,
        sha256=digest(archive.read_bytes()), active_revision_fingerprint=manifest['fingerprint'],
        model_count=28, total_pipeline_bytes=sum(item['pipeline_bytes'] for item in records),
        archive_members=len(files), internal_manifest='MANIFEST.json', all_crc_and_sha_checks_passed=True,
        isolated_load_validation=load_result, fit_calls=0, predict_calls=0,
        bootstrap_calls=0, shap_calls=0, created_utc=datetime.now(timezone.utc).isoformat())
    (output / 'manifest.json').write_bytes(encoded(result))
    (output / 'README.md').write_text(README + '\n## Download\n\n'
        + f'[{BUNDLE_NAME}]({BUNDLE_NAME}) — {archive.stat().st_size:,} bytes.\n\n'
        + f'SHA256: `{result["sha256"]}`.\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    build(parser.parse_args().project_root)
