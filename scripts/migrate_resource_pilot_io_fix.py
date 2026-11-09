"""Migrate verified fits after an atomic_json-only fix, without retraining.

The original pilot is fully preserved. The new fingerprint identifies updated I/O code;
each migrated checkpoint explicitly retains its original manifest and original hashes.
Any change to data, settings, or other AST nodes prevents migration.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.model_training import (PreparedData,read_json,atomic_json,sha256,
    initialize_run,_complete,_mark_complete)
from src.resource_pilot import resource_context

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('old_root',type=Path)
args=parser.parse_args()
old=args.old_root.resolve()
assert not (old/'run.lock').exists()
previous=read_json(old/'experiment_manifest.json')
for name,expected in previous['code_sha256'].items():
    assert sha256(old/'source_snapshot'/name)==expected, f'Original source snapshot mismatch: {name}'
current=resource_context(PreparedData(ROOT/'data/processed/full'),previous['threads'])
prior_protocol={k:v for k,v in previous.items() if k not in {'fingerprint','code_sha256'}}
new_protocol={k:v for k,v in current.items() if k not in {'fingerprint','code_sha256'}}
assert prior_protocol==new_protocol,'Protocol/data/settings changed; migration is forbidden.'
changes=[name for name in current['code_sha256'] if current['code_sha256'][name]!=previous['code_sha256'][name]]
assert changes==['model_training.py'],changes
def without_atomic(path):
    tree=ast.parse(Path(path).read_text(encoding='utf-8'))
    assert sum(isinstance(node,ast.FunctionDef) and node.name=='atomic_json' for node in tree.body)==1
    tree.body=[node for node in tree.body if not (isinstance(node,ast.FunctionDef) and node.name=='atomic_json')]
    return ast.dump(tree,include_attributes=False)
old_ast=without_atomic(old/'source_snapshot/model_training.py')
new_ast=without_atomic(ROOT/'src/model_training.py')
assert old_ast==new_ast,'The change affects training, not only JSON writing.'
models=[name for name in current['selected_configurations'] if _complete(old/name,previous)]
assert len(models)==6 and 'stacking' not in models,models
new=old.parent/f'resource_{current["fingerprint"][:12]}'
assert new!=old and not new.exists(),'The new directory already exists; automatic overwrite is disabled.'
initialize_run(new,current)
shutil.copytree(old,new/'origin_pilot')
assert sha256(new/'origin_pilot/experiment_manifest.json')==sha256(old/'experiment_manifest.json')
for name in models:
    assert _complete(new/'origin_pilot'/name,previous),name
proof={'operation':'verified_import_after_atomic_json_only_io_fix',
    'origin_root':str(old),'origin_pilot_relative':'origin_pilot',
    'origin_fingerprint':previous['fingerprint'],'current_fingerprint':current['fingerprint'],
    'origin_manifest_sha256':sha256(old/'experiment_manifest.json'),
    'protocol_inputs_versions_and_selected_configurations_equal':True,
    'changed_source_files':changes,'only_changed_ast_function':'atomic_json',
    'remaining_ast_sha256':hashlib.sha256(old_ast.encode()).hexdigest(),
    'imported_models':models,'new_fit_required':['stacking'],
    'original_models_not_refit':True}
atomic_json(new/'migration.json',proof)
for name in models:
    stage=new/name
    shutil.copytree(old/name,stage)
    assert _complete(stage,previous),name
    original=read_json(stage/'complete.json')
    shutil.copy2(stage/'complete.json',stage/'original_complete.json')
    provenance={'origin_pilot_relative':'origin_pilot','origin_model':name,
        'origin_fingerprint':previous['fingerprint'],
        'origin_manifest_sha256':proof['origin_manifest_sha256'],
        'original_complete_sha256':sha256(stage/'original_complete.json'),
        'migration_sha256':sha256(new/'migration.json'),
        'source_of_fit':'original pilot; unchanged serialized model and diagnostics',
        'refit_performed_during_import':False}
    atomic_json(stage/'imported_stage_provenance.json',provenance)
    names=list(original['artifact_sha256'])+['original_complete.json','imported_stage_provenance.json',
        'worker.log','memory_samples.csv']
    _mark_complete(stage,current,list(dict.fromkeys(names)))
    assert _complete(stage,current)
snapshot=new/'source_snapshot'; snapshot.mkdir()
for name,expected in current['code_sha256'].items():
    assert sha256(ROOT/'src'/name)==expected
    shutil.copy2(ROOT/'src'/name,snapshot/name)
print('New pilot root:',new)
print('Imported without fit:',models)
print('Original evidence preserved:',new/'origin_pilot')
