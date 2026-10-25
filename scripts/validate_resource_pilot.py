"""Проверить сохранённый пилот и обновить небольшие отчёты, без fit/test."""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.metadata
import shutil
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data_quality import artifact_manifest
from src.model_training import read_json, atomic_json, sha256, _complete, full_experiment_plan
from src.training_reporting import write_resource_reports, write_full_report

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('pilot_root',type=Path)
args=parser.parse_args()
pilot=args.pilot_root.resolve()
manifest=read_json(pilot/'experiment_manifest.json')
summary=read_json(pilot/'resource_summary.json')
assert not (pilot/'run.lock').exists(),'Пилот ещё работает; дождитесь снятия run.lock.'
plan=full_experiment_plan()
allowed_models=set(plan['models'])
assert set(manifest['selected_configurations'])==allowed_models,'Неверный состав моделей в manifest.'
results=summary['results']
result_models=[item['model'] for item in results]
assert len(result_models)==len(set(result_models)),'Модель повторяется в сводке пилота.'
assert set(result_models)<=allowed_models,'В сводке есть незапланированная модель.'
assert len(summary['planned_models'])==len(allowed_models) and set(summary['planned_models'])==allowed_models
assert summary['attempted_models']==len(results),'Счётчик попыток не совпадает с results.'
completed_count=sum(item['status']=='completed' for item in results)
assert summary['completed_models']==completed_count,'Счётчик завершений не совпадает с results.'
if summary['status']=='completed':
    assert set(result_models)==allowed_models and completed_count==len(allowed_models), \
        'Для completed требуются все семь завершённых моделей.'
prepared=ROOT/'data/processed/full'
assert artifact_manifest(prepared)==manifest['inputs'],'Изменились подготовленные входы.'
sources=pilot/'source_snapshot'
sources.mkdir(exist_ok=True)
for name,expected in manifest['code_sha256'].items():
    path=ROOT/'src'/name
    assert sha256(path)==expected,f'Источник изменился: {name}'
    shutil.copy2(path,sources/name)
origin=pilot/'origin_pilot'
migration=None
origin_manifest=None
if origin.exists():
    migration=read_json(pilot/'migration.json')
    origin_manifest=read_json(origin/'experiment_manifest.json')
    assert migration['origin_pilot_relative']=='origin_pilot'
    assert migration['origin_fingerprint']==origin_manifest['fingerprint']
    assert migration['current_fingerprint']==manifest['fingerprint']
    assert migration['origin_manifest_sha256']==sha256(origin/'experiment_manifest.json')
    assert migration['original_models_not_refit'] is True
    assert migration['protocol_inputs_versions_and_selected_configurations_equal'] is True
    assert {k:v for k,v in origin_manifest.items() if k not in {'fingerprint','code_sha256'}}== \
        {k:v for k,v in manifest.items() if k not in {'fingerprint','code_sha256'}}
    assert set(origin_manifest['code_sha256'])==set(manifest['code_sha256'])
    changed=[name for name in manifest['code_sha256']
             if manifest['code_sha256'][name]!=origin_manifest['code_sha256'][name]]
    assert changed==migration['changed_source_files']==['model_training.py']
    for name,expected in origin_manifest['code_sha256'].items():
        assert sha256(origin/'source_snapshot'/name)==expected,f'Изменён исходный снимок: {name}'
    def without_atomic_json(path):
        tree=ast.parse(path.read_text(encoding='utf-8'))
        matches=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='atomic_json']
        assert len(matches)==1
        tree.body=[node for node in tree.body if node is not matches[0]]
        return ast.dump(tree,include_attributes=False)
    old_ast=without_atomic_json(origin/'source_snapshot/model_training.py')
    assert old_ast==without_atomic_json(sources/'model_training.py')
    assert migration['only_changed_ast_function']=='atomic_json'
    assert migration['remaining_ast_sha256']==hashlib.sha256(old_ast.encode()).hexdigest()
    assert len(migration['imported_models'])==len(set(migration['imported_models']))
    assert set(migration['imported_models'])<=allowed_models
checks=[]
imported_models_checked=[]
for item in results:
    stage=pilot/item['model']
    if item['status']!='completed': continue
    assert _complete(stage,manifest),f'Неверный checkpoint: {stage}'
    marker=read_json(stage/'complete.json')
    assert 'resource_result.json' in marker['artifact_sha256'], \
        f'Результат этапа не защищён checkpoint-хэшем: {stage}'
    provenance_path=stage/'imported_stage_provenance.json'
    if provenance_path.exists():
        assert migration is not None and origin_manifest is not None
        assert {'original_complete.json','imported_stage_provenance.json'}<=set(marker['artifact_sha256'])
        provenance=read_json(provenance_path)
        assert provenance['origin_pilot_relative']=='origin_pilot'
        assert provenance['origin_model']==item['model']
        assert provenance['origin_fingerprint']==origin_manifest['fingerprint']
        assert provenance['origin_manifest_sha256']==sha256(origin/'experiment_manifest.json')
        assert provenance['migration_sha256']==sha256(pilot/'migration.json')
        assert provenance['refit_performed_during_import'] is False
        original_marker=read_json(stage/'original_complete.json')
        assert original_marker['fingerprint']==origin_manifest['fingerprint']
        assert provenance['original_complete_sha256']==sha256(stage/'original_complete.json')
        assert sha256(stage/'original_complete.json')==sha256(origin/item['model']/'complete.json')
        assert _complete(origin/item['model'],origin_manifest)
        for name,expected in original_marker['artifact_sha256'].items():
            assert sha256(stage/name)==expected,f'Изменён перенесённый артефакт: {stage/name}'
        imported_models_checked.append(item['model'])
    stored_result=read_json(stage/'resource_result.json')
    summary_result=dict(item)
    # Resume добавляет только служебный флаг в новую сводку, не переписывая этап.
    stored_result.pop('resumed',None)
    summary_result.pop('resumed',None)
    assert summary_result==stored_result,f'Сводка не совпадает с проверенным результатом: {stage}'
    assert item['rows']==manifest['train_rows'] and item['raw_features']==manifest['raw_features']
    assert item['model_size_bytes']==(stage/'pipeline.joblib').stat().st_size
    assert read_json(stage/'selected_configuration.json')==manifest['selected_configurations'][item['model']]
    assert item['warnings']==read_json(stage/'warnings.json')
    diagnostics=read_json(stage/'fit_diagnostics.json')
    assert diagnostics['processed_feature_counts']==item['processed_features']
    for row in diagnostics['estimators']:
        if 'preprocessor_width_matches_classifier' in row['checks']:
            assert row['checks']['preprocessor_width_matches_classifier']
    for row in diagnostics['mlp']:
        assert row['checks']['loss_curve_length_matches_n_iter']
        assert row['fitted']['loss_curve_'][-1]==row['fitted']['loss_']
    monitor=read_json(stage/'resource_monitor.json')
    assert monitor['status']=='completed' and monitor['exit_code']==0
    assert not monitor.get('remaining_child_pids',[])
    checks.append({'model':item['model'],'stage_hashes_checked':True,
        'imported_provenance_verified':provenance_path.exists(),
        'summary_matches_hash_checked_stage':True,
        'configuration_matches_manifest':True,'diagnostics_verified':True,
        'warnings_verified':True,'model_bytes_verified':True})
if migration is not None:
    assert set(imported_models_checked)==set(migration['imported_models'])
evidence={str(path.relative_to(pilot)):sha256(path) for path in sorted(pilot.rglob('*'))
    if path.is_file() and (path.name in {'worker.log','memory_samples.csv','phase.json','failure.json',
        'experiment_manifest.json','resource_summary.json','migration.json','original_complete.json',
        'imported_stage_provenance.json'} or 'source_snapshot' in path.relative_to(pilot).parts)}
atomic_json(pilot/'supplementary_evidence_sha256.json',evidence)
reports=ROOT/'reports/training'
atomic_json(reports/'resource_pilot_summary.json',summary)
atomic_json(reports/'resource_pilot_configurations.json',manifest['selected_configurations'])
shutil.copy2(pilot/'resource_metrics.csv',reports/'resource_pilot_metrics.csv')
write_resource_reports(pilot.parent,reports/'resource')
old=ROOT/'artifacts/training/a2_full_v1'
unchanged_factory=None
if (old/'experiment_manifest.json').exists():
    original=read_json(old/'experiment_manifest.json')
    unchanged_factory=all(sha256(ROOT/'src'/name)==original['code_sha256'][name]
        for name in ['model_pipelines.py','fold_preprocessing.py'])
    assert unchanged_factory
    write_full_report(old,reports/'full_v1_interrupted',prepared_dir=prepared)
assert (plan['searches'],plan['candidate_cv_outer_fits'],plan['main_outer_fits'])==(28,1680,140)
invocations=[]
if origin.exists():
    for source in sorted(origin.glob('attempt_history/*/resource_summary.json')):
        invocations.append((f'origin/attempt_history/{source.parent.name}',read_json(source),source.parent))
    invocations.append(('origin/current',read_json(origin/'resource_summary.json'),origin))
for source in sorted(pilot.glob('attempt_history/*/resource_summary.json')):
    invocations.append((f'current/attempt_history/{source.parent.name}',read_json(source),source.parent))
invocations.append(('current',summary,pilot))
primitive_counts={spec['model']:spec['primitive_fits_per_outer_fit'] for spec in plan['model_specs']}
attempt_rows=[]
for invocation,snapshot,directory in invocations:
    for item in snapshot['results']:
        if item.get('resumed'): continue
        stage=directory/item['model']
        monitoring=stage/'resource_monitor.json'
        observed=read_json(monitoring) if monitoring.exists() else {}
        phase=read_json(stage/'phase.json') if (stage/'phase.json').exists() else {}
        fit_seconds=item.get('fit_seconds')
        fit_time_source='resource_result' if fit_seconds is not None else None
        # phase получает fit_seconds только после успешного estimator.fit.
        # Это сохраняет факт завершения fit при последующей ошибке записи.
        if fit_seconds is None and phase.get('fit_seconds') is not None:
            fit_seconds=phase['fit_seconds']
            fit_time_source='phase_after_successful_fit'
        fit_completed=fit_seconds is not None
        fit_started=fit_completed or observed.get('fit_sample_count',0)>0
        peak=item.get('peak_ram_bytes')
        primitive_count=primitive_counts[item['model']]
        # Для прерванного одиночного RF известен один начатый fit. Число
        # завершённых баз прерванного ансамбля без дополнительных логов неизвестно.
        finished_primitive=primitive_count if fit_completed else 0 if primitive_count==1 else None
        unfinished_primitive=1 if fit_started and not fit_completed and primitive_count==1 else 0 if fit_completed else None
        attempt_rows.append({'invocation':invocation,'model':item['model'],'status':item['status'],
            'evidence_directory':str(stage.relative_to(pilot)),
            'fit_started':fit_started,'fit_completed':fit_completed,
            'fit_seconds_if_completed':fit_seconds,'fit_time_source':fit_time_source,
            'primitive_fits_finished':finished_primitive,'primitive_fits_unfinished':unfinished_primitive,
            'model_size_bytes':item.get('model_size_bytes'),
            'rows':item.get('rows'),'raw_features':item.get('raw_features'),
            'monitor_elapsed_seconds':observed.get('elapsed_seconds'),
            'peak_ram_gib':peak/1024**3 if peak is not None else None})
pd.DataFrame(attempt_rows).to_csv(reports/'resource_pilot_attempts.csv',index=False)
validation={'status':summary['status'],'pilot_root':str(pilot),'completed_models_checked':len(checks),
    'checks':checks,'input_files_sha256_unchanged':True,'input_files_checked':len(manifest['inputs']),
    'unique_allowed_models_and_counters_verified':True,
    'source_snapshot_verified':True,'model_factory_and_preprocessing_unchanged':unchanged_factory,
    'imported_models_verified':imported_models_checked,'migration_proof_verified':migration is not None,
    'main_budget':{'searches':28,'candidate_cv_outer_fits':1680,'main_outer_fits':140},
    'test_used':False,'psutil_version':importlib.metadata.version('psutil'),
    'supplementary_evidence_files_hashed':len(evidence),
    'total_completed_fit_seconds':sum(r.get('fit_seconds',0) for r in summary['results']),
    'elapsed_seconds_this_invocation':summary['elapsed_seconds_this_invocation'],
    'invocations_recorded':len(invocations),'outer_fit_attempts_recorded':len(attempt_rows),
    'outer_fits_started_recorded':sum(r['fit_started'] for r in attempt_rows),
    'outer_fits_finished_recorded':sum(r['fit_completed'] for r in attempt_rows),
    'primitive_fits_finished_recorded':sum(r['primitive_fits_finished'] or 0 for r in attempt_rows),
    'primitive_fits_unfinished_recorded':sum(r['primitive_fits_unfinished'] or 0 for r in attempt_rows),
    'primitive_fit_counts_fully_known':all(r['primitive_fits_finished'] is not None and
        r['primitive_fits_unfinished'] is not None for r in attempt_rows),
    'completed_fit_seconds_all_attempts':sum(r['fit_seconds_if_completed'] or 0 for r in attempt_rows),
    'monitor_elapsed_seconds_all_attempts':sum(r['monitor_elapsed_seconds'] or 0 for r in attempt_rows),
    'interrupted_attempts':[r for r in attempt_rows if r['status']!='completed'],
    'elapsed_seconds_all_invocations':sum(s['elapsed_seconds_this_invocation'] for _,s,_ in invocations)}
atomic_json(reports/'resource_pilot_validation.json',validation)
print('Verified completed models:',len(checks),'status:',summary['status'])
print('Reports:',reports)
