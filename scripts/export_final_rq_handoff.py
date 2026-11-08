"""Package completed final RQ1/RQ2 reports; no fitting or statistical recomputation."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/pilot_20k_20config_final/evaluation'
TRAINING = ROOT / 'artifacts/training/pilot_20k_20config_final'
CV = ROOT / 'reports/training/pilot_20k_20config_combined'


def main():
    summary = json.loads((REPORT / 'evaluation_summary.json').read_text())
    readiness = json.loads((REPORT / 'results_readiness.json').read_text())
    training = json.loads((TRAINING / 'experiment_summary.json').read_text())
    assert summary['status'] == 'completed' and summary['completed_bootstrap'] == 1000
    assert readiness['status'] == 'ready' and not readiness['blockers']
    assert training['completed_pairs'] == training['completed_final_fits'] == 28
    payload = {}
    for path in sorted(REPORT.rglob('*')):
        if path.is_file() and 'bootstrap' not in path.relative_to(REPORT).parts and path.suffix not in ('.parquet', '.npz'):
            payload['evaluation/' + path.relative_to(REPORT).as_posix()] = path.read_bytes()
    for name in ('all_1680_cv_folds.csv', 'all_560_candidates.csv', 'best_28_by_cv.csv',
                 'selected_parameters.json', 'import_audit.json', 'mlp_audit.json',
                 'mlp_audit.txt', 'mlp_audit.csv', 'mlp_preprocessing_audit.csv'):
        payload['cv_selection/' + name] = (CV / name).read_bytes()
    for name in ('experiment_manifest.json', 'experiment_summary.json', 'current_launch.json',
                 'final_metrics.csv', 'old_preresults_integrity_validation.json'):
        payload['training/' + name] = (TRAINING / name).read_bytes()
    for name in ('final_pilot_training.py', 'final_pilot_evaluation.py', 'final_pilot_reporting.py',
                 'model_pipelines.py', 'fold_preprocessing.py', 'fit_diagnostics.py',
                 'pilot_evaluation.py', 'pilot_evaluation_bootstrap.py',
                 'pilot_evaluation_decisions.py', 'pilot_delong.py'):
        payload['source/' + name] = (ROOT / 'src' / name).read_bytes()
    note = '''RMT: итоговые результаты пилота 20k / 20 конфигураций\n\nОткройте evaluation/results_report.html. Графики и таблицы находятся рядом.\nВыбор параметров: 1680 завершённых CV-fit; затем 28 новых итоговых fit на прежних 20000 клиентах. Test: прежние61503.\nСтарые предварительные результаты pilot_20k_5config сохранены отдельно без изменений.\nВ архив включены таблицы, графики,1000bootstrap-распределений, CV-оценки и аудит MLP.\nИндивидуальные вероятности и28файлов Pipeline остаются в artifacts/training/pilot_20k_20config_final проекта; в этот небольшой отчётный архив они не включены.\nОтчёт отражает пилот на20000 и один итоговый fit каждой пары; пятьповторов A2 здесь не выполнялись.\nSHA256 в MANIFEST.json позволяет проверить содержимое архива.\n'''
    payload['START_HERE_RU.txt'] = note.encode('utf-8-sig')
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), scope='pilot_20k_20config_final_RQ1_RQ2',
                    files=[dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
                           for name, data in sorted(payload.items())])
    payload['MANIFEST.json'] = json.dumps(manifest, ensure_ascii=False, indent=2).encode()
    destination = ROOT / 'handoff/RMT_RQ1_RQ2_final20config_2026-11-08.zip'
    destination.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(destination, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name, data in sorted(payload.items()):
            z.writestr(name, data)
    with zipfile.ZipFile(destination) as z:
        assert z.testzip() is None
        for name, data in payload.items():
            assert hashlib.sha256(z.read(name)).digest() == hashlib.sha256(data).digest()
    print(json.dumps(dict(file=str(destination), bytes=destination.stat().st_size,
                         sha256=hashlib.sha256(destination.read_bytes()).hexdigest(), files=len(payload)), indent=2))


if __name__ == '__main__':
    main()
