"""Check the Git index before publishing; never print matched secret contents.

Reads staged blobs (not working-tree substitutes). No model or data processing.
"""
from pathlib import Path
import csv
import io
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOTS = {'data', 'artifacts', 'handoff', 'local_private', 'tmp', '.venv', 'venv'}
PRIVATE_SUFFIXES = {'.parquet', '.joblib', '.pickle', '.pkl', '.npz', '.npy', '.zip', '.7z'}
SECRET_PATTERNS = {
    'private_key': rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'github_token': rb'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,}',
    'aws_access_key': rb'(?:AKIA|ASIA)[A-Z0-9]{16}',
    'token_pattern': rb'sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{35,}',
}


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def audit_index():
    files = [p.decode('utf-8') for p in git('diff', '--cached', '--name-only', '-z',
                                         '--diff-filter=ACMR').split(b'\0') if p]
    if not files:
        raise RuntimeError('No staged files to review')
    findings, inventory = [], []
    for name in files:
        path = Path(name)
        blob = git('show', ':' + name)
        inventory.append({'path': name, 'bytes': len(blob)})
        if path.parts[0] in PRIVATE_ROOTS or path.suffix.lower() in PRIVATE_SUFFIXES:
            findings.append({'path': name, 'issue': 'private path or binary artifact'})
        if path.name.lower() == 'kaggle.json' or path.name == '.env' or (
                path.name.startswith('.env.') and path.name != '.env.example'):
            findings.append({'path': name, 'issue': 'credential filename'})
        if '/bootstrap/' in name or re.search(r'bootstrap_.*_distribution\.csv$', name):
            findings.append({'path': name, 'issue': 'individual resample or bulky distribution'})
        if len(blob) > 10 * 1024**2:
            findings.append({'path': name, 'issue': 'file over 10 MiB review threshold'})
        if path.suffix.lower() not in {'.pdf', '.png', '.jpg', '.jpeg'}:
            for label, pattern in SECRET_PATTERNS.items():
                if re.search(pattern, blob):
                    findings.append({'path': name, 'issue': label})
        if path.suffix == '.csv':
            header = next(csv.reader(io.StringIO(blob.decode('utf-8-sig'))), [])
            if any(col.upper() in {'SK_ID_CURR', 'SK_ID_PREV', 'SK_ID_BUREAU'} for col in header):
                findings.append({'path': name, 'issue': 'identifier column in CSV; manual review required'})
        if path.suffix == '.ipynb':
            import nbformat
            notebook = nbformat.reads(blob.decode('utf-8'), as_version=4)
            nbformat.validate(notebook)
            if 'widgets' in notebook.metadata:
                findings.append({'path': name, 'issue': 'transient notebook widgets metadata'})
            for index, cell in enumerate(notebook.cells):
                for output in cell.get('outputs', []):
                    if any('widget' in mime for mime in output.get('data', {})):
                        findings.append({'path': name, 'cell': index, 'issue': 'transient widget output'})
    return {'passed': not findings, 'staged_files': len(files),
            'total_bytes': sum(row['bytes'] for row in inventory),
            'largest_files': sorted(inventory, key=lambda r: r['bytes'], reverse=True)[:10],
            'findings': findings,
            'scope': 'Staged blobs; path/size/secret-pattern/CSV-header/notebook checks. Complements manual report review.'}


if __name__ == '__main__':
    result = audit_index()
    destination = ROOT / 'tmp/publication_staged_audit.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
