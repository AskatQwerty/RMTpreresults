from pathlib import Path
import hashlib,json,os,shutil,zipfile
r=Path(__file__).resolve().parent
d=Path(os.environ['SHAP_PROJECT'])
batch=r.parent/'RMT_batch_11_15/data'
config=json.loads((r/'frozen_plan.json').read_bytes())
def sha(b):return hashlib.sha256(b).hexdigest()
for name,info in config['inputs'].items():
    assert sha((batch/Path(name).name).read_bytes())==info['sha256'],name
official=json.loads((r/'active_bundle_manifest.json').read_bytes())
bundle=r/'active_models.zip'
assert sha(bundle.read_bytes())==official['sha256']
with zipfile.ZipFile(bundle) as z:
    m=json.loads(z.read('MANIFEST.json'))
    assert not z.testzip()
    for name,expected in m['files_sha256'].items():
        assert sha(z.read(name))==expected,name
        if '/level_4/' in name:z.extract(name,r/'active_bundle')
for name in ['mlp_repair.py','model_pipelines.py','fold_preprocessing.py']:
    assert sha((r/'src'/name).read_bytes())==config['code_sha256'][name],name
out=r/'rq3/out';out.mkdir(exist_ok=True,parents=True)
for name in ['explained_ids.parquet','background_ids.parquet','selection.json']:
    shutil.copyfile(d/'rq3/out'/name,out/name)
dest=r/'data/handoff/pilot_20k_5config';dest.mkdir(exist_ok=True,parents=True)
handoff=d/'Get-FileHash pilot_20k_5config_shap/data/handoff/pilot_20k_5config'
shutil.copyfile(handoff/'features_level_4.json',dest/'features_level_4.json')
files={k:sha((handoff/k).read_bytes()) for k in ['features_level_4.json','X_test_level4.parquet','target_test.parquet','test_ids.parquet']}
for k in ['explained_ids.parquet','background_ids.parquet','selection.json']:files['rq3/out/'+k]=sha((out/k).read_bytes())
(r/'verification').mkdir(exist_ok=True)
(r/'verification/input_checksums.json').write_text(json.dumps(files,indent=2))
print('Verified training data, source modules, selection provenance and original model archive.')
