# Active RMT models: four updated MLPs

[Download all 28 models (ZIP, 18.5 MB)](RMT_active_28_models.zip) · [Download manifest and validation results (JSON)](manifest.json)

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

## Download

[RMT_active_28_models.zip](RMT_active_28_models.zip) — 18,499,348 bytes.

SHA256: `33bc897cfee188810a15ce4850a2e59144f6f523b45ebb3ce56cea152f37e355`.
