#!/bin/bash
# Chained RQ3 runs: wait for the fixed run, then explainer-noise run, then five retrainings.
cd "$(dirname "$0")/.." || exit 1
PY="${RQ3_PYTHON:-.venv/bin/python}"
while pgrep -f "rq3_shap.py explain --run fixed " > /dev/null; do sleep 30; done
"$PY" rq3/rq3_shap.py explain --run fixed_noise --seed 1 --n_perm 2 --n_perm_ensemble 1 --models logistic_regression,mlp,soft_voting,stacking > rq3/out/run_fixed_noise.log 2>&1
for s in 1 2 3 4 5; do
  "$PY" rq3/rq3_shap.py explain --run r$s --refit-seed $s --n_perm 2 --n_perm_ensemble 1 > rq3/out/run_r$s.log 2>&1
done
echo ALL_DONE > rq3/out/ALL_DONE
