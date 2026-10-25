#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
PY="${RQ3_PYTHON:-.venv/bin/python}"
sleep 60
for s in 2 4; do "$PY" rq3/rq3_shap.py explain --run r$s --refit-seed $s --n 500 --n_perm 1 --n_perm_ensemble 1 > rq3/out/run_r$s.log 2>&1; done
"$PY" rq3/rq3_shap.py explain --run fixed_noise --seed 1 --n 500 --n_perm 2 --models logistic_regression,mlp > rq3/out/run_fixed_noise.log 2>&1
"$PY" rq3/rq3_shap.py explain --run fixed_noise --seed 1 --n 300 --n_perm_ensemble 1 --models soft_voting,stacking >> rq3/out/run_fixed_noise.log 2>&1
echo DONE_B > rq3/out/DONE_B
