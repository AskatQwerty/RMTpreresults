#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
PY="${RQ3_PYTHON:-.venv/bin/python}"
sleep 30
"$PY" rq3/rq3_shap.py explain --run r2 --refit-seed 2 --n 500 --n_perm 1 --n_perm_ensemble 1 >> rq3/out/run_r2.log 2>&1
"$PY" rq3/rq3_shap.py explain --run r4 --refit-seed 4 --n 500 --n_perm 1 --n_perm_ensemble 1 > rq3/out/run_r4.log 2>&1
for s in a b; do
  seed=$([ $s = a ] && echo 11 || echo 12)
  "$PY" rq3/rq3_shap.py explain --run noise_$s --seed $seed --n 500 --n_perm 1 --models logistic_regression,mlp > rq3/out/run_noise_$s.log 2>&1
  "$PY" rq3/rq3_shap.py explain --run noise_$s --seed $seed --n 300 --n_perm_ensemble 1 --models soft_voting,stacking >> rq3/out/run_noise_$s.log 2>&1
done
echo DONE_B > rq3/out/DONE_B
