#!/bin/bash
cd /home/claude/rq3_work; PY=/home/claude/venv_rq3/bin/python
$PY rq3/rq3_shap.py explain --run fixed --n 500 --n_perm_ensemble 1 --models soft_voting,stacking >> rq3/out/run_fixed.log 2>&1
for s in 1 3 5; do $PY rq3/rq3_shap.py explain --run r$s --refit-seed $s --n 500 --n_perm 1 --n_perm_ensemble 1 > rq3/out/run_r$s.log 2>&1; done
echo DONE_A > rq3/out/DONE_A
