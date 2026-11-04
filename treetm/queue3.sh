#!/bin/bash
# Sequential, resumable queue (one job at a time to stay within memory).
cd "$(dirname "$0")"
for s in 1 2; do python train.py wos --run results/wos/main/seed$s --clauses 100 --T 10 --s 5 --epochs 30 --per_class 600 --seed $s --procs 2 >> results/wos/main/seed$s.out 2>&1; done
for s in 0 1 2; do [ -f results/wos/leafbal/seed$s/root_final_masks.npz ] || ./leafbal.sh wos $s root 100 10 5 30 600; done
for s in 1 2; do python baselines.py alerce $s >> results/alerce/baselines.out 2>&1; done
for s in 0 1 2; do python baselines.py wos $s --models flat_rf,lcpn_rf,flat_mlp >> results/wos/baselines.out 2>&1; done
python train.py wos --run results/wos/sweep/C50 --clauses 50 --T 10 --s 5 --epochs 30 --per_class 600 --seed 0 --procs 2 >> results/wos/sweep.out 2>&1
python train.py alerce --run results/alerce/sweep/C250 --clauses 250 --T 100 --s 4 --epochs 50 --per_class 4000 --seed 0 --procs 2 >> results/alerce/sweep.out 2>&1
echo Q3_DONE
