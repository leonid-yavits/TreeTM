#!/bin/bash
# Sequential, resumable queue (one job at a time to stay within memory).
cd "$(dirname "$0")"
for s in 1 2; do python baselines.py alerce $s >> results/alerce/baselines.out 2>&1; done
python train.py alerce --run results/alerce/sweep/C250 --clauses 250 --T 100 --s 4 --epochs 50 --per_class 4000 --seed 0 --procs 2 >> results/alerce/sweep.out 2>&1
echo Q3_DONE
