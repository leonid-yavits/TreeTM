#!/bin/bash
# ALeRCE, same configuration as seed 0 of the first study (C=1000, T=100, s=4, unweighted, P=4000, 50 epochs)
cd "$(dirname "$0")"
for seed in "$@"; do
  python train.py alerce --run results/alerce/main/seed$seed --clauses 1000 --T 100 --s 4 --epochs 50 --per_class 4000 --seed $seed --procs 2 > results/alerce/main/seed$seed.out 2>&1
done
