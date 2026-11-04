#!/bin/bash
cd "$(dirname "$0")"
for s in 1 2 3 4; do [ -f results/nslkdd/leafbal/seed$s/attack_final_masks.npz ] || ./leafbal.sh nslkdd $s root,attack 500 50 4 50 4000; done
for C in 50 100 250 1000; do
  python train.py nslkdd --run results/nslkdd/sweep/C$C --clauses $C --T 50 --s 4 --epochs 50 --per_class 4000 --seed 0 --procs 1 >> results/nslkdd/sweep.out 2>&1
done
for s in 1 2 3 4; do python baselines.py nslkdd $s >> results/nslkdd/baselines.out 2>&1; done
echo NSL_DONE
