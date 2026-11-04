#!/bin/bash
# Second queue: leaf-balanced variants, clause sweeps. Resumable.
cd "$(dirname "$0")"
for s in 1 2; do [ -f results/alerce/leafbal/seed$s/root_final_masks.npz ] || ./leafbal.sh alerce $s root 1000 100 4 50 4000; done
for s in 0 1 2; do [ -f results/wos/leafbal/seed$s/root_final_masks.npz ] || ./leafbal.sh wos $s root 100 10 5 30 600; done
# clause sweep, seed 0, child-balanced, tree + flat
for C in 250; do
  python train.py alerce --run results/alerce/sweep/C$C --clauses $C --T 100 --s 4 --epochs 50 --per_class 4000 --seed 0 --procs 2 >> results/alerce/sweep.out 2>&1
done
for C in 50; do
  python train.py wos --run results/wos/sweep/C$C --clauses $C --T 10 --s 5 --epochs 30 --per_class 600 --seed 0 --procs 2 >> results/wos/sweep.out 2>&1
done
echo QUEUE2_DONE
