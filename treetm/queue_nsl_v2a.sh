#!/bin/bash
cd "$(dirname "$0")"; mkdir -p results/nslkdd/main results/nslkdd/leafbal results/nslkdd/budget results/nslkdd/sweep
for s in 0 1 2 3 4; do
  [ -f results/nslkdd/main/seed$s/u2r_final_masks.npz ] && [ -f results/nslkdd/main/seed$s/flat_final_masks.npz ] && continue
  python train.py nslkdd --run results/nslkdd/main/seed$s --clauses 500 --T 50 --s 4 --epochs 50 --per_class 4000 --seed $s --procs 2 > results/nslkdd/main/seed$s.out 2>&1
done
echo QA_DONE
