#!/bin/bash
cd "$(dirname "$0")"; mkdir -p results/nslkdd/main results/nslkdd/leafbal results/nslkdd/budget results/nslkdd/sweep
for C in 50 100 250 1000; do
  [ -f results/nslkdd/sweep/C$C/flat_final_masks.npz ] || python train.py nslkdd --run results/nslkdd/sweep/C$C --clauses $C --T 50 --s 4 --epochs 50 --per_class 4000 --seed 0 --procs 2 >> results/nslkdd/sweep.out 2>&1
done
[ -f results/nslkdd/abl_root2000_child/seed0/root_final_masks.npz ] || ./abl_root.sh nslkdd 2000 50 4 50 4000
echo QC_DONE
