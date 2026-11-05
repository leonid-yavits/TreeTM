#!/bin/bash
# Validation-selected flat TM: C=1000 on NSL-KDD (on ALeRCE, validation selects the main C=1000)
cd "$(dirname "$0")"
link() { mkdir -p $2; for f in $1/flat_*; do ln -sfn $(realpath --relative-to=$2 $f) $2/$(basename $f); done; }
link results/nslkdd/sweep/C1000 results/nslkdd/flattuned/seed0
for s in 1 2 3 4; do d=results/nslkdd/flattuned/seed$s; [ -f $d/flat_final_masks.npz ] || python train.py nslkdd --run $d --clauses 1000 --T 50 --s 4 --epochs 50 --per_class 4000 --seed $s --procs 1 --nodes flat >> $d.out 2>&1; done
echo FT_DONE
