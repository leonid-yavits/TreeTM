#!/bin/bash
# Exploratory root-capacity run: TreeTM-C seed 0 with a root of C_root clauses
# per class; all other experts and the flat TM are shared with main/seed0.
# Usage: abl_root.sh <ds> <C_root> T s E P
cd "$(dirname "$0")"
ds=$1; CR=$2; T=$3; S=$4; E=$5; P=$6
src=results/$ds/main/seed0; dst=results/$ds/abl_root${CR}_child/seed0
mkdir -p $dst
for f in $src/*_final_masks.npz; do
  n=$(basename $f _final_masks.npz); [ $n = root ] && continue
  for ext in _final_masks.npz _scores.npz _hist.json _novel_scores.npy; do
    [ -e $src/$n$ext ] && ln -sfn ../../main/seed0/$n$ext $dst/$n$ext
  done
done
python train.py $ds --run $dst --clauses $CR --T $T --s $S --epochs $E --per_class $P --seed 0 --procs 1 --nodes root >> $dst.out 2>&1
