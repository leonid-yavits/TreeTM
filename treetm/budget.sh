#!/bin/bash
# Equal-budget root (review M-4): TreeTM-C whose root draws P_root = P * L / K_root
# samples per child and epoch, i.e. as many samples per epoch as the flat TM.
# All other experts and the flat TM are shared with results/<ds>/main/seed<s>.
# Usage: budget.sh <ds> <seed> <P_root> C T s E
cd "$(dirname "$0")"
ds=$1; s=$2; PR=$3; C=$4; T=$5; S=$6; E=$7
src=results/$ds/main/seed$s; dst=results/$ds/budget/seed$s
mkdir -p $dst
for f in $src/*_final_masks.npz; do
  n=$(basename $f _final_masks.npz); [ $n = root ] && continue
  for ext in _final_masks.npz _scores.npz _hist.json _novel_scores.npy; do
    [ -e $src/$n$ext ] && ln -sfn ../../main/seed$s/$n$ext $dst/$n$ext
  done
done
python train.py $ds --run $dst --clauses $C --T $T --s $S --epochs $E --per_class $PR --seed $s --procs 1 --nodes root >> $dst.out 2>&1
