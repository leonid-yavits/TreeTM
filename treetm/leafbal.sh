#!/bin/bash
# Leaf-balanced variant of one seed: experts whose children are all leaves are
# identical in distribution under both resampling schemes and are shared with
# results/<ds>/main/seed<s> (symlinks); the remaining experts are retrained
# with --resample leaf.  Usage: leafbal.sh <ds> <seed> <nodes,to,train> C T s E P
cd "$(dirname "$0")"
ds=$1; s=$2; nodes=$3; C=$4; T=$5; S=$6; E=$7; P=$8
src=results/$ds/main/seed$s; dst=results/$ds/leafbal/seed$s
mkdir -p $dst
for f in $src/*_final_masks.npz; do
  n=$(basename $f _final_masks.npz)
  if [[ ",$nodes," == *",$n,"* ]]; then continue; fi
  for ext in _final_masks.npz _scores.npz _hist.json _novel_scores.npy; do
    [ -e $src/$n$ext ] && ln -sfn ../../main/seed$s/$n$ext $dst/$n$ext
  done
done
python train.py $ds --run $dst --clauses $C --T $T --s $S --epochs $E --per_class $P --seed $s --procs 2 --resample leaf --nodes $nodes >> $dst.out 2>&1
