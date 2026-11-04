#!/bin/bash
# Usage: run_ds_seeds.sh <dataset> <clauses> <T> <s> <epochs> <per_class> <procs> <seeds...>
cd "$(dirname "$0")"
ds=$1; C=$2; T=$3; S=$4; E=$5; P=$6; PR=$7; shift 7
for seed in "$@"; do
  python train.py $ds --run results/$ds/main/seed$seed --clauses $C --T $T --s $S --epochs $E --per_class $P --seed $seed --procs $PR > results/$ds/main/seed$seed.out 2>&1
done
