#!/bin/bash
# Resumable experiment queue. Completed nodes and baselines are skipped;
# interrupted TM nodes resume from their per-epoch checkpoints.
cd "$(dirname "$0")"
run_tm () {  # ds seed C T s E P
  mkdir -p results/$1/main
  python train.py $1 --run results/$1/main/seed$2 --clauses $3 --T $4 --s $5 --epochs $6 --per_class $7 --seed $2 --procs 2 >> results/$1/main/seed$2.out 2>&1
}
for s in 1 2; do run_tm alerce $s 1000 100 4 50 4000; done
for s in 1 2 3 4; do python baselines.py nslkdd $s >> results/nslkdd/baselines.out 2>&1; done
for s in 0 1 2; do python baselines.py alerce $s >> results/alerce/baselines.out 2>&1; done
echo QUEUE_DONE
