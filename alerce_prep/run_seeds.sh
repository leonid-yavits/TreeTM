#!/bin/bash
# Seeds 1-4 of the unweighted main configuration (seed 0 = results/main).
cd "$(dirname "$0")"
CFG='{"root":{"clauses":1000,"T":100,"s":4,"weighted":false},"Transient":{"clauses":1000,"T":100,"s":4,"weighted":false},"Stochastic":{"clauses":1000,"T":100,"s":4,"weighted":false},"Periodic":{"clauses":1000,"T":100,"s":4,"weighted":false},"flat":{"clauses":1000,"T":100,"s":4,"weighted":false}}'
for s in "$@"; do
  python run_tms.py --run results/seeds/seed$s --epochs 50 --per_class 4000 --cfg "$CFG" --seed $s --procs 2 > results/seeds/seed$s.out 2>&1
done
