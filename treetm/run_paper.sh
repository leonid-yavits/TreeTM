#!/bin/bash
# Every run and analysis behind the paper, in order. Run from treetm/.
# Data preparation (step 0) is described in the README.
set -e
cd "$(dirname "$0")"

# 1. Flat TM and category experts, one run per seed.
#    train.py trains every node of the taxonomy and the flat TM. The paper uses the
#    flat TM and the experts below the root from these runs.
for s in 0 1 2 3 4; do
  python train.py nslkdd --run results/nslkdd/main/seed$s --clauses 500 --T 50 --s 4 --epochs 50 --per_class 4000 --seed $s --procs 2 > results/nslkdd/main/seed$s.out 2>&1
done
# ALeRCE seed 0 was trained with ../alerce_prep/run_tms.py (same configuration).
for s in 1 2; do
  python train.py alerce --run results/alerce/main/seed$s --clauses 1000 --T 100 --s 4 --epochs 50 --per_class 4000 --seed $s --procs 2 > results/alerce/main/seed$s.out 2>&1
done

# 2. TreeTM routing experts.
#    NSL-KDD: leaf-balanced resampling at the root and the attack expert.
for s in 0 1 2 3 4; do ./leafbal.sh nslkdd $s root,attack 500 50 4 50 4000; done
#    ALeRCE: budget-matched root, P_root = P * L / K_root = 4000 * 15 / 3 = 20000.
for s in 0 1 2; do ./budget.sh alerce $s 20000 1000 100 4 50; done

# 3. Clause-budget sweep of the flat TM (seed 0), for the validation selection of C.
#    (The original sweep runs also trained the experts, which the paper does not use.)
for C in 50 100 250 1000; do
  python train.py nslkdd --run results/nslkdd/sweep/C$C --clauses $C --T 50 --s 4 --epochs 50 --per_class 4000 --seed 0 --procs 2 --nodes flat
done
python train.py alerce --run results/alerce/sweep/C250 --clauses 250 --T 100 --s 4 --epochs 50 --per_class 4000 --seed 0 --procs 2 --nodes flat

# 4. Flat TM with the validation-selected C (NSL-KDD: C = 1000; seed 0 is the sweep run).
mkdir -p results/nslkdd/flattuned/seed0
for f in results/nslkdd/sweep/C1000/flat_*; do ln -sfn ../../sweep/C1000/$(basename $f) results/nslkdd/flattuned/seed0/$(basename $f); done
for s in 1 2 3 4; do
  python train.py nslkdd --run results/nslkdd/flattuned/seed$s --clauses 1000 --T 50 --s 4 --epochs 50 --per_class 4000 --seed $s --procs 1 --nodes flat > results/nslkdd/flattuned/seed$s.out 2>&1
done

# 5. Unclipped class sums, evaluation, cost, and analyses.
python unclip.py nslkdd results/nslkdd/main/seed* results/nslkdd/leafbal/seed* results/nslkdd/sweep/C* results/nslkdd/flattuned/seed*
python unclip.py alerce results/alerce/main/seed* results/alerce/budget/seed* results/alerce/sweep/C*
python evaluate.py nslkdd results/nslkdd/main/seed* results/nslkdd/leafbal/seed*
python evaluate.py alerce results/alerce/main/seed* results/alerce/budget/seed*
python flat_tuned.py nslkdd results/nslkdd/flattuned/seed*
python cost.py nslkdd results/nslkdd/leafbal/seed0
python cost.py alerce results/alerce/budget/seed0
python clause_len.py nslkdd leafbal
python interpret.py nslkdd results/nslkdd/main/seed0 dos neptune 2
python analysis.py

# 6. Paper numbers, tables, and figure.
python gen_eaai.py "${1:-paper}"
python figs_eaai.py "${1:-paper}"
