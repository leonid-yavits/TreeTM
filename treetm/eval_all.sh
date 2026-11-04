#!/bin/bash
# Evaluate every finished run (all experts and the flat TM have final masks) that lacks treetm_eval.json
cd "$(dirname "$0")"
for ds in nslkdd alerce wos; do
  n=$(python -c "import json;from taxonomy import Taxonomy;print(len(Taxonomy(json.load(open('work/$ds/meta.json'))['taxonomy']).internal)+1)")
  for r in results/$ds/main/seed* results/$ds/leafbal/seed* results/$ds/budget/seed* results/$ds/sweep/C*; do
    [ -d $r ] || continue
    c=$(ls $r/*_final_masks.npz 2>/dev/null | wc -l)
    if [ "$c" -eq "$n" ] && [ ! -f $r/treetm_eval.json ]; then python evaluate.py $ds $r 2>/dev/null | head -1; fi
  done
done
