#!/bin/bash
cd "$(dirname "$0")"; mkdir -p results/nslkdd/main results/nslkdd/leafbal results/nslkdd/budget results/nslkdd/sweep
for s in 0 1 2 3 4; do [ -f results/nslkdd/leafbal/seed$s/attack_final_masks.npz ] || ./leafbal.sh nslkdd $s root,attack 500 50 4 50 4000; done
for s in 0 1 2 3 4; do [ -f results/nslkdd/budget/seed$s/root_final_masks.npz ] || ./budget.sh nslkdd $s 46000 500 50 4 50; done
echo QB_DONE
