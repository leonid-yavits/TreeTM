#!/bin/bash
# Unclipped class sums for every trained run (main first, since other variants symlink to it)
cd "$(dirname "$0")"
for ds in nslkdd alerce wos; do
  python unclip.py $ds results/$ds/main/seed*
done
for ds in nslkdd alerce wos; do
  python unclip.py $ds results/$ds/leafbal/seed* results/$ds/budget/seed* results/$ds/sweep/C* $(ls -d results/$ds/abl_root*/seed0 2>/dev/null)
done
echo UNCLIP_DONE
