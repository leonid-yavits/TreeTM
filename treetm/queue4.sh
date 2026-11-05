#!/bin/bash
cd "$(dirname "$0")"
for s in 0 1 2 3 4; do ./budget.sh nslkdd $s 46000 500 50 4 50; done
for s in 0 1 2; do ./budget.sh alerce $s 20000 1000 100 4 50; done
echo Q4_DONE
