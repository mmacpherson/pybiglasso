#!/bin/bash
set -euo pipefail
cd /tmp/pybiglasso-m1/candidate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
{
 date -u
 sysctl -n machdep.cpu.brand_string hw.model hw.perflevel0.physicalcpu
 sw_vers
 rustc --version --verbose
 .venv/bin/python --version
 git rev-parse HEAD
 git -C ../baseline rev-parse HEAD
 uptime
 pmset -g batt
 pmset -g therm
} > ../environment-before.txt
/usr/bin/caffeinate -i -w $$ &
vm_stat 1 > ../vm-stat.txt &
VM_PID=$!
trap 'kill "$VM_PID" 2>/dev/null || true' EXIT
BASE=../baseline/src/pybiglasso/_rust.abi3.so
.venv/bin/python ../int8-macos.py --baseline-extension "$BASE" --problem wide-2k-20k wide-5k-50k --dtype int8 --threads 1 4 8 --alpha 1 .5 --repeats 6 --out ../int8-binomial.jsonl > ../int8-binomial.stdout
.venv/bin/python ../int8-macos.py --baseline-extension "$BASE" --problem tall-5k-1k wide-2k-20k --dtype float64 --threads 1 4 8 --alpha 1 .5 --repeats 6 --out ../float64-binomial.jsonl > ../float64-binomial.stdout
.venv/bin/python ../int8-macos.py --baseline-extension "$BASE" --family gaussian --problem tall-5k-1k wide-2k-20k --dtype float64 --threads 1 4 8 --alpha .5 --repeats 6 --out ../float64-gaussian.jsonl > ../float64-gaussian.stdout
{
 date -u
 uptime
 pmset -g batt
 pmset -g therm
} > ../environment-after.txt
