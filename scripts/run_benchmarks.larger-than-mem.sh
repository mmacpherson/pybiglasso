#!/usr/bin/env bash
set -euo pipefail


THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
OUTFILE=${THIS_DIR}/../data/benchmarks.larger-than-mem.csv

venv/bin/python scripts/benchmark_vs_r_biglasso.py --nrow 1000000 --ncol 5000 --nlambda 11 --nreps 1 --alpha 0.8 --dtype float32 --mmap >> ${OUTFILE}
venv/bin/python scripts/benchmark_vs_r_biglasso.py --nrow 1000000 --ncol 5000 --nlambda 11 --nreps 1 --alpha 0.8 --dtype float64 --mmap >> ${OUTFILE}
