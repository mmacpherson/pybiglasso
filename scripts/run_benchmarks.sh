#!/usr/bin/env bash
set -euox pipefail

RUN_BENCHMARK="venv/bin/python scripts/benchmark_vs_r_biglasso.py"

RESULTS_FILE=${1:-data/benchmarks.csv}
rm -f ${RESULTS_FILE}

${RUN_BENCHMARK} --nrow 100000 --ncol 100 --nlambda 10 --nreps 1 --alpha 0.8 --dtype float64 --mmap >> ${RESULTS_FILE}
${RUN_BENCHMARK} --nrow 100000 --ncol 100 --nlambda 20 --nreps 1 --alpha 0.8 --dtype float64 --mmap >> ${RESULTS_FILE}
${RUN_BENCHMARK} --nrow 100000 --ncol 100 --nlambda 50 --nreps 1 --alpha 0.8 --dtype float64 --mmap >> ${RESULTS_FILE}
${RUN_BENCHMARK} --nrow 100000 --ncol 200 --nlambda 10 --nreps 1 --alpha 0.8 --dtype float64 --mmap >> ${RESULTS_FILE}
${RUN_BENCHMARK} --nrow 100000 --ncol 200 --nlambda 20 --nreps 1 --alpha 0.8 --dtype float64 --mmap >> ${RESULTS_FILE}
${RUN_BENCHMARK} --nrow 100000 --ncol 200 --nlambda 50 --nreps 1 --alpha 0.8 --dtype float64 --mmap >> ${RESULTS_FILE}
${RUN_BENCHMARK} --nrow 100000 --ncol 500 --nlambda 10 --nreps 1 --alpha 0.8 --dtype float64 --mmap >> ${RESULTS_FILE}
${RUN_BENCHMARK} --nrow 100000 --ncol 500 --nlambda 20 --nreps 1 --alpha 0.8 --dtype float64 --mmap >> ${RESULTS_FILE}
${RUN_BENCHMARK} --nrow 100000 --ncol 500 --nlambda 50 --nreps 1 --alpha 0.8 --dtype float64 --mmap >> ${RESULTS_FILE}
${RUN_BENCHMARK} --nrow 200000 --ncol 100 --nlambda 10 --nreps 1 --alpha 0.8 --dtype float64 --mmap >> ${RESULTS_FILE}
${RUN_BENCHMARK} --nrow 200000 --ncol 100 --nlambda 20 --nreps 1 --alpha 0.8 --dtype float64 --mmap >> ${RESULTS_FILE}
${RUN_BENCHMARK} --nrow 200000 --ncol 100 --nlambda 50 --nreps 1 --alpha 0.8 --dtype float64 --mmap >> ${RESULTS_FILE}
${RUN_BENCHMARK} --nrow 200000 --ncol 200 --nlambda 10 --nreps 1 --alpha 0.8 --dtype float64 --mmap >> ${RESULTS_FILE}
${RUN_BENCHMARK} --nrow 200000 --ncol 200 --nlambda 20 --nreps 1 --alpha 0.8 --dtype float64 --mmap >> ${RESULTS_FILE}
${RUN_BENCHMARK} --nrow 200000 --ncol 200 --nlambda 50 --nreps 1 --alpha 0.8 --dtype float64 --mmap >> ${RESULTS_FILE}
${RUN_BENCHMARK} --nrow 200000 --ncol 500 --nlambda 10 --nreps 1 --alpha 0.8 --dtype float64 --mmap >> ${RESULTS_FILE}
${RUN_BENCHMARK} --nrow 200000 --ncol 500 --nlambda 20 --nreps 1 --alpha 0.8 --dtype float64 --mmap >> ${RESULTS_FILE}
${RUN_BENCHMARK} --nrow 200000 --ncol 500 --nlambda 50 --nreps 1 --alpha 0.8 --dtype float64 --mmap >> ${RESULTS_FILE}
${RUN_BENCHMARK} --nrow 500000 --ncol 100 --nlambda 10 --nreps 1 --alpha 0.8 --dtype float64 --mmap >> ${RESULTS_FILE}
${RUN_BENCHMARK} --nrow 500000 --ncol 100 --nlambda 20 --nreps 1 --alpha 0.8 --dtype float64 --mmap >> ${RESULTS_FILE}
${RUN_BENCHMARK} --nrow 500000 --ncol 100 --nlambda 50 --nreps 1 --alpha 0.8 --dtype float64 --mmap >> ${RESULTS_FILE}
${RUN_BENCHMARK} --nrow 500000 --ncol 200 --nlambda 10 --nreps 1 --alpha 0.8 --dtype float64 --mmap >> ${RESULTS_FILE}
${RUN_BENCHMARK} --nrow 500000 --ncol 200 --nlambda 20 --nreps 1 --alpha 0.8 --dtype float64 --mmap >> ${RESULTS_FILE}
${RUN_BENCHMARK} --nrow 500000 --ncol 200 --nlambda 50 --nreps 1 --alpha 0.8 --dtype float64 --mmap >> ${RESULTS_FILE}
${RUN_BENCHMARK} --nrow 500000 --ncol 500 --nlambda 10 --nreps 1 --alpha 0.8 --dtype float64 --mmap >> ${RESULTS_FILE}
${RUN_BENCHMARK} --nrow 500000 --ncol 500 --nlambda 20 --nreps 1 --alpha 0.8 --dtype float64 --mmap >> ${RESULTS_FILE}
${RUN_BENCHMARK} --nrow 500000 --ncol 500 --nlambda 50 --nreps 1 --alpha 0.8 --dtype float64 --mmap >> ${RESULTS_FILE}
${RUN_BENCHMARK} --nrow 1000000 --ncol 100 --nlambda 10 --nreps 1 --alpha 0.8 --dtype float64 --mmap >> ${RESULTS_FILE}
${RUN_BENCHMARK} --nrow 1000000 --ncol 100 --nlambda 20 --nreps 1 --alpha 0.8 --dtype float64 --mmap >> ${RESULTS_FILE}
${RUN_BENCHMARK} --nrow 1000000 --ncol 100 --nlambda 50 --nreps 1 --alpha 0.8 --dtype float64 --mmap >> ${RESULTS_FILE}
${RUN_BENCHMARK} --nrow 1000000 --ncol 200 --nlambda 10 --nreps 1 --alpha 0.8 --dtype float64 --mmap >> ${RESULTS_FILE}
${RUN_BENCHMARK} --nrow 1000000 --ncol 200 --nlambda 20 --nreps 1 --alpha 0.8 --dtype float64 --mmap >> ${RESULTS_FILE}
${RUN_BENCHMARK} --nrow 1000000 --ncol 200 --nlambda 50 --nreps 1 --alpha 0.8 --dtype float64 --mmap >> ${RESULTS_FILE}
${RUN_BENCHMARK} --nrow 1000000 --ncol 500 --nlambda 10 --nreps 1 --alpha 0.8 --dtype float64 --mmap >> ${RESULTS_FILE}
${RUN_BENCHMARK} --nrow 1000000 --ncol 500 --nlambda 20 --nreps 1 --alpha 0.8 --dtype float64 --mmap >> ${RESULTS_FILE}
${RUN_BENCHMARK} --nrow 1000000 --ncol 500 --nlambda 50 --nreps 1 --alpha 0.8 --dtype float64 --mmap >> ${RESULTS_FILE}
