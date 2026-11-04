#!/usr/bin/env bash
# Defaults to a non-reportable smoke. Coordinate an idle CPU before PHASE=reportable.
set -euo pipefail
cd "$(dirname "$0")/.."
export CONTAINER="${CONTAINER:-docker}"
export R_BENCH_IMAGE="${R_BENCH_IMAGE:-pybiglasso-bench:r4.6.1-snpnet}"
if ! "$CONTAINER" image inspect "$R_BENCH_IMAGE" >/dev/null 2>&1; then
  echo "Missing optional image; build it first with make r-bench-image" >&2
  exit 1
fi
datasets=()
for dataset in ${DATASETS:-smoke}; do datasets+=(--dataset "$dataset"); done
threads=()
for count in ${THREADS:-1}; do threads+=(--threads "$count"); done
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
uv run --locked --group bench python benchmarks/cv.py \
  "${datasets[@]}" "${threads[@]}" --alpha 1 --alpha 0.5 \
  --early-stopping on --early-stopping off --folds "${FOLDS:-3}" \
  --nlambda "${NLAMBDA:-100}" --lambda-min "${LAMBDA_MIN:-0.05}" \
  --repeats "${REPEATS:-3}" --phase "${PHASE:-smoke}" \
  --out "${1:-/tmp/bigstatsr-smoke.jsonl}"
