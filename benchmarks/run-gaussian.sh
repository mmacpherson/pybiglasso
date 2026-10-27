#!/usr/bin/env bash
# Gaussian baselines, before residual-sum reuse or other solver changes.
# Override DATASETS, THREADS, REPEATS or CAP for a smoke/longer run.
set -euo pipefail
cd "$(dirname "$0")/.."
datasets=()
for dataset in ${DATASETS:-tall-5k-1k wide-2k-20k}; do
  datasets+=(--dataset "$dataset")
done
threads=()
for count in ${THREADS:-1 $(($(nproc --all) / 2))}; do
  threads+=(--threads "$count")
done
export CONTAINER="${CONTAINER:-docker}"
export R_IMAGE="${R_IMAGE:-pybiglasso-r:r4.6.1}"
if ! "$CONTAINER" image inspect "$R_IMAGE" >/dev/null 2>&1; then
  echo "Missing R image $R_IMAGE; build it with make r-image R_VERSION=4.6.1" >&2
  exit 1
fi
uv run --locked --group bench python benchmarks/path.py --family gaussian \
  "${datasets[@]}" "${threads[@]}" --alpha 1 --alpha 0.5 \
  --repeats "${REPEATS:-3}" --cap "${CAP:-900}" --ratio-cap 1000000 \
  --out "${1:-benchmarks/results/path-gaussian.jsonl}"
