#!/usr/bin/env bash
# Run the in-memory path benchmark, logging to benchmarks/results/.
#
#   nohup benchmarks/run-path.sh [tag] </dev/null >/dev/null 2>&1 &
#
# An optional tag is appended to the output names (path-TAG.jsonl). Competitors are stopped at RATIO_CAP (default
# 10) times pybiglasso's single-threaded time on each problem; the smallest
# problem also runs effectively uncapped (skip with UNCAPPED=0), to show how
# large the gaps get. DATASETS="a b" and SOLVERS="a b" limit the capped run;
# THREADS="1 6 12" sets the thread counts (default: 1 and half the hardware
# threads).
# R biglasso solvers need CONTAINER/R_IMAGE (see tests/support/rcompare.py); their
# scratch data goes to R_WORKDIR (default: benchmarks/results/tmp, on disk).
set -euo pipefail
cd "$(dirname "$0")/.."
tag="${1:+-$1}"
ratio_cap="${RATIO_CAP:-10}"
threads=()
for t in ${THREADS:-1 $(($(nproc --all) / 2))}; do
  threads+=(--threads "$t")
done
out=benchmarks/results
mkdir -p "$out"
exec >>"$out/path${tag}.log" 2>&1
datasets=()
for d in ${DATASETS:-tall-5k-1k tall-50k-2k wide-2k-20k wide-5k-50k}; do
  datasets+=(--dataset "$d")
done
solvers=()
for s in ${SOLVERS:-}; do
  solvers+=(--solver "$s")
done
export R_WORKDIR="${R_WORKDIR:-$PWD/$out/tmp}"
mkdir -p "$R_WORKDIR"

common=(--alpha 1.0 --alpha 0.5 "${threads[@]}" --repeats 3 --cap 900)

if [[ "${UNCAPPED:-1}" != 0 ]]; then
  uv run -q --group bench python benchmarks/path.py \
    --dataset tall-5k-1k "${common[@]}" --ratio-cap 1000 \
    --out "$out/path${tag}-uncapped.jsonl"
fi

uv run -q --group bench python benchmarks/path.py \
  "${datasets[@]}" ${solvers[@]+"${solvers[@]}"} "${common[@]}" \
  --ratio-cap "$ratio_cap" \
  --out "$out/path${tag}.jsonl"

echo "done"
