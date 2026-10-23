#!/usr/bin/env bash
# The second round of benchmarks, run after run-path.sh:
#
#   1. glum and pybiglasso at every hardware thread (the most generous
#      setting for glum, the one competitor that parallelizes)
#   2. out of core: a design 4x larger than the memory cap, pybiglasso on a
#      memmap vs R biglasso on a file-backed big.matrix, same cap
#
#   nohup benchmarks/run-extra.sh [tag] </dev/null >/dev/null 2>&1 &
#
# OOC_DIR holds the generated matrix (default /scratch/biglasso-ooc; ~24 GB
# including R's file-backed copy).
set -euo pipefail
cd "$(dirname "$0")/.."
tag="${1:+-$1}"
out=benchmarks/results
mkdir -p "$out"
exec >>"$out/extra${tag}.log" 2>&1

all_threads="$(nproc --all)"
uv run -q --group bench python benchmarks/path.py \
  --dataset tall-5k-1k --dataset tall-50k-2k \
  --dataset wide-2k-20k --dataset wide-5k-50k \
  --solver pybiglasso --solver glum \
  --alpha 1.0 --alpha 0.5 --threads 1 --threads "$all_threads" \
  --repeats 3 --cap 900 --ratio-cap 10 \
  --out "$out/path${tag}-allthreads.jsonl"

CONTAINER="${CONTAINER:-docker}" R_IMAGE="${R_IMAGE:-pybiglasso-r:latest}" \
  uv run -q python benchmarks/out_of_core.py run "${OOC_DIR:-/scratch/biglasso-ooc}" \
  --nrow 20000 --ncol 75000 --mem 3G \
  --threads 1 --threads "$((all_threads / 2))"

echo "done"
