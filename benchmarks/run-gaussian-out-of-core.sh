#!/usr/bin/env bash
# Linux/cgroup v2. Apply the quiet-machine protocol before recording timings.
# PREPARE_ONLY=1 prepares data and R storage without fitting; DIAGNOSTIC=1
# runs unrecorded correctness checks. OOC_DIR, MEM and THREADS override the case.
set -euo pipefail
cd "$(dirname "$0")/.."
args=(run "${OOC_DIR:-/scratch/biglasso-ooc-gaussian}"
  --family gaussian --nrow "${NROW:-20000}" --ncol "${NCOL:-75000}"
  --mem "${MEM:-3G}" --nlambda 20)
for count in ${THREADS:-1 $(($(nproc --all) / 2))}; do
  args+=(--threads "$count")
done
if [[ "${PREPARE_ONLY:-0}" == 1 ]]; then args+=(--prepare-only); fi
if [[ "${DIAGNOSTIC:-0}" == 1 ]]; then args+=(--diagnostic); fi
if [[ -n "${OUT:-}" ]]; then args+=(--out "$OUT"); fi
if [[ -n "${BUILD_INFO:-}" ]]; then args+=(--build-info "$BUILD_INFO"); fi
export CONTAINER="${CONTAINER:-docker}"
export R_IMAGE="${R_IMAGE:-pybiglasso-r:r4.6.1}"
if ! "$CONTAINER" image inspect "$R_IMAGE" >/dev/null 2>&1; then
  echo "Missing R image $R_IMAGE; build it with make r-image R_VERSION=4.6.1" >&2
  exit 1
fi
uv run -p "${UV_PYTHON:-3.14}" --locked python benchmarks/out_of_core.py "${args[@]}"
