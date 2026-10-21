#!/usr/bin/env bash
# Start benchmarks/run-path.sh detached from the terminal (e.g. over ssh).
#
#   [DATASETS=...] [UNCAPPED=0] [RATIO_CAP=10] benchmarks/start-path.sh [tag]
set -euo pipefail
cd "$(dirname "$0")"
setsid nohup ./run-path.sh "$@" </dev/null >/dev/null 2>&1 &
echo "started run-path.sh (pid $!); log in benchmarks/results/"
