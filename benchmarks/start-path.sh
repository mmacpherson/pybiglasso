#!/usr/bin/env bash
# Start a benchmark script detached from the terminal (e.g. over ssh).
#
#   [DATASETS=...] [UNCAPPED=0] [RATIO_CAP=10] benchmarks/start-path.sh [tag]
#   SCRIPT=run-extra.sh benchmarks/start-path.sh [tag]
set -euo pipefail
cd "$(dirname "$0")"
script="${SCRIPT:-run-path.sh}"
setsid nohup "./$script" "$@" </dev/null >/dev/null 2>&1 &
echo "started $script (pid $!); log in benchmarks/results/"
