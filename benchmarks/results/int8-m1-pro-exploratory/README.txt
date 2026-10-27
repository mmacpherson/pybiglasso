Apple M1 Pro paired A/B for PR #49 (baseline origin/main 7cdd687, candidate int8-kernels 2b03702).
SHARED-DESKTOP RUN WITH APPS OPEN, NOT A QUIET HOST. Exploratory evidence.
30 settings, 6 timed fits per build per setting (AB/BA x3), threads 1/4/8. 360 timed fits, none dropped.
All fits 100 lambdas, no disallowed warnings; hashes equal across builds, repeats, threads.
A later quiet-gated rerun never started: 1-minute load never below 2 in 30 minutes (quiet-gate-summary.json).
Setup: clone; worktrees baseline=origin/main, candidate=origin/int8-kernels; uv sync --group bench in each;
copy candidate benchmarks/int8.py to int8-macos.py with the CPU lookup replaced by
`sysctl -n machdep.cpu.brand_string` (the portable lookup now in 6a893bb makes this unnecessary); run.sh.
JSONL SHA256: int8-binomial 1c4ca3d3..., float64-binomial 344db71b..., float64-gaussian 7ad6f3a1... (verified on transfer).
