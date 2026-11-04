# Zen 3 three-build whole-path evidence

AMD Ryzen 5 5600X, 20 settings, 360 timed fits and 60 warm-ups. See
[the analysis](../../int8.md#median-cpu-gated-zen-3-three-build-run) for methods
and all median comparisons.

`run.py` is the exact executed source. Build the three revisions named in
`provenance.json` in `baseline/`, `ungated/` and `gated/` beside this driver,
using `uv sync --no-dev --python 3.14` in each checkout before starting it.
Run `baseline/.venv/bin/python -u run.py` in a fresh output directory, with
the host's `powersave` governor and `balance_performance` energy preference.
The driver discovers its scratch location; no original machine path is needed.

Before every setting, 37 five-second samples span at least 180 seconds.
The median user + nice + system busy fraction must be at most 15%, with no
compiler, test or benchmark process at any sample. Individual busy samples
may exceed 15%. Load and iowait are not admission conditions.

`results.jsonl` contains every measured setting and fit, without sample
selection. `gate-qualified.jsonl` contains all 740 gate acquisition samples,
including the full 37-sample window for every setting, copied byte-for-byte
from the completed run. `gate-passed.jsonl` records the matching endpoints.
`environment.jsonl` records every setting's CPU/load/policy bookends. Both
governor and energy preference are checked in every gate sample and bookend;
the desktop exposes no AC/battery sensor. Timing marker files delimit the run.

`manifest.json` records byte counts and SHA256s for the eight evidence files.
It does not treat fitted outputs as numerical test expectations.
