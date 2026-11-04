# Zen 4 three-build whole-path evidence

AMD Ryzen 5 7640U, 20 settings, 360 timed fits and 60 warm-ups. See
[the analysis](../../int8.md#quiet-gated-zen-4-three-build-run) for methods and
all median comparisons.

`run.py` is the exact executed source. Build the three revisions named in
`provenance.json` in `baseline/`, `ungated/` and `gated/` beside this driver,
using `uv sync --no-dev --python 3.15` in each checkout before starting it.
Run `baseline/.venv/bin/python -u run.py` in a fresh output directory on the
specified AC/performance configuration. The driver discovers its scratch
location; no original machine path is needed.

`results.jsonl` contains every measured setting and fit, without sample
selection. `gate-qualified.jsonl` contains the complete 180-second admission
interval for each setting, copied byte-for-byte from the gate acquisition
log (37 five-second samples per interval). It is an admission-interval
extract, not a continuous whole-session trace. `gate-passed.jsonl` records
the corresponding endpoints. `environment.jsonl` records every setting's
CPU/load/power bookends. Timing marker files delimit the completed run.

`manifest.json` records byte counts and SHA256s for the eight evidence files.
It does not treat fitted outputs as numerical test expectations.
