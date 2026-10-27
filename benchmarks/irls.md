# Tall-binomial IRLS experiment

IRLS computes each row's probability, weight and residual independently.
The tall path now shares that work, including portable exp/log, across the
existing Rayon pool. The loss still adds individual terms in the original
sequential order, omitting masked rows. Chunk subtotals would change rounding
and could tip saturation. The old probability buffer holds the parallel loss
terms; there is no extra row buffer or copy of X.

The serial path keeps its two original passes. A fused serial exp/log loop
was slower, so it was discarded. The Rayon implementation is outlined from
the serial function. The IRLS dispatch threshold is 16,384 rows, separate
from the sweep's threshold; changing sweep thresholds is a later experiment.

## Whole-fit measurement

AMD Ryzen 5 7640U, six physical cores, twelve logical threads; Linux,
Python 3.15.0, NumPy 2.5.4, rustc 1.98.0. Baseline: main `9ed067e` (the same
tree as the locally built `16b964f`). Candidate: this PR. Four fits per
version and thread count, alternating AB/BA order in one process with the
same X/y and Python wrapper. Medians in seconds, including standardization
and the whole 30-lambda elastic-net path:

| Rows × columns | Threads | Baseline | Candidate |
|---|---:|---:|---:|
| 10,000 × 100 | 1 | 0.0325 | 0.0340 |
| | 6 | 0.0292 | 0.0276 |
| | 12 | 0.0303 | 0.0317 |
| 50,000 × 100 | 1 | 0.1650 | 0.1748 |
| | 6 | 0.1581 | 0.1015 |
| | 12 | 0.1806 | 0.1213 |
| 500,000 × 40 | 1 | 1.4327 | 1.4670 |
| | 6 | 1.3573 | 0.8296 |
| | 12 | 1.4231 | 0.7881 |

These low-dimensional tall problems emphasize IRLS; they do not predict the
same speedup on wide data or paths with much larger active sets. This was a
development workstation, not an isolated benchmark runner. The serial
medians remain 2–6% slower in this sample despite preserving the original
serial loop; dedicated-machine confirmation follows below rather than
treating these samples as settled. No timing tolerance or pass/fail
performance threshold is fitted to these measurements.

### Dedicated CPU confirmation

A separate run on an AMD Ryzen 5 5600X (six physical cores, twelve logical
threads), Python 3.14.0, NumPy 2.5.4 and rustc 1.99.1 compared main `9ed067e`,
IRLS `a6ea67a` and the weight-sum follow-up `5f7e1a9` (PR #42). All three
release extensions were freshly built with the same toolchain. Every fit
used the same wrapper and input arrays. IRLS-heavy cases used all six orders
of the three revisions, six timed fits each; the standard cases used three
cyclic orders, so each revision occupied each position once. Each revision
also had an untimed warm-up for each problem and thread count.

Selected median seconds (the IRLS column isolates this PR):

| Problem | Alpha | Threads | Main | IRLS | IRLS + weight reuse |
|---|---:|---:|---:|---:|---:|
| IRLS 500,000 × 40 | 0.5 | 1 | 1.3387 | 1.3869 | 1.3533 |
| | 0.5 | 6 | 1.4187 | 0.8012 | 0.7717 |
| | 0.5 | 12 | 1.4243 | 0.7735 | 0.7654 |
| Standard tall 50k × 2k | 1 | 1 | 9.5773 | 9.6273 | 9.7020 |
| | 1 | 6 | 6.3255 | 5.8975 | 5.9300 |
| | 0.5 | 1 | 9.7043 | 9.7382 | 9.7153 |
| | 0.5 | 6 | 6.3556 | 6.1595 | 6.0059 |

The multithreaded tall gain is confirmed, but the serial trade-off remains:
IRLS alone was 3.6% slower on the low-dimensional 500k case, versus about
0.4–0.5% on the standard 50k cases. The follow-up reduces that low-dimensional
serial gap to 1.1%, but is not consistently faster on the other problems.
The standard small and wide cases were roughly level, with individual
regressions as well as improvements. Short multithreaded IRLS-heavy cases
had substantial timing scatter; do not generalize their median ratios.

Standard cases use `benchmarks/path.py`'s data construction and shared lambda
grid, with explicit `family="binomial"`, alpha 1 and 0.5, and 1/6 threads.
The timer surrounds `biglasso()` (including standardization), not the
benchmark's subsequent coefficient-scale conversion. Input constants are
dropped as in that benchmark. No competitor timings or README rankings
are updated by this experiment.

The timing window was coordinated, but the machine still ran background
services. Recent `vmstat` samples were 95–97% idle before the IRLS set,
94–99% afterward/before the standard set, and 96–97% after the standard
set. Load averages before and after each comparison are included with
every sample in [the IRLS results](results/irls-ryzen-irls.jsonl) and
[the standard results](results/irls-ryzen-standard.jsonl). All path fields,
including centering, scaling, deviance, iterations and rejections, matched
bit for bit across versions, repetitions and thread counts. The hashes are
same-input comparisons, not expected solver values or fixture oracles.

Build both revisions, then run from the candidate environment:

```sh
python benchmarks/irls.py --repeats 4 \
  --baseline-extension ../baseline/src/pybiglasso/_rust.abi3.so
```

The script loads both compiled extensions and switches the wrapper's Rust
module between fits, leaving input arrays shared. It prints every timing
and asserts exact output identity across versions, repetitions and thread
counts. Hashes cover lambdas, coefficients and their sparse indices,
intercepts, iterations, rejections and deviance. They are runtime comparison
artifacts, not committed expected solver values.

## Numerical checks

The native kernel test checks each row's clipped Bernoulli equations and
the sequential loss sum independently, with and without a mask, at 1, 2 and
6 threads. Sizes straddle the row-parallel threshold and have chunk/vector
tails. The cross-platform snapshot grid also includes a 16,391-row problem
so CI exercises IRLS on worker threads on every supported OS.

Local pre/post probes additionally matched all bits for float64, float32
and int8, with full rows, a permuted mask, a small gathered subset and
repeated row indices. Independent R fixtures and the live R comparison
remain the correctness oracle; the same-input comparisons establish that
this optimization preserves existing results.
