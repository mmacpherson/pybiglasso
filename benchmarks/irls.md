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
serial loop, so dedicated-machine confirmation is needed before treating
the performance trade-off as settled. No timing tolerance or pass/fail
performance threshold is fitted to these measurements.

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

## Weight-sum reuse follow-up

The intercept update and the coordinate sweep both need `sum(w)`, and neither
changes `w`. Compute that same sequential sum once per IRLS iteration and pass
it to both consumers. This removes one row scan without changing arithmetic
order, allocations, or parallel dispatch. A dyadic native test independently
checks the intercept's weighted normal equation and the unchanged weights.

The same alternating DLL harness compared this follow-up against `a6ea67a`
(the parallel-IRLS parent), with four fits per version and thread count on the
development workstation above. Selected medians in seconds:

| Rows × columns | Threads | Parent | Weight-sum reuse |
|---|---:|---:|---:|
| 500,000 × 40 | 1 | 1.8239 | 1.7785 |
| | 6 | 0.9628 | 0.9174 |
| | 12 | 0.9273 | 0.8772 |

No competing build or test process was observed during this short run. Absolute
times differ from the earlier experiment, so these are within-run comparisons,
not evidence that the earlier serial trade-off has disappeared. Smaller cases
were near the timing noise floor. All fit hashes matched across both versions,
repetitions and thread counts. Dedicated-runner confirmation remains necessary
before making a general performance claim.
