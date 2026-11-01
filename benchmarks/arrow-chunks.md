# Arrow chunk runs

The multi-chunk fits take 1.01–1.03× one chunk with four batches and
1.09–1.16× with 64 batches on this in-memory workload. The ndarray control
is 0.363 → 0.359 seconds. Every path hash agrees across both extensions, all three
Arrow types, the ndarray control and all partitions.

| Input | Chunks | Before (s) | After (s) | After / one chunk |
|---|---:|---:|---:|---:|
| NumPy uint8 | — | 0.363 | 0.359 | — |
| Arrow uint8 | 1 | 0.490 | 0.427 | 1.00× |
| Arrow uint8 | 4 | 0.987 | 0.435 | 1.02× |
| Arrow uint8 | 64 | 1.230 | 0.495 | 1.16× |
| Arrow float64 | 1 | 0.603 | 0.562 | 1.00× |
| Arrow float64 | 4 | 1.004 | 0.577 | 1.03× |
| Arrow float64 | 64 | 1.257 | 0.635 | 1.13× |
| Arrow bool | 1 | 0.780 | 0.720 | 1.00× |
| Arrow bool | 4 | 1.236 | 0.725 | 1.01× |
| Arrow bool | 64 | 1.475 | 0.783 | 1.09× |

The [raw paired measurements](results/arrow-chunks.jsonl) retain all six
times per build, extension hashes, native source hashes and full-path hashes.
Before is `fc38525` (#72); after uses the run scans in this change.

The fixture matches the performance probe: seed 0, 262,144 rows, 300
Bernoulli(0.3) features; the response is the sum of the first five features
plus standard normal noise. Each Gaussian fit computes 20 lambdas with
`lambda_min=0.1` and one thread. Tables partition the same arrays into
1, 4 or 64 chunks, with identical values.

Both native release builds run in the same Python process on shared inputs.
Each is warmed once, then six repetitions alternate AB/BA order; the table
reports medians. Fits must complete the whole path without warnings. Hashes
include centering, scaling, coefficients, intercepts, lambdas, iterations,
rejections and deviance; they check invariance and are never fitted oracles.

Environment: AMD Ryzen 5 7640U, one logical CPU pinned, performance governor,
OpenBLAS/OMP limited to one thread; Python 3.15.0, NumPy 2.5.4, PyArrow 26.0.0,
rustc 1.98.0. Both extensions used the same release profile, with symbols
retained for `cpu-clock:u` profiling. The committed stripping setting is unchanged;
the measured and normal stripped builds have byte-identical `.text` sections.

A [binomial uint8 control](results/arrow-chunks-binomial.jsonl), using the
same continuous response thresholded at its median, also preserves all path
bits: one/four/64 chunks take 1.783/1.786/1.810 seconds after the change
(two paired repetitions). The ndarray control is 0.790 → 0.792 seconds.

The block scans now hold one borrowed chunk fixed for an inner loop.
Crossing blocks retain their original logical lanes and use the existing
reader. No accumulator restarts at a storage boundary: the sweep still
combines its fixed 2,048-row work chunks in order. Gaussian residual updates
also scan runs, keeping their cached sum in serial row order; their old
`column.at(i)` loop searched for the chunk on every row.

Reproduce with [arrow_chunks.py](arrow_chunks.py), pointing to a separately
built baseline extension:

```sh
uv run python benchmarks/arrow_chunks.py \
  --baseline-extension /path/to/baseline/_rust.abi3.so \
  --out timings.jsonl
```

The default is the Gaussian matrix above. `--family binomial --kind u8
--repeats 2` runs the supplemental control. These measurements cover this
synthetic in-memory workload on one CPU.
