# Compact column conversion

`Contig::block` reads eight elements from a borrowed column. Mapping the
fixed array by value can make LLVM unpack byte fields from a temporary
integer before converting them to doubles. Indexing the borrowed array with
`from_fn` exposes packed sign-extension loads instead. Each element still
undergoes the same conversion; no floating-point expression, lane assignment,
sum order, allocation or matrix access policy changes.

The native regression expands all signed int8 values independently and checks
the shift/moments passes against scalar formulas at every available SIMD
width, with both bounds settings and chunk/tail cases. Full paths retain
identical bytes, including centering/scaling, losses and iteration/rejection
counts, across the before/after extensions and thread counts. The hashes below
check that relation; they are not stored expected solver values.

## Paired whole-path measurements

These measurements use an AMD Ryzen 5 5600X (6 cores/12 threads), Python 3.14.0,
NumPy 2.5.4 and Rust 1.99.1. The baseline is `bfd527f`; the numerical change is
`fb9e54f`. The paired runs record source `cd8afa0`, and the subsequent numba
control records `810295a`; those later revisions change only the harness or
tests. Both extensions were built serially in separate checkouts.
Their fingerprints are included in every row.

[`int8.py`](int8.py) alternates the actual extensions in one process on the
same read-only Fortran array. It generates the original seed-1 genotype data
with floating effect sizes, then converts X's storage; passing `dtype=int8`
to the generator itself would also truncate its effects and change y.
Both libraries are warmed before timing. Clocks include the complete fit,
validation and standardization, and exclude generation, warm-up and hashing.
Paths use the default 100-lambda binomial grid, at alpha 1 or 0.5, and both
builds use the clipping policy introduced for 0.4.0.

The harness now rejects every warning except the optional numba control's
deprecation, and requires all 100 requested lambdas, during both warm-up and
timed fits. The warning filter is installed before the clock starts; the path
length check follows it. The original measurements predate this guard. A
[separate validation audit](results/int8-ryzen-validation.jsonl), using the
guard in `e711368`, repeated each recorded setting and variant on the exact
same extension files: all 70 validation fits completed 100 lambdas without
disallowed warnings and reproduced their recorded path hashes. These are
validation fits, not replacement timing measurements; all original times
remain unchanged.

There were ordinary background services on the machine: idle CPU was 86–93%
in preflight samples and 85–94% in the final postflight samples. These are
paired shared-machine results, not an otherwise idle-host measurement.
[One-second load samples](results/int8-ryzen-load.csv) cover the complete sets,
including generation and warm-up; the initial since-boot averages are omitted.
No competing compiler was observed during the retained sets. Earlier local
compiler-overlapped probes are excluded. All repetitions and small regressions
in the controls below are retained, including medians up to 2.5% slower.
This shared-machine run does not establish whether such small changes persist
without the background load.

The [int8 target runs](results/int8-ryzen-binomial.jsonl) use four repetitions
per version, in AB/BA order. Times below are median seconds, baseline → candidate:

| Problem | Alpha | 1 thread | 6 threads |
|---|---:|---:|---:|
| 2,000 × 20,000 | 1 | 1.561 → 0.862 | 0.870 → 0.520 |
| 2,000 × 20,000 | 0.5 | 1.657 → 0.925 | 0.968 → 0.569 |
| 5,000 × 50,000 | 1 | 8.806 → 4.926 | 4.285 → 2.639 |
| 5,000 × 50,000 | 0.5 | 9.333 → 5.223 | 4.867 → 2.976 |

The before/after ratio spans 1.62–1.81x on these int8 targets.
[Floating-storage controls](results/int8-ryzen-controls.jsonl) cover tall
5,000 × 1,000 and wide 2,000 × 20,000 designs, both penalties and 1/6 threads.
[Gaussian controls](results/int8-ryzen-gaussian.jsonl) use the same two shapes,
alpha 0.5, and a continuous dyadic response.

| Control | Baseline/candidate time range |
|---|---:|
| Binomial float32 | 0.984–1.015x |
| Binomial float64 | 0.987–1.020x |
| Gaussian int8 | 1.106–1.368x |
| Gaussian float64 | 0.975–1.006x |

## Current numba context

The original issue compared Rust int8 with numba. A
[separate same-input control](results/int8-ryzen-numba.jsonl) uses numba 0.68.0,
six threads and alpha 1. It runs each of the six orders of the three variants
once, after warming all three. This gives six timed fits per variant and
balances their positions. All three reach 100 lambdas and the same total
iterations (559 and 524, respectively).

| Problem | Baseline Rust | Candidate Rust | Numba |
|---|---:|---:|---:|
| 2,000 × 20,000 | 0.877 s | 0.512 s | 0.558 s |
| 5,000 × 50,000 | 4.316 s | 2.679 s | 2.764 s |

The candidate is competitive with numba here; this is not a general ranking
of the two backends. Numba uses different reductions and its hashes are
recorded separately. Only Rust's before/after hashes are required to coincide;
each backend must retain its own repeat/thread identity.

In total, the four files retain 34 records and 292 timed fits. Every median
and ratio can be recomputed from the raw times. No README competitor table
or out-of-core performance claim is updated by these measurements.

## Reproduce

Build `bfd527f` in a separate baseline checkout, then run from the candidate:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python benchmarks/int8.py \
  --baseline-extension ../baseline/src/pybiglasso/_rust.abi3.so \
  --problem wide-2k-20k wide-5k-50k --dtype int8 \
  --threads 1 6 --alpha 1 .5 --repeats 4

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python benchmarks/int8.py \
  --baseline-extension ../baseline/src/pybiglasso/_rust.abi3.so \
  --problem wide-2k-20k wide-5k-50k --dtype int8 \
  --threads 6 --alpha 1 --repeats 6 --with-numba
```

Use `--dtype float32 float64` for the floating controls and `--family gaussian`
for the Gaussian control. `--out` appends raw JSONL; choose a fresh destination.
