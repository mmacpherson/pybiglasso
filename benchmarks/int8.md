# Compact column conversion

`Contig::block` reads eight elements from a borrowed column. On x86, mapping
the fixed array by value can make LLVM unpack byte fields from a temporary
integer before converting them to doubles. Indexing the borrowed array with
`from_fn` exposes packed sign-extension loads instead. Aarch64 retains the
original array mapping: indexed conversion was slower in the M1 Pro runs below.
Each element still
undergoes the same conversion; no floating-point expression, lane assignment,
sum order, allocation or matrix access policy changes.

The native regression expands all signed int8 values independently and checks
the shift/moments passes against scalar formulas at every available SIMD
width, with both bounds settings and chunk/tail cases. Full paths retain
identical bytes, including centering/scaling, losses and iteration/rejection
counts, across the before/after extensions and thread counts. The hashes below
check that relation; they are not stored expected solver values.

## CPU-idle-gated M1 Pro three-build run

On the measured int8 problems, aarch64-gated median durations range from
0.38% longer to 2.53% shorter than baseline. The
ungated indexed form is 6.65–11.69% longer in all eight settings. The small
gated differences are retained, not assumed to be noise or a universal
neutrality guarantee. The x86 gains on two Ryzen chips are recorded below.

The [provenance](results/int8-m1-pro-quiet/provenance.json) records Apple M1
Pro (eight performance cores), macOS 26.7.2, arm64 Python 3.13.13, NumPy 2.5.4
and Homebrew rustc 1.99.1 (`58cf8f9c3`, LLVM 23.1.3). Baseline is `eeff8da`,
ungated is `2b03702`, and gated is `352c6ac` (including the gate in `f61a985`).
All three extensions were built before the gate; their SHA256s are recorded.
No custom `RUSTFLAGS` or `target-cpu` was reported.

The [start gate](results/int8-m1-pro-quiet/gate.jsonl) samples the second CPU
sample of `top -l 2 -n 0 -s 5` every fifteen seconds, requiring idle at least
90% and one-minute load below 3 for at least 180 seconds of qualifying samples.
It [passed](results/int8-m1-pro-quiet/gate-passed.json) at 22:44:29 UTC on
2026-10-27 after 195 seconds, with idle 90.7% and load 1.13. Timing bookends
are 22:44:34–23:09:21 UTC. This is a start-gate-admitted run, not continuous
idle gating during fits: ordinary macOS services remained. Core residency
was not measured or pinned, and there is no high-QoS claim. The
[per-set bookends](results/int8-m1-pro-quiet/set-environment.jsonl) retain
load, power and thermal reports: all twenty power snapshots show AC/100%,
with no recorded thermal/performance warnings. Load goes from 1.44 before the
first set to 3.80 after the last; post-set load includes benchmark work.

One Python wrapper loads the three actual extensions and fits the same
seed-1 read-only Fortran array at 1 and 8 threads. Each setting uses all six
permutations of baseline/ungated/gated once, giving six timed fits per build
with each build twice in each position. All 20 settings, 360 timed fits and
60 warm-ups are retained. Every fit completes all 100 lambdas; the
[executed harness](results/int8-m1-pro-quiet/int8-threeway.py) rejects warnings
other than the unused optional numba control's deprecation. Generation,
warm-up, hashing and UTC bookkeeping are outside the fit clock. Hashes agree
across builds, repetitions and thread counts.

Median seconds, B = baseline, U = ungated, G = gated:

| Int8 problem | Alpha | Threads | B | U | G |
|---|---:|---:|---:|---:|---:|
| 2,000 × 20,000 | 1 | 1 | 1.699684 | 1.876271 | 1.702690 |
| 2,000 × 20,000 | 1 | 8 | 0.871508 | 0.935604 | 0.874795 |
| 2,000 × 20,000 | 0.5 | 1 | 1.817221 | 2.000411 | 1.813093 |
| 2,000 × 20,000 | 0.5 | 8 | 0.993310 | 1.059359 | 0.990480 |
| 5,000 × 50,000 | 1 | 1 | 10.194382 | 11.369099 | 10.098489 |
| 5,000 × 50,000 | 1 | 8 | 4.676605 | 5.223508 | 4.618729 |
| 5,000 × 50,000 | 0.5 | 1 | 11.218697 | 12.393641 | 10.934378 |
| 5,000 × 50,000 | 0.5 | 8 | 5.385018 | 6.006070 | 5.380967 |

The [raw int8 lists](results/int8-m1-pro-quiet/binomial-int8.jsonl) also give
minima: gated durations are about 0.1–3.0% shorter than baseline, whereas the
ungated minima remain longer. Fastest-three medians in the
[complete report](results/int8-m1-pro-quiet/report.txt) are a uniform
sensitivity summary, not mode classification or proof of performance-core
execution. No sample is dropped or selected for a replacement run.

[Wide float64 binomial](results/int8-m1-pro-quiet/binomial-float64.jsonl)
and [Gaussian controls](results/int8-m1-pro-quiet/gaussian-float64.jsonl) have
gated median durations within 0.7% of baseline. Tall float64 binomial controls
still have distinct timing bands, despite the start gate: baseline/gated
median ratios span 0.914–1.331. Minima do not solve unequal sampling of those
bands: at alpha 0.5 and one thread, baseline has no observation near 3.2 s,
while gated has five. This does not establish stable tall-control parity or
identify the cause; all samples remain in the records.

The [manifest](results/int8-m1-pro-quiet/manifest.json) pins all thirteen
evidence files, including the editorial report, original scripts,
[execution notes](results/int8-m1-pro-quiet/execution.txt) and harness patch.
Local auditing verified every hash/byte count, record count, permutation,
median, ratio, timestamp order and recorded path identity without rerunning
the benchmarks. Only the report's prose has been edited; raw data and executed
sources retain their original bytes.

## Architecture gate

Assembly probes cross-compiled with Rust 1.98 / LLVM 22.1.8 for
`aarch64-apple-darwin` (`target-cpu=apple-m1`) and generic
`aarch64-unknown-linux-gnu` show different lowering. These are not the measured
M1 run's compiler and flags. In the probes, the old array map
loads eight int8 values with one 64-bit load and extracts them in registers;
`from_fn` emits eight byte loads. Both use eight scalar integer-to-double
conversions, not a packed NEON conversion. This persists in the actual score,
moment and fused shift/moment loops. More load/address work is a plausible
explanation, not a hardware-counter attribution of the measured slowdown.

Only this small conversion function has a compile-time architecture branch:
`aarch64` keeps the original `rows.map(Elem::f)`; other targets keep `from_fn`.
The conversions, lane assignments and all floating-point operations remain
identical. There is no runtime branch, new allocation or matrix copy. The
gated ARM probe bodies match baseline assembly after normalizing only symbol
hashes, so the gate retains the old ARM lowering rather than asking ARM to pay
for the x86 gain. This is assembly evidence, not a new whole-fit timing claim.
CI's cross-platform fit-bit comparison exercises both branches with Fortran
int8 and float64 inputs, both families and both penalties on each of its
chunk/tail, wide and parallel-IRLS designs: 24 fits per platform. The comparator
requires every case, so a float64-only snapshot cannot satisfy the check.

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
paired shared-machine results.
[One-second load samples](results/int8-ryzen-load.csv) cover the complete sets,
including generation and warm-up; the initial since-boot averages are omitted.
All repetitions and small regressions
in the controls below are retained, including medians up to 2.5% slower.
These measurements do not establish the cause of such small changes.

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

## Quiet-gated Zen 4 three-build run

The [provenance](results/int8-zen4-quiet/provenance.json) records an AMD Ryzen
5 7640U (6 physical cores/12 threads), Python 3.15.0, NumPy 2.5.4 and Rust
1.98.0 / LLVM 22.1.8. Baseline is main `5065100`, ungated is `2b03702`, and
gated is `d5e0c8b`. All three extensions were built before the first gate with
`uv sync --no-dev --python 3.15`, then loaded by one baseline Python wrapper.
Their exact source revisions and binary SHA256s are recorded. The compiler
and Python differ from the Zen 3 set, so this is not an isolated comparison
of CPU generations.

Before every setting, five-second `/proc/stat` deltas must show at most 10%
user + nice + system CPU, with no detected compiler, test or benchmark worker,
continuously for 180 seconds. Iowait and load average are recorded but do not
affect admission. The [twenty admitted intervals](results/int8-zen4-quiet/gate-qualified.jsonl)
retain all 37 samples in each qualifying interval; the
[pass records](results/int8-zen4-quiet/gate-passed.jsonl) record their endpoints.
AC power and the performance governor are checked at every gate sample and
in the [one-second CPU/load bookends](results/int8-zen4-quiet/environment.jsonl)
before and after every setting. Bookend user/system CPU spans 3.33–7.78%.
The first timing marker is 2026-10-27 23:43:57 UTC and completion is
2026-10-28 01:27:56 UTC. The gate admits each setting; it does not monitor
background CPU continuously during the fits.

The [executed driver](results/int8-zen4-quiet/run.py) uses the same seed-1
read-only Fortran inputs, original floating effects, and dyadic Gaussian
response as above. Each setting warms all three builds, then uses every
baseline/ungated/gated permutation once: six timed fits per build, twice in
each position. All 20 settings, 360 timed fits and 60 warm-ups are retained
in the [raw records](results/int8-zen4-quiet/results.jsonl), including UTC
start/end times for every fit. Generation, warm-up, hashing and UTC bookkeeping
are outside the fit clock. Every warning is fatal and every fit must complete
100 lambdas. Path hashes and iteration counts agree across builds,
repetitions and 1/6 threads.

Int8 median seconds, B = baseline, U = ungated, G = gated:

| Problem | Alpha | Threads | B | U | G |
|---|---:|---:|---:|---:|---:|
| 2,000 × 20,000 | 1 | 1 | 1.149906 | 0.847923 | 0.842463 |
| 2,000 × 20,000 | 1 | 6 | 0.652210 | 0.485625 | 0.491206 |
| 2,000 × 20,000 | 0.5 | 1 | 1.220298 | 0.897854 | 0.895341 |
| 2,000 × 20,000 | 0.5 | 6 | 0.722431 | 0.536846 | 0.540207 |
| 5,000 × 50,000 | 1 | 1 | 6.524844 | 4.874266 | 4.868373 |
| 5,000 × 50,000 | 1 | 6 | 3.197500 | 2.411411 | 2.403147 |
| 5,000 × 50,000 | 0.5 | 1 | 6.907092 | 5.159312 | 5.154129 |
| 5,000 × 50,000 | 0.5 | 6 | 3.576949 | 2.707395 | 2.694797 |

Baseline/gated ratios span 1.327–1.365x on these int8 targets; baseline/ungated
ratios span 1.321–1.359x. Both retain the indexed conversion on x86.

Float64 control median seconds, baseline → gated:

| Family | Problem | Alpha | 1 thread | 6 threads |
|---|---|---:|---:|---:|
| Binomial | 5,000 × 1,000 | 1 | 3.102194 → 3.085336 | 3.033210 → 3.055058 |
| Binomial | 5,000 × 1,000 | 0.5 | 2.135752 → 2.148400 | 2.096562 → 2.116675 |
| Binomial | 2,000 × 20,000 | 1 | 1.497089 → 1.496649 | 1.026277 → 1.019479 |
| Binomial | 2,000 × 20,000 | 0.5 | 1.581058 → 1.579238 | 1.084883 → 1.089735 |
| Gaussian | 5,000 × 1,000 | 0.5 | 1.202710 → 1.182065 | 1.142284 → 1.136609 |
| Gaussian | 2,000 × 20,000 | 0.5 | 2.211592 → 2.203663 | 1.828677 → 1.810536 |

Baseline/gated ratios span 0.9905–1.0067x for float64 binomial and
1.0036–1.0175x for float64 Gaussian. The tall six-thread elastic-net binomial
median is about 0.96% longer, not discarded or assumed to be noise. These
are measured ranges, not universal performance bounds or out-of-core claims.
The [manifest](results/int8-zen4-quiet/manifest.json) pins every evidence file;
all recorded medians, ratios, timestamps, permutations, admission intervals
and path identities were independently checked without rerunning fits.

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
