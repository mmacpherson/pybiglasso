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

## Median-CPU-gated Zen 3 three-build run

The [provenance](results/int8-zen3-quiet/provenance.json) records an AMD Ryzen
5 5600X (6 physical cores/12 threads), Python 3.14.0, NumPy 2.5.4 and Rust
1.99.1 / LLVM 23.1.3. Baseline is main `5065100`, ungated is `2b03702`, and
gated is `8128489`. All three extensions were built before the first gate
with `uv sync --no-dev --python 3.14`; their exact source revisions and binary
SHA256s are recorded. One baseline Python wrapper loads all three extensions.

The host uses the `powersave` governor and `balance_performance` energy
preference on all twelve logical CPUs. Both are recorded in provenance and
checked at every gate sample and bookend. The desktop exposes no AC/battery
sensor. Its regular application and monitoring services remain running.

Before each setting, the gate collects five-second `/proc/stat` samples over
180 seconds: 37 samples including both endpoints. The median user + nice +
system busy fraction must be at most 15%, and no compiler, test or benchmark
process may appear at any sample in that window. Individual busy samples may
exceed 15%; admission uses the median of the complete window. Iowait and load
are recorded but do not affect admission. All
[740 admission samples](results/int8-zen3-quiet/gate-qualified.jsonl) and
[20 pass records](results/int8-zen3-quiet/gate-passed.jsonl) are retained.
The admitted window medians span 8.63–9.60%. One-second
[CPU/load bookends](results/int8-zen3-quiet/environment.jsonl) precede and follow
every setting; their busy fractions span 3.55–18.16%. The gate admits settings;
it does not monitor background CPU continuously during fits.

Timing starts at 2026-10-28 03:05:29 UTC and ends at 04:21:21 UTC. All
20 settings, 360 timed fits and 60 warm-ups are retained in the
[raw records](results/int8-zen3-quiet/results.jsonl), including per-fit UTC
intervals. Every setting warms all three builds, then runs all six
baseline/ungated/gated permutations once: six timed fits per build, twice in
each position. The [executed driver](results/int8-zen3-quiet/run.py) rejects
every warning and requires all 100 lambdas in warm-ups and timed fits.
Path hashes and iteration counts agree across builds, repetitions and 1/6
threads; the hashes check identity, not external expected solver values.

Inputs are seed-1 read-only Fortran arrays. The genotype generator uses its
original floating effect sizes before X is converted to int8 storage;
Gaussian uses the same continuous dyadic response as the other three-build
runs. The fit clock includes validation, standardization and the complete
path, excluding generation, warm-up, hashing and UTC bookkeeping.

Int8 median seconds, B = baseline, U = ungated, G = gated:

| Problem | Alpha | Threads | B | U | G |
|---|---:|---:|---:|---:|---:|
| 2,000 × 20,000 | 1 | 1 | 1.546570 | 0.873216 | 0.861139 |
| 2,000 × 20,000 | 1 | 6 | 0.881457 | 0.551922 | 0.590317 |
| 2,000 × 20,000 | 0.5 | 1 | 1.654180 | 0.927049 | 0.928497 |
| 2,000 × 20,000 | 0.5 | 6 | 0.976873 | 0.609124 | 0.587592 |
| 5,000 × 50,000 | 1 | 1 | 8.741718 | 4.838774 | 4.805325 |
| 5,000 × 50,000 | 1 | 6 | 4.341952 | 2.609067 | 2.616984 |
| 5,000 × 50,000 | 0.5 | 1 | 9.341159 | 5.164698 | 5.131425 |
| 5,000 × 50,000 | 0.5 | 6 | 4.827652 | 2.920433 | 2.937895 |

Baseline/gated ratios span 1.493–1.820x; baseline/ungated ratios span
1.597–1.809x. Both use the indexed conversion on x86. At 2,000 × 20,000,
alpha 1 and six threads, the gated median is 6.96% longer than ungated while
remaining 1.493x faster than baseline; that difference is retained, not
assumed to be noise.

Float64 control median seconds, baseline → gated:

| Family | Problem | Alpha | 1 thread | 6 threads |
|---|---|---:|---:|---:|
| Binomial | 5,000 × 1,000 | 1 | 2.981246 → 2.941526 | 2.908529 → 2.906966 |
| Binomial | 5,000 × 1,000 | 0.5 | 2.071206 → 2.084123 | 2.023834 → 2.011012 |
| Binomial | 2,000 × 20,000 | 1 | 1.580772 → 1.600400 | 1.340692 → 1.334999 |
| Binomial | 2,000 × 20,000 | 0.5 | 1.622992 → 1.626633 | 1.435907 → 1.382778 |
| Gaussian | 5,000 × 1,000 | 0.5 | 1.087825 → 1.089677 | 1.076457 → 1.092858 |
| Gaussian | 2,000 × 20,000 | 0.5 | 2.197399 → 2.205904 | 2.006180 → 1.975193 |

Baseline/gated ratios span 0.9877–1.0384x for float64 binomial and
0.9850–1.0157x for float64 Gaussian. The wide single-thread binomial lasso
median is 1.24% longer; the tall six-thread Gaussian median is 1.52% longer.
All samples and small regressions remain in the records. These measured
ranges are not universal performance bounds or out-of-core claims.
The [manifest](results/int8-zen3-quiet/manifest.json) pins every evidence file;
counts, orders, timestamps, gate-window medians, path identities, timing
summaries and unchanged build fingerprints were checked without rerunning
fits.

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

The [Zen 3 bundle](results/int8-zen3-quiet/README.md) and
[Zen 4 bundle](results/int8-zen4-quiet/README.md) describe their pinned builds,
output layout and exact driver commands. Use a fresh output directory and
the stated host policy and gate rule. The
[M1 execution notes](results/int8-m1-pro-quiet/execution.txt) record its build,
gate and run commands; its scripts are included in that bundle.
