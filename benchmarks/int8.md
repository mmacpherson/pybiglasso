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
original evidence files byte-for-byte, including the report, scripts,
[execution notes](results/int8-m1-pro-quiet/execution.txt) and harness patch.
Local auditing verified every hash/byte count, record count, permutation,
median, ratio, timestamp order and recorded path identity without rerunning
the benchmarks. Earlier exploratory M1 records remain only in Git history.

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

## Zen 4 / AVX-512 follow-up

A separate run uses an AMD Ryzen 5 7640U (6 physical cores/12 threads), on
mains power with the performance governor and energy preference. Runtime
dispatch was independently checked to select pulp V4 (AVX-512), rather than
just checking the advertised CPU flags. Python is 3.15.0, NumPy 2.5.4 and Rust
1.98.0; the compiler and Python differ from the Zen 3 set above, so this is
not an isolated comparison of CPU generations.

The candidate is the main-integrated `53b88b1`; the baseline extension's source
is `378df63`, whose Rust sources are byte-identical to main `7cdd687` (including
the clipping change). The latter additionally contains test-only changes.
The same candidate Python wrapper calls both extensions. Each setting has
six repetitions per extension: three complete AB/BA pairs, after warming
both. The warning/full-path guard was active throughout these measurements.
Every fit reaches 100 lambdas; all before/after and thread-count hashes agree,
and agree with the corresponding Zen 3 records above. Extension fingerprints
are recorded in every row.

The [int8 targets](results/int8-zen4-binomial.jsonl) give these median seconds,
baseline → candidate:

| Problem | Alpha | 1 thread | 6 threads |
|---|---:|---:|---:|
| 2,000 × 20,000 | 1 | 1.129 → 0.841 | 0.657 → 0.493 |
| 2,000 × 20,000 | 0.5 | 1.350 → 0.927 | 0.957 → 0.735 |
| 5,000 × 50,000 | 1 | 8.003 → 5.865 | 4.185 → 3.085 |
| 5,000 × 50,000 | 0.5 | 7.007 → 5.198 | 3.728 → 2.714 |

The int8 gain spans 1.30–1.46x. The original float64 controls use tall
5,000 × 1,000 and wide 2,000 × 20,000 problems at both thread counts:

| Original control (build-overlapped; see correction below) | Baseline/candidate time range |
|---|---:|
| [Binomial](results/int8-zen4-controls.jsonl), alpha 1 and 0.5 | 0.998–1.047x |
| [Gaussian](results/int8-zen4-gaussian.jsonl), alpha 0.5 | 1.006–1.028x |

`Contig::block` is generic, so the expression change is also instantiated for
float64. The earlier Zen 3 Gaussian median up to 2.5% slower remains in the
record; this follow-up does not establish it was noise. Small differences can
reflect code generation/layout as well as background load. The binomial
control range is an observation, not a statistical bound on that variation.

[Environment records](results/int8-zen4-environment.json) capture mains,
governor, versions and source revisions before and after; all are unchanged.
Java/CCH and ordinary desktop/agent services remained running. The original
claim that both pybiglasso agents held all builds/tests was incorrect (see
the independently recovered build history below). There was also elevated
background I/O wait: preflight idle CPU was 55–66%
with 25–32% iowait; postflight idle was 61–71% with 23–32% iowait. These are
in-memory, paired shared-workstation measurements, not quiet-machine or
out-of-core claims. All original samples remain unchanged.

The [load record](results/int8-zen4-load.csv) retains 692 continuous one-second
samples, plus ten preflight and ten postflight samples. The bookend streams
overlap the continuous stream; each omits its initial since-boot average.
All 20 settings and 240 timed fits are retained in the three JSONL files.

### Build-overlap correction

Docker BuildKit's retained history and logs subsequently established two
overlaps, both after the last int8 target had finished at 21:01:06 UTC:

| Build | UTC interval | Work observed in its logs |
|---|---|---|
| bigstatsr | 21:02:31–21:02:51 | Binary-package installation; no source compilation |
| snpnet | 21:04:53–21:06:13 | Package installation including source compilation |

Subsequently supplied command-execution records corroborate those build
intervals and identify three additional jobs. These are reported whole-command
bounds, not independently verified per-stage or fit timings:

| Reported command | UTC interval | Conservative continuous-load samples |
|---|---|---:|
| Dependency sync / ruff | 21:01:17.271–21:01:17.883 | 366–371 |
| Python CV smoke | 21:04:54.484–21:05:00.414 | 584–593 |
| Targeted pytest / ruff | 21:05:58.983–21:06:01.426 | 648–654 |

All also start after the final int8 target. They reinforce the controls'
contaminated status without changing the follow-up scope.

These overlap the original float64 binomial and Gaussian controls, so their
small differences above cannot establish an uncontaminated floating-control
result. The int8 targets preceded both builds; they are not rerun on this
evidence. The [overlap manifest](results/int8-zen4-overlap.json) preserves the
precise Docker timestamps and original result endpoints.

The [load annotations](results/int8-zen4-overlap.csv) conservatively mark
continuous samples 441–464 for bigstatsr and 582–666 for snpnet. Original
vmstat output has no timestamps: annotations use each log's creation time
plus its sample index, with a two-second expansion for scheduling uncertainty.
They do not claim exact per-fit attribution. Original JSONL also has no fit
wall-clock bookends, so all twelve control settings are scheduled for one
balanced follow-up, not selected individual fast/slow repetitions. The
harness now records per-fit UTC epoch-nanosecond bookends outside the measured
region, to allow direct attribution in the follow-up. Original times, hashes,
and load samples are preserved, not replaced.

The first timestamped follow-up was aborted rather than used as clean
evidence. Its twenty-second preflight passed (median user+system 8%, maximum
19%, no compiler/test process observed), but a new unrelated multi-core
Python job started at 21:27:10 UTC. Only the benchmark worker and its own
monitor were stopped; the unrelated job was not changed. The
[two completed settings](results/int8-zen4-controls-followup-aborted.jsonl)
retain all 24 recorded timed fits, and the
[timestamped load](results/int8-zen4-followup-aborted-load.csv) retains 123
samples. The [attempt manifest](results/int8-zen4-followup-aborted.json) marks
17 fit windows as potentially overlapping the new job. The next setting was
interrupted before its repetitions were written, so it has no durable
per-fit record. This is an incomplete, contaminated attempt, not a
replacement control set. At that point, a coordinated quiet window was still
required; no uncontaminated-control conclusion is drawn from that attempt.

### Completed timestamped control follow-up

After a separately authorized quiet window, all twelve float64 settings were
repeated at `df2f381`, using the same extension fingerprints and six fits per
version (three complete AB/BA pairs). The
[binomial](results/int8-zen4-controls-followup.jsonl) and
[Gaussian](results/int8-zen4-gaussian-followup.jsonl) records retain all 144
timed fits, with per-fit UTC epoch-nanosecond intervals. Every fit completes
100 lambdas without disallowed warnings and reproduces its original
before/after/thread path hash. Original and aborted records remain separate.

Median seconds, baseline → candidate:

| Family | Problem | Alpha | 1 thread | 6 threads |
|---|---|---:|---:|---:|
| Binomial | 5,000 × 1,000 | 1 | 3.155 → 3.144 | 3.232 → 3.319 |
| Binomial | 5,000 × 1,000 | 0.5 | 2.190 → 2.178 | 2.118 → 2.135 |
| Binomial | 2,000 × 20,000 | 1 | 1.474 → 1.456 | 1.024 → 1.028 |
| Binomial | 2,000 × 20,000 | 0.5 | 1.534 → 1.554 | 1.096 → 1.097 |
| Gaussian | 5,000 × 1,000 | 0.5 | 1.198 → 1.185 | 1.166 → 1.150 |
| Gaussian | 2,000 × 20,000 | 0.5 | 2.232 → 2.214 | 1.843 → 1.843 |

Baseline/candidate ratios span 0.974–1.012x for binomial and
0.99995–1.01397x for Gaussian. The tall six-thread binomial lasso candidate
is about 2.7% slower; no floating-storage neutrality or universal speedup is
claimed. As above, generic code generation/layout can affect float64, and
these small changes are not assumed to be noise.

The [environment audit](results/int8-zen4-followup-environment.json) records
unchanged mains/performance settings, versions and extension source revisions.
No Docker build in retained BuildKit history overlaps this window, and sampled
process checks observed no competing compiler/test worker. The agents held
their builds/tests, while Java/CCH and ordinary desktop services remained.
Persistent background I/O wait is not a quiet-machine claim: preflight idle
CPU was 60–71%, with 22–30% iowait; postflight idle was 62–70%, with 24–32%
iowait. The [timestamped load](results/int8-zen4-followup-load.csv) has 20
preflight, 325 continuous and ten postflight samples, omitting each stream's
since-boot average. Postflight overlaps the continuous stream's end.

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
