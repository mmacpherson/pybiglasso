# Automatic Python thread selection (post-0.4.0)

`n_threads="auto"` and `None` use one thread when X is memory mapped or
fitted rows times input columns is below 10,000,000; otherwise they use
available physical cores (performance cores on Apple silicon).
A positive `RAYON_NUM_THREADS` caps auto mode.
Explicit positive integers override this policy, including for memmaps.
Unknown topology falls back to one. There is no timing, calibration fit,
scan or copy of X. This affects the Python API, not WASM.

Fitted rows include repeated gather indices. This count is conservative for
tiny subsets: masked-row kernels can still walk the full stored column.
Input columns are counted before constant-column filtering or screening.
The proxy cannot know the active set without fitting; it is not a promise
of the fastest setting for every distribution.

## Recorded baseline measurements

`python benchmarks/thread_policy.py` regenerates the comparison table and
cutoff from `results/thread-policy-evidence.jsonl`. It only reads JSON and
computes medians. The extract contains 37 baseline records with unchanged
time samples and original commit, file, line and full-file SHA256 references.

The source datasets are:

- Ryzen 5 5600X: `irls-ryzen-irls.jsonl` and
  `irls-ryzen-standard.jsonl`, using the merged IRLS/weight-sum build.
- M1 Pro: 16 baseline settings from
  `int8-m1-pro-quiet/{binomial-int8,binomial-float64,gaussian-float64}.jsonl`.
  The source bundle's thirteen manifest entries were verified against their
  byte counts and SHA256 hashes.

The M1 calibration subset excludes all four 5k × 1k float64-binomial
settings (alpha 1 and 0.5, each at one and eight workers). Their distinct
timing bands make median thread ratios unsuitable for calibration. The
[complete intended-run source bundle](https://github.com/mmacpherson/pybiglasso/tree/ee29459b51077806da6ff40d9d124a5ac5654137/benchmarks/results/int8-m1-pro-quiet)
retains all 20 baseline settings, including these four.

The M1 experiment admits a run after at least 180 seconds with idle CPU
at least 90% and one-minute load below 3. It uses all six permutations
of three builds, measuring one and eight workers on the same inputs.

For each CPU, build, problem, dtype and alpha, divide the one-worker median
by the six-worker (Ryzen) or eight-worker (M1) median. Use a declared 10%
material-gain margin: a workload is near-neutral when any of its ratios is
below 1.1, and a sustained parallel win when all its ratios are at least 1.1.
All retained baseline settings at these worker counts participate.
Twelve-worker SMT measurements are displayed separately.

The largest near-neutral workload is 5,000,000 entries. The smallest sustained
parallel win is 20,000,000. Their logarithmic midpoint is
`sqrt(5,000,000 * 20,000,000) = 10,000,000`. This gives a coarse size rule
between the measured workloads, not an exact crossover fitted to clocks.

Representative within-build median ratios (one worker / multiple workers):

| CPU | Workload | Workers | Ratio |
| --- | --- | ---: | ---: |
| Ryzen 5 5600X | 50k × 100, IRLS-heavy | 6 | 1.063 |
| Ryzen 5 5600X | 500k × 40, IRLS-heavy | 6 | 1.754 |
| Ryzen 5 5600X | 500k × 40, IRLS-heavy | 12 | 1.768 |
| M1 Pro | 5k × 1k, f64 Gaussian | 8 | 1.011 |
| M1 Pro | 2k × 20k, f64 binomial | 8 | 1.460–1.492 |
| M1 Pro | 2k × 20k, f64 Gaussian | 8 | 1.291 |

Physical cores avoid automatically allocating SMT workers: the Ryzen
50k × 100 IRLS case has a ratio of 1.063 with six workers and 0.774 with
twelve. The M1 measurements compare one and eight workers; the macOS
implementation uses the performance-core count reported by the OS.

For memory-mapped X, the [out-of-core results](../README.md#performance)
support a conservative one-thread default under limited page-cache budgets.
The selector does not estimate RAM residency. Users can request an explicit
worker count for their storage configuration.

## Portable topology without new dependencies

Linux counts distinct physical-core sibling sets among current process CPU
affinity, using [kernel sysfs topology](https://kernel.org/doc/html/latest/admin-guide/cputopology.html)
and its older sibling-list name as fallback. This handles partial SMT affinity
and repeated core IDs across sockets without assuming two threads per core.
It does not interpret cgroup CPU-time quotas: use explicit `n_threads` or the
Rayon ceiling if a quota is smaller than CPU affinity.

macOS queries `hw.perflevel0.physicalcpu`, falling back to `hw.physicalcpu`
when the performance-core key is unavailable or invalid. Each sysctl call
has a one-second timeout, and a successful result is cached. See
[Apple's system-capability documentation](https://developer.apple.com/documentation/kernel/1387446-sysctlbyname/determining_system_capabilities).
Windows uses standard-library ctypes with native
[topology records](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-system_logical_processor_information)
and process affinity, counting physical-core masks intersecting it. Detection
is conservative on multi-processor-group systems: it does not claim cores
outside the calling group. Errors or unsupported systems choose one, not
half the logical count. Explicit integers bypass probing.

Only the pool size passed to the existing adapter changes; numerical kernels
and fixed reduction order are unchanged. Tests cover selection, ownership,
environment/explicit overrides, native topology records, and actual masked
and gathered path bit identity.
