# Automatic Python thread selection (post-0.4.0)

`n_threads="auto"` and `None` use one thread when X is memory mapped or
fitted rows times input columns is below 10,000,000; otherwise they use
available physical cores. A positive `RAYON_NUM_THREADS` caps auto mode.
Explicit positive integers override this policy, including for memmaps.
Unknown topology falls back to one. There is no timing, calibration fit,
scan or copy of X. This affects the Python API, not WASM.

Fitted rows include repeated gather indices. This count is conservative for
tiny subsets: masked-row kernels can still walk the full stored column.
Input columns are counted before constant-column filtering or screening.
The proxy cannot know the active set without fitting; it is not a promise
of the fastest setting for every distribution.

## Derivation from retained records

`python benchmarks/thread_policy.py` regenerates a table and cutoff from
`results/thread-policy-evidence.jsonl`. It only reads JSON and computes
medians, never imports/runs pybiglasso. Every baseline time is unchanged,
with its original commit/file/line and full-file SHA256. All baseline records
in the named files are included, not just favorable cases. These are #49's
**baseline** times, not its unmerged conversion optimization. Existing Ryzen
IRLS files use the merged post-IRLS/weight-sum build.

For retained six-thread x86 and start-gated eight-thread M1 records, excluding
the banded M1 tall-binomial controls, a gain below 10% is insufficient to
automatically use more cores. This declared material-gain margin avoids
interpreting near-neutral changes as general parallel wins. The largest
workload with any gain below the margin is 5,000,000 entries. The smallest
workload whose recorded gains all clear it is 20,000,000. The logarithmic
midpoint is `sqrt(5,000,000 * 20,000,000) = 10,000,000`. The intervening range
is unmeasured: this is a coarse initial policy, **not a measured crossover**.
No new timing or threshold search was performed.

Representative within-build median ratios (one thread / multiple threads):

| CPU | Workload | Threads | Ratio |
| --- | --- | ---: | ---: |
| Ryzen 5 5600X | 50k × 100, IRLS-heavy | 6 | 1.063 |
| Ryzen 5 5600X | 500k × 40, IRLS-heavy | 6 | 1.754 |
| Ryzen 5 5600X | 500k × 40, IRLS-heavy | 12 | 1.768 |
| Ryzen 5 7640U | 5k × 1k, f64 binomial | 6 | 0.976–1.034 |
| Ryzen 5 7640U | 2k × 20k, f64 binomial | 6 | 1.400–1.439 |
| Ryzen 5 7640U | 2k × 20k, f64 Gaussian | 6 | 1.211 |
| M1 Pro (start-gated, banded) | 5k × 1k, f64 binomial | 8 | 0.887–1.107 |
| M1 Pro (start-gated) | 5k × 1k, f64 Gaussian | 8 | 1.011 |
| M1 Pro (start-gated) | 2k × 20k, f64 binomial | 8 | 1.460–1.492 |
| M1 Pro (start-gated) | 2k × 20k, f64 Gaussian | 8 | 1.291 |

SMT has no general advantage: the Ryzen 50k × 100 IRLS case is 1.063 at six
threads but 0.774 at twelve. Hence physical cores, not logical CPUs. M1 data
uses one/eight workers; its provenance reports eight performance cores, not
a total physical-core census including efficiency cores. Worker residency was
not pinned. It supports the direction of the size rule, not an unmeasured
total-core count or a P/E-core policy. The generic macOS default still counts
all physical cores, so exact performance at that count is not established here.

Qualifications:

- Ryzen 5 5600X: existing IRLS files and #49's retained shared-host baseline
  matrix with its disclosed background services. Compare ratios within the
  same build/problem/file, not absolute times across runs.
- Zen 4: int8 targets finished before overlapping builds. Float64 evidence
  uses **only** the completed timestamped `controls-followup` and
  `gaussian-followup` files. Original overlapped controls and the aborted
  follow-up are excluded, not reclassified as clean. The completed follow-up
  still had persistent 22–32% pre/post I/O wait and Java/desktop activity:
  no recorded build overlap does **not** make it a quiet-machine run.
- M1 Pro: the start-gated three-build bundle replaces the earlier exploratory
  extract. All 20 baseline settings and six samples per setting are retained;
  the source bundle's thirteen manifest entries were checked for exact byte
  counts and SHA256. Only baseline samples enter this report. The idle/load
  gate admitted the run **before** timing, not during fits. Core residency and
  QoS were unmeasured. Tall float64 binomial still has distinct timing bands:
  both alpha settings are reported, but neither calibrates the cutoff. The
  Gaussian tall control remains near-neutral and wide cases favor threads,
  supporting the same coarse cutoff without changing it. This is not a
  noise-free or universally fastest ARM claim.
- Mapped X: [out-of-core results](../README.md#performance) show that more
  threads need not help and can hurt with a tight page-cache budget. A memmap
  may fit RAM, but guessing its residency/available memory makes defaults
  fragile. Auto always uses one; users can override it. ndarray views backed
  by a mapping also count.

## Portable topology without new dependencies

Linux counts distinct physical-core sibling sets among current process CPU
affinity, using [kernel sysfs topology](https://kernel.org/doc/html/latest/admin-guide/cputopology.html)
and its older sibling-list name as fallback. This handles partial SMT affinity
and repeated core IDs across sockets without assuming two threads per core.
It does not interpret cgroup CPU-time quotas: use explicit `n_threads` or the
Rayon ceiling if a quota is smaller than CPU affinity.

macOS queries `hw.physicalcpu` through a one-second bounded, success-cached
sysctl call. It includes P/E physical cores, not a logical-CPU guess; see
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
