# Least-squares benchmark methods and achieved accuracy

The in-memory results are in
[120 setting records](results/path-gaussian-memory-conditioned.jsonl), including
all declared thread counts. The out-of-core results contain
[all nine observations](results/out_of_core-gaussian.json), three per setting.
The README selects the smallest observed native-time median among each
solver's declared thread counts. This is descriptive selection, not evidence
that the count is generally optimal. A completed single observation is marked
†; a censored result is a lower bound (≥), with no complete-path accuracy claim.

## Inputs, controls and timing

All in-memory solvers receive the same generated problem and analytic
100-lambda grid, for lasso and elastic net (α=0.5). Each uses its own default
convergence tolerance. pybiglasso, R biglasso and glmnet standardize raw X within
the timed fit. Other Python solvers receive pre-standardized float64 Fortran X;
scikit-learn additionally receives centered X and y. That preparation and
intercept recovery are untimed. Its default `copy_X=True` defensive copy and
native Gram construction are timed. Native path/warm-start APIs are used where
provided. These are default-control comparisons on prepared inputs, not
end-to-end loading or matched-accuracy comparisons.

The settings are 1, 6 and 12 threads, except that skglm, R glmnet and statsmodels
use one thread. BLAS/OpenMP limits match the setting, including scikit-learn's
Gram build. Three fresh child fits supply a median. The prospective slow rule
ends repetition after a completed observation exceeds one third of the common
Python-relative budget; that observation is labelled n=1 even for an R solver.
Python's native-time cap is the smaller of 900 s and
`max(5 s, 10 × pybiglasso's accepted 1-thread median)` for that problem/α.
The initial pybiglasso 1-thread baseline uses the initial 900 s budget before
that derived cap exists. R retains its absolute 900 s native-time cap. A timeout is not a measured fit
of exactly the cap duration. Imports, model construction, benchmark input
preparation and result export lie outside native timing, including the cap
clock. Statsmodels' per-lambda intervals accumulate toward the same cap.

The CPU is an AMD Ryzen 5 5600X (6 cores, 12 threads), with 64 GB RAM. Python is
3.14.0, NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1, skglm 0.5, glum 3.4.1,
statsmodels 0.15.0 and R 4.6.1. Exact package/R versions, thread settings and
build identities accompany the records. The native release wheel uses Rust
1.99.1, LLVM 23.1.1, generic x86_64, optimization level 3, fat LTO and one
codegen unit, with no RUSTFLAGS. The separately recorded development compiler
is not the compiler that built that wheel.

## Conditional in-memory recording policy

These measurements used an **external recording controller**. The repository
supplies the solver adapters and benchmark runners, not that controller's
conditioning, machine-wide page-cache resets or campaign cgroup management.
Running `benchmarks/run-gaussian.sh` alone does not reproduce this policy.
Results are conditional on the policy; it excludes allocation/reclaim costs
that can occur in ordinary shared-machine use.

Before each setting attempt, the controller waited for previous owned work to
exit, ran `sync` and dropped clean page cache once, then required at least
32 GiB MemFree and 40 GiB MemAvailable. It created a fresh dataset parent,
verified inputs/builds and required 180 consecutive seconds below 10% total
CPU use. Before each child it required free memory of at least
`8 × requested_X_bytes + 4 GiB`; at native start, at least 4 GiB. There was no
extra cache reset or input reconstruction between repetitions in a group.
One-second pre/post CPU bookends had to be at most 15%.

Required VM counters were validated at both ends, failing closed on missing,
invalid or decreasing values. Positive swap-out, direct-reclaim scan/steal or
any declared allocation-stall counter rejected the whole setting attempt.
Swap-ins were recorded without a threshold or attribution. Existing swap
usage was not itself a rejection. Transparent huge pages were left at
`madvise`, with defrag `defer+madvise`; no swapoff, forced compaction or THP
setting change was used. Neither low faults, huge-page success nor fast times
were acceptance criteria. A pressure increment is system-wide evidence, not
proof that the measured solver caused it.

The first valid complete group was accepted, even if slow or censored. A group
normally had at most three attempts, retaining all failures and never mixing
samples between attempts. Two kinds of reviewed exceptions must be disclosed:

- On 50,000 × 2,000, two exhausted admission-only windows were reopened after
  the same 180-second idle and memory-reserve start conditions held. Three
  attempts before pybiglasso 1t and three before skglm 1t failed admission;
  none reached a timed fit. Accepted checkpoints and all earlier failures
  remained unchanged.
- On 2,000 × 20,000, R biglasso 12t exhausted a mixed window: one timed fit was
  rejected for two allocation-stall events, followed by two reserve-admission
  failures. A separately authorized new window supplied that cell. This is
  a timed-rejection budget exception, not an admission-only restart or an
  unqualified three-attempt limit across restarts.

Across the four datasets there were 15 rejected attempts: pybiglasso 5,
skglm 3, R biglasso 3, glum 1 and scikit-learn 3. The other rejections were
CPU-bookend or declared pressure failures, followed by whole-attempt retries
within their original windows. None was rejected for slow time, large fault
count or accuracy ranking. The counts include the six admission-only failures
and the three mixed-window failures above. All accepted records, including
valid single-slow and censored outcomes, remain available.

The machine's other services were stopped during the out-of-core comparison,
but not during the in-memory campaign. Their environmental protocols differ
and must not be conflated.

## Achieved accuracy

KKT residuals are evaluated independently from the data and returned
coefficients, in response units. Relative objective difference is the largest
absolute relative difference across the path against a newly computed,
pinned, **untimed accuracy-only** R glmnet reference. A zero for that reference
is an identity, not a certificate of the exact optimum. These differences are
not certified objective gaps.

Every completed path reached all 100 lambdas without warnings. The generated
constant columns leave 973 used features in the smallest dataset. For glmnet's
response rescaling by sᵧ, the supplied α′ is
`α / (α + (1−α)sᵧ)` and λ′ is `λ(α + (1−α)sᵧ)`, giving the common objective;
the lasso's α and λ are unchanged.

The table uses the README's selected setting, retaining the achieved KKT and
objective difference beside its time. All other thread settings are in the
JSONL. † means one completed observation; ≥ means censored, with zero completed
path observations. Bounds below are rounded down for display.

| Dataset | α | Solver | Threads | Native seconds | Completed observations | Max KKT | Relative objective difference |
|---|---:|---|---:|---:|---:|---:|---:|
| tall-5k-1k | 1 | pybiglasso | 6 | 0.14 | 3 | 9.96263e-05 | 1.79695e-06 |
| tall-5k-1k | 1 | R biglasso | 6 | 0.68 | 3 | 9.96263e-05 | 1.79695e-06 |
| tall-5k-1k | 1 | glmnet (R) | 1 | 0.28 | 3 | 0.000105457 | 0 |
| tall-5k-1k | 1 | skglm | 1 | 1.36 | 3 | 9.89308e-05 | 1.87807e-06 |
| tall-5k-1k | 1 | glum | 6 | 1.51 | 3 | 1.03194e-05 | 1.87797e-06 |
| tall-5k-1k | 1 | scikit-learn | 6 | 0.06 | 3 | 5.00118e-05 | 1.80183e-06 |
| tall-5k-1k | 1 | statsmodels | 1 | ≥ 5.00 | 0 | — | — |
| tall-5k-1k | 0.5 | pybiglasso | 6 | 0.15 | 3 | 7.58111e-05 | 1.83284e-06 |
| tall-5k-1k | 0.5 | R biglasso | 6 | 0.72 | 3 | 7.58111e-05 | 1.83284e-06 |
| tall-5k-1k | 0.5 | glmnet (R) | 1 | 0.26 | 3 | 8.49491e-05 | 0 |
| tall-5k-1k | 0.5 | skglm | 1 | 1.35 | 3 | 9.96761e-05 | 1.84303e-06 |
| tall-5k-1k | 0.5 | glum | 6 | 1.68† | 1 | 9.73072e-06 | 1.84292e-06 |
| tall-5k-1k | 0.5 | scikit-learn | 6 | 0.07 | 3 | 5.4097e-05 | 1.76514e-06 |
| tall-5k-1k | 0.5 | statsmodels | 1 | ≥ 5.00 | 0 | — | — |
| tall-50k-2k | 1 | pybiglasso | 6 | 1.26 | 3 | 1.0887e-05 | 2.01114e-08 |
| tall-50k-2k | 1 | R biglasso | 6 | 4.49 | 3 | 1.0887e-05 | 2.01114e-08 |
| tall-50k-2k | 1 | glmnet (R) | 1 | 4.71 | 3 | 1.18529e-05 | 0 |
| tall-50k-2k | 1 | skglm | 1 | ≥ 17.51 | 0 | — | — |
| tall-50k-2k | 1 | glum | 6 | 9.77† | 1 | 2.28896e-06 | 2.02111e-08 |
| tall-50k-2k | 1 | scikit-learn | 6 | 0.84 | 3 | 4.48314e-05 | 4.69764e-07 |
| tall-50k-2k | 1 | statsmodels | 1 | ≥ 17.51 | 0 | — | — |
| tall-50k-2k | 0.5 | pybiglasso | 6 | 1.40 | 3 | 1.69467e-05 | 2.07117e-08 |
| tall-50k-2k | 0.5 | R biglasso | 6 | 4.65 | 3 | 1.69467e-05 | 2.07117e-08 |
| tall-50k-2k | 0.5 | glmnet (R) | 1 | 4.78 | 3 | 1.69466e-05 | 0 |
| tall-50k-2k | 0.5 | skglm | 1 | 16.80† | 1 | 9.59995e-05 | 4.8986e-07 |
| tall-50k-2k | 0.5 | glum | 6 | 10.94† | 1 | 2.14739e-06 | 2.11413e-08 |
| tall-50k-2k | 0.5 | scikit-learn | 6 | 0.85 | 3 | 4.77324e-06 | 1.87944e-08 |
| tall-50k-2k | 0.5 | statsmodels | 1 | ≥ 19.35 | 0 | — | — |
| wide-2k-20k | 1 | pybiglasso | 12 | 0.21 | 3 | 7.81038e-06 | 1.92679e-11 |
| wide-2k-20k | 1 | R biglasso | 6 | 0.92 | 3 | 7.81038e-06 | 1.92677e-11 |
| wide-2k-20k | 1 | glmnet (R) | 1 | 1.75 | 3 | 1.92845e-07 | 0 |
| wide-2k-20k | 1 | skglm | 1 | 6.82† | 1 | 9.80529e-05 | 2.12e-08 |
| wide-2k-20k | 1 | glum | 1 | 1.72 | 3 | 5.24751e-06 | 1.8755e-11 |
| wide-2k-20k | 1 | scikit-learn | 1 | 3.19† | 1 | 0.0123656 | 0.000220422 |
| wide-2k-20k | 1 | statsmodels | 1 | ≥ 8.69 | 0 | — | — |
| wide-2k-20k | 0.5 | pybiglasso | 12 | 0.22 | 3 | 3.6498e-06 | 4.50334e-12 |
| wide-2k-20k | 0.5 | R biglasso | 6 | 0.93 | 3 | 3.6498e-06 | 4.50334e-12 |
| wide-2k-20k | 0.5 | glmnet (R) | 1 | 1.74 | 3 | 9.75775e-08 | 0 |
| wide-2k-20k | 0.5 | skglm | 1 | 7.09† | 1 | 1.26809e-05 | 6.00652e-11 |
| wide-2k-20k | 0.5 | glum | 1 | 1.73 | 3 | 3.27927e-06 | 4.50334e-12 |
| wide-2k-20k | 0.5 | scikit-learn | 1 | 3.72† | 1 | 1.96958e-06 | 3.32937e-12 |
| wide-2k-20k | 0.5 | statsmodels | 1 | ≥ 8.76 | 0 | — | — |
| wide-5k-50k | 1 | pybiglasso | 12 | 0.90 | 3 | 5.42689e-06 | 7.93219e-12 |
| wide-5k-50k | 1 | R biglasso | 6 | 5.57 | 3 | 5.42689e-06 | 7.93219e-12 |
| wide-5k-50k | 1 | glmnet (R) | 1 | 11.18† | 1 | 3.6128e-08 | 0 |
| wide-5k-50k | 1 | skglm | 1 | ≥ 32.06 | 0 | — | — |
| wide-5k-50k | 1 | glum | 1 | 12.43† | 1 | 4.39096e-06 | 5.7772e-12 |
| wide-5k-50k | 1 | scikit-learn | 1 | 22.62† | 1 | 0.0121605 | 0.00022406 |
| wide-5k-50k | 1 | statsmodels | 1 | ≥ 32.06 | 0 | — | — |
| wide-5k-50k | 0.5 | pybiglasso | 12 | 0.92 | 3 | 2.21393e-06 | 2.08446e-12 |
| wide-5k-50k | 0.5 | R biglasso | 6 | 5.49 | 3 | 2.21393e-06 | 2.08446e-12 |
| wide-5k-50k | 0.5 | glmnet (R) | 1 | 11.09† | 1 | 1.63726e-08 | 0 |
| wide-5k-50k | 0.5 | skglm | 1 | ≥ 32.61 | 0 | — | — |
| wide-5k-50k | 0.5 | glum | 1 | 12.40† | 1 | 2.21393e-06 | 2.08446e-12 |
| wide-5k-50k | 0.5 | scikit-learn | 1 | 25.68† | 1 | 2.21393e-06 | 2.08446e-12 |
| wide-5k-50k | 0.5 | statsmodels | 1 | ≥ 32.61 | 0 | — | — |

## Out-of-core recording policy

The file is an uncompressed, NOCOW btrfs float64 matrix, 20,000 × 75,000,
12,000,000,000 logical bytes. Read-only `compsize` reported 100% uncompressed
extents for both Python's memmap and R's separate file-backed `big.matrix`.
The filesystem's zstd mount option did not compress these files. Each feature
inode was evicted, then verified to have zero resident bytes before its fit.
A 3 GiB cgroup cap included the process and page cache; child swap stayed zero.
The same 20-lambda lasso path used the same convergence settings for both.

Three rounds ran in fixed order Py1 → Py6 → R1, with the original CPU gates.
All nine fits were retained without retry. Each cell reports its median and
min–max spread, n=3. The order is not randomized or balanced. Read bytes come
from process storage-I/O accounting; refault bytes come from the cgroup's
file-workingset refault count times page size. Neither counts exact logical
matrix passes, and refaults do not count every reread. The raw observations
include both metrics, faults, cgroup anon/current/peak and swap. Round-2 Py6
major faults are 137413, retained without exclusion or causal inference.

The machine's other services were stopped while the benchmark lock was held.
A read-only environment-status log was sampled once per second and refreshed
approximately every 30 seconds. These samples are not independent one-second
checks or proof that services stayed stopped continuously. The audit required
that no unrelated process held the lock and no unrelated container ran,
with no missing reads, polling gaps or unexplained status changes. The status
file could be at most 90 seconds old; its maximum observed age was
30.600916936 s. All nine fits had no flags. The benchmark's own R containers
were checked against their recorded name, immutable ID, PID and command.
Cleanup, lock release and governor/EPP restoration were verified afterward.

All six Python paths are byte-identical across threads and rounds, including
sparse storage. All three R comparisons pass the existing rtol=1e-9,
atol=1e-11, with exactly equal iterations and rejection counts. Loss is not
compared because of the predeclared difference in final-sweep timing.
The disk-bound native times are roughly even; the descriptive R/Py median
ratios are 0.9484603122 at 1 thread and 1.0924508937 at 6 threads. They are not
a general speed claim. R's prepared backing-file creation is untimed; its
native fit and process wall including attachment/setup are recorded separately.

## Source provenance and rerunning

[Provenance](results/gaussian-records-provenance.json) records the measured
source commit `9a5adbdfbe50b03457a378975bcfb4de22995fcf`, native source commit,
wheel identity and the landing harness's Git blob identities. The measured
and transplanted harness differ only in history: their full Git trees are
both `88f84a1a74cf50cc13c61c1317689510d11fb882`. Every harness file is identical.
The published JSONL encodes unavailable censored durations as JSON `null`,
retaining the exact cap and censor flag. Accepted measured values are unchanged.

The repository's `benchmarks/run-gaussian.sh` and
`benchmarks/run-gaussian-out-of-core.sh` run the corresponding comparisons.
The external controller's memory conditioning and background-service exclusion
must be arranged separately for this recording policy; these scripts alone do
not provide it. Do not combine results from a different environment into these
conditional ratios.
