# Logistic benchmark methods and achieved accuracy

The [112 primary setting records](results/path-binomial-memory-conditioned.jsonl)
cover four problems, with a separate [29-setting relaxed cohort](results/path-binomial-relaxed.jsonl)
on the smallest problem. The [out-of-core records](results/out_of_core-logistic.json) have
three observations per setting, in a fixed order. Times describe these inputs
and controls; they are not matched-accuracy or end-to-end loading comparisons.
A completed single observation is marked †. A native timeout is a lower bound
(≥), with no complete-path accuracy claim.
The [provenance file](results/logistic-records-provenance.json) records CPU,
package versions, build settings and the distinct timing and accuracy identities.
The figures use the primary cohort only; their points carry the same accuracy,
control and observation-count qualifications as the table below.

## Inputs, controls and timing

The dataset names give nominal dimensions. Constant features are removed from
the in-memory inputs before fitting; the records also give the used feature
count. Each penalty uses the retained 100-lambda grid prepared by pybiglasso
before timing and supplied to the other adapters. The grid is not regenerated
for an accuracy export. Returned R grids are retained and checked; a grid that
is not exactly shared does not silently receive a matched-lambda objective
comparison. A path that stops early retains unavailable tail metrics; the
records give the number of fitted lambdas for each completed observation.

pybiglasso, R biglasso and R glmnet standardize raw X within the fit. Other
Python solvers receive standardized float64 Fortran inputs. This preparation,
model construction and result export are untimed; validation and defensive
copies performed inside a fit remain timed. Native path APIs or warm starts
are used where supplied. A loop over lambda counts all its native fit intervals
toward the reported time and the same cap. R process/container startup is
excluded from its native interval.

Baseline adapters retain each package's convergence tolerance. pybiglasso's
separately labelled tight rows use `eps=1e-10`; `r_compatible` is an alternate
arithmetic contract. scikit-learn uses `max_iter=10000`; liblinear has an
intercept scaling of 100 and still penalizes the intercept. Its intercept is
not the same optimization contract as an unpenalized intercept. Requested
threads are 1 and 6 where the adapter supports both; R glmnet uses one. Python BLAS/OpenMP limits follow the declared setting. R BLAS remains
sequential while biglasso's explicit `ncores` controls its OpenMP setting.
This restriction does not establish
that another count, including 12, could not be faster. The smallest observed
thread median is descriptive, not an optimal-thread claim.

Three fresh child fits normally supply a median. A prospective slow rule ends
repetition after a completed observation exceeds one third of the common
Python-relative budget, leaving a labelled single observation. The primary
Python cap is the smaller of 900 seconds and
`max(5 seconds, 10 × the accepted pybiglasso 1-thread baseline median)`.
The initial baseline has a 900-second budget before that derived cap exists.
R retains its absolute 900-second native cap; the common repetition budget
still controls the slow rule. The relaxed cohort raises the relative multiplier
to 1000; it is a separate cohort, not additional repeats pooled into primary
medians. Some wide statsmodels settings are omitted prospectively because of
the adapter's dense workspace requirement, rather than reported as timeouts.

## Conditional in-memory recording policy

An external controller stopped the machine's other services, reset clean page
cache once per in-memory setting attempt, and required fixed free-memory
reserves. Each group needed 30 consecutive seconds below 10% total CPU, with
one-second pre/post bookends at most 15%. Missing or invalid required counters
failed closed. Positive swap-out, direct-reclaim scan/steal or allocation-stall
counters rejected the whole attempt. Swap-ins were recorded without a threshold
or causal attribution. Existing swap use alone was not a rejection. There was
no swapoff, forced compaction or transparent-hugepage setting change.

The 30-second gate was chosen prospectively under service isolation. It is not
proven equivalent to a 180-second gate or to thermal stabilization; least-squares
records retain their separately documented protocol. Monitoring costs remain
included and are not subtracted. These results are conditional on that memory
and isolation policy. The repository provides adapters and runners, not the
external controller, its global cache reset or campaign cgroup management;
running the repository scripts alone does not reproduce this recording policy.

The first valid complete setting was kept even if slow or censored, with at most
three attempts and no pooling of partial attempts. Boundary admission refusals
created no fit attempt. One accepted RAM setting followed a pressure rejection
of its preceding whole attempt; rejected timings are excluded. A retained
pre-interval audit annotation was reviewed against the fit's own coverage;
it did not waive an invalid timed interval. No observation was rejected or
selected for its speed, fault count or achieved accuracy.

## Accuracy and export provenance

Independent binary-label logistic loss and active/inactive/intercept KKT
residuals accompany completed paths at their actual lambdas. Relative objective
differences use a pinned, untimed R glmnet accuracy reference only where input,
grid and reference identity match. These are achieved-accuracy descriptions,
not certified gaps to an exact optimum. A reference's zero self-difference is
an identity, not a certificate. A solver's stopping tolerance is not a universal
ceiling on this independent KKT metric.

At α=0.5, R biglasso and pybiglasso's `r_compatible` rows use the legacy update;
they are not matched to the common elastic-net objective. Their times and
achieved metrics remain labelled rather than presented as equal-objective
speed comparisons. Default, tight and alternate-control rows are distinct.

The RAM timings retain source `9b44b7e` and the native artifact built from
`cde044d`. The wide Python accuracy exports use the deterministic standardized
intercept recovery from `6e0dc58`, evaluated in a separate untimed Python 3.14.3
environment. The measured environment was Python 3.14.0. Every regenerated
coefficient equals its retained coefficient byte for byte; all 60 intercept
exports changed somewhere, with maximum absolute change about 8.3e-15.
Thread/repeat identity was checked for this overlay. This is neither a new
timing measurement nor a general cross-version guarantee for `math.fsum`.
All in-memory accuracy metrics are evaluated untimed in that separate environment,
with one BLAS thread; wide-panel metrics use the corrected models.
Exact model, grid, source and build hashes accompany the public provenance.

## Memory-capped out of core

The out-of-core panel uses source `9dde400`, separately identified from the RAM
build. It fits an uncompressed 12 GB float64 20,000 × 75,000 design under a
3 GiB cap per fit, including the process and page cache, with zero swap allowance
for the fit. Python maps its feature file; R attaches a separately prepared
file-backed matrix. Both backing files have retained compression, allocation,
COW and content-integrity evidence. Each file is evicted and verified
nonresident before its fit. This is a 20-lambda lasso comparison.

Three rounds use a fixed Python-1, Python-6, R-1 order. A round is the unit of
acceptance; no partial round is pooled. All recorded rounds completed on their
first attempts. R's single-thread configuration is evidenced by `ncores=1`
and its retained execution environment, not by a measurement that the entire
runtime has one operating-system thread.

Report medians with n=3 and min–max ranges. The approximately 1.1× and 1.4×
R/Python median ratios are descriptive; the one-thread ranges overlap. This is
an I/O-intensive, memory-capped comparison. Read bytes, faults and pressure vary
between observations; the records do not identify a cause. Process read bytes
come from `/proc/self/io` for both packages. Python major faults are a process
delta; R reports cgroup major faults. Refault counters do not count logical
passes, and host device/pressure counters cannot be assigned uniquely to a fit.

The six Python paths are byte-identical in every exported array, including
sparse storage. All three R comparisons pass the retained coefficient/intercept
and lambda tolerances with identical iterations and rejections. Independent
bounded scans outside native timing report achieved objective/KKT values.
Only the accepted recorded comparison supplies these rows; diagnostic,
failed and earlier observations are excluded and not pooled.

<!-- BENCHMARK DATA TABLES -->

## Achieved accuracy beside the observed times

The table uses the smallest observed median among the declared settings,
with single observations labelled † and censored native bounds rounded down
for display (≥). KKT is the largest achieved residual across the returned path.
Objective differences are against the pinned untimed reference only at exactly
shared grids. “Legacy” marks an unmatched elastic-net objective; complete
vectors and every other setting remain in the result files.

| Dataset | α | Solver | Threads | Native s | Completed observations | Fitted λ | Max KKT | Relative objective difference |
|---|---:|---|---:|---:|---:|---|---:|---:|
| tall-50k-2k | 0.5 | glum | 6 | 109.652† | 1 | 100 | 3.15116e-05 | — |
| tall-50k-2k | 0.5 | pybiglasso | 6 | 6.437 | 3 | 100 | 1.22331e-05 | — |
| tall-50k-2k | 0.5 | pybiglasso-rcompat | 6 | 6.314 | 3 | 100 | 0.0234989 | legacy |
| tall-50k-2k | 0.5 | pybiglasso-tight | 6 | 11.462 | 3 | 100 | 4.23305e-07 | — |
| tall-50k-2k | 0.5 | r-biglasso-ssr | 6 | 53.508† | 1 | 100 | 0.0234989 | legacy |
| tall-50k-2k | 0.5 | r-glmnet | 1 | 30.956 | 3 | 100 | 1.0329e-05 | 0 |
| tall-50k-2k | 0.5 | skglm | 1 | 55.009† | 1 | 100 | 9.75264e-05 | — |
| tall-50k-2k | 0.5 | sklearn-saga | 1 | ≥ 110.99 | 0 | — | — | — |
| tall-50k-2k | 1 | glum | 6 | 106.699† | 1 | 100 | 3.88177e-05 | — |
| tall-50k-2k | 1 | pybiglasso | 6 | 6.369 | 3 | 100 | 1.25273e-05 | — |
| tall-50k-2k | 1 | pybiglasso-tight | 6 | 11.571 | 3 | 100 | 3.55621e-07 | — |
| tall-50k-2k | 1 | r-biglasso-adaptive | 6 | 53.551† | 1 | 100 | 1.25273e-05 | — |
| tall-50k-2k | 1 | r-biglasso-ssr | 6 | 53.610† | 1 | 100 | 1.25273e-05 | — |
| tall-50k-2k | 1 | r-glmnet | 1 | 31.373 | 3 | 100 | 1.13174e-05 | 0 |
| tall-50k-2k | 1 | skglm | 1 | 47.076† | 1 | 100 | 9.6513e-05 | — |
| tall-50k-2k | 1 | sklearn-liblinear | 1 | ≥ 110.74 | 0 | — | — | — |
| tall-50k-2k | 1 | sklearn-saga | 1 | ≥ 110.74 | 0 | — | — | — |
| tall-50k-2k | 1 | statsmodels | 1 | ≥ 110.74 | 0 | — | — | — |
| tall-5k-1k | 0.5 | glum | 1 | ≥ 6.91 | 0 | — | — | — |
| tall-5k-1k | 0.5 | pybiglasso | 6 | 0.649 | 3 | 100 | 3.45287e-05 | — |
| tall-5k-1k | 0.5 | pybiglasso-rcompat | 6 | 0.640 | 3 | 100 | 0.0133795 | legacy |
| tall-5k-1k | 0.5 | pybiglasso-tight | 6 | 1.428 | 3 | 100 | 9.35766e-07 | — |
| tall-5k-1k | 0.5 | r-biglasso-ssr | 6 | 4.162† | 1 | 100 | 0.0133795 | legacy |
| tall-5k-1k | 0.5 | r-glmnet | 1 | 1.537 | 3 | 100 | 2.17974e-05 | 0 |
| tall-5k-1k | 0.5 | skglm | 1 | 4.611† | 1 | 100 | 9.85223e-05 | — |
| tall-5k-1k | 0.5 | sklearn-saga | 1 | ≥ 6.91 | 0 | — | — | — |
| tall-5k-1k | 1 | glum | 1 | ≥ 6.78 | 0 | — | — | — |
| tall-5k-1k | 1 | pybiglasso | 6 | 0.633 | 3 | 100 | 4.18363e-05 | — |
| tall-5k-1k | 1 | pybiglasso-tight | 6 | 1.451 | 3 | 100 | 1.07594e-06 | — |
| tall-5k-1k | 1 | r-biglasso-adaptive | 6 | 4.023† | 1 | 100 | 4.18443e-05 | — |
| tall-5k-1k | 1 | r-biglasso-ssr | 6 | 4.028† | 1 | 100 | 4.18443e-05 | — |
| tall-5k-1k | 1 | r-glmnet | 1 | 1.492 | 3 | 100 | 4.9462e-05 | 0 |
| tall-5k-1k | 1 | skglm | 1 | 4.208† | 1 | 100 | 9.74717e-05 | — |
| tall-5k-1k | 1 | sklearn-liblinear | 1 | ≥ 6.78 | 0 | — | — | — |
| tall-5k-1k | 1 | sklearn-saga | 1 | ≥ 6.78 | 0 | — | — | — |
| tall-5k-1k | 1 | statsmodels | 1 | ≥ 6.78 | 0 | — | — | — |
| wide-2k-20k | 0.5 | glum | 1 | ≥ 13.51 | 0 | — | — | — |
| wide-2k-20k | 0.5 | pybiglasso | 6 | 1.162 | 3 | 100 | 2.84651e-05 | — |
| wide-2k-20k | 0.5 | pybiglasso-rcompat | 6 | 1.170 | 3 | 100 | 0.00770539 | legacy |
| wide-2k-20k | 0.5 | pybiglasso-tight | 6 | 1.786 | 3 | 100 | 2.6123e-06 | — |
| wide-2k-20k | 0.5 | r-biglasso-ssr | 6 | 3.101 | 3 | 100 | 0.00770539 | legacy |
| wide-2k-20k | 0.5 | r-glmnet | 1 | 2.546 | 3 | 100 | 3.203e-05 | 0 |
| wide-2k-20k | 0.5 | skglm | 1 | 7.098† | 1 | 100 | 2.25303e-05 | — |
| wide-2k-20k | 0.5 | sklearn-saga | 1 | ≥ 13.51 | 0 | — | — | — |
| wide-2k-20k | 1 | glum | 1 | ≥ 13.09 | 0 | — | — | — |
| wide-2k-20k | 1 | pybiglasso | 6 | 1.116 | 3 | 100 | 2.62042e-05 | — |
| wide-2k-20k | 1 | pybiglasso-tight | 6 | 1.663 | 3 | 100 | 2.47633e-06 | — |
| wide-2k-20k | 1 | r-biglasso-adaptive | 6 | 2.928 | 3 | 100 | 2.62042e-05 | — |
| wide-2k-20k | 1 | r-biglasso-ssr | 6 | 3.040 | 3 | 100 | 2.62042e-05 | — |
| wide-2k-20k | 1 | r-glmnet | 1 | 2.478 | 3 | 100 | 3.13128e-05 | 0 |
| wide-2k-20k | 1 | skglm | 1 | 7.650† | 1 | 100 | 2.48326e-05 | — |
| wide-2k-20k | 1 | sklearn-liblinear | 1 | ≥ 13.09 | 0 | — | — | — |
| wide-2k-20k | 1 | sklearn-saga | 1 | ≥ 13.09 | 0 | — | — | — |
| wide-5k-50k | 0.5 | glum | 1 | ≥ 80.36 | 0 | — | — | — |
| wide-5k-50k | 0.5 | pybiglasso | 6 | 7.020 | 3 | 100 | 1.59058e-05 | — |
| wide-5k-50k | 0.5 | pybiglasso-rcompat | 6 | 7.046 | 3 | 100 | 0.00533715 | legacy |
| wide-5k-50k | 0.5 | pybiglasso-tight | 6 | 8.773 | 3 | 100 | 1.72214e-06 | — |
| wide-5k-50k | 0.5 | r-biglasso-ssr | 6 | 16.491 | 3 | 100 | 0.00533715 | legacy |
| wide-5k-50k | 0.5 | r-glmnet | 1 | 16.139 | 3 | 100 | 2.30762e-05 | 0 |
| wide-5k-50k | 0.5 | skglm | 1 | 33.653† | 1 | 100 | 9.03203e-05 | — |
| wide-5k-50k | 0.5 | sklearn-saga | 1 | ≥ 80.36 | 0 | — | — | — |
| wide-5k-50k | 1 | glum | 6 | 74.641† | 1 | 100 | 6.55964e-06 | — |
| wide-5k-50k | 1 | pybiglasso | 6 | 6.928 | 3 | 100 | 1.85133e-05 | — |
| wide-5k-50k | 1 | pybiglasso-tight | 6 | 8.833 | 3 | 100 | 1.88846e-06 | — |
| wide-5k-50k | 1 | r-biglasso-adaptive | 6 | 15.559 | 3 | 100 | 1.85133e-05 | — |
| wide-5k-50k | 1 | r-biglasso-ssr | 6 | 16.146 | 3 | 100 | 1.85133e-05 | — |
| wide-5k-50k | 1 | r-glmnet | 1 | 15.738 | 3 | 100 | 2.59668e-05 | 0 |
| wide-5k-50k | 1 | skglm | 1 | 34.660† | 1 | 100 | 3.35078e-05 | — |
| wide-5k-50k | 1 | sklearn-liblinear | 1 | ≥ 79.38 | 0 | — | — | — |
| wide-5k-50k | 1 | sklearn-saga | 1 | ≥ 79.38 | 0 | — | — | — |

## Out-of-core observations

Times below are native medians with min–max ranges across three fixed-order
rounds. Per-round reads and major faults are retained as observed, without
selecting a run or inferring a cause. Both packages’ process read counters use
decimal GB; R major faults are cgroup counts and Python’s are process deltas.

| Setting | Native median s (min–max) | Read GB, rounds 1/2/3 | Major faults, rounds 1/2/3 |
|---|---|---|---|
| py1 | 142.516 (137.103–161.819) | 332.683 / 342.760 / 376.676 | 23020 / 25807 / 34858 |
| py6 | 114.582 (113.144–118.227) | 327.938 / 322.044 / 317.134 | 22997 / 20883 / 20006 |
| r1 | 162.783 (156.258–184.835) | 337.977 / 385.816 / 347.890 | 27778 / 40577 / 30517 |
