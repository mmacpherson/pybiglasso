# Optional file-backed CV and genotype benchmarks

Phase 1 is code, correctness checks and smoke runs only. There are **no new
reportable timings** in this change, and the README's existing tables are
unchanged. Historical results from the old unmerged branch are not reused.
Reportable runs will use the AMD Ryzen 5 5600X benchmark CPU after a coordinated
quiet window, with before/after load records and medians of at least three
independent-process repetitions.

## Separate image

```bash
make r-bench-image  # optional snpnet stage, includes bigstatsr
# Or build only bigstatsr; neither command modifies the fixture image:
make r-bench-image BENCH_R_TARGET=bigstatsr
```

The image uses rocker R 4.6.1 and an explicit P3M snapshot (2026-06-24), not
conda or P3M's moving `latest`. An R profile also applies that snapshot to
GitHub dependency installation and new R sessions. The smoke installed
bigstatsr 1.6.2, snpnet 1.4.1 and glmnet 5.0. The fixture Dockerfile, fixtures
and stored benchmark results are untouched. `BUILD_JOBS` defaults to two.

snpnet's GitHub-only dependencies retain the original branch's commit pins:
glmnetPlus `755b0e58`, cindex/plink-ng `5a44c663`, snpnet `f7fca345`.
PLINK 2 alpha6 (2026-09-19) is checksum-pinned. The optional stage is Linux
x86-64 only. cindex builds with native CPU flags: rebuild the image on the
benchmark CPU rather than moving the locally built image to a different CPU.

## Cross-validation protocol

```bash
# Functional only: shared small inputs, both mixes, both early-stop modes.
REPEATS=1 NLAMBDA=60 benchmarks/run-bigstatsr.sh /tmp/cv-smoke.jsonl
# A two-worker functional check:
THREADS=2 REPEATS=1 NLAMBDA=60 benchmarks/run-bigstatsr.sh /tmp/cv-smoke-2.jsonl
```

Both packages read exactly the same column-major training file, as a read-only
NumPy memmap or a bigstatsr FBM. Preparation holds out a stratified 20% test
set, creates shared stratified fold IDs, and removes columns that are constant
in any fold's training rows **before timing, for both solvers**. Test rows are
not used in this filter. Input and fold fingerprints are recorded.

The matrix crosses solver, alpha, thread count and early stopping on/off.
bigstatsr receives `n.abort=10` or a huge integer (internally clamped below
the integer maximum). pybiglasso receives `early_stopping=10` or `None`.
The classifier explicitly passes `family="binomial"`; the harness selects
Rust and checks the returned family. big_spLogReg is intrinsically binomial.
All runs use an explicitly configured lambda count (default 100) and minimum
ratio (default 0.05); bigstatsr's minimum path length before early stopping
remains 50. Python has no corresponding minimum. These are not identical
stopping rules. Record actual fold path lengths, not just requested lengths.

This is an end-to-end **model-selection comparison**, not identical penalized
objectives: [bigstatsr CMSA](https://privefl.github.io/bigstatsr/reference/big_spLogReg.html)
averages selected fold models using fold-local grids. Python evaluates a common
full-data grid and refits on all training rows. Their default inner tolerances
also differ (`eps=1e-5` versus `1e-7`). Compare test log loss and AUC alongside
time; do not require their coefficients, selected lambdas or supports to agree.

The clock covers the complete fit/CV call, including standardization. File
creation, attachment/imports, Python warm-up, coefficient export and test-set
scoring are excluded. R's exported averaged coefficients are checked against
its own `predict(..., proba=FALSE)` outside the clock. Non-early-stopped folds
must complete the requested path. The misleadingly named `summary()$all_conv`
only says every fold ended for "No more improvement"; actual optimization
failures are rejected using warnings and per-fold iteration maxima instead.
The associated summary advisory is retained in diagnostics, not mistaken for
an inner convergence failure.

BLAS limits reuse `path.py:r_thread_limits` from the pinned-BLAS harness.
BLAS/OpenMP are sequential inside each fold; explicit `ncores`/`n_jobs` own
fold concurrency. `RAYON_NUM_THREADS` also bounds Python's full-data fits,
which otherwise default to all cores. Worker environments and per-repeat
diagnostics are recorded. Every raw time is retained; the summary is the
**median**, not the best run. Timeouts/errors are not successful rows, and
only this runner's uniquely named container is removed on an R timeout.

Ordinary file-backed runs do not prove out-of-core behavior under a memory
cap. A later memory-constrained comparison must also record the cgroup cap,
page-cache eviction and actual reads, following `out_of_core.py`; phase 1
makes no claim about that comparison.

## snpnet paths

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
  uv run --locked --group bench python benchmarks/snpnet.py --out /tmp/snpnet-smoke.jsonl
# Later, only in an approved reportable window; opt-in, never a default solver:
# R_BENCH_IMAGE=pybiglasso-bench:r4.6.1-snpnet \
#   uv run --group bench benchmarks/path.py --family binomial --solver r-snpnet ...
```

The adapter prepares BED/PGEN outside the clock and times the complete snpnet
call. It retains the original branch's unpatched workarounds: raw dosages,
penalty factors equal to their population standard deviations, and a zero
covariate required by the upstream penalty-factor indexing. The resulting
standardized penalty is equivalent **only for lasso**; elastic net is rejected.
Every requested lambda must be returned, with finite coefficients and an exact
round-tripped input grid. Small designs can exhaust all variants while snpnet
is still extending the path; that is an incomplete run, not a fast benchmark.

The 20-lambda correctness smoke completed and passed the independent mean-NLL
KKT ceiling `1e-6` at `eps=1e-14` (the numerical suite's established domain),
including the intercept. This stricter smoke is not a performance comparison
at default tolerances. The pinned package emits two known advisories: its
old `thresh` argument to current glmnet, and construction of a zero-extent
matrix for the dummy covariate. Those exact messages are retained in output;
all other warnings, particularly nonconvergence, are fatal. No upstream source
is patched, and the advisories are not represented as clean stderr.

## Not compared yet

No new large CV, snpnet or memory-capped performance rows have been measured.
snpnet installed and completed the bounded smoke, so it is not excluded for
installation failure. If a later bounded run cannot complete, preserve its
reason here rather than reporting its partial path as a timing result.
