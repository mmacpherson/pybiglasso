# pybiglasso

[![CI](https://github.com/mmacpherson/pybiglasso/actions/workflows/ci.yml/badge.svg)](https://github.com/mmacpherson/pybiglasso/actions/workflows/ci.yml)
[![Live R comparison](https://github.com/mmacpherson/pybiglasso/actions/workflows/r-live.yml/badge.svg)](https://github.com/mmacpherson/pybiglasso/actions/workflows/r-live.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/mmacpherson/pybiglasso/main.svg)](https://results.pre-commit.ci/latest/github/mmacpherson/pybiglasso/main)

[![PyPI](https://img.shields.io/pypi/v/pybiglasso.svg)](https://pypi.org/project/pybiglasso/)
[![Python versions](https://img.shields.io/pypi/pyversions/pybiglasso.svg)](https://pypi.org/project/pybiglasso/)
[![Downloads](https://img.shields.io/pypi/dm/pybiglasso.svg)](https://pypi.org/project/pybiglasso/)

[![License](https://img.shields.io/github/license/mmacpherson/pybiglasso.svg)](https://github.com/mmacpherson/pybiglasso/blob/main/LICENSE)
[![Operating System](https://img.shields.io/badge/OS-Linux_%7C_macOS_%7C_Windows-blue)](https://pypi.org/project/pybiglasso/)
[![Built with uv](https://img.shields.io/badge/Built%20with-uv-purple.svg)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**Lasso and elastic-net logistic regression for data larger than memory.**

If your penalized regression won't fit in memory, pybiglasso fits it where the
data already is: on disk. A 12 GB design matrix fits under a 3 GB memory limit,
on an ordinary desktop, in under two minutes (see [Performance](#performance)).

Save the matrix as a numpy memmap and pybiglasso fits the whole regularization
path, with cross-validation to choose the penalty, without loading it into
memory.

It's also fast when the data does fit in memory. On a full 100-step lasso path
over 5,000 samples by 50,000 features, it is 2.3x faster than R biglasso and
4.7x faster than skglm, the fastest of the other Python solvers we tested.
Across the four benchmark problems, with 1,000 to 50,000 features, it runs 2–5x
faster than R biglasso and 4–7x faster than skglm.

It is an unofficial Python port of the R package
[biglasso](https://github.com/pbreheny/biglasso), and reproduces its results to
floating-point precision (see [Validation](#validation)).

## Install

```
pip install pybiglasso
```

The core depends only on numba, numpy, and scipy. Extras add optional
integrations:

| extra | adds |
|---|---|
| `pybiglasso[polars]` | `BiglassoPath.to_polars()` |
| `pybiglasso[pandas]` | `BiglassoPath.to_pandas()` |
| `pybiglasso[sklearn]` | `BiglassoClassifier`, `BiglassoClassifierCV` |
| `pybiglasso[viz]` | `plot_coefficients()`, `plot_deviance()` (seaborn.objects) |
| `pybiglasso[all]` | all of the above |

## Quick start

```python
import numpy as np
import pybiglasso as bl

# open the matrix on disk without loading it (a read-only memmap); store it
# column-major (Fortran order)
X = np.load("X.npy", mmap_mode="r")
y = np.load("y.npy")

path = bl.biglasso(X, y, alpha=1.0, nlambda=100)   # alpha=1: lasso
path
# BiglassoPath(500000 features, 100 lambdas from 0.112 to 0.000112, alpha=1.0)

path.coef                  # scipy.sparse (features x lambdas)
path.n_nonzero             # model size along the path
path.predict_proba(X_new)  # (rows x lambdas); reads only the columns in use
path.score(X_val, y_val, metric="auc")

path.to_polars()           # lambda_, feature, coefficient (nonzero only)
```

To write such a file, open it as a column-major memmap and fill it a block of
columns at a time, so the whole matrix never has to be in memory:

```python
X = np.lib.format.open_memmap(
    "X.npy", mode="w+", dtype=np.float32,
    shape=(n_samples, n_features), fortran_order=True,
)
for start, block in read_blocks():
    X[:, start : start + block.shape[1]] = block
X.flush()
```

X can be float64, float32, any integer type, or bool. Use the narrowest type
that holds your values exactly: the fit is mostly reading X, so a narrower type
is faster as well as smaller (see
[storage](#threads-cross-validation-and-storage)).

Unpenalized covariates (say age, sex, and principal components) take a penalty
factor of 0:

```python
pf = np.ones(X.shape[1])
pf[:12] = 0
path = bl.biglasso(X, y, penalty_factor=pf)
```

### scikit-learn

pybiglasso also offers a scikit-learn interface: `BiglassoClassifier` fits at
one penalty strength, and `BiglassoClassifierCV` chooses it by cross-validation.
Both pass scikit-learn's estimator checks, so they work in pipelines, grid
searches, and anywhere else a scikit-learn classifier does (install
`pybiglasso[sklearn]`).

```python
from pybiglasso import BiglassoClassifierCV

clf = BiglassoClassifierCV(l1_ratio=1.0, cv=5, rule="1se").fit(X, y)
clf.alpha_, clf.coef_
```

The estimators use scikit-learn's parameter names, which differ from R's:

| scikit-learn (`BiglassoClassifier`) | R / `pybiglasso.biglasso` | meaning |
|---|---|---|
| `alpha` | `lambda` | penalty strength |
| `l1_ratio` | `alpha` | L1/L2 mix; 1 is the lasso |

## Why this exists

At 23andMe, I used R's biglasso package to fit penalized models on large
genotype/phenotype datasets without loading them into memory. Over a holiday
break, I ported it to Python, partly to gain a better understanding of how it
worked and partly because, assuming I could get it working, it would be
convenient to stay within the Python ecosystem. I validated it against R
biglasso, matching its coefficients, lambda grid, and iteration counts to
floating-point precision, and benchmarked it on data both smaller and larger
than memory.

In 2026 I revived the port for release, working with coding agents (Claude Code
and Codex). That work added the scikit-learn estimators and cross-validation,
extended the validation (against R across many more configurations, and with
checks independent of R, which turned up the differences listed below), and
carried the performance work and benchmarks further.

This is an independent port, not affiliated with or endorsed by the biglasso
authors: Yaohui Zeng, Chuyi Wang, Tabitha Peter, and Patrick Breheny. If you use
it, please cite their paper (below).

## Validation

pybiglasso is checked two ways: against R biglasso, which it ports, and
against checks that don't depend on R, which also cover the few places where
the two differ (see [Differences](#differences-from-r-biglasso)).

**Against R.** Every push runs pybiglasso against stored R biglasso 1.7.2
output for 128 configurations, comparing coefficients, deviance, the lambda
grid, centering and scaling, strong-rule rejections, and the iteration count at
every lambda, with `rtol=1e-9`. The grid covers wide and tall problems,
genotype-like dosage data (including int8 and uint8 storage), memmaps, float32,
row subsets, penalty factors (including zeros), user-supplied and linear lambda
grids, `dfmax` truncation, and saturation. The comparison runs pybiglasso with
`r_compatible=True`; for cases with penalty factors it passes R pybiglasso's
lambda grid, which sets aside the zero-penalty-factor case under Differences.
A weekly job reruns the grid live against the current CRAN release, to catch
upstream changes.

**Independent of R.**

- **Optimality.** At every lambda along the path, the fit satisfies the
  elastic-net optimality (KKT) conditions to 1e-6, on wide, correlated,
  partly unpenalized, and genotype-like designs.
- **Other solvers.** scikit-learn's saga solver, tightly converged, finds the
  same coefficients: to about 1e-7 for the lasso and the elastic net alike. In
  the benchmarks, glum reaches the same elastic-net objective as well.
- **Where R differs.** R's elastic-net update (reproduced by
  `r_compatible=True`) doesn't satisfy the optimality conditions, and its
  coefficients differ from saga's by 2–50% of their size on the test problem.
  This is the basis for the elastic-net difference under Differences.
- **Shifted data.** Adding 1e8 to every column of X leaves the solution
  unchanged and still optimal; this is the case described under Differences
  as data far from zero.

```
uv run pytest               # against stored R output; no R needed
make test-r                 # live, in a container (CONTAINER=podman works too)
make test-r R_VERSION=latest
make soak                   # larger memory-mapped problems; for a big machine
```

## Performance

These benchmarks measure speed only. Correctness is established separately,
against R biglasso and by checks independent of it (see
[Validation](#validation)). Each solver fits the
same problem along the same 100-value lambda path, at its own default
tolerance, and its accuracy is reported alongside its time. Machine: AMD Ryzen 5
5600X (6 cores, 12 threads) and 64 GB of RAM, with Python 3.14 and R 4.6.1. Each
time is the best of 3 runs. Every multithreaded solver is shown at its fastest
thread count: 1, 6, or 12. The scripts and raw results are in
[`benchmarks/`](benchmarks/).

![Time relative to pybiglasso, by dataset and solver](https://raw.githubusercontent.com/mmacpherson/pybiglasso/v0.1.0/docs/benchmarks.png)

**Which solvers.** R biglasso, the original, and every maintained Python
package, installable with pip, that fits L1 or elastic-net logistic regression
on a CPU: scikit-learn (its liblinear and saga solvers), glum, skglm, and
statsmodels.

**Other Python solvers.** Each competitor was stopped once it took 10x as long
as pybiglasso, which would otherwise have stretched the run to many hours. Only
skglm finishes within that on every dataset, at 4–6x for the lasso and 5–7x for
the elastic net. statsmodels
(lasso only) ran only on the tall datasets: its L1 fit uses a general-purpose
optimizer with dense matrices over twice the columns, which would need about
12 GB and 76 GB on the wide ones. To show how
large the gaps get, the smallest problem (5,000 x 1,000) also ran with no cap:

| Solver | Lasso | Elastic net (α=0.5) | Worst relative suboptimality |
|---|---|---|---|
| pybiglasso | 0.71 s | 0.74 s | 1e-6 |
| pybiglasso, `eps=1e-10` | 1.62 s | 1.54 s | 3e-10 |
| skglm | 4.1 s (6x) | 4.4 s (6x) | 2e-4 |
| glum, 6 threads | 9.6 s (14x) | 10.9 s (15x) | 6e-9 |
| sklearn liblinear | 20.5 s (29x) | — | 5e-6 |
| sklearn saga | 90.7 s (128x) | 94.8 s (128x) | 1e-5 |
| statsmodels | over 900 s (over 1,000x) | — | — |

**R biglasso 1.7.2.** Times are in seconds; "best" is each package at its
fastest thread count. For the lasso, both packages fit identical paths. For the
elastic net, each ran its own update (see
[Differences](#differences-from-r-biglasso)); pybiglasso with
`r_compatible=True`, which matches R, is within about 20% of its default's
time.

| Dataset | Penalty | pybiglasso, 1 thread | R, 1 thread | pybiglasso, best | R, best | Speedup (best) |
|---|---|---|---|---|---|---|
| 5,000 x 1,000 | lasso | 0.77 | 4.06 | 0.71 | 3.92 | 5.5x |
| 5,000 x 1,000 | elastic net | 0.74 | 4.17 | 0.74 | 4.04 | 5.5x |
| 50,000 x 2,000 | lasso | 11.5 | 55.7 | 10.6 | 54.1 | 5.1x |
| 50,000 x 2,000 | elastic net | 11.7 | 56.1 | 10.6 | 52.9 | 5.0x |
| 2,000 x 20,000 | lasso | 1.33 | 4.71 | 1.13 | 2.90 | 2.6x |
| 2,000 x 20,000 | elastic net | 1.35 | 4.80 | 1.15 | 2.99 | 2.6x |
| 5,000 x 50,000 | lasso | 8.61 | 27.2 | 6.86 | 15.5 | 2.3x |
| 5,000 x 50,000 | elastic net | 8.05 | 27.5 | 6.86 | 15.9 | 2.3x |

R's adaptive screening rule is no faster than SSR on these problems.

**Out of core.** This test fits a 12 GB float64 matrix (20,000 x 75,000) with
memory capped at 3 GB. pybiglasso read a numpy memmap under a cgroup memory
limit; R read a file-backed `big.matrix` in a container with the same limit,
and both were measured from their cgroup's accounting. Before each run, the
matrix was evicted from the page cache. Both fit the same 20-lambda lasso path
and agree to 1e-9, with identical iteration counts.

| | Fit time | Peak memory | Re-read from disk during the fit |
|---|---|---|---|
| pybiglasso, 1 thread | 108 s | 3.2 GB (at the cap) | 8.6 GB |
| pybiglasso, 6 threads | 137 s | 3.2 GB (at the cap) | 16.0 GB |
| R biglasso 1.7.2 | 182 s, plus 43 s to build the `big.matrix` | 3.2 GB (at the cap) | 16.5 GB |

Under the same limit, R re-reads about twice as much of the matrix from disk.
Extra threads hurt here: they evict pages each other still need.

**Not compared**, by the rule above:

- Other languages: R's glmnet, in C++ since its 4.1 releases; Julia's
  GLMNet.jl, a wrapper of glmnet's earlier Fortran solver; and Julia's
  Lasso.jl, written in Julia.
- Other hardware: RAPIDS cuML fits on a GPU, and only data that fits in GPU
  memory.
- Unmaintained: python-glmnet, archived, with its last release in 2020.
- A different problem: celer's logistic regression fits the lasso only, and no
  intercept. skglm, from the same authors, fits an intercept and the elastic
  net, and is compared above.
- Not a solver: benchopt is a benchmarking framework; its logistic-regression
  benchmark fits without an intercept or standardization, which pybiglasso
  doesn't support yet.

## Differences from R biglasso

pybiglasso follows R biglasso 1.7.2 closely, matching it coefficient for
coefficient and down to its iteration and screening counts. It departs in a few
places, mostly where we think R's behavior may not be what was intended; if
we've erred we'd be glad to be corrected. Each departure is marked in the code
with a comment beginning "Differs from R biglasso".

- **Elastic net.** For `alpha < 1`, R's logistic coordinate update divides by
  `v * (1 + l2)`, where the elastic-net objective in its documentation calls
  for `v + l2`, so the two solve somewhat different problems. pybiglasso uses
  `v + l2`, as glmnet and scikit-learn do; `r_compatible=True` reproduces R
  exactly. The lasso is the same either way.
- **Zero penalty factors.** With a `penalty.factor` of 0 and the default
  lambda grid, R's grid comes out as `lambda = 0` at every step. pybiglasso
  instead fits the unpenalized columns first, and starts the path at the
  smallest lambda that keeps every penalized column at zero.
- **Default `lambda_min`.** R chooses 0.001 or 0.05 by comparing `nrow(X)` with
  `ncol(X)`; pybiglasso compares the number of rows actually fit, which differs
  when `row_indices` is given.
- **Data far from zero.** For columns whose mean is large next to their spread
  (say, values near 1e8 that vary by 1), standardizing from raw sums loses
  precision in floating point. R's scales, and so its fit, can be affected
  without a warning; in our tests some scales came out as NaN. pybiglasso
  computes these columns' moments from centered values, so a constant shift of
  X leaves the solution unchanged.
- **Invalid input.** NaN or infinite values in the rows being fit, and
  non-finite penalty factors or lambdas, raise `ValueError`; R instead drops
  the affected column or returns the null model.
- **Deviance.** `deviance` is the deviance, -2 log-likelihood, of the returned
  model. R reports `loss`, half the deviance, computed before the final
  coordinate sweep and with probabilities clipped at 0 and 1, so it can
  occasionally be infinite. `r_compatible=True` keeps R's version.
- **Saturation.** `stop_on_saturation=False`, an option R doesn't have, fits
  every lambda, even past R's saturation cutoff.

## Threads, cross-validation, and storage

pybiglasso reads the matrix one column at a time, through the memmap, and
never loads or copies it; strong-rule screening skips most columns at most
steps. Cross-validation folds are row indices into the same matrix on disk.

- **A single fit is effectively single-threaded**, as is R biglasso's.
  Coordinate descent updates one coefficient at a time, and that sequential
  sweep is most of the work: profiling an elastic-net path on 5,000 x 1,000
  puts over 80% of the time in it, against 1-2% in the strong-rule and KKT
  scans that run in parallel (`n_threads=`). `make profile` reproduces this
  with Python 3.15's sampling profiler. Out of core, threads can help when memory to spare
  holds the page cache, by keeping more disk reads in flight. Under a tight
  memory limit they hurt (see [Performance](#performance)), so use
  `n_threads=1` there. Results are bit-identical at any thread count.
- **Cross-validation scales with cores.** `BiglassoClassifierCV(n_jobs=...)`
  fits folds in parallel threads that share one X (and, for a memmap, one page
  cache); results don't depend on `n_jobs`. This needs numba's tbb or omp
  threading layer; where numba falls back to workqueue, as on macOS by default,
  folds run one at a time.
- **Store X compactly.** Since the work is mostly moving bytes, a narrower
  dtype is faster: small integers (counts, categories, 0/1 indicators) as
  `int8` or `uint8` take 8x less memory and disk than float64, and fit about
  2.5x faster, with the same results to about 1e-13. `float32` halves both.

## How the code is organized

R biglasso implements the logistic solver in two long C++ functions:
`cdfit_binomial_ssr` standardizes, sets up the lambda grid, and runs the lambda
loop with its strong rule and KKT checks, and `cdfit_binomial_lambda_core` runs
the coordinate descent at each lambda. That keeps the algorithm in a few places,
close to how it's written in the papers.

pybiglasso splits the same algorithm into one small numba kernel per step, each
documenting the state it changes, plus a short driver, `_fit_lambda`, that reads
like the algorithm's outline. To my eye that's easier to read, test, and modify.
It's a matter of taste, though; some excellent programmers prefer the long
inlined function you can read straight through (John Carmack has
[a good essay](http://number-none.com/blow/john_carmack_on_inlined_code.html) on
why). The kernels compile to native code, so Python only sequences them, and the
solver takes exactly the same iterations as R's at every lambda; the R
comparison checks this in every case.

The driver, in plain numpy, works on whole arrays where the C++ writes out
loops: the strong rule is one mask over all columns, and the lambda grid one
`logspace`. Inside the kernels, loops are explicit, which lets one pass compute
several sums at once. Explicit loops also fix the order of summation, so results
are bit-identical at any thread count; that's why the kernels use small `dot`
and `seqsum` helpers rather than `@` and `.sum()`.

| pybiglasso (`src/pybiglasso/binomial.py`) | R biglasso (`src/`) |
|---|---|
| `center_scale`, `compute_z` | `standardize_and_get_residual` (utilities.cpp) |
| `update_probs` | probability and weight update in `cdfit_binomial_lambda_core` (binomial.cpp) |
| `update_intercept` | intercept update in `cdfit_binomial_lambda_core` |
| `update_weights` (`column_moments`, `apply_shift`) | coordinate loop in `cdfit_binomial_lambda_core`; `wcrossprod_resid`, `wsqsum_bm`, `lasso` (utilities.cpp) |
| `check_strong_set`, `check_rest` | `check_strong_set`, `check_rest_set` (utilities.cpp) |
| `_fit_lambda` | `cdfit_binomial_lambda_core` |
| `binomial_hsr` | `cdfit_binomial_ssr` |

## Scope

- Binomial (logistic) family only; R's Gaussian and Cox families are not ported.
- SSR (sequential strong rule) screening only. R uses its newer Hybrid and
  Adaptive rules only for the plain lasso without penalty factors, and they
  were no faster than SSR on our benchmarks.
- The first call in a session pays a few seconds of numba compilation; after
  that, compiled kernels are cached on disk.

## Citation

> Zeng Y, Breheny P (2021). The biglasso Package: A Memory- and
> Computation-Efficient Solver for Lasso Model Fitting with Big Data in R.
> *The R Journal*, 12(2), 6–19.
> [doi:10.32614/RJ-2021-001](https://doi.org/10.32614/RJ-2021-001)

## License

GPL-3.0-or-later, as R biglasso is GPL-3.
