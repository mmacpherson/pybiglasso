# pybiglasso

**Lasso and elastic-net logistic regression for data larger than memory.**

If your penalized regression runs out of memory, or you're renting a huge
instance just to hold the data, pybiglasso fits it where the data already is:
on disk. A 12 GB design matrix fits under a 3 GB memory limit, on an ordinary
desktop, in under two minutes (see [Performance](#performance)).

Save the matrix as a numpy memmap and pybiglasso fits the whole regularization
path, with cross-validation to choose the penalty, without loading it into
memory.

It's also fast when the data does fit: along a full 100-step path, 2.3–5.5x
faster than R biglasso and 4–6x faster than skglm.

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

# -- a column-major (Fortran-order) matrix on disk, e.g. genotype dosages
X = np.load("dosages.npy", mmap_mode="r")
y = np.load("phenotype.npy")

path = bl.biglasso(X, y, alpha=1.0, nlambda=100)   # alpha=1: lasso
path
# BiglassoPath(500000 features, 100 lambdas from 0.112 to 0.000112, alpha=1.0)

path.coef                  # scipy.sparse (features x lambdas)
path.n_nonzero             # model size along the path
path.predict_proba(X_new)  # (rows x lambdas); reads only the columns in use
path.score(X_val, y_val, metric="auc")

path.to_polars()           # lambda_, feature, coefficient (nonzero only)
```

Unpenalized covariates (say age, sex, and principal components) take a penalty
factor of 0:

```python
pf = np.ones(X.shape[1])
pf[:12] = 0
path = bl.biglasso(X, y, penalty_factor=pf)
```

### scikit-learn

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

At 23andMe I used R's biglasso to fit penalized models on genotype matrices too
big for memory. Over the 2020 holidays I ported it to Python, partly to
understand how it works under the hood, and partly because a Python version
fits into a Python stack without a detour through R.

This is an independent port, not affiliated with or endorsed by the biglasso
authors: Yaohui Zeng, Chuyi Wang, Tabitha Peter, and Patrick Breheny. The solver
follows their C++ implementation closely, which is why pybiglasso shares its
GPL-3 license. If you use it, please cite their paper (below).

## Validation

Every push runs pybiglasso against stored R biglasso 1.4.1 output for 126
configurations, comparing coefficients, deviance, the lambda grid, centering
and scaling, strong-rule rejections, and the iteration count at every lambda,
with `rtol=1e-9`.
The grid covers wide and tall problems, genotype-like dosage data (including int8 and uint8 storage), memmaps,
float32, row subsets, penalty factors (including zeros), user-supplied and
linear lambda grids, `dfmax` truncation, and saturation. A weekly job reruns the
grid live against R biglasso 1.4.1 and current CRAN (1.7.2 at last count); both
agree on all 126.

The lasso also agrees with scikit-learn's `saga` solver to about 1e-7, and the
elastic net to about 1e-6.

```
uv run pytest               # against stored R output; no R needed
make test-r                 # live, in a container (CONTAINER=podman works too)
make test-r R_VERSION=latest
make soak                   # larger memory-mapped problems; for a big machine
```

## Performance

These benchmarks measure speed only. Correctness is established separately,
against R biglasso itself (see [Validation](#validation)). Each solver fits the
same problem along the same 100-value lambda path, at its own default
tolerance, and its accuracy is reported alongside its time. Machine: AMD Ryzen 5
5600X (6 cores, 12 threads) and 64 GB of RAM, with Python 3.14 and R 4.6.1. Each
time is the best of 3 runs. Every multithreaded solver is shown at its fastest
thread count: 1, 6, or 12. The scripts and raw results are in
[`benchmarks/`](benchmarks/).

![Time relative to pybiglasso, by dataset and solver](https://raw.githubusercontent.com/mmacpherson/pybiglasso/v0.1.0/docs/benchmarks.png)

**Other Python solvers.** Each competitor was stopped once it took 10x as long
as pybiglasso, which would otherwise have stretched the run to many hours. Only
skglm (lasso only) finishes within that on every dataset, at 4–6x. To show how
large the gaps get, the smallest problem (5,000 x 1,000) also ran with no cap:

| Solver | Lasso | Elastic net (α=0.5) | Worst relative suboptimality |
|---|---|---|---|
| pybiglasso | 0.71 s | 0.74 s | 1e-6 |
| pybiglasso, `eps=1e-10` | 1.62 s | 1.54 s | 3e-10 |
| skglm | 4.1 s (6x) | — | 2e-4 |
| glum, 6 threads | 9.6 s (14x) | 10.9 s (15x) | 6e-9 |
| sklearn liblinear | 20.5 s (29x) | — | 5e-6 |
| sklearn saga | 90.7 s (128x) | 94.8 s (128x) | 1e-5 |

**R biglasso 1.7.2.** Both packages fit identical paths. Times are in seconds;
"best" is each package at its fastest thread count. For elastic net, R ran with
its own update (see [Differences](#differences-from-r-biglasso)), and
pybiglasso with the corrected one; `r_compatible=True` is within about 20%
of the default's time.

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

**Not compared.** RAPIDS cuML fits elastic-net logistic regression on a GPU,
but only on data that fits in GPU memory, so it isn't an out-of-core
alternative. benchopt's logistic-regression benchmark fits without an intercept
or standardization, which pybiglasso doesn't support yet.

## Differences from R biglasso

pybiglasso matches R biglasso except where R's behavior looks unintended:

- **Elastic net.** For `alpha < 1`, R's logistic coordinate update divides by
  `v * (1 + l2)` rather than `v + l2`, so it does not minimize the elastic-net
  objective it documents. pybiglasso minimizes the documented objective (agreeing
  with glmnet and scikit-learn); `r_compatible=True` reproduces R exactly. The
  lasso is unaffected.
- **Zero penalty factors.** With any `penalty.factor` of 0 and the default
  lambda grid, R fits `lambda = 0` at every step. pybiglasso fits the
  unpenalized columns first and starts the path at the smallest lambda that
  zeroes every penalized column.
- **Non-unit penalty factors.** pybiglasso's `lambda_max` accounts for penalty
  factors (`max |z_j| / (alpha * pf_j)`); R's ignores them.
- **Default `lambda_min`.** R chooses 0.001 or 0.05 by comparing `nrow(X)` with
  `ncol(X)`; pybiglasso compares the number of rows actually fit, which differs
  when `row_indices` is given.
- **Data far from zero.** On columns whose mean is large relative to their
  spread (say, values near 1e8 that vary by 1), R biglasso's standardization
  loses precision (its scales come out wrong, even NaN), and it silently
  returns a wrong fit. pybiglasso detects the cancellation and recomputes those columns'
  moments centered, so a constant shift of X doesn't change the solution.
- **Invalid input is an error.** NaN or infinite values in the rows being fit,
  and non-finite penalty factors or lambdas, raise `ValueError` instead of
  quietly dropping a column or returning the null model.
- `deviance` is the deviance, -2 log-likelihood, of the returned model; R's
  `loss` is half of that, and is computed before the final coordinate sweep,
  with probabilities clipped at 0 and 1 (so it can be infinite).
  `r_compatible=True` keeps R's version.
- `stop_on_saturation=False` (not in R) fits every lambda, even past R's
  saturation cutoff.

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
  cache); results don't depend on `n_jobs`.
- **Store genotypes compactly.** Since the work is mostly moving bytes, a
  narrower dtype is faster: hard-called genotypes as `int8` or `uint8` take 8x
  less memory and disk than float64, and fit about 2.5x faster, with the same
  results to about 1e-13. `float32` dosages halve both.

## How the code is organized

R biglasso implements each family and screening rule as a single C++ function
that runs the whole algorithm top to bottom: standardize, set up the lambda
grid, then the coordinate-descent loop and its KKT checks at each lambda. That
keeps each entry point self-contained and close to the algorithm as written in
the papers.

pybiglasso splits the same algorithm into one small numba kernel per step, each
documenting the state it changes, plus a short driver, `_fit_lambda`, that reads
like the algorithm's outline. To my eye that's easier to read, test, and modify.
It's a matter of taste, though; some excellent programmers prefer the long
inlined function you can read straight through (John Carmack has
[a good essay](http://number-none.com/blow/john_carmack_on_inlined_code.html) on
why). The split costs nothing at runtime: the kernels compile to native code,
and the solver takes exactly the same iterations as R's.

The kernels also lean on numpy expressions, which numba compiles, where the
C++ writes out loops: a column's score is `X[:, j] @ s` rather than a loop
over rows, and the strong rule is one mask over all columns. Explicit loops
remain where they pay their way: where the arithmetic branches, where one pass
can compute several sums at once, and where a fixed summation order keeps
results identical at any thread count (hence the small `dot` and `seqsum`
kernels in place of `@` and `.sum()`).

| pybiglasso (`src/pybiglasso/binomial.py`) | R biglasso 1.4.1 (`src/`) |
|---|---|
| `center_scale`, `compute_z` | `standardize_and_get_residual` (utilities.cpp) |
| `update_probs` | probability and weight update in `cdfit_binomial_ssr` (binomial.cpp) |
| `update_intercept` | intercept update in `cdfit_binomial_ssr` |
| `update_weights` (`column_moments`, `apply_shift`) | coordinate loop in `cdfit_binomial_ssr`; `wcrossprod_resid`, `wsqsum_bm`, `lasso` (utilities.cpp) |
| `check_strong_set`, `check_rest` | `check_strong_set`, `check_rest_set` (utilities.cpp) |
| `binomial_hsr`, `_fit_lambda` | the lambda loop of `cdfit_binomial_ssr` |

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
