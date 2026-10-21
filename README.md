# pybiglasso

**Lasso and elastic-net logistic regression for data larger than memory.**

pybiglasso fits the whole regularization path of a penalized logistic regression
on a design matrix that lives on disk. Point it at a numpy memmap and it reads
one column at a time, never loading or copying the matrix, and uses strong-rule
screening to skip most columns at most steps. Cross-validation works the same
way: folds are row indices into the one matrix on disk.

It is an unofficial Python port of the R package
[biglasso](https://github.com/pbreheny/biglasso), and reproduces its results to
floating-point precision: coefficients typically agree to about 1e-15, and never
by worse than about 1e-12 (see [Validation](#validation)).

<!-- PERFORMANCE -->

## Install

```
pip install pybiglasso
```

The core depends only on numpy, scipy, numba, and threadpoolctl. Extras add
optional integrations:

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

Every push runs pybiglasso against stored R biglasso 1.4.1 output for 123
configurations, comparing coefficients, deviance, the lambda grid, centering
and scaling, strong-rule rejections, and the iteration count at every lambda,
with `rtol=1e-9`.
The grid covers wide and tall problems, genotype-like dosage data, memmaps,
float32, row subsets, penalty factors (including zeros), user-supplied and
linear lambda grids, `dfmax` truncation, and saturation. A weekly job reruns the
grid live against R biglasso 1.4.1 and current CRAN (1.7.2 at last count); both
agree on all 123.

The lasso also agrees with scikit-learn's `saga` solver to about 1e-7, and the
elastic net to about 1e-6.

```
uv run pytest               # against stored R output; no R needed
make test-r                 # live, in a container (CONTAINER=podman works too)
make test-r R_VERSION=latest
make soak                   # larger memory-mapped problems; for a big machine
```

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
- `deviance` is the deviance, -2 log-likelihood; R's `loss` is half of that.
- `stop_on_saturation=False` (not in R) fits every lambda, even past R's
  saturation cutoff.

## Scope

- Binomial (logistic) family only; R's Gaussian and Cox families are not ported.
- The SSR (sequential strong rule) screening; R's newer Hybrid and Adaptive
  rules are not ported.
- Single-threaded.
- The first call in a session pays a few seconds of numba compilation.

## Citation

> Zeng Y, Breheny P (2021). The biglasso Package: A Memory- and
> Computation-Efficient Solver for Lasso Model Fitting with Big Data in R.
> *The R Journal*, 12(2), 6–19.
> [doi:10.32614/RJ-2021-001](https://doi.org/10.32614/RJ-2021-001)

## License

GPL-3.0-or-later, as R biglasso is GPL-3.
