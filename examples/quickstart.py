"""A short tour of pybiglasso on a genotype-like matrix stored on disk.

    uv run --all-groups python examples/quickstart.py

Uses the optional extras: polars, scikit-learn, and seaborn.
"""

import pathlib
import tempfile

import numpy as np

import pybiglasso as bl

workdir = pathlib.Path(tempfile.mkdtemp())

# 1. a design matrix on disk: 2,000 samples x 5,000 hard-called genotypes,
#    stored as int8 in column-major (Fortran) order, as pybiglasso reads it
data = bl.common.make_biglasso_dataset(nrow=2_000, ncol=5_000, seed=1)
X = np.lib.format.open_memmap(
    workdir / "genotypes.npy",
    mode="w+",
    dtype=np.int8,
    shape=data["X"].shape,
    fortran_order=True,
)
X[:] = data["X"]
X.flush()
y = data["y"]

X = np.load(workdir / "genotypes.npy", mmap_mode="r")  # never loaded into memory
print(f"X: {X.shape}, {X.dtype}, {X.nbytes / 1e6:.0f} MB on disk")

# 2. an elastic-net path, leaving the first 3 columns (think: covariates)
#    unpenalized
penalty_factor = np.ones(X.shape[1])
penalty_factor[:3] = 0
path = bl.biglasso(
    X, y, alpha=0.5, penalty_factor=penalty_factor, nlambda=50, family="binomial"
)
print(path)
print("model size along the path:", path.n_nonzero[::10])

# 3. the coefficients as a polars frame: one row per nonzero coefficient
coef = path.to_polars()
print(coef.filter(coef["lambda_"] == path.lambdas[-1]).sort("coefficient").tail(5))

# 4. choose lambda by 5-fold cross-validation; folds are row indices into
#    the same memmap, fit in parallel threads
from pybiglasso.sklearn import BiglassoClassifierCV  # noqa: E402

cv = BiglassoClassifierCV(
    l1_ratio=0.5, penalty_factor=penalty_factor, n_alphas=50, cv=5, rule="1se"
).fit(X, y)
print(f"chosen alpha (R's lambda): {cv.alpha_:.4g}, {(cv.coef_ != 0).sum()} features")

# 5. plot the path (seaborn.objects)
plot = bl.plot_coefficients(path, max_features=12)
plot.save(workdir / "path.png", bbox_inches="tight")  # legend sits outside the axes
print(f"wrote {workdir / 'path.png'}")
