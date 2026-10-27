"""The default elastic net solves its documented problem, at every lambda.

The R fixtures only cover `r_compatible=True`, so these tests check the default
solver against the problem itself, independently of any other implementation:

    minimize  (1/n) * NLL(a, b)
              + lambda * sum_j pf_j * (alpha * |b_j| + (1 - alpha)/2 * b_j^2)

on standardized columns (b_j = coefficient * scale_j). The problem is convex,
so the KKT conditions certify a solution: with g_j the gradient of the mean
log-likelihood along standardized column j,

    b_j != 0:  g_j = lambda * pf_j * (alpha * sign(b_j) + (1 - alpha) * b_j)
    b_j == 0:  |g_j| <= lambda * pf_j * alpha

and the intercept's gradient, mean(y - p), is zero.
"""

import warnings
import zlib

import numpy as np
import pytest
from scipy.special import expit
from support.certificates import KKT_TOL, kkt_residual
from support.certificates import path_objective as objective

import pybiglasso as bl


def labels(X, rng, beta):
    return rng.binomial(1, expit(X @ beta - 0.3)).astype(np.float64)


def wide(rng):
    n, p = 120, 600
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[:8] = rng.normal(scale=1.5, size=8)
    return X, labels(X, rng, beta), None


def correlated(rng):
    n, p, rho = 400, 60, 0.9
    X = np.empty((n, p))
    X[:, 0] = rng.normal(size=n)
    for j in range(1, p):  # AR(1) columns: neighbours correlate at rho
        X[:, j] = rho * X[:, j - 1] + np.sqrt(1 - rho**2) * rng.normal(size=n)
    beta = np.zeros(p)
    beta[[3, 4, 30]] = [1.0, -1.0, 0.8]
    return X, labels(X, rng, beta), None


def unpenalized(rng):
    n, p = 300, 40
    X = rng.normal(size=(n, p))
    X[:, :3] = rng.normal(
        loc=[50, 0.5, -3], scale=[10, 0.2, 1], size=(n, 3)
    )  # covariates
    beta = np.zeros(p)
    beta[[0, 1, 5, 6]] = [0.05, 1.0, 0.7, -0.7]
    pf = np.ones(p)
    pf[:3] = 0
    return X, labels(X - X.mean(axis=0), rng, beta), pf


def genotypes(rng):
    n, p = 500, 200
    X = rng.binomial(2, rng.uniform(0.05, 0.5, size=p), size=(n, p)).astype(np.int8)
    beta = np.zeros(p)
    beta[:10] = rng.normal(scale=0.5, size=10)
    return X, labels(X - X.mean(axis=0), rng, beta), None


def duplicates(rng):
    """Twin columns, among them the strongest: identical gradients, and only the
    L2 term to split their coefficients, so the problem is ill-conditioned along
    their difference (a KKT residual r allows a gap of 2r / (lambda (1-alpha)))."""
    n, p = 300, 40
    X = rng.normal(size=(n, p))
    X[:, 30:34] = X[:, :4]  # exact copies
    X[:, 34] = 3.0 * X[:, 4] + 1.5  # standardizes to column 4
    beta = np.zeros(p)
    beta[:5] = [1.2, -1.0, 0.8, 0.6, -0.6]
    return X, labels(X, rng, beta), None


# coordinate descent closes the twins' gap by about 2 lambda (1-alpha) / v of it
# per sweep, so at eps=1e-14 the duplicates need more than the default budget
MAX_ITER = dict(duplicates=100_000)

DESIGNS = dict(
    wide=wide,
    correlated=correlated,
    unpenalized=unpenalized,
    genotypes=genotypes,
    duplicates=duplicates,
)


@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("design", DESIGNS)
def test_kkt_conditions_hold_along_the_path(design, alpha):
    rng = np.random.default_rng(zlib.crc32(f"{design}-{alpha}".encode()))
    X, y, pf = DESIGNS[design](rng)
    with warnings.catch_warnings():
        # saturation truncating a wide path is fine here; failing to converge
        # within max_iter is not, so it raises
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", message="Model saturated")
        path = bl.biglasso(
            np.asfortranarray(X),
            y,
            alpha=alpha,
            penalty_factor=pf,
            nlambda=30,
            eps=1e-14,
            max_iter=MAX_ITER.get(design, 1000),
            family="binomial",
        )
    assert len(path) >= 10
    assert path.n_nonzero[-1] > 0
    Xf = X.astype(np.float64)
    worst = max(kkt_residual(path, Xf, y, k) for k in range(len(path)))
    assert worst < KKT_TOL, f"worst KKT residual {worst:.2e}"


def test_kkt_check_detects_r_compatible_update():
    """The check has teeth: R's elastic-net update fails it."""
    rng = np.random.default_rng(5)
    X, y, _ = correlated(rng)
    path = bl.biglasso(
        np.asfortranarray(X),
        y,
        alpha=0.5,
        nlambda=20,
        eps=1e-12,
        r_compatible=True,
        family="binomial",
    )
    worst = max(kkt_residual(path, X, y, k) for k in range(len(path)))
    assert worst > 100 * KKT_TOL


@pytest.mark.parametrize("design", ["wide", "correlated", "unpenalized", "duplicates"])
def test_no_nearby_point_does_better(design):
    """The objective is at a minimum: small moves in any coefficient increase it."""
    rng = np.random.default_rng(11)
    X, y, pf = DESIGNS[design](rng)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        path = bl.biglasso(
            np.asfortranarray(X),
            y,
            alpha=0.5,
            penalty_factor=pf,
            nlambda=15,
            eps=1e-12,
            family="binomial",
        )
    k = len(path) - 1
    best = objective(path, X, y, k)
    coef = path.coef[:, [k]].toarray().ravel()
    for j in rng.choice(X.shape[1], 25, replace=False):
        for step in (1e-4, -1e-4):
            moved = coef.copy()
            moved[j] += step / path.scale[j]
            trial = bl.path.BiglassoPath(
                **{
                    **{f: getattr(path, f) for f in path.__dataclass_fields__},
                    "coef": _replace_column(path.coef, k, moved),
                }
            )
            assert objective(trial, X, y, k) >= best - 1e-12


def _replace_column(coef, k, values):
    dense = coef.toarray()
    dense[:, k] = values
    return type(coef)(dense)
