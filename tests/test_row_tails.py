"""Fits of selected rows whose count leaves a partial block, checked by KKT.

The kernels handle rows in blocks of LANES (8) within chunks of CHUNK (2048),
and a masked fit handles the rows past the last whole block separately. Here
the fitted rows leave such a tail, in the last chunk and in a later one, and
each fit must satisfy its own optimality conditions, computed in numpy from
the selected rows alone. A residual update that missed those rows would leave
the coefficients optimal for a different residual.
"""

import numpy as np
import pytest
from support.certificates import (
    KKT_TOL,
    gaussian_subgradient_residual,
    standardize,
    standardized_kkt,
    standardized_theta,
    with_ones,
)

import pybiglasso as bl
from pybiglasso import _rust

EPS = 1e-14


def problem(nrow, family, rng):
    X = rng.standard_normal((nrow, 12))
    X[:, 1] += 0.7 * X[:, 0]
    eta = X[:, :3] @ np.array([1.0, -0.8, 0.6])
    if family == "gaussian":
        y = eta + rng.standard_normal(nrow)
    else:
        y = (rng.random(nrow) < 1 / (1 + np.exp(-eta))).astype(float)
    return np.asfortranarray(X), y


def selected_rows(nrow, kind):
    if kind == "mask":
        # most rows, so they are fit through a mask; the first and last rows
        # stay in, so the tail of every chunk holds fitted rows
        keep = np.ones(nrow, dtype=bool)
        keep[np.arange(5, nrow - 1, 9)] = False
    else:
        # a few rows of a tall X are gathered
        keep = np.zeros(nrow, dtype=bool)
        keep[np.arange(0, nrow, 3)] = True
    return np.flatnonzero(keep)


@pytest.mark.parametrize("nrow", [203, 2 * 2048 + 13])
@pytest.mark.parametrize("kind", ["mask", "gather"])
@pytest.mark.parametrize("family", ["gaussian", "binomial"])
@pytest.mark.parametrize("r_compatible", [False, True])
def test_selected_rows_with_a_partial_block_are_optimal(
    monkeypatch, nrow, kind, family, r_compatible
):
    rows = selected_rows(nrow, kind)
    assert len(rows) % 8 != 0 and len(rows) % 2048 != 0
    X, y = problem(nrow, family, np.random.default_rng(nrow))
    # R-compatible binomial elastic net solves another problem (see
    # test_default_enet.py); the lasso is the same problem in both modes
    alpha = 1.0 if r_compatible and family == "binomial" else 0.5

    direct, used = _rust.fit_path, []

    def recording(*args, **kwargs):
        result = direct(*args, **kwargs)
        used.append(result["row_path"])
        return result

    monkeypatch.setattr(_rust, "fit_path", recording)
    fit = bl.biglasso(
        X,
        y,
        family=family,
        alpha=alpha,
        row_indices=rows,
        nlambda=8,
        lambda_min=0.05,
        eps=EPS,
        max_iter=100_000,
        stop_on_saturation=False,
        r_compatible=r_compatible,
        n_threads=1,
    )
    assert used == [kind]

    Z, center, scale = standardize(X[rows])
    Z = with_ones(Z)
    np.testing.assert_allclose(fit.center, center, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(fit.scale, scale, rtol=1e-10)
    coef = fit.coef.toarray()
    for k, lam in enumerate(fit.lambdas):
        theta = standardized_theta(fit.intercept[k], coef[:, k], center, scale)
        if family == "gaussian":
            residual = gaussian_subgradient_residual(Z, y[rows], theta, lam, alpha)
            worst = np.max(np.abs(residual))
        else:
            worst = standardized_kkt(Z, y[rows], theta, lam, alpha)
        assert worst < KKT_TOL, f"lambda {k}: KKT residual {worst:.2e}"
