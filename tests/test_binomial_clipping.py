"""Clipping beyond R's threshold, with an analytic binomial optimum."""

import warnings

import numpy as np
import pytest
from support.certificates import (
    assert_same_optimum,
    standardize,
    standardized_theta,
    with_ones,
)
from support.paths import assert_bitwise

import pybiglasso as bl


@pytest.mark.parametrize("backend", ["rust", "numba"])
def test_default_fits_analytic_log_odds_above_ten(backend):
    # Two equal groups, one wrong label in each. The intercept is zero by
    # symmetry. With unit-variance X in {-1,1}, stationarity for beta > 0
    # gives expit(-beta) = q + lambda, q = 1 / group_size.
    # All counts, X entries and lambda are dyadic; no fitted value is an oracle.
    group_size = 32768
    x = np.repeat([-1.0, 1.0], group_size)
    X = np.asfortranarray(x[:, None])
    y = (x > 0).astype(float)
    y[0], y[group_size] = 1.0, 0.0
    lam = 2.0**-20
    tail = 1 / group_size + lam
    optimum = np.array([0.0, np.log1p(-tail) - np.log(tail)])
    assert 10 < optimum[1] < np.log(1 / 1e-9 - 1)
    Z, center, scale = standardize(X)
    Z = with_ones(Z)
    paths = []
    parallel_threads = 6
    if backend == "numba":
        # Numba cannot exceed the pool created at import (four on some CI
        # runners), even though Rust may explicitly request more workers.
        parallel_threads = min(6, pytest.importorskip("numba").config.NUMBA_NUM_THREADS)
    for threads in (1, parallel_threads):
        # Any non-convergence warning is a failure, rather than a way to
        # obtain a loose distance bound from a truncated fit.
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            fit = bl.biglasso(
                X,
                y,
                family="binomial",
                backend=backend,
                alpha=1.0,
                lambdas=[lam],
                eps=1e-14,
                max_iter=10000,
                stop_on_saturation=False,
                n_threads=threads,
            )
        theta = standardized_theta(
            fit.intercept[0], fit.coef.toarray()[:, 0], center, scale
        )
        assert_same_optimum(Z, y, theta, optimum, lam, 1.0, 1.0)
        paths.append(fit)
    assert_bitwise(*paths)
