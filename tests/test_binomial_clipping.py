"""Clipping beyond R's threshold, with an analytic binomial optimum."""

import warnings

import numpy as np
from support.certificates import (
    assert_same_optimum,
    standardize,
    standardized_theta,
    with_ones,
)
from support.paths import assert_bitwise

import pybiglasso as bl


def test_default_fits_analytic_log_odds_above_ten():
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
    for threads in (1, parallel_threads):
        # Any non-convergence warning is a failure, rather than a way to
        # obtain a loose distance bound from a truncated fit.
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            fit = bl.biglasso(
                X,
                y,
                family="binomial",
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


def test_default_boundary_is_glmnets_pmin():
    # Separable data: y = 1 exactly where x = 1. The intercept is zero by
    # symmetry, so every row has |eta| = beta and stationarity for beta > 0 is
    # mean(x * (y - p)) = 1 - expit(beta) = lambda: beta = logit(1 - lambda).
    # lambda = 2^-28 puts that optimum, about 19.4, between the eta at which a
    # 1e-8 floor would clip (about 18.4) and glmnet's pmin = 1e-9 boundary
    # (about 20.7), so it is reachable only with the 1e-9 boundary. Nothing
    # here comes from a fit.
    lam = 2.0**-28
    optimum = np.log1p(-lam) - np.log(lam)
    assert np.log(1 / 1e-8 - 1) < optimum < np.log(1 / 1e-9 - 1)
    x = np.repeat([-1.0, 1.0], 32)
    X = np.asfortranarray(x[:, None])
    y = (x > 0).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        fit = bl.biglasso(
            X,
            y,
            family="binomial",
            alpha=1.0,
            lambdas=[lam],
            # The sweep stops once w * shift^2 < eps * null deviance / n, with
            # w = lambda * (1 - lambda) at the optimum: here the last Newton
            # step is below 1e-10, and the error after it is far smaller.
            eps=1e-30,
            max_iter=10000,
            stop_on_saturation=False,
            n_threads=1,
        )
    # rounding need not cancel the symmetric rows exactly; the intercept gets
    # the same allowance as beta
    assert abs(fit.intercept[0]) < 1e-6
    beta = fit.coef.toarray()[0, 0]
    assert abs(beta - optimum) < 1e-6, (beta, optimum)
