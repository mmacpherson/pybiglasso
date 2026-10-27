"""Gaussian optima from closed forms, augmented lasso and active-set enumeration."""

from itertools import product

import numpy as np
import pytest
from sklearn.linear_model import Lasso
from support.certificates import (
    gaussian_curvature,
    gaussian_subgradient_residual,
    standardize,
    standardized_theta,
    with_ones,
)
from support.gaussian_oracles import assert_optimal, enumerate_optimum, objective

import pybiglasso as bl


def fit(X, y, lambdas, alpha, pf):
    return bl.biglasso(
        X,
        y,
        family="gaussian",
        backend="rust",
        lambdas=lambdas,
        alpha=alpha,
        penalty_factor=pf,
        eps=1e-22,
        max_iter=100_000,
        n_threads=1,
    )


def theta_at(path, center, scale, k=-1):
    return standardized_theta(
        path.intercept[k], path.coef[:, [k]].toarray().ravel(), center, scale
    )


def roundoff(y):
    # Our designs have <=32 rows and <=5 parameters, of order one. Allow 256
    # rounded operations at response scale for mapping and matrix products.
    return 256 * np.finfo(float).eps * max(1.0, np.linalg.norm(y))


@pytest.mark.parametrize("alpha", [0.25, 0.5, 1.0])
def test_multifeature_orthogonal_path_has_closed_form(alpha):
    Q = np.array(list(product((-1.0, 1.0), repeat=4)))
    center, scale = np.array([2, -4, 8, 1]), np.array([0.5, 2, 4, 1])
    X = np.asfortranarray(center + Q * scale)
    signal = np.array([2.0, -1.0, 0.5, -3.0])
    pf = np.array([0.0, 0.5, 2.0, 1.5])
    noise = 0.25 * Q[:, 0] * Q[:, 1]  # orthogonal to all four slopes and 1
    y = 3 + Q @ signal + noise
    entries = abs(signal[1:]) / (alpha * pf[1:])
    lambdas = np.unique(np.r_[0, entries / 2, entries, entries * 2])[::-1]
    path = fit(X, y, lambdas, alpha, pf)
    Z = with_ones(Q)
    for k, lam in enumerate(lambdas):
        beta = np.sign(signal) * np.maximum(abs(signal) - lam * alpha * pf, 0)
        beta /= 1 + lam * (1 - alpha) * pf
        expected = np.r_[3.0, beta]
        theta = theta_at(path, center, scale, k)
        residual = gaussian_subgradient_residual(Z, y, theta, lam, alpha, pf)
        assert np.max(abs(residual)) <= 1e-6
        # Hessian diag(1, 1+lambda*(1-alpha)*pf) has least curvature 1.
        # Strong monotonicity bounds theta error by ||subgradient||/mu.
        allowance = np.linalg.norm(residual) + roundoff(y)
        assert np.linalg.norm(theta - expected) <= allowance
        np.testing.assert_allclose(
            path.coef[:, [k]].toarray().ravel(),
            beta / scale,
            rtol=0,
            atol=allowance / min(scale),
        )
        expected_intercept = 3 - center @ (beta / scale)
        assert (
            abs(path.intercept[k] - expected_intercept)
            <= (1 + np.linalg.norm(center / scale)) * allowance
        )
        expected_rss = len(y) * (np.sum((signal - beta) ** 2) + 0.25**2)
        # ||Z||=sqrt(n). Translate theta error into the RSS perturbation bound.
        delta = np.sqrt(len(y)) * allowance
        rss_error = 2 * np.sqrt(expected_rss) * delta + delta**2 + roundoff(y)
        assert abs(path.deviance[k] - expected_rss) <= rss_error


def correlated_problem(duplicate=False):
    q = np.array(list(product((-1.0, 1.0), repeat=4)))
    X = np.column_stack([q[:, 0], q[:, 0] + q[:, 1], q[:, 2]])
    if duplicate:
        X = np.column_stack([X, X[:, 0]])
    X = np.asfortranarray(X * np.arange(1, X.shape[1] + 1) + 2)
    y = 1.5 + 2 * q[:, 0] - q[:, 1] - 0.75 * q[:, 2] + 0.25 * q[:, 3]
    Xs, center, scale = standardize(X)
    pf = np.array([1.0, 0.5, 2.0, 1.0])[: X.shape[1]]
    return X, y, with_ones(Xs), center, scale, pf


def augmented_lasso(Z, y, lam, alpha, pf, *, wrong=None):
    n, p = len(y), len(pf)
    # Eliminate the free intercept by centering. With u_j=pf_j*beta_j,
    # append sqrt(n*lambda*(1-alpha)/pf_j) ridge rows. Scaling BOTH arrays
    # by sqrt((n+p)/n) cancels sklearn's changed sample normalization.
    # fit_intercept=False prevents recentering the pseudo-observations;
    # sklearn Lasso does not standardize the augmented design.
    ridge = np.sqrt(n * lam * (1 - alpha) / pf)
    if wrong == "ridge":
        ridge /= np.sqrt(n)
    A = np.vstack([Z[:, 1:] / pf, np.diag(ridge)])
    response = np.r_[y - y.mean(), np.zeros(p)]
    if wrong != "sample count":
        A *= np.sqrt((n + p) / n)
        response *= np.sqrt((n + p) / n)
    model = Lasso(alpha=lam * alpha, fit_intercept=False, tol=1e-13, max_iter=100_000)
    model.fit(A, response)
    # Also check the transformed problem itself, independent of sklearn's
    # reported stopping status or dual_gap_. It has no fitted intercept.
    u = model.coef_
    gradient = A.T @ (A @ u - response) / (n + p)
    error = gradient + np.where(
        u != 0, lam * alpha * np.sign(u), np.clip(-gradient, -lam * alpha, lam * alpha)
    )
    assert np.max(abs(error)) <= 1e-6
    return np.r_[y.mean(), u / pf]


@pytest.mark.parametrize("alpha", [0.2, 0.5, 0.8, 1.0])
@pytest.mark.parametrize("lam", [0.1, 0.5])
def test_elastic_net_equals_independent_augmented_lasso(alpha, lam):
    X, y, Z, center, scale, pf = correlated_problem()
    oracle = augmented_lasso(Z, y, lam, alpha, pf)
    got = theta_at(fit(X, y, [lam], alpha, pf), center, scale)
    a = assert_optimal(Z, y, got, lam, alpha, pf)
    b = assert_optimal(Z, y, oracle, lam, alpha, pf)
    mu = gaussian_curvature(Z, lam, alpha, pf)
    allowance = np.sqrt(2 * a / mu) + np.sqrt(2 * b / mu) + roundoff(y)
    assert np.linalg.norm(got - oracle) <= allowance


@pytest.mark.parametrize("wrong", ["ridge", "sample count"])
def test_augmented_oracle_rejects_wrong_normalization(wrong):
    _, y, Z, _, _, pf = correlated_problem()
    theta = augmented_lasso(Z, y, 0.5, 0.5, pf, wrong=wrong)
    with pytest.raises(AssertionError):
        assert_optimal(Z, y, theta, 0.5, 0.5, pf)


@pytest.mark.parametrize("duplicate", [False, True])
@pytest.mark.parametrize("alpha", [0.25, 0.5, 1.0])
def test_cold_coarse_and_fine_paths_match_enumerated_optimum(duplicate, alpha):
    X, y, Z, center, scale, pf = correlated_problem(duplicate)
    lam = 0.15
    oracle = enumerate_optimum(Z, y, lam, alpha, pf)
    oracle_gap = assert_optimal(Z, y, oracle, lam, alpha, pf)
    for grid in ([lam], [1.0, lam], [1.0, 0.7, 0.4, 0.25, lam]):
        theta = theta_at(fit(X, y, grid, alpha, pf), center, scale)
        gap = assert_optimal(Z, y, theta, lam, alpha, pf)
        # Even a nonunique lasso has a unique fitted response. Quadratic loss
        # bounds its distance by sqrt(2*n*gap), without coefficient uniqueness.
        prediction_bound = np.sqrt(2 * len(y) * gap) + np.sqrt(2 * len(y) * oracle_gap)
        assert np.linalg.norm(Z @ (theta - oracle)) <= prediction_bound + roundoff(y)
        assert abs(
            objective(Z, y, theta, lam, alpha, pf)
            - objective(Z, y, oracle, lam, alpha, pf)
        ) <= max(gap, oracle_gap) + roundoff(y)
        if not duplicate or alpha < 1:
            mu = gaussian_curvature(Z, lam, alpha, pf)
            assert np.linalg.norm(theta - oracle) <= (
                np.sqrt(2 * gap / mu) + np.sqrt(2 * oracle_gap / mu) + roundoff(y)
            )


def test_optimum_check_rejects_wrong_ridge_and_intercept():
    X, y, Z, center, scale, pf = correlated_problem()
    theta = enumerate_optimum(Z, y, 0.5, 0.5, pf)
    for perturbation in ([0.5, 0, 0, 0], [0, 0.25, 0, 0]):
        with pytest.raises(AssertionError):
            assert_optimal(Z, y, theta + perturbation, 0.5, 0.5, pf)
    # Orthogonal closed form with an incorrect doubled ridge denominator.
    q = np.array(list(product((-1.0, 1.0), repeat=2)))
    wrong = np.r_[3.0, (2 - 0.25) / (1 + 0.5), 0.0]
    with pytest.raises(AssertionError):
        assert_optimal(with_ones(q), 3 + 2 * q[:, 0], wrong, 0.5, 0.5, np.ones(2))
