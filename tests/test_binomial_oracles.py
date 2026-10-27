"""Binomial optima without screening, solver trajectories or fitted expectations.

The tiny oracle enumerates signed active sets and solves their full smooth
stationarity systems with scipy. It has no IRLS, coordinate descent, screening
or warm start. Analytic examples also cover grouping and nonunique lasso fits.
"""

import warnings
from itertools import product

import numpy as np
import pytest
from scipy.optimize import brentq, root
from scipy.special import expit, logit
from support.certificates import (
    residual_rounding,
    standardize,
    standardized_theta,
    subgradient_residual,
    with_ones,
)

import pybiglasso as bl

# An independently specified accuracy target, not a claim that the stopping
# parameter itself guarantees coefficient accuracy. Residual acceptance below
# is chosen to imply this target through strong convexity. Fits must pass it.
COEFFICIENT_ACCURACY = 1e-8
FIT_EPS = 1e-24


def curvature(Z, lam, alpha, pf):
    # Every compared point must have |eta|<=4, hence so does their segment.
    # exp(4)<60, and 60/61^2>1/64: p(1-p)>=1/64 on that domain.
    hessian = Z.T @ Z / (64 * len(Z)) + np.diag(np.r_[0, lam * (1 - alpha) * pf])
    eigenvalues = np.linalg.eigvalsh(hessian)
    # Same practical eigensolver margin as the existing certificates; these
    # small well-scaled problems are not interval linear-algebra enclosures.
    lower = eigenvalues[0] - 1e-12 * eigenvalues[-1]
    assert lower > 0
    return lower


def residual_and_radius(Z, y, theta, lam, alpha, pf):
    assert np.max(abs(Z @ theta)) <= 4, "outside the declared curvature domain"
    mu = curvature(Z, lam, alpha, pf)
    residual = subgradient_residual(Z, y, theta, lam, alpha, pf)
    rounding = residual_rounding(Z, y, theta, lam, alpha, pf)
    # ||g||_2 <= sqrt(d)*||g||_inf. Strong monotonicity on the segment
    # gives ||a-b|| <= (||g(a)||+||g(b)||)/mu for ANY two compared points;
    # we do not assume the unknown exact optimum lies in the bounded domain.
    # Reserve a factor four for the other endpoint and rounding. This cap is
    # fixed by the requested accuracy and problem curvature, never fit output.
    cap = mu * COEFFICIENT_ACCURACY / (4 * np.sqrt(Z.shape[1]))
    assert np.max(abs(residual)) <= cap
    radius = (np.linalg.norm(residual) + np.linalg.norm(rounding)) / mu
    assert radius <= COEFFICIENT_ACCURACY / 2
    return residual, radius


def assert_same_optimum(Z, y, actual, expected, lam, alpha, pf):
    radii = [
        residual_and_radius(Z, y, point, lam, alpha, pf)[1]
        for point in (actual, expected)
    ]
    # The two curvature-normalized residual norms bound their distance. Include
    # rounding of the subtraction/norm, gamma_(d+2) times endpoint magnitudes.
    operations = Z.shape[1] + 2
    unit = np.finfo(float).eps
    rounding = (
        operations
        * unit
        / (1 - operations * unit)
        * (np.linalg.norm(actual) + np.linalg.norm(expected))
    )
    allowance = sum(radii) + rounding
    assert np.linalg.norm(actual - expected) <= allowance
    return allowance


def fit(X, y, lambdas, alpha=1.0, pf=None):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        path = bl.biglasso(
            X,
            y,
            family="binomial",
            backend="rust",
            lambdas=lambdas,
            alpha=alpha,
            penalty_factor=pf,
            eps=FIT_EPS,
            max_iter=100_000,
            n_threads=1,
        )
    assert len(path) == len(lambdas)
    np.testing.assert_array_equal(path.lambdas, lambdas)
    return path


def theta_at(path, center, scale, index):
    return standardized_theta(
        path.intercept[index], path.coef[:, [index]].toarray().ravel(), center, scale
    )


def enumerate_optimum(Z, y, lam, alpha, pf):
    """Find a KKT-feasible signed active set, solving every variable jointly."""
    p = Z.shape[1] - 1
    assert 0 < p <= 3
    candidates = []
    for state in product((-1, 0, 1), repeat=p):
        signs = np.array(state)
        active = np.flatnonzero(signs)
        columns = np.r_[0, active + 1]
        design = Z[:, columns]
        ridge = np.r_[0, lam * (1 - alpha) * pf[active]]
        lasso = np.r_[0, lam * alpha * pf[active] * signs[active]]

        def gradient(theta, design=design, ridge=ridge, lasso=lasso):
            return (
                design.T @ (expit(design @ theta) - y) / len(y) + ridge * theta + lasso
            )

        def hessian(theta, design=design, ridge=ridge):
            probability = expit(design @ theta)
            return design.T @ (
                design * (probability * (1 - probability))[:, None]
            ) / len(y) + np.diag(ridge)

        solution = root(gradient, np.zeros(len(columns)), jac=hessian, tol=1e-11).x
        if np.any(solution[1:] * signs[active] <= 0):
            continue
        theta = np.zeros(p + 1)
        theta[columns] = solution
        if np.max(abs(Z @ theta)) > 4:
            continue
        # scipy's success flag is not an accuracy certificate. Independently
        # check every active equation and every inactive inequality instead.
        residual = subgradient_residual(Z, y, theta, lam, alpha, pf)
        cap = curvature(Z, lam, alpha, pf) * COEFFICIENT_ACCURACY / (4 * np.sqrt(p + 1))
        if np.max(abs(residual)) <= cap:
            residual_and_radius(Z, y, theta, lam, alpha, pf)
            candidates.append(theta)
    assert candidates, "no enumerated state passed all independent KKT conditions"
    return candidates[0]


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("penalty", [0.5, 2.0])
def test_one_feature_lasso_path_has_closed_form(direction, penalty):
    # Mixed labels in each equally sized +/-1 group: P(y=1|q)=1/2+q/4.
    # Symmetry fixes the standardized intercept at zero. For positive beta,
    # stationarity is expit(beta)-3/4+lambda*pf=0; at lambda*pf>=1/4 it is zero.
    q = np.repeat([-1.0, 1.0], 8)
    y = np.r_[np.ones(2), np.zeros(6), np.ones(6), np.zeros(2)]
    center, scale = np.array([3.0]), np.array([2.0])
    X = np.asfortranarray(center + scale * direction * q[:, None])
    Z = with_ones(direction * q[:, None])
    lambdas = np.array([0.5, 0.25, 0.125, 0.0625, 0.0]) / penalty
    path = fit(X, y, lambdas, pf=np.array([penalty]))
    for k, lam in enumerate(lambdas):
        probability = 0.75 - min(lam * penalty, 0.25)
        expected = np.array([0.0, direction * logit(probability)])
        actual = theta_at(path, center, scale, k)
        allowance = assert_same_optimum(
            Z, y, actual, expected, lam, 1.0, np.array([penalty])
        )
        assert abs(path.coef[0, k] - expected[1] / scale[0]) <= allowance / scale[0]
        assert (
            abs(path.intercept[k] + center[0] * expected[1] / scale[0])
            <= (1 + center[0] / scale[0]) * allowance
        )
        if lam * penalty >= 0.25:
            assert path.coef[0, k] == 0
        # Both groups have the same signed-label loss. Binary deviance is
        # twice NLL (the saturated Bernoulli log likelihood is zero).
        expected_nll = -len(y) * (
            0.75 * np.log(probability) + 0.25 * np.log1p(-probability)
        )
        # Each softplus is 1-Lipschitz; ||Z||=sqrt(n). A gamma_(4n)
        # allowance covers the n log/add terms and their reduction at eta<=4.
        operations = 4 * len(y)
        unit = np.finfo(float).eps
        rounding = operations * unit / (1 - operations * unit) * len(y) * 5
        assert abs(path.deviance[k] - 2 * expected_nll) <= 2 * (
            len(y) * allowance + rounding
        )


def tiny_problem():
    q = np.repeat(np.array(list(product((-1.0, 1.0), repeat=3))), 4, axis=0)
    counts = [1, 2, 3, 1, 3, 2, 2, 3]
    y = np.concatenate([np.r_[np.ones(count), np.zeros(4 - count)] for count in counts])
    slopes = np.column_stack([q[:, 0], (q[:, 0] + q[:, 1]) / np.sqrt(2), q[:, 2]])
    X = np.asfortranarray(np.array([2, -4, 8]) + slopes * np.array([0.5, 2, 4]))
    standardized, center, scale = standardize(X)
    return X, y, with_ones(standardized), center, scale


@pytest.mark.parametrize("alpha", [0.25, 0.5, 1.0])
def test_screening_free_optimum_is_independent_of_grid_and_warm_start(alpha):
    X, y, Z, center, scale = tiny_problem()
    pf = np.array([0.5, 1.5, 1.0])
    coarse = np.array([0.3, 0.12, 0.04])
    fine = np.array([0.3, 0.2, 0.12, 0.08, 0.04])
    paths = [(coarse, fit(X, y, coarse, alpha, pf)), (fine, fit(X, y, fine, alpha, pf))]
    oracles = {lam: enumerate_optimum(Z, y, lam, alpha, pf) for lam in fine}
    assert any(np.any(theta[1:] != 0) for theta in oracles.values())
    for lambdas, path in paths:
        for k, lam in enumerate(lambdas):
            assert_same_optimum(
                Z, y, theta_at(path, center, scale, k), oracles[lam], lam, alpha, pf
            )
    for lam in coarse:
        cold = fit(X, y, [lam], alpha, pf)
        assert_same_optimum(
            Z, y, theta_at(cold, center, scale, 0), oracles[lam], lam, alpha, pf
        )


@pytest.mark.parametrize("separation", [0.0, 0.0625, 0.25])
@pytest.mark.parametrize("alpha", [0.25, 0.5])
def test_elastic_net_groups_equal_and_near_equal_columns(separation, alpha):
    q = np.repeat(np.array(list(product((-1.0, 1.0), repeat=2))), 8, axis=0)
    y = np.concatenate(
        [
            np.r_[np.ones(int(4 + 2 * a + b)), np.zeros(int(4 - 2 * a - b))]
            for a, b in product((-1, 1), repeat=2)
        ]
    )
    slopes = np.column_stack([q[:, 0], q[:, 0] + separation * q[:, 1]])
    X = np.asfortranarray(np.array([2, -4]) + slopes * np.array([0.5, 2]))
    standardized, center, scale = standardize(X)
    Z = with_ones(standardized)
    lam, pf = 0.04, np.ones(2)
    expected = enumerate_optimum(Z, y, lam, alpha, pf)
    path = fit(X, y, [lam], alpha, pf)
    theta = theta_at(path, center, scale, 0)
    allowance = assert_same_optimum(Z, y, theta, expected, lam, alpha, pf)
    assert min(theta[1:]) > allowance  # common positive sign is certified
    # Subtract the two active KKT equations: the common L1 term cancels.
    # Cauchy-Schwarz gives l2*|b1-b2| <= ||z1-z2||*||y-p||/n.
    # Use the difference norm directly, avoiding cancellation in 2*(1-rho).
    residual, radius = residual_and_radius(Z, y, theta, lam, alpha, pf)
    difference = np.linalg.norm(Z[:, 1] - Z[:, 2])
    score_norm = np.linalg.norm(y - expit(Z @ theta))
    unit = np.finfo(float).eps

    def gamma(operations):
        return operations * unit / (1 - operations * unit)

    # Enclose subtraction/norm rounding and expit's propagated dot-product
    # error (1/4-Lipschitz plus 4u, as in residual_rounding). A norm's squared
    # products, reduction and sqrt use at most n+2 dependent roundings.
    difference_error = gamma(len(y) + 2) * difference + gamma(1) * np.linalg.norm(
        abs(Z[:, 1]) + abs(Z[:, 2])
    )
    score_error = (
        np.linalg.norm(
            gamma(Z.shape[1]) * (abs(Z) @ abs(theta)) / 4
            + 4 * unit
            + unit * abs(y - expit(Z @ theta))
        )
        + gamma(len(y) + 2) * score_norm
    )
    grouping_bound = (difference + difference_error) * (score_norm + score_error)
    grouping_bound /= len(y) * (1 - gamma(4))
    # Approximate stationarity contributes |g1-g2|<=2*cap; evaluating the
    # contrast and penalty contributes their independently computed roundoff.
    rounding = residual_rounding(Z, y, theta, lam, alpha, pf)
    correction = (
        abs(residual[1])
        + abs(residual[2])
        + rounding[1]
        + rounding[2]
        + gamma(4) * lam * (1 - alpha) * (abs(theta[1]) + abs(theta[2]))
    )
    assert lam * (1 - alpha) * abs(theta[1] - theta[2]) <= grouping_bound + correction
    if separation == 0:
        # Ridge makes the equal split UNIQUE, unlike the following lasso case.
        assert abs(theta[1] - theta[2]) <= np.sqrt(2) * radius


@pytest.mark.parametrize("lam", [0.0625, 0.125])
def test_duplicate_lasso_has_distinct_optima_but_unique_predictions(lam):
    q = np.repeat([-1.0, 1.0], 8)
    y = np.r_[np.ones(2), np.zeros(6), np.ones(6), np.zeros(2)]
    center, scale = np.array([3.0, -2.0]), np.array([2.0, 4.0])
    X = np.asfortranarray(center + q[:, None] * scale)
    Z = with_ones(np.column_stack([q, q]))
    total = logit(0.75 - lam)
    alternatives = [
        np.array([0, share * total, (1 - share) * total]) for share in (0.0, 0.25, 1.0)
    ]
    # Exhibit independently valid alternatives; don't infer nonuniqueness from
    # whether coordinate descent happens to choose different representations.
    assert np.linalg.norm(alternatives[0] - alternatives[-1]) > 0.5
    for theta in alternatives:
        residual = subgradient_residual(Z, y, theta, lam, 1.0)
        assert np.max(abs(residual)) <= np.max(residual_rounding(Z, y, theta, lam, 1.0))
        np.testing.assert_allclose(
            Z @ theta, q * total, rtol=0, atol=np.finfo(float).eps * total
        )
        assert abs(theta[1:]).sum() == total
    reduced = with_ones(q[:, None])
    expected = np.array([0.0, total])
    for grid in ([0.5, 0.25, lam], [0.5, 0.25, 0.2, 0.15, lam], [lam]):
        path = fit(X, y, grid)
        theta = theta_at(path, center, scale, len(grid) - 1)
        collapsed = np.array([theta[0], theta[1] + theta[2]])
        allowance = assert_same_optimum(
            reduced, y, collapsed, expected, lam, 1.0, np.ones(1)
        )
        assert (
            np.linalg.norm(path.decision_function(X)[:, -1] - q * total)
            <= np.sqrt(len(y)) * allowance
        )


def test_oracle_checks_reject_missing_penalty_weight_and_wrong_ridge():
    X, y, Z, _, _ = tiny_problem()
    assert X.shape[1] == 3
    lam, alpha, pf = 0.04, 0.5, np.array([0.5, 1.5, 1.0])
    correct = enumerate_optimum(Z, y, lam, alpha, pf)
    residual_and_radius(Z, y, correct, lam, alpha, pf)
    wrong_weights = enumerate_optimum(Z, y, lam, alpha, np.ones(3))
    wrong_ridge = enumerate_optimum(Z, y, lam * alpha, 1.0, pf)
    for wrong in (wrong_weights, wrong_ridge):
        with pytest.raises(AssertionError):
            residual_and_radius(Z, y, wrong, lam, alpha, pf)


def test_closed_form_check_rejects_wrong_lambda_and_unmapped_intercept():
    q = np.repeat([-1.0, 1.0], 8)
    y = np.r_[np.ones(2), np.zeros(6), np.ones(6), np.zeros(2)]
    Z, lam, pf = with_ones(q[:, None]), 0.0625, np.ones(1)
    correct = np.array([0.0, logit(0.75 - lam)])
    residual_and_radius(Z, y, correct, lam, 1.0, pf)
    wrong_lambda = np.array([0.0, logit(0.75 - 0.9 * lam)])
    # X=3+2q needs raw intercept=-3*beta/2. Forgetting that map introduces
    # the following standardized-intercept error, independently of any fit.
    unmapped_intercept = correct + np.array([3 * correct[1] / 2, 0])
    for wrong in (wrong_lambda, unmapped_intercept):
        with pytest.raises(AssertionError):
            residual_and_radius(Z, y, wrong, lam, 1.0, pf)


def test_ridge_grouping_check_rejects_a_lasso_style_unequal_split():
    q = np.repeat([-1.0, 1.0], 8)
    y = np.r_[np.ones(2), np.zeros(6), np.ones(6), np.zeros(2)]
    Z, lam, alpha, pf = with_ones(np.column_stack([q, q])), 0.04, 0.5, np.ones(2)
    # Symmetry forces b1=b2=t/2. The total t solves the following scalar
    # equation, strictly increasing from negative to positive on this bracket.
    total = brentq(
        lambda t: expit(t) - 0.75 + lam * alpha + lam * (1 - alpha) * t / 2,
        0,
        logit(0.75),
        xtol=np.finfo(float).eps,
    )
    correct = np.array([0.0, total / 2, total / 2])
    wrong = np.array([0.0, total, 0.0])
    residual_and_radius(Z, y, correct, lam, alpha, pf)
    np.testing.assert_array_equal(Z @ correct, Z @ wrong)
    with pytest.raises(AssertionError):
        residual_and_radius(Z, y, wrong, lam, alpha, pf)
