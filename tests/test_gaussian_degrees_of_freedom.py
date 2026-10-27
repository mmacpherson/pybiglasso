"""Local lasso divergence, not a pointwise statistical degrees-of-freedom claim.

For Gaussian responses, statistical df is the expectation of local divergence.
Away from selection boundaries, the lasso prediction map projects perturbations
onto the active column space. Include the free intercept in that space.
See Tibshirani & Taylor (2012), https://www.stat.cmu.edu/~ryantibs/papers/lassodf-aos.pdf.
"""

from itertools import product

import numpy as np
import pytest
from support.certificates import (
    KKT_TOL,
    gaussian_duality_gap,
    gaussian_subgradient_residual,
    standardized_theta,
    with_ones,
)

import pybiglasso as bl


@pytest.mark.parametrize("duplicate", [False, True])
@pytest.mark.parametrize("lam", [0.25, 0.5])
def test_local_lasso_divergence_is_active_rank_including_intercept(duplicate, lam):
    q = np.array(list(product((-1.0, 1.0), repeat=3)))
    slopes = q[:, [0, 1, 0, 2]] if duplicate else q
    p, n = slopes.shape[1], len(q)
    center = np.arange(p) * 2.0 - 1
    scale = 2.0 ** np.arange(p)
    X = np.asfortranarray(center + slopes * scale)
    y = 3 + 2 * q[:, 0] - q[:, 1] + 0.125 * q[:, 2]
    Z = with_ones(slopes)

    # Soft-thresholding orthogonal scores: the first two directions stay
    # active; the third is strictly inactive. Duplicates share the same total
    # coefficient, so split that total to exhibit a redundant optimal support.
    expected_beta = np.array([2 - lam, -(1 - lam), 0.0])
    if duplicate:
        expected_beta = np.array([(2 - lam) / 2, -(1 - lam), (2 - lam) / 2, 0.0])
    expected_theta = np.r_[3.0, expected_beta]
    active = np.flatnonzero(expected_beta)
    active_design = Z[:, np.r_[0, active + 1]]
    active_rank = np.linalg.matrix_rank(active_design)
    assert active_rank == 3  # intercept plus two independent directions
    if duplicate:
        assert len(active) + 1 == 4 > active_rank
    projection = (
        np.ones((n, n)) + q[:, [0]] @ q[:, [0]].T + q[:, [1]] @ q[:, [1]].T
    ) / n
    np.testing.assert_allclose(
        active_design @ np.linalg.pinv(active_design), projection, atol=1e-14, rtol=0
    )

    def predict_and_radius(response):
        path = bl.biglasso(
            X,
            response,
            family="gaussian",
            backend="rust",
            lambdas=[lam],
            alpha=1.0,
            eps=1e-24,
            max_iter=100_000,
            n_threads=1,
        )
        theta = standardized_theta(
            path.intercept[0], path.coef.toarray()[:, 0], center, scale
        )
        residual = gaussian_subgradient_residual(Z, response, theta, lam, 1.0)
        assert np.max(abs(residual)) <= KKT_TOL
        gap = gaussian_duality_gap(Z, response, theta, lam, 1.0)
        assert gap <= 1e-10
        # Squared loss bounds prediction error by sqrt(2*n*objective gap),
        # including when coefficients are nonunique. The gap uses exact
        # rational dual feasibility and outward-rounded objective algebra.
        radius = np.sqrt(2 * n * gap)
        # The gap's square root can be loose for lasso: dual contraction has
        # first-order rounding error. Collapse identical directions instead.
        # Their penalty is |total| at the optimum, so the reduced problem has
        # the SAME fitted response and Hessian I (including the intercept).
        # Strong monotonicity bounds its coefficient error by the subgradient
        # norm; ||[1,q]||=sqrt(n) converts it to prediction error.
        collapsed = (
            theta[[0, 1, 2, 4]] + np.array([0, theta[3], 0, 0]) if duplicate else theta
        )
        reduced_residual = gaussian_subgradient_residual(
            with_ones(q), response, collapsed, lam, 1.0
        )
        radius = min(radius, np.sqrt(n) * np.linalg.norm(reduced_residual))
        rounding = 256 * np.finfo(float).eps * max(1.0, np.linalg.norm(response))
        return path.predict(X)[:, 0], radius + rounding

    baseline, radius = predict_and_radius(y)
    assert np.linalg.norm(baseline - Z @ expected_theta) <= radius
    assert gaussian_duality_gap(Z, y, expected_theta, lam, 1.0) <= 1e-12

    for h in (0.01, 0.005):
        # A coordinate perturbation changes each centered orthogonal score
        # by at most h/n. These analytic margins justify the SAME active
        # subspace/signs throughout both finite-difference neighborhoods.
        assert h / n < min(1 - lam, lam - 0.125)
        jacobian = np.empty((n, n))
        column_bounds = []
        for i in range(n):
            perturbation = np.eye(n)[i] * h
            plus, plus_radius = predict_and_radius(y + perturbation)
            minus, minus_radius = predict_and_radius(y - perturbation)
            jacobian[:, i] = (plus - minus) / (2 * h)
            # The exact map is affine in this neighborhood: no O(h^2)
            # truncation term. Only optimization and arithmetic remain.
            column_bounds.append((plus_radius + minus_radius) / (2 * h))
        # Reject an uninformatively loose certificate, rather than allowing
        # a poorly converged fit to make the derivative comparison vacuous.
        assert sum(column_bounds) <= 1e-5
        for i, allowance in enumerate(column_bounds):
            assert np.linalg.norm(jacobian[:, i] - projection[:, i]) <= allowance
        divergence = np.trace(jacobian)
        assert abs(divergence - active_rank) <= sum(column_bounds)
        if duplicate:
            # Nonzero count + intercept is wrong for this valid lasso optimum.
            assert abs(divergence - (len(active) + 1)) > 0.9
