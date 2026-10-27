"""Local lasso divergence, not a pointwise statistical degrees-of-freedom claim.

For Gaussian responses, statistical df is the expectation of local divergence.
Away from selection boundaries, the lasso prediction map projects perturbations
onto the active column space. Include the free intercept in that space.
See Tibshirani & Taylor (2012), https://www.stat.cmu.edu/~ryantibs/papers/lassodf-aos.pdf.
"""

from fractions import Fraction
from itertools import product

import numpy as np
import pytest
from support.certificates import (
    KKT_TOL,
    _sqrt_fraction_upper,
    gaussian_duality_gap,
    gaussian_subgradient_residual,
    standardized_theta,
    with_ones,
)

import pybiglasso as bl


def exact_norm_upper(values):
    """Outward-rounded Euclidean norm of exact rational values."""
    return Fraction(_sqrt_fraction_upper(sum(value * value for value in values)))


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
        raw_coef = path.coef.toarray()[:, 0]
        theta = standardized_theta(path.intercept[0], raw_coef, center, scale)
        residual = gaussian_subgradient_residual(Z, response, theta, lam, 1.0)
        assert np.max(abs(residual)) <= KKT_TOL
        gap = gaussian_duality_gap(Z, response, theta, lam, 1.0)
        assert gap <= 1e-10
        # Collapse identical directions using EXACT binary64 input values.
        # Their penalty is |total| at the optimum, so the reduced problem has
        # the SAME fitted response and Hessian I (including the intercept).
        # Strong monotonicity bounds its coefficient error by the subgradient
        # norm; ||[1,q]||=sqrt(n) converts it to prediction error.
        raw = [Fraction(float(value)) for value in raw_coef]
        exact_theta = [
            Fraction(float(path.intercept[0]))
            + sum(Fraction(float(c)) * b for c, b in zip(center, raw, strict=True)),
            *(Fraction(float(s)) * b for s, b in zip(scale, raw, strict=True)),
        ]
        collapsed = (
            [
                exact_theta[0],
                exact_theta[1] + exact_theta[3],
                exact_theta[2],
                exact_theta[4],
            ]
            if duplicate
            else exact_theta
        )
        reduced = [[1, *(int(value) for value in row)] for row in q]
        exact_prediction = [
            sum(z * b for z, b in zip(row, collapsed, strict=True)) for row in reduced
        ]
        errors = [
            prediction - Fraction(float(value))
            for prediction, value in zip(exact_prediction, response, strict=True)
        ]
        subgradient = []
        for j, coefficient in enumerate(collapsed):
            gradient = (
                sum(row[j] * error for row, error in zip(reduced, errors, strict=True))
                / n
            )
            if j:
                penalty = Fraction(lam)
                gradient += (
                    penalty * (1 if coefficient > 0 else -1)
                    if coefficient
                    else -max(-penalty, min(penalty, gradient))
                )
            subgradient.append(gradient)
        # sqrt(n)*||g||, rounded outward once, without a rounded residual,
        # coefficient collapse, or raw-to-standardized conversion in the bound.
        radius = Fraction(
            _sqrt_fraction_upper(n * sum(value * value for value in subgradient))
        )
        prediction = [Fraction(float(value)) for value in path.predict(X)[:, 0]]
        rounding = exact_norm_upper(
            value - exact
            for value, exact in zip(prediction, exact_prediction, strict=True)
        )
        return prediction, radius + rounding

    baseline, radius = predict_and_radius(y)
    assert (
        sum(
            (value - Fraction(float(expected))) ** 2
            for value, expected in zip(baseline, Z @ expected_theta, strict=True)
        )
        <= radius**2
    )
    assert gaussian_duality_gap(Z, y, expected_theta, lam, 1.0) <= 1e-12

    for h in (0.01, 0.005):
        # A coordinate perturbation changes each centered orthogonal score
        # by at most h/n. These analytic margins justify the SAME active
        # subspace/signs throughout both finite-difference neighborhoods.
        assert h / n < min(1 - lam, lam - 0.125)
        columns = []
        column_bounds = []
        for i in range(n):
            perturbation = np.eye(n)[i] * h
            responses = (y + perturbation, y - perturbation)
            plus, plus_radius = predict_and_radius(responses[0])
            minus, minus_radius = predict_and_radius(responses[1])
            denominator = 2 * Fraction(h)
            columns.append(
                [(a - b) / denominator for a, b in zip(plus, minus, strict=True)]
            )
            # Account for the ACTUAL rounded y +/- h, not ideal real additions.
            # Their score changes remain inside the same analytic margins.
            for response in responses:
                displacement = sum(
                    abs(Fraction(float(a)) - Fraction(float(b)))
                    for a, b in zip(response, y, strict=True)
                )
                assert displacement / n < min(
                    1 - Fraction(lam), Fraction(lam) - Fraction(0.125)
                )
            input_error = [
                (Fraction(float(a)) - Fraction(float(b))) / denominator - int(j == i)
                for j, (a, b) in enumerate(zip(*responses, strict=True))
            ]
            projected_error = [
                sum(
                    Fraction(float(weight)) * error
                    for weight, error in zip(row, input_error, strict=True)
                )
                for row in projection
            ]
            # The exact map is affine in this neighborhood: no O(h^2)
            # truncation term. Only optimization and arithmetic remain.
            column_bounds.append(
                (plus_radius + minus_radius) / denominator
                + exact_norm_upper(projected_error)
            )
        # Reject an uninformatively loose certificate, rather than allowing
        # a poorly converged fit to make the derivative comparison vacuous.
        assert sum(column_bounds) <= 1e-5
        # Central differences, norm comparisons and trace additions below are
        # rational too: no unbudgeted subtraction/division/reduction rounding.
        for i, allowance in enumerate(column_bounds):
            assert (
                sum(
                    (value - Fraction(float(expected))) ** 2
                    for value, expected in zip(
                        columns[i], projection[:, i], strict=True
                    )
                )
                <= allowance**2
            )
        divergence = sum(column[i] for i, column in enumerate(columns))
        assert abs(divergence - active_rank) <= sum(column_bounds)
        if duplicate:
            # Nonzero count + intercept is wrong for this valid lasso optimum.
            assert abs(divergence - (len(active) + 1)) > 0.9
