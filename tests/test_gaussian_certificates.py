"""Analytic and negative controls for independent least-squares certificates."""

import numpy as np
import pytest
from support.certificates import (
    gap_coefficient_distance,
    gaussian_curvature,
    gaussian_duality_gap,
    gaussian_gap_safe_zeros,
    gaussian_subgradient_residual,
)


def test_gaussian_curvature_promotes_float32_before_arithmetic():
    Z = np.column_stack([np.ones(8), np.tile([-0.1, 0.1], 4)]).astype(np.float32)
    # Orthogonal columns: the smaller Hessian eigenvalue is exactly x**2.
    exact_curvature = float(np.float32(0.1)) ** 2
    lower = float(gaussian_curvature(Z))
    assert 0 < lower <= exact_curvature
    assert lower == gaussian_curvature(Z.astype(np.float64))


@pytest.fixture
def orthogonal():
    # Each column has mean zero, unit population variance, and X'X/n = I.
    X = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    return np.column_stack([np.ones(4), X]), 3 + 2 * X[:, 0]


@pytest.mark.parametrize("alpha", [0.2, 0.5, 1.0])
def test_gaussian_certificates_at_analytic_optimum(orthogonal, alpha):
    Z, y = orthogonal
    lam = 0.5
    theta = np.array([3.0, (2 - lam * alpha) / (1 + lam * (1 - alpha)), 0.0])
    np.testing.assert_allclose(
        gaussian_subgradient_residual(Z, y, theta, lam, alpha),
        0,
        atol=8 * np.finfo(float).eps,
    )
    assert gaussian_duality_gap(Z, y, theta, lam, alpha) <= 16 * np.finfo(float).eps
    np.testing.assert_array_equal(
        gaussian_gap_safe_zeros(Z, y, theta, lam, alpha), [False, True]
    )
    assert 0.99 < gaussian_curvature(Z, lam, alpha) <= 1


@pytest.mark.parametrize("alpha", [0.5, 1.0])
def test_gaussian_gap_bounds_wrong_intercept_and_coefficients(orthogonal, alpha):
    Z, y = orthogonal
    lam = 0.5
    optimum = np.array([3.0, (2 - lam * alpha) / (1 + lam * (1 - alpha)), 0.0])
    wrong = optimum + [1.0, -0.75, 0.2]

    def objective(theta):
        return np.mean((Z @ theta - y) ** 2) / 2 + lam * (
            alpha * abs(theta[1:]).sum() + (1 - alpha) * (theta[1:] @ theta[1:]) / 2
        )

    gap = gaussian_duality_gap(Z, y, wrong, lam, alpha)
    assert gap >= objective(wrong) - objective(optimum) > 0.5
    assert np.linalg.norm(gaussian_subgradient_residual(Z, y, wrong, lam, alpha)) > 1
    if alpha < 1:
        assert np.linalg.norm(wrong[1:] - optimum[1:]) <= gap_coefficient_distance(
            gap, lam, alpha
        )


def test_gaussian_curvature_bounds_distance(orthogonal):
    Z, y = orthogonal
    theta_a, theta_b = np.array([3.0, 1.0, 0.0]), np.array([2.0, 1.5, -0.2])
    e1 = np.linalg.norm(gaussian_subgradient_residual(Z, y, theta_a, 0.5, 0.5))
    e2 = np.linalg.norm(gaussian_subgradient_residual(Z, y, theta_b, 0.5, 0.5))
    assert np.linalg.norm(theta_a - theta_b) <= (e1 + e2) / gaussian_curvature(
        Z, 0.5, 0.5
    )


def test_gaussian_wide_gap_and_ridge_distance():
    # Three identical columns and only two rows. Symmetry gives the EN optimum.
    Z = np.array([[1.0, -1.0, -1.0, -1.0], [1.0, 1.0, 1.0, 1.0]])
    y = np.array([1.0, 5.0])
    lam, alpha = 0.5, 0.5
    b = (2 - lam * alpha) / (3 + lam * (1 - alpha))
    optimum = np.r_[3.0, np.full(3, b)]
    np.testing.assert_allclose(
        gaussian_subgradient_residual(Z, y, optimum, lam, alpha),
        0,
        atol=8 * np.finfo(float).eps,
    )
    assert gaussian_duality_gap(Z, y, optimum, lam, alpha) < 16 * np.finfo(float).eps
    assert 0.24 < gaussian_curvature(Z, lam, alpha) <= 0.25
    with pytest.raises(AssertionError, match="no curvature"):
        gaussian_curvature(Z)


def test_gaussian_gap_rejects_unpenalized_columns(orthogonal):
    Z, y = orthogonal
    with pytest.raises(ValueError):
        gaussian_duality_gap(Z, y, np.zeros(3), 1.0, 0.5, [0.0, 1.0])
