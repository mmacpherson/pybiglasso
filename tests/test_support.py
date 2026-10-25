"""Negative controls and analytic examples for the independent test oracles."""

import copy
from dataclasses import dataclass

import numpy as np
import pytest
from scipy import sparse
from support.certificates import (
    gradient_lipschitz,
    objective,
    standardized_kkt,
    subgradient_residual,
    support_is_certified,
)
from support.paths import PATH_FIELDS, assert_bitwise, assert_same_trajectory


def test_subgradient_includes_intercept_and_penalty_factors():
    # At eta=log(3), p=3/4. Opposite X rows cancel X'p, while X'y/n=1/2.
    # The intercept's mean gradient is 3/4-1/2=1/4, and is never penalized.
    Z = np.array([[1.0, -1.0], [1.0, 1.0]])
    y = np.array([0.0, 1.0])
    theta = np.array([np.log(3), 0.0])
    residual = subgradient_residual(Z, y, theta, 0.1, 0.5, np.array([0.0]))
    np.testing.assert_allclose(
        residual, [0.25, -0.5], atol=8 * np.finfo(float).eps, rtol=0
    )
    np.testing.assert_allclose(
        standardized_kkt(Z, y, theta, 0.1, 0.5, np.array([0.0])),
        0.5,
        atol=8 * np.finfo(float).eps,
        rtol=0,
    )
    # At zero the nearest L1 subgradient removes only the available .1*.5*2.
    penalized = subgradient_residual(Z, y, theta, 0.1, 0.5, np.array([2.0]))
    np.testing.assert_allclose(
        penalized, [0.25, -0.4], atol=8 * np.finfo(float).eps, rtol=0
    )


def test_subgradient_batch_and_objective_have_analytic_values():
    # Each feature value has one row of each class. At b=log(3), its gradient
    # is 1/4 and mean NLL is (log(4)+log(4/3))/2. All targets are analytic.
    X = np.array([[-1.0], [-1.0], [1.0], [1.0]])
    Z = np.column_stack([np.ones(4), X])
    y = np.array([0.0, 1.0, 0.0, 1.0])
    b = np.log(3)
    theta = np.array([[0.0, 0.0], [0.0, b]])
    result = subgradient_residual(
        Z, y, theta, np.array([0.2, 0.4]), 0.5, np.array([2.0])
    )
    # Eight elementary operations on well-scaled values, each with a few ulps
    # for exp/log; this budget comes from their count, not solver output.
    tol = 32 * np.finfo(float).eps
    np.testing.assert_allclose(
        result, [[0.0, 0.0], [0.0, 0.25 + 0.4 * (1 + b)]], atol=tol, rtol=tol
    )
    expected = (np.log(4) + np.log(4 / 3)) / 2 + 0.4 * 2 * (0.5 * b + 0.25 * b * b)
    np.testing.assert_allclose(
        objective(X, y, 0.0, np.array([b]), np.ones(1), np.array([2.0]), 0.4, 0.5),
        expected,
        atol=tol,
        rtol=tol,
    )
    # Z'Z/n=I, so ||Z||_2^2/(4n)=1/4 (including the intercept).
    np.testing.assert_allclose(gradient_lipschitz(Z), 0.25, rtol=tol, atol=0)


def test_support_certificate_requires_both_strict_margins():
    kwargs = dict(distance=0.1, residual_bound=0.01, lipschitz=0.25)
    # A zero needs slack > .01+.25*.1=.035; a nonzero needs magnitude > .1.
    assert support_is_certified(
        np.array([0.0, 0.2]), np.array([0.16, 0.0]), 0.2, **kwargs
    )
    assert not support_is_certified(
        np.array([0.0, 0.2]), np.array([0.17, 0.0]), 0.2, **kwargs
    )
    assert not support_is_certified(
        np.array([0.0, 0.1]), np.array([0.16, 0.0]), 0.2, **kwargs
    )
    assert not support_is_certified(
        np.array([0.0, 0.2]), np.array([0.16, 0.0]), 0.2, rounding=0.006, **kwargs
    )
    # All dyadic: slack=1/4-1/8=1/8, exactly the bound
    # 1/16 + (1/4)*(1/4). Equality does not certify a zero.
    assert not support_is_certified(
        np.array([0.0]),
        np.array([0.125]),
        0.25,
        distance=0.25,
        residual_bound=0.0625,
        lipschitz=0.25,
    )


@dataclass
class ToyPath:
    """A model-shaped container, not recorded output from a solver."""

    coef: sparse.csc_array
    lambdas: np.ndarray
    intercept: np.ndarray
    deviance: np.ndarray
    iterations: np.ndarray
    n_rejected: np.ndarray

    def __len__(self):
        return len(self.lambdas)


def toy_path():
    return ToyPath(
        sparse.csc_array([[0.0, 0.25], [0.0, -0.5]]),
        np.array([0.1, 0.05]),
        np.zeros(2),
        np.array([10.0, 8.0]),
        np.array([1, 3]),
        np.array([2, 0]),
    )


@pytest.mark.parametrize("check", [assert_bitwise, assert_same_trajectory])
@pytest.mark.parametrize("field", (*PATH_FIELDS, "coef"))
def test_path_assertions_detect_corrupted_fields(check, field):
    path, changed = toy_path(), toy_path()
    check(path, changed)
    if field == "coef":
        coef = changed.coef.toarray()
        coef[0, 0] += 0.125
        changed.coef = sparse.csc_array(coef)
    else:
        getattr(changed, field)[0] += 1
    with pytest.raises(AssertionError):
        check(changed, path)


@pytest.mark.parametrize("change", ["sign-of-zero", "dtype", "shape"])
def test_bitwise_means_shape_dtype_and_bytes(change):
    a = np.array([0.0, 1.0])
    b = a.copy()
    if change == "sign-of-zero":
        b[0] = -0.0
    elif change == "dtype":
        b = b.astype(np.float32)
    else:
        b = b[None, :]
    with pytest.raises(AssertionError):
        assert_bitwise(a, b)


def test_prefix_is_explicit_and_checks_length():
    path = toy_path()
    short = copy.deepcopy(path)
    for field in PATH_FIELDS:
        setattr(short, field, getattr(short, field)[:1])
    short.coef = short.coef[:, :1]
    assert_bitwise(short, path, length=1)
    with pytest.raises(AssertionError):
        assert_bitwise(short, path)
    with pytest.raises(AssertionError):
        assert_bitwise(short, path, length=2)


def test_same_trajectory_rejects_relative_error_in_small_lambda():
    path = toy_path()
    path.lambdas[-1] = 1e-4
    changed = copy.deepcopy(path)
    changed.lambdas[-1] *= 1 + 1e-10
    # The lambda contract is relative (rtol=1e-12), including at the small
    # end of the grid. A fixed absolute floor would hide this corruption.
    with pytest.raises(AssertionError):
        assert_same_trajectory(changed, path)
