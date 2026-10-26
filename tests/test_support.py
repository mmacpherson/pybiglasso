"""Negative controls and analytic examples for the independent test oracles."""

import copy
import decimal
from dataclasses import dataclass

import numpy as np
import pytest
from scipy import sparse
from support.certificates import (
    assert_same_optimum,
    certified_distance,
    gradient_lipschitz,
    objective,
    residual_rounding,
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


def test_same_optimum_bound_holds_for_any_two_points_but_caps_bite():
    """Monotonicity of the subdifferential gives mu |d|^2 <= (e_a + e_b) |d|
    for any two points, not only near-optima, so random pairs, some with
    zeros, check the bound with no fitted expectation. Such points fail the
    caps, which assert_same_optimum checks first."""
    rng = np.random.default_rng(3)
    Z = np.column_stack([np.ones(40), rng.normal(size=(40, 3))])
    y = rng.binomial(1, 0.4, size=40).astype(float)
    pf = np.array([0.5, 1.0, 2.0])
    for _ in range(50):
        a, b = rng.normal(size=(2, 4)) * rng.binomial(1, 0.7, size=(2, 4))
        e = [
            np.linalg.norm(subgradient_residual(Z, y, t, 0.05, 0.5, pf)) for t in (a, b)
        ]
        assert np.linalg.norm(a - b) <= certified_distance(Z, y, a, b, *e)
        with pytest.raises(AssertionError):
            assert_same_optimum(Z, y, a, b, 0.05, 0.5, pf)


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


def test_residual_rounding_bounds_the_float_error():
    """Against the residual in 50-digit decimals, from the exact values of the
    same float inputs: near the optimum and away from it, with a zero
    coefficient and unequal penalty factors."""
    rng = np.random.default_rng(5)
    Z = np.column_stack([np.ones(200), rng.normal(size=(200, 3)) * [1, 10, 0.1]])
    y = rng.binomial(1, 0.3, size=200).astype(float)
    lam, alpha, pf = 0.02, 0.7, np.array([0.5, 1.0, 2.0])
    Dec = decimal.Decimal

    def exact(theta):
        with decimal.localcontext() as ctx:
            ctx.prec = 50
            t = [Dec(v) for v in theta]
            eta = [sum(Dec(z) * tj for z, tj in zip(row, t, strict=True)) for row in Z]
            score = [
                1 / (1 + (-e).exp()) - Dec(yi) for e, yi in zip(eta, y, strict=True)
            ]
            n = len(y)
            g = [sum(Dec(Z[i, j]) * score[i] for i in range(n)) / n for j in range(4)]
            out = [g[0]]
            for j, b in enumerate(t[1:]):
                l1 = Dec(lam) * Dec(alpha) * Dec(pf[j])
                gj = g[j + 1] + Dec(lam) * (1 - Dec(alpha)) * Dec(pf[j]) * b
                if b == 0:
                    out.append(gj - max(-l1, min(l1, gj)))
                else:
                    out.append(gj + (l1 if b > 0 else -l1))
            return np.array([float(v) for v in out])

    for theta in (np.array([-0.8, 0.3, 0.0, -2.0]), np.array([0.1, -0.05, 0.02, 0.0])):
        computed = subgradient_residual(Z, y, theta, lam, alpha, pf)
        want = exact(theta)
        # and half an ulp, to round the exact value to a float
        bound = residual_rounding(Z, y, theta, lam, alpha, pf)
        assert np.all(np.abs(computed - want) <= bound + np.spacing(abs(want)) / 2)


def test_same_optimum_computes_in_float64_whatever_the_inputs(monkeypatch):
    """residual_rounding's unit roundoff is float64's: float32 inputs must be
    promoted, not computed in float32 under a float64 error bound."""
    import support.certificates as certificates

    rng = np.random.default_rng(6)
    Z = np.column_stack([np.ones(40), rng.normal(size=(40, 2))]).astype(np.float32)
    y = rng.binomial(1, 0.5, size=40).astype(np.float32)
    theta = np.array([0.1, 0.3, -0.2], dtype=np.float32)
    promoted = [v.astype(np.float64) for v in (Z, y, theta)]
    np.testing.assert_array_equal(
        residual_rounding(Z, y, theta, 0.05, 0.5),
        residual_rounding(*promoted, 0.05, 0.5),
    )

    seen = []
    real = certificates.subgradient_residual

    def spy(Z, y, theta, *args, **kwargs):
        seen.append((Z.dtype, y.dtype, theta.dtype))
        return real(Z, y, theta, *args, **kwargs)

    monkeypatch.setattr(certificates, "subgradient_residual", spy)
    with pytest.raises(AssertionError):  # not an optimum: the caps refuse it
        certificates.assert_same_optimum(Z, y, theta, theta, 0.05, 0.5)
    assert seen and all(dtypes == (np.float64,) * 3 for dtypes in seen)
