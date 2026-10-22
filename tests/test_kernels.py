"""Small numerical reference checks, independent of path orchestration."""

import numpy as np
import pytest
from scipy.special import expit

from pybiglasso import binomial as solver


@pytest.fixture
def arrays():
    rng = np.random.default_rng(81)
    X = np.asfortranarray(rng.normal(size=(20, 4)))
    return X, rng.normal(size=20), rng.uniform(0.05, 0.25, size=20)


@pytest.mark.parametrize("subset", [False, True])
def test_center_scale_and_initial_crossproducts(arrays, subset):
    X, _, _ = arrays
    X = np.asfortranarray(X.astype(np.float32))
    rows = np.arange(1, 20, 2) if subset else None
    A = X if rows is None else X[rows]
    center, scale = solver.center_scale(X, rows)
    np.testing.assert_allclose(center, A.astype(float).mean(axis=0), atol=1e-15)
    np.testing.assert_allclose(scale, A.astype(float).std(axis=0), atol=1e-15)
    y = np.arange(20) % 2
    cols = np.array([0, 2, 3])
    expected = A[:, cols].astype(float).T @ (y if rows is None else y[rows])
    np.testing.assert_allclose(solver.build_sum_xy(X, y, rows, cols), expected)


@pytest.mark.parametrize("subset", [False, True])
def test_moments_and_shift(arrays, subset):
    X, residual, weight = arrays
    rows = np.arange(1, 20, 2) if subset else None
    A = X if rows is None else X[rows]
    r = residual.copy() if rows is None else residual[rows].copy()
    w = weight if rows is None else weight[rows]
    col = A[:, 2]
    expected = [col @ (w * r), (col**2) @ w, col @ w]
    np.testing.assert_allclose(solver.column_moments(X, 2, rows, r, w), expected)
    eta = np.zeros(len(r))
    before = r.copy()
    center = col.mean()
    got = solver.apply_shift(X, 2, rows, 0.7, center, r, eta, w)
    np.testing.assert_allclose(eta, 0.7 * (col - center))
    np.testing.assert_allclose(r, before - eta)
    np.testing.assert_allclose(got, w @ r)


def test_centered_moments_and_score_at_large_offset(arrays):
    X, r, w = arrays
    X += 1e8
    center, scale = X.mean(axis=0), X.std(axis=0)
    z = (X[:, 1] - center[1]) / scale[1]
    np.testing.assert_allclose(
        solver.centered_moments(X, 1, None, r, w, center[1], scale[1]),
        [z @ (w * r), (w * z**2).mean()],
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        solver.standardized_score(X, 1, None, r, r.sum(), center[1], scale[1]),
        z @ r / len(r),
        rtol=1e-14,
    )


@pytest.mark.parametrize("alpha,compat", [(1.0, False), (0.3, False), (0.3, True)])
@pytest.mark.parametrize("offset", [0.0, 1e8])
def test_coordinate_sweep_against_explicit_standardized_reference(
    arrays, alpha, compat, offset
):
    X, r, w = arrays
    X += offset
    center, scale = X.mean(axis=0), X.std(axis=0)
    Z = (X - center) / scale
    pf = np.array([1.0, 0.0, 2.0, 0.5])
    active = np.array([1, 1, 0, 1])
    a = np.array([0.2, -0.1, 0, 0.4])
    eta = np.zeros(len(r))
    expected_a, expected_r = a.copy(), r.copy()
    expected_eta = eta.copy()
    expected_max = 0.0
    for j in np.flatnonzero(active):
        v = np.dot(w, Z[:, j] ** 2) / len(r)
        u = Z[:, j] @ (w * expected_r) / len(r) + v * expected_a[j]
        l1, l2 = 0.05 * pf[j] * alpha, 0.05 * pf[j] * (1 - alpha)
        denominator = v * (1 + l2) if compat else v + l2
        b = np.sign(u) * max(abs(u) - l1, 0) / denominator
        shift = b - expected_a[j]
        expected_r -= shift * Z[:, j]
        expected_eta += shift * Z[:, j]
        expected_a[j] = b
        expected_max = max(expected_max, v * shift**2)
    total, maximum = solver.update_weights(
        X,
        None,
        0.05,
        alpha,
        pf,
        np.arange(4),
        center,
        scale,
        active,
        w,
        r,
        w @ r,
        eta,
        a,
        compat,
    )
    np.testing.assert_allclose(a, expected_a, atol=1e-12)
    np.testing.assert_allclose(r, expected_r, atol=1e-12)
    np.testing.assert_allclose(eta, expected_eta, atol=1e-12)
    np.testing.assert_allclose(
        [total, maximum], [w @ expected_r, expected_max], atol=1e-12
    )


def test_invalid_curvature_cannot_be_reported_as_convergence(arrays):
    X, r, w = arrays
    with pytest.raises(ValueError, match="curvature"):
        solver.update_weights(
            X,
            None,
            0.1,
            1.0,
            np.ones(4),
            np.arange(4),
            X.mean(0),
            X.std(0),
            np.ones(4, dtype=int),
            np.zeros_like(w),
            r,
            0.0,
            np.zeros(20),
            np.zeros(4),
            False,
        )


@pytest.mark.parametrize("tiny_weight", [1e-9, 1e-16])
def test_curvature_fallback_handles_positive_or_zero_cancelled_moment(tiny_weight):
    # The mean is modest, but nearly all the weight lies exactly at the mean.
    # Subtracting raw moments loses the tiny positive curvature, even when the
    # raw result remains positive. A nonpositive-only guard would miss this.
    X = np.array([[0.0], [1.0], [1.0], [2.0]], order="F")
    center, scale = X.mean(0), X.std(0)
    w = np.array([tiny_weight, 1.0, 1.0, tiny_weight])
    r = np.array([-1.0, 0.0, 0.0, 1.0])
    z = (X[:, 0] - center[0]) / scale[0]
    v = np.dot(w, z**2) / len(r)
    expected = (z @ (w * r) / len(r)) / v
    a = np.zeros(1)
    solver.update_weights(
        X,
        None,
        0.0,
        1.0,
        np.ones(1),
        np.array([0]),
        center,
        scale,
        np.ones(1, dtype=int),
        w,
        r,
        w @ r,
        np.zeros(4),
        a,
        False,
    )
    np.testing.assert_allclose(a[0], expected, rtol=1e-14)


@pytest.mark.parametrize("subset", [False, True])
def test_kkt_scans_match_explicit_standardized_scores(arrays, subset):
    X, residual, _ = arrays
    rows = np.arange(1, 20, 2) if subset else None
    A = X if rows is None else X[rows]
    s = residual if rows is None else residual[rows]
    center, scale = A.mean(0), A.std(0)
    scores = ((A - center) / scale).T @ s / len(s)
    a, pf = np.array([0.2, 0, 0, 0]), np.array([1.0, 0.1, 10.0, 0.1])
    l1, l2 = 0.1 * pf * 0.3, 0.1 * pf * 0.7
    violates = abs(scores - a * l2) > l1
    for scan, strong in [(solver.check_strong_set, True), (solver.check_rest, False)]:
        e1 = np.array([1, 0, 0, 0])
        e2 = np.array([1, 1, 1, 0])
        selected = (e1 == 0) & (e2 == 1) if strong else e2 == 0
        z = np.full(4, 123.0)
        count = scan(
            X, rows, 0.1, 0.3, pf, np.arange(4), center, scale, e1, e2, s, z, a
        )
        np.testing.assert_allclose(z[selected], scores[selected])
        np.testing.assert_array_equal(z[~selected], 123.0)
        assert count == np.count_nonzero(selected & violates)
        np.testing.assert_array_equal(
            e1, np.array([1, 0, 0, 0]) | (selected & violates)
        )


def test_intercept_update_and_probability_working_state(arrays):
    _, r, w = arrays
    before = r.copy()
    eta = np.zeros(len(r))
    delta = w @ r / w.sum()
    intercept, a0, total = solver.update_intercept(w, r, 0.4, eta)
    np.testing.assert_allclose([intercept, a0], 0.4 + delta)
    np.testing.assert_allclose(eta, delta)
    np.testing.assert_allclose(r, before - delta)
    assert abs(total) < 1e-14
    y = np.arange(len(r)) % 2
    eta = np.linspace(-5, 5, len(r))
    p, w, s, r = (np.zeros_like(eta) for _ in range(4))
    loss = solver.update_probs(y, None, eta, p, w, s, r)
    np.testing.assert_allclose(p, expit(eta))
    np.testing.assert_allclose(w, p * (1 - p))
    np.testing.assert_allclose(s, y - p)
    np.testing.assert_allclose(r, s / w)
    np.testing.assert_allclose(loss, np.logaddexp(0, np.where(y == 1, -eta, eta)).sum())


def test_final_loss_handles_extreme_logits_and_row_selection():
    eta = np.array([-1000.0, 1000.0, -1000.0, 1000.0])
    y = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    rows = np.array([0, 1, 4, 5])
    expected = np.logaddexp(0, np.where(y[rows] == 1, -eta, eta)).sum()
    assert solver.final_loss(y, rows, eta) == expected == 2000.0
