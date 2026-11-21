"""Centered prediction, the raw-scale export, and old fitted states.

Expectations come from exact rational arithmetic on the stored floats and the
binary64 projection of X, never from pybiglasso's own output.
"""

import dataclasses
import math
import pickle
import sys
import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import scipy.sparse as sps

import pybiglasso as bl
from pybiglasso import _rust
from pybiglasso import path as path_module

U = Fraction(1, 2**53)
ETA = Fraction(1, 2**1074)
MAX = sys.float_info.max
FIXTURES = Path(__file__).parent / "fixtures" / "old_state"


def exact_centered(path, X, k, rows=None):
    """The model at lambda k, evaluated exactly on the stored floats:
    centered_intercept + sum_j coef_j ((x_j - center_j) - offset_j)."""
    X = np.asarray(X, dtype=np.float64)
    if rows is not None:
        X = X[rows]
    coef = path.coef[:, [k]].tocoo()
    base = Fraction(float(path.centered_intercept[k]))
    out = []
    for row in X:
        value = base
        for j, b in zip(coef.row, coef.data, strict=True):
            offset = path.center_offset[j] if path.anchored[j] else 0.0
            value += Fraction(b) * (
                Fraction(row[j]) - Fraction(path.center[j]) - Fraction(offset)
            )
        out.append(value)
    return out


def forward_bound(path, X, k, rows=None):
    """gamma_(k+3) M + 2 k eta per row (centered_predictor's docstring)."""
    X = np.asarray(X, dtype=np.float64)
    if rows is not None:
        X = X[rows]
    coef = path.coef[:, [k]].tocoo()
    nnz = len(coef.data)
    gamma = (nnz + 3) * U / (1 - (nnz + 3) * U)
    bounds = []
    for row in X:
        m = abs(Fraction(float(path.centered_intercept[k])))
        for j, b in zip(coef.row, coef.data, strict=True):
            offset = path.center_offset[j] if path.anchored[j] else 0.0
            m += abs(Fraction(b)) * (
                abs(Fraction(row[j]) - Fraction(path.center[j])) + abs(Fraction(offset))
            )
        bounds.append(gamma * m + 2 * nnz * ETA)
    return bounds


def assert_within_bound(path, X, eta, rows=None):
    for k in range(len(path)):
        exact = exact_centered(path, X, k, rows)
        bound = forward_bound(path, X, k, rows)
        for i, (e, b) in enumerate(zip(exact, bound, strict=True)):
            error = abs(Fraction(float(eta[i, k])) - e)
            assert error <= b, (
                f"lambda {k}, row {i}: error {float(error):.3g} > {float(b):.3g}"
            )


def far_columns():
    """Columns far from zero next to their spread, each centered by today's
    arithmetic to well under the imprecise-center warning, so the fit itself
    is ordinary; only the raw form of its predictions loses accuracy."""
    rng = np.random.default_rng(20261121)
    n = 120
    cases = {}
    X = 1e8 + rng.standard_normal((n, 3))
    cases["1e8 + N(0, 1)"] = (
        np.asfortranarray(X),
        (X[:, 0] - 1e8) + rng.standard_normal(n),
    )
    # exactly centered: binary64 holds the mean 2^54 + 4 (README's example)
    x0 = np.where(np.arange(n) % 2 == 0, 2.0**54, 2.0**54 + 8)
    X = np.column_stack([x0, rng.standard_normal(n)])
    cases["2^54 and 2^54 + 8"] = (
        np.asfortranarray(X),
        (x0 - 2.0**54) / 8 + rng.standard_normal(n),
    )
    # alternating 2^50 and 2^50 + 2, which default mode centers exactly (#111)
    x0 = np.where(np.arange(n) % 2 == 0, 2.0**50, 2.0**50 + 2)
    X = np.column_stack([x0, rng.standard_normal(n)])
    cases["2^50 and 2^50 + 2"] = (
        np.asfortranarray(X),
        0.5 * (x0 - 2.0**50 - 1) + rng.standard_normal(n),
    )
    return cases


# Fails on main, observed: main predicts with the raw pair, which on these
# constructions exceeds the bound by up to 2.2e-8, 0.15 and 0.077.
@pytest.mark.parametrize("name", list(far_columns()))
def test_far_but_ordinary_columns_predict_within_the_forward_bound(name):
    X, y = far_columns()[name]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # an ordinary fit: no imprecise-center warning
        path = bl.biglasso(X, y, family="gaussian", nlambda=6)
    assert path.n_nonzero[-1] > 0
    assert not path.anchored.any()
    assert_within_bound(path, X, path.decision_function(X))
    # the raw pair, for the same model, does not meet that bound
    raw = path_module.linear_predictor(X, path.coef, path.intercept)
    worst = max(
        abs(Fraction(float(raw[i, k])) - e) - b
        for k in range(len(path))
        for i, (e, b) in enumerate(
            zip(exact_centered(path, X, k), forward_bound(path, X, k), strict=True)
        )
    )
    assert worst > 0


def test_ordinary_data_storage_rows_and_blocks_meet_the_bound(monkeypatch):
    # several column and row blocks, row subsets, Arrow chunks and int8
    monkeypatch.setattr(path_module, "CHUNK_ELEMENTS", 64)
    rng = np.random.default_rng(3)
    X = np.asfortranarray(rng.integers(-3, 4, size=(90, 7)).astype(np.int8))
    y = X[:, 0] - 0.5 * X[:, 3] + rng.standard_normal(90)
    path = bl.biglasso(X, y, family="gaussian", nlambda=5)
    assert path.n_nonzero[-1] >= 3
    rows = rng.permutation(90)[:40]
    assert_within_bound(path, X, path.decision_function(X))
    assert_within_bound(path, X, path.decision_function(X, row_indices=rows), rows)
    table = pa.table(
        {
            f"x{j}": pa.chunked_array([X[:13, j], X[13:14, j], X[14:, j]])
            for j in range(7)
        }
    )
    np.testing.assert_array_equal(
        path.decision_function(table), path.decision_function(X)
    )


def test_prediction_does_not_modify_X():
    rng = np.random.default_rng(4)
    X = np.asfortranarray(1e6 + rng.standard_normal((50, 4)))
    before = X.copy()
    path = bl.biglasso(X, X[:, 0] - 1e6, family="gaussian", nlambda=4)
    path.decision_function(X)
    assert X.tobytes() == before.tobytes()


def test_nonfinite_predictions_raise():
    rng = np.random.default_rng(5)
    X = np.asfortranarray(rng.standard_normal((40, 2)))
    path = bl.biglasso(X, 1e10 * X[:, 0], family="gaussian", nlambda=4)
    assert path.coef[0, -1] > 1  # so coef * (MAX - center) overflows
    huge = np.full((1, 2), MAX)
    with pytest.raises(ValueError, match="too large to predict"):
        path.decision_function(huge)


@pytest.mark.parametrize("family", ["gaussian", "binomial"])
def test_r_compatible_predicts_and_scores_with_the_raw_pair(family):
    rng = np.random.default_rng(6)
    X = np.asfortranarray(1e8 + rng.standard_normal((80, 3)))
    signal = X[:, 0] - 1e8
    y = signal if family == "gaussian" else (signal > 0).astype(float)
    path = bl.biglasso(X, y, family=family, nlambda=5, r_compatible=True)
    assert path.r_compatible
    raw = path_module.linear_predictor(X, path.coef, path.intercept)
    np.testing.assert_array_equal(path.decision_function(X), raw)
    restored = pickle.loads(pickle.dumps(path))
    np.testing.assert_array_equal(restored.decision_function(X), raw)
    default = dataclasses.replace(path, r_compatible=False)
    assert not np.array_equal(default.decision_function(X), raw)


@pytest.mark.parametrize("compatible", [False, True])
def test_early_stopping_predicts_folds_as_their_paths_predict(compatible):
    # the callback's held-out predictions are the fold path's own, bit for
    # bit, in the form the mode predicts with
    from pybiglasso.path import _predictor

    rng = np.random.default_rng(7)
    X = np.asfortranarray(1e7 + rng.standard_normal((150, 4)))
    y = ((X[:, 0] - 1e7) + 0.5 * rng.standard_normal(150) > 0).astype(float)
    train, test = (
        np.arange(0, 150, 3),
        np.setdiff1d(np.arange(150), np.arange(0, 150, 3)),
    )
    seen = []

    def stop(il, lambdas, cols, beta, scoring):
        coef = sps.csc_array(
            (beta, (cols, np.zeros(len(cols), dtype=np.int64))), shape=(X.shape[1], 1)
        )
        seen.append(_predictor(X, coef, scoring, test)[:, 0])
        return False

    path = bl.biglasso(
        X,
        y,
        family="binomial",
        row_indices=train,
        nlambda=8,
        r_compatible=compatible,
        _stop=stop,
    )
    assert path.r_compatible == compatible
    np.testing.assert_array_equal(
        np.column_stack(seen), path.decision_function(X, row_indices=test)
    )


def test_estimators_predict_in_centered_form():
    pytest.importorskip("sklearn")
    from pybiglasso.sklearn import BiglassoRegressor

    X, y = far_columns()["1e8 + N(0, 1)"]
    model = BiglassoRegressor(alpha=0.05).fit(X, y)
    path = model.path_
    assert model.centered_intercept_ == path.centered_intercept[0]
    eta = model.predict(X)[:, None]
    assert_within_bound(path, X, eta)


# The raw-scale export ----------------------------------------------------


def correctly_rounded(values):
    # int / int true division is correctly rounded in CPython
    return float(sum((Fraction(v) for v in values), Fraction(0)))


EXPORT_CASES = {
    "ties to even": (1.0, [1.0, 2.0**-53]),
    "tie above": (0.0, [1.0 + 2.0**-52, 2.0**-53]),
    "subnormals": (0.0, [5e-324, 5e-324, -1e-323 * 0.5]),
    "signed cancellation": (0.25, [1e16, *([1.0] * 100), -1e16]),
    "intermediate overflow": (0.0, [MAX, MAX, -MAX]),
    "a half-way sum of two products": (
        1.0 + 2.0**-26,
        [(1.0 + 2.0**-27) ** 2, 2.0**-53],
    ),
}


@pytest.mark.parametrize("name", list(EXPORT_CASES))
def test_exported_intercept_is_the_exactly_rounded_sum_subtracted(name):
    centered, products = EXPORT_CASES[name]
    got = _rust.exported_intercept(centered, np.array(products))
    assert got == centered - correctly_rounded(products)


def test_exported_intercept_forward_bound_with_a_rounded_away_product():
    # Centers and coefficients (1 + 2^-27, 1 + 2^-27) and (u, 1) about 1 + 2^-26:
    # the first product loses 2^-54, and the sum is then a tie that rounds
    # back down, so the export is 0 against an exact -1.5u. The bound must
    # count the sum's own rounding as well as the products' and the final
    # subtraction's.
    a = 1.0 + 2.0**-27
    centers, coefs = [a, 2.0**-53], [a, 1.0]
    centered = 1.0 + 2.0**-26
    products = [c * b for c, b in zip(centers, coefs, strict=True)]
    got = _rust.exported_intercept(centered, np.array(products))
    exact = Fraction(centered) - sum(
        (Fraction(c) * Fraction(b) for c, b in zip(centers, coefs, strict=True)),
        Fraction(0),
    )
    assert got == 0.0 and exact == -Fraction(3, 2**54)
    s = correctly_rounded(products)
    bound = (
        U * abs(Fraction(got))
        + U * abs(Fraction(s))
        + sum(
            (U * (abs(Fraction(p)) + ETA) / (1 - U) + ETA for p in products),
            Fraction(0),
        )
    )
    assert abs(Fraction(got) - exact) <= bound


@pytest.mark.parametrize(
    "centered,products",
    [
        (1.0, [math.inf, -math.inf]),
        (1.0, [math.nan]),
        (0.0, [MAX, MAX]),
        (MAX, [-MAX]),
    ],
    ids=["mixed infinities", "nan", "sum overflows", "subtraction overflows"],
)
def test_exported_intercept_is_none_when_not_representable(centered, products):
    assert _rust.exported_intercept(centered, np.array(products)) is None


def test_finite_exports_keep_their_bits():
    # Preservation, not correctness: on this IEEE build math.fsum is exactly
    # rounded, as the shared sum is, so every finite intercept main exported
    # (b0 - fsum(center * beta)) is unchanged.
    rng = np.random.default_rng(8)
    for _ in range(2000):
        n = int(rng.integers(1, 40))
        center = rng.standard_normal(n) * 10.0 ** rng.integers(-20, 20, n)
        beta = rng.standard_normal(n) * 10.0 ** rng.integers(-20, 20, n)
        b0 = float(rng.standard_normal() * 10.0 ** rng.integers(-5, 5))
        products = center * beta
        expected = b0 - math.fsum(products)
        assert _rust.exported_intercept(b0, products) == expected


def test_unrepresentable_raw_intercept_is_nan_with_a_warning(monkeypatch):
    # Native results whose raw-scale correction overflows (no real fit reaches
    # this: the response's spread would overflow first); main raised
    # OverflowError from math.fsum here.
    ncol = 2

    def solver(*args):
        return dict(
            iters=np.array([1]),
            col_map=np.arange(ncol),
            center=np.array([MAX, MAX]),
            offset=np.zeros(ncol),
            anchored=np.zeros(ncol, dtype=bool),
            scale=np.ones(ncol),
            intercept=np.array([0.5]),
            coef_rows=[np.arange(ncol)],
            coef_vals=[np.ones(ncol)],
            lambda_=np.array([0.1]),
            deviance=np.zeros(1),
            n_reject=np.zeros(1, dtype=np.int64),
        )

    monkeypatch.setattr(bl.fit, "_fit_path", lambda *a: solver(*a))
    X = np.zeros((3, ncol), order="F")
    with pytest.warns(UserWarning, match="not representable"):
        path = bl.biglasso(X, np.zeros(3), family="gaussian", lambdas=[0.1])
    assert np.isnan(path.intercept).all()
    assert path.centered_intercept[0] == 0.5
    near = np.full((2, ncol), MAX)
    # the centered form still predicts: 0.5 + (x - center) terms
    np.testing.assert_array_equal(path.decision_function(near), 0.5)
    compatible = dataclasses.replace(path, r_compatible=True)
    with pytest.raises(ValueError, match="raw intercept is not representable"):
        compatible.decision_function(near)


# Old fitted states --------------------------------------------------------
#
# The fixtures are a subset of those made by scripts/make_old_state_fixtures.py
# with a release-profile wheel built from commit b88db8b: pickle protocol 4
# of a fitted Gaussian path (default mode), a fitted BiglassoClassifier
# (which keeps no path) and an unfitted BiglassoRegressor. joblib rebuilds
# objects through the same __setstate__, so pickle alone covers the paths.


def load(name):
    return pickle.loads((FIXTURES / name).read_bytes())


@pytest.mark.parametrize(
    "name,kind",
    [
        ("path_gaussian_r_compatible_false.pkl", "BiglassoPath"),
        ("BiglassoClassifier_fitted.pkl", "BiglassoClassifier"),
    ],
)
def test_old_fitted_states_are_refused_clearly(name, kind):
    pytest.importorskip("sklearn")
    with pytest.raises(
        ValueError, match=f"This {kind} was saved by an earlier version"
    ):
        load(name)


def test_old_unfitted_estimators_load_and_fit():
    pytest.importorskip("sklearn")
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted

    model = load("BiglassoRegressor_unfitted.pkl")
    with pytest.raises(NotFittedError):
        check_is_fitted(model)
    rng = np.random.default_rng(9)
    X = np.asfortranarray(rng.standard_normal((40, 3)))
    model.fit(X, X[:, 0])
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))


# Defects that anchored centering will fix ----------------------------------


def unrepresentable_mean_column(n=64):
    """2^54 + 4 k for k in {0, 1, 2, 3}: binary64 spacing there is 4, and the
    exact mean, 2^54 + 4 mean(k), lies between representable centers."""
    rng = np.random.default_rng(10)
    k = rng.integers(0, 4, size=(n, 2))
    while (k.mean(axis=0) * 4 % 4 == 0).any():
        k = rng.integers(0, 4, size=(n, 2))
    return np.asfortranarray(2.0**54 + 4.0 * k), k


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Anchored centering, not yet implemented: one float64 center can't hold a mean between representable "
    "values, so the fixed Gaussian intercept misses its stationarity condition",
)
def test_gaussian_intercept_stationarity_for_unrepresentable_means():
    X, k = unrepresentable_mean_column()
    y = k[:, 0] - 0.5 * k[:, 1] + np.random.default_rng(11).standard_normal(len(k))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        path = bl.biglasso(X, y, family="gaussian", lambdas=[0.05], eps=1e-14)
    residual_mean = sum(
        (
            Fraction(float(v)) - e
            for v, e in zip(y, exact_centered(path, X, 0), strict=True)
        ),
        Fraction(0),
    ) / len(y)
    assert abs(residual_mean) <= 1e-8 * (1 + abs(path.coef.toarray()).sum())


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Not yet fixed: the binomial automatic grid scores y, not the null "
    "residual, so lambda_max picks up ybar times each standardized mean",
)
def test_binomial_lambda_max_is_the_null_gradient():
    X, k = unrepresentable_mean_column()
    y = (k[:, 0] + np.random.default_rng(12).standard_normal(len(k)) > 1.5).astype(
        float
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        path = bl.biglasso(X, y, family="binomial", nlambda=3)
    p0 = sum(map(Fraction, y), Fraction(0)) / len(y)
    gradient = max(
        abs(
            sum(
                (
                    (Fraction(x) - Fraction(path.center[j]))
                    / Fraction(path.scale[j])
                    * (Fraction(v) - p0)
                    for x, v in zip(X[:, j], y, strict=True)
                ),
                Fraction(0),
            )
            / len(y)
        )
        for j in range(X.shape[1])
    )
    assert abs(Fraction(path.lambdas[0]) - gradient) <= 1e-12 * gradient
