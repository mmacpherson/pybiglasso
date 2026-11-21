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
    # The target after anchored centering: tau = 1e-8 standardized units per
    # unit of standardized slope, plus the rounding of ybar itself and an
    # allowance for the residual updates' rounding, in y's units.
    beta = abs(path.coef.toarray()[:, 0] * path.scale)
    ybar = Fraction(float(path.centered_intercept[0]))
    exact_mean = sum(map(Fraction, map(float, y)), Fraction(0)) / len(y)
    z_max = max(
        abs(Fraction(float(x)) - Fraction(float(path.center[j])))
        / Fraction(float(path.scale[j]))
        for j in range(X.shape[1])
        for x in X[:, j]
    )
    updates = (
        64
        * len(y)
        * U
        * (
            max(abs(Fraction(float(v)) - ybar) for v in y)
            + sum(map(Fraction, beta)) * z_max
        )
    )
    bound = (
        Fraction(1, 10**8) * sum(map(Fraction, beta)) + abs(exact_mean - ybar) + updates
    )
    assert abs(residual_mean) <= bound


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


# Reduction order, scratch memory and floating-point settings --------------


def test_each_prediction_is_independent_of_the_other_lambdas(monkeypatch):
    # Summed in blocks of columns, the union of the requested lambdas' columns
    # set the partial sums: this logit was 0 alone and 1 beside the second
    # lambda, and so changed class.
    monkeypatch.setattr(path_module, "CHUNK_ELEMENTS", 4)
    path = bl.BiglassoPath(
        lambdas=np.array([1.0, 0.5]),
        intercept=np.zeros(2),
        centered_intercept=np.array([-1e16, 0.0]),
        coef=sps.csc_array(np.array([[1e16, 0], [0, 1], [1, 0], [0, 1]])),
        deviance=np.zeros(2),
        iterations=np.zeros(2, dtype=int),
        n_rejected=np.zeros(2, dtype=int),
        center=np.zeros(4),
        center_offset=np.zeros(4),
        anchored=np.zeros(4, dtype=bool),
        scale=np.ones(4),
        penalty_factor=np.ones(4),
        feature_names=tuple("abcd"),
        alpha=1.0,
        family="binomial",
        named_features=False,
        r_compatible=False,
    )
    X = np.ones((1, 4))
    # ((-1e16 + 1e16) + 1), the intercept and then each term in column order
    assert path.decision_function(X, lambdas=[1.0])[0, 0] == 1.0
    assert path.decision_function(X)[0, 0] == 1.0
    assert path.predict(X, lambdas=[1.0])[0, 0] == path.predict(X)[0, 0] == 1


def test_selected_whole_path_and_callback_predictions_agree(monkeypatch):
    # far columns cancel; a budget of 8 values splits the 12 columns into
    # blocks, so the union of the requested lambdas' columns sets their bounds
    monkeypatch.setattr(path_module, "CHUNK_ELEMENTS", 8)
    rng = np.random.default_rng(13)
    X = np.asfortranarray(1e8 + rng.standard_normal((90, 12)))
    y = ((X[:, :4] - 1e8) @ [1.0, -1.0, 0.5, 0.5] + rng.standard_normal(90) > 0).astype(
        float
    )
    train, test = np.arange(0, 90, 2), np.arange(1, 90, 2)
    seen = []

    def stop(il, lambdas, cols, beta, scoring):
        coef = sps.csc_array(
            (beta, (cols, np.zeros(len(cols), dtype=np.int64))), shape=(X.shape[1], 1)
        )
        seen.append(path_module._predictor(X, coef, scoring, test)[:, 0])
        return False

    path = bl.biglasso(
        X, y, family="binomial", row_indices=train, nlambda=10, _stop=stop
    )
    assert path.n_nonzero[-1] >= 6
    whole = path.decision_function(X, row_indices=test)
    selected = np.column_stack(
        [
            path.decision_function(X, lambdas=[lam], row_indices=test)[:, 0]
            for lam in path.lambdas
        ]
    )
    assert whole.tobytes() == np.asfortranarray(selected).tobytes()
    np.testing.assert_array_equal(np.column_stack(seen), whole)


def test_raw_compatible_prediction_is_unchanged_across_blocks(monkeypatch):
    # r_compatible keeps the historical raw evaluation, blocks and all: the
    # same as summing each block's sparse product into the raw intercept
    monkeypatch.setattr(path_module, "CHUNK_ELEMENTS", 96)
    rng = np.random.default_rng(14)
    X = np.asfortranarray(1e8 + rng.standard_normal((90, 12)))
    path = bl.biglasso(
        X, X[:, 0] - 1e8, family="gaussian", nlambda=8, r_compatible=True
    )
    expected = np.broadcast_to(path.intercept, (90, len(path))).copy()
    used = np.unique(path.coef.indices)
    width = min(len(used), 96)
    for c0 in range(0, len(used), width):
        cols = used[c0 : c0 + width]
        for start in range(0, 90, 96 // width):
            stop = min(start + 96 // width, 90)
            expected[start:stop] += X[start:stop][:, cols] @ sps.csr_array(
                path.coef[cols, :]
            )
    assert path.decision_function(X).tobytes() == expected.tobytes()


def scratch_peak(fn):
    import tracemalloc

    tracemalloc.start()
    try:
        result = fn()
        return tracemalloc.get_traced_memory()[1], result
    finally:
        tracemalloc.stop()


def test_centered_prediction_scratch_stays_within_its_budget():
    # Fortran float64 X, every column used, one lambda: blocks overlapped
    # their predecessors' buffers and reached 50 MiB against 32 MiB
    rng = np.random.default_rng(15)
    X = np.asfortranarray(rng.standard_normal((6144, 1024)))
    coef = sps.csc_array(np.ones((1024, 1)))
    peak, eta = scratch_peak(
        lambda: path_module.centered_predictor(
            X,
            coef,
            np.zeros(1),
            np.zeros(1024),
            np.zeros(1024),
            np.zeros(1024, dtype=bool),
        )
    )
    assert peak <= path_module.CHUNK_ELEMENTS * 8 + eta.nbytes


@pytest.mark.parametrize("storage", ["int8 rows", "arrow rows", "float64 C"])
def test_centered_prediction_scratch_budget_over_many_blocks(monkeypatch, storage):
    monkeypatch.setattr(path_module, "CHUNK_ELEMENTS", 1 << 14)
    rng = np.random.default_rng(16)
    n, p, nlambda = 20000, 6, 3
    values = rng.integers(-3, 4, size=(n, p)).astype(np.int8)
    rows = rng.permutation(n)[: n // 2]
    if storage == "int8 rows":
        X = np.asfortranarray(values)
    elif storage == "arrow rows":
        X = pa.table(
            {
                f"x{j}": pa.chunked_array([values[:777, j], values[777:, j]])
                for j in range(p)
            }
        )
    else:
        X, rows = np.ascontiguousarray(values, dtype=np.float64), None
    coef = sps.csc_array(rng.standard_normal((p, nlambda)))
    peak, eta = scratch_peak(
        lambda: path_module.centered_predictor(
            X,
            coef,
            np.zeros(nlambda),
            np.ones(p),
            np.full(p, 0.5),
            np.ones(p, dtype=bool),
            rows,
        )
    )
    # the scratch budget, the result, and the sparse coefficients' index
    assert peak <= path_module.CHUNK_ELEMENTS * 8 + eta.nbytes + (1 << 14)


def test_overflow_reaches_its_specified_result_under_raising_settings(monkeypatch):
    ncol = 1

    def solver(*args):
        return dict(
            iters=np.array([1]),
            col_map=np.arange(ncol),
            center=np.array([MAX]),
            offset=np.zeros(ncol),
            anchored=np.zeros(ncol, dtype=bool),
            scale=np.ones(ncol),
            intercept=np.array([0.5]),
            coef_rows=[np.arange(ncol)],
            coef_vals=[np.array([2.0])],  # MAX * 2 overflows
            lambda_=np.array([0.1]),
            deviance=np.zeros(1),
            n_reject=np.zeros(1, dtype=np.int64),
        )

    monkeypatch.setattr(bl.fit, "_fit_path", lambda *a: solver(*a))
    X = np.zeros((3, ncol), order="F")
    for settings in (dict(all="raise"), dict()):
        with np.errstate(**settings), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            path = bl.biglasso(X, np.zeros(3), family="gaussian", lambdas=[0.1])
        assert np.isnan(path.intercept[0])
        # the fit's own warning, and no low-level overflow warning beside it
        assert [str(w.message)[:40] for w in caught] == [
            "The intercept on X's raw scale is not re"
        ]
    with (
        np.errstate(all="raise"),
        pytest.raises(ValueError, match="too large to predict"),
    ):
        path.decision_function(np.full((1, ncol), -MAX))


def test_centered_prediction_scratch_budget_with_many_lambdas():
    # At 100 lambdas a block of rows held products for every lambda and a
    # finiteness mask of the same shape, over the budget together.
    X = np.asfortranarray(np.ones((40000, 1)))
    coef = sps.csc_array(np.full((1, 100), 0.5))
    peak, eta = scratch_peak(
        lambda: path_module.centered_predictor(
            X, coef, np.zeros(100), np.zeros(1), np.zeros(1), np.zeros(1, dtype=bool)
        )
    )
    np.testing.assert_array_equal(eta, 0.5)
    assert peak <= path_module.CHUNK_ELEMENTS * 8 + eta.nbytes + (1 << 14)


def test_finite_underflow_keeps_its_bits_under_raising_settings(monkeypatch):
    # Predicting at the smallest normal float: 0.1 * tiny is subnormal
    x = np.array([-1.0, 1.0, -1.0, 1.0])
    path = bl.biglasso(x[:, None], 0.2 * x, family="gaussian", lambdas=[0.1])
    probe = np.full((1, 1), np.finfo(float).tiny)
    expected = path.decision_function(probe)
    assert 0 < abs(expected[0, 0]) < np.finfo(float).tiny
    with np.errstate(all="raise"):
        assert path.decision_function(probe).tobytes() == expected.tobytes()

    # an export product that underflows to a subnormal
    def solver(*args):
        return dict(
            iters=np.array([1]),
            col_map=np.arange(1),
            center=np.array([np.finfo(float).tiny]),
            offset=np.zeros(1),
            anchored=np.zeros(1, dtype=bool),
            scale=np.array([4.0]),  # and the slope 0.5 / 4 is exact
            intercept=np.array([0.0]),
            coef_rows=[np.arange(1)],
            coef_vals=[np.array([0.5])],
            lambda_=np.array([0.1]),
            deviance=np.zeros(1),
            n_reject=np.zeros(1, dtype=np.int64),
        )

    monkeypatch.setattr(bl.fit, "_fit_path", lambda *a: solver(*a))
    X = np.zeros((3, 1), order="F")
    default = bl.biglasso(X, np.zeros(3), family="gaussian", lambdas=[0.1])
    assert 0 < abs(default.intercept[0]) < np.finfo(float).tiny
    with np.errstate(all="raise"):
        strict = bl.biglasso(X, np.zeros(3), family="gaussian", lambdas=[0.1])
    assert strict.intercept.tobytes() == default.intercept.tobytes()
    assert strict.coef.data.tobytes() == default.coef.data.tobytes()
