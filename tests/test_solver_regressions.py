"""Regression tests for the numerical and input failures found before release."""

import warnings
from fractions import Fraction

import numpy as np
import pytest
from scipy.special import expit
from support.certificates import (
    assert_same_optimum,
    path_theta,
    standardize,
    standardized_theta,
    subgradient_residual,
    with_ones,
)

from pybiglasso import biglasso


@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    X = np.asfortranarray(rng.normal(size=(100, 3)))
    y = rng.binomial(1, expit(X[:, 0]))
    return X, y


@pytest.mark.parametrize("linear", [False, True])
@pytest.mark.parametrize("unpenalized", [False, True])
def test_automatic_singleton_rejected_but_explicit_singleton_fits(
    data, linear, unpenalized
):
    X, y = data
    pf = np.array([0, 1, 1]) if unpenalized else None
    kwargs = dict(penalty_factor=pf, lambda_log_scale=not linear, eps=1e-14)
    with pytest.raises(ValueError, match=r"at least 2.*lambdas=\[value\]"):
        biglasso(X, y, nlambda=1, family="binomial", **kwargs)
    full = biglasso(X, y, nlambda=5, family="binomial", **kwargs)
    lam = full.lambdas[-1]
    explicit = biglasso(X, y, nlambda=1, lambdas=[lam], family="binomial", **kwargs)
    assert explicit.n_nonzero[0] > 0
    # two routes to the optimum at lam (alpha 1, the default): each passes
    # its KKT conditions, and convexity bounds the gap
    Xs, center, scale = standardize(X)
    assert_same_optimum(
        with_ones(Xs),
        y,
        path_theta(full, len(full) - 1, center, scale),
        path_theta(explicit, 0, center, scale),
        lam,
        1.0,
        1.0 if pf is None else pf,
    )


@pytest.mark.parametrize("r_compatible", [False, True])
@pytest.mark.parametrize("subset", [False, True])
def test_translation_preserves_solution_and_kkt(data, r_compatible, subset):
    X, y = data
    rows = np.arange(0, len(y), 2) if subset else None
    lambdas = np.array([0.03, 0.01])
    kwargs = dict(
        lambdas=lambdas, eps=1e-12, row_indices=rows, r_compatible=r_compatible
    )
    shifted = np.asfortranarray(X + 1e8)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        base = biglasso(X, y, family="binomial", **kwargs)
        fit = biglasso(shifted, y, family="binomial", **kwargs)
    B, A, yy = (X, shifted, y) if rows is None else (X[rows], shifted[rows], y[rows])
    eta = fit.intercept + A @ fit.coef.toarray()
    Xs, center, scale = standardize(A)
    coef = fit.coef.toarray()
    theta = np.vstack([fit.intercept + center @ coef, coef * scale[:, None]])
    # the lasso (alpha 1, the default) at the lambdas requested, with
    # predictions in the shifted fit's own arithmetic. 5e-7 is the
    # first-release review's ceiling, tighter than KKT_TOL; it is kept as a
    # stated cap, not derived
    cap = 5e-7
    residual = subgradient_residual(with_ones(Xs), yy, theta, lambdas, 1.0, eta=eta)
    assert abs(residual).max() < cap

    # the shift rounds X to 1e8's spacing (1.5e-8), so the shifted problem is
    # a slightly different one. The unshifted fit must pass its KKT conditions
    # too, at the same cap: then convexity bounds the two fits' distance
    _, base_center, base_scale = standardize(B)
    for k, lam in enumerate(lambdas):
        assert_same_optimum(
            with_ones(Xs),
            yy,
            path_theta(base, k, base_center, base_scale),
            path_theta(fit, k, center, scale),
            lam,
            1.0,
            cap=cap,
        )

    # predictions on the shifted rows cancel terms near 1e8: in any summation
    # order, a sum of m float products is within gamma_m of the exact sum of
    # the terms' magnitudes (Higham, Accuracy and Stability, section 3.1)
    m = X.shape[1] + 1
    unit = np.finfo(float).eps / 2
    gamma = m * unit / (1 - m * unit)
    got = fit.decision_function(shifted)
    for k in range(len(fit)):
        c = [Fraction(v) for v in coef[:, k]]
        for i, row in enumerate(shifted):
            terms = [Fraction(fit.intercept[k])]
            terms += [Fraction(x) * cj for x, cj in zip(row, c, strict=True)]
            exact = sum(terms)
            size = sum(abs(t) for t in terms)
            # and half an ulp to round the exact value for the comparison
            assert abs(got[i, k] - float(exact)) <= gamma * size + unit * abs(exact)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("subset", [False, True])
def test_nonfinite_memmap_rejected_in_standardization(data, tmp_path, value, subset):
    X, y = data
    mm = np.lib.format.open_memmap(
        tmp_path / "X.npy", mode="w+", dtype=X.dtype, shape=X.shape, fortran_order=True
    )
    mm[:] = X
    mm[0, 1] = value
    rows = np.arange(0, len(y), 2) if subset else None
    with pytest.raises(ValueError, match="finite"):
        biglasso(mm, y, row_indices=rows, family="binomial")


@pytest.mark.parametrize("value", [np.nan, np.inf, 1e300], ids=str)
def test_unused_rows_do_not_change_fit(data, value):
    X, y = data
    rows = np.arange(0, len(y), 2)
    copy = biglasso(np.asfortranarray(X[rows]), y[rows], nlambda=3, family="binomial")
    X[1, :] = value
    fit = biglasso(X, y, row_indices=rows, nlambda=3, family="binomial")
    np.testing.assert_array_equal(fit.iterations, copy.iterations)
    # the same arithmetic in another grouping: rounding, far below eps
    np.testing.assert_allclose(
        fit.coef.toarray(), copy.coef.toarray(), rtol=1e-9, atol=1e-12
    )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"lambdas": [np.nan]}, "lambdas"),
        ({"lambdas": [np.inf]}, "lambdas"),
        ({"lambdas": np.array([0.1 + 1j])}, "real values"),
        ({"lambdas": [[0.1, 0.01]]}, "lambdas"),
        ({"penalty_factor": [1, 1, np.nan]}, "penalty_factor"),
        ({"penalty_factor": [1, 1, np.inf]}, "penalty_factor"),
        ({"penalty_factor": np.array([1, 1, 1j])}, "real values"),
        ({"row_indices": [0, 1.1, 2]}, "integer indices"),
        ({"row_indices": [0, np.nan, 2]}, "integer indices"),
        ({"row_indices": [[0, 1], [2, 3]]}, "integer indices"),
        ({"row_indices": []}, "integer indices"),
        ({"row_indices": [True, False]}, "integer indices"),
        ({"nlambda": 0}, "nlambda"),
        ({"nlambda": 1.5}, "nlambda"),
        ({"nlambda": True}, "nlambda"),
        ({"max_iter": 0}, "max_iter"),
        ({"max_iter": -1}, "max_iter"),
        ({"eps": 0}, "eps"),
        ({"eps": np.nan}, "eps"),
        ({"eps": np.inf}, "eps"),
        ({"lambda_min": 0}, "lambda_min"),
        ({"lambda_min": 2}, "lambda_min"),
        ({"lambda_min": np.nan}, "lambda_min"),
        ({"dfmax": -1}, "dfmax"),
        ({"n_threads": 0}, "n_threads"),
        ({"alpha": np.nan}, "Alpha"),
    ],
)
def test_invalid_parameters_rejected(data, kwargs, match):
    with pytest.raises(ValueError, match=match):
        biglasso(*data, family="binomial", **kwargs)


@pytest.mark.parametrize("shape", [(100,), (100, 0), (0, 3), (100, 3, 1)])
def test_invalid_design_shape(data, shape):
    with pytest.raises(ValueError, match="two-dimensional"):
        biglasso(np.empty(shape), data[1], family="binomial")


def test_response_must_be_vector(data):
    X, y = data
    with pytest.raises(ValueError, match="one-dimensional"):
        biglasso(X, y[:, None], family="binomial")


@pytest.mark.parametrize("dtype", [object, np.complex128, np.float16])
def test_unsupported_design_dtype(data, dtype):
    X, y = data
    with pytest.raises(ValueError, match="dtype"):
        biglasso(X.astype(dtype), y, family="binomial")


def test_constant_design_has_clear_error(data):
    X, y = data
    with pytest.raises(ValueError, match="no varying features"):
        biglasso(np.ones_like(X), y, family="binomial")


def test_only_unpenalized_features_vary(data):
    X, y = data
    X[:, 2] = 1
    with pytest.raises(ValueError, match="varying penalized feature"):
        biglasso(X, y, penalty_factor=[0, 0, 1], family="binomial")
    explicit = biglasso(
        X,
        y,
        penalty_factor=[0, 0, 1],
        lambdas=[0.1, 0.01],
        eps=1e-14,
        family="binomial",
    )
    # with column 2 constant, both lambdas pose one problem, the unpenalized
    # fit of columns 0 and 1: each passes KKT, and convexity bounds the gap
    coef = explicit.coef.toarray()
    assert not coef[2].any()
    Xs, center, scale = standardize(X[:, :2])
    Z = with_ones(Xs)
    a, b = (
        standardized_theta(explicit.intercept[k], coef[:2, k], center, scale)
        for k in range(2)
    )
    assert_same_optimum(Z, y, a, b, 0.1, 1.0, 0.0)


@pytest.mark.parametrize("linear", [False, True])
def test_zero_scores_produce_finite_null_path(linear):
    X = np.asfortranarray(np.column_stack([np.tile([-1.0, 1.0], 50), np.ones(100)]))
    y = np.tile([0, 0, 1, 1], 25)
    fit = biglasso(X, y, nlambda=5, lambda_log_scale=not linear, family="binomial")
    np.testing.assert_array_equal(fit.lambdas, np.zeros(5))
    assert fit.coef.nnz == 0
    np.testing.assert_allclose(fit.deviance, 200 * np.log(2))


def test_unconverged_prefit_rejected(data):
    with pytest.raises(ValueError, match="Unpenalized prefit failed to converge"):
        biglasso(
            *data, penalty_factor=[0, 1, 1], max_iter=1, nlambda=2, family="binomial"
        )


@pytest.mark.parametrize("subset", [False, True])
def test_final_deviance_stable_for_extreme_misclassified_logit(subset):
    rng = np.random.default_rng(7)
    X = np.asfortranarray(rng.normal(size=(1000, 2)))
    y = rng.binomial(1, expit(3 * X[:, 0]))
    X[0, 0], y[0] = 20, 0
    rows = np.arange(0, 1000, 2) if subset else None
    fit = biglasso(X, y, lambdas=[0.01], eps=1e-12, row_indices=rows, family="binomial")
    eta = fit.decision_function(X, row_indices=rows)[:, 0]
    yy = y if rows is None else y[rows]
    reference = 2 * np.logaddexp(0, np.where(yy == 1, -eta, eta)).sum()
    assert eta[0] > 10
    assert np.isfinite(fit.deviance).all()
    np.testing.assert_allclose(fit.deviance, reference, rtol=1e-12)


def test_nonconverged_fit_reports_final_not_previous_loss(data):
    X, y = data
    with pytest.warns(UserWarning, match="Failed to converge"):
        fit = biglasso(X, y, lambdas=[0.01], max_iter=2, family="binomial")
    np.testing.assert_allclose(fit.deviance, fit.score(X, y), rtol=1e-12)


# (nrow(X), ncol(X), rows fit): R's rule, nrow(X) > ncol(X), and the rows
# actually fit choose different defaults in the first two
@pytest.mark.parametrize(
    "shape,rows,ratio",
    [
        ((120, 60), np.arange(0, 120, 3), 0.05),  # tall X, 40 rows fit
        ((50, 60), np.r_[np.arange(50), np.arange(25)], 0.001),  # repeats: 75
        ((120, 60), np.arange(60), 0.05),  # exactly ncol rows: not more
        ((120, 60), np.arange(61), 0.001),
    ],
    ids=["tall-subset", "wide-repeats", "rows-equal-cols", "rows-just-over"],
)
def test_default_lambda_min_counts_the_rows_fit(shape, rows, ratio):
    # Differs from R biglasso, which compares nrow(X) with ncol(X)
    rng = np.random.default_rng(11)
    X = np.asfortranarray(rng.normal(size=shape))
    y = rng.binomial(1, expit(X[:, :3].sum(axis=1))).astype(np.float64)
    # the whole grid, past saturation, so its last lambda is the end
    options = dict(row_indices=rows, nlambda=5, stop_on_saturation=False)
    auto = biglasso(X, y, family="binomial", **options)
    explicit = biglasso(X, y, lambda_min=ratio, family="binomial", **options)
    # log and exp of the grid's ends, and a division: a few roundings
    np.testing.assert_allclose(
        auto.lambdas[-1] / auto.lambdas[0], ratio, rtol=16 * np.finfo(float).eps
    )
    np.testing.assert_array_equal(auto.lambdas, explicit.lambdas)
    np.testing.assert_array_equal(auto.coef.toarray(), explicit.coef.toarray())
