"""Regression tests for the numerical and input failures found before release."""

import warnings

import numpy as np
import pytest
from scipy.special import expit

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
    kwargs = dict(penalty_factor=pf, lambda_log_scale=not linear, eps=1e-12)
    with pytest.raises(ValueError, match=r"at least 2.*lambdas=\[value\]"):
        biglasso(X, y, nlambda=1, **kwargs)
    full = biglasso(X, y, nlambda=5, **kwargs)
    explicit = biglasso(X, y, nlambda=1, lambdas=full.lambdas[-1:], **kwargs)
    assert explicit.n_nonzero[0] > 0
    np.testing.assert_allclose(
        full.coef[:, [-1]].toarray(), explicit.coef.toarray(), atol=2e-6
    )
    np.testing.assert_allclose(full.intercept[-1:], explicit.intercept, atol=2e-6)


@pytest.mark.parametrize("r_compatible", [False, True])
@pytest.mark.parametrize("subset", [False, True])
def test_translation_preserves_solution_and_kkt(data, r_compatible, subset):
    X, y = data
    rows = np.arange(0, len(y), 2) if subset else None
    kwargs = dict(
        lambdas=[0.03, 0.01], eps=1e-12, row_indices=rows, r_compatible=r_compatible
    )
    shifted = np.asfortranarray(X + 1e8)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        base = biglasso(X, y, **kwargs)
        fit = biglasso(shifted, y, **kwargs)
    np.testing.assert_allclose(fit.coef.toarray(), base.coef.toarray(), atol=3e-7)
    np.testing.assert_allclose(
        fit.decision_function(shifted), base.decision_function(X), atol=3e-7
    )
    A, yy = (shifted, y) if rows is None else (shifted[rows], y[rows])
    eta = fit.intercept + A @ fit.coef.toarray()
    residual = expit(eta) - yy[:, None]
    Z = (A - A.mean(axis=0)) / A.std(axis=0)
    beta = fit.coef.toarray() * A.std(axis=0)[:, None]
    gradient = Z.T @ residual / len(yy)
    l1 = fit.lambdas[None, :]
    violation = np.where(
        beta != 0, abs(gradient + l1 * np.sign(beta)), np.maximum(0, abs(gradient) - l1)
    )
    assert violation.max() < 5e-7
    assert abs(residual.mean(axis=0)).max() < 5e-7


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
        biglasso(mm, y, row_indices=rows)


def test_unused_nonfinite_rows_do_not_invalidate_fit(data):
    X, y = data
    X[1, :] = np.nan
    rows = np.arange(0, len(y), 2)
    fit = biglasso(X, y, row_indices=rows, nlambda=3)
    assert np.isfinite(fit.coef.data).all()


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
        biglasso(*data, **kwargs)


@pytest.mark.parametrize("shape", [(100,), (100, 0), (0, 3), (100, 3, 1)])
def test_invalid_design_shape(data, shape):
    with pytest.raises(ValueError, match="two-dimensional"):
        biglasso(np.empty(shape), data[1])


def test_response_must_be_vector(data):
    X, y = data
    with pytest.raises(ValueError, match="one-dimensional"):
        biglasso(X, y[:, None])


@pytest.mark.parametrize("dtype", [object, np.complex128, np.float16])
def test_unsupported_design_dtype(data, dtype):
    X, y = data
    with pytest.raises(ValueError, match="dtype"):
        biglasso(X.astype(dtype), y)


def test_constant_design_has_clear_error(data):
    X, y = data
    with pytest.raises(ValueError, match="no varying features"):
        biglasso(np.ones_like(X), y)


def test_only_unpenalized_features_vary(data):
    X, y = data
    X[:, 2] = 1
    with pytest.raises(ValueError, match="varying penalized feature"):
        biglasso(X, y, penalty_factor=[0, 0, 1])
    explicit = biglasso(X, y, penalty_factor=[0, 0, 1], lambdas=[0.1, 0.01])
    np.testing.assert_allclose(
        explicit.coef.toarray()[:, 0], explicit.coef.toarray()[:, 1], atol=1e-4
    )


@pytest.mark.parametrize("linear", [False, True])
def test_zero_scores_produce_finite_null_path(linear):
    X = np.asfortranarray(np.column_stack([np.tile([-1.0, 1.0], 50), np.ones(100)]))
    y = np.tile([0, 0, 1, 1], 25)
    fit = biglasso(X, y, nlambda=5, lambda_log_scale=not linear)
    np.testing.assert_array_equal(fit.lambdas, np.zeros(5))
    assert fit.coef.nnz == 0
    np.testing.assert_allclose(fit.deviance, 200 * np.log(2))


def test_unconverged_prefit_rejected(data):
    with pytest.raises(ValueError, match="Unpenalized prefit failed to converge"):
        biglasso(*data, penalty_factor=[0, 1, 1], max_iter=1, nlambda=2)


@pytest.mark.parametrize("subset", [False, True])
def test_final_deviance_stable_for_extreme_misclassified_logit(subset):
    rng = np.random.default_rng(7)
    X = np.asfortranarray(rng.normal(size=(1000, 2)))
    y = rng.binomial(1, expit(3 * X[:, 0]))
    X[0, 0], y[0] = 20, 0
    rows = np.arange(0, 1000, 2) if subset else None
    fit = biglasso(X, y, lambdas=[0.01], eps=1e-12, row_indices=rows)
    eta = fit.decision_function(X, row_indices=rows)[:, 0]
    yy = y if rows is None else y[rows]
    reference = 2 * np.logaddexp(0, np.where(yy == 1, -eta, eta)).sum()
    assert eta[0] > 10
    assert np.isfinite(fit.deviance).all()
    np.testing.assert_allclose(fit.deviance, reference, rtol=1e-12)


def test_nonconverged_fit_reports_final_not_previous_loss(data):
    X, y = data
    with pytest.warns(UserWarning, match="Failed to converge"):
        fit = biglasso(X, y, lambdas=[0.01], max_iter=2)
    np.testing.assert_allclose(fit.deviance, fit.score(X, y), rtol=1e-12)
