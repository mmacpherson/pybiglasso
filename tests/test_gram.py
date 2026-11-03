"""Covariance updates checked against analytic optima and independent KKT."""

import numpy as np
import pyarrow as pa
import pytest
from sklearn.datasets import load_breast_cancer
from support.certificates import (
    KKT_TOL,
    gaussian_curvature,
    gaussian_subgradient_residual,
    standardize,
    standardized_theta,
    with_ones,
)
from support.paths import assert_bitwise

import pybiglasso as bl
from pybiglasso.arrow import _open_mapped
from pybiglasso.sklearn import (
    BiglassoClassifier,
    BiglassoClassifierCV,
    BiglassoRegressor,
    BiglassoRegressorCV,
)

EPS = 1e-16
DTYPES = [
    "bool",
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "float32",
    "float64",
]


def certify_pair(X, y, *, rows=None, alpha=0.5, penalty_factor=None, **kwargs):
    """Both fits obey KKT; curvature bounds their objective difference."""
    selected = np.arange(len(y)) if rows is None else rows
    values = np.asarray(X)[selected].astype(float)
    response = y[selected]
    Zs, center, scale = standardize(values)
    Z = with_ones(Zs)
    pf = np.ones(X.shape[1]) if penalty_factor is None else np.asarray(penalty_factor)
    fits = [
        bl.biglasso(
            X,
            y,
            update_from=mode,
            row_indices=rows,
            alpha=alpha,
            penalty_factor=pf,
            eps=EPS,
            max_iter=100_000,
            **kwargs,
        )
        for mode in ("gram", "residuals")
    ]
    objectives, errors = [], []
    for fit in fits:
        assert len(fit) == len(fits[0])
        theta = np.column_stack(
            [
                standardized_theta(
                    fit.intercept[k], fit.coef[:, [k]].toarray().ravel(), center, scale
                )
                for k in range(len(fit))
            ]
        )
        error = gaussian_subgradient_residual(
            Z, response, theta, fit.lambdas, alpha, pf
        )
        assert np.max(np.abs(error)) < KKT_TOL
        residual = response[:, None] - Z @ theta
        objective = np.mean(residual**2, axis=0) / 2 + fit.lambdas * np.sum(
            pf[:, None] * (alpha * abs(theta[1:]) + (1 - alpha) * theta[1:] ** 2 / 2),
            axis=0,
        )
        objectives.append(objective)
        errors.append(np.sum(error**2, axis=0))
    for k, lam in enumerate(fits[0].lambdas):
        mu = gaussian_curvature(Z, lam, alpha, pf)
        # Strong convexity bounds each fit's objective gap by ||g||²/(2 mu).
        # The independent BLAS objective evaluation gets a fixed rounding term.
        bound = (errors[0][k] + errors[1][k]) / (2 * mu) + 1e-12 * max(
            1, np.var(response)
        )
        assert abs(objectives[0][k] - objectives[1][k]) <= bound
    return fits


@pytest.mark.parametrize("alpha", [1.0, 0.5])
def test_gram_orthogonal_closed_form_and_support(alpha):
    X = np.asfortranarray(
        [[-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]
    )
    signal = np.array([2.0, -1.0, 0.25])
    y = 3 + X @ signal
    pf = np.array([0.0, 1.0, 2.0])
    lambdas = np.array([1.0, 0.2, 0.05, 0.0])
    expected = (
        np.sign(signal[:, None])
        * np.maximum(abs(signal[:, None]) - lambdas * alpha * pf[:, None], 0)
        / (1 + lambdas * (1 - alpha) * pf[:, None])
    )
    for mode in ["gram", "residuals"]:
        fit = bl.biglasso(
            X,
            y,
            alpha=alpha,
            lambdas=lambdas,
            penalty_factor=pf,
            eps=EPS,
            update_from=mode,
        )
        np.testing.assert_allclose(fit.coef.toarray(), expected, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(fit.coef.toarray() != 0, expected != 0)
        np.testing.assert_allclose(fit.intercept, 3, rtol=0, atol=1e-12)
        assert fit.update_from == mode


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("alpha", [1.0, 0.4])
def test_gram_all_dtypes_certified(dtype, alpha):
    rng = np.random.default_rng(781)
    integers = rng.integers(0, 2 if dtype == "bool" else 8, (97, 7))
    X = np.asfortranarray(integers.astype(dtype))
    y = (
        1
        + integers @ np.array([1.0, -0.5, 0.0, 0.25, 0.0, 0.0, 0.0])
        + rng.normal(size=97) / 8
    )
    certified, _ = certify_pair(
        X,
        y,
        alpha=alpha,
        penalty_factor=[0.0, 0.5, 1.0, 2.0, 1.0, 1.0, 1.0],
        lambdas=[0.5, 0.1, 0.01],
    )
    arrays = [pa.array(X[:, j]) for j in range(X.shape[1])]
    names = [f"x{j}" for j in range(X.shape[1])]
    record_batch = pa.record_batch(arrays, names=names)
    chunked = pa.Table.from_arrays(
        [pa.chunked_array([a[: j + 1], a[j + 1 :]]) for j, a in enumerate(arrays)],
        names=names,
    )
    for arrow in [record_batch, chunked]:
        actual = bl.biglasso(
            arrow,
            y,
            alpha=alpha,
            update_from="gram",
            eps=EPS,
            penalty_factor=[0.0, 0.5, 1.0, 2.0, 1.0, 1.0, 1.0],
            lambdas=[0.5, 0.1, 0.01],
        )
        assert_bitwise(actual, certified)


@pytest.mark.parametrize(
    "storage", ["strided", "mapped_numpy", "arrow", "mapped_arrow"]
)
@pytest.mark.parametrize("selection", ["all", "mask", "gather", "repeated", "fallback"])
def test_gram_storage_rows_and_thread_invariance(tmp_path, storage, selection):
    rng = np.random.default_rng(521)
    n = 2057
    raw = rng.integers(0, 2, (n, 6))
    y = 1 + raw[:, 0] - raw[:, 1] / 2 + rng.normal(size=n) / 8
    values = raw.astype(float)
    rows = {
        "all": None,
        "mask": np.arange(7, n),
        "gather": np.arange(3, n, 4),
        "repeated": np.r_[np.arange(3, n, 4), 8, 8],
        "fallback": np.arange(7, n),
    }[selection]
    if selection == "fallback":
        values[0, 0] = np.nan
    if storage == "strided":
        X = np.array(values, order="C")
    elif storage == "mapped_numpy":
        np.save(tmp_path / "X.npy", np.asfortranarray(values))
        X = np.load(tmp_path / "X.npy", mmap_mode="r")
    else:
        # Bool slice starts mid-byte; odd independent chunks cross LANES.
        columns = []
        for j in range(values.shape[1]):
            a = pa.array(
                np.r_[0, values[:, j]]
                if j == 0
                else np.r_[False, raw[:, j].astype(bool)]
            ).slice(1)
            cut = 7 + 2 * j
            columns.append(
                pa.chunked_array([a[:cut], a[cut : 17 + cut], a[17 + cut :]])
            )
        X = pa.Table.from_arrays(columns, names=[f"x{j}" for j in range(6)])
        if storage == "mapped_arrow":
            with pa.ipc.new_file(tmp_path / "X.feather", X.schema) as writer:
                writer.write_table(X, max_chunksize=101)
            X = _open_mapped(tmp_path / "X.feather")
    fits = [
        bl.biglasso(
            X,
            y,
            update_from="gram",
            alpha=0.5,
            lambdas=[0.2, 0.05, 0.01],
            row_indices=rows,
            eps=EPS,
            n_threads=t,
        )
        for t in [1, 2, 4]
    ]
    for other in fits[1:]:
        assert_bitwise(fits[0], other)
    independent, _ = certify_pair(
        np.asfortranarray(values), y, rows=rows, lambdas=[0.2, 0.05, 0.01]
    )
    assert_bitwise(fits[0], independent)


@pytest.mark.parametrize("alpha", [1.0, 0.5])
def test_gram_near_collinear_breast_cancer(alpha):
    X, y = load_breast_cancer(return_X_y=True)
    X = np.asfortranarray(X)
    Xs, _, _ = standardize(X)
    assert np.corrcoef(Xs.T)[0, 3] > 0.98
    assert 0 < np.linalg.eigvalsh(Xs.T @ Xs / len(Xs)).min() < 1e-3
    certify_pair(X, y.astype(float), alpha=alpha, lambdas=[0.05, 0.01, 0.001])


def test_gram_large_offset_and_automatic_unpenalized_prefit():
    rng = np.random.default_rng(5)
    X = np.asfortranarray(1e9 + rng.integers(0, 32, (101, 5)) / 8)
    y = 1 + (X[:, 0] - 1e9) - (X[:, 1] - 1e9) / 2
    certify_pair(X, y, penalty_factor=[0.0, 1.0, 1.0, 1.0, 1.0], nlambda=7)


@pytest.mark.parametrize("bad", ["auto", "naive", 1])
def test_update_from_invalid(bad):
    with pytest.raises(ValueError, match="update_from"):
        bl.biglasso(np.ones((4, 2)), np.arange(4.0), update_from=bad)


@pytest.mark.parametrize("kwargs", [dict(family="binomial"), dict(r_compatible=True)])
def test_update_from_gram_incompatible(kwargs):
    with pytest.raises(ValueError, match="update_from='residuals'"):
        bl.biglasso(
            np.ones((4, 2)),
            np.array([0.0, 1.0, 0.0, 1.0]),
            update_from="gram",
            **kwargs,
        )


def test_default_and_explicit_residuals_unchanged():
    X = np.asfortranarray(np.random.default_rng(2).normal(size=(31, 3)))
    y = np.arange(31.0) % 5
    default = bl.biglasso(X, y, nlambda=5)
    assert default.update_from == "residuals"
    assert_bitwise(default, bl.biglasso(X, y, nlambda=5, update_from="residuals"))


def test_estimators_and_cv_pass_update_mode():
    X = np.asfortranarray(np.random.default_rng(2).normal(size=(61, 5)))
    y = 2 + X[:, 0] + np.random.default_rng(3).normal(size=61) / 8
    for estimator in [
        BiglassoRegressor(update_from="gram"),
        BiglassoRegressorCV(update_from="gram", n_alphas=5, cv=3),
    ]:
        estimator.fit(X, y)
        assert estimator.path_.update_from == "gram"
    for estimator in [
        BiglassoClassifier(update_from="gram"),
        BiglassoClassifierCV(update_from="gram", n_alphas=5, cv=3),
    ]:
        with pytest.raises(ValueError, match="update_from='residuals'"):
            estimator.fit(X, (y > 2).astype(int))


@pytest.mark.parametrize("response", ["constant", "unpenalized", "limited", "dfmax"])
def test_gram_path_boundaries(response):
    rng = np.random.default_rng(77)
    X = np.asfortranarray(rng.normal(size=(71, 5)))
    y = np.ones(71) if response == "constant" else 2 + X @ np.arange(1.0, 6.0)
    kwargs = dict(nlambda=5, eps=EPS, update_from="gram")
    if response == "unpenalized":
        y = 2 + X[:, :4] @ np.arange(1.0, 5.0)
        kwargs.update(lambdas=[1.0, 0.1], penalty_factor=[0.0, 0.0, 0.0, 0.0, 1.0])
    elif response == "limited":
        kwargs.update(lambdas=[0.01], max_iter=1)
    elif response == "dfmax":
        kwargs.update(dfmax=1)
    if response in ("limited", "dfmax"):
        with pytest.warns(UserWarning, match="max_iter|dfmax"):
            fit = bl.biglasso(X, y, **kwargs)
    else:
        fit = bl.biglasso(X, y, **kwargs)
    prediction = fit.predict(X)
    residual = y[:, None] - prediction
    # RSS reconstructed on standardized X and public prediction on original X
    # differ in arithmetic order. Bound that roundoff, even at zero RSS:
    # eight operations per feature plus eight for the intercept/response,
    # then 2 ||r|| ||dr|| + ||dr||² and the length-n RSS reduction error.
    beta = abs(fit.coef.toarray())
    unit = np.finfo(float).eps
    operations = 8 * X.shape[1] + 8
    gamma = operations * unit / (1 - operations * unit)
    delta = np.linalg.norm(
        gamma
        * (
            abs(y[:, None])
            + abs(fit.intercept)
            + abs(X) @ beta
            + abs(fit.center) @ beta
        ),
        axis=0,
    )
    rss = np.sum(residual**2, axis=0)
    bound = (
        2 * np.linalg.norm(residual, axis=0) * delta + delta**2 + len(y) * unit * rss
    )
    assert np.all(abs(fit.deviance - rss) <= bound)
    if response == "constant":
        assert not fit.coef.nnz
        np.testing.assert_array_equal(prediction, 1)
    if response == "unpenalized":
        np.testing.assert_allclose(
            fit.coef.toarray(), np.tile([1.0, 2.0, 3.0, 4.0, 0.0], (2, 1)).T, atol=1e-7
        )


def test_gram_callback_and_explicit_residual_compatibility():
    X = np.asfortranarray(np.random.default_rng(9).normal(size=(71, 5)))
    y = X[:, 0] + X[:, 1] / 2
    seen = []

    def stop(il, lambdas, cols, beta, intercept):
        seen.append(il)
        return il == 2

    fit = bl.biglasso(X, y, nlambda=7, update_from="gram", _stop=stop)
    assert seen == [0, 1, 2] and len(fit) == 3
    assert_bitwise(
        bl.biglasso(X, y, r_compatible=True, nlambda=5),
        bl.biglasso(X, y, r_compatible=True, nlambda=5, update_from="residuals"),
    )


@pytest.mark.parametrize("checked", [False, True])
@pytest.mark.parametrize("row_path", ["mask", "gather"])
@pytest.mark.parametrize("fallback", [False, True])
def test_gram_native_row_dispatch_and_bounds(checked, row_path, fallback):
    from pybiglasso import _rust
    from pybiglasso.arrow import _ArrowMatrix

    X = np.asfortranarray(np.random.default_rng(7).normal(size=(107, 4)))
    if fallback:
        X[0, 0] = np.nan
    table = pa.Table.from_arrays(
        [
            pa.chunked_array([pa.array(X[: 7 + j, j]), pa.array(X[7 + j :, j])])
            for j in range(4)
        ],
        names=[f"x{j}" for j in range(4)],
    )
    arguments = dict(
        y=np.arange(107.0) % 5,
        row_indices=np.arange(7, 107, dtype=np.int64),
        alpha=0.5,
        lambdas=np.array([0.2, 0.05]),
        nlambda=2,
        eps=EPS,
        lambda_min=0.2,
        lambda_log_scale=True,
        penalty_factor=np.ones(4),
        dfmax=5,
        max_iter=10_000,
        stop_on_saturation=False,
        r_compatible=False,
        n_threads=2,
        boundscheck=checked,
        family="gaussian",
        update_from="gram",
        row_path=row_path,
    )
    a = _rust.fit_path(x=X, **arguments)
    b = _rust.fit_path(x=(107, _ArrowMatrix(table)._columns), **arguments)
    assert a["row_path"] == b["row_path"] == ("gather" if fallback else row_path)
    for field in [
        "center",
        "scale",
        "lambda_",
        "deviance",
        "iters",
        "n_reject",
        "intercept",
    ]:
        assert a[field].tobytes() == b[field].tobytes(), field
    for a_col, b_col in zip(a["coef_vals"], b["coef_vals"], strict=True):
        assert a_col.tobytes() == b_col.tobytes()


@pytest.mark.parametrize("dtype", ["int64", "uint64"])
def test_gram_wide_integer_conversion(dtype):
    X = np.asfortranarray(
        ((1 << 53) + 1 + np.arange(164).reshape(41, 4) * 3).astype(dtype)
    )
    y = np.arange(41.0) % 7
    options = dict(update_from="gram", lambdas=[0.2, 0.05], eps=EPS)
    assert_bitwise(
        bl.biglasso(X, y, **options), bl.biglasso(X.astype(float), y, **options)
    )
