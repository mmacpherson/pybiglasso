"""Production Gaussian solver: outside R trajectories and independent optima."""

import dataclasses

import numpy as np
import pytest
from support import rcompare, reference
from support.certificates import KKT_TOL, gaussian_subgradient_residual, with_ones
from support.paths import assert_bitwise

import pybiglasso as bl


def _fit(*args, **kwargs):
    # Gaussian remains Rust only, including CI jobs selecting numba globally.
    kwargs.setdefault("backend", "rust")
    return bl.biglasso(*args, **kwargs)


@pytest.fixture(scope="module")
def fixtures():
    return rcompare.load_fixtures()


@pytest.mark.parametrize("compatible", [False, True])
@pytest.mark.parametrize("log_scale", [False, True])
def test_gaussian_unrepresentable_automatic_grid_rejected(compatible, log_scale):
    X = np.asfortranarray([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    y = 1 + X[:, 0]
    options = dict(
        family="gaussian",
        alpha=np.nextafter(0.0, 1.0),
        r_compatible=compatible,
        lambda_log_scale=log_scale,
        eps=1e-14,
    )
    with pytest.raises(ValueError, match="automatic.*finite"):
        _fit(X, y, nlambda=3, **options)
    # The unrepresentable automatic maximum must not invalidate supplied grids.
    fit = _fit(X, y, lambdas=[1.0, 0.5], **options)
    np.testing.assert_array_equal(fit.lambdas, [1.0, 0.5])
    np.testing.assert_allclose(fit.coef.toarray(), [[0.5, 2 / 3], [0, 0]])
    np.testing.assert_array_equal(fit.intercept, 1)
    assert np.isfinite(fit.deviance).all()


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("case", rcompare.gaussian_grid(), ids=lambda c: c.name)
def test_gaussian_matches_stored_r(case, fixtures, tmp_path):
    digest, lambdas, expected = fixtures[case.name]
    got, new_digest, _ = rcompare.prepare_case(case, tmp_path, lambdas)
    assert digest == new_digest
    assert not (problems := rcompare.compare(got, expected)), "\n".join(problems)


@pytest.fixture(scope="module")
def live_gaussian(tmp_path_factory):
    root = tmp_path_factory.mktemp("gaussian-production-r")
    cases = rcompare.gaussian_grid()
    dirs = [root / case.name for case in cases]
    prepared = [
        rcompare.prepare_case(case, d)[0] for case, d in zip(cases, dirs, strict=True)
    ]
    outputs = rcompare.run_r(dirs)
    return {
        case.name: (got, want)
        for case, got, want in zip(cases, prepared, outputs, strict=True)
    }


@pytest.mark.r
@pytest.mark.skipif(not rcompare.r_available(), reason="R not available")
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("case", rcompare.gaussian_grid(), ids=lambda c: c.name)
def test_gaussian_matches_live_r(case, live_gaussian):
    got, expected = live_gaussian[case.name]
    assert not (problems := rcompare.compare(got, expected)), "\n".join(problems)


@pytest.mark.parametrize("penalty", ["none", "varied", "zeros"])
def test_gaussian_default_matches_reference_and_kkt(penalty, tmp_path):
    case = rcompare.Case(100, 10, family="gaussian", alpha=0.5, penalty=penalty)
    data = rcompare.make_case_data(case, tmp_path)
    options = dict(
        family="gaussian",
        alpha=0.5,
        penalty_factor=data["penalty_factor"],
        nlambda=5,
        eps=1e-18,
        max_iter=10000,
    )
    fit = _fit(data["X"], data["y"], **options)
    expected = reference.fit_path(data["X"], data["y"], **options)
    got = dict(
        center=fit.center,
        scale=fit.scale,
        lambda_=fit.lambdas,
        loss=fit.deviance,
        iter=fit.iterations,
        rejections=fit.n_rejected,
        beta=np.vstack([fit.intercept, fit.coef.toarray()]),
    )
    if penalty == "zeros":
        # The default prefit puts unpenalized scores exactly on the strong
        # rule's zero boundary in exact arithmetic. NumPy and fixed-order
        # Rust sums need not round those scores to zero together. Compare
        # optima here; the separate R cases still pin discrete trajectories.
        for key in ("center", "scale", "lambda_", "loss", "beta"):
            np.testing.assert_allclose(
                got[key], expected[key], rtol=rcompare.RTOL, atol=rcompare.ATOL
            )
    else:
        assert not (problems := rcompare.compare(got, expected)), "\n".join(problems)
    Xs = (data["X"] - fit.center) / fit.scale
    beta = fit.coef.toarray()
    theta = np.vstack([fit.intercept + fit.center @ beta, fit.scale[:, None] * beta])
    pf = 1 if data["penalty_factor"] is None else data["penalty_factor"]
    assert (
        np.max(
            np.abs(
                gaussian_subgradient_residual(
                    with_ones(Xs), data["y"], theta, fit.lambdas, fit.alpha, pf
                )
            )
        )
        <= KKT_TOL
    )


@pytest.mark.parametrize("rows", [None, [0, 2, 4, 6], [0, 0, 2, 4, 6], [6, 4]])
def test_gaussian_threads_rows_and_memmap(rows, tmp_path):
    X = np.memmap(
        tmp_path / "X.bin", mode="w+", dtype="float64", shape=(8, 2), order="F"
    )
    X[:] = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]] * 2)
    X.flush()
    X = np.memmap(
        tmp_path / "X.bin", mode="r", dtype="float64", shape=(8, 2), order="F"
    )
    y = 3 + 2 * X[:, 0] + X[:, 1]
    options = dict(
        family="gaussian", row_indices=rows, lambdas=[1, 0.5, 0.1], alpha=0.5, eps=1e-14
    )
    a = _fit(X, y, n_threads=1, **options)
    b = _fit(X, y, n_threads=4, **options)
    assert_bitwise(a, b)
    used = np.arange(len(y)) if rows is None else np.asarray(rows)
    c = _fit(np.asfortranarray(X[used]), y[used], **dict(options, row_indices=None))
    np.testing.assert_allclose(
        a.predict(X)[used], c.predict(X[used]), rtol=1e-9, atol=1e-11
    )


def test_gaussian_prediction_scoring_and_constant_response():
    X = np.asfortranarray([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    y = 3 + 2 * X[:, 0]
    fit = _fit(X, y, family="gaussian", lambdas=[0.5], alpha=0.5, eps=1e-14)
    assert fit.family == "gaussian"
    np.testing.assert_allclose(fit.predict(X)[:, 0], 3 + 1.4 * X[:, 0])
    np.testing.assert_allclose(fit.deviance, [4 * 0.6**2])
    np.testing.assert_allclose(fit.score(X, y), fit.deviance)
    np.testing.assert_allclose(fit.score(X, y, "mse"), [0.6**2])
    np.testing.assert_allclose(fit.score(X, y, "r2"), [1 - 0.6**2 / 4])
    for metric in ["auc", "accuracy"]:
        with pytest.raises(ValueError):
            fit.score(X, y, metric)
    with pytest.raises(ValueError, match="binomial"):
        fit.predict_proba(X)
    constant = _fit(X, np.full(4, 3.0), family="gaussian", nlambda=3)
    np.testing.assert_array_equal(constant.lambdas, 0)
    np.testing.assert_array_equal(constant.predict(X), 3)
    np.testing.assert_array_equal(constant.deviance, 0)
    np.testing.assert_array_equal(constant.score(X, np.full(4, 3.0), "r2"), 1)
    with pytest.warns(UserWarning, match="max_iter"):
        compatible = _fit(
            X,
            np.full(4, 3.0),
            family="gaussian",
            nlambda=3,
            r_compatible=True,
            max_iter=4,
        )
    np.testing.assert_array_equal(compatible.predict(X), constant.predict(X))
    np.testing.assert_array_equal(compatible.iterations[1:], 4)
    wrong = dataclasses.replace(constant, intercept=constant.intercept + 1)
    np.testing.assert_array_equal(wrong.score(X, np.full(4, 3.0), "r2"), 0)


def test_gaussian_reports_actual_rss_after_iteration_limit():
    rng = np.random.default_rng(4)
    X = np.asfortranarray(rng.normal(size=(30, 50)))
    y = rng.normal(size=30)
    with pytest.warns(UserWarning, match="max_iter"):
        fit = _fit(X, y, family="gaussian", lambdas=[0.1, 0.05], max_iter=1)
    np.testing.assert_allclose(
        fit.deviance, np.square(fit.predict(X) - y[:, None]).sum(axis=0)
    )


def test_gaussian_validates_family_backend_and_selected_y():
    X = np.asfortranarray([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    with pytest.raises(NotImplementedError, match="rust"):
        _fit(X, np.arange(4), family="gaussian", backend="numba")
    with pytest.raises(ValueError, match="finite"):
        _fit(X, [0, 1, 2, np.nan], family="gaussian")
    fit = _fit(
        X, [0, 1, 2, np.nan], family="gaussian", row_indices=[0, 1, 2], nlambda=3
    )
    assert np.isfinite(fit.predict(X)).all()


def test_gaussian_regressor_sklearn_contract():
    from sklearn.base import clone, is_regressor

    X = np.asfortranarray([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    y = 3 + 2 * X[:, 0]
    model = bl.BiglassoRegressor(alpha=0.5, l1_ratio=0.5, tol=1e-14, backend="rust")
    assert is_regressor(model)
    model = clone(model).fit(X, y)
    assert model.coef_.shape == (2,)
    assert isinstance(model.intercept_, float)
    np.testing.assert_allclose(model.coef_, [1.4, 0])
    np.testing.assert_allclose(model.predict(X), 3 + 1.4 * X[:, 0])
    assert model.score(X, y) == pytest.approx(1 - 0.6**2 / 4)


@pytest.mark.parametrize("rule", ["min", "1se"])
def test_gaussian_regressor_cv_has_analytic_fold_scores(rule):
    X = np.asfortranarray(
        np.tile([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]], (3, 1))
    )
    y = 3 + 2 * X[:, 0]
    alphas = np.array([1.0, 0.5, 0.1])

    def fit(jobs):
        return bl.BiglassoRegressorCV(
            alphas=alphas,
            l1_ratio=0.5,
            cv=3,
            rule=rule,
            n_jobs=jobs,
            tol=1e-14,
            backend="rust",
        ).fit(X, y)

    a, b = fit(1), fit(2)
    # Each training and held-out fold is the same balanced orthogonal design.
    coefficient = (2 - alphas * 0.5) / (1 + alphas * 0.5)
    expected = np.tile((2 - coefficient) ** 2, (3, 1))
    np.testing.assert_allclose(a.cv_scores_, expected)
    np.testing.assert_array_equal(a.cv_scores_, b.cv_scores_)
    assert a.alpha_ == 0.1
    np.testing.assert_allclose(a.coef_, [coefficient[-1], 0])
