import fractions
import functools
import operator
import warnings

import numba
import numpy as np
import pytest
import scipy.sparse as sps

import pybiglasso as bl


def saga(sklm, C, l1_ratio):
    """sklearn's saga at the given elastic-net mix, tightly converged.

    Before scikit-learn 1.8, l1_ratio is ignored unless penalty="elasticnet";
    from 1.8 the penalty argument is deprecated and l1_ratio sets the mix.
    """
    import sklearn

    major, minor = (int(v) for v in sklearn.__version__.split(".")[:2])
    extra = {} if (major, minor) >= (1, 8) else {"penalty": "elasticnet"}
    return sklm.LogisticRegression(
        C=C, l1_ratio=l1_ratio, solver="saga", tol=1e-12, max_iter=200_000, **extra
    )


@pytest.fixture(scope="module")
def data():
    d = bl.common.generate_test_data(300, 12, seed=3)
    return d["X"], d["y"]


@pytest.fixture(scope="module")
def path(data):
    X, y = data
    return bl.biglasso(X, y, alpha=1.0, nlambda=15)


# the path object ------------------------------------------------------


def test_path_shapes(path, data):
    X, _ = data
    L = len(path)
    assert L == 15
    assert isinstance(path.coef, sps.csc_array)
    assert path.coef.shape == (X.shape[1], L)
    for arr in (path.lambdas, path.intercept, path.deviance, path.iterations):
        assert arr.shape == (L,)
    assert np.all(np.diff(path.lambdas) < 0)
    assert path.n_nonzero[0] == 0 and path.n_nonzero[-1] > 0
    assert path.feature_names == tuple(f"x{j}" for j in range(X.shape[1]))


def test_feature_names_from_dataframe(data):
    pd = pytest.importorskip("pandas")
    X, y = data
    df = pd.DataFrame(X, columns=[f"snp{j}" for j in range(X.shape[1])])
    p = bl.biglasso(df, y, nlambda=3)
    assert p.feature_names[0] == "snp0"


def test_lasso_matches_independent_solver(data):
    """At alpha=1 the objective is unambiguous; compare against sklearn's saga."""
    sklm = pytest.importorskip("sklearn.linear_model")
    X, y = data
    n, sd = len(y), X.std(axis=0)
    p = bl.biglasso(X, y, alpha=1.0, nlambda=8, lambda_min=0.02, eps=1e-12)
    for k in (2, 4, 7):
        lam = p.lambdas[k]
        sk = saga(sklm, C=1 / (n * lam), l1_ratio=1.0).fit((X - X.mean(axis=0)) / sd, y)
        np.testing.assert_allclose(
            p.coef[:, [k]].toarray().ravel() * sd, sk.coef_.ravel(), atol=1e-6
        )


@pytest.mark.parametrize("alpha", [0.5, 0.2])
def test_elastic_net_matches_independent_solver(data, alpha):
    """The default elastic net minimizes the documented (glmnet) objective."""
    sklm = pytest.importorskip("sklearn.linear_model")
    X, y = data
    n, sd = len(y), X.std(axis=0)
    Xs = (X - X.mean(axis=0)) / sd
    p = bl.biglasso(X, y, alpha=alpha, nlambda=8, lambda_min=0.02, eps=1e-12)
    r = bl.biglasso(X, y, alpha=alpha, lambdas=p.lambdas, eps=1e-12, r_compatible=True)
    for k in (2, 4, 7):
        sk = saga(sklm, C=1 / (n * p.lambdas[k]), l1_ratio=alpha).fit(Xs, y)
        ours = p.coef[:, [k]].toarray().ravel() * sd
        np.testing.assert_allclose(ours, sk.coef_.ravel(), atol=1e-6)
        # R's update solves a different problem, measurably so
        r_coef = r.coef[:, [k]].toarray().ravel() * sd
        assert np.abs(r_coef - sk.coef_.ravel()).max() > 1e-3


# prediction and scoring -----------------------------------------------


def test_predict(path, data):
    X, _ = data
    eta = path.decision_function(X)
    manual = path.intercept + X @ path.coef.toarray()
    np.testing.assert_allclose(eta, manual)
    np.testing.assert_allclose(path.predict_proba(X), 1 / (1 + np.exp(-manual)))
    np.testing.assert_array_equal(path.predict(X), (manual > 0).astype(np.int8))


def test_predict_subsets(path, data):
    X, _ = data
    lams = path.lambdas[[3, 7]]
    np.testing.assert_allclose(
        path.decision_function(X, lambdas=lams), path.decision_function(X)[:, [3, 7]]
    )
    rows = np.arange(0, X.shape[0], 3)
    np.testing.assert_allclose(
        path.decision_function(X, row_indices=rows), path.decision_function(X)[rows]
    )
    with pytest.raises(ValueError, match="not on the fitted path"):
        path.decision_function(X, lambdas=[0.123456])
    with pytest.raises(ValueError, match="columns"):
        path.decision_function(X[:, :-1])


def test_score(path, data):
    metrics = pytest.importorskip("sklearn.metrics")
    X, y = data
    # default-mode deviance describes the returned model, after its last sweep
    np.testing.assert_allclose(path.score(X, y), path.deviance, rtol=1e-12)
    eta = path.decision_function(X)
    auc = [metrics.roc_auc_score(y, eta[:, k]) for k in range(1, len(path))]
    np.testing.assert_allclose(path.score(X, y, "auc")[1:], auc)
    acc = [metrics.accuracy_score(y, eta[:, k] > 0) for k in range(len(path))]
    np.testing.assert_allclose(path.score(X, y, "accuracy"), acc)
    with pytest.raises(ValueError, match="metric"):
        path.score(X, y, "r2")


def test_dataframe_exports(path):
    pl = pytest.importorskip("polars")
    coef = path.to_polars()
    assert coef.columns == ["lambda_", "feature", "coefficient"]
    assert coef.height == len(path) + path.coef.nnz
    assert coef.filter(pl.col("feature") == "(Intercept)").height == len(path)
    assert path.to_polars("path").height == len(path)
    assert path.to_polars("features").height == path.coef.shape[0]
    pytest.importorskip("pandas")
    assert path.to_pandas().shape == (coef.height, 3)


def test_plots(path):
    so = pytest.importorskip("seaborn.objects")
    for plot in (bl.plot_coefficients(path), bl.plot_deviance(path)):
        assert isinstance(plot, so.Plot)
        plot.plot()


# input handling -------------------------------------------------------


@pytest.mark.parametrize(
    "make_y, rows, match",
    [
        (lambda y: np.zeros_like(y), None, "both 0s and 1s"),
        (lambda y: np.ones_like(y), None, "both 0s and 1s"),
        (
            lambda y: np.where(np.arange(len(y)) < 50, 1.0, 0.0),
            np.arange(60, 100),
            "both 0s and 1s",
        ),
        (lambda y: 2 * y, None, "both 0s and 1s"),
    ],
    ids=["all-zero", "all-one", "one-class-in-row-subset", "not-0-1"],
)
def test_rejects_degenerate_response(data, make_y, rows, match):
    X, y = data
    with pytest.raises(ValueError, match=match):
        bl.biglasso(X, make_y(y), row_indices=rows)


def test_rejects_bad_inputs(data):
    X, y = data
    with pytest.raises(ValueError, match="Alpha"):
        bl.biglasso(X, y, alpha=1.5)
    with pytest.raises(ValueError, match="one entry per column"):
        bl.biglasso(X, y, penalty_factor=np.ones(3))
    with pytest.raises(ValueError, match="at least one positive"):
        bl.biglasso(X, y, penalty_factor=np.zeros(X.shape[1]))
    with pytest.raises(ValueError, match="decreasing"):
        bl.biglasso(X, y, lambdas=[0.01, 0.1])
    with pytest.raises(ValueError, match="outside valid range"):
        bl.biglasso(X, y, row_indices=np.array([0, X.shape[0]]))
    with pytest.warns(UserWarning, match="Fortran"):
        bl.biglasso(np.ascontiguousarray(X), y, nlambda=3)


def test_row_indices_match_subset(data):
    X, y = data
    rows = np.flip(np.arange(X.shape[0])[::2])
    a = bl.biglasso(np.asfortranarray(X[rows]), y[rows], nlambda=10)
    b = bl.biglasso(X, y, row_indices=rows, nlambda=10)
    np.testing.assert_allclose(a.coef.toarray(), b.coef.toarray())
    np.testing.assert_allclose(a.intercept, b.intercept)
    np.testing.assert_array_equal(a.iterations, b.iterations)


def test_memmap_matches_in_memory(data, tmp_path):
    X, y = data
    mm = np.lib.format.open_memmap(
        tmp_path / "X.npy", mode="w+", dtype=X.dtype, shape=X.shape, fortran_order=True
    )
    mm[:] = X
    mm.flush()
    a = bl.biglasso(X, y, nlambda=10)
    b = bl.biglasso(np.load(tmp_path / "X.npy", mmap_mode="r"), y, nlambda=10)
    np.testing.assert_array_equal(a.coef.toarray(), b.coef.toarray())


# the lambda path ------------------------------------------------------


@pytest.mark.parametrize("penalty", ["zeros", "varied"])
def test_default_grid_matches_same_grid_as_user_lambdas(data, penalty):
    """The default path's starting point must be a true solution at lambda_max."""
    X, y = data
    pf = np.ones(X.shape[1])
    if penalty == "zeros":
        pf[:2] = 0
    else:
        pf = np.random.default_rng(0).uniform(0.5, 2, X.shape[1])
    a = bl.biglasso(X, y, penalty_factor=pf, nlambda=10)
    b = bl.biglasso(X, y, penalty_factor=pf, lambdas=a.lambdas)
    np.testing.assert_allclose(a.coef.toarray(), b.coef.toarray(), atol=1e-5)
    np.testing.assert_allclose(a.intercept, b.intercept, atol=1e-5)


def test_unpenalized_columns_enter_at_lambda_max(data):
    X, y = data
    pf = np.ones(X.shape[1])
    pf[:2] = 0
    p = bl.biglasso(X, y, penalty_factor=pf, nlambda=10)
    first = p.coef[:, [0]].toarray().ravel()
    assert np.all(first[:2] != 0), "unpenalized columns should be fit at lambda_max"
    assert np.all(first[2:] == 0), "penalized columns should be zero at lambda_max"
    # and just below lambda_max, a penalized column enters
    assert p.n_nonzero[1] > 2


def test_dfmax_truncates_path(data):
    X, y = data
    with pytest.warns(UserWarning, match="dfmax"):
        p = bl.biglasso(X, y, dfmax=3, nlambda=30)
    assert len(p) < 30
    assert p.n_nonzero[:-1].max() <= 3


def test_saturation(data):
    X, _ = data
    y = (X[:, 0] > 0).astype(float)  # perfectly separable
    with pytest.warns(UserWarning, match="saturated"):
        p = bl.biglasso(X, y, nlambda=50, lambda_min=1e-4)
    assert len(p) < 50
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = bl.biglasso(X, y, nlambda=50, lambda_min=1e-4, stop_on_saturation=False)
    assert len(full) == 50


def test_float32_matches_float64(data):
    X, y = data
    X32 = np.asfortranarray(X.astype(np.float32))
    a = bl.biglasso(X32, y, nlambda=10)
    b = bl.biglasso(np.asfortranarray(X32.astype(np.float64)), y, nlambda=10)
    np.testing.assert_allclose(a.coef.toarray(), b.coef.toarray())


@pytest.mark.parametrize("penalty", [None, "zeros"])
def test_thread_count_does_not_change_results(penalty):
    # numba refuses more threads than the machine has (3 on CI's macOS)
    threads = min(4, numba.config.NUMBA_NUM_THREADS)
    if threads < 2:
        pytest.skip("needs at least two cores")
    d = bl.common.make_biglasso_dataset(nrow=150, ncol=400, seed=2)
    pf = None
    if penalty == "zeros":
        pf = np.ones(400)
        pf[:3] = 0
    one = bl.biglasso(d["X"], d["y"], alpha=0.7, penalty_factor=pf, n_threads=1)
    many = bl.biglasso(d["X"], d["y"], alpha=0.7, penalty_factor=pf, n_threads=threads)
    np.testing.assert_array_equal(one.coef.toarray(), many.coef.toarray())
    np.testing.assert_array_equal(one.iterations, many.iterations)
    np.testing.assert_array_equal(one.n_rejected, many.n_rejected)


def test_quickstart_example_runs():
    for mod in ("polars", "sklearn", "seaborn"):
        pytest.importorskip(mod)
    import pathlib
    import runpy

    import matplotlib

    matplotlib.use("Agg")
    example = pathlib.Path(__file__).parents[1] / "examples" / "quickstart.py"
    runpy.run_path(str(example), run_name="__main__")


def test_intercept_correction_does_not_depend_on_summation_order(monkeypatch):
    # The intercept on X's scale subtracts sum(center * beta). BLAS (@) picks
    # that sum's order by CPU and thread count; summed accurately, the order
    # can't matter. These terms cancel, so any plain order loses some of them.
    terms = np.concatenate([[1e16], np.ones(1000), [-1e16]])
    exact = float(sum(fractions.Fraction(t) for t in terms))
    left_to_right = functools.reduce(operator.add, terms.tolist())
    assert left_to_right != exact  # the input tells an ordered sum apart
    ncol = len(terms)

    def solver(X, y, row_indices, params, on_lambda=None):
        return dict(
            iters=np.array([1]),
            col_map=np.arange(ncol),
            center=terms,
            scale=np.ones(ncol),
            intercept=np.zeros(1),
            coef_rows=[np.arange(ncol)],
            coef_vals=[np.ones(ncol)],
            lambda_=np.array([0.1]),
            deviance=np.zeros(1),
            n_reject=np.zeros(1, dtype=np.int64),
        )

    # the correction runs on either backend's results
    monkeypatch.setattr(bl.binomial, "fit_path", solver)
    monkeypatch.setattr(
        bl.binomial,
        "fit_path_rust",
        lambda X, y, rows, params, threads, on_lambda: solver(X, y, rows, params),
    )
    X = np.zeros((4, ncol), order="F")
    fit = bl.biglasso(X, np.array([0, 1, 0, 1]), lambdas=[0.1])
    assert fit.intercept[0] == -exact
