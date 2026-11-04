import builtins
import fractions
import functools
import operator
import pickle
import warnings

import numpy as np
import pytest
import scipy.sparse as sps
from support.certificates import (
    KKT_TOL,
    assert_same_optimum,
    kkt_residual,
    path_theta,
    standardize,
    with_ones,
)
from support.paths import assert_bitwise

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
    return bl.biglasso(X, y, alpha=1.0, nlambda=15, family="binomial")


# the path object ------------------------------------------------------


@pytest.mark.parametrize("protocol", [0, pickle.HIGHEST_PROTOCOL])
def test_legacy_path_pickle_keeps_binomial_behavior(path, data, protocol):
    # Released paths predate the family field. Simulate their instance state
    # without changing the shared fixture, then round-trip that legacy state.
    legacy = pickle.loads(pickle.dumps(path, protocol=protocol))
    del legacy.__dict__["family"]
    restored = pickle.loads(pickle.dumps(legacy, protocol=protocol))
    assert restored.family == "binomial"
    X, y = data
    np.testing.assert_array_equal(restored.predict(X), path.predict(X))
    np.testing.assert_array_equal(restored.predict_proba(X), path.predict_proba(X))
    np.testing.assert_array_equal(restored.score(X, y, "auc"), path.score(X, y, "auc"))


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
    p = bl.biglasso(df, pd.Series(y), nlambda=3, family="binomial")
    assert p.feature_names[0] == "snp0"


@pytest.mark.parametrize("family", ["gaussian", "binomial"])
@pytest.mark.parametrize(
    ("dtypes", "common"),
    [
        (("bool", "int8", "float64"), np.float64),
        (("bool", "int8"), np.int8),
        (("bool", "uint16", "float32"), np.float32),
    ],
)
def test_pandas_frame_with_mixed_bool_and_numeric_columns(family, dtypes, common):
    # pandas makes an object array of such a frame; the fit and predictions
    # must be those of the same values as an array of their common type
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(7)
    n = 150
    columns = {
        f"{dtype}_{j}": (
            rng.random(n) < 0.4
            if dtype == "bool"
            else rng.integers(0, 5, n)
            if dtype in ("int8", "uint16")
            else rng.normal(size=n)
        ).astype(dtype)
        for j, dtype in enumerate(dtypes)
    }
    frame = pd.DataFrame(columns)
    values = np.asfortranarray(
        np.column_stack([v.astype(common) for v in columns.values()])
    )
    signal = values.astype(np.float64) @ np.linspace(1, -1, len(dtypes))
    y = signal + rng.normal(size=n)
    if family == "binomial":
        y = (y > np.median(y)).astype(np.float64)
    fit = bl.biglasso(frame, y, family=family, nlambda=5)
    reference = bl.biglasso(values, y, family=family, nlambda=5)
    assert_bitwise(fit, reference)
    assert fit.feature_names == tuple(columns)
    assert_bitwise(fit.decision_function(frame), reference.decision_function(values))


def test_predict_accepts_a_pandas_frame(data):
    pd = pytest.importorskip("pandas")
    X, y = data
    path = bl.biglasso(X, y, nlambda=4, family="binomial")
    frame = pd.DataFrame(X)
    assert_bitwise(path.decision_function(frame), path.decision_function(X))
    rows = np.array([3, 0, 7])
    assert_bitwise(
        path.decision_function(frame, row_indices=rows),
        path.decision_function(X, row_indices=rows),
    )


def assert_matches_saga(sklm, path, k, X, y, alpha):
    """saga at the path's k-th lambda: both pass KKT for that lambda and the
    alpha requested, and convexity bounds their distance."""
    Xs, center, scale = standardize(X)
    Z, lam = with_ones(Xs), path.lambdas[k]
    sk = saga(sklm, C=1 / (len(y) * lam), l1_ratio=alpha).fit(Xs, y)
    theirs = np.concatenate([sk.intercept_, sk.coef_.ravel()])
    ours = path_theta(path, k, center, scale)
    assert_same_optimum(Z, y, ours, theirs, lam, alpha)


def test_lasso_matches_independent_solver(data):
    """At alpha=1 the objective is unambiguous; compare against sklearn's saga."""
    sklm = pytest.importorskip("sklearn.linear_model")
    X, y = data
    p = bl.biglasso(
        X, y, alpha=1.0, nlambda=8, lambda_min=0.02, eps=1e-14, family="binomial"
    )
    for k in (2, 4, 7):
        assert_matches_saga(sklm, p, k, X, y, 1.0)


@pytest.mark.parametrize("alpha", [0.5, 0.2])
def test_elastic_net_matches_independent_solver(data, alpha):
    """The default elastic net minimizes the documented (glmnet) objective."""
    sklm = pytest.importorskip("sklearn.linear_model")
    X, y = data
    p = bl.biglasso(
        X, y, alpha=alpha, nlambda=8, lambda_min=0.02, eps=1e-14, family="binomial"
    )
    r = bl.biglasso(
        X,
        y,
        alpha=alpha,
        lambdas=p.lambdas,
        eps=1e-14,
        r_compatible=True,
        family="binomial",
    )
    for k in (2, 4, 7):
        assert_matches_saga(sklm, p, k, X, y, alpha)
        # R's update solves a different problem: its fit, as converged, fails
        # the documented objective's KKT conditions
        assert kkt_residual(r, X, y, k) > KKT_TOL


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


@pytest.mark.parametrize("package", ["pandas", "polars"])
def test_dataframe_export_missing_package_names_install(path, monkeypatch, package):
    original_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == package or name.startswith(package + "."):
            raise ImportError(f"{package} not installed", name=package)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match=f"pip install {package}") as error:
        getattr(path, f"to_{package}")()
    assert "pybiglasso[" not in str(error.value)


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
        bl.biglasso(X, make_y(y), row_indices=rows, family="binomial")


def test_rejects_bad_inputs(data):
    X, y = data
    with pytest.raises(ValueError, match="Alpha"):
        bl.biglasso(X, y, alpha=1.5, family="binomial")
    with pytest.raises(ValueError, match="one entry per column"):
        bl.biglasso(X, y, penalty_factor=np.ones(3), family="binomial")
    with pytest.raises(ValueError, match="at least one positive"):
        bl.biglasso(X, y, penalty_factor=np.zeros(X.shape[1]), family="binomial")
    with pytest.raises(ValueError, match="decreasing"):
        bl.biglasso(X, y, lambdas=[0.01, 0.1], family="binomial")
    with pytest.raises(ValueError, match="outside valid range"):
        bl.biglasso(X, y, row_indices=np.array([0, X.shape[0]]), family="binomial")
    with pytest.warns(UserWarning, match="Fortran"):
        bl.biglasso(np.ascontiguousarray(X), y, nlambda=3, family="binomial")


def test_row_indices_match_subset(data):
    X, y = data
    rows = np.flip(np.arange(X.shape[0])[::2])
    a = bl.biglasso(np.asfortranarray(X[rows]), y[rows], nlambda=10, family="binomial")
    b = bl.biglasso(X, y, row_indices=rows, nlambda=10, family="binomial")
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
    a = bl.biglasso(X, y, nlambda=10, family="binomial")
    b = bl.biglasso(
        np.load(tmp_path / "X.npy", mmap_mode="r"), y, nlambda=10, family="binomial"
    )
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
    a = bl.biglasso(X, y, penalty_factor=pf, nlambda=10, eps=1e-14, family="binomial")
    b = bl.biglasso(
        X, y, penalty_factor=pf, lambdas=a.lambdas, eps=1e-14, family="binomial"
    )
    np.testing.assert_array_equal(b.lambdas, a.lambdas)
    # two routes to each optimum (alpha 1, the default): both pass KKT at the
    # lambda b was given, and convexity bounds the gap
    Xs, center, scale = standardize(X)
    for k, lam in enumerate(a.lambdas):
        assert_same_optimum(
            with_ones(Xs),
            y,
            path_theta(a, k, center, scale),
            path_theta(b, k, center, scale),
            lam,
            1.0,
            pf,
        )


def test_unpenalized_columns_enter_at_lambda_max(data):
    X, y = data
    pf = np.ones(X.shape[1])
    pf[:2] = 0
    p = bl.biglasso(X, y, penalty_factor=pf, nlambda=10, family="binomial")
    first = p.coef[:, [0]].toarray().ravel()
    assert np.all(first[:2] != 0), "unpenalized columns should be fit at lambda_max"
    assert np.all(first[2:] == 0), "penalized columns should be zero at lambda_max"
    # and just below lambda_max, a penalized column enters
    assert p.n_nonzero[1] > 2


def test_dfmax_truncates_path(data):
    X, y = data
    with pytest.warns(UserWarning, match="dfmax"):
        p = bl.biglasso(X, y, dfmax=3, nlambda=30, family="binomial")
    assert len(p) < 30
    assert p.n_nonzero[:-1].max() <= 3


def test_saturation(data):
    X, _ = data
    y = (X[:, 0] > 0).astype(float)  # perfectly separable
    with pytest.warns(UserWarning, match="saturated"):
        p = bl.biglasso(X, y, nlambda=50, lambda_min=1e-4, family="binomial")
    assert len(p) < 50
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = bl.biglasso(
            X,
            y,
            nlambda=50,
            lambda_min=1e-4,
            stop_on_saturation=False,
            family="binomial",
        )
    assert len(full) == 50


def test_float32_matches_float64(data):
    X, y = data
    X32 = np.asfortranarray(X.astype(np.float32))
    a = bl.biglasso(X32, y, nlambda=10, family="binomial")
    b = bl.biglasso(
        np.asfortranarray(X32.astype(np.float64)), y, nlambda=10, family="binomial"
    )
    np.testing.assert_allclose(a.coef.toarray(), b.coef.toarray())


@pytest.mark.parametrize("penalty", [None, "zeros"])
def test_thread_count_does_not_change_results(penalty):
    threads = 4
    d = bl.common.make_biglasso_dataset(nrow=150, ncol=400, seed=2)
    pf = None
    if penalty == "zeros":
        pf = np.ones(400)
        pf[:3] = 0
    one = bl.biglasso(
        d["X"], d["y"], alpha=0.7, penalty_factor=pf, n_threads=1, family="binomial"
    )
    many = bl.biglasso(
        d["X"],
        d["y"],
        alpha=0.7,
        penalty_factor=pf,
        n_threads=threads,
        family="binomial",
    )
    assert_bitwise(one, many)


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

    monkeypatch.setattr(
        bl.fit,
        "fit_path_rust",
        lambda X, y, rows, params, threads, on_lambda, family: solver(
            X, y, rows, params
        ),
    )
    X = np.zeros((4, ncol), order="F")
    fit = bl.biglasso(X, np.array([0, 1, 0, 1]), lambdas=[0.1], family="binomial")
    assert fit.intercept[0] == -exact


@pytest.fixture
def named_fit():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(rng.normal(size=(120, 3)), columns=["a", "b", "c"])
    y = 2 * frame["a"] - frame["c"] + rng.normal(size=120)
    return frame, bl.biglasso(frame, y, nlambda=4)


def test_predict_matches_the_array_when_columns_match(named_fit):
    frame, path = named_fit
    assert_bitwise(path.predict(frame), path.predict(frame.to_numpy()))


def test_predict_rejects_reordered_columns(named_fit):
    frame, path = named_fit
    with pytest.raises(ValueError, match="different order"):
        path.predict(frame[["c", "b", "a"]])


def test_predict_rejects_reordered_arrow_columns(named_fit):
    pa = pytest.importorskip("pyarrow")
    frame, path = named_fit
    table = pa.Table.from_pandas(frame[["b", "a", "c"]], preserve_index=False)
    with pytest.raises(ValueError, match="different order"):
        path.predict(table)


def test_predict_rejects_other_column_names(named_fit):
    frame, path = named_fit
    with pytest.raises(ValueError, match="don't match"):
        path.predict(frame.set_axis(["a", "b", "z"], axis=1))


def test_predict_accepts_named_columns_on_an_unnamed_fit(named_fit):
    frame, _ = named_fit
    rng = np.random.default_rng(12)
    path = bl.biglasso(frame.to_numpy(), rng.normal(size=len(frame)), nlambda=4)
    assert not path.named_features
    assert_bitwise(path.predict(frame), path.predict(frame.to_numpy()))
    # even labels that look like the defaults, in any order
    xs = frame.set_axis(["x2", "x0", "x1"], axis=1)
    assert_bitwise(path.predict(xs), path.predict(frame.to_numpy()))


def test_predict_checks_explicit_default_looking_names(named_fit):
    frame, _ = named_fit
    rng = np.random.default_rng(13)
    y = rng.normal(size=len(frame))
    path = bl.biglasso(frame.to_numpy(), y, nlambda=4, feature_names=["x0", "x1", "x2"])
    assert path.named_features
    with pytest.raises(ValueError, match="don't match"):
        path.predict(frame)
    with pytest.raises(ValueError, match="different order"):
        path.predict(frame.set_axis(["x2", "x1", "x0"], axis=1))


@pytest.mark.parametrize(
    "labels", [["a", "a", "c"], [1, "1", "c"]], ids=["repeated", "collide"]
)
def test_predict_rejects_ambiguous_names(labels):
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(14)
    frame = pd.DataFrame(rng.normal(size=(60, 3)), columns=labels)
    path = bl.biglasso(frame, rng.normal(size=60), nlambda=4)
    with pytest.raises(ValueError, match="ambiguous"):
        path.predict(frame)
    assert_bitwise(
        path.predict(frame.to_numpy()), path.decision_function(frame.to_numpy())
    )


def test_reorder_message_remedies_work(named_fit):
    pa = pytest.importorskip("pyarrow")
    frame, path = named_fit
    shuffled = frame[["c", "a", "b"]]
    expected = path.predict(frame.to_numpy())
    assert_bitwise(path.predict(shuffled[list(path.feature_names)]), expected)
    table = pa.Table.from_pandas(shuffled, preserve_index=False)
    assert_bitwise(path.predict(table.select(list(path.feature_names))), expected)
