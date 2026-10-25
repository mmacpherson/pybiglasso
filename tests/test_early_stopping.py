"""Early stopping: biglasso's private _stop hook, and BiglassoClassifierCV's
early_stopping, which stops each fold once its held-out score stops improving.

Each check is against the same fit without stopping: a stopped path is the
unstopped one cut short, so its prefix must be bit-identical."""

import warnings

import numpy as np
import pytest
from support.paths import assert_bitwise

pytest.importorskip("sklearn")

import pybiglasso as bl  # noqa: E402
import pybiglasso.sklearn as bl_sklearn  # noqa: E402
from pybiglasso.sklearn import BiglassoClassifierCV  # noqa: E402


@pytest.fixture(scope="module")
def data():
    d = bl.common.generate_test_data(400, 40, seed=7)
    return d["X"], d["y"]


GRIDS = dict(automatic={}, supplied=dict(lambdas=np.geomspace(0.1, 0.002, 15)))
ROWS = dict(all=None, most=np.arange(400)[np.arange(400) % 7 != 0], few=np.arange(150))


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("grid", GRIDS)
@pytest.mark.parametrize("at", [0, 1, 6, 14])
def test_stopped_path_is_a_prefix(data, grid, rows, at):
    X, y = data
    kwargs = dict(nlambda=15, row_indices=ROWS[rows], **GRIDS[grid])
    full = bl.biglasso(X, y, **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # stopping isn't a truncation to warn of
        short = bl.biglasso(X, y, _stop=lambda il, *_: il >= at, **kwargs)
    assert len(short) == at + 1
    assert_bitwise(short, full, length=len(short))


def test_stop_sees_each_fit(data):
    # on X's scale, as the returned path has them
    X, y = data
    seen = []

    def record(il, lambdas, cols, beta, intercept):
        seen.append((il, lambdas.copy(), cols.copy(), beta.copy(), intercept))
        return False

    path = bl.biglasso(X, y, nlambda=12, _stop=record)
    assert [s[0] for s in seen] == list(range(12))
    for il, lambdas, cols, beta, intercept in seen:
        np.testing.assert_array_equal(lambdas, path.lambdas)
        column = path.coef[:, [il]]
        np.testing.assert_array_equal(cols, column.indices)
        np.testing.assert_array_equal(beta, column.data)
        assert intercept == path.intercept[il]


def test_stop_errors_propagate(data):
    X, y = data

    class Boom(Exception):
        pass

    def boom(il, *_):
        if il == 3:
            raise Boom
        return False

    with pytest.raises(Boom):
        bl.biglasso(X, y, nlambda=10, _stop=boom)


def test_cv_with_patience_past_the_path_matches_none(data):
    X, y = data
    plain = BiglassoClassifierCV(n_alphas=25, cv=4).fit(X, y)
    early = BiglassoClassifierCV(n_alphas=25, cv=4, early_stopping=25).fit(X, y)
    np.testing.assert_array_equal(early.alphas_, plain.alphas_)
    np.testing.assert_array_equal(early.cv_scores_, plain.cv_scores_)
    assert early.alpha_ == plain.alpha_
    np.testing.assert_array_equal(early.coef_, plain.coef_)
    np.testing.assert_array_equal(early.intercept_, plain.intercept_)
    assert early.n_iter_ == plain.n_iter_
    # the fit on all the rows ends at the chosen alpha
    best = int(np.flatnonzero(plain.alphas_ == plain.alpha_)[0])
    assert len(early.path_) == best + 1
    assert_bitwise(early.path_, plain.path_, length=len(early.path_))


def stopping_point(scores, n_lambdas, patience):
    """The last index a fold reaches under the rule (lower is better)."""
    best, since = np.inf, 0
    for k, score in enumerate(scores):
        if score < best:
            best, since = score, 0
        else:
            since += 1
        if since >= patience:
            return k
    return n_lambdas - 1


@pytest.mark.parametrize("patience", [1, 3, 8])
def test_cv_folds_stop_where_the_rule_says(data, monkeypatch, patience):
    # the rule is replayed on the scores each fold's stopping check computed
    # (a one-column prediction per lambda), not on cv_scores_ (all lambdas at
    # once), which agree only up to rounding: a near tie with a running best
    # could then decide differently
    X, y = data
    checks = []

    class Recording(bl_sklearn._NoImprovement):
        def __init__(self, *args):
            super().__init__(*args)
            self.scores = []
            checks.append(self)

        def score(self, *args):
            self.scores.append(super().score(*args))
            return self.scores[-1]

    monkeypatch.setattr(bl_sklearn, "_NoImprovement", Recording)
    plain = BiglassoClassifierCV(n_alphas=40, cv=5).fit(X, y)
    early = BiglassoClassifierCV(
        n_alphas=40, cv=5, early_stopping=patience, n_jobs=1
    ).fit(X, y)
    assert len(checks) == 5  # one per fold, in fold order (n_jobs=1)
    for fold, check in enumerate(checks):
        last = len(check.scores) - 1
        assert last == stopping_point(check.scores, len(plain.alphas_), patience)
        # the rule saw the fold's held-out deviance: the same sums as
        # cv_scores_, grouped differently, so equal to rounding (deviance of
        # order n, summed over n rows)
        np.testing.assert_allclose(
            np.array(check.scores) / len(check.rows),
            plain.cv_scores_[fold, : last + 1],
            rtol=1e-12,
        )
        np.testing.assert_array_equal(
            early.cv_scores_[fold, : last + 1], plain.cv_scores_[fold, : last + 1]
        )
        assert np.isnan(early.cv_scores_[fold, last + 1 :]).all()
    reached = ~np.isnan(early.cv_scores_).any(axis=0)
    mean = np.full(len(plain.alphas_), np.inf)
    mean[reached] = np.nanmean(plain.cv_scores_[:, reached], axis=0)
    assert early.alpha_ == plain.alphas_[np.argmin(mean)]
    assert len(early.path_) == np.argmin(mean) + 1
    assert_bitwise(early.path_, plain.path_, length=len(early.path_))


def test_cv_stops_early_on_this_data(data):
    # so the comparisons above aren't vacuous
    X, y = data
    early = BiglassoClassifierCV(n_alphas=40, cv=5, early_stopping=3).fit(X, y)
    assert np.isnan(early.cv_scores_).any(axis=1).all()


@pytest.mark.parametrize("rule", ["min", "1se"])
def test_cv_stopping_with_a_truncated_full_fit(data, monkeypatch, rule):
    # when the fit on all the rows ends sooner than the folds (here by dfmax),
    # only the alphas it reached are candidates, with or without stopping
    X, y = data
    fit = bl_sklearn.biglasso

    def truncated(X, y, row_indices=None, **kwargs):
        if row_indices is None:
            kwargs["dfmax"] = 6
        return fit(X, y, row_indices=row_indices, **kwargs)

    monkeypatch.setattr(bl_sklearn, "biglasso", truncated)
    cv = dict(n_alphas=30, cv=4, rule=rule)
    with pytest.warns(UserWarning, match="dfmax"):
        plain = BiglassoClassifierCV(**cv).fit(X, y)
    with pytest.warns(UserWarning, match="dfmax"):
        early = BiglassoClassifierCV(early_stopping=30, **cv).fit(X, y)
    reached = len(plain.alphas_)
    assert reached < 30
    np.testing.assert_array_equal(early.cv_scores_[:, :reached], plain.cv_scores_)
    assert early.alpha_ == plain.alpha_
    np.testing.assert_array_equal(early.coef_, plain.coef_)
    assert_bitwise(early.path_, plain.path_, length=len(early.path_))


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "10"])
def test_cv_rejects_bad_early_stopping(data, value):
    X, y = data
    with pytest.raises(ValueError, match="early_stopping"):
        BiglassoClassifierCV(early_stopping=value).fit(X, y)
