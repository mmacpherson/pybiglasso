import os

import numpy as np
import pytest

pytest.importorskip("sklearn")

from sklearn.utils.estimator_checks import parametrize_with_checks  # noqa: E402

import pybiglasso as bl  # noqa: E402
from pybiglasso.sklearn import BiglassoClassifier, BiglassoClassifierCV  # noqa: E402


@pytest.mark.filterwarnings("ignore::UserWarning")
@parametrize_with_checks(
    [
        BiglassoClassifier(alpha=0.01),
        BiglassoClassifierCV(n_alphas=10, cv=3),
    ]
)
def test_sklearn_compatible(estimator, check):
    check(estimator)


@pytest.fixture(scope="module")
def data():
    d = bl.common.generate_test_data(400, 15, seed=5)
    return d["X"], d["y"]


def test_classifier_matches_functional_api(data):
    X, y = data
    path = bl.biglasso(X, y, alpha=0.7, nlambda=10)
    lam = path.lambdas[6]
    clf = BiglassoClassifier(alpha=lam, l1_ratio=0.7).fit(X, y)
    ref = bl.biglasso(X, y, alpha=0.7, lambdas=[lam])
    np.testing.assert_allclose(clf.coef_.ravel(), ref.coef.toarray().ravel())
    np.testing.assert_allclose(clf.intercept_, ref.intercept)
    # and close to the warm-started path solution at the same lambda
    np.testing.assert_allclose(
        clf.coef_.ravel(), path.coef[:, [6]].toarray().ravel(), atol=1e-4
    )


def test_classifier_string_labels(data):
    X, y = data
    labels = np.where(y == 1, "case", "control")
    clf = BiglassoClassifier(alpha=0.01).fit(X, labels)
    assert set(clf.classes_) == {"case", "control"}
    proba = clf.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1)
    np.testing.assert_array_equal(
        clf.predict(X), clf.classes_[(proba[:, 1] > 0.5).astype(int)]
    )


@pytest.mark.parametrize("scoring", ["deviance", "auc", "accuracy"])
def test_cv_selects_from_path(data, scoring):
    X, y = data
    cv = BiglassoClassifierCV(n_alphas=20, cv=4, scoring=scoring).fit(X, y)
    assert cv.cv_scores_.shape == (4, len(cv.alphas_))
    assert cv.alpha_ in cv.alphas_
    k = int(np.flatnonzero(cv.alphas_ == cv.alpha_)[0])
    np.testing.assert_allclose(
        cv.coef_.ravel(), cv.path_.coef[:, [k]].toarray().ravel()
    )


def test_cv_1se_is_sparser(data):
    X, y = data
    best = BiglassoClassifierCV(n_alphas=30, cv=5).fit(X, y)
    onese = BiglassoClassifierCV(n_alphas=30, cv=5, rule="1se").fit(X, y)
    assert onese.alpha_ >= best.alpha_
    assert (onese.coef_ != 0).sum() <= (best.coef_ != 0).sum()


def test_cv_folds_use_row_indices(data, tmp_path):
    """CV on a memmap reads rows by index rather than copying the design."""
    X, y = data
    mm = np.lib.format.open_memmap(
        tmp_path / "X.npy", mode="w+", dtype=X.dtype, shape=X.shape, fortran_order=True
    )
    mm[:] = X
    mm.flush()
    a = BiglassoClassifierCV(n_alphas=10, cv=3).fit(X, y)
    b = BiglassoClassifierCV(n_alphas=10, cv=3).fit(
        np.load(tmp_path / "X.npy", mmap_mode="r"), y
    )
    np.testing.assert_allclose(a.cv_scores_, b.cv_scores_)


def test_cv_n_jobs_does_not_change_results(data):
    X, y = data
    one = BiglassoClassifierCV(n_alphas=15, cv=5, n_jobs=1).fit(X, y)
    many = BiglassoClassifierCV(n_alphas=15, cv=5, n_jobs=5).fit(X, y)
    np.testing.assert_array_equal(one.cv_scores_, many.cv_scores_)
    assert one.alpha_ == many.alpha_


def test_cv_under_workqueue_runs_folds_one_at_a_time():
    """numba's workqueue layer (macOS's default) aborts on concurrent kernels."""
    import subprocess
    import sys

    code = """
import warnings
import pybiglasso as bl
from pybiglasso import BiglassoClassifierCV
d = bl.common.generate_test_data(200, 8, seed=1)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    BiglassoClassifierCV(n_alphas=8, cv=4, n_jobs=4).fit(d["X"], d["y"])
    BiglassoClassifierCV(n_alphas=8, cv=4).fit(d["X"], d["y"])
print(sum("one at a time" in str(x.message) for x in w))
"""
    env = {**os.environ, "NUMBA_THREADING_LAYER": "workqueue"}
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr[-2000:]
    # a warning for the explicit n_jobs=4, none for the default
    assert out.stdout.split()[-1] == "1"
