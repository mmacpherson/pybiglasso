from decimal import Decimal, localcontext

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
        bl.BiglassoRegressor(alpha=0.01, backend="rust"),
        bl.BiglassoRegressorCV(n_alphas=10, cv=3, n_jobs=1, backend="rust"),
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
    path = bl.biglasso(X, y, alpha=0.7, nlambda=10, family="binomial")
    lam = path.lambdas[6]
    clf = BiglassoClassifier(alpha=lam, l1_ratio=0.7).fit(X, y)
    ref = bl.biglasso(X, y, alpha=0.7, lambdas=[lam], family="binomial")
    np.testing.assert_allclose(clf.coef_.ravel(), ref.coef.toarray().ravel())
    np.testing.assert_allclose(clf.intercept_, ref.intercept)
    # and close to the warm-started path solution at the same lambda. The
    # classifier fits at the default eps=1e-7, which bounds no KKT residual,
    # so no certificate applies (test_biglasso certifies warm against cold
    # starts at eps=1e-14); 1e-4 is a stated ceiling for default accuracy
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


def test_classifier_extreme_probabilities_are_accurate():
    eta = np.array([-700.0, -40.0, -36.0, -30.0, -20.0, 0.0, 20.0, 700.0])
    clf = BiglassoClassifier()
    clf.classes_ = np.array([0, 1])
    clf.n_features_in_ = 1
    clf.coef_ = np.ones((1, 1))
    clf.intercept_ = np.zeros(1)

    with localcontext() as context:
        context.prec = 400
        decimal_eta = [Decimal.from_float(value) for value in eta]
        probability = np.array(
            [float(1 / (1 + (-value).exp())) for value in decimal_eta]
        )
        log_probability = np.array(
            [float(-(1 + (-value).exp()).ln()) for value in decimal_eta]
        )
        log_complement = np.array(
            [float(-(1 + value.exp()).ln()) for value in decimal_eta]
        )

    proba = clf.predict_proba(eta[:, None])
    log_proba = clf.predict_log_proba(eta[:, None])
    np.testing.assert_array_max_ulp(proba[:, 1], probability, maxulp=2)
    np.testing.assert_array_max_ulp(log_proba[:, 1], log_probability, maxulp=2)
    np.testing.assert_array_max_ulp(log_proba[:, 0], log_complement, maxulp=2)
    assert np.isfinite(log_proba).all()


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
