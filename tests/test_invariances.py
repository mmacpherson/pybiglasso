"""Properties any correct solver has, with expected values from the math or from
another solver; none needs R. The ideas come from other solvers' test suites
(scikit-learn, glmnet, ncvreg, celer, skglm).

Two kinds of agreement appear here, and the tolerance follows from which:

- Same trajectory. When a change to the data leaves every quantity the solver
  computes unchanged in exact arithmetic (duplicating rows, flipping the
  labels, rescaling a column), the two fits run the same iterations and differ
  only by rounding. Iteration counts must then match, and results agree to
  rtol 1e-9, which leaves a factor of about 1e6 over double rounding for its
  accumulation through hundreds of sweeps. Scaling by a power of two commutes
  with every floating-point operation, so there the fits agree bit for bit.
- Same optimum, different route (permuted columns, another solver). Each
  answer is certified by its own KKT residual, and the distance between two
  certified answers is bounded by strong convexity: for minimizers of a
  mu-strongly convex objective, |theta1 - theta2| <= (|e1| + |e2|) / mu, with
  e_i the residuals. mu is measured from the Hessian at the solution, and
  halved to cover the segment between the two points.
"""

import warnings

import numpy as np
import pytest
from scipy.special import expit, logit
from test_biglasso import saga
from test_default_enet import KKT_TOL, kkt_residual, objective

import pybiglasso as bl

SAME_TRAJECTORY = dict(rtol=1e-9, atol=1e-12)


def dense(path):
    return path.coef.toarray()


def standardize(X):
    X = X.astype(np.float64)
    center, scale = X.mean(axis=0), X.std(axis=0)
    return (X - center) / scale, center, scale


def standardized_theta(intercept, coef, center, scale):
    """(intercept, coefficients) on standardized columns."""
    return np.concatenate([[intercept + center @ coef], coef * scale])


def with_ones(Xs):
    return np.column_stack([np.ones(len(Xs)), Xs])


def mean_nll_gradient(Z, y, theta):
    return Z.T @ (expit(Z @ theta) - y) / len(y)


def strong_convexity(Z, y, theta):
    """Half the smallest Hessian eigenvalue of the mean NLL at theta."""
    p = expit(Z @ theta)
    hessian = Z.T @ (Z * (p * (1 - p))[:, None]) / len(y)
    return np.linalg.eigvalsh(hessian).min() / 2


def assert_same_trajectory(a, b, coef_b=None, intercept_b=None, deviance_b=None):
    np.testing.assert_array_equal(a.iterations, b.iterations)
    np.testing.assert_array_equal(a.n_rejected, b.n_rejected)
    np.testing.assert_allclose(a.lambdas, b.lambdas, rtol=1e-12)
    np.testing.assert_allclose(
        dense(a), dense(b) if coef_b is None else coef_b, **SAME_TRAJECTORY
    )
    np.testing.assert_allclose(
        a.intercept,
        b.intercept if intercept_b is None else intercept_b,
        **SAME_TRAJECTORY,
    )
    np.testing.assert_allclose(
        a.deviance,
        b.deviance if deviance_b is None else deviance_b,
        **SAME_TRAJECTORY,
    )


@pytest.fixture(scope="module")
def data():
    d = bl.common.generate_test_data(300, 12, seed=17)
    return d["X"], d["y"]


# data transformations that leave the problem unchanged ---------------------


@pytest.mark.parametrize("rows", ["all", "subset"])
def test_duplicating_every_row_leaves_fit_unchanged(data, rows):
    """The objective is (1/n) * NLL, so stacking the rows twice changes nothing
    but n; standardization, the grid and every update are the same. Deviance is
    a sum over rows, so it doubles."""
    X, y = data
    once_rows = None if rows == "all" else np.arange(0, len(y), 2)
    base = np.arange(len(y)) if once_rows is None else once_rows
    once = bl.biglasso(X, y, alpha=0.7, row_indices=once_rows, nlambda=30)
    twice = bl.biglasso(X, y, alpha=0.7, row_indices=np.r_[base, base], nlambda=30)
    assert len(once) == 30 and once.n_nonzero[-1] > 0
    assert_same_trajectory(twice, once, deviance_b=2 * once.deviance)


@pytest.mark.parametrize("alpha, unpenalized", [(1.0, False), (0.5, True)])
def test_flipping_labels_negates_the_fit(data, alpha, unpenalized):
    """y -> 1 - y maps the likelihood at (a, b) to that at (-a, -b), and the
    penalty is even. The solver's steps are symmetric too: |z| sets the grid
    and strong rule, and the IRLS clamp at |eta| > 10 is two-sided."""
    X, y = data
    pf = np.ones(X.shape[1])
    if unpenalized:
        pf[:2] = 0
    fit = bl.biglasso(X, y, alpha=alpha, penalty_factor=pf, nlambda=30)
    flipped = bl.biglasso(X, 1 - y, alpha=alpha, penalty_factor=pf, nlambda=30)
    assert fit.n_nonzero[-1] > 0
    assert_same_trajectory(flipped, fit, coef_b=-dense(fit), intercept_b=-fit.intercept)


def test_permuting_columns_permutes_the_solution(data):
    """Permuting columns, and their penalty factors along with them, permutes
    the solution. Coordinate descent visits them in another order, so compare
    the certified optima, not the iterates."""
    X, y = data
    rng = np.random.default_rng(4)
    pf = rng.uniform(0.5, 2, X.shape[1])
    perm = rng.permutation(X.shape[1])
    Xp = np.asfortranarray(X[:, perm])
    kw = dict(alpha=0.8, nlambda=20, eps=1e-14)
    fit = bl.biglasso(X, y, penalty_factor=pf, **kw)
    permuted = bl.biglasso(Xp, y, penalty_factor=pf[perm], **kw)
    # lambda_max is the largest |z_j| / pf_j, the same set of numbers computed
    # the same way column by column, so the grid is exact
    np.testing.assert_array_equal(fit.lambdas, permuted.lambdas)
    assert len(fit) == len(permuted) == 20

    Xs, center, scale = standardize(X)
    Z = with_ones(Xs)
    for k in range(len(fit)):
        r_fit = kkt_residual(fit, X, y, k)
        r_perm = kkt_residual(permuted, Xp, y, k)
        assert max(r_fit, r_perm) < KKT_TOL
        theta = standardized_theta(fit.intercept[k], dense(fit)[:, k], center, scale)
        theta_p = np.empty_like(theta)
        theta_p[0] = permuted.intercept[k] + center[perm] @ dense(permuted)[:, k]
        theta_p[1 + perm] = dense(permuted)[:, k] * scale[perm]
        bound = np.sqrt(len(theta)) * (r_fit + r_perm) / strong_convexity(Z, y, theta)
        assert np.linalg.norm(theta - theta_p) <= bound
        np.testing.assert_allclose(
            objective(fit, X, y, k), objective(permuted, Xp, y, k), rtol=1e-10
        )


@pytest.mark.parametrize(
    "dtype, c",
    [(np.float64, 4), (np.float64, 3), (np.int8, 2), (np.int8, 3)],
    ids=["float-by-4", "float-by-3", "int8-by-2", "int8-by-3"],
)
def test_rescaling_a_column_divides_its_coefficient(dtype, c):
    """Columns are standardized, so x_j -> c * x_j (c > 0) gives the same
    standardized problem: the same grid, and coef_j / c. Genotype dosages are
    at most 2, so int8 holds them times 3 without overflow."""
    if dtype == np.int8:
        X = bl.common.make_biglasso_dataset(300, 12, seed=8, dtype=np.int8)
        X, y = X["X"], X["y"]
    else:
        d = bl.common.generate_test_data(300, 12, seed=8)
        X, y = d["X"], d["y"]
    scaled_cols = [0, 3, 7]
    Xc = X.copy(order="F")
    Xc[:, scaled_cols] *= dtype(c)
    # no int8 wraparound: the scaled design is exactly c times the original
    assert Xc.dtype == dtype
    np.testing.assert_array_equal(
        Xc[:, scaled_cols], c * X[:, scaled_cols].astype(float)
    )
    fit = bl.biglasso(X, y, alpha=0.9, nlambda=25)
    rescaled = bl.biglasso(Xc, y, alpha=0.9, nlambda=25)
    assert np.all(dense(fit)[scaled_cols].any(axis=1))
    expected = dense(fit)
    expected[scaled_cols] /= c
    if c & (c - 1) == 0:
        # a power of two scales every operation exactly
        np.testing.assert_array_equal(rescaled.lambdas, fit.lambdas)
        np.testing.assert_array_equal(rescaled.iterations, fit.iterations)
        np.testing.assert_array_equal(dense(rescaled), expected)
        np.testing.assert_array_equal(rescaled.intercept, fit.intercept)
        np.testing.assert_array_equal(rescaled.deviance, fit.deviance)
    else:
        assert_same_trajectory(rescaled, fit, coef_b=expected)


# the ends of the path -------------------------------------------------------


def test_lambdas_at_or_above_lambda_max_give_the_null_model(data):
    """lambda_max = max_j |x_j'(y - ybar)| / (n * alpha * pf_j) on standardized
    columns: at or above it the null model satisfies KKT, and just below it the
    column attaining the max enters (alone, as the next one is far off)."""
    X, y = data
    n, alpha = len(y), 0.6
    pf = np.random.default_rng(9).uniform(0.5, 2, X.shape[1])
    Xs, _, _ = standardize(X)
    ybar = y.mean()
    ratios = np.abs(Xs.T @ (y - ybar)) / n / pf / alpha
    lambda_max = ratios.max()
    assert np.sort(ratios)[-2] < 0.99 * lambda_max
    null_deviance = -2 * n * (ybar * np.log(ybar) + (1 - ybar) * np.log(1 - ybar))

    fit = bl.biglasso(
        X,
        y,
        alpha=alpha,
        penalty_factor=pf,
        lambdas=lambda_max * np.array([2, 1 + 1e-6, 1 - 1e-6]),
        eps=1e-12,
    )
    assert len(fit) == 3
    assert fit.coef[:, [0, 1]].nnz == 0
    np.testing.assert_allclose(fit.intercept[:2], logit(ybar), rtol=1e-12)
    np.testing.assert_allclose(fit.deviance[:2], null_deviance, rtol=1e-12)
    np.testing.assert_array_equal(np.flatnonzero(dense(fit)[:, 2]), [np.argmax(ratios)])


def _unpenalized_logistic(sklm):
    import sklearn

    major, minor = (int(v) for v in sklearn.__version__.split(".")[:2])
    # from 1.8 the penalty argument is deprecated, and C=inf means none
    extra = dict(C=np.inf) if (major, minor) >= (1, 8) else dict(penalty=None)
    return sklm.LogisticRegression(
        solver="newton-cholesky", tol=1e-14, max_iter=1000, **extra
    )


def test_tiny_lambda_approaches_the_unpenalized_mle():
    """With n >> p and lambda near zero the fit is the maximum-likelihood fit;
    compare with scikit-learn's unpenalized Newton solver. Each point's
    gradient of the unpenalized problem (lambda's pull included) bounds its
    distance to the MLE."""
    sklm = pytest.importorskip("sklearn.linear_model")
    d = bl.common.generate_test_data(3000, 4, seed=21)
    X, y = d["X"], d["y"]
    fit = bl.biglasso(X, y, lambdas=[1e-9], eps=1e-14)
    sk = _unpenalized_logistic(sklm).fit(X, y)

    Xs, center, scale = standardize(X)
    Z = with_ones(Xs)
    ours = standardized_theta(fit.intercept[0], dense(fit)[:, 0], center, scale)
    theirs = standardized_theta(sk.intercept_[0], sk.coef_.ravel(), center, scale)
    assert np.all(ours[1:] != 0)
    residuals = sum(np.linalg.norm(mean_nll_gradient(Z, y, t)) for t in (ours, theirs))
    assert np.linalg.norm(ours - theirs) <= residuals / strong_convexity(Z, y, ours)
    # and on X's own scale
    bound = residuals / strong_convexity(Z, y, ours) / scale
    assert np.all(np.abs(dense(fit)[:, 0] - sk.coef_.ravel()) <= bound)


@pytest.mark.parametrize("alpha", [1.0, 0.5])
def test_support_matches_saga(alpha):
    """saga's proximal step sets a coefficient exactly to zero once its
    gradient is inside the penalty's kink, so at convergence its zeros are
    exact. Compare supports where both solvers resolve them: every zero has
    KKT slack, and every nonzero a magnitude, of at least 1e-5, ten times the
    1e-6 the coefficients are already known to agree to."""
    sklm = pytest.importorskip("sklearn.linear_model")
    d = bl.common.generate_test_data(300, 12, seed=3)
    X, y = d["X"], d["y"]
    n = len(y)
    Xs, center, scale = standardize(X)
    fit = bl.biglasso(X, y, alpha=alpha, nlambda=8, lambda_min=0.02, eps=1e-12)
    margin = 1e-5
    compared = 0
    for k in (2, 4, 7):
        lam = fit.lambdas[k]
        b = dense(fit)[:, k] * scale
        eta = fit.intercept[k] + X @ dense(fit)[:, k]
        g = Xs.T @ (y - expit(eta)) / n
        zero = b == 0
        slack = lam * alpha - np.abs(g[zero])
        assert np.all(slack > margin) and np.all(np.abs(b[~zero]) > margin)
        sk = saga(sklm, C=1 / (n * lam), l1_ratio=alpha).fit(Xs, y)
        np.testing.assert_array_equal(sk.coef_.ravel() != 0, ~zero)
        compared += zero.sum()
    assert compared > 0, "no zeros to compare"


# cross-validation bookkeeping -----------------------------------------------


def _held_out_score(fold_path, X, y, test, scoring):
    metrics = pytest.importorskip("sklearn.metrics")
    eta = fold_path.intercept + X[test] @ dense(fold_path)
    yt = y[test]
    if scoring == "deviance":
        # -2 log-likelihood per observation, computed stably
        return 2 * np.mean(np.logaddexp(0, eta) - yt[:, None] * eta, axis=0)
    if scoring == "auc":
        return np.array([metrics.roc_auc_score(yt, e) for e in eta.T])
    return np.array([metrics.accuracy_score(yt, e > 0) for e in eta.T])


def _rebuild_cv(cv, X, y, n_folds, l1_ratio, scoring):
    """cv_scores_ and alpha_ by hand, from biglasso fits of each training fold."""
    from sklearn.model_selection import StratifiedKFold

    grid = cv.alphas_
    scores = np.full((n_folds, len(grid)), np.nan)
    folds = StratifiedKFold(n_folds).split(np.zeros((len(y), 1)), y)
    for k, (train, test) in enumerate(folds):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fold_path = bl.biglasso(
                X, y, row_indices=train, lambdas=grid, alpha=l1_ratio
            )
        s = _held_out_score(fold_path, X, y, test, scoring)
        scores[k, : len(s)] = s
    return scores


def _choose(scores, scoring, rule):
    """glmnet's rules, over the alphas every fold reached."""
    complete = ~np.isnan(scores).any(axis=0)
    mean = scores[:, complete].mean(axis=0)
    if scoring != "deviance":
        mean = -mean
    best = np.argmin(mean)
    if rule == "1se":
        se = scores[:, complete][:, best].std(ddof=1) / np.sqrt(len(scores))
        best = np.flatnonzero(mean <= mean[best] + se)[0]
    return np.flatnonzero(complete)[best]


@pytest.mark.parametrize(
    "scoring, rule",
    [("deviance", "min"), ("auc", "min"), ("accuracy", "min"), ("deviance", "1se")],
)
def test_cv_scores_rebuilt_by_hand(data, scoring, rule):
    from pybiglasso.sklearn import BiglassoClassifierCV

    X, y = data
    cv = BiglassoClassifierCV(
        l1_ratio=0.8, n_alphas=20, cv=4, scoring=scoring, rule=rule
    ).fit(X, y)
    scores = _rebuild_cv(cv, X, y, 4, 0.8, scoring)
    assert not np.isnan(scores).any()
    np.testing.assert_allclose(cv.cv_scores_, scores, rtol=1e-12)
    assert cv.alpha_ == cv.alphas_[_choose(scores, scoring, rule)]


@pytest.mark.parametrize("rule", ["min", "1se"])
def test_cv_masks_folds_that_stop_on_saturation(rule):
    """Nearly separable labels saturate some folds before others; their unfit
    alphas are NaN, and the choice is among alphas every fold reached."""
    from pybiglasso.sklearn import BiglassoClassifierCV

    rng = np.random.default_rng(6)
    X = np.asfortranarray(rng.normal(size=(200, 6)))
    y = (X[:, 0] + 0.4 * X[:, 1] + 0.05 * rng.normal(size=200) > 0).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = BiglassoClassifierCV(
            n_alphas=40, alpha_min_ratio=1e-4, cv=5, rule=rule
        ).fit(X, y)
    scores = _rebuild_cv(cv, X, y, 5, 1.0, "deviance")
    fit_lengths = (~np.isnan(scores)).sum(axis=1)
    # the case under test: folds stop at different alphas, short of the grid
    assert len(set(fit_lengths)) > 1 and fit_lengths.min() < len(cv.alphas_)
    np.testing.assert_array_equal(np.isnan(cv.cv_scores_), np.isnan(scores))
    np.testing.assert_allclose(cv.cv_scores_, scores, rtol=1e-12)
    chosen = _choose(scores, "deviance", rule)
    assert chosen < fit_lengths.min()
    assert cv.alpha_ == cv.alphas_[chosen]


# edge shapes ------------------------------------------------------------------


# a one-column array is both C- and F-contiguous, and np.isfortran says False
@pytest.mark.filterwarnings("ignore:The design matrix is not in Fortran")
def test_single_feature():
    """With one column, lambda_max is |x'(y - ybar)| / (n * alpha) on the
    standardized column, and KKT holds along the path."""
    d = bl.common.generate_test_data(300, 1, seed=12)
    X, y = d["X"], d["y"]
    Xs, _, _ = standardize(X)
    alpha = 0.7
    fit = bl.biglasso(X, y, alpha=alpha, nlambda=20, eps=1e-14)
    lambda_max = abs(Xs[:, 0] @ (y - y.mean())) / len(y) / alpha
    np.testing.assert_allclose(fit.lambdas[0], lambda_max, rtol=1e-12)
    assert fit.n_nonzero[0] == 0 and fit.n_nonzero[-1] == 1
    assert max(kkt_residual(fit, X, y, k) for k in range(len(fit))) < KKT_TOL


def test_unpenalized_column_beside_a_penalized_one():
    """At lambda_max, the fit is the maximum-likelihood fit on the unpenalized
    column alone (here from scikit-learn), and lambda_max is the penalized
    column's score there. Along the path, KKT holds."""
    sklm = pytest.importorskip("sklearn.linear_model")
    d = bl.common.generate_test_data(400, 2, seed=13)
    X, y = d["X"], d["y"]
    fit = bl.biglasso(X, y, penalty_factor=[0, 1], nlambda=20, eps=1e-14)
    assert dense(fit)[1, 0] == 0 and dense(fit)[0, 0] != 0 and fit.n_nonzero[-1] == 2
    assert max(kkt_residual(fit, X, y, k) for k in range(len(fit))) < KKT_TOL

    sk = _unpenalized_logistic(sklm).fit(X[:, :1], y)
    Xs, center, scale = standardize(X)
    Z = with_ones(Xs[:, :1])
    ours = standardized_theta(
        fit.intercept[0], dense(fit)[:1, 0], center[:1], scale[:1]
    )
    theirs = standardized_theta(
        sk.intercept_[0], sk.coef_.ravel(), center[:1], scale[:1]
    )
    mu = strong_convexity(Z, y, ours)
    distance = (
        sum(np.linalg.norm(mean_nll_gradient(Z, y, t)) for t in (ours, theirs)) / mu
    )
    assert np.linalg.norm(ours - theirs) <= distance
    # the score is Lipschitz in theta, with constant at most |[1, x_2]|^2 / (4n)
    score = Xs[:, 1] @ (y - expit(Z @ theirs)) / len(y)
    lipschitz = np.linalg.norm(np.column_stack([Z, Xs[:, 1]]), 2) ** 2 / (4 * len(y))
    assert abs(fit.lambdas[0] - abs(score)) <= lipschitz * distance
