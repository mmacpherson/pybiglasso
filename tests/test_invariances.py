"""Properties any correct solver has, with expected values from the math or from
another solver; none needs R. The ideas come from other solvers' test suites
(scikit-learn, glmnet, ncvreg, celer, skglm).

Two kinds of agreement appear here, and the tolerance follows from which:

- Same trajectory. Duplicating rows, flipping the labels and rescaling a column
  leave every quantity the solver computes unchanged in exact arithmetic, so
  the two fits differ only by rounding. In general rounding can still push a
  strong-rule, KKT or stopping test across its boundary, so the exact
  iteration and rejection counts asserted here are properties of these
  fixtures, not a theorem; the fixtures also keep n > p after duplication (the
  lambda_min default switches there) and keep every rescaled column's scale
  far above the 1e-6 cutoff. Values are compared at rtol 1e-9, a stated
  policy: rounding with room to spare, not a proven error budget. Scaling by a
  power of two is exact away from overflow and subnormals, which these values
  are, so there the fits agree bit for bit.
- Same optimum, different route (permuted columns, another solver). Each
  answer must first pass a fixed accuracy test, its KKT or gradient residual
  below KKT_TOL. Then the distance between the two is bounded by convexity:
  if the mean NLL has curvature at least mu along the segment between them,
  |theta1 - theta2| <= (|e1| + |e2|) / mu, with e_i their residuals (the
  penalty is convex, so it only helps). `segment_curvature` gives such a mu.
"""

import decimal
import warnings

import numpy as np
import pytest
from scipy.special import expit, logit
from support.certificates import (
    KKT_TOL,
    certified_distance,
    gradient_lipschitz,
    kkt_residual,
    logistic_weight,
    mean_nll_gradient,
    segment_curvature,
    standardize,
    standardized_kkt,
    standardized_theta,
    support_is_certified,
    with_ones,
)
from support.certificates import path_objective as objective
from support.paths import assert_bitwise, assert_same_trajectory
from test_biglasso import saga

import pybiglasso as bl


def dense(path):
    return path.coef.toarray()


@pytest.mark.parametrize("eta", [20.0, 30.0, -30.0])
def test_segment_curvature_stays_below_the_true_weight(eta):
    """Where p rounds to nearly 1, p(1 - p) cancels; the weight must stay
    accurate, and the curvature bound must not exceed the true curvature,
    here e^-|eta| / (1 + e^-|eta|)^2 to 50 digits."""
    with decimal.localcontext() as ctx:
        ctx.prec = 50
        u = (-abs(decimal.Decimal(eta))).exp()
        true_weight = u / (1 + u) ** 2
        weight = logistic_weight(np.array([eta]))[0]
        assert abs(decimal.Decimal(weight) / true_weight - 1) < 1e-14
        # two rows, one of each class, at the same eta: the Hessian is the weight
        Z, y = np.ones((2, 1)), np.array([0.0, 1.0])
        theta = np.array([eta])
        assert decimal.Decimal(segment_curvature(Z, y, theta, theta)) <= true_weight


def test_certified_distance_accounts_for_flattening_curvature():
    """A check on the helper: logistic curvature at one point says little about
    it far away. One row of each class with only an intercept: theta = 0 is the
    minimum (gradient 0, curvature 1/4) and theta = 10 has gradient about 1/2,
    so curvature at 0 alone would claim a distance under 4; it is 10."""
    Z, y = np.ones((2, 1)), np.array([0.0, 1.0])
    t1, t2 = np.array([0.0]), np.array([10.0])
    e1, e2 = (abs(mean_nll_gradient(Z, y, t)[0]) for t in (t1, t2))
    assert certified_distance(Z, y, t1, t2, e1, e2) >= 10
    # and on random pairs the bound holds, as it must
    rng = np.random.default_rng(0)
    Zr = with_ones(rng.normal(size=(50, 3)))
    yr = rng.binomial(1, 0.4, size=50).astype(float)
    for _ in range(20):
        a, b = rng.normal(scale=3, size=(2, 4))
        ea, eb = (np.linalg.norm(mean_nll_gradient(Zr, yr, t)) for t in (a, b))
        assert np.linalg.norm(a - b) <= certified_distance(Zr, yr, a, b, ea, eb)


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
        # KKT residuals are max-norms; the bound wants 2-norms
        root = np.sqrt(len(theta))
        bound = certified_distance(Z, y, theta, theta_p, root * r_fit, root * r_perm)
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
        assert_bitwise(rescaled, fit, coef_b=expected)
    else:
        assert_same_trajectory(rescaled, fit, coef_b=expected)


# the ends of the path -------------------------------------------------------


def test_lambdas_above_lambda_max_give_the_null_model(data):
    """lambda_max = max_j |x_j'(y - ybar)| / (n * alpha * pf_j) on standardized
    columns: above it the null model satisfies KKT, and just below it the
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
    compare with scikit-learn's unpenalized Newton solver. Both must first
    nearly zero the unpenalized gradient: within KKT_TOL, plus lambda's pull
    of at most lambda per coordinate for biglasso. Their residuals then bound
    their distance."""
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
    lam = fit.lambdas[0]
    g_ours, g_theirs = (mean_nll_gradient(Z, y, t) for t in (ours, theirs))
    assert kkt_residual(fit, X, y, 0) < KKT_TOL
    assert np.abs(g_ours).max() < KKT_TOL + lam
    assert np.abs(g_theirs).max() < KKT_TOL
    distance = certified_distance(
        Z, y, ours, theirs, np.linalg.norm(g_ours), np.linalg.norm(g_theirs)
    )
    assert np.linalg.norm(ours - theirs) <= distance
    # and on X's own scale
    assert np.all(np.abs(dense(fit)[:, 0] - sk.coef_.ravel()) <= distance / scale)


@pytest.mark.parametrize("alpha", [1.0, 0.5])
def test_support_matches_saga(alpha):
    """saga's proximal step gives exact zeros, so supports can be compared.

    Both fits must pass KKT within KKT_TOL, which bounds their distance d. A
    saga coefficient within KKT_TOL of KKT can only be nonzero if its gradient
    has magnitude at least lambda * alpha - KKT_TOL (the elastic net's L2 term
    only pushes it further out), and gradients differ by at most L * d
    between the fits. So where biglasso's zero has slack above
    KKT_TOL + L * d, saga's is exactly zero too, and where biglasso's nonzero
    exceeds d, saga's is nonzero with the same sign. The test asserts that
    every coordinate is resolved this way, then compares.
    """
    sklm = pytest.importorskip("sklearn.linear_model")
    d = bl.common.generate_test_data(300, 12, seed=3)
    X, y = d["X"], d["y"]
    n = len(y)
    Xs, center, scale = standardize(X)
    Z = with_ones(Xs)
    root = np.sqrt(Z.shape[1])
    fit = bl.biglasso(X, y, alpha=alpha, nlambda=8, lambda_min=0.02, eps=1e-12)
    compared = 0
    for k in (2, 4, 7):
        lam = fit.lambdas[k]
        ours = standardized_theta(fit.intercept[k], dense(fit)[:, k], center, scale)
        sk = saga(sklm, C=1 / (n * lam), l1_ratio=alpha).fit(Xs, y)
        theirs = np.concatenate([sk.intercept_, sk.coef_.ravel()])
        r_ours, r_theirs = (
            standardized_kkt(Z, y, t, lam, alpha) for t in (ours, theirs)
        )
        assert max(r_ours, r_theirs) < KKT_TOL
        distance = certified_distance(
            Z, y, ours, theirs, root * r_ours, root * r_theirs
        )
        b = ours[1:]
        g = Xs.T @ (y - expit(Z @ ours)) / n
        zero = b == 0
        assert support_is_certified(
            b,
            g,
            lam * alpha,
            distance=distance,
            residual_bound=KKT_TOL,
            lipschitz=gradient_lipschitz(Z),
        )
        np.testing.assert_array_equal(theirs[1:] != 0, ~zero)
        np.testing.assert_array_equal(np.sign(theirs[1:]), np.sign(b))
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
    # this caps the prefit's gradient too: column 0 and the intercept
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
    g_ours, g_theirs = (mean_nll_gradient(Z, y, t) for t in (ours, theirs))
    assert np.abs(g_theirs).max() < KKT_TOL
    distance = certified_distance(
        Z, y, ours, theirs, np.linalg.norm(g_ours), np.linalg.norm(g_theirs)
    )
    assert np.linalg.norm(ours - theirs) <= distance
    # lambda_max is the penalized column's score at the prefit, which moves at
    # most L * distance between the two fits (L of the gradient on [Z, x_2]);
    # 1e-12 covers rounding in the two computations of the score
    score = Xs[:, 1] @ (y - expit(Z @ theirs)) / len(y)
    lipschitz = gradient_lipschitz(np.column_stack([Z, Xs[:, 1]]))
    assert abs(fit.lambdas[0] - abs(score)) <= lipschitz * distance + 1e-12
