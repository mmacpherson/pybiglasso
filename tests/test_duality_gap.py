"""Global certificates on wide logistic problems; no fitted reference values.

Ndiaye et al. (2017), Theorem 6 / Section 5.4 supplies conjugacy and the dual
sphere. Celer's test_celer_path_logreg also caps a path's gaps, but reports
summed loss without an intercept; these tests use mean loss and an intercept.
"""

import decimal
import warnings
from fractions import Fraction

import numpy as np
import pytest
from scipy.linalg import hadamard
from scipy.special import expit
from support import certificates, reference_path
from support.certificates import (
    KKT_TOL,
    _entropy_term_lower,
    _gap_certificate,
    _softplus_upper,
    _sqrt_fraction_upper,
    duality_gap,
    gap_coefficient_distance,
    gap_safe_zeros,
    standardize,
    standardized_kkt,
    standardized_theta,
    with_ones,
)

import pybiglasso as bl

# An accuracy requirement in nats per observation, chosen independently of
# solver output, not an inferred conversion of its stopping criterion.
GAP_TOL = 1e-8
FIT_OPTIONS = dict(lambdas=[0.2, 0.1, 0.03], eps=1e-16, max_iter=10_000, n_threads=1)


@pytest.mark.parametrize(
    "z", [Fraction(-30), Fraction(0), Fraction(1, 3), Fraction(30)]
)
def test_directed_softplus_encloses_high_precision_value(z):
    with decimal.localcontext(decimal.Context(prec=60, rounding=decimal.ROUND_CEILING)):
        upper = _softplus_upper(z)
    with decimal.localcontext(decimal.Context(prec=100)):
        value = decimal.Decimal(z.numerator) / decimal.Decimal(z.denominator)
        exact = (1 + value.exp()).ln()
        assert upper >= exact
        assert upper - exact < decimal.Decimal("1e-55")  # 60-digit enclosure, |z|<=30


@pytest.mark.parametrize(
    "q",
    [
        Fraction(0),
        Fraction(1),
        Fraction(1, 2),
        Fraction(1, 3),  # nonterminating Decimal: lower and upper brackets differ
        Fraction(1, 10**20),
        1 - Fraction(1, 10**20),
    ],
)
def test_directed_entropy_term_encloses_high_precision_value(q):
    with decimal.localcontext(decimal.Context(prec=60, rounding=decimal.ROUND_FLOOR)):
        lower = _entropy_term_lower(q)
    with decimal.localcontext(decimal.Context(prec=100)):
        value = decimal.Decimal(q.numerator) / decimal.Decimal(q.denominator)
        exact = -value * value.ln() if value else decimal.Decimal(0)
        assert lower <= exact
        assert exact - lower < decimal.Decimal("1e-55")


@pytest.mark.parametrize(
    "value", [Fraction(0), Fraction(2), 1 + Fraction(1, 10**100), Fraction(1, 10**1000)]
)
def test_outward_root_is_a_bound_in_exact_arithmetic(value):
    upper = _sqrt_fraction_upper(value)
    assert np.isfinite(upper)
    assert Fraction(upper) ** 2 >= value
    if value:
        assert Fraction(np.nextafter(upper, 0)) ** 2 <= value


def test_null_optimum_has_zero_gap_and_bad_point_has_positive_gap():
    # Both classes at every X: the mean NLL is minimized at theta=0, and the
    # positive penalty is minimized there too. F*=log(2), even with p>n.
    X = np.tile(np.array([[-1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, -1.0]]), (1, 3))
    Z, y = with_ones(X), np.array([0.0, 1.0, 0.0, 1.0])
    null = np.zeros(Z.shape[1])
    for alpha in (1.0, 0.5):
        cert = _gap_certificate(Z, y, null, 0.1, alpha, 1.0)
        assert 0 <= cert.gap <= 4 * cert.rounding
        assert cert.rounding < 1e-10  # final binary64 endpoint conversions
        assert np.all(gap_safe_zeros(Z, y, null, 0.1, alpha))
        bad = null.copy()
        bad[0] = 4.0  # ridge does not penalize or control this intercept
        # The independent exact excess is softplus(4)-2-log(2).
        excess = np.logaddexp(0, 4.0) - 2 - np.log(2)
        gap = duality_gap(Z, y, bad, 0.1, alpha)
        assert gap >= excess and gap > 1


def test_dual_balances_intercept_and_obeys_lasso_constraints_exactly():
    # Deliberately not a stationary intercept; naive u=(p-y)/n is infeasible.
    Z = np.array([[1.0, -3.0], [1.0, -1.0], [1.0, 1.0], [1.0, 3.0]])
    y, theta = np.array([0.0, 0.0, 1.0, 1.0]), np.array([2.0, 0.3])
    assert abs(np.sum(expit(Z @ theta) - y)) > 1
    cert = _gap_certificate(Z, y, theta, 0.01, 1.0, 0.5)
    assert sum(cert.point, Fraction()) == 0
    q = [int(yi) + len(y) * u for yi, u in zip(y, cert.point, strict=True)]
    assert all(0 <= v <= 1 for v in q)
    score = sum(
        (Fraction(float(x)) * u for x, u in zip(Z[:, 1], cert.point, strict=True)),
        Fraction(),
    )
    assert score == cert.scores[0]
    assert abs(score) <= Fraction(0.01) * Fraction(0.5)
    assert abs(score) > 0  # not an always-zero, vacuous feasible point


def test_lasso_closed_form_optimum_and_active_feature_not_screened():
    # Complementary success fractions 1/4 and 3/4 at x=-1/+1. For lambda=1/8,
    # beta*=logit(3/4-1/8)=log(5/3), intercept=0; this comes from scalar KKT.
    x = np.repeat([-1.0, 1.0], 4)
    Z, y = with_ones(x[:, None]), np.array([0, 0, 0, 1, 0, 1, 1, 1], dtype=float)
    theta = np.array([0.0, np.log(5 / 3)])
    cert = _gap_certificate(Z, y, theta, 0.125, 1.0, 1.0)
    assert 0 <= cert.gap <= 4 * cert.rounding
    assert not gap_safe_zeros(Z, y, theta, 0.125, 1.0)[0]


def test_elastic_net_conjugate_and_weak_duality_against_decimal():
    Z = np.array([[1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [1.0, 1.0]])
    y, theta = np.array([0.0, 1.0, 0.0, 1.0]), np.array([0.0, 1.0])
    cert = _gap_certificate(Z, y, theta, 0.2, 0.5, 2.0)
    with decimal.localcontext() as ctx:
        ctx.prec = 80
        D = decimal.Decimal

        def exact(f):
            return D(f.numerator) / D(f.denominator)

        a, lam, pf = D(0.5), D(0.2), D(2)
        l1, l2 = lam * pf * a, lam * pf * (1 - a)
        score = abs(exact(cert.scores[0]))
        assert score > l1  # omitting the conjugate must change the answer
        conjugate = (score - l1) ** 2 / (2 * l2)
        entropy = D(0)
        for yi, u in zip(y, cert.point, strict=True):
            q = D(int(yi)) + 4 * exact(u)
            entropy -= q * q.ln() + (1 - q) * (1 - q).ln()
        dual = entropy / 4 - conjugate
        primal = (1 + D(1).exp()).ln() - D(0.5) + l1 + l2 / 2
        assert D(cert.dual) <= dual <= primal <= D(cert.primal)
        assert D(cert.gap) >= primal - dual
        assert cert.rounding < 1e-10


def test_ridge_distance_is_coefficient_only_and_uses_the_weakest_penalty():
    # mu=lambda*(1-alpha)*min(pf)=1/4. A gap of 1/8 gives radius 1.
    radius = gap_coefficient_distance(0.125, 0.5, 0.5, [1.0, 2.0])
    assert 1 <= radius <= 1 + 2 * np.finfo(float).eps
    assert gap_coefficient_distance(0, 0.5, 0.5) == 0
    with pytest.raises(ValueError, match="alpha < 1"):
        gap_coefficient_distance(0.125, 0.5, 1.0)


@pytest.mark.parametrize("sign", [-1, 1])
def test_gap_encloses_loss_at_extreme_logits(sign):
    Z, y = np.array([[1.0, -1.0], [1.0, 1.0]]), np.array([0.0, 1.0])
    theta = np.array([0.0, sign * 1000.0])
    cert = _gap_certificate(Z, y, theta, 0.001, 1.0, 1.0)
    # The penalty is exactly 1000*lambda on the coefficients supplied. In the
    # wrong direction, the mean NLL additionally exceeds 1000; in the right
    # direction, a trivial feasible dual u=0 leaves the penalty in the gap.
    floor = float(Fraction(1000) * Fraction(0.001))
    assert np.isfinite(cert.gap)
    assert cert.primal >= floor + (1000 if sign < 0 else 0)
    assert cert.gap > 0


def test_gap_and_radius_do_not_depend_on_callers_decimal_context():
    Z, y, theta = (
        np.array([[1.0, -1.0], [1.0, 1.0]]),
        np.array([0.0, 1.0]),
        np.array([0.3, 0.2]),
    )
    before = duality_gap(Z, y, theta, 0.1, 0.5)
    radius = gap_coefficient_distance(before, 0.1, 0.5)
    with decimal.localcontext() as ctx:
        ctx.prec = 6
        ctx.rounding = decimal.ROUND_FLOOR
        ctx.traps[decimal.Inexact] = True
        assert duality_gap(Z, y, theta, 0.1, 0.5) == before
        assert gap_coefficient_distance(before, 0.1, 0.5) == radius


@pytest.mark.parametrize("copies", [1, 4])
def test_gap_safe_radius_cannot_screen_a_known_nonzero(monkeypatch, copies):
    # A near-lambda_max lasso optimum is beta=log((1-lambda)/lambda)>0.
    # Supply another feasible dual center, q=(.49,.51), rather than the
    # constructor's near-optimal center. Shrinking the sphere by sqrt(2)
    # wrongly screens this nonzero. Replication leaves the mean objective,
    # scores and gap unchanged; the column norm grows by sqrt(copies) and
    # the correct radius shrinks by sqrt(copies). n=8 also distinguishes 2n
    # from n**2, which coincide at n=2 and would undersize the sphere here.
    from types import SimpleNamespace

    Z = np.tile(np.array([[1.0, -1.0], [1.0, 1.0]]), (copies, 1))
    y, lam = np.tile(np.array([0.0, 1.0]), copies), 0.499
    theta = np.array([0.0, np.log((1 - lam) / lam)])
    assert theta[1] > 0
    with decimal.localcontext() as ctx:
        ctx.prec = 80
        D = decimal.Decimal
        b, penalty, q = D(theta[1]), D(lam), D(49) / 100
        primal = (1 + (-b).exp()).ln() + penalty * b
        dual = -q * q.ln() - (1 - q) * (1 - q).ln()
        gap = np.nextafter(float(primal - dual), np.inf)
    witness = SimpleNamespace(gap=gap, scores=(Fraction(-49, 100),))
    monkeypatch.setattr(certificates, "_gap_certificate", lambda *_: witness)
    assert not gap_safe_zeros(Z, y, theta, lam, 1.0)[0]


@pytest.mark.parametrize("pf", [0, [-1], [np.nan]])
def test_zero_or_invalid_penalties_are_not_silently_certified(pf):
    Z = np.array([[1.0, -1.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="strictly positive"):
        duality_gap(Z, np.array([0.0, 1.0]), np.zeros(2), 0.1, 0.5, pf)
    with pytest.raises(ValueError, match="strictly positive"):
        gap_coefficient_distance(1e-8, 0.1, 0.5, pf)


def wide_problem(design):
    """Repeated predictor rows with overlapping labels; no separation or full
    column rank. The two constructions also include exact duplicate columns."""
    if design == "walsh":
        base = hadamard(8)[:, 1:].astype(float)
        X = np.repeat(np.tile(base, (1, 6)), 4, axis=0)  # 32 x 42
        counts = np.array([1, 1, 1, 1, 3, 3, 3, 3])
    else:
        rng = np.random.default_rng(19)
        base = rng.normal(size=(12, 40))
        X = np.repeat(np.tile(base, (1, 2)), 4, axis=0)  # 48 x 80
        counts = 1 + (base[:, 0] > 0).astype(int) + (base[:, 1] > 0)
    y = np.concatenate([np.arange(4) < count for count in counts]).astype(float)
    pf = np.resize(np.array([0.5, 1.0, 2.0]), X.shape[1])
    return np.asfortranarray(X), y, pf


def fit_wide(design, alpha):
    X, y, pf = wide_problem(design)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        path = bl.biglasso(
            X,
            y,
            alpha=alpha,
            penalty_factor=pf,
            backend="rust",
            family="binomial",
            **FIT_OPTIONS,
        )
    assert len(path) == len(FIT_OPTIONS["lambdas"])
    np.testing.assert_array_equal(path.lambdas, FIT_OPTIONS["lambdas"])
    assert path.alpha == alpha
    np.testing.assert_array_equal(path.penalty_factor, pf)
    Xs, center, scale = standardize(X)
    return path, with_ones(Xs), center, scale, y, pf


@pytest.mark.parametrize("alpha", [1.0, 0.5, 0.9])
@pytest.mark.parametrize("design", ["walsh", "random"])
def test_wide_path_has_small_global_gap_and_gap_safe_zeros(design, alpha):
    path, Z, center, scale, y, pf = fit_wide(design, alpha)
    assert Z.shape[1] - 1 > len(y)
    assert np.linalg.matrix_rank(Z) < Z.shape[1]
    certified_zeros = 0
    for k, lam in enumerate(path.lambdas):
        theta = standardized_theta(
            path.intercept[k], path.coef[:, [k]].toarray().ravel(), center, scale
        )
        assert (
            np.max(abs(Z @ theta)) < 10
        )  # accuracy domain, not needed for weak duality
        assert standardized_kkt(Z, y, theta, lam, alpha, pf) < KKT_TOL
        assert duality_gap(Z, y, theta, lam, alpha, pf) < GAP_TOL
        safe = gap_safe_zeros(Z, y, theta, lam, alpha, pf)
        # A safe zero is a property of the exact optimum. For this approximate
        # iterate, its own likelihood score also clears L1 by more than its
        # independently capped KKT residual; a nonzero would contradict KKT.
        gradient = (Z.T @ (expit(Z @ theta) - y) / len(y))[1:]
        assert np.all(lam * alpha * pf[safe] - abs(gradient[safe]) > KKT_TOL)
        assert np.all(theta[1:][safe] == 0)
        certified_zeros += np.sum(safe)
    assert certified_zeros > 0 and path.coef.nnz > 0


@pytest.mark.parametrize("alpha", [0.5, 0.9])
@pytest.mark.parametrize("design", ["walsh", "random"])
def test_wide_coefficients_agree_with_reference_within_ridge_gap_radii(design, alpha):
    # The reference (support/reference.py) sums in another order, with `@`;
    # each fit's gap is certified on its own, so that order doesn't enter.
    a, Z, center, scale, y, pf = fit_wide(design, alpha)
    X, _, _ = wide_problem(design)
    options = {k: v for k, v in FIT_OPTIONS.items() if k != "n_threads"}
    b = reference_path.fit(X, y, alpha=alpha, penalty_factor=pf, **options)
    for k, lam in enumerate(a.lambdas):
        ta, tb = (
            standardized_theta(
                p.intercept[k], p.coef[:, [k]].toarray().ravel(), center, scale
            )
            for p in (a, b)
        )
        ga, gb = (duality_gap(Z, y, t, lam, alpha, pf) for t in (ta, tb))
        assert max(ga, gb) < GAP_TOL
        radius = gap_coefficient_distance(
            ga, lam, alpha, pf
        ) + gap_coefficient_distance(gb, lam, alpha, pf)
        assert np.linalg.norm(ta[1:] - tb[1:]) <= radius


def test_global_gap_rejects_nonconverged_wide_iterate():
    X, y, pf = wide_problem("walsh")
    with pytest.warns(UserWarning, match="Failed to converge"):
        path = bl.biglasso(
            X,
            y,
            alpha=0.5,
            penalty_factor=pf,
            backend="rust",
            family="binomial",
            **dict(FIT_OPTIONS, max_iter=1),
        )
    Xs, center, scale = standardize(X)
    theta = standardized_theta(
        path.intercept[-1], path.coef[:, [-1]].toarray().ravel(), center, scale
    )
    assert duality_gap(with_ones(Xs), y, theta, path.lambdas[-1], 0.5, pf) > GAP_TOL
