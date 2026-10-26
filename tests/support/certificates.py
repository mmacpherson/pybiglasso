"""Independent numerical checks for logistic and Gaussian elastic-net objectives.

These helpers use numpy/scipy and the mathematical problem, not solver kernels.
KKT is not an accuracy promise for clipped, saturated or unconverged fits, or
for R-compatible elastic-net updates. Callers state and check that domain.
Curvature includes the unpenalized intercept. Its eigensolver margin is a
practical rounding allowance, not an interval-arithmetic certificate.
"""

import decimal
import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.special import expit

# On gradients of the mean NLL. With eps=1e-14 the solver's stopping measure
# v * shift**2 has natural gradient scale sqrt(eps), not eps. This is the
# pre-existing independent accuracy cap; extraction must not relax it.
KKT_TOL = 1e-6


def standardize(X):
    """Population mean and sd, independently of the solver (small test arrays)."""
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


def gaussian_subgradient_residual(Z, y, theta, lam, alpha, penalty_factor=1.0):
    """Minimum-norm subgradient of RSS/(2n) plus the standardized penalty."""
    theta = np.asarray(theta)
    batch = theta.ndim == 2
    response = y[:, None] if batch else y
    gradient = Z.T @ (Z @ theta - response) / len(y)
    pf = np.asarray(penalty_factor)
    if batch and pf.ndim:
        pf = pf[:, None]
    beta = theta[1:]
    l1 = lam * alpha * pf
    residual = gradient.copy()
    residual[1:] += lam * (1 - alpha) * pf * beta + np.where(
        beta != 0, l1 * np.sign(beta), np.clip(-gradient[1:], -l1, l1)
    )
    return residual


def gaussian_curvature(Z, lam=0.0, alpha=1.0, penalty_factor=1.0):
    """Least eigenvalue of the constant full Hessian, including the intercept.

    As with segment_curvature, subtract a relative eigensolver allowance;
    this distance check uses floating-point linear algebra, not intervals.
    Ridge can supply curvature even when X is wide.
    """
    hessian = Z.T @ Z / len(Z)
    ridge = np.broadcast_to(
        lam * (1 - alpha) * np.asarray(penalty_factor), Z.shape[1] - 1
    )
    hessian += np.diag(np.r_[0.0, ridge])
    eigenvalues = np.linalg.eigvalsh(hessian)
    lower = eigenvalues.min() - 1e-12 * np.abs(eigenvalues).max()
    assert np.isfinite(lower) and lower > 0, "no curvature to certify a distance"
    return lower


def subgradient_residual(Z, y, theta, lam, alpha, penalty_factor=1.0, *, eta=None):
    """A minimum-norm subgradient of mean NLL + elastic-net penalty.

    theta includes the intercept and may be a vector or a column per lambda.
    At zero, choose the L1 subgradient nearest the negative NLL gradient; it is
    valid for both the max-norm KKT cap and the 2-norm distance bound. An eta
    supplied by the caller preserves original-X prediction arithmetic (notably
    on large-offset regression fixtures); it must describe the same theta.
    """
    theta = np.asarray(theta)
    batch = theta.ndim == 2
    response = y[:, None] if batch else y
    eta = Z @ theta if eta is None else eta
    gradient = Z.T @ (expit(eta) - response) / len(y)
    beta = theta[1:]
    pf = np.asarray(penalty_factor)
    if batch and pf.ndim:
        pf = pf[:, None]
    l1 = lam * alpha * pf
    l2 = lam * (1 - alpha) * pf
    residual = gradient.copy()
    residual[1:] += l2 * beta + np.where(
        beta != 0,
        l1 * np.sign(beta),
        np.clip(-gradient[1:], -l1, l1),
    )
    return residual


def standardized_kkt(Z, y, theta, lam, alpha, penalty_factor=1.0):
    """Largest KKT violation, including the unpenalized intercept."""
    return np.max(np.abs(subgradient_residual(Z, y, theta, lam, alpha, penalty_factor)))


def kkt_residual(path, X, y, k):
    """Largest KKT violation at one lambda, checking standardization as well."""
    Xs, center, scale = standardize(X)
    np.testing.assert_allclose(path.center, center, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(path.scale, scale, rtol=1e-10)
    coef = path.coef[:, [k]].toarray().ravel()
    theta = standardized_theta(path.intercept[k], coef, center, scale)
    eta = path.intercept[k] + X @ coef
    return np.max(
        np.abs(
            subgradient_residual(
                with_ones(Xs),
                y,
                theta,
                path.lambdas[k],
                path.alpha,
                path.penalty_factor,
                eta=eta,
            )
        )
    )


def objective(X, y, intercept, coef, scale, pf, lam, alpha):
    """Mean NLL plus the penalty on standardized coefficients, from scratch."""
    eta = intercept + X @ coef
    nll = np.logaddexp(0, np.where(y == 1, -eta, eta)).mean()
    beta = coef * scale
    penalty = np.dot(pf, alpha * abs(beta) + (1 - alpha) * beta**2 / 2)
    return nll + lam * penalty


def path_objective(path, X, y, k):
    return objective(
        X,
        y,
        path.intercept[k],
        path.coef[:, [k]].toarray().ravel(),
        path.scale,
        path.penalty_factor,
        path.lambdas[k],
        path.alpha,
    )


def logistic_weight(eta):
    """p(1-p), without cancellation when p rounds to one."""
    u = np.exp(-np.abs(eta))
    return u / (1 + u) ** 2


def segment_curvature(Z, y, theta1, theta2):
    """Lower bound on mean NLL curvature along the entire endpoint segment.

    A row's logistic weight decreases with |eta|, so its minimum on a segment
    is at an endpoint. Z' diag(minimum weights) Z / n bounds every Hessian on
    that segment from below. The relative eigensolver margin is retained from
    the original helper; it is not interval arithmetic.
    """
    weight = np.minimum(logistic_weight(Z @ theta1), logistic_weight(Z @ theta2))
    floor = Z.T @ (Z * weight[:, None]) / len(y)
    eigenvalues = np.linalg.eigvalsh(floor)
    mu = eigenvalues.min() - 1e-12 * np.abs(eigenvalues).max()
    assert np.isfinite(mu) and mu > 0, "no curvature to certify a distance"
    return mu


def certified_distance(Z, y, theta1, theta2, e1, e2):
    """Distance bound for one convex objective, given subgradient 2-norms."""
    return (e1 + e2) / segment_curvature(Z, y, theta1, theta2)


def gradient_lipschitz(Z):
    """Mean NLL gradient Lipschitz constant ||Z||_2**2 / (4n)."""
    return np.linalg.norm(Z, 2) ** 2 / (4 * len(Z))


def support_is_certified(
    beta,
    gradient,
    l1,
    *,
    distance,
    residual_bound,
    lipschitz,
    rounding=0.0,
):
    """Whether another approximate solution must have these signs.

    Callers first cap both fits' residuals and bound their distance. For zeros,
    require L1 slack > the other fit's residual bound + Lipschitz * distance;
    for nonzeros, |beta| > distance. lipschitz may include L2's coefficientwise
    contribution where the caller uses that more conservative allowance.
    The intercept belongs in the distance and gradient bound, not in beta.
    """
    zero = beta == 0
    slack = l1 - np.abs(gradient)
    resolved = np.where(
        zero,
        slack > residual_bound + lipschitz * distance + rounding,
        np.abs(beta) > distance,
    )
    return bool(np.all(resolved))


def path_theta(path, k, center, scale):
    """A path's fit at index k, on standardized columns."""
    coef = path.coef[:, [k]].toarray().ravel()
    return standardized_theta(path.intercept[k], coef, center, scale)


def assert_same_optimum(
    Z, y, theta_a, theta_b, lam, alpha, penalty_factor=1.0, *, cap=KKT_TOL
):
    """Two approximate solutions of one problem, certified close.

    The problem is the caller's, as requested (lam, alpha and penalty_factor
    from the inputs, not from a fit's metadata). Both points must pass its KKT
    conditions within cap; then convexity bounds their distance by
    certified_distance from the residuals' 2-norms. That bound holds for any
    two points, so the caps carry the teeth: a fit of another problem fails
    them. KKT_TOL assumes eps=1e-14. Returns (distance, bound). Inputs are
    promoted to float64, the arithmetic residual_rounding assumes.
    """
    Z, y, theta_a, theta_b = (
        np.asarray(v, dtype=np.float64) for v in (Z, y, theta_a, theta_b)
    )
    residuals = [
        subgradient_residual(Z, y, t, lam, alpha, penalty_factor)
        for t in (theta_a, theta_b)
    ]
    assert max(np.abs(r).max() for r in residuals) < cap
    distance = np.linalg.norm(np.asarray(theta_a) - np.asarray(theta_b))
    # the true residuals, not the computed ones, bound the distance: near an
    # optimum (the null model at lambda_max, say) rounding is all there is
    norms = (
        np.linalg.norm(r)
        + np.linalg.norm(residual_rounding(Z, y, t, lam, alpha, penalty_factor))
        for r, t in zip(residuals, (theta_a, theta_b), strict=True)
    )
    bound = certified_distance(Z, y, theta_a, theta_b, *norms)
    assert distance <= bound
    return distance, bound


def residual_rounding(Z, y, theta, lam, alpha, penalty_factor=1.0):
    """Elementwise bound on the rounding error in subgradient_residual.

    The standard model, unit roundoff u and gamma_k = k u / (1 - k u)
    (Higham, Accuracy and Stability, chapter 3): eta = Z theta is within
    gamma_m |Z| |theta|; expit, 1/4-Lipschitz, passes a quarter of that on
    and adds at most 4 u; p - y adds one rounding; the mean of n products
    gamma_(n+1) of its magnitudes; the penalty terms three roundings each.
    These are first-order terms; doubling covers the higher-order rest. u is
    float64's, so inputs are promoted to float64, as subgradient_residual
    then computes in it too. An allowance for finite, well-scaled problems
    (no overflow or underflow), not an interval enclosure.
    """
    Z, y, theta = (np.asarray(v, dtype=np.float64) for v in (Z, y, theta))
    unit = np.finfo(np.float64).eps / 2

    def gamma(k):
        return k * unit / (1 - k * unit)

    n, m = Z.shape
    absZ = np.abs(Z)
    score = expit(Z @ theta) - y
    score_error = (
        gamma(m) * (absZ @ np.abs(theta)) / 4 + 4 * unit + unit * np.abs(score)
    )
    error = (gamma(n + 1) * (absZ.T @ np.abs(score)) + absZ.T @ score_error) / n
    pf = np.asarray(penalty_factor)
    error[1:] += 3 * unit * lam * pf * (alpha + (1 - alpha) * np.abs(theta[1:]))
    return 2 * error


@dataclass(frozen=True)
class _GapCertificate:
    primal: float
    dual: float
    gap: float
    point: tuple[Fraction, ...]  # exactly feasible dual point
    scores: tuple[Fraction, ...]  # X'u, evaluated exactly
    rounding: float


def _gap_parameters(lam, alpha, penalty_factor, p):
    pf = np.broadcast_to(np.asarray(penalty_factor, dtype=float), (p,))
    if not np.isfinite(lam) or lam <= 0 or not 0 < alpha <= 1:
        raise ValueError("gap certificates need lambda > 0 and 0 < alpha <= 1")
    if not np.isfinite(pf).all() or np.any(pf <= 0):
        # A zero penalty imposes another exact dual equality X_j'u=0. Merely
        # balancing the intercept does not enforce it; do not certify that case.
        raise ValueError("gap certificates need strictly positive penalty factors")
    penalty = tuple(Fraction(float(lam)) * Fraction(float(f)) for f in pf)
    a = Fraction(float(alpha))
    return tuple(v * a for v in penalty), tuple(v * (1 - a) for v in penalty)


def _fraction_float(value, *, upper):
    """One-sided conversion, for algebraic bounds (not transcendental calls)."""
    rounded = float(value)
    wrong_side = Fraction(rounded) < value if upper else Fraction(rounded) > value
    if wrong_side:
        rounded = math.nextafter(rounded, math.inf if upper else -math.inf)
    return rounded


def _decimal_fraction(value, rounding):
    with decimal.localcontext() as ctx:
        ctx.rounding = rounding
        return decimal.Decimal(value.numerator) / decimal.Decimal(value.denominator)


def _softplus_upper(value):
    """Upper bound at an exact rational logit; caller rounds algebra upward."""
    z = _decimal_fraction(value, decimal.ROUND_CEILING)
    # exp/ln are correctly rounded (half-even) independently of the context's
    # rounding mode. One next_plus encloses their exact value. The surrounding
    # monotone algebra is rounded upward; the stable form avoids exp(large z).
    small_exp = (-abs(z)).exp().next_plus()
    return max(z, 0) + (1 + small_exp).ln().next_plus()


def _entropy_term_lower(value):
    """Lower bound on -v*log(v) for an exact rational v in [0,1]."""
    if value == 0 or value == 1:
        return decimal.Decimal(0)
    lo = _decimal_fraction(value, decimal.ROUND_FLOOR)
    hi = _decimal_fraction(value, decimal.ROUND_CEILING)
    # v>=lo and -log(v)>=-log(hi)>=0, so their product is a lower bound.
    # Evaluate both q and 1-q as Fractions: a rounded q must not erase its tail.
    log_hi = hi.ln().next_plus()
    return lo * max(-log_hi, 0)


def _sqrt_fraction_upper(value):
    if value == 0:
        return 0.0
    with decimal.localcontext(decimal.Context(prec=60)):
        z = _decimal_fraction(value, decimal.ROUND_CEILING)
        upper = z.sqrt().next_plus()
    return _fraction_float(Fraction(upper), upper=True)


def _gap_certificate(Z, y, theta, lam, alpha, penalty_factor):
    Z, y, theta = (np.asarray(a, dtype=float) for a in (Z, y, theta))
    if (
        Z.ndim != 2
        or Z.shape[0] == 0
        or Z.shape[1] < 2
        or y.shape != (len(Z),)
        or theta.shape != (Z.shape[1],)
        or not all(np.isfinite(a).all() for a in (Z, y, theta))
        or not np.all(Z[:, 0] == 1)
        or not np.all((y == 0) | (y == 1))
    ):
        raise ValueError("expected finite Z=[1,X], binary y and one parameter vector")
    n, d = Z.shape
    l1, l2 = _gap_parameters(lam, alpha, penalty_factor, d - 1)
    eta = Z @ theta

    # Balancing each sign by a factor <=1 preserves q=y+n*u in [0,1].
    # Fractions enforce sum(u)=0 exactly, not merely to a residual tolerance.
    # Unlike the solver, a small test oracle can afford exact dot products.
    residual = [
        Fraction(float(q)) - int(yi) for q, yi in zip(expit(eta), y, strict=True)
    ]
    pos = sum((r for r in residual if r > 0), Fraction())
    neg = -sum((r for r in residual if r < 0), Fraction())
    mass = min(pos, neg)
    point = tuple(
        r * mass / (pos if r > 0 else neg) / n if r else Fraction() for r in residual
    )
    exact_Z = [tuple(Fraction(float(x)) for x in row) for row in Z]
    columns = tuple(zip(*(row[1:] for row in exact_Z), strict=True))
    scores = tuple(
        sum((x * u for x, u in zip(col, point, strict=True)), Fraction())
        for col in columns
    )
    if alpha == 1:
        # Lasso conjugacy is a hard constraint, not a soft penalty. A common
        # exact contraction preserves both intercept balance and the box.
        factor = min(
            [
                Fraction(1),
                *(bound / abs(s) for bound, s in zip(l1, scores, strict=True) if s),
            ]
        )
        point = tuple(u * factor for u in point)
        scores = tuple(s * factor for s in scores)
        assert all(abs(s) <= b for s, b in zip(scores, l1, strict=True))
        conjugate = Fraction()
    else:
        # (l1*|b| + l2*b^2/2)^*(s) = (|s|-l1)_+^2/(2*l2).
        conjugate = sum(
            (
                max(abs(s) - a, 0) ** 2 / (2 * b)
                for s, a, b in zip(scores, l1, l2, strict=True)
            ),
            Fraction(),
        )
    assert sum(point, Fraction()) == 0
    q = tuple(int(yi) + n * u for yi, u in zip(y, point, strict=True))
    assert all(0 <= v <= 1 for v in q)
    exact_theta = tuple(Fraction(float(v)) for v in theta)
    exact_eta = tuple(
        sum((x * b for x, b in zip(row, exact_theta, strict=True)), Fraction())
        for row in exact_Z
    )
    penalty = sum(
        (
            a * abs(b) + c * b**2 / 2
            for a, c, b in zip(l1, l2, exact_theta[1:], strict=True)
        ),
        Fraction(),
    )
    # Only the dual proposal uses binary64 expit. It need not be accurate: its
    # feasibility was checked exactly. Bound the *actual objective* separately
    # with directed Decimal algebra and enclosing exp/ln calls at 60 digits.
    # No fitted result, arbitrary epsilon floor or empirical tolerance enters.
    with decimal.localcontext(decimal.Context(prec=60)) as ctx:
        ctx.rounding = decimal.ROUND_CEILING
        losses = [
            _softplus_upper(-z if yi else z) for yi, z in zip(y, exact_eta, strict=True)
        ]
        primal_up = sum(losses, decimal.Decimal(0)) / n + _decimal_fraction(
            penalty, decimal.ROUND_CEILING
        )
        ctx.rounding = decimal.ROUND_FLOOR
        entropy = sum(
            (_entropy_term_lower(v) + _entropy_term_lower(1 - v) for v in q),
            decimal.Decimal(0),
        )
        dual_lo = entropy / n - _decimal_fraction(conjugate, decimal.ROUND_CEILING)
    primal = _fraction_float(Fraction(primal_up), upper=True)
    dual = _fraction_float(Fraction(dual_lo), upper=False)
    gap = _fraction_float(Fraction(primal) - Fraction(dual), upper=True)
    rounding = math.ulp(primal) + math.ulp(dual)
    assert np.isfinite(gap) and gap >= 0, "invalid primal-dual gap"
    return _GapCertificate(primal, dual, gap, point, scores, rounding)


def duality_gap(Z, y, theta, lam, alpha, penalty_factor=1.0):
    """Upper bound on objective suboptimality, in mean NLL units (nats/row).

    Z=[1, standardized X], theta=[intercept, standardized beta]. Positive
    lambda and penalty factors, 0<alpha<=1; zero-penalty nuisance columns need
    extra dual equalities and are deliberately rejected. A valid bound does
    not require the fit to converge, remain unclipped, or have n>p.

    The dual is -mean(h(q)) - sum_j (|X_j'u|-l1_j)_+^2/(2*l2_j), with
    q=y+n*u in [0,1] and sum(u)=0; for l2=0 enforce |X_j'u|<=l1_j.
    See Ndiaye et al. (2017), Theorem 6 and Section 5.4, for the logistic
    conjugate/gap construction: https://jmlr.org/papers/v18/16-577.html.
    The elastic-net conjugate follows by maximizing one scalar quadratic.
    Linear constraints are exact rational arithmetic; objective bounds use
    directed Decimal algebra and correctly-rounded exp/ln, then outward
    binary64 conversion. This oracle is for small test designs, not production.
    """
    return _gap_certificate(Z, y, theta, lam, alpha, penalty_factor).gap


def gap_coefficient_distance(gap, lam, alpha, penalty_factor=1.0):
    """Bound ||beta-beta*||_2 by sqrt(2*gap/min_j(l2_j)), for alpha<1.

    Ridge supplies curvature on coefficients even when p>n. This is NOT a
    bound on theta: its intercept is unpenalized. Two fits of one objective
    differ in coefficients by at most the sum of their two radii.
    """
    pf = np.atleast_1d(penalty_factor)
    _, l2 = _gap_parameters(lam, alpha, pf, len(pf))
    if alpha == 1 or not np.isfinite(gap) or gap < 0:
        raise ValueError(
            "coefficient distance needs alpha < 1 and a finite nonnegative gap"
        )
    if gap == 0:
        return 0.0
    squared = Fraction(2) * Fraction(float(gap)) / min(l2)
    return _sqrt_fraction_upper(squared)


def gap_safe_zeros(Z, y, theta, lam, alpha, penalty_factor=1.0):
    """Features zero at EVERY optimum, from a logistic gap-safe dual sphere.

    Mean logistic loss has conjugate curvature 4n in u=(q-y)/n, so the dual
    optimum lies within sqrt(gap/(2n)). If |X_j'u|+||X_j||*radius < l1_j,
    KKT forces beta*_j=0, also for elastic net. This certifies the optimum,
    not literal zeros in arbitrary approximate iterates. Comparisons are
    strict; the algebraic conversions/products are rounded outward.
    """
    cert = _gap_certificate(Z, y, theta, lam, alpha, penalty_factor)
    l1, _ = _gap_parameters(lam, alpha, penalty_factor, Z.shape[1] - 1)
    r2 = Fraction(float(cert.gap)) / (2 * len(y))
    radius = _sqrt_fraction_upper(r2)
    safe = []
    for col, score, threshold in zip(Z[:, 1:].T, cert.scores, l1, strict=True):
        norm2 = sum((Fraction(float(x)) ** 2 for x in col), Fraction())
        norm = _sqrt_fraction_upper(norm2)
        uncertainty = math.nextafter(radius * norm, math.inf)
        upper = math.nextafter(
            _fraction_float(abs(score), upper=True) + uncertainty, math.inf
        )
        safe.append(upper < _fraction_float(threshold, upper=False))
    return np.array(safe)


def _gaussian_gap_certificate(Z, y, theta, lam, alpha, penalty_factor):
    """Exact rational primal/dual algebra for small least-squares test data.

    With u=(y-prediction)/n, the dual is y'u-n||u||²/2-g*(X'u).
    The intercept imposes sum(u)=0. Lasso additionally constrains |X'u|<=l1;
    elastic net instead subtracts the soft-threshold quadratic conjugate.
    Positive penalty factors only, as for the logistic gap oracle.
    """
    Z, y, theta = np.asarray(Z), np.asarray(y), np.asarray(theta)
    if Z.ndim != 2 or y.shape != (len(Z),) or theta.shape != (Z.shape[1],):
        raise ValueError("expected Z=(n,p+1), y=(n,), theta=(p+1,)")
    if not np.all(Z[:, 0] == 1) or not all(np.isfinite(v).all() for v in (Z, y, theta)):
        raise ValueError("finite inputs and an intercept column of ones required")
    l1, l2 = _gap_parameters(lam, alpha, penalty_factor, Z.shape[1] - 1)
    design = [tuple(Fraction(float(x)) for x in row) for row in Z]
    response = tuple(Fraction(float(v)) for v in y)
    coefficients = tuple(Fraction(float(v)) for v in theta)
    residual = tuple(
        yi - sum((x * b for x, b in zip(row, coefficients, strict=True)), Fraction())
        for row, yi in zip(design, response, strict=True)
    )
    n = len(y)
    mean_residual = sum(residual, Fraction()) / n
    point = tuple((r - mean_residual) / n for r in residual)
    columns = tuple(zip(*(row[1:] for row in design), strict=True))
    scores = tuple(
        sum((x * u for x, u in zip(col, point, strict=True)), Fraction())
        for col in columns
    )
    if alpha == 1:
        contraction = min(
            [Fraction(1), *(a / abs(s) for a, s in zip(l1, scores, strict=True) if s)]
        )
        point = tuple(u * contraction for u in point)
        scores = tuple(s * contraction for s in scores)
        conjugate = Fraction()
        assert all(abs(s) <= a for s, a in zip(scores, l1, strict=True))
    else:
        conjugate = sum(
            (
                max(abs(s) - a, 0) ** 2 / (2 * b)
                for s, a, b in zip(scores, l1, l2, strict=True)
            ),
            Fraction(),
        )
    assert sum(point, Fraction()) == 0
    penalty = sum(
        (
            a * abs(b) + c * b * b / 2
            for a, c, b in zip(l1, l2, coefficients[1:], strict=True)
        ),
        Fraction(),
    )
    primal_exact = sum((r * r for r in residual), Fraction()) / (2 * n) + penalty
    dual_exact = (
        sum((yi * u for yi, u in zip(response, point, strict=True)), Fraction())
        - n * sum((u * u for u in point), Fraction()) / 2
        - conjugate
    )
    assert primal_exact >= dual_exact
    primal = _fraction_float(primal_exact, upper=True)
    dual = _fraction_float(dual_exact, upper=False)
    gap = _fraction_float(primal_exact - dual_exact, upper=True)
    return _GapCertificate(
        primal, dual, gap, point, scores, math.ulp(primal) + math.ulp(dual)
    )


def gaussian_duality_gap(Z, y, theta, lam, alpha, penalty_factor=1.0):
    """Outward-rounded upper bound on Gaussian objective suboptimality."""
    return _gaussian_gap_certificate(Z, y, theta, lam, alpha, penalty_factor).gap


def gaussian_gap_safe_zeros(Z, y, theta, lam, alpha, penalty_factor=1.0):
    """Zeros at every optimum, using the dual sphere of radius sqrt(2 gap/n)."""
    cert = _gaussian_gap_certificate(Z, y, theta, lam, alpha, penalty_factor)
    l1, _ = _gap_parameters(lam, alpha, penalty_factor, Z.shape[1] - 1)
    radius = _sqrt_fraction_upper(Fraction(2) * Fraction(float(cert.gap)) / len(y))
    safe = []
    for col, score, threshold in zip(Z[:, 1:].T, cert.scores, l1, strict=True):
        norm = _sqrt_fraction_upper(
            sum((Fraction(float(x)) ** 2 for x in col), Fraction())
        )
        uncertainty = math.nextafter(radius * norm, math.inf)
        upper = math.nextafter(
            _fraction_float(abs(score), upper=True) + uncertainty, math.inf
        )
        safe.append(upper < _fraction_float(threshold, upper=False))
    return np.array(safe)
