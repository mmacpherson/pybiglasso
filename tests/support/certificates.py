"""Independent numerical checks for the default logistic elastic-net objective.

These helpers use numpy/scipy and the mathematical problem, not solver kernels.
KKT is not an accuracy promise for clipped, saturated or unconverged fits, or
for R-compatible elastic-net updates. Callers state and check that domain.
Curvature includes the unpenalized intercept. Its eigensolver margin is a
practical rounding allowance, not an interval-arithmetic certificate.
"""

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
