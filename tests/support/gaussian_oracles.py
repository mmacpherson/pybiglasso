"""Small Gaussian optima from stationarity systems, without solver kernels.

All parameters use independently standardized columns and a free intercept.
Enumeration is deliberately exponential and restricted to at most four slopes.
"""

from itertools import product

import numpy as np

from support.certificates import (
    KKT_TOL,
    gaussian_duality_gap,
    gaussian_subgradient_residual,
)

# A declared objective accuracy target for these small, O(1)-scaled problems.
# This is independent of the production stopping rule and its observed output.
GAP_TOL = 1e-10


def objective(Z, y, theta, lam, alpha, pf):
    beta = theta[1:]
    return np.mean((Z @ theta - y) ** 2) / 2 + lam * (
        alpha * (pf @ abs(beta)) + (1 - alpha) * (pf @ beta**2) / 2
    )


def assert_optimal(Z, y, theta, lam, alpha, pf):
    """Cap both stationarity and an outward-rounded rational primal/dual gap."""
    residual = gaussian_subgradient_residual(Z, y, theta, lam, alpha, pf)
    assert np.max(abs(residual)) <= KKT_TOL
    gap = gaussian_duality_gap(Z, y, theta, lam, alpha, pf)
    assert gap <= GAP_TOL
    return gap


def enumerate_optimum(Z, y, lam, alpha, pf):
    """Enumerate absent/positive/negative slopes and check every KKT equation.

    On a signed active set A, solve
      (Z_A' Z_A/n + diag(0, lambda*(1-alpha)*pf_A)) theta_A
        = Z_A' y/n - (0, lambda*alpha*pf_A*sign_A).
    The intercept is always present; no screening, cached score or warm start.
    A least-squares solve also admits redundant active columns. KKT, rather
    than an arbitrary coefficient representation, decides feasibility.
    """
    p = Z.shape[1] - 1
    assert 0 < p <= 4 and np.all(np.asarray(pf) > 0)
    candidates = []
    gram, score = Z.T @ Z / len(y), Z.T @ y / len(y)
    # These systems have order <=5. This conservative allowance covers double
    # precision formation/solve error; it is checked again by exact-gap algebra.
    oracle_kkt = 1e-10
    for state in product((-1, 0, 1), repeat=p):
        signs = np.asarray(state)
        active = np.flatnonzero(signs)
        cols = np.r_[0, active + 1]
        hessian = gram[np.ix_(cols, cols)] + np.diag(
            np.r_[0, lam * (1 - alpha) * pf[active]]
        )
        rhs = score[cols] - np.r_[0, lam * alpha * pf[active] * signs[active]]
        solution = np.linalg.lstsq(hessian, rhs, rcond=None)[0]
        if np.any(solution[1:] * signs[active] <= 0):
            continue
        theta = np.zeros(p + 1)
        theta[cols] = solution
        residual = gaussian_subgradient_residual(Z, y, theta, lam, alpha, pf)
        if np.max(abs(residual)) <= oracle_kkt:
            assert_optimal(Z, y, theta, lam, alpha, pf)
            candidates.append(theta)
    assert candidates, "no sign/active set satisfied the independent KKT check"
    return min(candidates, key=lambda b: objective(Z, y, b, lam, alpha, pf))
