"""Small, shrinkable numerical problems, with expectations from the objective.

Theorems: the objective identities, fixed-grid prefixes, and each backend's
thread-count independence. Across backends compare optimality and the certified
accuracy of the coefficients, including at exact ties. Different reduction orders
can move a strict screening or stopping comparison across its boundary, so
cross-backend iteration, rejection and exact-support equality is tested only on a
separate grid away from those ties.

KKT is tested only in default mode, on overlapping-label, well-conditioned
designs, with no saturation, exhausted iteration budget, or |eta| > 10 clamp.
It is not a universal property of every input the solver accepts. In particular
alpha=0 raises NotImplementedError; it is not an invalid-input ValueError.

No fit generates a grid, a target coefficient, or an acceptance tolerance here.
Metamorphic comparisons fit both sides of an identity specified in advance.
The segment-curvature bound is shared with test_invariances, including its
explicit allowance (not interval arithmetic) for floating-point eigensolving.
"""

import dataclasses
import warnings

import numba
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from rcompare import ATOL, RTOL
from scipy.linalg import hadamard
from scipy.special import expit
from test_default_enet import KKT_TOL
from test_invariances import (
    certified_distance,
    standardize,
    standardized_theta,
    with_ones,
)

import pybiglasso as bl

BACKENDS = ["numba", "rust"]
FIT_EPS = 1e-14
# The stopping measure is v*shift^2 < eps*null_deviance/n. Its natural gradient
# scale is sqrt(eps), not eps; use the independently established KKT_TOL=1e-6
# from test_default_enet, with the same eps=1e-14 and no relaxed acceptance.
FIT_OPTIONS = dict(eps=FIT_EPS, max_iter=10_000, n_threads=1)


@dataclasses.dataclass(frozen=True)
class SmoothProblem:
    """Walsh columns with modest dyadic shears, and both labels at every point.

    Shrinking removes columns, shears, offsets and repeated observations. Unlike
    an opaque PRNG seed, each field describes the numerical problem it changes.
    A shear of at most 1/4 has singular values between 3/4 and 5/4. After
    standardizing, the Gram eigenvalues are at least (3/4)^2/(17/16)=9/17;
    the curvature comparison therefore cannot become singular from collinearity.
    """

    order: int
    columns: int
    repeats: int
    ones: tuple[int, ...]
    shear: float
    offset: int
    alpha: float
    penalties: tuple[float, ...]
    grid: tuple[float, ...]

    def arrays(self):
        base = hadamard(self.order)[:, 1 : self.columns + 1].astype(float)
        mixing = np.eye(self.columns)
        for j in range(1, self.columns):
            mixing[j - 1, j] = self.shear
        X = np.repeat(base @ mixing + self.offset, self.repeats, axis=0)
        y = np.concatenate(
            [np.arange(self.repeats) < count for count in self.ones]
        ).astype(float)
        return np.asfortranarray(X), y

    def fit(self, backend, **overrides):
        X, y = self.arrays()
        options = dict(
            alpha=self.alpha,
            penalty_factor=self.penalties,
            lambdas=self.grid,
            backend=backend,
            **FIT_OPTIONS,
        )
        options.update(overrides)
        # Non-convergence and saturation are failures of this stated domain,
        # rather than an excuse to omit a lambda from the optimality checks.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # A one-column array is both C- and F-contiguous; np.isfortran
            # returns False there, so even asfortranarray gets this advice.
            warnings.filterwarnings(
                "ignore", message="The design matrix is not in Fortran"
            )
            path = bl.biglasso(X, y, **options)
        # Check metadata against inputs, so optimality cannot pass by silently
        # solving a problem with a different alpha, penalty or lambda grid.
        assert path.alpha == options["alpha"]
        assert_bits(
            path.penalty_factor, np.array(options["penalty_factor"], dtype=float)
        )
        assert len(path) == len(options["lambdas"])
        assert_bits(path.lambdas, np.array(options["lambdas"], dtype=float))
        return path


@st.composite
def smooth_problems(draw):
    order = draw(st.sampled_from([4, 8]))
    columns = draw(st.integers(1, min(6, order - 1)))
    repeats = draw(st.integers(2, min(5, 40 // order)))
    ones = tuple(
        draw(st.lists(st.integers(1, repeats - 1), min_size=order, max_size=order))
    )
    penalties = tuple(
        draw(
            st.lists(
                st.sampled_from([0.5, 1.0, 2.0]),
                min_size=columns,
                max_size=columns,
            )
        )
    )
    size = draw(st.integers(2, 5))
    return SmoothProblem(
        order,
        columns,
        repeats,
        ones,
        draw(st.sampled_from([0.0, 0.25, -0.25])),
        draw(st.integers(-2, 2)),
        draw(st.sampled_from([0.25, 0.5, 0.75, 1.0])),
        penalties,
        (0.5, 0.25, 0.125, 0.0625, 0.03125)[:size],
    )


# A non-null example prevents a collection of all-null shrunken problems from
# being the only exercised domain. Its targets are still computed from math.
SIGNAL = SmoothProblem(
    4, 2, 4, (1, 1, 3, 3), 0.25, 1, 0.5, (1.0, 0.5), (0.25, 0.125, 0.0625)
)

SCREENING_TIE = SmoothProblem(4, 1, 3, (1, 1, 1, 1), 0.0, 0, 0.25, (0.5,), (0.5, 0.25))
ITERATION_TIE = SmoothProblem(4, 1, 4, (2, 1, 3, 1), 0.0, 1, 0.75, (0.5,), (0.5, 0.25))
SUPPORT_TIE = SmoothProblem(
    8, 1, 3, (1, 1, 2, 1, 2, 1, 2, 1), 0.0, 1, 0.5, (0.5,), (0.5, 0.25)
)
# At lambda=1/4, the third column's score and L1 threshold are both 1/8.
# Different tie decisions alter the finite stopping iterate at the NEXT lambda.
STOPPING_TIE = SmoothProblem(
    8, 3, 3, (2, 1, 1, 2, 2, 1, 2, 2), 0.0, 1, 0.25, (0.5, 1.0, 2.0), (0.5, 0.25, 0.125)
)


def assert_bits(a, b):
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    assert a.tobytes() == b.tobytes()


def assert_path_bits(a, b, length=None, rows=None):
    length = len(a) if length is None else length
    assert len(a) == length
    for field in ("lambdas", "intercept", "deviance", "iterations", "n_rejected"):
        assert_bits(getattr(a, field), getattr(b, field)[:length])
    coef = b.coef.toarray()[:, :length]
    assert_bits(a.coef.toarray(), coef if rows is None else coef[rows])


def theta_and_residuals(path, X, y):
    """Independently recompute subgradient residuals, including the intercept.

    At zero choose the L1 subgradient nearest the likelihood gradient; this is
    a valid subgradient, so its 2-norm can enter the convex distance bound.
    """
    Xs, center, scale = standardize(X)
    Z = with_ones(Xs)
    # The smooth family has Gram floor >=9/17; the suppressor construction's
    # floor is 1-1/sqrt(5)>1/2, and the storage fixtures are orthogonal.
    assert np.linalg.eigvalsh(Z.T @ Z / len(y)).min() > 0.5
    theta = np.column_stack(
        [
            standardized_theta(
                path.intercept[k], path.coef[:, [k]].toarray().ravel(), center, scale
            )
            for k in range(len(path))
        ]
    )
    eta = Z @ theta
    assert np.max(np.abs(eta)) < 10, "outside the unclipped KKT domain"
    gradient = Z.T @ (expit(eta) - y[:, None]) / len(y)
    beta = theta[1:]
    l1 = path.lambdas * path.alpha * path.penalty_factor[:, None]
    l2 = path.lambdas * (1 - path.alpha) * path.penalty_factor[:, None]
    residual = gradient.copy()
    residual[1:] += l2 * beta + np.where(
        beta != 0,
        l1 * np.sign(beta),
        np.clip(-gradient[1:], -l1, l1),
    )
    assert np.max(np.abs(residual)) < KKT_TOL, (
        "KKT residual exceeds the fixed accuracy cap"
    )
    assert len(path) == theta.shape[1]
    return Z, theta, residual


def assert_certified_solutions(a, b, X, y):
    """Cap KKT before bounding the distance between routes to one optimum.

    Changing storage dtype or C/F layout changes LLVM's vectorized reduction
    order in numba. Those are objective identities, not bitwise identities.
    """
    assert len(a) == len(b)
    assert_bits(a.lambdas, b.lambdas)
    Z, ta, ra = theta_and_residuals(a, X, y)
    _, tb, rb = theta_and_residuals(b, X, y)
    for k in range(len(a)):
        ea, eb = np.linalg.norm(ra[:, k]), np.linalg.norm(rb[:, k])
        bound = certified_distance(Z, y, ta[:, k], tb[:, k], ea, eb)
        # <=40 rows and 7 parameters: a generous gamma_(64*n*d) allowance
        # for forming and comparing the two standardized parameter vectors.
        rounding = 64 * Z.size * np.finfo(float).eps * (1 + np.linalg.norm(ta[:, k]))
        assert np.linalg.norm(ta[:, k] - tb[:, k]) <= bound + rounding


def assert_backend_rounding_contract(a, b, X, y):
    assert_certified_solutions(a, b, X, y)
    # The cross-backend contract uses the project's existing R-agreement policy,
    # not tolerances inferred from these fits. The independent KKT cap and
    # segment bound above additionally check that both solve the stated problem.
    np.testing.assert_allclose(a.coef.toarray(), b.coef.toarray(), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(a.intercept, b.intercept, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(a.deviance, b.deviance, rtol=RTOL, atol=ATOL)


def independent_objective(X, y, intercept, beta, scale, pf, lam, alpha):
    """No pybiglasso routine or fitted path supplies this value."""
    eta = intercept + X @ beta
    nll = np.logaddexp(0, np.where(y == 1, -eta, eta)).mean()
    b = beta * scale
    penalty = np.dot(pf, alpha * abs(b) + (1 - alpha) * b**2 / 2)
    return nll + lam * penalty


@given(problem=smooth_problems(), exponent=st.integers(-3, 3))
@example(problem=SIGNAL, exponent=2)
def test_objective_identities(problem, exponent):
    """Rescale pf and lambda inversely; complement y and negate theta.

    Unlike a solver comparison, these identities hold at arbitrary coefficients.
    The coefficients below are dyadic constants, not a fit's output.
    """
    X, y = problem.arrays()
    beta = (np.arange(X.shape[1]) + 1) / 16
    scale = X.std(axis=0)
    pf = np.array(problem.penalties)
    lam, factor, intercept = problem.grid[-1], 2.0**exponent, 0.25
    value = independent_objective(X, y, intercept, beta, scale, pf, lam, problem.alpha)
    rescaled = independent_objective(
        X, y, intercept, beta, scale, pf * factor, lam / factor, problem.alpha
    )
    flipped = independent_objective(
        X, 1 - y, -intercept, -beta, scale, pf, lam, problem.alpha
    )
    # Powers of two are exact here; positive sum reductions of <=40 terms have
    # relative error <= gamma_40. Allow two evaluations and 16 scalar operations
    # per term, also covering the few-ulp exp/log implementations.
    tol = 2 * 16 * len(y) * np.finfo(float).eps
    np.testing.assert_allclose([rescaled, flipped], value, rtol=tol, atol=0)


@pytest.mark.parametrize("backend", BACKENDS)
@given(problem=smooth_problems(), exponent=st.integers(-3, 3), insert=st.integers(0, 6))
@example(problem=SIGNAL, exponent=2, insert=1)
def test_penalty_rescaling_and_constant_insertion(problem, exponent, insert, backend):
    """Supplied grids avoid changing the n>p default; removed columns cannot
    alter any used-column coordinate, even with a different raw column map.
    """
    fit = problem.fit(backend)
    factor = 2.0**exponent
    scaled = problem.fit(
        backend,
        penalty_factor=np.array(problem.penalties) * factor,
        lambdas=np.array(problem.grid) / factor,
    )
    # Compare all fields other than the known, inversely rescaled lambda grid.
    assert_bits(scaled.lambdas * factor, fit.lambdas)
    for field in ("intercept", "deviance", "iterations", "n_rejected"):
        assert_bits(getattr(fit, field), getattr(scaled, field))
    assert_bits(fit.coef.toarray(), scaled.coef.toarray())

    X, y = problem.arrays()
    at = min(insert, X.shape[1])
    padded = np.asfortranarray(np.insert(X, at, 7.0, axis=1))
    pf = np.insert(problem.penalties, at, 2.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        added = bl.biglasso(
            padded,
            y,
            alpha=problem.alpha,
            penalty_factor=pf,
            lambdas=problem.grid,
            backend=backend,
            **FIT_OPTIONS,
        )
    keep = np.delete(np.arange(padded.shape[1]), at)
    assert np.all(added.coef.toarray()[at] == 0)
    # Removing a constant changes a one-column array's C/F classification and
    # thus numba's reduction order. It can even move a zero-score screening
    # tie; the objective identity is what holds for all these problems.
    projected = dataclasses.replace(
        added,
        coef=added.coef[keep],
        center=added.center[keep],
        scale=added.scale[keep],
        penalty_factor=added.penalty_factor[keep],
        feature_names=tuple(added.feature_names[j] for j in keep),
    )
    assert_certified_solutions(fit, projected, X, y)


@pytest.mark.parametrize("backend", BACKENDS)
@given(problem=smooth_problems(), prefix=st.integers(1, 4))
@example(problem=SIGNAL, prefix=1)
def test_supplied_grid_prefix_is_exact(problem, prefix, backend):
    full = problem.fit(backend)
    size = min(prefix, len(problem.grid) - 1)
    short = problem.fit(backend, lambdas=problem.grid[:size])
    assert_path_bits(short, full, length=size)


@pytest.mark.parametrize("backend", BACKENDS)
@given(problem=smooth_problems(), compatible=st.booleans())
@example(problem=SIGNAL, compatible=True)
@example(problem=SCREENING_TIE, compatible=False)
@example(problem=SCREENING_TIE, compatible=True)
@example(problem=ITERATION_TIE, compatible=False)
@example(problem=ITERATION_TIE, compatible=True)
@example(problem=SUPPORT_TIE, compatible=False)
@example(problem=SUPPORT_TIE, compatible=True)
@example(problem=STOPPING_TIE, compatible=False)
@example(problem=STOPPING_TIE, compatible=True)
def test_threads_do_not_change_bits(problem, compatible, backend):
    threads = 2 if backend == "rust" else min(2, numba.config.NUMBA_NUM_THREADS)
    one = problem.fit(backend, r_compatible=compatible)
    many = problem.fit(backend, n_threads=threads, r_compatible=compatible)
    assert_path_bits(one, many)
    assert_bits(one.center, many.center)
    assert_bits(one.scale, many.scale)


@pytest.mark.parametrize("backend", BACKENDS)
@given(problem=smooth_problems())
@example(problem=SIGNAL)
def test_kkt_and_penalty_likelihood_order(problem, backend):
    """KKT plus residual-certified regularization order, only in default mode.

    For exact minimizers of L+lambda*P, adding the two optimality inequalities
    gives P_high<=P_low, then L_high>=L_low. At an approximate optimum x with
    a subgradient residual e, convexity gives F(x)-F(z)<=e'(x-z). Thus, with d
    the actual endpoint distance, the penalty inequality's slack is bounded
    by (|e_high|+|e_low|)*d/(lambda_high-lambda_low).
    """
    X, y = problem.arrays()
    fit = problem.fit(backend)
    assert len(fit) == len(problem.grid)
    Z, theta, residual = theta_and_residuals(fit, X, y)
    eta = Z @ theta
    nll = np.logaddexp(0, np.where(y[:, None] == 1, -eta, eta)).mean(axis=0)
    beta = theta[1:]
    pf = np.array(problem.penalties)[:, None]
    penalty = (
        pf * (problem.alpha * abs(beta) + (1 - problem.alpha) * beta**2 / 2)
    ).sum(axis=0)
    # Dot/log/sum evaluations in these <=40-by-7 well-scaled arrays: a generous
    # gamma_(64*n*d) budget, scaled by the positive objective terms it bounds.
    rounding = 64 * Z.size * np.finfo(float).eps * (1 + max(nll) + max(penalty))
    for k in range(len(fit) - 1):
        high, low = fit.lambdas[k : k + 2]
        distance = np.linalg.norm(theta[:, k] - theta[:, k + 1])
        errors = np.linalg.norm(residual[:, k : k + 2], axis=0)
        slack = errors.sum() * distance
        # These optimality inequalities use the actual endpoint distance; the
        # segment helper is for two solutions of the SAME objective below.
        assert penalty[k] <= penalty[k + 1] + (slack + rounding) / (high - low)
        assert (
            nll[k] >= nll[k + 1] - high * (slack + rounding) / (high - low) - rounding
        )


@given(problem=smooth_problems())
@example(problem=SIGNAL)
@example(problem=SCREENING_TIE)
@example(problem=ITERATION_TIE)
@example(problem=SUPPORT_TIE)
@example(problem=STOPPING_TIE)
def test_backends_agree_within_certified_solution_accuracy(problem):
    X, y = problem.arrays()
    a, b = (problem.fit(backend) for backend in BACKENDS)
    assert len(a) == len(b) == len(problem.grid)
    assert_certified_solutions(a, b, X, y)


@pytest.mark.parametrize(
    "problem",
    [
        pytest.param(SCREENING_TIE, id="zero-score-screening"),
        pytest.param(ITERATION_TIE, id="lambda-max-iterations"),
        pytest.param(SUPPORT_TIE, id="lambda-max-support"),
        pytest.param(STOPPING_TIE, id="tie-changes-later-stopping-iterate"),
    ],
)
def test_backend_ties_satisfy_optimality_and_agree_within_certified_bound(problem):
    """At a strict-comparison tie either discrete branch is valid to rounding.

    Keep these reproducers portable: an exactly zero coefficient in one backend
    and a rounding-sized nonzero in the other is not a support regression.
    A tie can also change a later finite stopping iterate, so its numerical
    tolerance comes from independently capped residuals and segment curvature,
    not a fixed relative summation-error budget for the whole trajectory.
    """
    X, y = problem.arrays()
    a, b = (problem.fit(backend) for backend in BACKENDS)
    assert_certified_solutions(a, b, X, y)


@given(problem=smooth_problems())
@example(problem=SIGNAL)
def test_backend_contract_with_positive_strong_cutoffs(problem):
    """Discrete equality only on a construction away from the exact ties.

    Positive sequential cutoffs .09 and .05 avoid a zero-score screening tie;
    the grid also avoids the dyadic lambda_max ties of this family. Equality
    of counts and support is a property of this construction, not a universal
    floating-point theorem or a requirement at the explicit ties above.
    """
    problem = dataclasses.replace(problem, grid=(0.37, 0.23, 0.14))
    X, y = problem.arrays()
    a, b = (problem.fit(backend) for backend in BACKENDS)
    np.testing.assert_array_equal(a.iterations, b.iterations)
    np.testing.assert_array_equal(a.n_rejected, b.n_rejected)
    np.testing.assert_array_equal(a.coef.toarray() != 0, b.coef.toarray() != 0)
    assert_backend_rounding_contract(a, b, X, y)
