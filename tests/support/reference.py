"""R biglasso's paths, written plainly in numpy: a reference for the tests.

This is the algorithm `pybiglasso.biglasso` runs (R biglasso 1.7.2's
`biglasso(..., family = "binomial", screen = "SSR")`, with pybiglasso's
documented differences from it), in the same steps as the Rust solver
(`rust/src/solver.rs`) and the numba one (`src/pybiglasso/binomial.py`); each
function's docstring names its counterparts in binomial.py and in R. What the
solvers do for speed or for a fixed summation order (one pass for several sums,
reading X a column at a time, fixed-order sums) is plain numpy here: the fit rows are copied and standardized outright, and
sums are `@` and `.sum()`. So results agree with the solvers' to rounding, not
bit for bit. Nothing here imports pybiglasso, so the tests can hold the
solver to it.

The problem, on columns standardized to mean 0 and variance 1:

    minimize  (1/n) * NLL(intercept, beta)
              + lambda * sum_j pf_j * (alpha * |beta_j| + (1 - alpha)/2 * beta_j^2)

solved by coordinate descent on a quadratic approximation of the NLL (IRLS),
for a decreasing grid of lambdas, each fit starting from the last. At each
lambda the sequential strong rule guesses which columns can be nonzero;
descent runs on the active set, and KKT checks correct the guess.

`gaussian_path` ports the least-squares solver directly, with RSS/(2n) in
place of mean NLL. It is held to R before the production Gaussian kernel.
"""

import dataclasses

import numpy as np

# columns whose standard deviation is at most this are left out of the fit
MIN_SCALE = 1e-6
# beyond this linear predictor, IRLS takes p as exactly 0 or 1, with a small
# fixed weight
ETA_MAX, MIN_WEIGHT = 10.0, 1e-4
# glmnet's probability boundary; the working-residual representation also
# needs a positive boundary weight (glmnet can represent zero curvature).
P_MIN = 1e-9
# the path stops once the loss falls below this fraction of the null loss
SATURATED = 0.01


def gaussian_path(
    X,
    y,
    alpha=1.0,
    nlambda=100,
    lambdas=None,
    eps=1e-7,
    lambda_min=None,
    lambda_log_scale=True,
    penalty_factor=None,
    row_indices=None,
    dfmax=None,
    max_iter=1000,
    r_compatible=False,
):
    """Plain numpy port of R biglasso 1.7.2's `cdfit_gaussian_ssr`.

    Unlike logistic IRLS, Gaussian sweeps update the exact residual and use
    unit curvature. There is no saturation cutoff. R's loss is RSS, set only
    after both KKT scans pass (even if the iteration budget was exhausted).
    Arguments have already been validated, as in the binomial reference.
    """
    X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if row_indices is not None:
        X, y = X[row_indices], y[row_indices]
    nrow, ncol = X.shape
    center, scale = center_scale(X)
    used = np.flatnonzero(scale > MIN_SCALE)
    Xs = (X[:, used] - center[used]) / scale[used]
    pf = (
        np.ones(len(used))
        if penalty_factor is None
        else np.asarray(penalty_factor)[used]
    )
    dfmax = ncol + 1 if dfmax is None else dfmax
    lambda_min = (0.001 if nrow > ncol else 0.05) if lambda_min is None else lambda_min
    ybar = y.mean()
    residual = y - ybar
    tolerance = eps * (residual @ residual) / nrow
    beta = np.zeros(len(used))
    scores = compute_z(Xs, residual)
    active = np.zeros(len(used), dtype=bool)
    strong = active.copy()
    events = []

    def sweep(lam):
        largest = 0.0
        for j in np.flatnonzero(active):
            scores[j] = Xs[:, j] @ residual / nrow + beta[j]
            l1, l2 = lam * alpha * pf[j], lam * (1 - alpha) * pf[j]
            value = np.sign(scores[j]) * max(abs(scores[j]) - l1, 0.0) / (1 + l2)
            shift = value - beta[j]
            if shift != 0:
                largest = max(largest, shift**2)
                residual[:] -= shift * Xs[:, j]
                beta[j] = value
        return largest

    def scan(columns, lam, *, rest=False):
        scores[columns] = Xs[:, columns].T @ residual / nrow
        violations = columns[
            np.abs(scores[columns] - beta[columns] * lam * (1 - alpha) * pf[columns])
            > lam * alpha * pf[columns]
        ]
        active[violations] = True
        if rest:
            strong[violations] = True
        return len(violations)

    lambda_max = largest_lambda(scores, pf, alpha)
    automatic = lambdas is None
    if automatic and np.any(pf == 0):
        # Differs from R biglasso: fit unpenalized columns first, so the grid
        # starts where the penalized coefficients can all be zero.
        active[pf == 0] = True
        for _ in range(max_iter):
            change = sweep(0.0)
            if change < tolerance or change == 0:
                break
        else:
            raise ValueError("Unpenalized prefit failed to converge.")
        scores = compute_z(Xs, residual)
        lambda_max = largest_lambda(scores, pf, alpha)

    if automatic and not np.isfinite(lambda_max):
        raise ValueError("The automatic lambda grid must be finite.")
    grid = (
        lambda_grid(lambda_max, max(lambda_min, 1e-6), nlambda, lambda_log_scale)
        if automatic
        else np.asarray(lambdas, dtype=np.float64)
    )
    if automatic and not np.isfinite(grid).all():
        raise ValueError("The automatic lambda grid must be finite.")
    path = dict(lambda_=[], loss=[], iter=[], rejections=[], beta=[])

    def record(lam, loss, iterations, rejected):
        coefficients = np.zeros(ncol)
        coefficients[used] = beta / scale[used]
        path["lambda_"].append(lam)
        path["loss"].append(loss)
        path["iter"].append(iterations)
        path["rejections"].append(rejected)
        path["beta"].append(np.r_[ybar - center @ coefficients, coefficients])

    if automatic:
        record(grid[0], residual @ residual, 0, len(used))
    previous = lambda_max
    for lam in grid[1:] if automatic else grid:
        index = len(path["lambda_"])
        if index and np.count_nonzero(beta) > dfmax:
            events.append(("dfmax", index))
            break
        strong[:] = np.abs(scores) > (2 * lam - previous) * alpha * pf
        rejected = len(used) - np.count_nonzero(strong)
        iterations, loss = 0, 0.0
        while iterations < max_iter:
            while iterations < max_iter:
                while iterations < max_iter:
                    iterations += 1
                    change = sweep(lam)
                    # Differs from R biglasso: with zero null RSS, strict
                    # comparison never succeeds; default mode accepts no change.
                    if change < tolerance or (not r_compatible and change == 0):
                        break
                if not scan(np.flatnonzero(strong & ~active), lam):
                    break
            if not scan(np.flatnonzero(~strong), lam, rest=True):
                loss = residual @ residual
                break
        if iterations == max_iter:
            events.append(("max_iter", index))
        record(lam, loss if r_compatible else residual @ residual, iterations, rejected)
        previous = lam
    return dict(
        center=center,
        scale=scale,
        events=events,
        **{
            k: np.asarray(v).T if k == "beta" else np.asarray(v)
            for k, v in path.items()
        },
    )


@dataclasses.dataclass
class Problem:
    """What one lambda's fit needs to know, besides the lambda; fixed for the path."""

    X: np.ndarray  # the fit rows of the columns that vary, standardized
    y: np.ndarray
    alpha: float
    penalty_factor: np.ndarray  # of the columns in X
    max_iter: int
    stop_on_saturation: bool
    r_compatible: bool
    null_loss: float
    # descent at a lambda has converged once no coordinate's update moves the
    # quadratic approximation by this much
    tolerance: float


@dataclasses.dataclass
class State:
    """The fit in progress, carried from lambda to lambda."""

    intercept: float
    eta: np.ndarray  # the linear predictor, intercept + X @ beta
    beta: np.ndarray  # coefficients of the standardized columns
    scores: np.ndarray  # each column's score x_j's / n, as last computed
    active: np.ndarray  # columns descent updates; it only grows
    strong: np.ndarray  # the strong rule's guess at this lambda, grown by checks
    # IRLS's quadratic approximation, at eta as of the last refresh
    weights: np.ndarray = None
    residual: np.ndarray = None  # y - p
    working_residual: np.ndarray = None  # residual / weights, kept current by descent
    loss: float = np.nan
    iterations: int = 0


def fit_path(
    X,
    y,
    alpha=1.0,
    nlambda=100,
    lambdas=None,
    eps=1e-7,
    lambda_min=None,
    lambda_log_scale=True,
    penalty_factor=None,
    row_indices=None,
    dfmax=None,
    max_iter=1000,
    stop_on_saturation=True,
    r_compatible=False,
    family="binomial",
):
    """Fit the path; arguments are `pybiglasso.biglasso`'s, already validated.

    Returns a dict in R biglasso's layout: `center` and `scale` of every column,
    and for each lambda state, `lambda_`, `loss` (the negative log-likelihood),
    `iter`, `rejections` (columns the strong rule set aside), and `beta`, the
    intercept and coefficients on X's scale, one column per lambda. `events`
    lists what stopped or troubled the state, as (kind, lambda index).

    binomial.py: `biglasso` and `fit_path`; R: `biglasso` and
    `cdfit_binomial_ssr`.
    """
    if family == "gaussian":
        return gaussian_path(
            X,
            y,
            alpha,
            nlambda,
            lambdas,
            eps,
            lambda_min,
            lambda_log_scale,
            penalty_factor,
            row_indices,
            dfmax,
            max_iter,
            r_compatible,
        )
    if family != "binomial":
        raise ValueError(f"Unknown family {family!r}")
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if row_indices is not None:
        X, y = X[row_indices], y[row_indices]
    nrow, ncol = X.shape
    if penalty_factor is None:
        penalty_factor = np.ones(ncol)
    if dfmax is None:
        dfmax = ncol + 1
    # Differs from R biglasso: R compares nrow(X) with ncol(X); this compares
    # the rows actually fit
    if lambda_min is None:
        lambda_min = 0.001 if nrow > ncol else 0.05
    lambda_min = max(lambda_min, 1e-6)

    center, scale = center_scale(X)
    used = np.flatnonzero(scale > MIN_SCALE)
    pf = penalty_factor[used]
    penalized = pf > 0

    ybar = y.mean()
    null_loss = -(y * np.log(ybar) + (1 - y) * np.log(1 - ybar)).sum()
    problem = Problem(
        X=(X[:, used] - center[used]) / scale[used],
        y=y,
        alpha=alpha,
        penalty_factor=pf,
        max_iter=max_iter,
        stop_on_saturation=stop_on_saturation,
        r_compatible=r_compatible,
        null_loss=null_loss,
        tolerance=eps * null_loss / nrow,
    )
    # start from the null model, the intercept alone
    state = State(
        intercept=np.log(ybar / (1 - ybar)),
        eta=np.full(nrow, np.log(ybar / (1 - ybar))),
        beta=np.zeros(len(used)),
        scores=compute_z(problem.X, y - ybar),
        active=np.zeros(len(used), dtype=bool),
        strong=np.zeros(len(used), dtype=bool),
    )

    path = dict(lambda_=[], loss=[], iter=[], rejections=[], beta=[], intercept=[])
    events = []

    def record(lam, rejections):
        path["lambda_"].append(lam)
        path["loss"].append(state.loss if r_compatible else final_loss(y, state.eta))
        path["iter"].append(state.iterations)
        path["rejections"].append(rejections)
        path["beta"].append(state.beta.copy())
        path["intercept"].append(state.intercept)

    if lambdas is not None:
        # a given grid is fit from its first value, starting from the null model,
        # with R's lambda_max as the strong rule's previous lambda
        grid = np.asarray(lambdas, dtype=np.float64)
        lambda_prev = largest_lambda(state.scores, pf, alpha)
    else:
        # the grid runs down from lambda_max, the smallest lambda at which every
        # penalized coefficient is zero; that fit is the path's first point
        if not penalized.all():
            # Differs from R biglasso, whose default grid is then all zeros
            # (an unregularized fit): here the solution at lambda_max is the fit
            # of the unpenalized columns alone, and its residuals set lambda_max
            state.active[~penalized] = state.strong[~penalized] = True
            status = _fit_lambda(problem, state, 0.0, check_kkt=False)
            if status == "saturated":
                events.append(("prefit_saturated", 0))
                return _results(path, events, center, scale, used, ncol)
            if status == "max_iter":
                raise ValueError("Unpenalized prefit failed to converge.")
            # the refresh would replace the loss, which R reports from the
            # prefit's last iteration
            prefit_loss = state.loss
            update_irls(problem, state)
            state.scores = compute_z(problem.X, state.residual)
            state.loss = prefit_loss
        else:
            state.loss = null_loss
        lambda_max = largest_lambda(state.scores, pf, alpha)
        grid = lambda_grid(lambda_max, lambda_min, nlambda, lambda_log_scale)
        record(grid[0], rejections=len(used))
        lambda_prev = lambda_max
        grid = grid[1:]

    for lam in grid:
        index = len(path["lambda_"])
        if index > 0 and np.count_nonzero(state.beta) > dfmax:
            events.append(("dfmax", index))
            break

        rejections = strong_rule(problem, state, lam, lambda_prev)
        state.iterations = 0
        status = _fit_lambda(problem, state, lam, check_kkt=True)
        if status == "saturated":
            events.append(("saturated", index))
            break
        if status == "max_iter":
            events.append(("max_iter", index))
        record(lam, rejections)
        lambda_prev = lam

    return _results(path, events, center, scale, used, ncol)


def center_scale(X):
    """Each column's mean and population standard deviation.

    binomial.py: `center_scale`; R: `standardize_and_get_residual`.
    """
    return X.mean(axis=0), X.std(axis=0)


def compute_z(X, residual):
    """Each standardized column's score against a residual, x_j'residual / n.

    binomial.py: `compute_z`; R: `standardize_and_get_residual`.
    """
    return X.T @ residual / len(residual)


def largest_lambda(scores, penalty_factor, alpha):
    """The smallest lambda at which every penalized coefficient is zero.

    binomial.py: inline in `fit_path`; R: `standardize_and_get_residual`.
    """
    penalized = penalty_factor > 0
    return (
        np.max(np.abs(scores[penalized]) / penalty_factor[penalized], initial=0.0)
        / alpha
    )


def lambda_grid(lambda_max, lambda_min, nlambda, log_scale):
    """The default grid, from lambda_max down to lambda_min * lambda_max.

    binomial.py: inline in `fit_path`; R: in `biglasso`'s setup.
    """
    if lambda_max == 0:
        return np.zeros(nlambda)
    lo, hi = lambda_min * lambda_max, lambda_max
    if log_scale:
        return np.logspace(np.log(hi), np.log(lo), nlambda, base=np.e)
    return np.linspace(hi, lo, nlambda)


def strong_rule(problem, state, lam, lambda_prev):
    """The sequential strong rule's guess at the columns that can be nonzero:
    |score_j| > alpha * pf_j * (2 lambda - lambda_prev), with each column's
    score as last computed. Returns how many columns it sets aside.

    binomial.py: inline in `fit_path`; R: in `cdfit_binomial_ssr`.
    """
    cutoff = (2 * lam - lambda_prev) * problem.alpha * problem.penalty_factor
    state.strong = np.abs(state.scores) > cutoff
    return len(state.strong) - np.count_nonzero(state.strong)


def update_irls(problem, state):
    """Refresh the quadratic approximation at eta: weights p(1 - p), residual
    y - p, and working residual (y - p) / weights; and the loss at eta.

    binomial.py: `update_irls`; R: the probability and weight update in
    `cdfit_binomial_lambda_core`.
    """
    eta, y = state.eta, problem.y
    eta_max = ETA_MAX if problem.r_compatible else np.log(1 / P_MIN - 1)
    min_weight = MIN_WEIGHT if problem.r_compatible else P_MIN * (1 - P_MIN)
    clipped = np.abs(eta) > eta_max
    # p is exactly 0 or 1 where clipped, so the loss there can be infinite
    with np.errstate(divide="ignore"):
        p = np.empty_like(eta)
        p[clipped] = (eta[clipped] > 0).astype(float)
        e = np.exp(eta[~clipped])
        p[~clipped] = e / (1 + e)
        state.loss = -np.where(y == 1, np.log(p), np.log(1 - p)).sum()
    state.weights = np.where(clipped, min_weight, p * (1 - p))
    state.residual = y - p
    state.working_residual = state.residual / state.weights


def final_loss(y, eta):
    """The negative log-likelihood at eta, computed stably.

    binomial.py: `final_loss`. Differs from R biglasso, which reports the loss
    from before the last sweep, with clipped probabilities; r_compatible keeps
    R's.
    """
    return np.logaddexp(0, np.where(y == 1, -eta, eta)).sum()


def move(state, x, step):
    """Move the linear predictor by step * x, and the working residual with it."""
    state.eta += step * x
    state.working_residual -= step * x


def update_intercept(state):
    """Move the intercept to the weighted mean of the working residual.

    binomial.py: `update_intercept`; R: the intercept update in
    `cdfit_binomial_lambda_core`.
    """
    step = (state.weights @ state.working_residual) / state.weights.sum()
    state.intercept += step
    move(state, 1.0, step)


def coordinate_sweep(problem, state, lam):
    """Minimize the approximation along each active column in turn.

    Returns the convergence measure: the largest curvature-weighted squared
    change in a coefficient, v * (change in beta_j)^2.

    binomial.py: `coordinate_sweep`; R: the coordinate loop in
    `cdfit_binomial_lambda_core`, with `wcrossprod_resid`, `wsqsum_bm` and
    `lasso`.
    """
    nrow, w = len(problem.y), state.weights
    largest_change = 0.0
    for j in np.flatnonzero(state.active):
        x = problem.X[:, j]
        curvature = (w * x) @ x / nrow
        # the unpenalized minimizer along x_j, times the curvature
        u = (w * x) @ state.working_residual / nrow + curvature * state.beta[j]
        l1, l2 = penalties(problem, lam, j)

        if abs(u) < l1:
            new = 0.0
        elif problem.r_compatible:
            # R biglasso's update, which divides by v * (1 + l2)
            new = np.sign(u) * (abs(u) - l1) / curvature / (1 + l2)
        else:
            # Differs from R biglasso: the documented objective calls for
            # v + l2; the two agree for the lasso (l2 = 0)
            new = np.sign(u) * (abs(u) - l1) / (curvature + l2)

        step = new - state.beta[j]
        if step != 0:
            largest_change = max(largest_change, curvature * step**2)
            move(state, x, step)
        state.beta[j] = new
    return largest_change


def penalties(problem, lam, j):
    """Column j's L1 and L2 penalties at lam."""
    penalty = lam * problem.penalty_factor[j]
    return penalty * problem.alpha, penalty * (1 - problem.alpha)


def kkt_violations(problem, state, lam, columns):
    """Recompute the scores of `columns`; return those breaking the KKT
    conditions, |score_j - l2_j * beta_j| > l1_j.

    The scores use the residual from the last IRLS refresh, as R's do: from
    before the sweep that just converged.
    """
    state.scores[columns] = compute_z(problem.X[:, columns], state.residual)
    l1, l2 = penalties(problem, lam, columns)
    return columns[np.abs(state.scores[columns] - l2 * state.beta[columns]) > l1]


def check_strong_set(problem, state, lam):
    """Check the strong set's inactive columns; violators become active.

    Returns how many violated.

    binomial.py: `check_strong_set`; R: `check_strong_set`.
    """
    bad = kkt_violations(
        problem, state, lam, np.flatnonzero(state.strong & ~state.active)
    )
    state.active[bad] = True
    return len(bad)


def check_rest_set(problem, state, lam):
    """Check every column outside the strong set; violators join both sets.

    An active column outside the strong set is checked too, and counts as a
    violation if its score breaks the conditions. Returns how many violated.

    binomial.py: `check_rest_set`; R: `check_rest_set`.
    """
    bad = kkt_violations(problem, state, lam, np.flatnonzero(~state.strong))
    state.active[bad] = state.strong[bad] = True
    return len(bad)


def _fit_lambda(problem, state, lam, check_kkt):
    """Descend to convergence at one lambda: "converged", "saturated" or "max_iter".

    With check_kkt=False, only the columns already active are fit.

    binomial.py: `_fit_lambda`; R: `cdfit_binomial_lambda_core`.
    """
    unscanned = "both"  # the scans the last iteration didn't run
    while state.iterations < problem.max_iter:
        state.iterations += 1
        unscanned = "both"
        update_irls(problem, state)
        # Differs from R biglasso, which always stops here; going on
        # (stop_on_saturation=False) is not in R
        if problem.stop_on_saturation and state.loss / problem.null_loss < SATURATED:
            return "saturated"

        update_intercept(state)
        if coordinate_sweep(problem, state, lam) >= problem.tolerance:
            continue
        if not check_kkt:
            return "converged"
        # if the strong set breaks the conditions, descend again with the
        # enlarged active set; then the same for the rest
        if check_strong_set(problem, state, lam) > 0:
            unscanned = "rest"
            continue
        if check_rest_set(problem, state, lam) == 0:
            return "converged"
        unscanned = "none"

    # as R biglasso does, finish the scans when max_iter cuts a fit short:
    # they set the scores and the strong set the next lambda starts from
    if check_kkt and unscanned == "both":
        check_strong_set(problem, state, lam)
    if check_kkt and unscanned != "none":
        check_rest_set(problem, state, lam)
    return "max_iter"


def _results(path, events, center, scale, used, ncol):
    """The path in R's layout, with coefficients back on X's scale."""
    nfit = len(path["lambda_"])
    coef = np.array(path["beta"]).reshape(nfit, len(used)).T / scale[used, None]
    beta = np.zeros((ncol + 1, nfit))
    beta[1 + used] = coef
    beta[0] = np.array(path["intercept"]) - center[used] @ coef
    return dict(
        center=center,
        scale=scale,
        lambda_=np.array(path["lambda_"]),
        loss=np.array(path["loss"]),
        iter=np.array(path["iter"], dtype=np.int64),
        rejections=np.array(path["rejections"], dtype=np.int64),
        beta=beta,
        events=events,
    )
