#!/usr/bin/env python3
import collections
import logging
import time
import warnings

import numba as nb
import numpy as np
import scipy.sparse as sps
from threadpoolctl import threadpool_limits

from .path import INTERCEPT, BiglassoPath

logger = logging.getLogger(__name__)


BigLassoParams = collections.namedtuple(
    "BigLassoParams",
    "alpha lambdas nlambda eps lambda_min lambda_log_scale multiplier dfmax max_iter "
    "stop_on_saturation r_compatible",
)


def biglasso(
    X,
    y,
    alpha=1.0,
    family="binomial",
    nlambda=100,
    lambdas=None,
    eps=1e-7,
    lambda_min=None,
    lambda_log_scale=True,
    penalty_factor=None,
    row_indices=None,
    dfmax=None,
    max_iter=1000,
    feature_names=None,
    stop_on_saturation=True,
    r_compatible=False,
):
    """Fit a lasso or elastic-net penalized logistic regression path.

    This is a port of R biglasso's `biglasso(..., family = "binomial",
    screen = "SSR")`, and uses its parameter names: `alpha` mixes the L1 and L2
    penalties (1 is the lasso), and `lambdas` are penalty strengths.

    X may be a numpy array or memmap, ideally in Fortran (column-major) order;
    it is read a column at a time and never copied.

    Like R, the path stops early once training deviance falls below 1% of the
    null deviance; `stop_on_saturation=False` (not in R) fits every lambda.

    For the elastic net (alpha < 1), the coordinate update minimizes the
    documented objective, (1/n) * NLL + lambda * (alpha * |b|_1 +
    (1 - alpha) / 2 * |b|_2^2), as glmnet and scikit-learn do. R biglasso's
    logistic update instead divides by v * (1 + l2) rather than v + l2, which
    solves a different problem; `r_compatible=True` reproduces R exactly. The
    two agree for the lasso (alpha = 1).

    Returns a `BiglassoPath`.
    """

    if feature_names is None and hasattr(X, "columns"):
        feature_names = [str(c) for c in X.columns]
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    y = np.asarray(y)

    # -- first, a few checks on data shape and type
    if not np.isfortran(X):
        warnings.warn(
            "The design matrix is not in Fortran (column-major) order; "
            "this algorithm performs best on Fortran-ordered arrays.",
            stacklevel=2,
        )

    if alpha == 0:
        raise NotImplementedError(
            "Sorry, this implementation does not support zero alpha."
        )
    elif not (0 < alpha <= 1):
        raise ValueError(f"Alpha must be in (0, 1]; you gave [{alpha}]")

    if family != "binomial":
        raise NotImplementedError("Sorry, we only do binomial endpoints now.")

    if X.shape[0] != len(y):
        raise ValueError(
            f"Number of rows in X [{X.shape[0]}] is different from length of y [{len(y)}]."
        )

    if row_indices is not None:
        row_indices = np.asarray(row_indices)
        if np.any((row_indices < 0) | (row_indices >= X.shape[0])):
            raise ValueError("row_indices has indices outside valid range.")
        row_indices = row_indices.astype(np.uint32)

    y_used = y if row_indices is None else y[row_indices]
    if set(np.unique(y_used)) != {0, 1}:
        raise ValueError(
            "Family `binomial` requires a response with both 0s and 1s; "
            f"got values {set(np.unique(y_used))}."
        )

    nrow, ncol = get_nrow(X, row_indices), X.shape[1]

    if feature_names is not None:
        feature_names = tuple(str(f) for f in feature_names)
        if len(feature_names) != ncol:
            raise ValueError("feature_names must have one entry per column of X.")
        if INTERCEPT in feature_names:
            raise ValueError(f"{INTERCEPT!r} is reserved for the intercept.")

    lambda_min_thresh = (0.001, 0.05)
    if lambda_min is None:
        lambda_min = lambda_min_thresh[0] if (nrow > ncol) else lambda_min_thresh[1]

    if penalty_factor is None:
        multiplier = np.ones(ncol)
    else:
        multiplier = np.array(penalty_factor, dtype=np.float64)
        if multiplier.shape != (ncol,):
            raise ValueError("penalty_factor must have one entry per column of X.")
        if np.any(multiplier < 0) or not np.any(multiplier > 0):
            raise ValueError(
                "penalty_factor must be non-negative, with at least one positive entry."
            )

    if lambdas is not None:
        lambdas = np.array(lambdas, dtype=np.float64).ravel()
        if len(lambdas) == 0 or np.any(lambdas < 0):
            raise ValueError("lambdas must be a non-empty sequence of values >= 0.")
        if np.any(np.diff(lambdas) > 0):
            raise ValueError("lambdas must be in decreasing order.")
        nlambda = len(lambdas)

    if dfmax is None:
        dfmax = ncol + 1

    if feature_names is None:
        feature_names = tuple(f"x{e}" for e in range(ncol))

    y = y.astype(np.float64)

    params = BigLassoParams(
        alpha=alpha,
        lambdas=lambdas,
        nlambda=nlambda,
        eps=eps,
        lambda_min=lambda_min,
        lambda_log_scale=lambda_log_scale,
        multiplier=multiplier,
        dfmax=dfmax,
        max_iter=max_iter,
        stop_on_saturation=stop_on_saturation,
        r_compatible=r_compatible,
    )
    logger.info(
        "Starting fit. Design matrix size: [%d, %d]  Num λ values: %d",
        nrow,
        ncol,
        nlambda,
    )
    # -- at least for the vector-vector dot products we do here, the overhead of
    #    openblas/mkl multithreading appears to slow down the program in tests,
    #    so we clamp those operations to a single thread
    with threadpool_limits(limits=1, user_api="blas"):
        results = binomial_hsr(X, y, row_indices, params)

    # -- keep the part of the path that was fit (it stops early on saturation
    #    or dfmax), and untransform coefficients back to the scale of X
    fitted = np.flatnonzero(results["iters"] >= 0)
    col_map, center, scale = results["col_map"], results["center"], results["scale"]
    rows, vals, intercept = [], [], results["intercept"][fitted].copy()
    for k, il in enumerate(fitted):
        cols = col_map[results["coef_rows"][il]]
        beta = results["coef_vals"][il] / scale[cols]
        intercept[k] -= center[cols] @ beta
        rows.append(cols)
        vals.append(beta)
    indptr = np.concatenate([[0], np.cumsum([len(r) for r in rows])])
    coef = sps.csc_array(
        (
            np.concatenate(vals) if vals else np.zeros(0),
            np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64),
            indptr,
        ),
        shape=(ncol, len(fitted)),
    )

    return BiglassoPath(
        lambdas=results["lambda_"][fitted],
        intercept=intercept,
        coef=coef,
        # -- the solver tracks -log-likelihood, as R does; report true deviance
        deviance=2.0 * results["deviance"][fitted],
        iterations=results["iters"][fitted],
        n_rejected=results["n_reject"][fitted],
        center=center,
        scale=scale,
        penalty_factor=multiplier,
        feature_names=feature_names,
        alpha=alpha,
    )


@nb.njit(fastmath=True)
def dot3(a, b, c):  # pragma: no cover
    """Faster version of (a * b * c).sum()"""
    out = 0.0
    for ix in range(len(a)):
        out += a[ix] * b[ix] * c[ix]
    return out


@nb.njit(fastmath=True)
def dot(a, b):  # pragma: no cover
    """Dot product accumulated in float64; works for mixed float32/float64 inputs."""
    out = 0.0
    for ix in range(len(a)):
        out += a[ix] * b[ix]
    return out


@nb.njit()
def center_scale(X, row_indices):  # pragma: no cover
    """Compute the mean and std of matrix X.

    Spares memory by working a column at a time; suitable for use with memory
    mapped arrays (or in-memory arrays).

    Becase by column, you want X to be in Fortran order.
    """

    mean = np.zeros(X.shape[1], dtype=np.float64)
    std = mean.copy()

    for ix in range(X.shape[1]):
        if row_indices is None:
            col = X[:, ix]
        else:
            col = X[row_indices, ix]

        # -- two passes, accumulating in float64 regardless of X's dtype
        acc = 0.0
        for v in col:
            acc += v
        mean[ix] = acc / len(col)
        acc = 0.0
        for v in col:
            acc += (v - mean[ix]) ** 2
        std[ix] = np.sqrt(acc / len(col))

    return mean, std


@nb.njit()
def build_sum_xy(X, y, row_indices, col_map):  # pragma: no cover
    """Compute columnwise dot product of X with y.

    Memory-map-friendly, columnwise version of:

    # sum_xy = (X[:, col_map] * y[:, np.newaxis]).sum(0)
    """

    out = np.zeros(len(col_map))

    if row_indices is None:
        for j in range(len(col_map)):
            out[j] = dot(X[:, col_map[j]], y)
    else:
        for j in range(len(col_map)):
            out[j] = dot(X[row_indices, col_map[j]], y[row_indices])

    return out


@nb.njit()
def np_logspace(lo, hi, num):  # pragma: no cover
    """Alternate version of np.logspace that's closer to np.linspace."""
    # np.logspace(np.log(3), np.log(4), num=5, base=np.e)
    return np.exp(np.linspace(np.log(lo), np.log(hi), num))


@nb.njit()
def weighted_sum_squares(Xcol, w, c, s):  # pragma: no cover
    """Compute weighted sum of squares."""
    sum_wx_sq, sum_wx, sum_w = 0.0, 0.0, 0.0
    for ix in range(len(Xcol)):
        sum_wx_sq += w[ix] * (Xcol[ix] ** 2)
        sum_wx += w[ix] * Xcol[ix]
        sum_w += w[ix]

    return (sum_wx_sq - 2 * c * sum_wx + (c**2) * sum_w) / (s**2)


@nb.njit()
def get_nrow(X, row_indices):  # pragma: no cover
    """Get number of rows."""
    if row_indices is None:
        nrow = X.shape[0]
    else:
        nrow = len(row_indices)

    return nrow


@nb.njit()
def update_probs(
    y, row_indices, eta, p, w, s, r, eta_max=10, w_min=0.0001
):  # pragma: no cover
    """Perform one round of probability updates.

    Mutates: p, w, s, r

    """

    for ix in range(len(eta)):
        _eta = eta[ix]

        if _eta > eta_max:
            p[ix] = 1
            w[ix] = w_min
        elif _eta < -eta_max:
            p[ix] = 0
            w[ix] = w_min
        else:
            p[ix] = np.exp(_eta) / (1 + np.exp(_eta))
            w[ix] = p[ix] * (1 - p[ix])

        if row_indices is None:
            s[ix] = y[ix] - p[ix]
        else:
            s[ix] = y[row_indices[ix]] - p[ix]
        r[ix] = s[ix] / w[ix]

    # -- faster to do this, like above, in numba bc lazy eval
    # deviance += -1.0 * np.log(np.where(y == 1, p, 1 - p)).sum()
    dev_acc = 0.0
    for ix in range(len(eta)):
        if row_indices is None:
            yval = y[ix]
        else:
            yval = y[row_indices[ix]]

        if yval == 1:
            dev_acc += np.log(p[ix])
        else:
            dev_acc += np.log(1 - p[ix])
    deviance = -1.0 * dev_acc

    return deviance


@nb.njit()
def update_intercept(w, r, a0, eta):  # pragma: no cover
    """Update intercept based on current state.

    Mutates: w, r, eta
    """

    xwr = w @ r
    _si = xwr / w.sum()
    intercept = _si + a0

    if _si != 0:
        a0_out = intercept
        r -= _si
        eta += _si
        sum_w_residuals = w @ r
    else:
        # -- don't recompute if r hasn't changed
        a0_out = a0
        sum_w_residuals = xwr

    return intercept, a0_out, sum_w_residuals


@nb.njit()
def update_weights(
    X,
    # y,
    row_indices,
    lam,
    # params,
    alpha,
    multiplier,
    # bl,
    col_map,
    center,
    scale,
    # bl_i
    e1,
    w,
    r,
    sum_w_residuals,
    eta,
    a,
    r_compatible,
):  # pragma: no cover
    """Perform one round of weight updates.

    Mutates: r, eta, a
    """

    nrow = get_nrow(X, row_indices)

    max_update = 0.0

    for j in range(len(col_map)):
        jj = col_map[j]
        if not e1[j]:
            continue

        if row_indices is None:
            Xcol = X[:, jj]
        else:
            Xcol = X[row_indices, jj]

        xwr = (dot3(Xcol, r, w) - center[jj] * sum_w_residuals) / scale[jj]
        v = weighted_sum_squares(Xcol, w, center[jj], scale[jj]) / nrow

        u = xwr / nrow + v * a[j]

        l1 = lam * multiplier[jj] * alpha
        l2 = lam * multiplier[jj] * (1 - alpha)

        if abs(u) < l1:
            new_weight = 0.0
        else:
            sign = 1 if (u > 0) else -1
            if r_compatible:
                new_weight = sign * (abs(u) - l1) / v / (1 + l2)
            else:
                new_weight = sign * (abs(u) - l1) / (v + l2)

        shift = new_weight - a[j]
        if shift != 0:
            update = v * (shift**2)
            max_update = max(update, max_update)
            _si = shift * (Xcol - center[jj]) / scale[jj]
            r -= _si
            eta += _si
            sum_w_residuals = r @ w

        a[j] = new_weight

    return sum_w_residuals, max_update


@nb.njit()
def check_strong_set(
    X,
    row_indices,
    lam,
    # params
    alpha,
    multiplier,
    # bl,
    col_map,
    center,
    scale,
    # bl_i
    e1,
    e2,
    s,
    z,
    a,
):  # pragma: no cover
    """Scan for violations in strong set."""

    nrow = get_nrow(X, row_indices)
    violations = 0
    for j in range(len(col_map)):
        jj = col_map[j]
        if (e1[j] != 0) or (e2[j] != 1):
            continue
        if row_indices is None:
            _sum = dot(X[:, jj], s)
        else:
            _sum = dot(X[row_indices, jj], s)
        z[j] = (_sum - center[jj] * s.sum()) / (scale[jj] * nrow)
        l1 = lam * multiplier[jj] * alpha
        l2 = lam * multiplier[jj] * (1 - alpha)
        if abs(z[j] - a[j] * l2) > l1:
            e1[j] = 1
            violations += 1

    return violations


@nb.njit()
def check_rest(
    X,
    row_indices,
    lam,
    # params
    alpha,
    multiplier,
    # bl,
    col_map,
    center,
    scale,
    # bl_i
    e1,
    e2,
    s,
    z,
    a,
):  # pragma: no cover
    """Scan for violations in rest of predictor columns."""

    nrow = get_nrow(X, row_indices)
    violations = 0
    for j in range(len(col_map)):
        jj = col_map[j]
        if e2[j] != 0:
            continue
        if row_indices is None:
            _sum = dot(X[:, jj], s)
        else:
            _sum = dot(X[row_indices, jj], s)
        z[j] = (_sum - center[jj] * s.sum()) / (scale[jj] * nrow)
        l1 = lam * multiplier[jj] * alpha
        l2 = lam * multiplier[jj] * (1 - alpha)
        if abs(z[j] - a[j] * l2) > l1:
            e1[j] = e2[j] = 1
            violations += 1

    return violations


def binomial_hsr(
    X,
    y,
    row_indices,
    params,
    # -- constants basically
    scale_min_thresh=1e-6,
):

    # -- PHASE 1: Initialization
    nrow = get_nrow(X, row_indices)
    y_used = y if row_indices is None else y[row_indices]

    # -- compute means and stds online, ie one column at a time
    _center, _scale = center_scale(X, row_indices)

    # -- find reduced set of variable-enough columns/predictors to use during fit
    _col_map = np.where(_scale > scale_min_thresh)[0]
    ncol_used = len(_col_map)
    multiplier = params.multiplier[_col_map]

    # -- a few preliminary calculations, to initialize the data structures
    sum_xy = build_sum_xy(X, y, row_indices, _col_map)
    z = (sum_xy - _center[_col_map] * y_used.sum()) / (
        _scale[_col_map] * nrow
    )  # residuals
    ybar = y_used.mean()
    null_deviance = (
        -1.0 * (y_used * np.log(ybar) + (1 - y_used) * np.log(1 - ybar)).sum()
    )
    a0 = np.log(ybar / (1 - ybar))
    thresh = params.eps * null_deviance / nrow

    # -- R biglasso's lambda_max, which ignores penalty factors; used as the
    #    strong-rule reference for the first user-supplied lambda, as in R
    lambda_max_r = np.abs(z).max() / params.alpha

    # -- here we initialize two largish collections of state, mainly to slim
    #    down the function signatures and make it easier to follow the code flow
    # -- the reason there are two collections, instead of one, is that the first
    #    contains the state that is to be returned from this function, where the
    #    second contains state internal to this function that needn't be
    #    returned

    # -- init to-be-returned state
    bl = dict(
        center=_center,
        scale=_scale,
        col_map=_col_map,
        lambda_=np.zeros(params.nlambda),
        intercept=np.zeros(params.nlambda),
        # -- nonzero coefficients at each lambda, as (used-column index, value)
        coef_rows=[None] * params.nlambda,
        coef_vals=[None] * params.nlambda,
        iters=np.zeros(params.nlambda, dtype=np.int64),
        deviance=np.zeros(params.nlambda),
        n_reject=np.zeros(params.nlambda, np.int64),
    )

    # -- init internal-only state
    bl_i = dict(
        eta=np.repeat(a0, nrow),
        z=z,
        p=np.zeros(nrow),
        w=np.zeros(nrow),
        s=np.zeros(nrow),
        r=np.zeros(nrow),
        a=np.zeros(ncol_used),
        e1=np.zeros(ncol_used, dtype=np.int64),
        e2=np.zeros(ncol_used, dtype=np.int64),
        a0=a0,
        sum_w_residuals=0.0,
    )

    # -- kernels index penalty factors by raw column; `multiplier` is by used column
    fit_args = (
        X,
        y,
        row_indices,
        params.alpha,
        params.multiplier,
        params.max_iter,
        params.stop_on_saturation,
        params.r_compatible,
        bl,
        bl_i,
    )

    # -- PHASE 2: Lambda grid and starting point
    if params.lambdas is not None:
        # -- user-supplied grid: fit every value, starting from the null model
        bl["lambda_"][:] = params.lambdas
        lambda_max = lambda_max_r
        lstart = 0
    else:
        unpenalized = multiplier == 0
        if unpenalized.any():
            # -- fit the unpenalized columns alone; that model is the solution at
            #    lambda_max, and its residuals determine lambda_max
            bl_i["e1"][unpenalized] = 1
            bl_i["e2"][unpenalized] = 1
            status = _fit_lambda(*fit_args, 0, 0.0, null_deviance, thresh, False)
            if status == "saturated":
                warnings.warn(
                    "Model saturated by the unpenalized columns alone; no path fit.",
                    stacklevel=3,
                )
                bl["iters"][:] = -1
                return bl
            update_probs(
                y, row_indices, bl_i["eta"], bl_i["p"], bl_i["w"], bl_i["s"], bl_i["r"]
            )
            bl_i["z"] = z = compute_z(
                X, row_indices, _col_map, _center, _scale, bl_i["s"]
            )
        else:
            bl["deviance"][0] = null_deviance
            bl["intercept"][0] = a0

        penalized = multiplier > 0
        lambda_max = (np.abs(z[penalized]) / multiplier[penalized]).max() / params.alpha

        if params.lambda_log_scale:
            bl["lambda_"][:] = np.flip(
                np_logspace(
                    params.lambda_min * lambda_max,
                    lambda_max,
                    num=params.nlambda,
                )
            )
        else:
            bl["lambda_"][:] = np.linspace(
                lambda_max, params.lambda_min * lambda_max, params.nlambda
            )

        _record_coefs(bl, bl_i, 0)
        bl["n_reject"][0] = ncol_used
        lstart = 1

    logger.info("λ[1/%d] = %0.2g", params.nlambda, bl["lambda_"][0])

    # -- PHASE 3: Fitting
    for il in range(lstart, params.nlambda):
        lam = bl["lambda_"][il]
        logger.info("λ[%d/%d] = %0.2g", il + 1, params.nlambda, lam)
        t_begin = time.process_time()

        if il > 0:
            if (bl_i["a"] != 0).sum() > params.dfmax:
                warnings.warn(
                    f"Exceeded dfmax={params.dfmax} nonzero coefficients; "
                    f"path truncated at λ[{il + 1}/{params.nlambda}].",
                    stacklevel=3,
                )
                bl["iters"][il:] = -1
                return bl

            cutoff = 2 * lam - bl["lambda_"][il - 1]
        else:
            cutoff = 2 * lam - lambda_max

        bl_i["e2"] = (np.abs(z) > cutoff * params.alpha * multiplier).astype(np.int64)
        bl["n_reject"][il] = ncol_used - bl_i["e2"].sum()

        status = _fit_lambda(*fit_args, il, lam, null_deviance, thresh, True)

        if status == "saturated":
            warnings.warn(
                f"Model saturated; path truncated at λ[{il + 1}/{params.nlambda}].",
                stacklevel=3,
            )
            bl["iters"][il:] = -1
            return bl
        elif status == "max_iter":
            warnings.warn(
                f"Failed to converge within max_iter={params.max_iter} iterations "
                f"at λ[{il + 1}/{params.nlambda}].",
                stacklevel=3,
            )
        _record_coefs(bl, bl_i, il)

        t_end = time.process_time()
        logger.info(
            "λ[%d/%d] complete in %dms; %d nonzero coefficients",
            il + 1,
            params.nlambda,
            round(1000 * (t_end - t_begin)),
            (bl_i["a"] != 0).sum(),
        )

    return bl


def _record_coefs(bl, bl_i, il):
    nz = np.flatnonzero(bl_i["a"])
    bl["coef_rows"][il] = nz
    bl["coef_vals"][il] = bl_i["a"][nz].copy()


@nb.njit()
def compute_z(X, row_indices, col_map, center, scale, s):  # pragma: no cover
    """Standardized score x_j'(y - p) / n for every used column."""
    nrow = get_nrow(X, row_indices)
    s_sum = s.sum()
    z = np.zeros(len(col_map))
    for j in range(len(col_map)):
        jj = col_map[j]
        if row_indices is None:
            _sum = dot(X[:, jj], s)
        else:
            _sum = dot(X[row_indices, jj], s)
        z[j] = (_sum - center[jj] * s_sum) / (scale[jj] * nrow)
    return z


def _fit_lambda(
    X,
    y,
    row_indices,
    alpha,
    multiplier,
    max_iter,
    stop_on_saturation,
    r_compatible,
    bl,
    bl_i,
    il,
    lam,
    null_deviance,
    thresh,
    check_kkt,
):
    """Run coordinate descent at one lambda value, writing results to slot `il`.

    With `check_kkt=False`, only the columns already in the active set are fit.

    Returns "converged", "saturated", or "max_iter".
    """

    while bl["iters"][il] < max_iter:
        bl["iters"][il] += 1

        bl["deviance"][il] = update_probs(
            y, row_indices, bl_i["eta"], bl_i["p"], bl_i["w"], bl_i["s"], bl_i["r"]
        )

        # -- break early if model is saturated
        if stop_on_saturation and bl["deviance"][il] / null_deviance < 0.01:
            return "saturated"

        # -- perform one round of weight updates
        bl["intercept"][il], bl_i["a0"], bl_i["sum_w_residuals"] = update_intercept(
            bl_i["w"], bl_i["r"], bl_i["a0"], bl_i["eta"]
        )
        bl_i["sum_w_residuals"], max_update = update_weights(
            X,
            row_indices,
            lam,
            alpha,
            multiplier,
            bl["col_map"],
            bl["center"],
            bl["scale"],
            bl_i["e1"],
            bl_i["w"],
            bl_i["r"],
            bl_i["sum_w_residuals"],
            bl_i["eta"],
            bl_i["a"],
            r_compatible,
        )

        if max_update >= thresh:
            continue
        if not check_kkt:
            return "converged"

        kkt_args = (
            X,
            row_indices,
            lam,
            alpha,
            multiplier,
            bl["col_map"],
            bl["center"],
            bl["scale"],
            bl_i["e1"],
            bl_i["e2"],
            bl_i["s"],
            bl_i["z"],
            bl_i["a"],
        )

        # -- check strong set; if violated, re-fit with the enlarged active set
        if check_strong_set(*kkt_args) > 0:
            continue

        # -- check rest of predictors
        if check_rest(*kkt_args) == 0:
            return "converged"

    return "max_iter"
