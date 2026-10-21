#!/usr/bin/env python3
import collections
import logging
import time

import numba as nb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

logFormatter = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(format=logFormatter, level=logging.INFO)
logger = logging.getLogger(__name__)


BigLassoParams = collections.namedtuple(
    "BigLassoParams",
    "alpha nlambda eps lambda_min lambda_log_scale multiplier dfmax max_iter",
)


def biglasso(
    X,
    y,
    alpha=1.0,
    family="binomial",
    nlambda=100,
    eps=1e-7,
    lambda_min=None,
    penalty_factor=None,
    row_indices=None,
    X_names=None,
    intercept_label="Intercept",
):

    # -- first, a few checks on data shape and type
    if not np.isfortran(X):
        logger.warn(
            """The provided design matrix is not in Fortran order.
            This algorithm performs best on Fortran-ordered arrays."""
        )

    if alpha == 0:
        raise NotImplementedError(
            "Sorry, this implementation does not support zero alpha."
        )
    elif not (0 < alpha <= 1):
        raise ValueError(f"Alpha must be in (0, 1]; you gave [{alpha}]")

    if family != "binomial":
        raise NotImplementedError("Sorry, we only do binomial endpoints now.")
    elif family == "binomial":
        if set(y) != {0, 1}:
            raise ValueError(
                f"Family `binomial` specified, but set(response) is {set(y)} not {{0,1}}."
            )

    if X.shape[0] != len(y):
        raise ValueError(
            f"Number of rows in X [{X.shape[0]}] is different from length of y [{len(y)}]."
        )

    if row_indices is not None:
        if np.any((row_indices < 0) | (row_indices >= X.shape[0])):
            raise ValueError("row_ix has indices outside valid range.")
        row_indices = row_indices.astype(np.uint32)

    nrow, ncol = get_nrow(X, row_indices), X.shape[1]
    print("nr nc", nrow, ncol)

    if X_names is not None:
        if len(X_names) != ncol:
            raise ValueError(
                "X_names has different length than number of columns in X."
            )
        if intercept_label in X_names:
            raise ValueError(
                f"Column names conflict with intercept label: [{intercept_label}]"
            )

    lambda_min_thresh = (0.001, 0.05)
    if lambda_min is None:
        lambda_min = lambda_min_thresh[0] if (nrow > ncol) else lambda_min_thresh[1]

    if penalty_factor is None:
        multiplier = np.ones(ncol)
    else:
        assert len(penalty_factor) == ncol
        multiplier = penalty_factor.copy()

    if X_names is None:
        X_names = [f"c{e}" for e in range(ncol)]

    y = y.astype(np.float64)

    params = BigLassoParams(
        alpha=alpha,
        nlambda=nlambda,
        eps=eps,
        lambda_min=lambda_min,
        lambda_log_scale=True,
        multiplier=multiplier,
        dfmax=-1,
        max_iter=1000,
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

    # -- postprocess
    #    1. untransform coefficients
    results["weights"] = (
        results["weights"] / results["scale"][results["col_map"], np.newaxis]
    )
    results["intercept"] -= results["center"][results["col_map"]] @ results["weights"]
    _weights = np.zeros(
        (len(results["center"]), len(results["lambda_"])),
        dtype=results["weights"].dtype,
    )
    _weights[results["col_map"], :] = results["weights"]

    # -- 2. truncate saturated models
    sat_mask = np.isfinite(results["deviance"])

    # -- 3. build output structures
    df_ncol = pd.DataFrame(
        dict(
            feature=X_names,
            center=results["center"],
            scale=results["scale"],
            penalty_factor=multiplier,
        )
    ).assign(feature_retained=[e in results["col_map"] for e in range(ncol)])

    df_nlambda = pd.DataFrame(
        dict(
            deviance=results["deviance"][sat_mask],
            lambda_=results["lambda_"][sat_mask],
            iterations=results["iters"][sat_mask],
            num_rejected=results["n_reject"][sat_mask],
        )
    )

    coefs = np.column_stack(
        [results["intercept"][sat_mask], _weights[:, sat_mask].transpose()]
    )
    df_weights = pd.DataFrame(
        coefs,
        index=results["lambda_"][sat_mask],
        columns=[intercept_label] + list(X_names),
    ).astype(pd.SparseDtype(coefs.dtype, fill_value=0.0))

    return dict(
        df_ncol=df_ncol,
        df_nlambda=df_nlambda,
        df_weights=df_weights,
    )


@nb.njit()
def dot3(a, b, c):  # pragma: no cover
    """Faster version of (a * b * c).sum()"""
    out = 0.0
    for ix in range(len(a)):
        out += a[ix] * b[ix] * c[ix]
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

        mean[ix] = np.mean(col)
        std[ix] = np.std(col)

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
            out[j] = X[:, col_map[j]] @ y
    else:
        for j in range(len(col_map)):
            out[j] = X[row_indices, col_map[j]] @ y[row_indices]

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

    return (sum_wx_sq - 2 * c * sum_wx + (c ** 2) * sum_w) / (s ** 2)


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
):  # pragma: no cover
    """Perform one round of weight updates.

    Mutates: r, eta, a
    """

    nrow = get_nrow(X, row_indices)

    max_update = 0.0

    next_weights = a.copy()
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
            next_weights[j] = 0
        else:
            sign = 1 if (u > 0) else -1
            next_weights[j] = sign * (abs(u) - l1) / v / (1 + l2)

        shift = next_weights[j] - a[j]
        if shift != 0:
            update = v * (shift ** 2)
            max_update = max(update, max_update)
            _si = shift * (Xcol - center[jj]) / scale[jj]
            r -= _si
            eta += _si
            sum_w_residuals = r @ w

        a[j] = next_weights[j]

    return next_weights, sum_w_residuals, max_update


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
            _sum = X[:, jj] @ s
        else:
            _sum = X[row_indices, jj] @ s
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
            _sum = X[:, jj] @ s
        else:
            _sum = X[row_indices, jj] @ s
        z[j] = (_sum - center[jj] * s.sum()) / (scale[jj] * nrow)
        l1 = lam * multiplier[jj] * alpha
        l2 = lam * multiplier[jj] * (1 - alpha)
        if abs(z[j] - a[j] * l2) > l1:
            e1[j] = 1
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

    # -- compute means and stds online, ie one column at a time
    _center, _scale = center_scale(X, row_indices)

    # -- find reduced set of variable-enough columns/predictors to use during fit
    _col_map = np.where(_scale > scale_min_thresh)[0]
    ncol_used = len(_col_map)

    # -- a few preliminary calculations, to initialize the data structures
    sum_xy = build_sum_xy(X, y, row_indices, _col_map)
    if row_indices is None:
        z = (sum_xy - _center[_col_map] * y.sum()) / (
            _scale[_col_map] * nrow
        )  # residuals
        ybar = y.mean()
        null_deviance = -1.0 * (y * np.log(ybar) + (1 - y) * np.log(1 - ybar)).sum()
    else:
        z = (sum_xy - _center[_col_map] * y[row_indices].sum()) / (
            _scale[_col_map] * nrow
        )
        ybar = y[row_indices].mean()
        null_deviance = (
            -1.0
            * (
                y[row_indices] * np.log(ybar) + (1 - y[row_indices]) * np.log(1 - ybar)
            ).sum()
        )
    a0 = np.log(ybar / (1 - ybar))
    thresh = params.eps * null_deviance / nrow

    lambda_max = np.abs(z).max() / params.alpha  # FIXFIX: alpha zero case?

    # -- set up lambda-values grid if not provided
    if params.lambda_log_scale:
        lambda_ = np.flip(
            np_logspace(
                params.lambda_min * lambda_max,
                lambda_max,
                num=params.nlambda,
            )
        )
    else:
        lambda_ = np.linspace(params.lambda_min, lambda_max, params.nlambda)

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
        lambda_=lambda_,
        intercept=np.zeros(params.nlambda),
        weights=np.zeros((ncol_used, params.nlambda)),
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

    # -- PHASE 2: Fitting

    # -- population state for first lambda value
    bl["deviance"][0] = null_deviance
    bl["intercept"][0] = a0
    lstart = 1
    bl["n_reject"][0] = ncol_used

    logger.info("λ[1/%d] = %0.2g", params.nlambda, bl["lambda_"][0])

    # -- loop over remaining lambda values
    for il in range(lstart, len(bl["lambda_"])):

        lam = bl["lambda_"][il]
        logger.info("λ[%d/%d] = %0.2g", il + 1, params.nlambda, lam)
        t_begin = time.process_time()

        if il > 0:
            if params.dfmax > 0:
                nv = (bl_i["a"] != 0).sum()
                if nv > params.dfmax:
                    raise ValueError("Exceeded max coefficients.")

            cutoff = 2 * lam - bl["lambda_"][il - 1]
        else:
            cutoff = 2 * lam - lambda_max

        bl_i["e2"] = (
            np.abs(z) > cutoff * params.alpha * params.multiplier[bl["col_map"]]
        )
        bl["n_reject"][il] = ncol_used - bl_i["e2"].sum()

        while bl["iters"][il] < params.max_iter:

            bl["iters"][il] += 1

            bl["deviance"][il] = update_probs(
                y, row_indices, bl_i["eta"], bl_i["p"], bl_i["w"], bl_i["s"], bl_i["r"]
            )

            # -- break early if model is saturated
            if bl["deviance"][il] / null_deviance < 0.01:
                logger.info("Model is saturated, exiting early.")
                bl["deviance"][il:] = np.nan
                return bl

            # -- perform one round of weight updates
            bl["intercept"][il], bl_i["a0"], bl_i["sum_w_residuals"] = update_intercept(
                bl_i["w"], bl_i["r"], bl_i["a0"], bl_i["eta"]
            )
            bl["weights"][:, il], bl["sum_w_residuals"], max_update = update_weights(
                X,
                # y,
                row_indices,
                lam,
                params.alpha,
                params.multiplier,
                bl["col_map"],
                bl["center"],
                bl["scale"],
                bl_i["e1"],
                bl_i["w"],
                bl_i["r"],
                bl_i["sum_w_residuals"],
                bl_i["eta"],
                bl_i["a"],
            )

            if max_update < thresh:
                logger.debug("max_update l.t. thresh")
                pass
            else:
                continue

            # -- check strong set
            violations = check_strong_set(
                X,
                row_indices,
                lam,
                params.alpha,
                params.multiplier,
                bl["col_map"],
                bl["center"],
                bl["scale"],
                bl_i["e1"],
                bl_i["e2"],
                bl_i["s"],
                bl_i["z"],
                bl_i["a"],
            )
            if violations == 0:
                logger.debug("no strong set violations, proceeding to check of rest")
            else:
                # -- then re-eval icept and weights with updated strong set
                continue

            # -- check rest of predictors
            violations = check_rest(
                X,
                row_indices,
                lam,
                params.alpha,
                params.multiplier,
                bl["col_map"],
                bl["center"],
                bl["scale"],
                bl_i["e1"],
                bl_i["e2"],
                bl_i["s"],
                bl_i["z"],
                bl_i["a"],
            )
            if violations == 0:
                logger.debug("no rest-of-set violations, lambda value is finished")
                break
            else:
                continue  # redundant, but kept for continuity with strong-set check

        t_end = time.process_time()
        logger.info(
            "λ[%d/%d] complete in %dms; %d nonzero coefficients",
            il + 1,
            params.nlambda,
            round(1000 * (t_end - t_begin)),
            (bl_i["a"] != 0).sum(),
        )

    return bl
