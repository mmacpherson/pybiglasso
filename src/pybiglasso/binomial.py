#!/usr/bin/env python3
import collections
import logging
import math
import numbers
import warnings

import numpy as np
import scipy.sparse as sps

from .config import check_backend, settings
from .path import INTERCEPT, BiglassoPath

logger = logging.getLogger(__name__)


def _integer(name, value, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def _finite_scalar(name, value):
    if not isinstance(value, numbers.Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number.")


BigLassoParams = collections.namedtuple(
    "BigLassoParams",
    "alpha lambdas nlambda eps lambda_min lambda_log_scale penalty_factor dfmax max_iter "
    "stop_on_saturation r_compatible",
)


def biglasso(
    X,
    y,
    alpha=1.0,
    family="gaussian",
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
    n_threads=None,
    backend=None,
    _stop=None,
):
    """Fit a lasso or elastic-net binomial or Gaussian regression path.

    The default family="gaussian" minimizes RSS/(2n) plus the standardized elastic-net
    penalty, with an unpenalized intercept. It requires the Rust backend and
    reports RSS as deviance. Gaussian fits do not use a saturation cutoff.

    This ports R biglasso's Gaussian and binomial solvers with screen="SSR",
    and uses its parameter names: `alpha` mixes the L1 and L2
    penalties (1 is the lasso), and `lambdas` are penalty strengths.

    X may be a numpy array or memmap, ideally in Fortran (column-major) order;
    it is read a column at a time and never copied.

    Like R, a binomial path stops early once training deviance falls below 1% of the
    null deviance; `stop_on_saturation=False` (not in R) fits every lambda.

    For binomial elastic net (alpha < 1), the coordinate update minimizes the
    documented objective, (1/n) * NLL + lambda * (alpha * |b|_1 +
    (1 - alpha) / 2 * |b|_2^2), as glmnet and scikit-learn do. R biglasso's
    logistic update instead divides by v * (1 + l2) rather than v + l2, which
    solves a different problem; `r_compatible=True` reproduces R exactly.
    The default binomial probability cutoff also differs from R: it uses
    glmnet's pmin=1e-9 boundary with a positive weight floor; compatibility
    mode retains R's |eta| > 10 cutoff and 1e-4 weights.

    Column-wise work (standardization and the strong-rule and KKT scans) runs on
    `n_threads` threads, by default all available cores; coordinate descent
    itself is sequential. Results do not depend on the thread count.

    The solver is implemented in Rust. `backend=None` and `backend="rust"` are
    accepted for compatibility; `backend="numba"` was removed in 0.4.0.

    Returns a `BiglassoPath`.

    Automatic grids require nlambda >= 2; use lambdas=[value] for a single fit.
    Designs with no varying
    features (or no varying penalized features for an automatic grid) are
    rejected. An exactly zero lambda_max produces a finite all-zero grid.
    `deviance` is evaluated at the final fitted model; compatibility mode
    retains R's historical pre-sweep loss instead.
    """
    # _stop (private, for BiglassoClassifierCV's early stopping): called as
    # _stop(il, lambdas, cols, beta, intercept) after each lambda is fit, with
    # the whole grid and that fit's nonzero coefficients on X's scale; a true
    # result ends the path there, as dfmax does, but without a warning

    if feature_names is None and hasattr(X, "columns"):
        feature_names = [str(c) for c in X.columns]
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    y = np.asarray(y)

    # first, a few checks on data shape and type
    if X.ndim != 2 or not all(X.shape):
        raise ValueError("X must be a non-empty two-dimensional design matrix.")
    if X.dtype.kind not in "biuf" or X.dtype == np.float16 or not X.dtype.isnative:
        raise ValueError(
            "X must have a native real numeric dtype (float32/64 or integer)."
        )
    if X.dtype.kind == "f" and X.dtype.itemsize not in (4, 8):
        raise ValueError("Floating-point X must have dtype float32 or float64.")
    if y.ndim != 1 or y.dtype.kind not in "biuf":
        raise ValueError("y must be a one-dimensional numeric response.")
    _finite_scalar("Alpha", alpha)
    _finite_scalar("eps", eps)
    if eps <= 0:
        raise ValueError("eps must be positive.")
    _integer("max_iter", max_iter)
    if lambdas is None:
        if (
            isinstance(nlambda, (bool, np.bool_))
            or not isinstance(nlambda, numbers.Integral)
            or nlambda < 2
        ):
            raise ValueError(
                "nlambda must be at least 2 (an integer); to fit one value, pass lambdas=[value]."
            )
    if dfmax is not None:
        _integer("dfmax", dfmax, minimum=0)
    if n_threads is not None:
        _integer("n_threads", n_threads)
    if backend is not None:
        check_backend(backend)
    if family not in ("binomial", "gaussian"):
        raise NotImplementedError("Supported families are 'binomial' and 'gaussian'.")
    # not np.isfortran, which is False for one column: both C and F order
    if not X.flags.f_contiguous:
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

    if X.shape[0] != len(y):
        raise ValueError(
            f"Number of rows in X [{X.shape[0]}] is different from length of y [{len(y)}]."
        )

    if row_indices is not None:
        row_indices = np.asarray(row_indices)
        if (
            row_indices.ndim != 1
            or row_indices.size == 0
            or row_indices.dtype.kind not in "iuf"
            or not np.isfinite(row_indices).all()
            or np.any(row_indices != np.floor(row_indices))
        ):
            raise ValueError(
                "row_indices must be a non-empty one-dimensional sequence of integer indices."
            )
        if np.any((row_indices < 0) | (row_indices >= X.shape[0])):
            raise ValueError("row_indices has indices outside valid range.")
        row_indices = row_indices.astype(np.intp)

    y_used = y if row_indices is None else y[row_indices]
    if family == "gaussian" and not np.isfinite(y_used).all():
        raise ValueError("Gaussian y must be finite in the rows used for fitting.")
    if family == "binomial" and set(np.unique(y_used)) != {0, 1}:
        raise ValueError(
            "Family `binomial` requires a response with both 0s and 1s; "
            f"got values {set(np.unique(y_used))}."
        )

    nrow = X.shape[0] if row_indices is None else len(row_indices)
    ncol = X.shape[1]

    if feature_names is not None:
        feature_names = tuple(str(f) for f in feature_names)
        if len(feature_names) != ncol:
            raise ValueError("feature_names must have one entry per column of X.")
        if INTERCEPT in feature_names:
            raise ValueError(f"{INTERCEPT!r} is reserved for the intercept.")

    # Differs from R biglasso: R compares nrow(X) with ncol(X); this
    # compares the rows actually fit, which differ when row_indices is given
    lambda_min_thresh = (0.001, 0.05)
    if lambda_min is None:
        lambda_min = lambda_min_thresh[0] if (nrow > ncol) else lambda_min_thresh[1]
    _finite_scalar("lambda_min", lambda_min)
    if not 0 < lambda_min <= 1:
        raise ValueError("lambda_min must be in (0, 1].")
    lambda_min = max(lambda_min, 1e-6)  # as R does

    if penalty_factor is None:
        penalty_factor = np.ones(ncol)
    else:
        if np.iscomplexobj(penalty_factor):
            raise ValueError("penalty_factor must contain real values.")
        penalty_factor = np.array(penalty_factor, dtype=np.float64)
        if penalty_factor.shape != (ncol,):
            raise ValueError("penalty_factor must have one entry per column of X.")
        if (
            not np.isfinite(penalty_factor).all()
            or np.any(penalty_factor < 0)
            or not np.any(penalty_factor > 0)
        ):
            raise ValueError(
                "penalty_factor must be finite and non-negative, with at least one positive entry."
            )

    if lambdas is not None:
        if np.iscomplexobj(lambdas):
            raise ValueError("lambdas must contain real values.")
        lambdas = np.array(lambdas, dtype=np.float64)
        if (
            lambdas.ndim != 1
            or lambdas.size == 0
            or not np.isfinite(lambdas).all()
            or np.any(lambdas < 0)
        ):
            raise ValueError(
                "lambdas must be a non-empty one-dimensional sequence of finite values >= 0."
            )
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
        penalty_factor=penalty_factor,
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
    on_lambda = None
    if _stop is not None:

        def on_lambda(il, lambdas, cols, vals, intercept, center, scale):
            beta, intercept = _unstandardize(vals, intercept, center, scale)
            return bool(_stop(il, lambdas, cols, beta, intercept))

    results = fit_path_rust(X, y, row_indices, params, n_threads, on_lambda, family)

    # keep the part of the path that was fit (it stops early on saturation
    # or dfmax), and untransform coefficients back to the scale of X
    fitted = np.flatnonzero(results["iters"] >= 0)
    col_map, center, scale = results["col_map"], results["center"], results["scale"]
    rows, vals, intercept = [], [], results["intercept"][fitted].copy()
    for k, il in enumerate(fitted):
        cols = col_map[results["coef_rows"][il]]
        beta, intercept[k] = _unstandardize(
            results["coef_vals"][il], intercept[k], center[cols], scale[cols]
        )
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
        # the solver tracks -log-likelihood, as R does; report true deviance
        deviance=(2.0 if family == "binomial" else 1.0) * results["deviance"][fitted],
        iterations=results["iters"][fitted],
        n_rejected=results["n_reject"][fitted],
        center=center,
        scale=scale,
        penalty_factor=penalty_factor,
        feature_names=feature_names,
        alpha=alpha,
        family=family,
    )


def _unstandardize(vals, intercept, center, scale):
    """Coefficients and intercept on X's scale, from those fit on standardized
    columns with this `center` and `scale`."""
    beta = vals / scale
    # fsum, not @: BLAS picks its summation order by CPU and thread count
    return beta, intercept - math.fsum(center * beta)


def fit_path_rust(
    X, y, row_indices, params, n_threads, on_lambda=None, family="binomial"
):
    """fit_path, run by the Rust backend in one call."""
    from . import _rust

    if not X.flags.aligned:
        # the kernels read X through typed pointers; numpy can build arrays
        # whose elements aren't aligned for their type (offset buffers)
        raise ValueError("X must be aligned to its dtype for the Rust solver.")
    path = _rust.fit_path(
        X,
        y,
        None if row_indices is None else row_indices.astype(np.int64),
        float(params.alpha),
        params.lambdas,
        params.nlambda,
        float(params.eps),
        float(params.lambda_min),
        bool(params.lambda_log_scale),
        params.penalty_factor,
        params.dfmax,
        params.max_iter,
        bool(params.stop_on_saturation),
        bool(params.r_compatible),
        n_threads or 0,
        settings.boundscheck,
        on_lambda=on_lambda,
        family=family,
    )
    path.pop("row_path")  # gathered or masked rows; reported for tests
    for kind, il in path.pop("events"):
        where = f"λ[{il + 1}/{params.nlambda}]"
        message = {
            "prefit_saturated": "Model saturated by the unpenalized columns alone; no path fit.",
            "dfmax": f"Exceeded dfmax={params.dfmax} nonzero coefficients; path truncated at {where}.",
            "saturated": f"Model saturated; path truncated at {where}.",
            "max_iter": f"Failed to converge within max_iter={params.max_iter} iterations at {where}.",
        }[kind]
        warnings.warn(message, stacklevel=3)
    return path
