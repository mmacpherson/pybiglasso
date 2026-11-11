import collections
import logging
import math
import numbers
import warnings

import numpy as np
import scipy.sparse as sps

from ._threads import resolve_threads
from .arrow import _ArrowMatrix, _as_array, _as_arrow_matrix, _response
from .config import settings
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

_ValidatedInput = collections.namedtuple(
    "_ValidatedInput",
    "X y row_indices nrow ncol n_threads lambda_min penalty_factor lambdas nlambda "
    "dfmax feature_names named_features",
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
    n_threads="auto",
    _stop=None,
):
    """Fit a lasso or elastic-net binomial or Gaussian regression path.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Numeric NumPy array or memory map, preferably in Fortran order, or a
        PyArrow Table or RecordBatch. Columns are read without copying X.
    y : array-like of shape (n_samples,)
        Numeric response. Binomial fits require both 0 and 1 among the rows
        used; Gaussian fits require finite values there.
    alpha : float, default=1.0
        Elastic-net mixing parameter in (0, 1]. One is the lasso; smaller
        values mix L1 and L2 penalties.
    family : {"gaussian", "binomial"}, default="gaussian"
        Gaussian least-squares or binomial logistic regression.
    nlambda : int, default=100
        Number of penalty strengths in an automatically generated path. Must
        be at least two; use ``lambdas=[value]`` for one fit.
    lambdas : array-like, optional
        Explicit nonnegative penalty strengths in decreasing order. When
        supplied, these replace the automatic grid and ``nlambda``.
    eps : float, default=1e-7
        Positive convergence tolerance, corresponding to R biglasso's ``eps``.
    lambda_min : float, optional
        Smallest automatic penalty as a fraction of the largest. The default
        is 0.001 when fitted rows outnumber features and 0.05 otherwise.
    lambda_log_scale : bool, default=True
        Space an automatic grid geometrically when true, linearly when false.
    penalty_factor : array-like of shape (n_features,), optional
        Nonnegative multiplier for each feature's penalty. Zero leaves a
        feature unpenalized; at least one multiplier must be positive.
    row_indices : array-like, optional
        Nonempty integer indices of rows to fit, without copying those rows.
        Predictions and path metadata still refer to the original X.
    dfmax : int, optional
        Stop after a fit exceeds this many nonzero feature coefficients. The
        default, ``n_features + 1``, does not truncate on model size.
    max_iter : int, default=1000
        Maximum coordinate-descent iterations at each penalty strength.
    feature_names : sequence of str, optional
        One name per feature for path metadata and named-column prediction.
        Arrow and dataframe column names are inferred when this is omitted.
    stop_on_saturation : bool, default=True
        For binomial fits, stop when training deviance falls below 1% of null
        deviance, as R biglasso does. False attempts every penalty strength.
    r_compatible : bool, default=False
        Reproduce R biglasso's results for both families. This includes its
        centering sums, logistic elastic-net update and probability cutoff,
        Gaussian residual sum, strict constant-response convergence comparison,
        and loss bookkeeping after an iteration limit. See README.md's
        "Differences from R biglasso" for the full list.
    n_threads : {"auto", None, int}, default="auto"
        Threads used by parallel native reductions. Auto mode uses one thread
        for Arrow, memory-mapped, or small designs and otherwise uses available
        physical cores, capped by ``RAYON_NUM_THREADS``. Explicit positive
        integers override that policy. Results do not depend on thread count.

    Returns
    -------
    BiglassoPath
        Fitted path with ``lambdas``, sparse ``coef``, ``intercept``,
        ``deviance``, ``iterations``, ``n_rejected``, centering and scaling,
        penalty factors, feature names, ``alpha``, and ``family``.

    Notes
    -----
    On standardized columns, the binomial objective is ``(1/n) * NLL + lambda
    * sum_j pf_j * (alpha * |b_j| + (1 - alpha)/2 * b_j^2)``. The Gaussian
    objective replaces ``(1/n) * NLL`` with ``RSS/(2n)`` and uses the same
    penalty. The intercept is unpenalized. R biglasso's logistic elastic-net
    update solves a slightly different problem, reproduced by
    ``r_compatible=True``.

    Automatic grids reject designs without varying penalized features. An
    exactly zero largest penalty produces a finite all-zero grid. Default-mode
    deviance is evaluated at the final fitted model.
    """
    # _stop (private, for BiglassoClassifierCV's early stopping): called as
    # _stop(il, lambdas, cols, beta, intercept) after each lambda is fit, with
    # the whole grid and that fit's nonzero coefficients on X's scale; a true
    # result ends the path there, as dfmax does, but without a warning

    data = _validate_args(
        X,
        y,
        alpha,
        family,
        nlambda,
        lambdas,
        eps,
        lambda_min,
        penalty_factor,
        row_indices,
        dfmax,
        max_iter,
        feature_names,
        n_threads,
    )

    params = BigLassoParams(
        alpha=alpha,
        lambdas=data.lambdas,
        nlambda=data.nlambda,
        eps=eps,
        lambda_min=data.lambda_min,
        lambda_log_scale=lambda_log_scale,
        penalty_factor=data.penalty_factor,
        dfmax=data.dfmax,
        max_iter=max_iter,
        stop_on_saturation=stop_on_saturation,
        r_compatible=r_compatible,
    )
    logger.info(
        "Starting fit. Design matrix size: [%d, %d]  Num λ values: %d",
        data.nrow,
        data.ncol,
        data.nlambda,
    )
    on_lambda = None
    if _stop is not None:

        def on_lambda(il, lambdas, cols, vals, intercept, center, scale):
            beta, intercept = _unstandardize(vals, intercept, center, scale)
            return bool(_stop(il, lambdas, cols, beta, intercept))

    results = _fit_path(
        data.X,
        data.y,
        data.row_indices,
        params,
        data.n_threads,
        on_lambda,
        family,
    )
    return _assemble_path(results, data, alpha, family)


def _validate_args(
    X,
    y,
    alpha,
    family,
    nlambda,
    lambdas,
    eps,
    lambda_min,
    penalty_factor,
    row_indices,
    dfmax,
    max_iter,
    feature_names,
    n_threads,
):
    X = _as_arrow_matrix(X)
    y = _response(y)
    if feature_names is None and hasattr(X, "columns"):
        feature_names = [str(c) for c in X.columns]
    X = _as_array(X)
    mapped = isinstance(X, _ArrowMatrix)
    y = np.asarray(y)

    if (not mapped and X.ndim != 2) or not all(X.shape):
        raise ValueError("X must be a non-empty two-dimensional design matrix.")
    if not mapped and (
        X.dtype.kind not in "biuf" or X.dtype == np.float16 or not X.dtype.isnative
    ):
        raise ValueError(
            "X must have a native real numeric dtype (float32/64 or integer)."
        )
    if not mapped and X.dtype.kind == "f" and X.dtype.itemsize not in (4, 8):
        raise ValueError("Floating-point X must have dtype float32 or float64.")
    if y.ndim != 1 or y.dtype.kind not in "biuf":
        raise ValueError("y must be a one-dimensional numeric response.")
    _finite_scalar("Alpha", alpha)
    _finite_scalar("eps", eps)
    if eps <= 0:
        raise ValueError("eps must be positive.")
    _integer("max_iter", max_iter)
    if lambdas is None and (
        isinstance(nlambda, (bool, np.bool_))
        or not isinstance(nlambda, numbers.Integral)
        or nlambda < 2
    ):
        raise ValueError(
            "nlambda must be at least 2 (an integer); to fit one value, "
            "pass lambdas=[value]."
        )
    if dfmax is not None:
        _integer("dfmax", dfmax, minimum=0)
    if family not in ("binomial", "gaussian"):
        raise NotImplementedError("Supported families are 'binomial' and 'gaussian'.")
    # not np.isfortran, which is False for one column: both C and F order
    if not mapped and not X.flags.f_contiguous:
        warnings.warn(
            "The design matrix is not in Fortran (column-major) order; "
            "this algorithm performs best on Fortran-ordered arrays.",
            stacklevel=3,
        )

    if alpha == 0:
        raise NotImplementedError(
            "Sorry, this implementation does not support zero alpha."
        )
    if not 0 < alpha <= 1:
        raise ValueError(f"Alpha must be in (0, 1]; you gave [{alpha}]")
    if X.shape[0] != len(y):
        raise ValueError(
            f"Number of rows in X [{X.shape[0]}] is different from length of y "
            f"[{len(y)}]."
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
                "row_indices must be a non-empty one-dimensional sequence of "
                "integer indices."
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
    n_threads = resolve_threads(n_threads, X, nrow)

    if feature_names is not None:
        feature_names = tuple(str(f) for f in feature_names)
        if len(feature_names) != ncol:
            raise ValueError("feature_names must have one entry per column of X.")
        if INTERCEPT in feature_names:
            raise ValueError(f"{INTERCEPT!r} is reserved for the intercept.")

    # Differs from R biglasso: R compares nrow(X) with ncol(X); this
    # compares the rows actually fit, which differ when row_indices is given
    if lambda_min is None:
        lambda_min = 0.001 if nrow > ncol else 0.05
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
                "penalty_factor must be finite and non-negative, with at least "
                "one positive entry."
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
                "lambdas must be a non-empty one-dimensional sequence of finite "
                "values >= 0."
            )
        if np.any(np.diff(lambdas) > 0):
            raise ValueError("lambdas must be in decreasing order.")
        nlambda = len(lambdas)

    if dfmax is None:
        dfmax = ncol + 1
    named_features = feature_names is not None
    if feature_names is None:
        feature_names = tuple(f"x{index}" for index in range(ncol))

    return _ValidatedInput(
        X=X,
        y=y.astype(np.float64),
        row_indices=row_indices,
        nrow=nrow,
        ncol=ncol,
        n_threads=n_threads,
        lambda_min=lambda_min,
        penalty_factor=penalty_factor,
        lambdas=lambdas,
        nlambda=nlambda,
        dfmax=dfmax,
        feature_names=feature_names,
        named_features=named_features,
    )


def _assemble_path(results, data, alpha, family):
    if imprecise := results.pop("imprecise_center", []):
        names = ", ".join(data.feature_names[index] for index in imprecise[:5])
        more = f" and {len(imprecise) - 5} more" if len(imprecise) > 5 else ""
        warnings.warn(
            f"Column(s) {names}{more} of X are too far from zero, relative to their "
            "spread, for float64 to center them precisely, which can make the fit, "
            "its intercept especially, inaccurate. Subtract a reference value, such "
            "as each column's minimum, before fitting.",
            stacklevel=3,
        )

    # Keep the fitted prefix (saturation and dfmax can stop it early), then
    # return coefficients to X's original scale.
    fitted = np.flatnonzero(results["iters"] >= 0)
    col_map, center, scale = results["col_map"], results["center"], results["scale"]
    rows, vals, intercept = [], [], results["intercept"][fitted].copy()
    for output_index, lambda_index in enumerate(fitted):
        cols = col_map[results["coef_rows"][lambda_index]]
        beta, intercept[output_index] = _unstandardize(
            results["coef_vals"][lambda_index],
            intercept[output_index],
            center[cols],
            scale[cols],
        )
        rows.append(cols)
        vals.append(beta)
    indptr = np.concatenate([[0], np.cumsum([len(row) for row in rows])])
    coef = sps.csc_array(
        (
            np.concatenate(vals) if vals else np.zeros(0),
            np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64),
            indptr,
        ),
        shape=(data.ncol, len(fitted)),
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
        penalty_factor=data.penalty_factor,
        feature_names=data.feature_names,
        alpha=alpha,
        family=family,
        named_features=data.named_features,
    )


def _unstandardize(vals, intercept, center, scale):
    """Coefficients and intercept on X's scale, from those fit on standardized
    columns with this `center` and `scale`."""
    beta = vals / scale
    # fsum, not @: BLAS picks its summation order by CPU and thread count
    return beta, intercept - math.fsum(center * beta)


def _fit_path(X, y, row_indices, params, n_threads, on_lambda, family):
    """Run the native solver for a complete regularization path."""
    from . import _rust

    if not isinstance(X, _ArrowMatrix) and not X.flags.aligned:
        # the kernels read X through typed pointers; numpy can build arrays
        # whose elements aren't aligned for their type (offset buffers)
        raise ValueError("X must be aligned to its dtype for the Rust solver.")
    path = _rust.fit_path(
        (X.shape[0], X._columns) if isinstance(X, _ArrowMatrix) else X,
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
    events = path.pop("events")
    path["imprecise_center"] = [j for kind, j in events if kind == "imprecise_center"]
    for kind, il in events:
        if kind == "imprecise_center":
            continue
        where = f"λ[{il + 1}/{params.nlambda}]"
        message = {
            "prefit_saturated": "Model saturated by the unpenalized columns alone; no path fit.",
            "dfmax": f"Exceeded dfmax={params.dfmax} nonzero coefficients; path truncated at {where}.",
            "saturated": f"Model saturated; path truncated at {where}.",
            "max_iter": f"Failed to converge within max_iter={params.max_iter} iterations at {where}.",
        }[kind]
        warnings.warn(message, stacklevel=3)
    return path
