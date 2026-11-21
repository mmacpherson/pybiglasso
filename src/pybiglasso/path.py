"""The fitted regularization path returned by `pybiglasso.biglasso`."""

import dataclasses
from typing import NamedTuple

import numpy as np
import scipy.sparse as sps
from scipy.special import expit as _expit

from .arrow import _ArrowMatrix, _as_array, _as_arrow_matrix, _response

INTERCEPT = "(Intercept)"
METRICS = ("deviance", "auc", "accuracy")


@dataclasses.dataclass(frozen=True, eq=False)
class BiglassoPath:
    """A binomial or Gaussian regression fit along decreasing lambdas.

    Coefficients are on the original (unstandardized) scale of X. The model at
    lambda k is ``centered_intercept[k] + sum_j coef[j, k] * ((x_j - center_j)
    - center_offset_j)``, the offset applying where ``anchored`` is true;
    ``intercept[k] + X @ coef[:, k]`` is the same model on X's raw scale.

    Attributes:
      lambdas: (nlambda,) penalty strengths, decreasing
      intercept: (nlambda,) intercept on X's raw scale at each lambda; NaN
        where that isn't a representable float64 (a warning says so)
      centered_intercept: (nlambda,) the model's value where every column is
        at its center (plus its offset, if anchored)
      coef: (ncol, nlambda) sparse coefficients, one column per lambda
      deviance: (nlambda,) training deviance: RSS for Gaussian, twice negative
        log likelihood for binomial. r_compatible=True retains R's loss bookkeeping.
      iterations: (nlambda,) coordinate-descent iterations at each lambda
      n_rejected: (nlambda,) columns excluded by the strong rule at each lambda
      center, scale: (ncol,) column means and standard deviations of X
      center_offset: (ncol,) what anchored columns subtract after the center
      anchored: (ncol,) whether each column is standardized as
        ((x - center) - center_offset) / scale
      penalty_factor: (ncol,) per-column penalty multipliers
      feature_names: (ncol,) column names
      named_features: whether feature_names came from the user or X's
        columns, rather than being the defaults x0, x1, ...
      alpha: elastic-net mixing parameter (1 is the lasso)
      family: "binomial" or "gaussian"
      r_compatible: whether the path was fit with r_compatible=True, which
        also predicts and scores with the raw coefficients and intercept, as
        R biglasso does
    """

    lambdas: np.ndarray
    intercept: np.ndarray
    centered_intercept: np.ndarray
    coef: sps.csc_array
    deviance: np.ndarray
    iterations: np.ndarray
    n_rejected: np.ndarray
    center: np.ndarray
    center_offset: np.ndarray
    anchored: np.ndarray
    scale: np.ndarray
    penalty_factor: np.ndarray
    feature_names: tuple
    alpha: float
    family: str
    named_features: bool
    r_compatible: bool

    def __setstate__(self, state):
        _require_current_state(self, state, _PATH_STATE)
        self.__dict__.update(state)

    def __len__(self):
        return len(self.lambdas)

    def __repr__(self):
        span = (
            f" from {self.lambdas[0]:.3g} to {self.lambdas[-1]:.3g}"
            if len(self)
            else ""
        )
        return (
            f"BiglassoPath({self.coef.shape[0]} features, {len(self)} lambdas"
            f"{span}, alpha={self.alpha})"
        )

    @property
    def n_nonzero(self):
        """Number of nonzero coefficients (excluding the intercept) at each lambda."""
        return np.diff(self.coef.indptr)

    def _lambda_index(self, lambdas):
        if lambdas is None:
            return np.arange(len(self))
        lambdas = np.atleast_1d(np.asarray(lambdas, dtype=np.float64))
        index = np.empty(len(lambdas), dtype=np.int64)
        for k, lam in enumerate(lambdas):
            hits = np.flatnonzero(np.isclose(self.lambdas, lam, rtol=1e-10, atol=0))
            if len(hits) == 0:
                raise ValueError(f"lambda {lam} is not on the fitted path")
            index[k] = hits[0]
        return index

    def _check_column_names(self, X):
        """Reject named columns that aren't the fitted ones in order: a
        coefficient would otherwise be applied to the wrong feature. A path
        fit without names checks the column count only."""
        if not self.named_features or not hasattr(X, "columns"):
            return
        names = tuple(str(c) for c in X.columns)
        if len(set(names)) != len(names):
            raise ValueError(
                "X's column names are ambiguous: some repeat, or are equal once "
                "written as strings (1 and '1', say). Pass X as an array in the "
                "fitted column order, or rename its columns."
            )
        fitted = tuple(self.feature_names)
        if names == fitted:
            return
        if sorted(names) == sorted(fitted):
            raise ValueError(
                "X has the fitted columns in a different order; select them in "
                "the fitted order first, by their original labels (with string "
                "labels: X[list(path.feature_names)] for pandas or Polars, "
                "X.select(list(path.feature_names)) for an Arrow table)."
            )
        raise ValueError("X's column names don't match the fitted feature_names.")

    def decision_function(self, X, lambdas=None, row_indices=None):
        """Linear predictor (log-odds for binomial), shape (nrow, nlambda).

        Only the columns of X with a nonzero coefficient somewhere among the
        selected lambdas are read, which matters when X is a large memmap.
        With `row_indices`, predict for those rows of X only. If X names its
        columns (a DataFrame or Arrow table) and the path was fit with names,
        they must be the fitted `feature_names`, in the same order. A path fit
        on an unnamed array, or unpickled from before names were recorded,
        checks only the number of columns.
        """
        X = _as_arrow_matrix(X)
        index = self._lambda_index(lambdas)
        if X.shape[1] != self.coef.shape[0]:
            raise ValueError(
                f"X has {X.shape[1]} columns; the model was fit on {self.coef.shape[0]}."
            )
        self._check_column_names(X)
        return _predictor(
            X,
            self.coef[:, index],
            Scoring(
                self.intercept[index],
                self.centered_intercept[index],
                self.center,
                self.center_offset,
                self.anchored,
                self.r_compatible,
            ),
            row_indices,
        )

    def predict_proba(self, X, lambdas=None, row_indices=None):
        """Probability that y == 1, shape (nrow, nlambda)."""
        if self.family != "binomial":
            raise ValueError("predict_proba is only available for binomial paths.")
        return _expit(self.decision_function(X, lambdas, row_indices))

    def predict(self, X, lambdas=None, row_indices=None):
        """Predicted class or Gaussian response, shape (nrow, nlambda)."""
        if self.family == "gaussian":
            return self.decision_function(X, lambdas, row_indices)
        return (self.decision_function(X, lambdas, row_indices) > 0).astype(np.int8)

    def score(self, X, y, metric="deviance", lambdas=None, row_indices=None):
        """Evaluate the path on held-out data; one value per lambda.

        Binomial metrics are "deviance", "auc", and "accuracy"; Gaussian
        metrics are "deviance" (RSS), "mse", and "r2".
        `y` covers all rows of X; with `row_indices`, only those rows are scored.
        """
        metrics = ("deviance", "mse", "r2") if self.family == "gaussian" else METRICS
        if metric not in metrics:
            raise ValueError(f"metric must be one of {metrics}, not {metric!r}")
        y = np.asarray(_response(y), dtype=np.float64)
        if row_indices is not None:
            y = y[row_indices]
        eta = self.decision_function(X, lambdas, row_indices)
        if self.family == "gaussian":
            rss = np.square(eta - y[:, None]).sum(axis=0)
            if metric == "deviance":
                return rss
            if metric == "mse":
                return rss / len(y)
            total = np.square(y - y.mean()).sum()
            return 1 - rss / total if total > 0 else (rss == 0).astype(float)
        return score_logits(y, eta, metric)

    def _frames(self):
        """Column data for the dataframe exports."""
        names = np.asarray(self.feature_names, dtype=object)
        coo = self.coef.tocoo()
        coef = dict(
            lambda_=np.concatenate([self.lambdas, self.lambdas[coo.col]]),
            feature=np.concatenate(
                [np.full(len(self), INTERCEPT, dtype=object), names[coo.row]]
            ),
            coefficient=np.concatenate([self.intercept, coo.data]),
        )
        path = dict(
            lambda_=self.lambdas,
            centered_intercept=self.centered_intercept,
            deviance=self.deviance,
            n_nonzero=self.n_nonzero,
            iterations=self.iterations,
            n_rejected=self.n_rejected,
        )
        features = dict(
            feature=names,
            center=self.center,
            center_offset=self.center_offset,
            anchored=self.anchored,
            scale=self.scale,
            penalty_factor=self.penalty_factor,
        )
        return dict(coef=coef, path=path, features=features)

    def to_polars(self, kind="coef"):
        """Export as a polars DataFrame.

        kind="coef": long format, one row per nonzero coefficient (and the
          intercept) per lambda: lambda_, feature, coefficient
        kind="path": one row per lambda: centered_intercept, deviance,
          n_nonzero, iterations, ...
        kind="features": one row per column of X: center, center_offset,
          anchored, scale, penalty_factor
        """
        try:
            import polars as pl
        except ImportError as e:
            raise ImportError("to_polars() needs polars: pip install polars") from e
        data = self._frames()[kind]
        return pl.DataFrame(
            {k: (v.tolist() if v.dtype == object else v) for k, v in data.items()}
        )

    def to_pandas(self, kind="coef"):
        """Export as a pandas DataFrame; see `to_polars` for `kind`."""
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("to_pandas() needs pandas: pip install pandas") from e
        return pd.DataFrame(self._frames()[kind])


def score_logits(y, eta, metric):
    """`metric` of 0/1 labels `y` under (nrow, k) logits `eta`; one value per
    column."""
    if metric == "deviance":
        return binomial_deviance(y, eta)
    if metric == "auc":
        return np.array([roc_auc(y, eta[:, k]) for k in range(eta.shape[1])])
    return ((eta > 0) == (y[:, np.newaxis] == 1)).mean(axis=0)


def linear_predictor(X, coef, intercept, row_indices=None):
    """`X[row_indices] @ coef + intercept`, without copying X.

    `coef` is (ncol, k), sparse or dense; `intercept` is (k,). Only the columns
    with a nonzero coefficient are read, in blocks of rows and columns, so at
    most CHUNK_ELEMENTS values of X are converted to float64 at once, whatever
    X's dtype or size; coefficients stay sparse. Besides the (nrow, k) result,
    that bounds the memory used. Raises ValueError if the values read aren't
    all finite.

    This is the raw form of a fitted model, as R biglasso predicts. For
    columns far from zero its rounding error grows with their magnitude; a
    path's own `decision_function` uses the centered form in default mode.
    """
    X = _as_array(_as_arrow_matrix(X))
    coef = sps.csc_array(coef)
    eta = np.broadcast_to(intercept, (_nrow(X, row_indices), coef.shape[1])).copy()
    for start, stop, _, beta, block in _blocks(X, coef, row_indices, CHUNK_ELEMENTS):
        eta[start:stop] += block @ beta
    return eta


def centered_predictor(
    X, coef, centered_intercept, center, center_offset, anchored, row_indices=None
):
    """`centered_intercept + sum_j coef_j * ((x_j - center_j) - offset_j)`.

    The offset applies to `anchored` columns. Reads X as `linear_predictor`
    does, without copying it, subtracting each block's centers into a new
    array rather than into the block, which may be the caller's own memory;
    block and centered copy together stay within the same 32 MB.

    Forward error, for the stored floats: with k nonzero coefficients at a
    lambda and M = |centered_intercept| + sum_j |coef_j| (|x_j - center_j| +
    |offset_j|), the result is within gamma_(k+3) M + 2 k eta of the exact
    value (u = 2**-53, gamma_m = m u / (1 - m u), eta = 2**-1074), under
    round-to-nearest with gradual underflow, for (k + 3) u < 1/2. Each term
    meets at most k + 3 roundings: two subtractions, one product and at most k
    additions. scipy evaluates `block @ beta` as a column-ordered axpy into a
    zeroed output (csc_matvecs), and every block's partial sum is then added
    to the intercept, so each output takes at most k additions in all. The
    bound doesn't grow with a column's distance from zero, unlike the raw
    form's. Raises ValueError if a value read or a prediction isn't finite.
    """
    X = _as_array(_as_arrow_matrix(X))
    coef = sps.csc_array(coef)
    eta = np.broadcast_to(
        centered_intercept, (_nrow(X, row_indices), coef.shape[1])
    ).copy()
    # each block and its centered copy: the same budget as linear_predictor's
    blocks = _blocks(X, coef, row_indices, max(1, CHUNK_ELEMENTS // 2))
    for start, stop, cols, beta, block in blocks:
        centered = block - center[cols]
        offset = anchored[cols]
        if offset.any():
            centered[:, offset] -= center_offset[cols][offset]
        eta[start:stop] += centered @ beta
    if not np.isfinite(eta).all():
        raise ValueError("X has values too large to predict with this model.")
    return eta


class Scoring(NamedTuple):
    """What a fitted model predicts with, at its selected lambdas: the raw
    pair, or the centered form, as `r_compatible` chooses."""

    intercept: np.ndarray
    centered_intercept: np.ndarray
    center: np.ndarray
    center_offset: np.ndarray
    anchored: np.ndarray
    r_compatible: bool


def _predictor(X, coef, scoring, row_indices=None):
    """The linear predictor of a fitted model, in the form its mode scores
    with."""
    if scoring.r_compatible:
        if np.isnan(scoring.intercept).any():
            raise ValueError(
                "This r_compatible model's raw intercept is not representable, "
                "so it can't be scored on the raw scale; fit with "
                "r_compatible=False for centered scoring."
            )
        return linear_predictor(X, coef, scoring.intercept, row_indices)
    # Differs from R biglasso: R predicts with the raw coefficients and
    # intercept. Default mode evaluates the same model in centered form, whose
    # rounding error doesn't grow with a column's distance from zero;
    # r_compatible=True keeps the raw form.
    return centered_predictor(
        X,
        coef,
        scoring.centered_intercept,
        scoring.center,
        scoring.center_offset,
        scoring.anchored,
        row_indices,
    )


def _nrow(X, row_indices):
    return X.shape[0] if row_indices is None else len(row_indices)


def _blocks(X, coef, row_indices, chunk):
    """(start, stop, cols, beta, block) over the columns of X with a nonzero
    coefficient, in blocks of at most `chunk` values converted to float64."""
    used = np.unique(coef.indices)
    if len(used) == 0:
        return
    nrow = _nrow(X, row_indices)
    width = min(len(used), chunk)
    height = chunk // width
    for c0 in range(0, len(used), width):
        cols = used[c0 : c0 + width]
        beta = sps.csr_array(coef[cols, :])
        for start in range(0, nrow, height):
            stop = min(start + height, nrow)
            if isinstance(X, _ArrowMatrix):
                rows = (
                    slice(start, stop)
                    if row_indices is None
                    else row_indices[start:stop]
                )
                block = X._block(rows, cols)
            elif row_indices is None:
                block = X[start:stop][:, cols]
            else:
                block = X[np.ix_(row_indices[start:stop], cols)]
            block = np.asarray(block, dtype=np.float64)
            if not np.isfinite(block).all():
                raise ValueError("X contains NaN or infinity.")
            yield start, stop, cols, beta, block


# rows x columns of X gathered at once by `linear_predictor` (32 MB as
# float64); `centered_predictor` gathers half as many, beside their centered copy
CHUNK_ELEMENTS = 1 << 22

# Every field a fitted path records; an older pickle lacks some of them.
_PATH_STATE = frozenset(f.name for f in dataclasses.fields(BiglassoPath))


def _require_current_state(obj, state, fields):
    """Refuse fitted state from an earlier pybiglasso, which can't be scored
    the way this one scores: it lacks the centered state predictions use."""
    missing = fields - set(state)
    if missing:
        raise ValueError(
            f"This {type(obj).__name__} was saved by an earlier version of "
            "pybiglasso that didn't record the fitted state predictions now use "
            f"({', '.join(sorted(missing))}). Refit it with this version."
        )


def binomial_deviance(y, eta):
    """Deviance, -2 * log-likelihood, of 0/1 labels `y` under logits `eta`.

    `eta` is (nrow,) or (nrow, k); returns a scalar or (k,) array.
    """
    y = np.asarray(_response(y), dtype=np.float64)
    if eta.ndim == 2:
        y = y[:, np.newaxis]
    # log(1 + exp(eta)) - y * eta, computed stably
    return 2.0 * (np.logaddexp(0.0, eta) - y * eta).sum(axis=0)


def roc_auc(y, score):
    """Area under the ROC curve, via the Mann-Whitney U statistic (ties averaged)."""
    y = np.asarray(y) == 1
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    sorted_scores = score[order]
    # average ranks within ties
    _, first, counts = np.unique(sorted_scores, return_index=True, return_counts=True)
    avg = first + (counts + 1) / 2.0
    ranks[order] = np.repeat(avg, counts)
    return (ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
