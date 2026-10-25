"""The fitted regularization path returned by `pybiglasso.biglasso`."""

import dataclasses

import numpy as np
import scipy.sparse as sps

INTERCEPT = "(Intercept)"
METRICS = ("deviance", "auc", "accuracy")


@dataclasses.dataclass(frozen=True, eq=False)
class BiglassoPath:
    """A penalized logistic regression fit along a decreasing sequence of lambdas.

    Coefficients are on the original (unstandardized) scale of X.

    Attributes:
      lambdas: (nlambda,) penalty strengths, decreasing
      intercept: (nlambda,) intercept at each lambda
      coef: (ncol, nlambda) sparse coefficients, one column per lambda
      deviance: (nlambda,) training deviance, -2 log-likelihood, of the model at
        each lambda (with r_compatible=True, R's pre-final-sweep value, doubled)
      iterations: (nlambda,) coordinate-descent iterations at each lambda
      n_rejected: (nlambda,) columns excluded by the strong rule at each lambda
      center, scale: (ncol,) column means and standard deviations of X
      penalty_factor: (ncol,) per-column penalty multipliers
      feature_names: (ncol,) column names
      alpha: elastic-net mixing parameter (1 is the lasso)
    """

    lambdas: np.ndarray
    intercept: np.ndarray
    coef: sps.csc_array
    deviance: np.ndarray
    iterations: np.ndarray
    n_rejected: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    penalty_factor: np.ndarray
    feature_names: tuple
    alpha: float

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

    def decision_function(self, X, lambdas=None, row_indices=None):
        """Linear predictor (log-odds), shape (nrow, nlambda).

        Only the columns of X with a nonzero coefficient somewhere among the
        selected lambdas are read, which matters when X is a large memmap.
        With `row_indices`, predict for those rows of X only.
        """
        index = self._lambda_index(lambdas)
        if X.shape[1] != self.coef.shape[0]:
            raise ValueError(
                f"X has {X.shape[1]} columns; the model was fit on {self.coef.shape[0]}."
            )
        return linear_predictor(
            X, self.coef[:, index], self.intercept[index], row_indices
        )

    def predict_proba(self, X, lambdas=None, row_indices=None):
        """Probability that y == 1, shape (nrow, nlambda)."""
        return _expit(self.decision_function(X, lambdas, row_indices))

    def predict(self, X, lambdas=None, row_indices=None):
        """Predicted class (0 or 1), shape (nrow, nlambda)."""
        return (self.decision_function(X, lambdas, row_indices) > 0).astype(np.int8)

    def score(self, X, y, metric="deviance", lambdas=None, row_indices=None):
        """Evaluate the path on held-out data; one value per lambda.

        `metric` is "deviance" (lower is better), "auc", or "accuracy".
        `y` covers all rows of X; with `row_indices`, only those rows are scored.
        """
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}, not {metric!r}")
        y = np.asarray(y, dtype=np.float64)
        if row_indices is not None:
            y = y[row_indices]
        return score_logits(y, self.decision_function(X, lambdas, row_indices), metric)

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
            deviance=self.deviance,
            n_nonzero=self.n_nonzero,
            iterations=self.iterations,
            n_rejected=self.n_rejected,
        )
        features = dict(
            feature=names,
            center=self.center,
            scale=self.scale,
            penalty_factor=self.penalty_factor,
        )
        return dict(coef=coef, path=path, features=features)

    def to_polars(self, kind="coef"):
        """Export as a polars DataFrame.

        kind="coef": long format, one row per nonzero coefficient (and the
          intercept) per lambda: lambda_, feature, coefficient
        kind="path": one row per lambda: deviance, n_nonzero, iterations, ...
        kind="features": one row per column of X: center, scale, penalty_factor
        """
        try:
            import polars as pl
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "to_polars() needs polars: pip install 'pybiglasso[polars]'"
            ) from e
        data = self._frames()[kind]
        return pl.DataFrame(
            {k: (v.tolist() if v.dtype == object else v) for k, v in data.items()}
        )

    def to_pandas(self, kind="coef"):
        """Export as a pandas DataFrame; see `to_polars` for `kind`."""
        try:
            import pandas as pd
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "to_pandas() needs pandas: pip install 'pybiglasso[pandas]'"
            ) from e
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
    """
    coef = sps.csc_array(coef)
    used = np.unique(coef.indices)
    nrow = X.shape[0] if row_indices is None else len(row_indices)
    eta = np.broadcast_to(intercept, (nrow, coef.shape[1])).copy()
    if len(used) == 0:
        return eta
    width = min(len(used), CHUNK_ELEMENTS)
    height = CHUNK_ELEMENTS // width
    for c0 in range(0, len(used), width):
        cols = used[c0 : c0 + width]
        beta = sps.csr_array(coef[cols, :])
        for start in range(0, nrow, height):
            stop = min(start + height, nrow)
            if row_indices is None:
                block = X[start:stop][:, cols]
            else:
                block = X[np.ix_(row_indices[start:stop], cols)]
            block = np.asarray(block, dtype=np.float64)
            if not np.isfinite(block).all():
                raise ValueError("X contains NaN or infinity.")
            eta[start:stop] += block @ beta
    return eta


# rows x columns of X gathered at once by `linear_predictor` (32 MB as float64)
CHUNK_ELEMENTS = 1 << 22


def _expit(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def binomial_deviance(y, eta):
    """Deviance, -2 * log-likelihood, of 0/1 labels `y` under logits `eta`.

    `eta` is (nrow,) or (nrow, k); returns a scalar or (k,) array.
    """
    y = np.asarray(y, dtype=np.float64)
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
