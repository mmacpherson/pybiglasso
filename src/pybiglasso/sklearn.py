"""scikit-learn estimators; install with `pip install 'pybiglasso[sklearn]'`.

These follow scikit-learn's naming, which differs from R biglasso's:

    scikit-learn (here)   R biglasso / pybiglasso.biglasso
    -------------------   --------------------------------
    alpha                 lambda   (penalty strength)
    l1_ratio              alpha    (L1/L2 mix; 1 is the lasso)
"""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import check_cv
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data

from .binomial import biglasso
from .path import METRICS, _expit, linear_predictor

__all__ = ["BiglassoClassifier", "BiglassoClassifierCV"]


def _as_design(X):
    """X as the kernels read it: numeric dtypes and memmaps pass through uncopied.

    Booleans become a uint8 view, and float16 becomes float32 (numba has no
    float16 arithmetic); everything else numeric is used as is, so an int8
    genotype memmap stays an int8 memmap. Finiteness is checked by `biglasso`
    and `linear_predictor` as they read X, not by a separate pass here.
    """
    if X.dtype.kind == "b":
        return X.view(np.uint8)
    if X.dtype.kind == "f" and X.dtype.itemsize < 4:
        return X.astype(np.float32)
    return X


class _BiglassoBase(ClassifierMixin, BaseEstimator):
    def _validate_xy(self, X, y):
        X, y = validate_data(
            self, X, y, dtype="numeric", order=None, copy=False, ensure_all_finite=False
        )
        X = _as_design(X)
        check_classification_targets(y)
        y_type = type_of_target(y, input_name="y", raise_unknown=True)
        if y_type != "binary":
            raise ValueError(
                "Only binary classification is supported. The type of the target "
                f"is {y_type}."
            )
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError(
                "This solver needs samples of 2 classes in the data, but the data "
                f"contains only one class: {self.classes_[0]!r}"
            )
        if not 0 < self.l1_ratio <= 1:
            raise ValueError(f"l1_ratio must be in (0, 1]; got {self.l1_ratio}.")
        return X, (y == self.classes_[1]).astype(np.float64)

    def _fit_kwargs(self):
        return dict(
            alpha=self.l1_ratio,
            penalty_factor=self.penalty_factor,
            eps=self.tol,
            max_iter=self.max_iter,
        )

    def decision_function(self, X):
        check_is_fitted(self)
        # -- unlike fit, check every column for NaN/inf: scikit-learn requires it,
        #    and for float X it costs one read (a sum), not a copy
        X = validate_data(self, X, reset=False, dtype="numeric", order=None)
        return linear_predictor(_as_design(X), self.coef_.T, self.intercept_)[:, 0]

    def predict(self, X):
        check_is_fitted(self)
        return self.classes_[(self.decision_function(X) > 0).astype(np.intp)]

    def predict_proba(self, X):
        p = _expit(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.input_tags.sparse = False
        return tags


class BiglassoClassifier(_BiglassoBase):
    """Lasso / elastic-net logistic regression at a single penalty strength.

    Parameters
    ----------
    alpha : float
        Penalty strength (R biglasso's `lambda`), on standardized features.
    l1_ratio : float in (0, 1]
        Elastic-net mix (R biglasso's `alpha`); 1 is the lasso.
    penalty_factor : array of shape (n_features,), optional
        Per-feature penalty multipliers; 0 leaves a feature unpenalized.
    tol, max_iter : convergence settings, as R biglasso's `eps` and `max.iter`.
    """

    def __init__(
        self, alpha=1e-3, l1_ratio=1.0, penalty_factor=None, tol=1e-7, max_iter=1000
    ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.penalty_factor = penalty_factor
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, y):
        X, y = self._validate_xy(X, y)
        path = biglasso(
            X,
            y,
            lambdas=[self.alpha],
            stop_on_saturation=False,
            feature_names=getattr(self, "feature_names_in_", None),
            **self._fit_kwargs(),
        )
        self.coef_ = path.coef.toarray().T
        self.intercept_ = path.intercept.copy()
        self.n_iter_ = int(path.iterations[0])
        return self


class BiglassoClassifierCV(_BiglassoBase):
    """Lasso / elastic-net logistic regression, choosing alpha by cross-validation.

    Fits the full regularization path once on all the data, then once per fold
    on that fold's training rows (via `row_indices`, so X is never copied), and
    picks the penalty strength with the best held-out score.

    Parameters
    ----------
    l1_ratio : float in (0, 1]
        Elastic-net mix (R biglasso's `alpha`); 1 is the lasso.
    n_alphas : int
        Length of the penalty path (R biglasso's `nlambda`).
    alpha_min_ratio : float, optional
        Smallest alpha as a fraction of the largest (R biglasso's `lambda.min`).
    alphas : array, optional
        Explicit decreasing penalty strengths; overrides the two above.
    cv : int, cross-validation generator, or iterable
        As in scikit-learn; integers mean stratified k-fold.
    scoring : {"deviance", "auc", "accuracy"}
    rule : {"min", "1se"}
        Pick the best-scoring alpha, or the largest alpha within one standard
        error of it (glmnet's lambda.1se).
    penalty_factor, tol, max_iter : as in `BiglassoClassifier`.
    n_jobs : int, optional
        Folds to fit at once, in threads that share X (so a memmap is read
        through one page cache). Each fold is single-threaded; results do not
        depend on n_jobs. None or -1 uses all cores.

    Attributes
    ----------
    alpha_ : chosen penalty strength
    alphas_ : the path's penalty strengths
    cv_scores_ : (n_folds, n_alphas) held-out scores; deviance is per observation
    path_ : `BiglassoPath` fit on all the data
    coef_, intercept_ : coefficients at `alpha_`
    """

    def __init__(
        self,
        l1_ratio=1.0,
        n_alphas=100,
        alpha_min_ratio=None,
        alphas=None,
        cv=5,
        scoring="deviance",
        rule="min",
        penalty_factor=None,
        tol=1e-7,
        max_iter=1000,
        n_jobs=None,
    ):
        self.l1_ratio = l1_ratio
        self.n_alphas = n_alphas
        self.alpha_min_ratio = alpha_min_ratio
        self.alphas = alphas
        self.cv = cv
        self.scoring = scoring
        self.rule = rule
        self.penalty_factor = penalty_factor
        self.tol = tol
        self.max_iter = max_iter
        self.n_jobs = n_jobs

    def fit(self, X, y):
        if self.scoring not in METRICS:
            raise ValueError(f"scoring must be one of {METRICS}; got {self.scoring!r}")
        if self.rule not in ("min", "1se"):
            raise ValueError(f"rule must be 'min' or '1se'; got {self.rule!r}")
        X, y01 = self._validate_xy(X, y)
        kwargs = self._fit_kwargs()
        names = getattr(self, "feature_names_in_", None)

        self.path_ = path = biglasso(
            X,
            y01,
            nlambda=self.n_alphas,
            lambda_min=self.alpha_min_ratio,
            lambdas=self.alphas,
            feature_names=names,
            **kwargs,
        )
        grid = path.lambdas

        splitter = check_cv(self.cv, y01, classifier=True)
        folds = list(splitter.split(np.zeros((len(y01), 1)), y01))
        if self.scoring == "auc":
            for k, (_, test) in enumerate(folds):
                if len(np.unique(y01[test])) < 2:
                    raise ValueError(
                        f"The held-out rows of fold {k} are all one class, so AUC "
                        "is undefined there; use fewer folds, a stratified "
                        "splitter, or scoring='deviance'."
                    )
        scores = np.full((len(folds), len(grid)), np.nan)

        def score_fold(fold):
            train, test = fold
            fold_path = biglasso(
                X, y01, row_indices=train, lambdas=grid, n_threads=1, **kwargs
            )
            s = fold_path.score(X, y01, self.scoring, row_indices=test)
            return s / len(test) if self.scoring == "deviance" else s

        n_jobs = self.n_jobs
        if n_jobs is None or n_jobs < 0:
            n_jobs = os.cpu_count() or 1
        with ThreadPoolExecutor(max_workers=min(n_jobs, len(folds))) as pool:
            for k, s in enumerate(pool.map(score_fold, folds)):
                scores[k, : len(s)] = s
        self.cv_scores_ = scores
        self.alphas_ = grid

        # -- only consider alphas that every fold reached (paths can stop early)
        complete = ~np.isnan(scores).any(axis=0)
        if not complete.any():
            raise ValueError(
                "No alpha was fit in every fold; try a larger alpha range."
            )
        sign = 1.0 if self.scoring == "deviance" else -1.0  # lower is better
        mean = np.where(complete, sign * np.nanmean(scores, axis=0), np.inf)
        best = int(np.argmin(mean))
        if self.rule == "1se":
            se = np.nanstd(scores[:, best], ddof=1) / np.sqrt(len(folds))
            # -- alphas are decreasing, so the first qualifying index is largest
            best = int(np.flatnonzero(mean <= mean[best] + se)[0])

        self.alpha_ = grid[best]
        self.n_iter_ = int(path.iterations[best])
        self.coef_ = path.coef[:, [best]].toarray().T
        self.intercept_ = path.intercept[[best]].copy()
        return self
