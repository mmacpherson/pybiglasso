#!/usr/bin/env python3

import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.metrics
from numpy.random import default_rng
from sklearn.linear_model import LogisticRegression

DEFAULT_SEED = 123 * 321
DEFAULT_DTYPE = np.float64


def get_iris_dataset(seed=DEFAULT_SEED, dtype=DEFAULT_DTYPE):

    iris = sklearn.datasets.load_iris()
    X = iris["data"].astype(dtype)
    y = (iris["target"] == 2).astype(dtype)  # 1 if Iris-Virginica, else 0'

    X = (X - X.mean(axis=0)) / X.std(axis=0)

    log_reg = LogisticRegression(solver="lbfgs", C=1e5, fit_intercept=False)
    log_reg.fit(X, y)

    return dict(w=log_reg.coef_.ravel(), X=X, y=y)


def generate_test_data(
    num_rows=50,
    num_features=5,
    dtype=DEFAULT_DTYPE,
    seed=DEFAULT_SEED,
    fortran_order=True,
):

    rng = default_rng(seed)

    features = rng.standard_normal(
        size=(num_rows, num_features),
        dtype=dtype,
    )

    if fortran_order:
        features = np.asfortranarray(features)

    weights = rng.standard_normal(size=num_features, dtype=dtype)
    weights = np.sqrt(2) * weights / np.linalg.norm(weights)

    s = 1 / (1 + np.exp(-1.0 * features @ weights))

    out = (rng.random(size=num_rows, dtype=dtype) < s).astype(dtype)

    return dict(
        y=out,
        X=features,
        w=weights,
    )


def sample_biglasso_weights(
    rng,
    ncol,
    proportion_associated_markers=0.5,
    marker_effect_sd=0.5,
):
    """Sample sparse, heavy-tailed marker effects.

    `rng` is a `numpy.random.Generator`, or anything `default_rng` accepts.
    """
    rng = default_rng(rng)
    is_associated = rng.random(ncol) < proportion_associated_markers
    raw_weights = marker_effect_sd * rng.standard_t(3, size=ncol)
    return is_associated * raw_weights


def biglasso_generator(
    rng,
    weights,
    nrow=50,
    allele_frequency_alpha=0.4,
    allele_frequency_beta=0.4,
    dtype=DEFAULT_DTYPE,
    fortran_order=True,
):
    """Yield an endless stream of independent PRS-like batches of `nrow` rows."""
    rng = default_rng(rng)
    weights = np.asarray(weights, dtype=dtype)
    order_str = "F" if fortran_order else "C"

    while True:
        # -- generate dosages array
        allele_frequencies = rng.beta(
            allele_frequency_alpha, allele_frequency_beta, size=len(weights)
        )
        dosages = rng.binomial(2, allele_frequencies, size=(nrow, len(weights)))

        # -- generate response
        probs = logit_to_prob(dosages @ weights)
        response = rng.random(nrow) < probs

        yield dict(
            X=np.asarray(dosages, dtype=dtype, order=order_str),
            w=weights,
            y=response.astype(dtype),
        )


def make_biglasso_dataset(
    nrow=50,
    ncol=5,
    proportion_associated_markers=0.5,
    allele_frequency_alpha=0.4,
    allele_frequency_beta=0.4,
    marker_effect_sd=0.5,
    dtype=DEFAULT_DTYPE,
    seed=DEFAULT_SEED,
    fortran_order=True,
):
    """Generate PRS-like dataset.

    This wrapper is good when you know your synthetic dataset will fit in
    memory; if it won't, you can drop down and use the generator functions this
    function uses.
    """
    rng = default_rng(seed)
    weights = sample_biglasso_weights(
        rng, ncol, proportion_associated_markers, marker_effect_sd
    )
    return next(
        biglasso_generator(
            rng,
            weights,
            nrow,
            allele_frequency_alpha,
            allele_frequency_beta,
            dtype,
            fortran_order,
        )
    )


def logit_to_prob(x):
    odds = np.exp(x)
    return odds / (1 + odds)


def predict(bl_fit, Xnew, lam=None, output="logit"):
    """Predict biglasso response given design matrix `Xnew`.

    Inputs:
      bl_fit: output of biglasso() call
      Xnew: ndarray (nrow, ncol)
      lam: iterable (nlambda) of lambda values; must be present in bl_fit

    Output:
      : ndarray (nrow, nlambda) of predictions (in nlambda=1 case, drops 2nd dim)


    If `lam` is None, prediction is done for all lambda values present in the
    fitted model object.

    Otherwise, predictions will be given for specified values of `lam`.

    """

    if output not in ("logit", "prob", "class"):
        raise ValueError(
            f"Only 'logit', 'prob', and 'class' output are supported, not [{output}]"
        )

    betas = bl_fit["df_weights"]
    if Xnew.shape[1] != betas.shape[1] - 1:
        raise ValueError(
            f"Expected design matrix to have [{betas.shape[1] - 1}] columns, but it has [{Xnew.shape[1]}]."
        )

    if lam is not None:
        if not hasattr(lam, "__iter__"):
            lam = [lam]
        betas = betas[betas.index.isin(lam)]

    responses = (
        betas.iloc[:, 0].to_numpy() + Xnew @ betas.iloc[:, 1:].to_numpy().transpose()
    ).squeeze()

    if output in ("prob", "class"):
        responses = logit_to_prob(responses)

    if output == "class":
        responses = np.round(responses)

    return responses


def select_model(bl_fit, Xtrue, ytrue, lam=None, stat="deviance"):
    def _compute_stat(ytrue, predmat, statfn):
        return np.apply_along_axis(lambda col: statfn(ytrue, col), 0, predmat)

    valid_stats = ("deviance", "logloss", "auc", "class")

    if stat not in valid_stats:
        raise ValueError(f"stat [{stat}] not among supported stats: [{valid_stats}]")

    if lam is None:
        lam = bl_fit["df_nlambda"].lambda_.to_numpy()

    if stat in ("deviance", "logloss"):
        predictions = predict(bl_fit, Xtrue, lam=lam, output="prob")
        computed_stat = _compute_stat(ytrue, predictions, sklearn.metrics.log_loss)
    elif stat == "auc":
        predictions = predict(bl_fit, Xtrue, lam=lam, output="logit")
        computed_stat = _compute_stat(ytrue, predictions, sklearn.metrics.roc_auc_score)
    elif stat == "class":
        predictions = predict(bl_fit, Xtrue, lam=lam, output="class")
        computed_stat = _compute_stat(
            ytrue, predictions, sklearn.metrics.accuracy_score
        )

    return pd.DataFrame({"lam": lam, stat: computed_stat})
