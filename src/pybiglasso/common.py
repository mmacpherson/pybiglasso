#!/usr/bin/env python3

import numpy as np
from numpy.random import default_rng

DEFAULT_SEED = 123 * 321
DEFAULT_DTYPE = np.float64


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
        # generate dosages array
        allele_frequencies = rng.beta(
            allele_frequency_alpha, allele_frequency_beta, size=len(weights)
        )
        dosages = rng.binomial(2, allele_frequencies, size=(nrow, len(weights)))

        # generate response; effects act on dosage deviations from the mean
        probs = logit_to_prob((dosages - 2 * allele_frequencies) @ weights)
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
