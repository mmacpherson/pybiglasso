"""Compare default prediction with the raw predictor in the same process."""

from statistics import median
from time import perf_counter_ns

import numpy as np
import pytest

from pybiglasso import BiglassoPath, biglasso

pytestmark = pytest.mark.speed

# Budget up to twice the raw work for centered prediction: it may need another
# pass over the active values to subtract their centers before multiplication.
# Allow another factor of two for machine and timing variation.
MAX_RATIO = 4


@pytest.fixture(scope="module")
def design():
    # A tall, modest-width problem makes the prediction work more expensive
    # than its Python setup, with only 4.4 MiB of input and 9.2 MiB of output.
    rng = np.random.default_rng(811)
    X = np.asfortranarray(rng.standard_normal((12_000, 48)))
    signal = X[:, :24] @ np.linspace(0.8, 0.3, 24)
    X += np.linspace(-0.5, 0.5, X.shape[1])
    probability = 1 / (1 + np.exp(-0.18 * signal))
    return X, {
        "gaussian": 2 + signal + rng.standard_normal(len(X)),
        "binomial": (rng.random(len(X)) < probability).astype(float),
    }


def prediction_ratio(
    X: np.ndarray,
    default: BiglassoPath,
    reference: BiglassoPath,
    lambdas: list[float] | None,
) -> float:
    for path in (default, reference):
        path.decision_function(X, lambdas=lambdas)

    # Alternate which mode goes first. The median ignores occasional stalls.
    samples = [[], []]
    paths = (default, reference)
    for repeat in range(5):
        first = repeat % 2
        for mode in (first, 1 - first):
            start = perf_counter_ns()
            prediction = paths[mode].decision_function(X, lambdas=lambdas)
            elapsed = perf_counter_ns() - start
            del prediction
            samples[mode].append(elapsed)
    return median(samples[0]) / median(samples[1])


@pytest.mark.parametrize("family", ["gaussian", "binomial"])
def test_default_prediction_speed(design, family):
    X, responses = design
    # A smaller training set keeps the guard focused on prediction.
    train = np.asfortranarray(X[:4096])
    y = responses[family][: len(train)]
    default = biglasso(train, y, family=family, nlambda=100, n_threads=1)
    reference = biglasso(
        train, y, family=family, lambdas=default.lambdas, n_threads=1, r_compatible=True
    )

    whole = prediction_ratio(X, default, reference, None)
    selected = prediction_ratio(X, default, reference, [float(default.lambdas[50])])
    ratios = f"{family}: whole path {whole:.3f}x, one lambda {selected:.3f}x"
    print(ratios)
    assert max(whole, selected) <= MAX_RATIO, (
        f"{ratios} against R-compatible prediction (limit {MAX_RATIO}x)"
    )
