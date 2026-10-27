"""Whole-path timing must reject unconverged or silently shortened fits."""

import importlib
import warnings
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def benchmark(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    return importlib.import_module("int8")


@pytest.mark.parametrize(
    "message, category",
    [
        ("Maximum number of iterations reached", UserWarning),
        ("Model saturated", UserWarning),
        ("overflow encountered", RuntimeWarning),
        ("An unrelated future change", FutureWarning),
    ],
)
def test_benchmark_rejects_fit_warnings(benchmark, monkeypatch, message, category):
    def fit(*args, **kwargs):
        warnings.warn(message, category, stacklevel=2)
        return range(100)

    monkeypatch.setattr(benchmark.bl, "biglasso", fit)
    with pytest.raises(category, match=message):
        benchmark.checked_fit(None, None)


def test_benchmark_rejects_silent_truncation(benchmark, monkeypatch):
    monkeypatch.setattr(benchmark.bl, "biglasso", lambda *args, **kwargs: range(99))
    with pytest.raises(ValueError, match="99 of 100 lambdas"):
        benchmark.checked_fit(None, None)


def test_benchmark_accepts_only_numba_deprecation(benchmark, monkeypatch):
    def fit(*args, **kwargs):
        warnings.warn(
            "The numba backend is deprecated and slated for removal in pybiglasso 0.4.0",
            FutureWarning,
            stacklevel=2,
        )
        return range(kwargs["nlambda"])

    monkeypatch.setattr(benchmark.bl, "biglasso", fit)
    path, seconds = benchmark.checked_fit(None, None, nlambda=7)
    assert len(path) == 7
    assert seconds >= 0


def test_benchmark_rejects_actual_unconverged_fit(benchmark):
    # Correlated predictors require several coordinate sweeps at this lambda.
    X = np.asfortranarray([[-1.0, -2.0], [-1.0, 0.0], [1.0, 0.0], [1.0, 2.0]])
    y = 3 + 2 * X[:, 0] - X[:, 1]
    with pytest.raises(UserWarning, match="converg|iteration"):
        benchmark.checked_fit(
            X, y, family="gaussian", backend="rust", max_iter=1, n_threads=1
        )
