"""The benchmark compares the same squared-loss objective, not rescaled y."""

import importlib
import math
from pathlib import Path

import numpy as np
import pytest
from support import rcompare


@pytest.fixture
def benchmark(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    return importlib.import_module("path")


@pytest.fixture
def orthogonal():
    X = np.asfortranarray([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    return X, 3 + 2 * X[:, 0]


def test_gaussian_benchmark_analytic_objective_and_kkt(benchmark, orthogonal):
    X, y = orthogonal
    beta = np.array([1.4, 0.0])
    assert benchmark.objective(X, y, 0.5, 0.5, beta, 3.0, "gaussian") == pytest.approx(
        31 / 40
    )
    assert (
        benchmark.gaussian_kkt(X, y, np.array([0.5]), 0.5, beta[None], np.array([3.0]))
        < 8 * np.finfo(float).eps
    )
    assert (
        benchmark.gaussian_kkt(
            X, y, np.array([0.5]), 0.5, np.zeros((1, 2)), np.array([3.0])
        )
        > 1
    )


def test_gaussian_benchmark_grid_is_analytic(benchmark, orthogonal):
    X, y = orthogonal
    grid = benchmark.lambda_grid(X, y, 0.5, family="gaussian")
    assert len(grid) == 100
    np.testing.assert_allclose(
        grid[[0, -1]], [4.0, 0.004], rtol=8 * np.finfo(float).eps
    )
    assert np.all(np.diff(grid) < 0)


def test_benchmark_default_objective_remains_binomial(benchmark, orthogonal):
    X, _ = orthogonal
    y = np.array([0.0, 0.0, 1.0, 1.0])
    assert benchmark.objective(X, y, 0.5, 0.5, np.zeros(2), 0.0) == math.log(2)


@pytest.mark.r
@pytest.mark.skipif(not rcompare.r_available(), reason="R not available")
@pytest.mark.parametrize("solver", ["pybiglasso", "r-biglasso-ssr", "r-glmnet"])
def test_gaussian_benchmark_adapters_have_same_analytic_optimum(
    benchmark, orthogonal, solver
):
    X, y = orthogonal
    out = benchmark.FITTERS[solver](
        X, X, y, np.array([1.0, 0.5]), 0.5, 1, family="gaussian"
    )
    np.testing.assert_allclose(
        out[0], [[1.0, 0.0], [1.4, 0.0]], atol=16 * np.finfo(float).eps, rtol=0
    )
    np.testing.assert_allclose(
        out[1], [3.0, 3.0], atol=16 * np.finfo(float).eps, rtol=0
    )
