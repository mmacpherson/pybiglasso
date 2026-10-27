"""The benchmark compares the same squared-loss objective, not rescaled y."""

import importlib
import json
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


@pytest.mark.parametrize("container", [None, "docker", "podman"])
def test_benchmark_r_thread_limits_cross_container_boundary(
    benchmark, monkeypatch, tmp_path, container
):
    # Exercise POSIX container command construction even on Windows CI.
    monkeypatch.setattr(rcompare.os, "getuid", lambda: 1000, raising=False)
    monkeypatch.setattr(rcompare.os, "getgid", lambda: 1000, raising=False)
    if container:
        monkeypatch.setenv("CONTAINER", container)
    else:
        monkeypatch.delenv("CONTAINER", raising=False)
    calls = []
    monkeypatch.setattr(
        rcompare.subprocess, "run", lambda cmd, **kwargs: calls.append((cmd, kwargs))
    )
    limits = benchmark.r_thread_limits(6)
    rcompare.run_r([tmp_path], load=lambda _: {}, env=limits)
    cmd, kwargs = calls[0]
    assert kwargs["env"]["OPENBLAS_NUM_THREADS"] == "1"
    assert kwargs["env"]["MKL_NUM_THREADS"] == "1"
    assert kwargs["env"]["OMP_NUM_THREADS"] == "6"
    if container:
        for key, value in limits.items():
            index = cmd.index(f"{key}={value}")
            assert cmd[index - 1] == "--env"


@pytest.mark.parametrize("family", ["binomial", "gaussian"])
@pytest.mark.parametrize("solver", ["r-biglasso-ssr", "r-glmnet"])
def test_all_r_benchmark_adapters_forward_thread_limits(
    benchmark, monkeypatch, orthogonal, family, solver
):
    calls = []

    def run_r(dirs, **kwargs):
        calls.append(kwargs["env"])
        return [
            dict(
                beta=np.zeros((3, 2)) if solver == "r-biglasso-ssr" else [0.0] * 4,
                beta_dim=[2, 2],
                scale=np.ones(2),
                center=np.zeros(2),
                intercept=[0.0, 0.0],
                seconds=1.0,
                biglasso_version="test",
                glmnet_version="test",
                r_version="test",
                warnings=[],
                iter=np.ones(2),
                glmnet_family=family,
                glmnet_alpha=0.5,
                glmnet_lambda_factor=1.0,
                response_scale=1.0,
            )
        ]

    monkeypatch.setattr(rcompare, "run_r", run_r)
    X, y = orthogonal
    benchmark.FITTERS[solver](X, X, y, np.array([1.0, 0.5]), 0.5, 6, family=family)
    assert calls == [benchmark.r_thread_limits(6)]


def test_gaussian_table_requires_native_glmnet(benchmark, tmp_path):
    table = importlib.import_module("gaussian-table")
    path = tmp_path / "results.jsonl"
    path.write_text(
        json.dumps(
            dict(
                family="gaussian",
                solver="r-glmnet",
                glmnet_family="gaussian()",
            )
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="native Gaussian"):
        table.render(path)


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


@pytest.mark.r
@pytest.mark.skipif(not rcompare.r_available(), reason="R not available")
@pytest.mark.parametrize("alpha", [1.0, 0.5, 0.2])
@pytest.mark.parametrize("response_scale", [0.25, 3.0])
def test_native_glmnet_mapping_matches_generic_optimality(
    benchmark, alpha, response_scale
):
    # Independent solvers of the same objective; no port-generated expectations.
    # Exercise both sides of unit response variance and correlated columns.
    rng = np.random.default_rng(42)
    X = rng.normal(size=(300, 60))
    X[:, 1:] += X[:, :1] / 2
    X = np.asfortranarray(X)
    Xs = (X - X.mean(axis=0)) / X.std(axis=0)
    y = 3 + response_scale * (Xs[:, 0] + Xs[:, 1] / 2 + rng.normal(size=300))
    lambdas = np.array([0.5, 0.1, 0.01]) * response_scale
    outputs = [
        benchmark.fit_r_glmnet(
            X,
            Xs,
            y,
            lambdas,
            alpha,
            1,
            family="gaussian",
            generic=generic,
            thresh=1e-13,
        )
        for generic in (False, True)
    ]
    from support.certificates import gaussian_curvature, with_ones

    Z = with_ones(Xs)
    for out in outputs:
        assert not out[3]["warnings"] and out[3]["jerr"] == 0
    scale = np.std(y)
    factor = alpha + (1 - alpha) * scale
    assert outputs[0][3]["glmnet_alpha"] == pytest.approx(alpha / factor)
    assert outputs[0][3]["glmnet_lambda_factor"] == pytest.approx(factor)
    for k, lam in enumerate(lambdas):
        a, b = [np.r_[out[1][k], out[0][k]] for out in outputs]
        residuals = [
            benchmark.gaussian_kkt(Xs, y, np.array([lam]), alpha, t[None, 1:], t[:1])
            for t in (a, b)
        ]
        # Fixed accuracy target, not a tolerance tuned to observed differences.
        assert max(residuals) <= 1e-5 * response_scale
        curvature = gaussian_curvature(Z, lam, alpha, np.ones(X.shape[1]))
        distance = np.sqrt(Z.shape[1]) * sum(residuals) / curvature
        rounding = 64 * len(y) * np.finfo(float).eps * (1 + np.linalg.norm(y))
        assert np.linalg.norm(a - b) <= distance + rounding / curvature
