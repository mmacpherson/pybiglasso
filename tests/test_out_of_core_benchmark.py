"""The out-of-core benchmark keeps its family and data provenance straight."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def benchmark(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    import out_of_core

    return out_of_core


@pytest.fixture
def gaussian_files(tmp_path):
    # Orthogonal, centered, unit-scale columns give beta_j = soft(x_j'y/n, lambda).
    X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype="<f8")
    y = 3 + 2 * X[:, 0] + 0.5 * X[:, 1]
    X.ravel(order="F").tofile(tmp_path / "X.bin")
    y.tofile(tmp_path / "y.bin")
    (tmp_path / "data.json").write_text(
        json.dumps(dict(nrow=4, ncol=2, family="gaussian"))
    )
    return tmp_path


def test_reuse_rejects_a_different_family(benchmark, gaussian_files):
    with pytest.raises(ValueError, match="family does not match"):
        benchmark.validate_data(gaussian_files, 4, 2, "binomial")


def test_reuse_requires_gaussian_metadata(benchmark, gaussian_files):
    (gaussian_files / "data.json").unlink()
    with pytest.raises(ValueError, match="generate Gaussian data in a new directory"):
        benchmark.validate_data(gaussian_files, 4, 2, "gaussian")


@pytest.mark.parametrize("file", ["X.bin", "y.bin"])
def test_reuse_rejects_truncation(benchmark, gaussian_files, file):
    with (gaussian_files / file).open("r+b") as stream:
        stream.truncate(8)
    with pytest.raises(ValueError, match="must contain exactly"):
        benchmark.validate_data(gaussian_files, 4, 2, "gaussian")


@pytest.mark.skipif(
    sys.platform != "linux", reason="fit worker measures Linux cgroup accounting"
)
def test_gaussian_worker_matches_closed_form(benchmark, gaussian_files):
    benchmark.write_params(gaussian_files, 4, 2, 2, 0.1, family="gaussian")
    benchmark.fit_py(gaussian_files, 1)
    with np.load(gaussian_files / "py-1.npz") as result:
        np.testing.assert_allclose(result["lambdas"], [2, 0.2], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            result["coef"], [[0, 1.8], [0, 0.3]], rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(result["intercept"], [3, 3], rtol=1e-12, atol=1e-12)


def test_quiet_gate_resets_after_a_busy_interval(benchmark, monkeypatch):
    samples = iter([0.01] * 35 + [0.11] + [0.01] * 36)
    monkeypatch.setattr(benchmark, "cpu_use", lambda seconds: next(samples))
    gate = benchmark.wait_for_quiet()
    assert len(gate["samples"]) == 72
    assert all(value < 0.1 for value in gate["samples"][-36:])


def test_preparation_does_not_wait_for_quiet(benchmark, gaussian_files, monkeypatch):
    def unexpected():
        pytest.fail("preparation must not wait for a quiet machine")

    monkeypatch.setattr(benchmark, "wait_for_quiet", unexpected)
    benchmark.run(
        SimpleNamespace(
            workdir=gaussian_files,
            nrow=4,
            ncol=2,
            family="gaussian",
            nlambda=2,
            lambda_min=0.1,
            diagnostic=False,
            mem="3G",
            skip_r=True,
            prepare_only=True,
        )
    )


def test_path_identity_includes_dtype_and_shape(benchmark):
    original = benchmark.path_digest({"coef": np.zeros((1, 2), dtype="<f8")})
    assert original != benchmark.path_digest({"coef": np.zeros((2, 1), dtype="<f8")})
    assert original != benchmark.path_digest({"coef": np.zeros((1, 4), dtype="<i4")})


@pytest.mark.skipif(
    sys.platform != "linux", reason="benchmark provenance reads Linux CPU data"
)
def test_build_info_must_match_the_installed_binary(benchmark, tmp_path):
    metadata = tmp_path / "build-info.json"
    metadata.write_text(json.dumps(dict(native_sha256="0" * 64)))
    with pytest.raises(ValueError, match="installed native extension"):
        benchmark.environment(metadata)
