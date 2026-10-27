"""Whole-path timing must reject unconverged or silently shortened fits."""

import importlib
import json
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

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


def test_benchmark_records_each_balanced_fit_interval(benchmark, monkeypatch, tmp_path):
    # Test harness bookkeeping with stand-ins, not model coefficients or
    # performance expectations. The supplied clock fixes the interval oracle.
    baseline_file = tmp_path / "baseline"
    candidate_file = tmp_path / "candidate"
    baseline_file.write_bytes(b"baseline")
    candidate_file.write_bytes(b"candidate")
    baseline = SimpleNamespace(__file__=str(baseline_file))
    candidate = SimpleNamespace(__file__=str(candidate_file))
    loader = SimpleNamespace(exec_module=lambda module: None)
    monkeypatch.setattr(
        benchmark.importlib.util,
        "spec_from_file_location",
        lambda *args: SimpleNamespace(loader=loader),
    )
    monkeypatch.setattr(
        benchmark.importlib.util, "module_from_spec", lambda spec: baseline
    )
    monkeypatch.setattr(benchmark, "_rust", candidate)
    # main restores its own candidate; restore the real package module after
    # this isolated bookkeeping test, too.
    monkeypatch.setattr(benchmark.bl, "_rust", benchmark.bl._rust)
    monkeypatch.setattr(benchmark, "PROBLEMS", {"toy": (8, 1)})
    monkeypatch.setattr(
        benchmark.bl.common,
        "make_biglasso_dataset",
        lambda **kwargs: {"X": np.ones((8, 1)), "y": np.arange(8) % 2},
    )

    class PathStub:
        iterations = np.ones(100, dtype=int)

        def __len__(self):
            return 100

    calls = []

    def fit(*args, **kwargs):
        calls.append(benchmark.bl._rust)
        return PathStub(), 0.25

    monkeypatch.setattr(benchmark, "checked_fit", fit)
    monkeypatch.setattr(benchmark, "path_hash", lambda fit: "same-test-path")
    clock = iter(range(100, 900, 100))
    monkeypatch.setattr(benchmark.time, "time_ns", lambda: next(clock))
    out = tmp_path / "fits.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "int8.py",
            "--baseline-extension",
            str(baseline_file),
            "--problem",
            "toy",
            "--threads",
            "1",
            "--alpha",
            "1",
            "--repeats",
            "2",
            "--out",
            str(out),
        ],
    )
    benchmark.main()
    record = json.loads(out.read_text())
    assert calls == [baseline, candidate, baseline, candidate, candidate, baseline]
    assert record["times"] == {"baseline": [0.25, 0.25], "candidate": [0.25, 0.25]}
    assert record["fit_intervals"] == [
        dict(
            repeat=i // 2,
            variant=name,
            start_unix_ns=100 + 200 * i,
            end_unix_ns=200 + 200 * i,
        )
        for i, name in enumerate(["baseline", "candidate", "candidate", "baseline"])
    ]
