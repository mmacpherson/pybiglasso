"""Whole-path timing must reject unconverged or silently shortened fits."""

import hashlib
import importlib
import itertools
import json
import statistics
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize(
    "bundle, file_count", [("int8-m1-pro-quiet", 13), ("int8-zen4-quiet", 8)]
)
def test_quiet_evidence_matches_manifest(bundle, file_count):
    # These are external measurements, not regenerated solver expectations.
    # Preserve executed sources too: formatting them would break provenance.
    directory = Path(__file__).resolve().parents[1] / "benchmarks/results" / bundle
    manifest = json.loads((directory / "manifest.json").read_text())
    assert len(manifest) == file_count
    for name, expected in manifest.items():
        data = (directory / name).read_bytes()
        assert len(data) == expected["bytes"], name
        assert hashlib.sha256(data).hexdigest() == expected["sha256"], name


def test_quiet_zen4_evidence_protocol():
    # Audit relations in measured records, not expected solver values or speed.
    directory = (
        Path(__file__).resolve().parents[1] / "benchmarks/results/int8-zen4-quiet"
    )

    def records(name):
        return [
            json.loads(line) for line in (directory / name).read_text().splitlines()
        ]

    rows = records("results.jsonl")
    gates = records("gate-passed.jsonl")
    samples = records("gate-qualified.jsonl")
    environment = records("environment.jsonl")
    assert len(rows) == len(gates) == 20
    assert len(samples) == 740
    assert len(environment) == 40
    expected = {
        (family, dtype, nrow, ncol, alpha, threads)
        for family, dtype, problems, alphas in [
            ("binomial", "int8", [(2000, 20000), (5000, 50000)], [1.0, 0.5]),
            ("binomial", "float64", [(5000, 1000), (2000, 20000)], [1.0, 0.5]),
            ("gaussian", "float64", [(5000, 1000), (2000, 20000)], [0.5]),
        ]
        for nrow, ncol in problems
        for alpha in alphas
        for threads in (1, 6)
    }
    keys = ("family", "dtype", "nrow", "ncol", "alpha", "threads")
    assert {tuple(row[key] for key in keys) for row in rows} == expected
    variants = ["baseline", "ungated", "gated"]
    orders = list(map(list, itertools.permutations(variants)))
    identities = {}
    previous_end = None
    for index, (row, gate) in enumerate(zip(rows, gates, strict=True)):
        label = row["label"]
        assert row["orders"] == orders and row["seed"] == 1
        assert gate["label"] == label and gate["qualifying_seconds"] >= 180
        interval = [sample for sample in samples if sample["label"] == label]
        assert len(interval) == 37 and interval[-1] == gate
        duration = (
            datetime.fromisoformat(interval[-1]["utc"])
            - datetime.fromisoformat(interval[0]["utc"])
        ).total_seconds()
        assert duration >= 180
        assert all(
            sample["qualifies"]
            and sample["user_system_pct"] <= 10
            and not sample["workers"]
            for sample in interval
        )
        before, after = environment[index * 2 : index * 2 + 2]
        assert before["label"] == label + "-before"
        assert after["label"] == label + "-after"
        assert not before["workers"] and not after["workers"]
        assert len(row["fits"]) == 18 and len(row["warmups"]) == 3
        assert [fit["variant"] for fit in row["warmups"]] == variants
        for repeat, order in enumerate(orders):
            group = row["fits"][repeat * 3 : repeat * 3 + 3]
            assert [fit["variant"] for fit in group] == order
            assert [fit["position"] for fit in group] == [0, 1, 2]
            assert all(fit["repeat"] == repeat for fit in group)
        fits = row["warmups"] + row["fits"]
        assert {fit["path_hash"] for fit in fits} == {row["path_hash"]}
        assert {fit["nlambda"] for fit in fits} == {100}
        assert len({fit["iterations"] for fit in fits}) == 1
        key = tuple(row[key] for key in keys[:-1])
        assert identities.setdefault(key, row["path_hash"]) == row["path_hash"]
        chronology = [gate["utc"], before["utc"]]
        for fit in fits:
            assert fit["seconds"] > 0
            chronology.extend([fit["utc_start"], fit["utc_end"]])
        chronology.append(after["utc"])
        timestamps = list(map(datetime.fromisoformat, chronology))
        assert timestamps == sorted(timestamps)
        if previous_end is not None:
            assert previous_end < timestamps[0]
        previous_end = timestamps[-1]
        for name in variants:
            times = [fit["seconds"] for fit in row["fits"] if fit["variant"] == name]
            assert len(times) == 6 and times == row["times"][name]
            assert row["median_seconds"][name] == statistics.median(times)
            assert row["minimum_seconds"][name] == min(times)
        for name in ("ungated", "gated"):
            assert row["baseline_over_" + name] == (
                row["median_seconds"]["baseline"] / row["median_seconds"][name]
            )
    assert all(
        record["ac_online"] and record["governors"] == ["performance"]
        for record in samples + environment
    )


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
    monkeypatch.setattr(benchmark, "cpu_model", lambda: "test CPU")
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
    assert record["cpu"] == "test CPU"
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


def test_cpu_metadata_linux_model(benchmark, monkeypatch):
    monkeypatch.setattr(benchmark.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        benchmark.Path,
        "read_text",
        lambda path: "processor : 0\nmodel name\t: Example CPU\n",
    )
    assert benchmark.cpu_model() == "Example CPU"


def test_cpu_metadata_macos_sysctl_without_proc(benchmark, monkeypatch):
    monkeypatch.setattr(benchmark.platform, "system", lambda: "Darwin")

    def no_proc(path):
        raise AssertionError("macOS must not read /proc/cpuinfo")

    def sysctl(command, **kwargs):
        assert command == ["sysctl", "-n", "machdep.cpu.brand_string"]
        assert kwargs == dict(text=True, stderr=subprocess.DEVNULL, timeout=5)
        return "Apple M1\n"

    monkeypatch.setattr(benchmark.Path, "read_text", no_proc)
    monkeypatch.setattr(benchmark.subprocess, "check_output", sysctl)
    assert benchmark.cpu_model() == "Apple M1"


@pytest.mark.parametrize(
    "reason",
    [
        FileNotFoundError(),
        subprocess.CalledProcessError(1, "sysctl"),
        subprocess.TimeoutExpired("sysctl", 5),
    ],
)
def test_cpu_metadata_macos_failure_falls_back(benchmark, monkeypatch, reason):
    monkeypatch.setattr(benchmark.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(benchmark.platform, "processor", lambda: "fallback CPU")

    def fail(*args, **kwargs):
        raise reason

    monkeypatch.setattr(benchmark.subprocess, "check_output", fail)
    assert benchmark.cpu_model() == "fallback CPU"


@pytest.mark.parametrize("content", [None, "CPU architecture: 8\n", "model name : \n"])
def test_cpu_metadata_linux_missing_model_falls_back(benchmark, monkeypatch, content):
    monkeypatch.setattr(benchmark.platform, "system", lambda: "Linux")
    monkeypatch.setattr(benchmark.platform, "processor", lambda: "fallback CPU")

    def read(path):
        if content is None:
            raise FileNotFoundError()
        return content

    monkeypatch.setattr(benchmark.Path, "read_text", read)
    assert benchmark.cpu_model() == "fallback CPU"


@pytest.mark.parametrize("machine, expected", [("arm64", "arm64"), ("", "unknown")])
def test_cpu_metadata_empty_processor_uses_architecture(
    benchmark, monkeypatch, machine, expected
):
    monkeypatch.setattr(benchmark.platform, "system", lambda: "Windows")
    monkeypatch.setattr(benchmark.platform, "processor", lambda: "")
    monkeypatch.setattr(benchmark.platform, "machine", lambda: machine)
    assert benchmark.cpu_model() == expected


def test_cpu_metadata_blank_sysctl_uses_fallback(benchmark, monkeypatch):
    monkeypatch.setattr(benchmark.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        benchmark.subprocess, "check_output", lambda *args, **kwargs: "\n"
    )
    monkeypatch.setattr(benchmark.platform, "processor", lambda: "fallback CPU")
    assert benchmark.cpu_model() == "fallback CPU"
