"""The bool storage benchmark's identity check: byte-exact, against the first fit."""

import importlib
import json
import types
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp


@pytest.fixture
def bench(monkeypatch):
    pytest.importorskip("resource", reason="the harness is POSIX-only")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    return importlib.import_module("bool_storage")


def fake_path(intercept=0.25, dtype=np.float64):
    fields = {f: np.array([1.0, 0.5], dtype=dtype) for f in ("lambdas", "deviance")}
    fields.update(
        intercept=np.array([intercept, 0.0], dtype=dtype),
        iterations=np.array([1, 3]),
        n_rejected=np.array([2, 0]),
        center=np.array([0.5, 0.25]),
        scale=np.array([0.5, 0.4]),
        coef=sp.csc_matrix(np.array([[0.0, 1.5], [0.0, -0.75]])),
    )
    return types.SimpleNamespace(**fields)


def digest(bench, tmp_path, name, path):
    np.savez(tmp_path / name, **bench.path_arrays(path))
    return bench.path_digest(tmp_path / name)


def test_digest_is_byte_exact(bench, tmp_path):
    first = digest(bench, tmp_path, "a.npz", fake_path())
    assert digest(bench, tmp_path, "b.npz", fake_path()) == first
    # array_equal would call each of these the same path
    assert digest(bench, tmp_path, "c.npz", fake_path(intercept=-0.0)) != digest(
        bench, tmp_path, "d.npz", fake_path(intercept=0.0)
    )
    assert digest(bench, tmp_path, "e.npz", fake_path(dtype=np.float32)) != first


def test_every_fit_is_compared_with_the_first(bench, tmp_path, monkeypatch):
    """Stub fits that each return a different path: the second must fail,
    whatever the first wrote over."""
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.setattr(bench, "HERE", tmp_path)
    monkeypatch.setattr(bench, "provenance", lambda workdir: {})
    monkeypatch.setattr(bench, "wait_until_quiet", lambda *args: 0.0)
    calls = []

    def fit(cmd, check):
        stem = Path(cmd[cmd.index("--out") + 1])
        calls.append(stem)
        np.savez(
            stem.with_suffix(".npz"),
            **bench.path_arrays(fake_path(intercept=0.25 * len(calls))),
        )
        stem.with_suffix(".json").write_text(json.dumps({"fit_seconds": 1.0}))

    monkeypatch.setattr(bench.subprocess, "run", fit)
    args = types.SimpleNamespace(
        workdir=str(workdir), tag="test", mem=[], batch_rows=[0], threads=[1],
        repeats=2, nlambda=2, lambda_min=0.1, gate_busy=0.1, gate_seconds=0,
    )  # fmt: skip
    with pytest.raises(SystemExit, match="differs from the first fit"):
        bench.run(args)
    assert len(set(calls)) == len(calls) == 2  # every fit keeps its own output
    records = [
        json.loads(line)
        for line in (tmp_path / "results/bool-storage-test/results.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [r["identical_to_first"] for r in records] == [True, False]
