"""Cross-platform snapshots must cover the compact-storage specialization too.

These test the snapshot protocol, not model expectations. Synthetic payloads
below deliberately contain no solver-generated coefficients or golden hashes.
"""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

import pybiglasso as bl


def expected_cases():
    # Independent protocol specification: do not derive this from the script's
    # DTYPES, so deleting int8 from record AND compare still fails this check.
    return {
        f"{shape}/dtype={dtype}/family={family}/alpha={alpha}"
        for shape in ("tall", "wide", "parallel-irls")
        for dtype in ("float64", "int8", "arrow-mixed")
        for family in ("binomial", "gaussian")
        for alpha in (1.0, 0.5)
    }


@pytest.fixture
def arch_bits(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("check_arch_bits")


def test_record_fits_fortran_int8_and_float64_in_both_families(
    arch_bits, monkeypatch, tmp_path
):
    calls = []
    real_fit = bl.biglasso

    def observe(X, y, **options):
        from pybiglasso.arrow import _is_arrow

        kind = "arrow-mixed" if _is_arrow(X) else X.dtype.name
        if kind != "arrow-mixed":
            assert X.flags.f_contiguous
        assert y.dtype == np.float64
        calls.append((X.shape, kind, options["family"], options["alpha"]))
        return real_fit(X, y, **options)

    monkeypatch.setattr(bl, "biglasso", observe)
    output = tmp_path / "record.json"
    arch_bits.record(output)
    snapshot = json.loads(output.read_text())
    assert set(snapshot["cases"]) == expected_cases()
    assert len(calls) == 36
    assert set(calls) == {
        (shape, dtype, family, alpha)
        for shape in ((2057, 17), (97, 129), (16391, 9))
        for dtype in ("float64", "int8", "arrow-mixed")
        for family in ("binomial", "gaussian")
        for alpha in (1.0, 0.5)
    }
    for key, case in snapshot["cases"].items():
        expected_dtype = "|i1" if "/dtype=int8/" in key else "<f8"
        assert case["inputs"]["X"]["dtype"] == expected_dtype


@pytest.fixture
def snapshots(tmp_path):
    paths = []
    for system, machine in (
        ("Linux", "x86_64"),
        ("Linux", "aarch64"),
        ("Darwin", "arm64"),
        ("Windows", "amd64"),
    ):
        snapshot = {
            "system": system,
            "machine": machine,
            "cases": {
                key: {"inputs": {"X": "identical-input"}, "fit": {"coef": "bits"}}
                for key in expected_cases()
            },
        }
        path = tmp_path / f"{system}-{machine}.json"
        path.write_text(json.dumps(snapshot))
        paths.append(path)
    return paths


def test_compare_accepts_complete_equal_snapshots(arch_bits, snapshots):
    arch_bits.compare(snapshots[0], snapshots[1:])


@pytest.mark.parametrize("index", [0, 1])
def test_compare_rejects_missing_int8_cases(arch_bits, snapshots, index):
    snapshot = json.loads(snapshots[index].read_text())
    snapshot["cases"] = {
        key: value
        for key, value in snapshot["cases"].items()
        if "/dtype=int8/" not in key
    }
    snapshots[index].write_text(json.dumps(snapshot))
    with pytest.raises(AssertionError):
        arch_bits.compare(snapshots[0], snapshots[1:])


@pytest.mark.parametrize("part", ["inputs", "fit"])
def test_compare_rejects_changed_int8_bits(arch_bits, snapshots, part):
    snapshot = json.loads(snapshots[1].read_text())
    key = "tall/dtype=int8/family=binomial/alpha=1.0"
    field = "X" if part == "inputs" else "coef"
    snapshot["cases"][key][part][field] = "changed-bits"
    snapshots[1].write_text(json.dumps(snapshot))
    with pytest.raises(AssertionError, match="inputs differ|bits differ"):
        arch_bits.compare(snapshots[0], snapshots[1:])
