"""Wheel provenance must identify the actual selected compiler, not just the pin."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def provenance(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("wheel_provenance")


def commands(monkeypatch, provenance, release):
    outputs = {
        (
            "rustc",
            "-vV",
        ): f"rustc {release}\nrelease: {release}\nLLVM version: 23.1.1\n",
        ("maturin", "--version"): "maturin 1.16.0\n",
    }
    monkeypatch.setattr(
        provenance.subprocess,
        "check_output",
        lambda command, **kwargs: outputs[tuple(command)],
    )


def test_logs_llvm_and_maturin(provenance, monkeypatch, capsys):
    commands(monkeypatch, provenance, "1.99.1")
    provenance.report("1.99.1")
    output = capsys.readouterr().out
    assert "LLVM version: 23.1.1" in output
    assert "maturin 1.16.0" in output


def test_rejects_a_different_active_compiler(provenance, monkeypatch):
    commands(monkeypatch, provenance, "1.98.0")
    with pytest.raises(SystemExit, match="Expected Rust 1.99.1, selected 1.98.0"):
        provenance.report("1.99.1")
