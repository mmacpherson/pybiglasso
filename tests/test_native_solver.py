"""The native solver does not load optional numerical compilers."""

import os
import subprocess
import sys

import numpy as np
import pytest

import pybiglasso as bl


def test_import_and_fits_never_import_numba():
    # A fresh interpreter avoids false confidence from numba already imported
    # by a test dependency. A finder makes attempted optional imports fatal.
    code = """
import importlib.abc
import sys

class NoNumba(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "numba" or fullname.startswith("numba."):
            raise AssertionError("pybiglasso attempted to import numba")

sys.meta_path.insert(0, NoNumba())
import pybiglasso as bl

d = bl.common.generate_test_data(60, 4, seed=1)
for family in ("binomial", "gaussian"):
    fit = bl.biglasso(d["X"], d["y"], nlambda=3, family=family)
    assert len(fit) == 3
assert "numba" not in sys.modules
"""
    child = subprocess.run(
        [sys.executable, "-c", code],
        env=os.environ,
        capture_output=True,
        text=True,
    )
    assert child.returncode == 0, child.stderr


@pytest.mark.parametrize("subset", [False, True])
def test_rust_uses_no_numpy_transcendentals(monkeypatch, subset):
    """The null loss is part of the stopping rule, so it too must be portable."""
    rng = np.random.default_rng(1)
    X = np.asfortranarray(rng.integers(-3, 4, size=(60, 4)) / 4)
    y = (np.arange(60) % 2).astype(float)
    rows = np.array([5, 2, 5, 0]) if subset else None

    def forbidden(*args, **kwargs):
        raise AssertionError("Rust fitting called a NumPy transcendental")

    for name in ("exp", "log", "log1p"):
        monkeypatch.setattr(np, name, forbidden)
    fit = bl.biglasso(
        X,
        y,
        row_indices=rows,
        nlambda=3,
        lambda_min=0.5,
        family="binomial",
    )
    # Both the full response and selected rows are balanced: log(1) = 0.
    assert fit.intercept[0] == 0
