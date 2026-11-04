"""Rust is the only solver; removed backend requests fail explicitly."""

import os
import subprocess
import sys

import numpy as np
import pytest
from support.paths import assert_bitwise

import pybiglasso as bl
from pybiglasso.config import Settings


def test_rust_is_the_only_backend():
    assert not hasattr(Settings.from_env({}), "backend")
    assert Settings.from_env({"PYBIGLASSO_BACKEND": "rust"}) == Settings.from_env({})
    with pytest.raises(ValueError, match="removed.*0.4.0.*PYBIGLASSO_BACKEND"):
        Settings.from_env({"PYBIGLASSO_BACKEND": "numba"})
    with pytest.raises(ValueError, match="backend must"):
        Settings.from_env({"PYBIGLASSO_BACKEND": "unknown"})


@pytest.mark.parametrize("family", ["binomial", "gaussian"])
def test_removed_backend_names_release_and_replacement(family):
    d = bl.common.generate_test_data(60, 4, seed=1)
    with pytest.raises(ValueError, match='removed.*0.4.0.*backend="rust"'):
        bl.biglasso(d["X"], d["y"], family=family, backend="numba")


@pytest.mark.parametrize("family", ["binomial", "gaussian"])
def test_explicit_rust_matches_implicit_solver(family):
    d = bl.common.generate_test_data(60, 4, seed=1)
    implicit = bl.biglasso(d["X"], d["y"], nlambda=3, family=family)
    explicit = bl.biglasso(d["X"], d["y"], nlambda=3, family=family, backend="rust")
    assert_bitwise(implicit, explicit)


@pytest.mark.parametrize(
    "name",
    [
        "BiglassoClassifier",
        "BiglassoClassifierCV",
        "BiglassoRegressor",
        "BiglassoRegressorCV",
    ],
)
def test_estimators_report_removed_backend(name):
    pytest.importorskip("sklearn")
    d = bl.common.generate_test_data(60, 4, seed=1)
    options = dict(n_alphas=3, cv=2) if name.endswith("CV") else {}
    estimator = getattr(bl, name)(backend="numba", **options)
    with pytest.raises(ValueError, match="removed.*0.4.0"):
        estimator.fit(d["X"], d["y"])


@pytest.mark.parametrize("backend", ["numba", "unknown"])
def test_removed_or_unknown_environment_backend_fails_at_import(backend):
    child = subprocess.run(
        [sys.executable, "-c", "import pybiglasso"],
        env=os.environ | {"PYBIGLASSO_BACKEND": backend},
        capture_output=True,
        text=True,
    )
    assert child.returncode != 0
    assert "PYBIGLASSO_BACKEND" in child.stderr
    if backend == "numba":
        assert "removed" in child.stderr and "0.4.0" in child.stderr
    else:
        assert "backend must" in child.stderr


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
        env=os.environ | {"PYBIGLASSO_BACKEND": "rust"},
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
        backend="rust",
        family="binomial",
    )
    # Both the full response and selected rows are balanced: log(1) = 0.
    assert fit.intercept[0] == 0
