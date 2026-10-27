"""The backend setting: Rust by default since 0.2.0; numba deprecated, an
optional extra, and gone in 0.4.0."""

import warnings

import numpy as np
import pytest

import pybiglasso as bl
from pybiglasso import binomial
from pybiglasso.config import Settings


def test_rust_is_the_default():
    assert Settings.from_env({}).backend == "rust"
    assert Settings.from_env({"PYBIGLASSO_BACKEND": "numba"}).backend == "numba"


def test_numba_warns_once_a_process(monkeypatch):
    pytest.importorskip("numba")
    monkeypatch.setattr(binomial, "_numba_warned", False)
    d = bl.common.generate_test_data(60, 4, seed=1)
    with pytest.warns(FutureWarning, match="0.4.0") as caught:
        bl.biglasso(d["X"], d["y"], nlambda=3, backend="numba")
    assert caught[0].filename == __file__  # pointed at the caller
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        bl.biglasso(d["X"], d["y"], nlambda=3, backend="numba")


def test_numba_backend_without_numba_says_how_to_install(monkeypatch):
    monkeypatch.setattr(binomial, "nb", None)
    d = bl.common.generate_test_data(60, 4, seed=1)
    with pytest.raises(ImportError, match=r"pybiglasso\[numba\]"):
        bl.biglasso(d["X"], d["y"], nlambda=3, backend="numba")
    bl.biglasso(d["X"], d["y"], nlambda=3, backend="rust")


@pytest.mark.parametrize("subset", [False, True])
def test_rust_uses_no_numpy_transcendentals(monkeypatch, subset):
    """The null loss is part of the stopping rule, so it too must be portable."""
    rng = np.random.default_rng(1)
    X = np.asfortranarray(rng.integers(-3, 4, size=(60, 4)) / 4)
    y = (np.arange(60) % 2).astype(float)
    rows = np.array([5, 2, 5, 0]) if subset else None

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Rust fitting called a NumPy transcendental or numba kernel"
        )

    monkeypatch.setattr(binomial, "get_nrow", forbidden)
    for name in ("exp", "log", "log1p"):
        monkeypatch.setattr(np, name, forbidden)
    fit = bl.biglasso(X, y, row_indices=rows, nlambda=3, lambda_min=0.5, backend="rust")
    # Both the full response and selected rows are balanced: log(1) = 0.
    assert fit.intercept[0] == 0
