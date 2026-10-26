"""The backend setting: Rust by default since 0.2.0; numba deprecated, an
optional extra, and gone in 0.3.0."""

import warnings

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
    with pytest.warns(FutureWarning, match="0.3.0") as caught:
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
