"""The Python bounds setting reaches Rust's checked/unchecked dispatch.

Native negative tests in kernel_tests.rs check that invalid checked reads
panic, including block and gathered reads. These check the binding's flag
for each layout/row path; malformed user inputs are validated in either mode.
"""

import inspect
from contextlib import nullcontext
from dataclasses import replace

import numpy as np
import pytest

import pybiglasso as bl
from pybiglasso import binomial

_rust = pytest.importorskip("pybiglasso._rust")


@pytest.mark.parametrize("checked", [False, True])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("selection", ["all", "mask", "gather"])
def test_bounds_setting_is_forwarded_to_rust(monkeypatch, checked, order, selection):
    X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]] * 4, dtype=float, order=order)
    rows = {
        "all": None,
        "mask": np.arange(8),
        "gather": np.array([3, 0, 1, 3, 0, 1]),
    }[selection]
    monkeypatch.setattr(
        binomial, "settings", replace(binomial.settings, boundscheck=checked)
    )
    original = _rust.fit_path
    signature = inspect.signature(original)
    calls = []

    def spy(*args, **kwargs):
        checked_arg = signature.bind(*args, **kwargs).arguments["boundscheck"]
        result = original(*args, **kwargs)
        calls.append((checked_arg, result["row_path"]))
        return result

    monkeypatch.setattr(_rust, "fit_path", spy)
    # Suppress only the C-order advisory; the bounds dispatch is the subject.
    with pytest.warns(UserWarning, match="Fortran") if order == "C" else nullcontext():
        bl.biglasso(
            X,
            3 + X[:, 0],
            family="gaussian",
            backend="rust",
            lambdas=[0.25],
            row_indices=rows,
            n_threads=1,
        )
    assert calls == [(checked, selection)]
