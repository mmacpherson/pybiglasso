"""The boundscheck setting changes no result beyond rounding.

The Rust backend's sums are in a fixed order, so its results are bit-identical
either way. numba's fastmath kernels are summed in whatever order LLVM
vectorizes them, which a bounds check changes; their results agree as closely
as pybiglasso must agree with R, and the path itself, its lambdas, iteration
counts and rejections, is identical.

numba reads the setting once, at import, so each setting is fit in its own
process.
"""

import importlib.util
import os
import subprocess
import sys

import numpy as np
import pytest
from rcompare import ATOL, RTOL

from pybiglasso.config import Settings

BACKENDS = ["numba"]
if importlib.util.find_spec("pybiglasso._rust") is not None:
    BACKENDS.append("rust")

# each case: Fortran float64, C-order float32 (strided reads), and a read-only
# int8 memmap fit on a shuffled row subset with repeats (gathered reads)
FIT = """
import sys
import numpy as np
import pybiglasso as bl
from pybiglasso import binomial

out, backends = sys.argv[1], sys.argv[2].split(",")
if bl.config.settings.boundscheck:
    # the kernels really check, and weren't loaded from an unchecked cache
    try:
        binomial.dot(np.ones(3), np.ones(2))
        raise SystemExit("numba kernels are not bounds-checked")
    except IndexError:
        pass
d = bl.common.make_biglasso_dataset(nrow=400, ncol=150, seed=5, marker_effect_sd=0.3)
X, y = d["X"], d["y"].astype(np.float64)
memmap = np.lib.format.open_memmap(
    out + ".X.npy", mode="w+", dtype=np.int8, shape=X.shape, fortran_order=True
)
memmap[:] = X
memmap.flush()
del memmap
rows = np.random.default_rng(1).integers(0, X.shape[0], 300)
cases = {
    "fortran-f64": (np.asfortranarray(X, dtype=np.float64), None, 1.0),
    "c-f32": (np.ascontiguousarray(X, dtype=np.float32), None, 0.5),
    "memmap-rows": (np.load(out + ".X.npy", mmap_mode="r"), rows, 0.5),
}
results = {}
for backend in backends:
    for name, (Xc, row_indices, alpha) in cases.items():
        p = bl.biglasso(
            Xc, y, alpha=alpha, nlambda=30, row_indices=row_indices, backend=backend
        )
        key = f"{backend}/{name}"
        results[f"{key}/lambdas"] = p.lambdas
        results[f"{key}/intercept"] = p.intercept
        results[f"{key}/coef"] = p.coef.toarray()
        results[f"{key}/iterations"] = p.iterations
        results[f"{key}/n_rejected"] = p.n_rejected
np.savez(out, **results)
"""


def fit_with(boundscheck, tmp_path):
    out = tmp_path / f"boundscheck-{boundscheck}.npz"
    env = dict(os.environ, PYBIGLASSO_BOUNDSCHECK=boundscheck)
    env.pop("NUMBA_BOUNDSCHECK", None)
    child = subprocess.run(
        [sys.executable, "-c", FIT, str(out), ",".join(BACKENDS)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert child.returncode == 0, child.stderr
    return dict(np.load(out))


def test_results_identical_with_and_without_checks(tmp_path):
    checked, unchecked = fit_with("1", tmp_path), fit_with("0", tmp_path)
    assert checked.keys() == unchecked.keys()
    assert {k.split("/")[0] for k in checked} == set(BACKENDS)
    for key in checked:
        backend, _case, field = key.split("/")
        if backend == "numba" and field in ("intercept", "coef"):
            np.testing.assert_allclose(
                checked[key], unchecked[key], rtol=RTOL, atol=ATOL, err_msg=key
            )
        else:
            # bits, not values: also catches 0.0 against -0.0
            assert checked[key].tobytes() == unchecked[key].tobytes(), key


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", True), ("on", True), ("True", True), ("0", False), (" off ", False)],
)
def test_setting_from_env(value, expected):
    assert Settings.from_env({"PYBIGLASSO_BOUNDSCHECK": value}).boundscheck is expected


def test_setting_default_and_bad_value():
    assert Settings.from_env({}).boundscheck is True
    with pytest.raises(ValueError, match="PYBIGLASSO_BOUNDSCHECK"):
        Settings.from_env({"PYBIGLASSO_BOUNDSCHECK": "maybe"})
