"""The boundscheck setting changes no result.

The Rust backend's sums are in a fixed order, so its results are bit-identical
either way.

The setting is read once, at import, so each setting is fit in its own process.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

from pybiglasso.config import Settings

# each case: Fortran float64, C-order float32 (strided reads), and a read-only
# int8 memmap fit on a shuffled row subset with repeats (gathered reads)
FIT = """
import sys
import numpy as np
import pybiglasso as bl

out = sys.argv[1]
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
for name, (Xc, row_indices, alpha) in cases.items():
    p = bl.biglasso(
        Xc, y, family="binomial", alpha=alpha, nlambda=30, row_indices=row_indices, backend="rust"
    )
    results[f"{name}/lambdas"] = p.lambdas
    results[f"{name}/intercept"] = p.intercept
    results[f"{name}/coef"] = p.coef.toarray()
    results[f"{name}/iterations"] = p.iterations
    results[f"{name}/n_rejected"] = p.n_rejected
np.savez(out, **results)
"""


def fit_with(boundscheck, tmp_path):
    out = tmp_path / f"boundscheck-{boundscheck}.npz"
    env = dict(os.environ, PYBIGLASSO_BOUNDSCHECK=boundscheck)
    env.pop("NUMBA_BOUNDSCHECK", None)
    child = subprocess.run(
        [sys.executable, "-c", FIT, str(out)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert child.returncode == 0, child.stderr
    return dict(np.load(out))


def test_results_identical_with_and_without_checks(tmp_path):
    checked, unchecked = fit_with("1", tmp_path), fit_with("0", tmp_path)
    assert checked.keys() == unchecked.keys()
    assert {k.split("/")[0] for k in checked} == {"fortran-f64", "c-f32", "memmap-rows"}
    for key in checked:
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
