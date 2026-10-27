"""X is never copied by native code either.

tracemalloc (test_out_of_core.py) sees numpy's allocations but not the Rust
extension's, which go straight to malloc. Here each fit runs in a child
process whose private writable memory is capped (RLIMIT_DATA) at what it
holds after a warm-up fit on a small, different X, plus half the size of
the rows the fit selects; the target X's first fit runs under the cap. A
copy of X, or of just those rows, in any dtype, by numpy or by Rust, can't
be allocated: numpy raises MemoryError,
and Rust aborts the child. The memmap itself is a read-only file mapping,
which the limit doesn't count. Linux only: other systems don't enforce
RLIMIT_DATA this way.
"""

import json
import subprocess
import sys
import textwrap

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="RLIMIT_DATA is enforced on Linux"
)

# over 16,384 rows, where the solver turns row-parallel
NROW, NCOL = 20_000, 2_000  # 40 MB as int8; 320 MB as float64

CHILD = textwrap.dedent(
    """
    import ctypes, json, resource, sys
    import numpy as np
    import pybiglasso as bl

    xfile, yfile, case = sys.argv[1:]
    target_X, target_y = np.load(xfile, mmap_mode="r"), np.load(yfile)
    keep = []

    def run(X, y):
        nrow = X.shape[0]
        gaussian = y + np.arange(nrow) % 7  # any continuous response
        if case == "binomial":
            bl.biglasso(X, y, family="binomial", nlambda=8, lambda_min=0.2)
        elif case == "gaussian":
            bl.biglasso(X, gaussian, family="gaussian", nlambda=8, lambda_min=0.2)
        elif case == "most rows":  # read through a mask
            rows = np.arange(0, nrow, 5)
            rows = np.setdiff1d(np.arange(nrow), rows)
            bl.biglasso(X, y, family="binomial", nlambda=8, lambda_min=0.2, row_indices=rows)
        elif case == "few rows":  # read through the row indices
            rows = np.arange(0, nrow, 5)
            bl.biglasso(X, y, family="binomial", nlambda=8, lambda_min=0.2, row_indices=rows)
        elif case == "copy the rows":  # a control: the selected rows, copied
            rows = np.arange(0, nrow, 5)
            keep.append(np.array(X[rows], order="F"))
            bl.biglasso(X, y, family="binomial", nlambda=8, lambda_min=0.2, row_indices=rows)
        elif case == "cross-validation":
            from pybiglasso.sklearn import BiglassoClassifierCV
            BiglassoClassifierCV(n_alphas=6, alpha_min_ratio=0.2, cv=3, n_jobs=1).fit(X, y)
        # the negative controls, which must fail: a copy of X kept by numpy,
        # and an allocation of X's size, as Rust's would be, before a fit
        elif case == "copy X":
            keep.append(np.array(X))
            bl.biglasso(X, y, family="binomial", nlambda=8, lambda_min=0.2)
        elif case == "malloc X":
            libc = ctypes.CDLL(None)
            libc.malloc.restype = ctypes.c_void_p
            if not libc.malloc(ctypes.c_size_t(X.nbytes)):
                raise MemoryError("malloc")
            bl.biglasso(X, y, family="binomial", nlambda=8, lambda_min=0.2)

    def private_bytes():
        with open("/proc/self/status") as status:
            for line in status:
                if line.startswith("VmData:"):
                    return int(line.split()[1]) * 1024

    # Warm up thread pools, imports and the allocator's arenas on a small,
    # different X, so anything a first fit of the target X keeps is measured.
    rng = np.random.default_rng(0)
    small = np.asfortranarray(rng.integers(0, 3, size=(2_000, 200), dtype=np.int8))
    run(small, (rng.random(2_000) < 0.5).astype(np.float64))
    del small, keep[:]
    # half of the rows the fit selects (a fold's training rows, for CV), so a
    # copy of just those rows can't be allocated either
    selected = {"most rows": 4 / 5, "few rows": 1 / 5, "copy the rows": 1 / 5, "cross-validation": 2 / 3}
    budget = int(target_X.nbytes * selected.get(case, 1)) // 2
    limit = private_bytes() + budget
    resource.setrlimit(resource.RLIMIT_DATA, (limit, resource.RLIM_INFINITY))
    try:
        run(target_X, target_y)
    except MemoryError:
        print(json.dumps({"fits": False}))
    else:
        print(json.dumps({"fits": True}))
    """
)


@pytest.fixture(scope="module")
def memmap(tmp_path_factory):
    rng = np.random.default_rng(5)
    directory = tmp_path_factory.mktemp("native")
    xfile, yfile = directory / "X.npy", directory / "y.npy"
    X = np.lib.format.open_memmap(
        xfile, mode="w+", dtype=np.int8, shape=(NROW, NCOL), fortran_order=True
    )
    for j in range(NCOL):
        X[:, j] = rng.integers(0, 3, size=NROW)
    signal = X[:, :10].astype(np.float64) @ rng.normal(size=10)
    y = (signal + rng.logistic(size=NROW) > np.median(signal)).astype(np.float64)
    X.flush()
    del X
    np.save(yfile, y)
    return xfile, yfile


def run_child(memmap, case):
    # one malloc arena, so threads don't each reserve one during the measured fit
    # and the Rust backend, whatever the environment asks for
    env = {
        "MALLOC_ARENA_MAX": "1",
        "PYTHONWARNINGS": "ignore",
        "PYBIGLASSO_BACKEND": "rust",
    }
    result = subprocess.run(
        [sys.executable, "-c", CHILD, *map(str, memmap), case],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, **env},
        timeout=600,
    )
    return result


@pytest.mark.parametrize(
    "case", ["binomial", "gaussian", "most rows", "few rows", "cross-validation"]
)
def test_fits_within_half_of_x(memmap, case):
    result = run_child(memmap, case)
    assert result.returncode == 0, result.stderr[-2000:]
    assert json.loads(result.stdout.splitlines()[-1]) == {"fits": True}


@pytest.mark.parametrize("case", ["copy X", "malloc X", "copy the rows"])
def test_a_copy_of_x_cannot_be_allocated(memmap, case):
    """The negative controls: a numpy copy of X or of the selected rows, and a
    bare malloc of X's size, as Rust's would be."""
    result = run_child(memmap, case)
    assert result.returncode == 0, result.stderr[-2000:]
    assert json.loads(result.stdout.splitlines()[-1]) == {"fits": False}
