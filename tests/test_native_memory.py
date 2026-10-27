"""X is never copied by native code either.

tracemalloc (test_out_of_core.py) sees numpy's allocations but not the Rust
extension's, which go straight to malloc. Here each fit runs in a child
process whose private writable memory is capped (RLIMIT_DATA) at what it
already holds after a warm-up fit, plus half of X's own size. A copy of X in
any dtype, by numpy or by Rust, can't be allocated: numpy raises MemoryError,
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
    X, y = np.load(xfile, mmap_mode="r"), np.load(yfile)
    nrow = X.shape[0]
    gaussian = y + np.arange(nrow) % 7  # any continuous response

    def run():
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
        elif case == "cross-validation":
            from pybiglasso.sklearn import BiglassoClassifierCV
            BiglassoClassifierCV(n_alphas=6, alpha_min_ratio=0.2, cv=3, n_jobs=1).fit(X, y)
        elif case == "copy X":  # the negative controls: these must fail
            np.array(X)
        elif case == "malloc X":
            libc = ctypes.CDLL(None)
            libc.malloc.restype = ctypes.c_void_p
            if not libc.malloc(ctypes.c_size_t(X.nbytes)):
                raise MemoryError("malloc")

    def private_bytes():
        with open("/proc/self/status") as status:
            for line in status:
                if line.startswith("VmData:"):
                    return int(line.split()[1]) * 1024

    if not case.endswith(" X"):
        run()  # thread pools, imports and the allocator's arenas, first
    budget = X.nbytes // 2
    limit = private_bytes() + budget
    resource.setrlimit(resource.RLIMIT_DATA, (limit, resource.RLIM_INFINITY))
    try:
        run()
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
    env = {"MALLOC_ARENA_MAX": "1", "PYTHONWARNINGS": "ignore"}
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


@pytest.mark.parametrize("case", ["copy X", "malloc X"])
def test_a_copy_of_x_cannot_be_allocated(memmap, case):
    """The negative controls: a numpy copy, and a bare malloc as Rust's would be."""
    result = run_child(memmap, case)
    assert result.returncode == 0, result.stderr[-2000:]
    assert json.loads(result.stdout.splitlines()[-1]) == {"fits": False}
