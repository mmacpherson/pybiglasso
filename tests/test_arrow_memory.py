"""The native memory cap covers the first target fit after opening Arrow X.

The packed-bool target is wider than the int8 target so that one copy of its
physical value buffers clearly exceeds an allowance derived from the fit's
row/feature buffers. The copy control retains those buffers without expanding
them to byte-per-value storage.
"""

import json
import os
import signal
import subprocess
import sys
import textwrap

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="Linux RLIMIT_DATA semantics"
)


def test_prediction_gather_does_not_copy_a_chunked_column():
    from pybiglasso.arrow import _ArrowMatrix

    chunks = [pa.array(np.arange(200_000, dtype=np.float64)) for _ in range(10)]
    table = pa.table({"x": pa.chunked_array(chunks)})
    previous = pa.default_memory_pool()
    pool = pa.proxy_memory_pool(previous)
    try:
        pa.set_memory_pool(pool)
        baseline = pool.bytes_allocated()
        actual = _ArrowMatrix(table)._block(np.array([1, 1_999_999]), [0])
        peak = pool.max_memory() - baseline
    finally:
        pa.set_memory_pool(previous)
    np.testing.assert_array_equal(actual[:, 0], [1.0, 199_999.0])
    # The result is 16 bytes; allow 1 MiB for Arrow metadata. ChunkedArray.take
    # allocates all 16,000,000 value bytes here and fails this derived ceiling.
    assert peak <= 1 << 20


CHILD = textwrap.dedent("""
    import ctypes, gc, json, os, resource, sys
    # glibc snapshots RLIMIT_STACK at exec for C++ workers. RUST_MIN_STACK
    # only bounds Rayon stacks; Arrow's lazily started I/O workers otherwise
    # inherit large stacks (typically 8 MiB), depending on task scheduling.
    if '_PYBIGLASSO_ARROW_STACK_SET' not in os.environ:
        _, hard = resource.getrlimit(resource.RLIMIT_STACK)
        resource.setrlimit(resource.RLIMIT_STACK, (1 << 20, hard))
        os.execve(sys.executable, sys.orig_argv,
                  {**os.environ, '_PYBIGLASSO_ARROW_STACK_SET': '1'})
    import numpy as np
    import pyarrow as pa
    from pybiglasso import biglasso
    from pybiglasso.arrow import _open_mapped

    target, warm, copy_target, case, storage = sys.argv[1:]
    n_threads = int(os.environ.get('PYBIGLASSO_MEMORY_TEST_THREADS', '2'))
    # Fully warm the bounded Arrow pools, rather than racing later worker
    # creation against the baseline.
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    # Imports, worker pools and small allocator arenas are initialized with
    # a different design. Opening the target is baselined; the cap covers its
    # first fit and whether a retained same-representation copy can coexist.
    small = _open_mapped(warm)
    biglasso(small, np.arange(small.shape[0]) % 3, nlambda=3, n_threads=n_threads)
    del small
    mapped = _open_mapped(copy_target if case == 'copy' else target)
    if storage == 'allocated':
        columns = []
        for column in mapped.itercolumns():
            chunks = [pa.Array.from_buffers(chunk.type, len(chunk),
                      [None, pa.py_buffer(bytearray(chunk.buffers()[1]))], offset=chunk.offset)
                      for chunk in column.chunks]
            columns.append(pa.chunked_array(chunks))
        X = pa.Table.from_arrays(columns, names=mapped.column_names)
        del mapped, columns, chunks
    else:
        X = mapped
    gc.collect()
    with open('/proc/self/status') as status:
        private = next(int(s.split()[1]) * 1024 for s in status if s.startswith('VmData:'))
    nrow, ncol = X.shape
    # Conservative live fit scratch: 24 nrow-length words cover eta/IRLS/w/s/r,
    # Python and native y selection/casting, masks and row indices, with a 2x
    # transient-allocation margin; seven ncol-length words cover
    # centers/scales, beta/z/multipliers and active
    # sets; four possible 256 KiB Rayon stacks; plus 4 MiB for warmed allocator
    # and runtime bookkeeping. This is derived from solver structures, not an
    # observed threshold. The packed-bool copy control is 20,000 x 9,000 =
    # 22.5 MB, over twice its 9.6 MB allowance; the int8 control is 40 MB
    # against 9.2 MB. Positive fits retain the original 20,000 x 2,000 shape,
    # so widening the negative control does not add solver work.
    allowance = (24 * nrow + 7 * ncol) * 8 + 4 * (256 << 10) + (4 << 20)
    resource.setrlimit(resource.RLIMIT_DATA,
                       (private + allowance, resource.RLIM_INFINITY))
    stage = 'open'
    try:
        stage = case
        if case == 'copy':
            # Copy each physical value buffer without expanding packed bools.
            copy = [bytes(chunk.buffers()[1]) for column in X.itercolumns()
                    for chunk in column.chunks]
        elif case == 'malloc':
            libc = ctypes.CDLL(None)
            libc.malloc.restype = ctypes.c_void_p
            if not libc.malloc(ctypes.c_size_t(X.shape[0] * X.shape[1])):
                raise MemoryError('native matrix allocation')
        stage = 'fit'
        rows = np.arange(0, X.shape[0], 2) if case == 'subset' else None
        fit = biglasso(X, np.arange(X.shape[0]) % 3, nlambda=3,
                       lambda_min=0.8, row_indices=rows, n_threads=n_threads)
        assert len(fit) == 3
    except (MemoryError, ValueError) as error:
        if isinstance(error, ValueError) and 'Resource temporarily unavailable' not in str(error):
            raise
        print(json.dumps({'fits': False, 'stage': stage}))
    else:
        print(json.dumps({'fits': True, 'stage': stage}))
""")


@pytest.fixture(scope="module", params=["int8", "bool"])
def files(request, tmp_path_factory):
    directory = tmp_path_factory.mktemp("arrow-memory")
    rng = np.random.default_rng(14)
    paths = []
    shapes = [("warm", 2_000, 20), ("target", 20_000, 2_000)]
    if request.param == "bool":
        shapes.append(("copy-target", 20_000, 9_000))
    for name, n, p in shapes:
        columns = [
            pa.array(rng.integers(0, 2, n).astype(request.param)) for _ in range(p)
        ]
        batch = pa.record_batch(columns, names=[f"x{j}" for j in range(p)])
        path = directory / (name + ".arrow")
        with pa.OSFile(str(path), "wb") as sink:
            with pa.ipc.new_file(sink, batch.schema) as writer:
                writer.write_batch(batch.slice(0, n // 2))
                writer.write_batch(batch.slice(n // 2))
        paths.append(path)
    warm, target, *copy_target = paths
    return target, warm, copy_target[0] if copy_target else target


_CASES = [
    *(
        pytest.param(case, storage, stack_cache, id=f"{case}-{storage}-{stack_cache}")
        for case in ("fit", "subset")
        for storage in ("mapped", "allocated")
        for stack_cache in (True, False)
    ),
    *(
        pytest.param("copy", storage, True, id=f"copy-{storage}")
        for storage in ("mapped", "allocated")
    ),
    pytest.param("malloc", "mapped", True, id="malloc"),
]


@pytest.mark.parametrize("case,storage,stack_cache", _CASES)
def test_open_and_first_fit_cap_with_copy_controls(files, case, stack_cache, storage):
    env = dict(
        os.environ,
        MALLOC_ARENA_MAX="1",
        RUST_MIN_STACK=str(256 * 1024),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    if not stack_cache:
        env["GLIBC_TUNABLES"] = "glibc.pthread.stack_cache_size=0"
    result = subprocess.run(
        [sys.executable, "-c", CHILD, *map(str, files), case, storage],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    positive = case in ("fit", "subset")
    if not positive and result.returncode == -signal.SIGABRT:
        assert "bad_alloc" in result.stderr or "memory allocation of" in result.stderr
        return
    assert result.returncode == 0, result.stderr[-3000:]
    outcome = json.loads(result.stdout.splitlines()[-1])
    assert outcome["fits"] is positive, result.stderr[-3000:]
    expected_stages = {"fit"} if positive else {case, "fit"}
    assert outcome["stage"] in expected_stages, result.stderr[-3000:]
