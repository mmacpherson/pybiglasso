"""The native memory cap covers opening IPC plus its first target fit."""

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="Linux RLIMIT_DATA semantics"
)

CHILD = textwrap.dedent("""
    import ctypes, gc, json, resource, sys
    import numpy as np
    import pyarrow as pa
    from pybiglasso import biglasso
    from pybiglasso.arrow import _ArrowMatrix, open_mapped

    target, warm, case, storage = sys.argv[1:]
    # Imports, worker pools and small allocator arenas are initialized with
    # a different design. The target is not even opened until under the cap.
    small = open_mapped(warm)
    biglasso(small, np.arange(small.shape[0]) % 3, nlambda=3, n_threads=2)
    del small
    if storage == 'allocated':
        mapped = open_mapped(target)
        columns = []
        for column in mapped.itercolumns():
            chunks = [pa.Array.from_buffers(chunk.type, len(chunk),
                      [None, pa.py_buffer(bytearray(chunk.buffers()[1]))], offset=chunk.offset)
                      for chunk in column.chunks]
            columns.append(pa.chunked_array(chunks))
        X = pa.Table.from_arrays(columns, names=mapped.column_names)
        del mapped, columns, chunks
    gc.collect()
    with open('/proc/self/status') as status:
        private = next(int(s.split()[1]) * 1024 for s in status if s.startswith('VmData:'))
    resource.setrlimit(resource.RLIMIT_DATA, (private + 20_000_000, resource.RLIM_INFINITY))
    stage = 'open'
    try:
        if storage == 'mapped':
            X = open_mapped(target)
        stage = case
        if case == 'copy':
            # Smallest primitive matrix copy; for packed bool this is the
            # byte-per-value expansion that must also fail the same cap.
            copy = np.empty(X.shape, dtype=np.uint8, order='F')
            for j, column in enumerate(_ArrowMatrix(X)._columns):
                start = 0
                for values, offset, count in column:
                    copy[start:start+count, j] = (values if offset is None else
                        np.unpackbits(values, bitorder='little')[offset:offset+count])
                    start += count
        elif case == 'malloc':
            libc = ctypes.CDLL(None)
            libc.malloc.restype = ctypes.c_void_p
            if not libc.malloc(ctypes.c_size_t(X.shape[0] * X.shape[1])):
                raise MemoryError('native matrix allocation')
        stage = 'fit'
        rows = np.arange(0, X.shape[0], 2) if case == 'subset' else None
        fit = biglasso(X, np.arange(X.shape[0]) % 3, nlambda=3,
                       lambda_min=0.8, row_indices=rows, n_threads=2)
        assert len(fit) == 3
    except MemoryError:
        print(json.dumps({'fits': False, 'stage': stage}))
    else:
        print(json.dumps({'fits': True, 'stage': stage}))
""")


@pytest.fixture(scope="module", params=["int8", "bool"])
def files(request, tmp_path_factory):
    directory = tmp_path_factory.mktemp("arrow-memory")
    rng = np.random.default_rng(14)
    paths = []
    for name, n, p in (("warm", 2000, 20), ("target", 20_000, 2000)):
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
    return paths[::-1]


@pytest.mark.parametrize("case", ["fit", "subset", "copy", "malloc"])
@pytest.mark.parametrize("stack_cache", [True, False])
@pytest.mark.parametrize("storage", ["mapped", "allocated"])
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
    assert result.returncode == 0, result.stderr[-3000:]
    positive = case in ("fit", "subset")
    assert json.loads(result.stdout.splitlines()[-1]) == {
        "fits": positive,
        "stage": "fit" if positive else case,
    }
