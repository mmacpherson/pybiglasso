"""First ingest under a private-memory cap, with a whole-table control."""

import json
import os
import signal
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("pyarrow")
pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="Linux RLIMIT_DATA accounting"
)


def test_wide_parquet_blocks_bound_converted_cells(tmp_path):
    import importlib

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    module = importlib.import_module("pybiglasso.ingest")
    nrow, ncol = 300, 2_000
    source = tmp_path / "wide.parquet"
    table = pa.table(
        {
            f"x{column}": pa.array(np.arange(nrow, dtype=np.int16))
            for column in range(ncol)
        }
    )
    pq.write_table(table, source, row_group_size=nrow)
    columns = tuple(table.column_names)
    largest = max(
        len(block[columns[0]])
        for block in module._blocks(source, columns, columns, 65_536, None, pa, 1 << 30)
    )
    assert largest <= module._MAX_BLOCK_CELLS // ncol


WIDE_CHILD = textwrap.dedent("""
    import ctypes, gc, json, sys, threading
    from pathlib import Path
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pybiglasso.ingest import ingest

    source, small, output, mode = map(Path, sys.argv[1:])
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    pa.set_memory_pool(pa.system_memory_pool())
    warm = ingest(small, output.with_suffix('.warm.arrow'), dtypes='preserve')
    warm_control = pq.read_table(small, use_threads=False).to_pydict()
    del warm, warm_control
    gc.collect()
    pa.default_memory_pool().release_unused()
    ctypes.CDLL(None).malloc_trim(0)
    def rss():
        with open('/proc/self/status') as status:
            return next(int(line.split()[1]) * 1024 for line in status
                        if line.startswith('VmRSS:'))
    baseline = rss()
    peak = [baseline]
    stop = threading.Event()
    def sample():
        while not stop.wait(.001):
            peak[0] = max(peak[0], rss())
    monitor = threading.Thread(target=sample)
    monitor.start()
    if mode.name == 'control':
        result = pq.read_table(source, use_threads=False).to_pydict()
        assert result
    else:
        result = ingest(source, output, dtypes='preserve')
        assert result.X.num_rows * result.X.num_columns == 1 << 22
    peak[0] = max(peak[0], rss())
    stop.set()
    monitor.join()
    print(json.dumps({'peak': peak[0] - baseline}))
""")


@pytest.fixture(scope="module")
def same_size_wide_parquet(tmp_path_factory):
    import gc

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    directory = tmp_path_factory.mktemp("wide-parquet-memory")
    paths = []
    for name, rows, columns in (
        ("narrow", 65_536, 64),
        ("wide", 2_048, 2_048),
    ):
        values = np.arange(rows, dtype=np.float64)
        table = pa.table({f"x{j}": pa.array(values) for j in range(columns)})
        path = directory / f"{name}.parquet"
        pq.write_table(table, path, row_group_size=rows, compression="none")
        paths.append(path)
        del table
        gc.collect()
    small = directory / "small.parquet"
    pq.write_table(pa.table({"x": [1.0, 2.0]}), small, compression="none")
    return paths, small


@pytest.mark.parametrize("width", ["narrow", "wide"])
@pytest.mark.parametrize("mode", ["ingest", "control"])
def test_wide_parquet_peak_is_bounded_by_source_size(
    tmp_path, same_size_wide_parquet, width, mode
):
    paths, small = same_size_wide_parquet
    source = paths[width == "wide"]
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            WIDE_CHILD,
            str(source),
            str(small),
            str(tmp_path / f"{width}-{mode}.arrow"),
            mode,
        ],
        env={
            **os.environ,
            "MALLOC_ARENA_MAX": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        },
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    peak = json.loads(completed.stdout.splitlines()[-1])["peak"]
    logical_bytes = (1 << 22) * 8
    columns = 64 if width == "narrow" else 2_048
    # Source decoding, staging, mapped writer input and the final file may each
    # account for the logical values. Add 32 MiB of fixed reader/writer state,
    # then 12 KiB per column. Isolating _write_ipc with one float64 value per
    # staged column measured at most 6.4 KiB of incremental cost per column
    # from 64 through 4,096 columns; 12 KiB leaves 87.5% headroom for dependency
    # and allocator differences.
    # The explicit term keeps the 2,048-column case honest without fitting the
    # ceiling to this test's end-to-end RSS.
    budget = 4 * logical_bytes + (32 << 20) + columns * (12 << 10)
    assert (peak <= budget) if mode == "ingest" else (peak > budget), (
        peak,
        budget,
    )


CHILD = textwrap.dedent("""
    import ctypes, gc, json, os, resource, sys
    # glibc snapshots RLIMIT_STACK at exec for C++ worker stacks. Bound those
    # stacks independently of whether warm-up thread stacks stay cached.
    if '_PYBIGLASSO_INGEST_STACK_SET' not in os.environ:
        _, hard = resource.getrlimit(resource.RLIMIT_STACK)
        resource.setrlimit(resource.RLIMIT_STACK, (1 << 20, hard))
        os.execve(sys.executable, sys.orig_argv,
                  {**os.environ, '_PYBIGLASSO_INGEST_STACK_SET': '1'})
    from pathlib import Path
    import pyarrow as pa
    import pyarrow.csv as pc
    from pybiglasso.ingest import ingest

    source, output, small, control = map(Path, sys.argv[1:])
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    # Use libc for Arrow buffers as well: a large pre-reserved allocator arena
    # would let the negative control fit without a fresh counted allocation.
    pa.set_memory_pool(pa.system_memory_pool())
    # Initialize the reader, writer, imports and allocator on different input.
    warm = ingest(small, output.with_suffix('.warm.arrow'), block_size=8192)
    warm_table = pc.read_csv(small, read_options=pc.ReadOptions(use_threads=False))
    del warm, warm_table
    gc.collect()
    pa.default_memory_pool().release_unused()
    libc = ctypes.CDLL(None)
    libc.malloc_trim(0)
    with open('/proc/self/status') as status:
        private = next(int(line.split()[1])*1024 for line in status
                       if line.startswith('VmData:'))
    # 40k * 8 * 8 = 2,560,000 output bytes. An extra 1,280,000 private
    # bytes allows bounded CSV blocks but cannot hold the numeric table.
    limit = private + 40_000 * 8 * 8 // 2
    resource.setrlimit(resource.RLIMIT_DATA, (limit, resource.RLIM_INFINITY))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    try:
        if str(control) == 'copy':
            whole = pc.read_csv(source, read_options=pc.ReadOptions(use_threads=False))
        result = ingest(source, output, block_size=8192)
        assert result.X.shape == (40_000, 8)
        print(json.dumps({'fits': True}))
    except MemoryError:
        print(json.dumps({'fits': False}))
""")


@pytest.mark.parametrize("control", ["stream", "copy"])
def test_csv_streams_under_less_than_table_memory(tmp_path, control):
    source = tmp_path / "large.csv"
    small = tmp_path / "small.csv"
    header = ",".join(f"x{i}" for i in range(8)) + "\n"
    row = ",".join("0.1" for _ in range(8)) + "\n"
    # Fixture construction is itself bounded; neither the parent nor child
    # creates the full table before establishing the child's memory limit.
    with source.open("w") as stream:
        stream.write(header)
        for _ in range(40):
            stream.write(row * 1000)
    small.write_text(header + row * 8)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            CHILD,
            str(source),
            str(tmp_path / "X.arrow"),
            str(small),
            control,
        ],
        env={
            **os.environ,
            "MALLOC_ARENA_MAX": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        },
        capture_output=True,
        text=True,
        timeout=180,
    )
    if control == "copy" and completed.returncode == -signal.SIGABRT:
        # Arrow's C++ parser can abort rather than translate an allocation
        # failure into ArrowMemoryError. Require allocation-specific evidence,
        # not an arbitrary failing child (e.g. thread-creation failure).
        assert "Out of memory: malloc of size" in completed.stderr
        return
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert json.loads(completed.stdout.splitlines()[-1]) == {
        "fits": control == "stream"
    }


STREAM_CHILD = textwrap.dedent("""
    import ctypes, gc, json, os, resource, sys
    import numpy as np
    import pyarrow as pa
    from pathlib import Path
    from pybiglasso.ingest import ingest

    output, mode = Path(sys.argv[1]), sys.argv[2]
    batch_size = int(sys.argv[3])
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    pa.set_memory_pool(pa.system_memory_pool())
    schema = pa.schema([(f'x{i}', pa.float64()) for i in range(8)])
    def reader(rows):
        def batches():
            for start in range(0, rows, batch_size):
                length = min(batch_size, rows-start)
                # Retain distinct ordinary-sized producer buffers so read_all
                # must retain the whole stream. Reuse the one-row buffers: that
                # case isolates retained batch metadata instead.
                if start == 0 or batch_size != 1:
                    values = pa.record_batch([
                        pa.array(np.full(batch_size, 0.25)) for _ in range(8)
                    ], schema=schema)
                yield values.slice(0, length)
        return pa.RecordBatchReader.from_batches(schema, batches())
    dtype_option = {"x0": "float64"} if mode == 'mapping' else (
        'auto' if mode == 'copy' else mode)
    # Warm both conversion routes and the whole-table control independently.
    for dtype in ('auto', 'preserve'):
        warm = ingest(reader(8), output.with_suffix(f'.{dtype}'), dtypes=dtype,
                      block_size=512)
        del warm
    whole = reader(8).read_all()
    del whole
    gc.collect()
    pa.default_memory_pool().release_unused()
    ctypes.CDLL(None).malloc_trim(0)
    with open('/proc/self/status') as status:
        private = next(int(line.split()[1])*1024 for line in status
                       if line.startswith('VmData:'))
    # Input has 2,560,000 bytes. The 1,280,000 allowance cannot hold it;
    # only a producer batch and bounded conversion work fit.
    resource.setrlimit(resource.RLIMIT_DATA,
                       (private + 1_280_000, resource.RLIM_INFINITY))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    try:
        source = reader(40_000)
        if mode == 'copy':
            print('WHOLE_TABLE_READ_START', file=sys.stderr, flush=True)
            source = source.read_all()
            print('WHOLE_TABLE_READ_DONE', file=sys.stderr, flush=True)
        result = ingest(source, output, dtypes=dtype_option,
                        block_size=512)
        assert result.X.shape == (40_000, 8)
        assert result.X['x0'][39_999].as_py() == 0.25
        print(json.dumps({'fits': True}))
    except MemoryError:
        print(json.dumps({'fits': False}))
""")


@pytest.mark.parametrize(
    "mode,batch_size,malloc_mmap_threshold",
    [
        ("preserve", 512, None),
        ("auto", 512, None),
        ("copy", 512, None),
        ("copy", 512, "4096"),
        ("preserve", 1, None),
        ("auto", 1, None),
        ("mapping", 1, None),
    ],
)
def test_arrow_stream_memory_is_bounded_by_batches(
    tmp_path, mode, batch_size, malloc_mmap_threshold
):
    allocator_env = (
        {}
        if malloc_mmap_threshold is None
        else {"MALLOC_MMAP_THRESHOLD_": malloc_mmap_threshold}
    )
    # A small mmap threshold reproduces the C++ bad_alloc path on glibc;
    # the same unchanged cap can otherwise raise a Python MemoryError.
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            STREAM_CHILD,
            str(tmp_path / "data.feather"),
            mode,
            str(batch_size),
        ],
        env={
            **os.environ,
            **allocator_env,
            "MALLOC_ARENA_MAX": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        },
        capture_output=True,
        text=True,
        timeout=180,
    )
    _assert_stream_memory_result(completed, mode)


def _assert_stream_memory_result(completed, mode):
    if mode == "copy" and completed.returncode == -signal.SIGABRT:
        # Exhausting the cap while read_all retains producer batches can fail
        # in Arrow's C++ NumPyConverter before it can translate bad_alloc into
        # MemoryError. Never accept an arbitrary crash, a positive-case abort,
        # or an allocation failure after the deliberate whole-table read.
        assert "WHOLE_TABLE_READ_START" in completed.stderr
        assert "WHOLE_TABLE_READ_DONE" not in completed.stderr
        assert "std::bad_alloc" in completed.stderr
        return
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert json.loads(completed.stdout.splitlines()[-1]) == {"fits": mode != "copy"}


@pytest.mark.parametrize(
    "mode,returncode,stderr",
    [
        ("auto", -signal.SIGABRT, "WHOLE_TABLE_READ_START\nstd::bad_alloc"),
        ("copy", -signal.SIGABRT, "std::bad_alloc"),
        ("copy", -signal.SIGABRT, "WHOLE_TABLE_READ_START\nthread creation failed"),
        (
            "copy",
            -signal.SIGABRT,
            "WHOLE_TABLE_READ_START\nWHOLE_TABLE_READ_DONE\nstd::bad_alloc",
        ),
        ("copy", -signal.SIGSEGV, "WHOLE_TABLE_READ_START\nstd::bad_alloc"),
    ],
)
def test_stream_memory_control_rejects_unrelated_failures(mode, returncode, stderr):
    completed = subprocess.CompletedProcess([], returncode, stdout="", stderr=stderr)
    with pytest.raises(AssertionError):
        _assert_stream_memory_result(completed, mode)
