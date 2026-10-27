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
    # 160k * 8 * 8 = 10,240,000 output bytes. An extra 5,120,000 private
    # bytes allows bounded CSV blocks but cannot hold the numeric table.
    limit = private + 160_000 * 8 * 8 // 2
    resource.setrlimit(resource.RLIMIT_DATA, (limit, resource.RLIM_INFINITY))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    try:
        if str(control) == 'copy':
            whole = pc.read_csv(source, read_options=pc.ReadOptions(use_threads=False))
        result = ingest(source, output, block_size=8192)
        assert result.design.shape == (160_000, 8)
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
        for _ in range(160):
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
