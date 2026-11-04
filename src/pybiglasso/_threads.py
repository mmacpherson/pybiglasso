"""A conservative, metadata-only thread policy; see benchmarks/threads.md."""

import ctypes
import mmap
import numbers
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from .arrow import _ArrowMatrix, _is_arrow

# Logarithmic midpoint between the recorded 5M-entry near-neutral cases and
# the smallest sustained parallel win (20M entries). Not a measured crossover.
PARALLEL_WORK = 10_000_000


def _mapped_array(array):
    # np.asarray(memmap) and ordinary slices can hide the subclass. Follow
    # ownership, not flags or size, without inspecting/copying any X values.
    seen = set()
    while array is not None and id(array) not in seen:
        seen.add(id(array))
        if isinstance(array, (np.memmap, mmap.mmap)):
            return True
        # frombuffer uses a memoryview; stride_tricks can insert DummyArray.
        array = (
            array.obj if isinstance(array, memoryview) else getattr(array, "base", None)
        )
    return False


def _linux_cores(cpus, root=Path("/sys/devices/system/cpu")):
    cores = set()
    for cpu in cpus:
        topology = root / f"cpu{cpu}" / "topology"
        for name in ("core_cpus_list", "thread_siblings_list"):
            try:
                siblings = set()
                for item in (topology / name).read_text().strip().split(","):
                    bounds = [int(value) for value in item.split("-")]
                    if len(bounds) == 1:
                        siblings.add(bounds[0])
                    elif len(bounds) == 2 and 0 <= bounds[0] <= bounds[1]:
                        siblings.update(range(bounds[0], bounds[1] + 1))
                    else:
                        raise ValueError("Invalid CPU topology")
                if cpu not in siblings:
                    raise ValueError("Missing CPU in topology")
                cores.add(frozenset(siblings))
                break
            except (OSError, ValueError):
                continue
        else:
            return 1
    return max(1, len(cores))


@lru_cache(maxsize=1)
def _mac_cores():
    # Apple silicon's measured worker count uses performance cores; Intel
    # Macs and older kernels can expose only the all-physical-core key.
    for key in ("hw.perflevel0.physicalcpu", "hw.physicalcpu"):
        try:
            count = int(
                subprocess.check_output(
                    ["/usr/sbin/sysctl", "-n", key],
                    text=True,
                    timeout=1,
                    stderr=subprocess.DEVNULL,
                ).strip()
            )
            if count > 0:
                return count
        except (OSError, ValueError, subprocess.SubprocessError):
            continue
    raise ValueError("Physical core count unavailable")


def _windows_cores():
    # Windows' fixed-size topology records include core masks, so partial
    # process affinity does not count both SMT siblings as separate cores.
    class Details(ctypes.Union):
        _fields_ = [("reserved", ctypes.c_uint64 * 2)]

    class Processor(ctypes.Structure):
        _fields_ = [
            ("mask", ctypes.c_size_t),
            ("relationship", ctypes.c_uint32),
            ("details", Details),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    information = kernel.GetLogicalProcessorInformation
    information.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    information.restype = ctypes.c_int
    size = ctypes.c_uint32()
    if information(None, ctypes.byref(size)) or ctypes.get_last_error() != 122:
        return 1  # ERROR_INSUFFICIENT_BUFFER is the expected sizing response.
    if not size.value or size.value % ctypes.sizeof(Processor):
        return 1
    records = (Processor * (size.value // ctypes.sizeof(Processor)))()
    if not information(records, ctypes.byref(size)):
        return 1
    process = kernel.GetCurrentProcess
    process.restype = ctypes.c_void_p
    affinity = kernel.GetProcessAffinityMask
    affinity.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    affinity.restype = ctypes.c_int
    allowed, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not affinity(process(), ctypes.byref(allowed), ctypes.byref(system)):
        return 1
    # RelationProcessorCore == 0. A zero affinity mask (including unsupported
    # multi-group arrangements) falls back conservatively instead of guessing.
    return max(
        1, sum(r.relationship == 0 and bool(r.mask & allowed.value) for r in records)
    )


def physical_cores():
    """Available physical cores (performance cores on macOS), or one."""
    try:
        if sys.platform.startswith("linux"):
            return _linux_cores(os.sched_getaffinity(0))
        if sys.platform == "darwin":
            return max(1, _mac_cores())
        if sys.platform == "win32":
            return _windows_cores()
    except (AttributeError, OSError, ValueError, subprocess.SubprocessError):
        pass
    return 1


def resolve_threads(requested, X, n_fit):
    """Resolve a public thread option without touching matrix values."""
    if isinstance(requested, numbers.Integral) and not isinstance(
        requested, (bool, np.bool_)
    ):
        if requested >= 1:
            return int(requested)
    elif requested is None or (isinstance(requested, str) and requested == "auto"):
        if (
            isinstance(X, _ArrowMatrix)
            or _is_arrow(X)
            or _mapped_array(X)
            or n_fit * X.shape[1] < PARALLEL_WORK
        ):
            return 1
        count = physical_cores()
        # Preserve the existing environment control as an auto-mode ceiling;
        # an explicit n_threads integer still takes precedence over it.
        try:
            limit = int(os.environ.get("RAYON_NUM_THREADS", "0"))
        except ValueError:
            limit = 0
        return min(count, limit) if limit > 0 else count
    raise ValueError("n_threads must be a positive integer, None, or 'auto'.")
