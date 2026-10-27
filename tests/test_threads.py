"""Thread selection is metadata-only; thread count cannot change a path."""

import ctypes
import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from support.paths import assert_bitwise

import pybiglasso as bl
from pybiglasso import _rust, _threads


@pytest.mark.parametrize("requested", [None, "auto"])
@pytest.mark.parametrize(
    "work, expected", [(9_999_999, 1), (10_000_000, 6), (20_000_000, 6)]
)
def test_auto_boundary_without_reading_x(monkeypatch, requested, work, expected):
    monkeypatch.delenv("RAYON_NUM_THREADS", raising=False)
    monkeypatch.setattr(_threads, "physical_cores", lambda: 6)
    # Not even an array: selection may inspect shape, never matrix values.
    X = SimpleNamespace(shape=(work, 1))
    assert _threads.resolve_threads(requested, X, work) == expected


@pytest.mark.parametrize("requested", [1, 7, np.int64(3)])
def test_explicit_threads_override_auto_and_environment(monkeypatch, requested):
    monkeypatch.setenv("RAYON_NUM_THREADS", "1")
    monkeypatch.setattr(
        _threads, "physical_cores", lambda: pytest.fail("Topology probed")
    )
    assert _threads.resolve_threads(requested, None, 0) == requested


@pytest.mark.parametrize(
    "requested", [0, -1, True, np.bool_(False), 1.5, "all", "", [], np.array([2])]
)
def test_invalid_thread_options(requested):
    with pytest.raises(ValueError, match="n_threads"):
        _threads.resolve_threads(requested, None, 0)


@pytest.mark.parametrize(
    "value, expected", [("1", 1), ("3", 3), ("12", 6), ("0", 6), ("-2", 6), ("bad", 6)]
)
def test_rayon_is_an_auto_ceiling(monkeypatch, value, expected):
    monkeypatch.setenv("RAYON_NUM_THREADS", value)
    monkeypatch.setattr(_threads, "physical_cores", lambda: 6)
    X = SimpleNamespace(shape=(10_000, 1_000))
    assert _threads.resolve_threads("auto", X, 10_000) == expected


def test_memmap_ownership_not_just_subclass(monkeypatch, tmp_path):
    monkeypatch.setattr(
        _threads, "physical_cores", lambda: pytest.fail("Topology probed")
    )
    X = np.memmap(tmp_path / "X", mode="w+", shape=(8, 4), order="F")
    for array in (
        X,
        X[:, :2],
        np.asarray(X),
        np.asarray(X)[:, :2],
        np.frombuffer(X.base, dtype="uint8").reshape(-1, 1),
        np.lib.stride_tricks.as_strided(X, shape=X.shape, strides=X.strides),
    ):
        assert _threads._mapped_array(array)
        assert _threads.resolve_threads(None, array, 10_000_000) == 1
    assert not _threads._mapped_array(np.array(X))


def test_recorded_work_cutoff():
    script = Path(__file__).parents[1] / "benchmarks/thread_policy.py"
    spec = importlib.util.spec_from_file_location("thread_policy", script)
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    rows = list(report.comparisons())
    small, large, midpoint = report.cutoff(rows)
    assert (small, large, midpoint) == (5_000_000, 20_000_000, _threads.PARALLEL_WORK)
    # Supplemental SMT worker counts do not set the physical-core cutoff.
    smt = dict(threads=12, work=200_000_000, speedup=0.5)
    assert report.cutoff([*rows, smt]) == (small, large, midpoint)
    # Eight-worker measurements participate in the same size rule.
    arm = [
        dict(threads=6, work=5_000_000, speedup=1.0),
        dict(threads=8, work=20_000_000, speedup=1.2),
    ]
    assert report.cutoff(arm) == (small, large, midpoint)
    contradictory = [
        dict(threads=6, work=1_000_000, speedup=1.2),
        dict(threads=6, work=5_000_000, speedup=1.0),
    ]
    with pytest.raises(ValueError, match="do not separate"):
        report.cutoff(contradictory)


def test_linux_smt_affinity_and_old_kernel_name(tmp_path):
    # Two sockets can reuse core IDs: sibling masks identify actual cores.
    for cpu, siblings in {
        0: "0,4",
        4: "0,4",
        1: "1,5",
        5: "1,5",
        8: "8-9",
        9: "8-9",
    }.items():
        topology = tmp_path / f"cpu{cpu}" / "topology"
        topology.mkdir(parents=True)
        (topology / "thread_siblings_list").write_text(siblings)
    assert _threads._linux_cores({0, 1, 4, 5, 8, 9}, tmp_path) == 3
    assert _threads._linux_cores({0, 4}, tmp_path) == 1
    assert _threads._linux_cores({4, 5, 9}, tmp_path) == 3
    assert _threads._linux_cores({0, 99}, tmp_path) == 1
    (tmp_path / "cpu0/topology/core_cpus_list").write_text("invalid")
    assert _threads._linux_cores({0, 1}, tmp_path) == 2
    (tmp_path / "cpu0/topology/thread_siblings_list").write_text("2-3")
    assert _threads._linux_cores({0, 1}, tmp_path) == 1


def test_mac_probe_is_bounded_and_cached(monkeypatch):
    calls = []

    def sysctl(command, **kwargs):
        calls.append((command, kwargs))
        return "8\n"

    _threads._mac_cores.cache_clear()
    monkeypatch.setattr(_threads.subprocess, "check_output", sysctl)
    try:
        assert _threads._mac_cores() == _threads._mac_cores() == 8
        assert calls == [
            (["/usr/sbin/sysctl", "-n", "hw.physicalcpu"], {"text": True, "timeout": 1})
        ]
    finally:
        _threads._mac_cores.cache_clear()


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32", "unknown"])
def test_physical_core_dispatch_and_safe_failure(monkeypatch, platform):
    monkeypatch.setattr(_threads.sys, "platform", platform)
    monkeypatch.setattr(
        _threads.os, "sched_getaffinity", lambda _: {2, 6}, raising=False
    )
    monkeypatch.setattr(
        _threads, "_linux_cores", lambda cpus: 1 if cpus == {2, 6} else 99
    )
    monkeypatch.setattr(_threads, "_mac_cores", lambda: 8)
    monkeypatch.setattr(_threads, "_windows_cores", lambda: 4)
    assert (
        _threads.physical_cores()
        == {"linux": 1, "darwin": 8, "win32": 4, "unknown": 1}[platform]
    )

    def unavailable(*_):
        raise OSError("Unavailable")

    for name in ("_linux_cores", "_mac_cores", "_windows_cores"):
        monkeypatch.setattr(_threads, name, unavailable)
    assert _threads.physical_cores() == 1


def test_mac_timeout_falls_back(monkeypatch):
    def timeout():
        raise subprocess.TimeoutExpired("sysctl", 1)

    monkeypatch.setattr(_threads.sys, "platform", "darwin")
    monkeypatch.setattr(_threads, "_mac_cores", timeout)
    assert _threads.physical_cores() == 1


def test_native_topology_probe_smoke():
    # Also execute the real OS interface in each CI platform, not only mocks.
    # Unknown topology is allowed to return one; ABI/type failures must not
    # escape into a fit or crash the process.
    count = _threads.physical_cores()
    assert isinstance(count, int) and count >= 1


class NativeFunction:
    def __init__(self, call):
        self.call = call

    def __call__(self, *args):
        return self.call(*args)


@pytest.mark.parametrize(
    "mask, expected", [(0b1111, 2), (0b0011, 1), (0b0101, 2), (0, 1)]
)
@pytest.mark.parametrize("failure", [None, "sizing", "read", "affinity"])
def test_windows_native_core_masks(monkeypatch, mask, expected, failure):
    # Actual ctypes records, not a mocked final count: two SMT cores plus a
    # package record which must never be counted as a third core.
    def information(records, length):
        if records is None:
            length._obj.value = (32 if ctypes.sizeof(ctypes.c_void_p) == 8 else 24) * 3
            return 0
        if failure == "read":
            return 0
        for record, bits, relationship in zip(
            records, [3, 12, 15], [0, 0, 3], strict=True
        ):
            record.mask, record.relationship = bits, relationship
        return 1

    def affinity(_handle, allowed, system):
        allowed._obj.value, system._obj.value = mask, 15
        return failure != "affinity"

    kernel = SimpleNamespace(
        GetLogicalProcessorInformation=NativeFunction(information),
        GetCurrentProcess=NativeFunction(lambda: -1),
        GetProcessAffinityMask=NativeFunction(affinity),
    )
    monkeypatch.setattr(
        _threads.ctypes, "WinDLL", lambda *a, **k: kernel, raising=False
    )
    monkeypatch.setattr(
        _threads.ctypes,
        "get_last_error",
        lambda: 5 if failure == "sizing" else 122,
        raising=False,
    )
    assert _threads._windows_cores() == (1 if failure else expected)
    if failure not in ("sizing", "read"):
        assert kernel.GetCurrentProcess.restype is ctypes.c_void_p


@pytest.mark.parametrize("family", ["binomial", "gaussian"])
@pytest.mark.parametrize("selection", ["all", "mask", "gather", "repeated"])
@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("dtype", ["int8", "float32", "float64"])
def test_auto_routing_and_bits(monkeypatch, tmp_path, family, selection, mapped, dtype):
    monkeypatch.delenv("RAYON_NUM_THREADS", raising=False)
    monkeypatch.setattr(_threads, "physical_cores", lambda: 3)
    monkeypatch.setattr(_threads, "PARALLEL_WORK", 4_096)
    rng = np.random.default_rng(11)
    X = np.asfortranarray(rng.normal(size=(128, 64)).astype(dtype))
    if mapped:
        data = np.memmap(
            tmp_path / "X", mode="w+", shape=X.shape, dtype=dtype, order="F"
        )
        data[:] = X
        data.flush()
        X = np.asarray(
            np.memmap(tmp_path / "X", mode="r", shape=X.shape, dtype=dtype, order="F")
        )
    y = (
        (np.arange(len(X)) % 7 < 3).astype(float)
        if family == "binomial"
        else 0.25 + X[:, 0] - 0.5 * X[:, 1]
    )
    rows = {
        "all": None,
        "mask": np.arange(0, 128, 2),
        "gather": np.arange(0, 128, 4),
        "repeated": np.r_[np.arange(64), 0],
    }[selection]
    options = dict(
        family=family,
        row_indices=rows,
        lambdas=[0.2, 0.1, 0.03],
        eps=1e-14,
        alpha=0.5,
        stop_on_saturation=False,
    )
    native = _rust.fit_path
    calls = []

    def fit(*args, **kwargs):
        assert args[0] is X
        calls.append(args[14])
        return native(*args, **kwargs)

    monkeypatch.setattr(_rust, "fit_path", fit)
    one = bl.biglasso(X, y, n_threads=1, **options)
    for requested in (None, "auto", 3):
        other = bl.biglasso(X, y, n_threads=requested, **options)
        assert_bitwise(one, other)
    expected_auto = 1 if mapped or selection == "gather" else 3
    assert calls == [1, expected_auto, expected_auto, 3]
