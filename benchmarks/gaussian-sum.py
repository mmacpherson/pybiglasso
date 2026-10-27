"""AB/BA Gaussian residual-sum measurements using two actual Rust extensions.

Whole-fit clocks include validation/standardization, but exclude data generation,
page eviction and output hashing. Hashes check trajectories between builds and
threads; they are not expected solver values. Run out-of-core separately under
a cgroup memory cap smaller than its 1.6 GB design, with scratch on real disk.
"""

import argparse
import hashlib
import importlib.util
import json
import mmap
import os
import platform
import statistics
import subprocess
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

import pybiglasso as bl
from pybiglasso import _rust

CASES = {
    "tall": (100_000, 100),
    "wide": (2_000, 20_000),
    "out-of-core": (4_000, 50_000),
}


def path_hash(path):
    digest = hashlib.sha256()
    coef = path.coef.tocsc()
    for value in (
        path.center,
        path.scale,
        path.lambdas,
        path.intercept,
        coef.data,
        coef.indices,
        coef.indptr,
        path.iterations,
        path.n_rejected,
        path.deviance,
    ):
        digest.update(str((value.dtype.str, value.shape)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def accounting():
    relative = Path("/proc/self/cgroup").read_text().strip().split("::", 1)[1]
    cgroup = Path("/sys/fs/cgroup") / relative.lstrip("/")
    cap = (cgroup / "memory.max").read_text().strip()
    stat = dict(
        line.split() for line in (cgroup / "memory.stat").read_text().splitlines()
    )
    io = dict(
        line.split(": ", 1) for line in Path("/proc/self/io").read_text().splitlines()
    )
    return dict(
        memory_cap=None if cap == "max" else int(cap),
        memory_peak=int((cgroup / "memory.peak").read_text()),
        refaulted_bytes=int(stat["workingset_refault_file"])
        * os.sysconf("SC_PAGE_SIZE"),
        read_bytes=int(io["read_bytes"]),
    )


def evict(filename):
    with filename.open("rb") as file:
        os.fsync(file.fileno())
        os.posix_fadvise(file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def data(case, dtype, directory):
    n, p = CASES[case]
    rng = np.random.default_rng(65)
    y = 3 + rng.standard_normal(n) * 0.5
    filename = directory / "X.bin" if case == "out-of-core" else None
    X = (
        np.memmap(filename, mode="w+", shape=(n, p), dtype=dtype, order="F")
        if filename
        else np.empty((n, p), dtype=dtype, order="F")
    )
    for j in range(p):
        column = (
            rng.integers(-2, 3, size=n, dtype=np.int8)
            if dtype == "int8"
            else rng.standard_normal(n).astype(dtype)
        )
        X[:, j] = column
        if j < 5:
            y += column.astype(float) * (1.5, -1.0, 0.75, 0.5, -0.25)[j]
    if filename:
        X.flush()
        del X
        X = np.memmap(filename, mode="r", shape=(n, p), dtype=dtype, order="F")
    else:
        X.flags.writeable = False
    return X, y, filename


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-extension", type=Path, required=True)
    parser.add_argument("--case", choices=CASES, nargs="+", default=["tall", "wide"])
    parser.add_argument(
        "--dtype", choices=["float64", "float32", "int8"], default="float64"
    )
    parser.add_argument(
        "--selection", choices=["all", "mask", "gather", "repeated"], default="all"
    )
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 6, 12])
    parser.add_argument("--alpha", type=float, nargs="+", default=[1.0, 0.5])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--nlambda", type=int, default=30)
    parser.add_argument("--scratch-dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.repeats < 2 or args.nlambda < 2 or min(args.threads) < 1:
        parser.error("Use at least two repetitions/lambdas and positive thread counts.")
    if "out-of-core" in args.case:
        if args.scratch_dir is None:
            parser.error("Out-of-core requires --scratch-dir on real disk.")
        filesystem = subprocess.check_output(
            ["stat", "-f", "-c", "%T", str(args.scratch_dir)], text=True
        ).strip()
        if filesystem == "tmpfs":
            parser.error("Out-of-core scratch must not use tmpfs.")
    spec = importlib.util.spec_from_file_location(
        "baseline._rust", args.baseline_extension
    )
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    variants = {"baseline": baseline, "candidate": _rust}
    cpu = next(
        line.split(":", 1)[1].strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines()
        if line.startswith("model name")
    )
    metadata = dict(
        cpu=cpu,
        python=platform.python_version(),
        numpy=np.__version__,
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        baseline_extension_sha256=hashlib.sha256(
            args.baseline_extension.read_bytes()
        ).hexdigest(),
        candidate_extension_sha256=hashlib.sha256(
            Path(_rust.__file__).read_bytes()
        ).hexdigest(),
        seed=65,
    )
    try:
        for case in args.case:
            with tempfile.TemporaryDirectory(
                dir=args.scratch_dir, prefix="gaussian-sum-"
            ) as temporary:
                X, y, filename = data(case, args.dtype, Path(temporary))
                n, p = X.shape
                rows = dict(
                    all=None,
                    mask=np.random.default_rng(66).permutation(n)[: 9 * n // 10],
                    gather=np.arange(0, n, 4),
                    repeated=np.r_[0, np.arange(0, n, 4)],
                )[args.selection]
                if filename:
                    cap = accounting()["memory_cap"]
                    if cap is None or cap >= X.nbytes:
                        parser.error(
                            "Out-of-core requires a cgroup memory cap smaller than X."
                        )
                for alpha in args.alpha:
                    hashes = set()
                    for threads in args.threads:
                        options = dict(
                            family="gaussian",
                            backend="rust",
                            alpha=alpha,
                            nlambda=args.nlambda,
                            lambda_min=0.05,
                            n_threads=threads,
                            row_indices=rows,
                        )
                        # Warm both libraries/pools; cold file pages are still evicted per timed fit.
                        for extension in variants.values():
                            bl._rust = extension
                            hashes.add(path_hash(bl.biglasso(X, y, **options)))
                        times = {name: [] for name in variants}
                        resources = {name: [] for name in variants}
                        for rep in range(args.repeats):
                            order = list(variants.items())
                            if rep % 2:
                                order.reverse()
                            for name, extension in order:
                                bl._rust = extension
                                if filename:
                                    # Drop this mapping's pages before requesting global clean-page eviction.
                                    X._mmap.madvise(mmap.MADV_DONTNEED)
                                    evict(filename)
                                before = accounting() if filename else None
                                with warnings.catch_warnings(record=True) as caught:
                                    start = time.perf_counter()
                                    fit = bl.biglasso(X, y, **options)
                                    elapsed = time.perf_counter() - start
                                after = accounting() if filename else None
                                times[name].append(elapsed)
                                assert not caught and len(fit) == args.nlambda
                                hashes.add(path_hash(fit))
                                if filename:
                                    resources[name].append(
                                        dict(
                                            memory_cap=after["memory_cap"],
                                            scope_peak_bytes=after["memory_peak"],
                                            read_bytes=after["read_bytes"]
                                            - before["read_bytes"],
                                            refaulted_bytes=after["refaulted_bytes"]
                                            - before["refaulted_bytes"],
                                        )
                                    )
                        assert len(hashes) == 1, (
                            "Baseline/candidate/thread trajectory changed bits."
                        )
                        medians = {
                            name: statistics.median(values)
                            for name, values in times.items()
                        }
                        result = dict(
                            **metadata,
                            family="gaussian",
                            case=case,
                            nrow=n,
                            ncol=p,
                            dtype=args.dtype,
                            selection=args.selection,
                            alpha=alpha,
                            threads=threads,
                            nlambda=args.nlambda,
                            lambda_min=0.05,
                            times=times,
                            median_seconds=medians,
                            speedup=medians["baseline"] / medians["candidate"],
                            path_hash=next(iter(hashes)),
                            resources=resources,
                        )
                        line = json.dumps(result)
                        print(line, flush=True)
                        if args.out:
                            with args.out.open("a") as output:
                                output.write(line + "\n")
    finally:
        bl._rust = _rust


if __name__ == "__main__":
    main()
