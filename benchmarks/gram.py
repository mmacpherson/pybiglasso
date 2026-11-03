#!/usr/bin/env python3
"""Compare explicit Gaussian update modes and sklearn's native path on Linux.

Run on an idle machine after building, with BLAS/OMP threads fixed before
starting Python, e.g. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
uv run python benchmarks/gram.py --dataset tall-50k-2k --cpu 0.
Data generation and sklearn centering are outside the intervals. The
pybiglasso intervals cover its public call (including its own standardization);
sklearn covers enet_path only. A setting first requires 180 seconds below
10% user+system CPU, then retains per-fit CPU/load bookends.
This records explicit modes; it does not choose an auto crossover.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

from bool_storage import wait_until_quiet
from path import lambda_grid, make_data


def bookend():
    def counters():
        values = [
            int(v) for v in Path("/proc/stat").read_text().splitlines()[0].split()[1:9]
        ]
        return sum(values), values[3] + values[4]

    total0, idle0 = counters()
    time.sleep(1)
    total1, idle1 = counters()
    busy = 1 - (idle1 - idle0) / (total1 - total0)
    return dict(cpu_busy=busy, load=os.getloadavg())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["smoke-500-100", "tall-5k-1k", "tall-50k-2k"],
        default="tall-5k-1k",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--solver", choices=["residuals", "gram", "sklearn-cd"], action="append"
    )
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--gate-seconds", type=int, default=180)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    import numpy as np
    from sklearn.linear_model import enet_path
    from threadpoolctl import threadpool_info

    import pybiglasso as bl
    from pybiglasso import _rust

    if any(pool["num_threads"] != 1 for pool in threadpool_info()):
        raise RuntimeError("Set BLAS/OMP thread limits to one before starting Python.")
    # No warning (including a nonconverged fit) is a valid timing sample.
    warnings.simplefilter("error")
    X, Xs, y = make_data(args.dataset, family="gaussian")
    grid = lambda_grid(X, y, 0.5, family="gaussian")
    Xs = np.asfortranarray(Xs - Xs.mean(axis=0))
    yc = y - y.mean()
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True):
        raise RuntimeError(
            "Commit the benchmark sources and run from a clean checkout."
        )
    cpu = next(
        line.split(":", 1)[1].strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines()
        if line.startswith("model name")
    )
    print(
        json.dumps(
            dict(
                dataset=args.dataset,
                revision=revision,
                shape=list(X.shape),
                alpha=0.5,
                lambdas=len(grid),
                seed=0,
                cpu=cpu,
                core=args.cpu,
                dtype=str(X.dtype),
                threads=1,
                extension_sha256=hashlib.sha256(
                    Path(_rust.__file__).read_bytes()
                ).hexdigest(),
                versions={
                    package: importlib.metadata.version(package)
                    for package in ["pybiglasso", "numpy", "scikit-learn"]
                },
                sources={
                    str(path): hashlib.sha256((root / path).read_bytes()).hexdigest()
                    for path in [
                        Path("rust/src/gaussian.rs"),
                        Path("rust/src/solver.rs"),
                        Path("src/pybiglasso/binomial.py"),
                        Path("benchmarks/gram.py"),
                    ]
                },
            )
        ),
        flush=True,
    )
    # Small unreported warmups initialize dispatch and thread-pool machinery.
    small = np.asfortranarray(X[:100, :10])
    for mode in ["residuals", "gram"]:
        bl.biglasso(small, y[:100], lambdas=grid[:2], update_from=mode, n_threads=1)
    enet_path(Xs[:100, :10], yc[:100], alphas=grid[:2], l1_ratio=0.5)
    waited = wait_until_quiet(0.10, args.gate_seconds)
    print(
        json.dumps(
            dict(
                gate_wait_seconds=waited, gate_seconds=args.gate_seconds, gate_busy=0.10
            )
        ),
        flush=True,
    )
    digests = {}
    solvers = args.solver or ["residuals", "gram", "sklearn-cd"]
    for repeat in range(args.repeats):
        turn = repeat % len(solvers)
        order = solvers[turn:] + solvers[:turn]
        for solver in order:
            before = bookend()
            if before["cpu_busy"] > 0.15:
                raise RuntimeError(f"No quiet benchmark window: {before}")
            started = datetime.now(UTC).isoformat()
            start = time.perf_counter()
            if solver == "sklearn-cd":
                result = enet_path(Xs, yc, alphas=grid, l1_ratio=0.5)
            else:
                result = bl.biglasso(
                    X, y, lambdas=grid, alpha=0.5, update_from=solver, n_threads=1
                )
            seconds = time.perf_counter() - start
            digest = hashlib.sha256()
            arrays = (
                result
                if solver == "sklearn-cd"
                else [
                    result.coef.toarray(),
                    result.intercept,
                    result.deviance,
                    result.iterations,
                ]
            )
            for array in arrays:
                digest.update(np.asarray(array).tobytes())
            fingerprint = digest.hexdigest()
            if fingerprint != digests.setdefault(solver, fingerprint):
                raise RuntimeError("Repeated fit changed its result.")
            after = bookend()
            print(
                json.dumps(
                    dict(
                        repeat=repeat,
                        solver=solver,
                        seconds=seconds,
                        result_sha256=fingerprint,
                        started=started,
                        before=before,
                        after=after,
                    )
                ),
                flush=True,
            )
            if after["cpu_busy"] > 0.15:
                raise RuntimeError(
                    "No quiet post-fit bookend; retain this invalid run separately."
                )
            del result


if __name__ == "__main__":
    main()
