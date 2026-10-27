"""Paired whole-path timings for compact column conversion.

Use two actual Rust extensions with one Python wrapper and shared inputs.
AB/BA ordering balances the two versions; hashes compare transformations of
the same fit and are never expected solver values. An optional numba control
uses every three-way order with separate repeat/thread identity checks,
because numba and Rust reduce in different orders.
"""

import argparse
import hashlib
import importlib.util
import itertools
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

import pybiglasso as bl
from pybiglasso import _rust

PROBLEMS = {
    "tall-5k-1k": (5000, 1000),
    "wide-2k-20k": (2000, 20000),
    "wide-5k-50k": (5000, 50000),
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-extension", type=Path, required=True)
    parser.add_argument(
        "--family", choices=["binomial", "gaussian"], default="binomial"
    )
    parser.add_argument(
        "--problem", nargs="+", choices=PROBLEMS, default=["wide-2k-20k", "wide-5k-50k"]
    )
    parser.add_argument(
        "--dtype", nargs="+", choices=["int8", "float32", "float64"], default=["int8"]
    )
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 6])
    parser.add_argument("--alpha", nargs="+", type=float, default=[1.0, 0.5])
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument(
        "--with-numba",
        action="store_true",
        help="include a numba control; use a multiple of six repetitions",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.repeats < 2 or args.repeats % 2 or min(args.threads) < 1:
        parser.error("Use an even number of repetitions >= 2 and positive threads.")
    if not all(0 < alpha <= 1 for alpha in args.alpha):
        parser.error("alpha must be in (0, 1].")
    if args.with_numba and (args.family != "binomial" or args.repeats % 6):
        parser.error(
            "The numba control is binomial-only and requires repetitions divisible by six."
        )
    spec = importlib.util.spec_from_file_location(
        "baseline._rust", args.baseline_extension
    )
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    variants = {"baseline": baseline, "candidate": _rust}
    fingerprints = {
        name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for name, module in variants.items()
    }
    if len(set(fingerprints.values())) != 2:
        parser.error("The baseline and candidate extension files must differ.")
    numba_version = None
    if args.with_numba:
        import numba

        if max(args.threads) > numba.config.NUMBA_NUM_THREADS:
            parser.error("Requested threads exceed numba's available pool.")
        numba_version = numba.__version__
        variants["numba"] = _rust
    # Every order for a three-way control balances position/predecessor effects.
    # With two variants this is the familiar AB/BA sequence.
    orders = list(itertools.permutations(variants.items()))
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
        extension_sha256=fingerprints,
        seed=1,
        family=args.family,
        numba=numba_version,
    )
    try:
        for problem in args.problem:
            nrow, ncol = PROBLEMS[problem]
            # Keep the original response/effects: passing dtype=int8 into the
            # generator would also truncate its effect sizes and change y.
            data = bl.common.make_biglasso_dataset(nrow=nrow, ncol=ncol, seed=1)
            response = data["y"]
            if args.family == "gaussian":
                # A continuous dyadic response for a shared-kernel control.
                response = 0.5 + 2 * response + data["X"][:, 1] / 4
            for dtype in args.dtype:
                X = np.asarray(data["X"], dtype=dtype, order="F")
                X.flags.writeable = False
                for alpha in args.alpha:
                    hashes = {name: set() for name in variants}
                    trajectories = {}
                    for threads in args.threads:
                        options = dict(
                            family=args.family,
                            backend="rust",
                            alpha=alpha,
                            n_threads=threads,
                        )
                        for name, module in variants.items():
                            bl._rust = module
                            options["backend"] = "numba" if name == "numba" else "rust"
                            hashes[name].add(
                                path_hash(bl.biglasso(X, response, **options))
                            )
                        times = {name: [] for name in variants}
                        for repeat in range(args.repeats):
                            order = orders[repeat % len(orders)]
                            for name, module in order:
                                bl._rust = module
                                options["backend"] = (
                                    "numba" if name == "numba" else "rust"
                                )
                                start = time.perf_counter()
                                fit = bl.biglasso(X, response, **options)
                                times[name].append(time.perf_counter() - start)
                                hashes[name].add(path_hash(fit))
                                trajectories[name] = dict(
                                    nlambda=len(fit),
                                    iterations=int(fit.iterations.sum()),
                                )
                        assert len(hashes["baseline"] | hashes["candidate"]) == 1, (
                            "A Rust fit changed bits across versions/threads"
                        )
                        assert all(len(h) == 1 for h in hashes.values()), (
                            "A backend changed bits across repetitions/threads"
                        )
                        medians = {
                            name: statistics.median(values)
                            for name, values in times.items()
                        }
                        result = dict(
                            **metadata,
                            problem=problem,
                            nrow=nrow,
                            ncol=ncol,
                            dtype=dtype,
                            alpha=alpha,
                            threads=threads,
                            times=times,
                            median_seconds=medians,
                            speedup=medians["baseline"] / medians["candidate"],
                            path_hash=next(iter(hashes["candidate"])),
                            variant_hashes={
                                name: next(iter(h)) for name, h in hashes.items()
                            },
                            trajectories=trajectories,
                            **trajectories["candidate"],
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
