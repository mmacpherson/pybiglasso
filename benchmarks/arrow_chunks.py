"""Alternate two extensions on the same values and Arrow chunk partitions.

The fixture and fit match the Arrow performance probe: n=262144, p=300,
Gaussian, 20 lambdas, lambda_min=0.1, one thread. AB/BA order balances the
builds within one process. Hashes check invariance, never expected fit values.
"""

import argparse
import hashlib
import importlib.util
import json
import platform
import statistics
from pathlib import Path

import numpy as np
import pyarrow as pa
from int8 import checked_fit, cpu_model, path_hash

import pybiglasso as bl
from pybiglasso import _rust


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-extension", required=True, type=Path)
    parser.add_argument(
        "--family", choices=["gaussian", "binomial"], default="gaussian"
    )
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--nrow", type=int, default=262144)
    parser.add_argument("--ncol", type=int, default=300)
    parser.add_argument(
        "--kind",
        nargs="+",
        choices=["u8", "f64", "bool"],
        default=["u8", "f64", "bool"],
    )
    parser.add_argument("--chunks", nargs="+", type=int, default=[1, 4, 64])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.repeats < 2 or args.repeats % 2:
        parser.error("Use an even number of repetitions >= 2.")
    if not all(0 < count <= args.nrow for count in args.chunks):
        parser.error("Chunk counts must be positive and no greater than the row count.")
    if args.nrow < 64 or args.ncol < 5:
        parser.error("Use at least 64 rows and 5 columns.")
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
        parser.error("Baseline and candidate must be different extensions.")
    metadata = dict(
        cpu=cpu_model(),
        python=platform.python_version(),
        numpy=np.__version__,
        pyarrow=pa.__version__,
        extension_sha256=fingerprints,
        source_sha256={
            name: hashlib.sha256(
                (Path(__file__).resolve().parents[1] / "rust" / name).read_bytes()
            ).hexdigest()
            for name in (
                "src/solver.rs",
                "src/gaussian.rs",
                "src/python.rs",
                "Cargo.lock",
            )
        },
        nrow=args.nrow,
        ncol=args.ncol,
        family=args.family,
        nlambda=20,
        lambda_min=0.1,
        threads=1,
        seed=0,
    )
    rng = np.random.default_rng(0)
    values = rng.random((args.nrow, args.ncol)) < 0.3
    response = values[:, :5] @ np.ones(5) + rng.normal(size=args.nrow)
    if args.family == "binomial":
        response = (response > np.median(response)).astype(float)
    all_hashes = set()

    def measure(X, kind, chunks):
        times = {name: [] for name in variants}
        for module in variants.values():
            bl._rust = module
            fit, _ = checked_fit(
                X,
                response,
                family=args.family,
                nlambda=20,
                lambda_min=0.1,
                n_threads=1,
            )
            all_hashes.add(path_hash(fit))
        order = list(variants.items())
        for repeat in range(args.repeats):
            for name, module in order if repeat % 2 == 0 else reversed(order):
                bl._rust = module
                fit, seconds = checked_fit(
                    X,
                    response,
                    family=args.family,
                    nlambda=20,
                    lambda_min=0.1,
                    n_threads=1,
                )
                times[name].append(seconds)
                all_hashes.add(path_hash(fit))
        assert len(all_hashes) == 1, (
            "Fit bits changed across builds, types or partitions"
        )
        result = dict(
            **metadata,
            kind=kind,
            chunks=chunks,
            times=times,
            median_seconds={name: statistics.median(t) for name, t in times.items()},
            path_hash=next(iter(all_hashes)),
        )
        line = json.dumps(result)
        print(line, flush=True)
        if args.out:
            with args.out.open("a") as stream:
                stream.write(line + "\n")

    try:
        ndarray = np.asfortranarray(values, dtype=np.uint8)
        measure(ndarray, "ndarray-u8", 1)
        for kind, dtype in (("u8", np.uint8), ("f64", np.float64), ("bool", np.bool_)):
            if kind not in args.kind:
                continue
            arrays = [pa.array(values[:, j].astype(dtype)) for j in range(args.ncol)]
            for count in args.chunks:
                bounds = np.linspace(0, args.nrow, count + 1, dtype=np.int64)
                columns = [
                    pa.chunked_array(
                        [
                            a.slice(int(start), int(end - start))
                            for start, end in zip(bounds[:-1], bounds[1:], strict=True)
                        ]
                    )
                    for a in arrays
                ]
                measure(
                    pa.Table.from_arrays(
                        columns, names=[f"c{j}" for j in range(args.ncol)]
                    ),
                    kind,
                    count,
                )
    finally:
        bl._rust = _rust


if __name__ == "__main__":
    main()
