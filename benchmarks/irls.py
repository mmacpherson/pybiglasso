"""Small tall-binomial fits for before/after IRLS measurements.

Run this same file with each revision's Python environment. Times include
standardization and the whole path; hashes compare transformations of the
same fit, and are never stored as expected solver values.
"""

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

import pybiglasso as bl
from pybiglasso import _rust


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 6, 12])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--baseline-extension",
        type=Path,
        help="alternate this baseline Rust extension with the running revision",
    )
    args = parser.parse_args()
    variants = {"candidate": _rust}
    if args.baseline_extension:
        spec = importlib.util.spec_from_file_location(
            "baseline._rust", args.baseline_extension
        )
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
        variants = {"baseline": baseline, **variants}
    for nrow, ncol in [(10_000, 100), (50_000, 100), (500_000, 40)]:
        rng = np.random.default_rng(2718)
        X = np.asfortranarray(rng.standard_normal((nrow, ncol)))
        eta = X[:, 0] * 0.8 - X[:, 1] * 0.6 + 0.2
        y = (rng.random(nrow) < 1 / (1 + np.exp(-eta))).astype(float)
        hashes = set()
        for threads in args.threads:
            times = {name: [] for name in variants}
            for rep in range(args.repeats):
                order = list(variants.items())
                if rep % 2:
                    order.reverse()
                for name, extension in order:
                    # Both compiled versions share one Python wrapper and the
                    # exact same input arrays. AB/BA order limits clock drift.
                    bl._rust = extension
                    start = time.perf_counter()
                    path = bl.biglasso(
                        X,
                        y,
                        family="binomial",
                        backend="rust",
                        alpha=0.5,
                        nlambda=30,
                        lambda_min=0.05,
                        n_threads=threads,
                    )
                    times[name].append(time.perf_counter() - start)
                    digest = hashlib.sha256()
                    coef = path.coef.tocsc()
                    for a in [
                        path.lambdas,
                        path.intercept,
                        coef.data,
                        coef.indices,
                        coef.indptr,
                        path.iterations,
                        path.n_rejected,
                        path.deviance,
                    ]:
                        a = np.asarray(a)
                        digest.update(str((a.dtype.str, a.shape)).encode())
                        digest.update(a.tobytes())
                    hashes.add(digest.hexdigest())
            bl._rust = _rust
            assert len(hashes) == 1, (
                "A fit changed bits across versions, repetitions or threads"
            )
            print(
                json.dumps(
                    dict(
                        nrow=nrow,
                        ncol=ncol,
                        threads=threads,
                        times=times,
                        fit_hash=next(iter(hashes)),
                    )
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
