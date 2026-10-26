"""Compare fixed Rust fits on Linux x86, Linux ARM, macOS ARM and Windows x86.

CI records each platform independently, then uses that run's x86 snapshot as
the reference. This checks equality across machines, not solver correctness:
there are no solver-generated golden values to bless or regenerate.

    uv run python scripts/check_arch_bits.py record x86.json
    python scripts/check_arch_bits.py compare x86.json linux-arm.json mac-arm.json win.json
"""

import argparse
import hashlib
import json
import platform
import warnings
from pathlib import Path

PROBLEMS = {"tall": (2057, 17), "wide": (97, 129)}
ALPHAS = (1.0, 0.5)
CANDIDATE_MACHINES = {
    "Linux": ("aarch64", "arm64"),
    "Darwin": ("aarch64", "arm64"),
    "Windows": ("amd64", "x86_64"),
}


def array_bits(array, *, include_data=False):
    # Shape and dtype matter too: equal raw bytes alone can mask a layout bug.
    raw = array.tobytes(order="C")
    result = dict(
        shape=list(array.shape),
        dtype=array.dtype.str,
        sha256=hashlib.sha256(raw).hexdigest(),
    )
    if include_data:
        # Small output arrays let a failed comparison identify the first
        # lambda/ulp, rather than requiring another native build to diagnose it.
        result["data"] = raw.hex()
    return result


def record(output):
    import numpy as np

    import pybiglasso as bl

    cases = {}
    for name, (nrow, ncol) in PROBLEMS.items():
        # Exact dyadic inputs avoid platform libm/BLAS in data generation.
        # The tall case crosses a 2048-row chunk and has a non-8-row tail.
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, nrow, dtype=np.int64).astype(np.float64)
        X = rng.integers(-8, 9, (nrow, ncol), dtype=np.int64).astype(np.float64)
        X[:, 0] += 4 * (2 * y - 1)
        X = np.asfortranarray(X / 8)
        for alpha in ALPHAS:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                fit = bl.biglasso(
                    X, y, alpha=alpha, nlambda=12, backend="rust", n_threads=2
                )
            assert len(fit) == 12 and fit.coef.nnz > 0, name
            coef = fit.coef.tocsc()
            arrays = dict(
                lambdas=fit.lambdas,
                intercept=fit.intercept,
                coef_data=coef.data,
                coef_indices=coef.indices,
                coef_indptr=coef.indptr,
                iterations=fit.iterations,
                n_rejected=fit.n_rejected,
                center=fit.center,
                scale=fit.scale,
                deviance=fit.deviance,
            )
            cases[f"{name}/alpha={alpha}"] = dict(
                inputs={"X": array_bits(X), "y": array_bits(y)},
                fit={
                    key: array_bits(value, include_data=True)
                    for key, value in arrays.items()
                },
            )
    snapshot = dict(
        machine=platform.machine().lower(),
        system=platform.system(),
        numpy=np.__version__,
        cases=cases,
    )
    output.write_text(json.dumps(snapshot, indent=2) + "\n")
    print(
        f"Recorded {len(cases)} Rust fits: {snapshot['system']} {snapshot['machine']}"
    )


def compare(reference, candidates):
    baseline = json.loads(reference.read_text())
    assert baseline["machine"] == "x86_64" and baseline["system"] == "Linux"
    expected = {f"{name}/alpha={alpha}" for name in PROBLEMS for alpha in ALPHAS}
    assert set(baseline["cases"]) == expected
    systems = set()
    differences = []
    for path in candidates:
        other = json.loads(path.read_text())
        assert other["system"] in CANDIDATE_MACHINES, path
        assert other["machine"] in CANDIDATE_MACHINES[other["system"]], path
        systems.add(other["system"])
        assert set(other["cases"]) == expected, path
        mismatches = []
        for case in sorted(expected):
            a, b = baseline["cases"][case], other["cases"][case]
            assert a["inputs"] == b["inputs"], f"{path}: {case} inputs differ"
            assert a["fit"].keys() == b["fit"].keys(), f"{path}: {case} fields differ"
            for field in a["fit"]:
                if a["fit"][field] != b["fit"][field]:
                    mismatches.append(
                        f"{path}: {case} {field} bits differ from {reference}"
                    )
        if mismatches:
            differences.extend(mismatches)
            print(
                f"{path}: {len(mismatches)} field comparisons differ from {reference}"
            )
        else:
            print(f"{path}: all {len(expected)} fits bit-identical to {reference}")
    assert len(candidates) == 3 and systems == CANDIDATE_MACHINES.keys()
    # Report every platform, even when an earlier candidate already differs.
    assert not differences, "\n".join(differences)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    writer = commands.add_parser("record")
    writer.add_argument("output", type=Path)
    checker = commands.add_parser("compare")
    checker.add_argument("reference", type=Path)
    checker.add_argument("candidates", type=Path, nargs="+")
    args = parser.parse_args()
    if args.command == "record":
        record(args.output)
    else:
        compare(args.reference, args.candidates)


if __name__ == "__main__":
    main()
