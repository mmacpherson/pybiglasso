"""Compare fixed Rust fits on Linux x86, Linux ARM, macOS ARM and Windows x86.

CI records each platform independently, then uses that run's x86 snapshot as
the reference. This checks equality across machines, not solver correctness:
there are no solver-generated golden values to bless or regenerate.

    uv run python scripts/check_arch_bits.py record x86.json
    python scripts/check_arch_bits.py compare x86.json linux-arm.json mac-arm.json win.json

`match` holds any snapshots, from any machine, to the reference: CI's
published wheels, each recorded from the installed wheel, against the test
build's Linux x86 snapshot.

    python scripts/check_arch_bits.py match x86.json wheel-*.json
"""

import argparse
import hashlib
import json
import platform
import tempfile
import warnings
from pathlib import Path

PROBLEMS = {"tall": (2057, 17), "wide": (97, 129), "parallel-irls": (16_391, 9)}
ALPHAS = (1.0, 0.5)
FAMILIES = ("binomial", "gaussian")
DTYPES = ("float64", "int8", "arrow-mixed")
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
    import pyarrow as pa

    import pybiglasso as bl
    from pybiglasso.arrow import _open_mapped

    cases = {}
    for name, (nrow, ncol) in PROBLEMS.items():
        # Exact dyadic inputs avoid platform libm/BLAS in data generation.
        # The tall case crosses a 2048-row chunk and has a non-8-row tail.
        # parallel-irls also crosses the row-parallel IRLS threshold, so the
        # cross-OS check covers portable math on worker threads.
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, nrow, dtype=np.int64).astype(np.float64)
        raw = rng.integers(-8, 9, (nrow, ncol), dtype=np.int64).astype(np.float64)
        raw[:, 0] += 4 * (2 * y - 1)
        designs = {
            "float64": np.asfortranarray(raw / 8),
            # All values are integers in [-12, 12], exactly representable in
            # int8. Exercise Contig<i8>::block, not just the f64 control.
            "int8": np.asfortranarray(raw.astype(np.int8)),
        }
        # All six supported physical types, including a bitmap crossing the
        # same logical chunk/tail boundaries. Compare values, not IPC bytes.
        columns = []
        for j in range(ncol):
            values = raw[:, j]
            column = (
                values > 0,
                values.astype(np.int8),
                (values + 16).astype(np.uint8),
                (values * 256).astype(np.int16),
                (values / 8).astype(np.float32),
                values / 8,
            )[j % 6]
            columns.append(column)
        reference = np.asfortranarray(np.column_stack(columns))
        temporary = tempfile.TemporaryDirectory()
        arrow_path = Path(temporary.name) / "mixed.arrow"
        batch = pa.record_batch(
            [pa.array(a) for a in columns], names=[f"x{j}" for j in range(ncol)]
        )
        with pa.OSFile(str(arrow_path), "wb") as sink:
            with pa.ipc.new_file(sink, batch.schema) as writer:
                cuts = sorted({0, min(7, nrow), min(2047, nrow), min(2048, nrow), nrow})
                for start, stop in zip(cuts[:-1], cuts[1:], strict=True):
                    writer.write_batch(batch.slice(start, stop - start))
        designs["arrow-mixed"] = _open_mapped(arrow_path)
        for dtype in DTYPES:
            X = designs[dtype]
            values = reference if dtype == "arrow-mixed" else X
            for family in FAMILIES:
                # All operations are exact dyadic arithmetic, including Gaussian y.
                response = y if family == "binomial" else 0.5 + 2 * y + values[:, 1] / 4
                for alpha in ALPHAS:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error")
                        fit = bl.biglasso(
                            X,
                            response,
                            family=family,
                            alpha=alpha,
                            nlambda=12,
                            n_threads=2,
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
                    cases[f"{name}/dtype={dtype}/family={family}/alpha={alpha}"] = dict(
                        inputs={"X": array_bits(values), "y": array_bits(response)},
                        fit={
                            key: array_bits(value, include_data=True)
                            for key, value in arrays.items()
                        },
                    )
        # Windows cannot remove a mapped file; release all views first.
        designs.pop("arrow-mixed")
        del X
        temporary.cleanup()
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


EXPECTED = {
    f"{name}/dtype={dtype}/family={family}/alpha={alpha}"
    for name in PROBLEMS
    for dtype in DTYPES
    for family in FAMILIES
    for alpha in ALPHAS
}


def load_reference(reference):
    baseline = json.loads(reference.read_text())
    assert baseline["machine"] == "x86_64" and baseline["system"] == "Linux"
    assert set(baseline["cases"]) == EXPECTED
    return baseline


def mismatches(baseline, reference, path, other):
    """The fields of `other` whose bits differ from the reference's."""
    assert set(other["cases"]) == EXPECTED, path
    found = []
    for case in sorted(EXPECTED):
        a, b = baseline["cases"][case], other["cases"][case]
        assert a["inputs"] == b["inputs"], f"{path}: {case} inputs differ"
        assert a["fit"].keys() == b["fit"].keys(), f"{path}: {case} fields differ"
        for field in a["fit"]:
            if a["fit"][field] != b["fit"][field]:
                found.append(f"{path}: {case} {field} bits differ from {reference}")
    if found:
        print(f"{path}: {len(found)} field comparisons differ from {reference}")
    else:
        print(f"{path}: all {len(EXPECTED)} fits bit-identical to {reference}")
    return found


def compare(reference, candidates, systems):
    baseline = load_reference(reference)
    seen = set()
    differences = []
    for path in candidates:
        other = json.loads(path.read_text())
        assert other["system"] in CANDIDATE_MACHINES, path
        assert other["machine"] in CANDIDATE_MACHINES[other["system"]], path
        seen.add(other["system"])
        differences += mismatches(baseline, reference, path, other)
    # every expected platform, once each
    assert len(candidates) == len(systems) and seen == systems, (seen, systems)
    # Report every platform, even when an earlier candidate already differs.
    assert not differences, "\n".join(differences)


def match(reference, candidates):
    baseline = load_reference(reference)
    # every named snapshot must be there, and complete; a missing or empty one
    # is a failed check, not one fewer to compare
    absent = [str(path) for path in candidates if not path.is_file()]
    assert not absent, f"missing snapshots: {absent}"
    empty = [str(path) for path in candidates if not path.read_text().strip()]
    assert not empty, f"empty snapshots: {empty}"
    differences = []
    for path in candidates:
        other = json.loads(path.read_text())
        assert other.get("cases"), f"{path}: no cases"
        differences += mismatches(baseline, reference, path, other)
    assert not differences, "\n".join(differences)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    writer = commands.add_parser("record")
    writer.add_argument("output", type=Path)
    matcher = commands.add_parser("match")
    matcher.add_argument("reference", type=Path)
    matcher.add_argument("candidates", type=Path, nargs="+")
    checker = commands.add_parser("compare")
    checker.add_argument("reference", type=Path)
    checker.add_argument("candidates", type=Path, nargs="+")
    checker.add_argument(
        "--without-macos",
        action="store_true",
        help="expect no Darwin snapshot (pull requests skip macOS)",
    )
    args = parser.parse_args()
    if args.command == "record":
        record(args.output)
    elif args.command == "match":
        match(args.reference, args.candidates)
    else:
        systems = set(CANDIDATE_MACHINES)
        if args.without_macos:
            systems.discard("Darwin")
        compare(args.reference, args.candidates, systems)


if __name__ == "__main__":
    main()
