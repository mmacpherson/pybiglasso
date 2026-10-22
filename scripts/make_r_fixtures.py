#!/usr/bin/env python3
"""Fit the fast comparison grid in R and store the results as test fixtures.

    make fixtures     # builds the pinned R image, then runs this

Fixtures come from R biglasso FIXTURES_BIGLASSO, pinned through the R
release whose package snapshot installs it (FIXTURES_R in the Makefile).
Prints any disagreements with the current Python fits.
"""

import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "tests"))

import rcompare  # noqa: E402

FIXTURES_BIGLASSO = "1.7.2"


def main():
    cases = rcompare.fast_grid()
    with tempfile.TemporaryDirectory() as tmp:
        dirs, prepared = [], []
        for case in cases:
            d = pathlib.Path(tmp) / case.name
            prepared.append(rcompare.prepare_case(case, d))
            dirs.append(d)
        r_outputs = rcompare.run_r(dirs)

    errors = [
        (c.name, r["error"])
        for c, r in zip(cases, r_outputs, strict=True)
        if "error" in r
    ]
    for name, err in errors:
        print(f"R ERROR {name}: {err}")
    if errors:
        sys.exit(f"{len(errors)} cases failed in R; no fixtures written")

    versions = {r["biglasso_version"] for r in r_outputs}
    if versions != {FIXTURES_BIGLASSO}:
        sys.exit(f"Fixtures must come from biglasso {FIXTURES_BIGLASSO}, got {versions}")

    failures = 0
    for case, (py, _, _), r in zip(cases, prepared, r_outputs, strict=True):
        problems = rcompare.compare(py, r)
        if problems:
            failures += 1
            print(f"DISAGREE {case.name}")
            for p in problems:
                print(f"    {p}")

    rcompare.save_fixtures(
        {
            c.name: (digest, lambdas, r)
            for c, (_, digest, lambdas), r in zip(
                cases, prepared, r_outputs, strict=True
            )
        }
    )
    print(
        f"wrote {len(cases)} cases to {rcompare.FIXTURE_PATH}; "
        f"{len(cases) - failures} agree, {failures} disagree"
    )


if __name__ == "__main__":
    main()
