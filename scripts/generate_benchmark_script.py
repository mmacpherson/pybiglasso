#!/usr/bin/env python3

import itertools

import click
import sh

HEADER = """\
#!/usr/bin/env bash
set -euox pipefail

RUN_BENCHMARK="venv/bin/python scripts/benchmark_vs_r_biglasso.py"

RESULTS_FILE=${1:-data/benchmarks.csv}
rm -f ${RESULTS_FILE}
"""


@click.command()
@click.option("--results-file", default="data/benchmarks.csv", type=click.Path())
@click.option("--script-file", default="scripts/run_benchmarks.sh", type=click.Path())
def main(results_file, script_file):

    grid = dict(
        nreps=1,
        nrow=(100_000, 200_000, 500_000, 1_000_000),
        ncol=(100, 200, 500),
        nlambda=(10, 20, 50),
        alpha=0.8,
        dtype="float64",
        # dtype=("float32", "float64"),
    )

    out_lines = [HEADER]

    params = (
        dict(zip(grid.keys(), e))
        for e in itertools.product(
            *(v if isinstance(v, tuple) else (v,) for v in grid.values())
        )
    )
    for param_set in params:
        cli_parts = (
            "${{RUN_BENCHMARK}}",
            "--nrow {nrow}",
            "--ncol {ncol}",
            "--nlambda {nlambda}",
            "--nreps {nreps}",
            "--alpha {alpha}",
            "--dtype {dtype}",
            "--mmap",
            ">> ${{RESULTS_FILE}}",
        )
        cli = " ".join(e.format(**param_set) for e in cli_parts)
        out_lines.append(cli)

    sh.rm("-f", script_file)
    with open(script_file, "wt") as fp:
        print("\n".join(out_lines), file=fp)
    sh.chmod("u+x", script_file)


if __name__ == "__main__":
    main()
