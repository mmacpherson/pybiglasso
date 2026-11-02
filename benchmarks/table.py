#!/usr/bin/env python3
"""Print the README's table of absolute times under each benchmark figure.

A row per solver, a column per dataset; each cell is lasso / elastic net in
seconds, every solver at its best thread count. A run stopped at its time cap
is shown as at least the cap. Reads the same files as plot.py.

    uv run benchmarks/table.py benchmarks/results/path.jsonl
"""

import argparse
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from plot import DATASETS, LABELS, PENALTIES, load  # noqa: E402

SHORT = {
    "tall-5k-1k": "5,000 × 1,000",
    "tall-50k-2k": "50,000 × 2,000",
    "wide-2k-20k": "2,000 × 20,000",
    "wide-5k-50k": "5,000 × 50,000",
}


def best(rows):
    out = {}
    for r in rows:
        key = r["dataset"], r["alpha"], r["solver"]
        seconds = r["cap"] if r["timed_out"] else r["seconds"]
        if key not in out or seconds < out[key][0]:
            out[key] = seconds, r["timed_out"]
    return out


def cell(entry):
    if entry is None:
        return "—"
    seconds, capped = entry
    digits = 2 if seconds < 10 else 1 if seconds < 100 else 0
    text = f"{seconds:.{digits}f}"
    return f"≥ {text}" if capped else text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+")
    times = best(load(parser.parse_args().results))
    datasets = [d for d in DATASETS if any(k[0] == d for k in times)]
    solvers = ["pybiglasso"] + [s for s in LABELS if any(k[2] == s for k in times)]
    names = {"pybiglasso": "pybiglasso", **LABELS}
    print("| Solver | " + " | ".join(SHORT[d] for d in datasets) + " |")
    print("|---|" + "---|" * len(datasets))
    for s in solvers:
        cells = [
            " / ".join(cell(times.get((d, a, s))) for a in PENALTIES) for d in datasets
        ]
        print(f"| {names[s]} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
