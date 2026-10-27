#!/usr/bin/env python3
"""Render the README Gaussian timing table from native-glmnet JSONL results."""

import argparse
import json
from pathlib import Path
from statistics import median


def render(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if row.get("family") != "gaussian":
            continue
        if row["solver"] == "r-glmnet" and row.get("glmnet_family") != "gaussian":
            raise ValueError("The timing table requires native Gaussian glmnet")
        if row["timed_out"] or len(row["times"]) != 3:
            raise ValueError("The timing table requires three completed repetitions")
        if row["seconds"] != median(row["times"]):
            raise ValueError("Stored time differs from the median")
        rows[row["dataset"], row["alpha"], row["solver"], row["threads"]] = row
    lines = [
        "| Dataset | Penalty | pybiglasso, 1 thread | pybiglasso, 6 threads | R biglasso, 1 thread | R biglasso, 6 threads | Native glmnet, 1 thread |",
        "|---|---|---|---|---|---|---|",
    ]
    columns = [
        ("pybiglasso", 1),
        ("pybiglasso", 6),
        ("r-biglasso-ssr", 1),
        ("r-biglasso-ssr", 6),
        ("r-glmnet", 1),
    ]
    for dataset, label in [
        ("tall-5k-1k", "5,000 x 1,000"),
        ("wide-2k-20k", "2,000 x 20,000"),
    ]:
        for alpha, penalty in [(1, "lasso"), (0.5, "elastic net (α=0.5)")]:
            times = [
                f"{rows[dataset, alpha, solver, threads]['seconds']:.2f}"
                for solver, threads in columns
            ]
            lines.append("| " + " | ".join([label, penalty, *times]) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    print(render(parser.parse_args().results))
