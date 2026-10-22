#!/usr/bin/env python3
"""Plot the in-memory benchmark results as time relative to pybiglasso.

Each solver is shown at its best thread count; runs stopped at the ratio cap
are drawn at the cap. Later files override earlier ones for the same
(dataset, alpha, solver, threads), so reruns on a quiet machine win.

    uv run --group bench benchmarks/plot.py results/path.jsonl ... -o out.png
"""

import argparse
import json
import pathlib

import pandas as pd
import seaborn.objects as so

LABELS = {
    "r-biglasso-ssr": "R biglasso 1.7.2",
    "skglm": "skglm",
    "glum": "glum",
    "sklearn-liblinear": "sklearn liblinear",
    "sklearn-saga": "sklearn saga",
}


def load(paths):
    rows = {}
    for path in paths:
        for line in pathlib.Path(path).read_text().splitlines():
            r = json.loads(line)
            rows[r["dataset"], r["alpha"], r["solver"], r["threads"]] = r
    return pd.DataFrame(rows.values())


def relative(df):
    best = df.groupby(["dataset", "alpha", "solver"], as_index=False).agg(
        seconds=("seconds", "min"),
        timed_out=("timed_out", "all"),
        ratio_cap=("ratio_cap", "first"),
    )
    base = best[best.solver == "pybiglasso"].set_index(["dataset", "alpha"]).seconds
    best = best[best.solver.isin(LABELS)].copy()
    best["ratio"] = (
        best.seconds
        / base.loc[list(zip(best.dataset, best.alpha, strict=True))].to_numpy()
    )
    best.loc[best.timed_out, "ratio"] = best.ratio_cap
    best["solver"] = best.solver.map(LABELS)
    best["penalty"] = best.alpha.map({1.0: "lasso", 0.5: "elastic net (α=0.5)"})
    best["result"] = best.timed_out.map({False: "finished", True: "stopped at cap"})
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+")
    parser.add_argument("-o", "--out", required=True)
    args = parser.parse_args()
    df = relative(load(args.results))
    order = list(LABELS.values())
    plot = (
        so.Plot(df, x="ratio", y="solver", color="penalty", marker="result")
        .facet(
            col="dataset",
            order=["tall-5k-1k", "tall-50k-2k", "wide-2k-20k", "wide-5k-50k"],
            wrap=2,
        )
        .add(so.Dot(pointsize=7), so.Dodge(empty="fill"))
        .scale(
            x=so.Continuous(trans="log").tick(at=[1, 2, 5, 10]).label(like="{x:g}×"),
            y=so.Nominal(order=order),
            marker=so.Nominal(["o", ">"], order=["finished", "stopped at cap"]),
        )
        .limit(x=(0.8, 13))
        .label(x="× pybiglasso time", y="", color="", marker="")
        .layout(size=(9, 6), engine="constrained")
    )
    plot.save(args.out, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
