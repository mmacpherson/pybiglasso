#!/usr/bin/env python3
"""Plot in-memory benchmark results as time relative to pybiglasso.

One figure per family, in one style: a panel per dataset, a row per solver,
lasso and elastic net side by side, on a log scale with pybiglasso at 1x.
Each solver is shown at its best thread count. A run stopped at its time cap
took at least that long, so it is drawn at the cap as a lower bound. Later
files override earlier ones for the same (dataset, alpha, solver, threads).
Writes a light and a dark PNG, for the README's <picture> element.

    uv run --group bench benchmarks/plot.py --family binomial \\
        benchmarks/results/path.jsonl -o docs/benchmarks-logistic
"""

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

LABELS = {
    "r-biglasso-ssr": "R biglasso",
    "r-glmnet": "glmnet (R)",
    "skglm": "skglm",
    "glum": "glum",
    "sklearn-cd": "scikit-learn",
    "sklearn-liblinear": "scikit-learn liblinear",
    "sklearn-saga": "scikit-learn saga",
    "statsmodels": "statsmodels",
}
DATASETS = {
    "tall-5k-1k": "5,000 rows × 1,000 features",
    "tall-50k-2k": "50,000 rows × 2,000 features",
    "wide-2k-20k": "2,000 rows × 20,000 features",
    "wide-5k-50k": "5,000 rows × 50,000 features",
}
HEADER = 0.6  # inches above the panels for the title and legend
TITLES = {"binomial": "Logistic regression", "gaussian": "Least squares"}
PENALTIES = {1.0: "lasso", 0.5: "elastic net (α = 0.5)"}
# reference palette, categorical slots 1 and 2, light and dark steps
THEMES = {
    "light": dict(
        surface="#fcfcfb", text="#0b0b0b", muted="#52514e", grid="#e4e3df",
        series=("#2a78d6", "#eb6834"),
    ),
    "dark": dict(
        surface="#1a1a19", text="#f0efec", muted="#c3c2b7", grid="#3a3a37",
        series=("#3987e5", "#d95926"),
    ),
}  # fmt: skip


def load(paths):
    rows = {}
    for path in paths:
        for line in pathlib.Path(path).read_text().splitlines():
            r = json.loads(line)
            rows[r["dataset"], r["alpha"], r["solver"], r["threads"]] = r
    return list(rows.values())


def best_runs(rows):
    """Each (dataset, alpha, solver) at its fastest thread count, as a ratio to
    pybiglasso's fastest; a capped run's ratio is its cap's, a lower bound."""
    best = {}
    for r in rows:
        key = r["dataset"], r["alpha"], r["solver"]
        seconds = r["cap"] if r["timed_out"] else r["seconds"]
        if key not in best or seconds < best[key]["seconds"]:
            best[key] = dict(r, seconds=seconds)
    out = []
    for (dataset, alpha, solver), r in best.items():
        base = best.get((dataset, alpha, "pybiglasso"))
        if solver not in LABELS or base is None:
            continue
        out.append(
            dict(
                dataset=dataset,
                alpha=alpha,
                solver=solver,
                ratio=r["seconds"] / base["seconds"],
                capped=r["timed_out"],
            )
        )
    return out


def draw(runs, family, theme, out):
    t = THEMES[theme]
    datasets = [d for d in DATASETS if any(r["dataset"] == d for r in runs)]
    solvers = [s for s in LABELS if any(r["solver"] == s for r in runs)]
    alphas = [a for a in PENALTIES if any(r["alpha"] == a for r in runs)]
    ncols = 2 if len(datasets) > 1 else 1
    nrows = -(-len(datasets) // ncols)
    # a fixed header band (title and legend), in inches, above the panels
    height = 0.42 * (len(solvers) + 1.5) * nrows + 0.5 + HEADER
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.2 * ncols, height),
        sharex=True,
        squeeze=False,
        facecolor=t["surface"],
    )
    ratios = [r["ratio"] for r in runs] + [1.0]
    lo = 2 ** np.floor(np.log2(min(ratios) / 1.15))
    hi = 2 ** np.ceil(np.log2(max(ratios) * 1.15))
    ticks = [x for x in (0.25, 0.5, 1, 2, 5, 10, 20, 50, 100) if lo <= x <= hi]
    offsets = np.linspace(-0.16, 0.16, len(alphas)) if len(alphas) > 1 else [0.0]
    for ax, dataset in zip(axes.flat, datasets, strict=False):
        ax.set_facecolor(t["surface"])
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        ax.set_xticks(ticks, [f"{x:g}×" for x in ticks])
        ax.minorticks_off()
        ax.grid(axis="x", color=t["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, colors=t["muted"])
        ax.axvline(1, color=t["text"], linewidth=1.2)
        ax.text(1, -0.75, " pybiglasso", color=t["text"], va="center", fontsize=9)
        ax.set_ylim(len(solvers) - 0.4, -1.1)
        ax.set_yticks(range(len(solvers)), [LABELS[s] for s in solvers])
        ax.tick_params(axis="y", colors=t["text"])
        ax.set_title(DATASETS[dataset], color=t["text"], fontsize=10, loc="left")
        for r in runs:
            if r["dataset"] != dataset:
                continue
            k = alphas.index(r["alpha"])
            ax.plot(
                r["ratio"],
                solvers.index(r["solver"]) + offsets[k],
                marker=">" if r["capped"] else "o",
                markersize=8.5 if r["capped"] else 7,
                color=t["series"][k],
                markeredgecolor=t["surface"],
                markeredgewidth=1.5,
                linestyle="none",
            )
    for ax in axes.flat[len(datasets) :]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(
            "time relative to pybiglasso  (← faster · slower →)", color=t["muted"]
        )
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=7,
               color=t["series"][k], label=PENALTIES[a])
        for k, a in enumerate(alphas)
    ] + [
        Line2D([], [], marker=">", linestyle="none", markersize=7,
               color=t["muted"], label="stopped at time cap (at least this slow)")
    ]  # fmt: skip
    fig.suptitle(
        f"{TITLES[family]}: time relative to pybiglasso, lower is better",
        x=0.01,
        y=1 - 0.12 / height,
        ha="left",
        va="top",
        color=t["text"],
        fontsize=12,
    )
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 1 - 0.42 / height),
        ncols=len(handles),
        frameon=False,
        labelcolor=t["text"],
        handletextpad=0.3,
        columnspacing=1.4,
    )
    fig.tight_layout(rect=(0, 0, 1, 1 - HEADER / height))
    fig.savefig(out, dpi=150, facecolor=t["surface"])
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+")
    parser.add_argument("--family", choices=list(TITLES), required=True)
    parser.add_argument("-o", "--out", required=True, help="path stem")
    args = parser.parse_args()
    runs = best_runs(load(args.results))
    for theme in THEMES:
        draw(runs, args.family, theme, f"{args.out}-{theme}.png")


if __name__ == "__main__":
    main()
