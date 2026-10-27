"""Summarize retained timings without fitting or timing anything.

python benchmarks/thread_policy.py
"""

import collections
import json
import math
import statistics
from pathlib import Path

EVIDENCE = Path(__file__).parent / "results/thread-policy-evidence.jsonl"


def comparisons(path=EVIDENCE):
    groups = collections.defaultdict(dict)
    for line in path.read_text().splitlines():
        row = json.loads(line)
        key = tuple(
            row[name]
            for name in (
                "source_commit",
                "source_file",
                "cpu",
                "quality",
                "problem",
                "family",
                "dtype",
                "alpha",
                "nrow",
                "ncol",
            )
        )
        if row["threads"] in groups[key]:
            raise ValueError("Duplicate thread-count record")
        groups[key][row["threads"]] = statistics.median(row["times"])
    for key, times in groups.items():
        if 1 not in times:
            raise ValueError("Missing single-thread reference")
        for threads, seconds in sorted(times.items()):
            if threads > 1:
                yield dict(
                    cpu=key[2],
                    quality=key[3],
                    problem=key[4],
                    family=key[5],
                    dtype=key[6],
                    alpha=key[7],
                    work=key[8] * key[9],
                    threads=threads,
                    speedup=times[1] / seconds,
                )


def cutoff(rows):
    # Declared 10% material-gain margin. Keep exploratory M1 and SMT counts
    # in the report, but neither calibrates the physical-core cutoff.
    by_work = collections.defaultdict(list)
    for row in rows:
        if row["quality"] == "retained" and row["threads"] == 6:
            by_work[row["work"]].append(row["speedup"])
    small = max(work for work, ratios in by_work.items() if min(ratios) < 1.1)
    large = min(work for work, ratios in by_work.items() if min(ratios) >= 1.1)
    if small >= large:
        raise ValueError("Records do not separate into a single workload cutoff")
    return small, large, math.isqrt(small * large)


def main():
    rows = list(comparisons())
    small, large, midpoint = cutoff(rows)
    print(f"Largest near-neutral work: {small:,}")
    print(f"Smallest sustained >=10% win: {large:,}")
    print(f"Logarithmic midpoint: {midpoint:,}\n")
    print("CPU | Problem | Family/dtype | Work | Threads | 1t / Nt range")
    print("--- | --- | --- | ---: | ---: | ---:")
    groups = collections.defaultdict(list)
    for row in rows:
        key = tuple(
            row[name]
            for name in ("cpu", "problem", "family", "dtype", "work", "threads")
        )
        groups[key].append(row["speedup"])
    for (cpu, problem, family, dtype, work, threads), ratios in sorted(groups.items()):
        print(
            f"{cpu} | {problem} | {family}/{dtype} | {work:,} | {threads} | {min(ratios):.3f}–{max(ratios):.3f}"
        )


if __name__ == "__main__":
    main()
