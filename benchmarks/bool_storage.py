#!/usr/bin/env python3
"""Packed bool columns vs uint8 .npy: the same binary design, two storages.

Writes one binary design (carrier status: 0/1) twice, as a Fortran-order
uint8 .npy (a byte per value) and as uncompressed Arrow files with bool
columns (a bit per value), one per batch size, then fits the same path on
each

- capped and cold: under a cgroup memory cap (systemd-run --user), each file
  evicted from the page cache first, so a design larger than the cap is
  re-read from disk at every pass, and
- warm: uncapped, after one discarded priming fit,

and checks that every fit is bit-identical to the first. Choose the cap
between the two sizes (a bit design 1/8 of the uint8 one) to see the case
packed bools are for: uint8 out of core, bits not.

A CPU gate waits for the machine to be quiet before every fit, as in
results/int8-zen4-quiet. Results append to results/bool-storage-<tag>/.

    uv run --with pyarrow benchmarks/bool_storage.py generate /scratch/bool \\
        --nrow 50000 --ncol 240000 --batch-rows 0 --batch-rows 65536
    uv run --with pyarrow benchmarks/bool_storage.py run /scratch/bool \\
        --tag zen3 --mem 4G --repeats 3 --threads 1 --threads 6

--batch-rows 0 means one batch. `scan` times a single pass over every
column, Parquet decode vs the mapped storages: what a solver working on
Parquet directly would pay at every pass over the data.
"""

import argparse
import datetime
import hashlib
import json
import pathlib
import platform
import resource
import subprocess
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from out_of_core import cgroup_stats, evict_from_page_cache, read_bytes  # noqa: E402


def arrow_name(batch_rows):
    return f"X-bool-{batch_rows or 'one'}.arrow"


def generate(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    import pybiglasso as bl

    workdir = pathlib.Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    nrow, ncol = args.nrow, args.ncol
    rng = np.random.default_rng(args.seed)
    weights = bl.common.sample_biglasso_weights(rng, ncol, marker_effect_sd=0.05)
    X = np.lib.format.open_memmap(
        workdir / "X-u8.npy", mode="w+", dtype=np.uint8, shape=(nrow, ncol),
        fortran_order=True,
    )  # fmt: skip
    eta = np.zeros(nrow)
    # bits are 1/8 of the uint8 design: held in memory to write one table
    columns = []
    for start in range(0, ncol, args.block):
        stop = min(ncol, start + args.block)
        af = rng.beta(0.4, 0.4, size=stop - start)
        carrier = rng.binomial(2, af, size=(nrow, stop - start)) > 0
        eta += (carrier - carrier.mean(axis=0)) @ weights[start:stop]
        X[:, start:stop] = carrier
        columns.extend(pa.array(carrier[:, k]) for k in range(stop - start))
    X.flush()
    del X
    y = (rng.random(nrow) < bl.common.logit_to_prob(eta)).astype(np.float64)
    np.save(workdir / "y.npy", y)

    table = pa.Table.from_arrays(columns, names=[f"v{j}" for j in range(ncol)])
    options = pa.ipc.IpcWriteOptions(compression=None)
    for batch_rows in args.batch_rows:
        with pa.ipc.new_file(
            workdir / arrow_name(batch_rows), table.schema, options=options
        ) as writer:
            writer.write_table(table, max_chunksize=batch_rows or nrow)
    if args.parquet:
        pq.write_table(table, workdir / "X.parquet", compression="zstd")
    (workdir / "params.json").write_text(
        json.dumps(dict(nrow=nrow, ncol=ncol, seed=args.seed))
    )


def open_x(workdir, storage):
    if storage == "npy-u8":
        return np.load(workdir / "X-u8.npy", mmap_mode="r")
    import pyarrow.feather as pf

    batch_rows = int(storage.removeprefix("arrow-bool-").replace("one", "0"))
    return pf.read_table(workdir / arrow_name(batch_rows), memory_map=True)


def fit(args):
    import pybiglasso as bl

    workdir = pathlib.Path(args.workdir)
    y = np.load(workdir / "y.npy")
    io0, t0 = read_bytes(), time.perf_counter()
    X = open_x(workdir, args.storage)
    opened = time.perf_counter() - t0
    path = bl.biglasso(
        X, y, family="binomial", nlambda=args.nlambda,
        lambda_min=args.lambda_min, n_threads=args.threads,
    )  # fmt: skip
    seconds = time.perf_counter() - t0 - opened
    io1 = read_bytes()
    np.savez(
        args.out.with_suffix(".npz"),
        lambdas=path.lambdas,
        intercept=path.intercept,
        coef=path.coef.toarray(),
        iterations=path.iterations,
    )
    record = dict(
        open_seconds=opened,
        fit_seconds=seconds,
        read_gb=None if io0 is None else (io1 - io0) / 1e9,
        max_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
        **cgroup_stats(),
    )
    args.out.with_suffix(".json").write_text(json.dumps(record))


def busy_fraction(seconds):
    def sample():
        values = list(map(int, open("/proc/stat").readline().split()[1:]))
        return sum(values[:8]), sum(values[:3])

    total0, busy0 = sample()
    time.sleep(seconds)
    total1, busy1 = sample()
    return (busy1 - busy0) / max(1, total1 - total0)


def wait_until_quiet(limit, quiet_seconds, sample_seconds=5):
    """Block until user+system CPU stays under `limit` for `quiet_seconds`."""
    started, quiet = time.perf_counter(), 0.0
    while quiet < quiet_seconds:
        if busy_fraction(sample_seconds) < limit:
            quiet += sample_seconds
        else:
            quiet = 0.0
    return time.perf_counter() - started


def provenance(workdir):
    import pyarrow

    from pybiglasso import _rust

    cpu = next(
        line.split(":", 1)[1].strip()
        for line in open("/proc/cpuinfo")
        if line.startswith("model name")
    )
    governors = sorted(
        {
            p.read_text().strip()
            for p in pathlib.Path("/sys/devices/system/cpu").glob(
                "cpu*/cpufreq/scaling_governor"
            )
        }
    )
    commit = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()  # fmt: skip
    return dict(
        cpu=cpu,
        governors=governors,
        python=platform.python_version(),
        numpy=np.__version__,
        pyarrow=pyarrow.__version__,
        source_commit=commit,
        extension_sha256=hashlib.sha256(
            pathlib.Path(_rust.__file__).read_bytes()
        ).hexdigest(),
        script_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        design=json.loads((workdir / "params.json").read_text()),
    )


def same_path(a, b):
    a, b = np.load(a), np.load(b)
    return all(np.array_equal(a[k], b[k]) for k in a.files)


def run(args):
    workdir = pathlib.Path(args.workdir).absolute()
    out = HERE / "results" / f"bool-storage-{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "provenance.json").write_text(json.dumps(provenance(workdir), indent=2))
    storages = ["npy-u8"] + [
        f"arrow-bool-{b or 'one'}" for b in sorted(args.batch_rows, reverse=True)
    ]
    files = {
        s: workdir / ("X-u8.npy" if s == "npy-u8" else arrow_name(int(
            s.removeprefix("arrow-bool-").replace("one", "0"))))
        for s in storages
    }  # fmt: skip
    reference = None
    regimes = [("capped-cold", mem) for mem in args.mem] + [("warm", None)]
    for regime, mem in regimes:
        for threads in args.threads:
            for storage in storages:
                for repeat in range(args.repeats + (regime == "warm")):
                    waited = wait_until_quiet(args.gate_busy, args.gate_seconds)
                    stem = workdir / f"fit-{regime}-{mem}-{storage}-{threads}"
                    cmd = [
                        sys.executable, __file__, "fit", str(workdir),
                        "--storage", storage, "--threads", str(threads),
                        "--nlambda", str(args.nlambda),
                        "--lambda-min", str(args.lambda_min), "--out", str(stem),
                    ]  # fmt: skip
                    if mem:
                        evict_from_page_cache(files[storage])
                        cmd = [
                            "systemd-run", "--user", "--scope", "-q",
                            "-p", f"MemoryMax={mem}", "-p", "MemorySwapMax=0",
                        ] + cmd  # fmt: skip
                    subprocess.run(cmd, check=True)
                    reference = reference or stem.with_suffix(".npz")
                    record = dict(
                        time=datetime.datetime.now(datetime.UTC).isoformat(),
                        regime=regime,
                        mem=mem,
                        storage=storage,
                        threads=threads,
                        repeat=repeat,
                        # the first warm fit only fills the page cache
                        priming=regime == "warm" and repeat == 0,
                        gate_wait_seconds=waited,
                        identical_to_first=same_path(
                            reference, stem.with_suffix(".npz")
                        ),
                        **json.loads(stem.with_suffix(".json").read_text()),
                    )
                    with (out / "results.jsonl").open("a") as stream:
                        stream.write(json.dumps(record) + "\n")
                    print(json.dumps(record), flush=True)
                    if not record["identical_to_first"]:
                        sys.exit(f"{storage} differs from the first fit: stopping")


def scan(args):
    """One pass over every column: decode Parquet vs read the mapped storages."""
    import pyarrow.compute as pc
    import pyarrow.feather as pf
    import pyarrow.parquet as pq

    workdir = pathlib.Path(args.workdir)
    results = {}
    for name, path in [
        ("parquet-zstd", workdir / "X.parquet"),
        ("npy-u8", workdir / "X-u8.npy"),
        ("arrow-bool-one", workdir / arrow_name(0)),
    ]:
        evict_from_page_cache(path)
        t0 = time.perf_counter()
        if name == "parquet-zstd":
            for batch in pq.ParquetFile(path).iter_batches(batch_size=args.block):
                for column in batch.columns:
                    pc.sum(column)
        elif name == "npy-u8":
            X = np.load(path, mmap_mode="r")
            for j in range(X.shape[1]):
                int(X[:, j].sum())
        else:
            for column in pf.read_table(path, memory_map=True).columns:
                pc.sum(column)
        results[name] = time.perf_counter() - t0
    print(json.dumps(results))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("workdir")
    g.add_argument("--nrow", type=int, default=50_000)
    g.add_argument("--ncol", type=int, default=240_000)
    g.add_argument("--batch-rows", type=int, action="append")
    g.add_argument("--block", type=int, default=2_000)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--parquet", action="store_true")
    f = sub.add_parser("fit")
    f.add_argument("workdir")
    f.add_argument("--storage", required=True)
    f.add_argument("--threads", type=int, default=1)
    f.add_argument("--nlambda", type=int, default=20)
    f.add_argument("--lambda-min", type=float, default=0.1)
    f.add_argument("--out", type=pathlib.Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("workdir")
    r.add_argument("--tag", required=True)
    r.add_argument("--mem", action="append", default=[])
    r.add_argument("--batch-rows", type=int, action="append")
    r.add_argument("--threads", type=int, action="append")
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--nlambda", type=int, default=20)
    r.add_argument("--lambda-min", type=float, default=0.1)
    r.add_argument("--gate-busy", type=float, default=0.10)
    r.add_argument("--gate-seconds", type=float, default=180)
    s = sub.add_parser("scan")
    s.add_argument("workdir")
    s.add_argument("--block", type=int, default=65_536)
    args = ap.parse_args()
    if getattr(args, "batch_rows", None) is None and args.cmd in ("generate", "run"):
        args.batch_rows = [0]
    if args.cmd == "run":
        args.threads = args.threads or [1]
    return dict(generate=generate, fit=fit, run=run, scan=scan)[args.cmd](args)


if __name__ == "__main__":
    main()
