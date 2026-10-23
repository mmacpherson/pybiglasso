#!/usr/bin/env python3
"""Fit a path on a design matrix several times larger than the memory allowed.

Generates a genotype-like dosage matrix on disk (raw column-major float64,
readable by np.memmap and by R), then fits the same lasso path with

- pybiglasso, on a memmap, under a cgroup memory cap (systemd-run --user), and
- R biglasso, on a file-backed big.matrix, in a container with the same cap,

and checks that the two agree. Under cgroup v2 the page cache counts against
the cap, so the matrix really is re-read from disk rather than cached.

    uv run benchmarks/out_of_core.py run /scratch/biglasso-ooc \\
        --nrow 20000 --ncol 150000 --mem 6G
"""

import argparse
import json
import os
import pathlib
import resource
import subprocess
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "tests"))


def generate(workdir, nrow, ncol, seed=0, block=2_000):
    """Write X.bin, y.bin, params.json a block of columns at a time."""
    import pybiglasso as bl

    workdir = pathlib.Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    weights = bl.common.sample_biglasso_weights(rng, ncol, marker_effect_sd=0.05)
    X = np.memmap(
        workdir / "X.bin", dtype=np.float64, mode="w+", shape=(nrow, ncol), order="F"
    )
    eta = np.zeros(nrow)
    for start in range(0, ncol, block):
        stop = min(ncol, start + block)
        af = rng.beta(0.4, 0.4, size=stop - start)
        dos = rng.binomial(2, af, size=(nrow, stop - start)).astype(np.float64)
        eta += (dos - 2 * af) @ weights[start:stop]
        X[:, start:stop] = dos
        X.flush()
    del X
    y = (rng.random(nrow) < bl.common.logit_to_prob(eta)).astype(np.float64)
    y.tofile(workdir / "y.bin")
    return y


def write_params(workdir, nrow, ncol, nlambda, lambda_min):
    params = dict(
        nrow=nrow,
        ncol=ncol,
        ny=nrow,
        alpha=1.0,
        nlambda=nlambda,
        lambda_min=lambda_min,
        lambda_log_scale=True,
        eps=1e-7,
        max_iter=1000,
        filebacked=True,
        cgroup_stats=True,  # R reports its container's memory accounting
    )
    (pathlib.Path(workdir) / "params.json").write_text(json.dumps(params))
    return params


def evict_from_page_cache(path):
    """Drop a file's clean pages from the page cache (no root needed).

    Pages cached earlier are charged to whichever cgroup first read or wrote
    them, so without this a capped run could reuse them without ever being
    charged, and would not really be out of core.
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def cgroup_stats():
    """Memory accounting for this process's cgroup (v2).

    Unlike RSS, this includes page cache, so it shows whether the cap held.
    Refaults count file pages read back in after being evicted: the re-reads
    that make a run genuinely out of core.
    """
    try:
        rel = open("/proc/self/cgroup").read().strip().split("::", 1)[1]
        base = pathlib.Path("/sys/fs/cgroup") / rel.lstrip("/")
        stat = dict(
            line.split() for line in (base / "memory.stat").read_text().splitlines()
        )
        page = os.sysconf("SC_PAGE_SIZE")
        return dict(
            cap_gb=int((base / "memory.max").read_text()) / 1e9,
            peak_gb=int((base / "memory.peak").read_text()) / 1e9,
            major_faults=int(stat.get("pgmajfault", 0)),
            refaulted_gb=int(stat.get("workingset_refault_file", 0)) * page / 1e9,
        )
    except (OSError, ValueError, IndexError):
        return {}


def read_bytes():
    try:
        for line in open("/proc/self/io"):
            if line.startswith("read_bytes"):
                return int(line.split()[1])
    except OSError:
        pass
    return None


def fit_py(workdir, threads):
    import pybiglasso as bl

    workdir = pathlib.Path(workdir)
    prm = json.loads((workdir / "params.json").read_text())
    X = np.memmap(
        workdir / "X.bin",
        dtype=np.float64,
        mode="r",
        shape=(prm["nrow"], prm["ncol"]),
        order="F",
    )
    y = np.fromfile(workdir / "y.bin")
    # compile on a tiny problem, outside the timing. It must be a read-only
    # Fortran memmap like X: numba compiles each array type separately, and
    # compiling inside the fit would also take memory from the page cache.
    warm = np.lib.format.open_memmap(
        workdir / "warm.npy", mode="w+", shape=(50, 20), fortran_order=True
    )
    warm[:] = X[:50, :20]
    warm.flush()
    bl.biglasso(np.load(workdir / "warm.npy", mmap_mode="r"), y[:50], nlambda=2)
    io0, t0 = read_bytes(), time.perf_counter()
    path = bl.biglasso(
        X,
        y,
        alpha=prm["alpha"],
        nlambda=prm["nlambda"],
        lambda_min=prm["lambda_min"],
        eps=prm["eps"],
        n_threads=threads,
    )
    seconds = time.perf_counter() - t0
    io1 = read_bytes()
    out = dict(
        seconds=seconds,
        read_gb=None if io0 is None else (io1 - io0) / 1e9,
        max_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
        **cgroup_stats(),
        threads=threads,
        n_nonzero=path.n_nonzero.tolist(),
    )
    np.savez(
        workdir / f"py-{threads}.npz",
        lambdas=path.lambdas,
        intercept=path.intercept,
        coef=path.coef.toarray(),
        loss=path.deviance / 2,
        iter=path.iterations,
        center=path.center,
        scale=path.scale,
        rejections=path.n_rejected,
    )
    (workdir / f"py-{threads}.json").write_text(json.dumps(out))
    print(json.dumps(out))


def run(args):
    workdir = pathlib.Path(args.workdir).absolute()
    gb = args.nrow * args.ncol * 8 / 1e9
    print(f"design: {args.nrow} x {args.ncol} float64 = {gb:.1f} GB; cap {args.mem}")
    if not (workdir / "X.bin").exists():
        t0 = time.perf_counter()
        generate(workdir, args.nrow, args.ncol)
        print(f"generated in {time.perf_counter() - t0:.0f}s")
    write_params(workdir, args.nrow, args.ncol, args.nlambda, args.lambda_min)

    results = {}
    for threads in args.threads:
        evict_from_page_cache(workdir / "X.bin")
        cmd = [
            "systemd-run", "--user", "--scope", "-q",
            "-p", f"MemoryMax={args.mem}", "-p", "MemorySwapMax=0",
            sys.executable, __file__, "fit-py", str(workdir),
            "--threads", str(threads),
        ]  # fmt: skip
        subprocess.run(cmd, check=True)
        results[f"pybiglasso threads={threads}"] = json.loads(
            (workdir / f"py-{threads}.json").read_text()
        )

    if not args.skip_r:
        image = os.environ.get("R_IMAGE", "pybiglasso-r:latest")
        container = os.environ.get("CONTAINER", "docker")
        script = REPO / "tests" / "r" / "fit_biglasso.R"
        cmd = [
            container, "run", "--rm",
            f"--memory={args.mem}", f"--memory-swap={args.mem}",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-v", f"{workdir}:{workdir}", "-v", f"{script.parent}:{script.parent}",
            image, "Rscript", str(script), str(workdir),
        ]  # fmt: skip
        evict_from_page_cache(workdir / "X.bin")
        # R builds its file-backed big.matrix from X.bin each run (timed in
        # wall_incl_filebacked_setup) and refuses to overwrite an old one
        for stale in ("X.bm", "X.desc"):
            (workdir / stale).unlink(missing_ok=True)
        t0 = time.perf_counter()
        subprocess.run(cmd, check=True)
        wall = time.perf_counter() - t0
        import rcompare

        r = rcompare.load_r_output(workdir / "out.json")
        if "error" in r:
            sys.exit(f"R failed: {r['error']}")
        raw = json.loads((workdir / "out.json").read_text())
        results["R biglasso"] = dict(
            seconds=r["seconds"],
            wall_incl_filebacked_setup=wall,
            **raw.get("cgroup", {}),
        )
        ref = np.load(workdir / f"py-{args.threads[0]}.npz")
        py = dict(
            center=ref["center"],
            scale=ref["scale"],
            lambda_=ref["lambdas"],
            loss=ref["loss"],
            iter=ref["iter"],
            rejections=ref["rejections"],
            beta=np.vstack([ref["intercept"], ref["coef"]]),
        )
        # pybiglasso's default deviance is the final model's; R's loss is
        # from before the final sweep (r_compatible=True would match it)
        py["loss"] = r["loss"]
        problems = rcompare.compare(py, r)
        results["agreement"] = problems or (
            "agree (rtol=1e-9), identical iterations; loss not compared"
        )

    print(json.dumps(results, indent=2))
    out = HERE / "results" / "out_of_core.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            dict(nrow=args.nrow, ncol=args.ncol, gb=gb, mem=args.mem, **results),
            indent=2,
        )
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("workdir")
    r.add_argument("--nrow", type=int, default=20_000)
    r.add_argument("--ncol", type=int, default=150_000)
    r.add_argument("--nlambda", type=int, default=20)
    r.add_argument("--lambda-min", type=float, default=0.1)
    r.add_argument("--mem", default="6G")
    r.add_argument("--threads", type=int, action="append")
    r.add_argument("--skip-r", action="store_true")
    f = sub.add_parser("fit-py")
    f.add_argument("workdir")
    f.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    if args.cmd == "fit-py":
        return fit_py(args.workdir, args.threads)
    args.threads = args.threads or [1]
    return run(args)


if __name__ == "__main__":
    main()
