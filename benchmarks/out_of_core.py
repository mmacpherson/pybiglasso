#!/usr/bin/env python3
"""Fit a path on a design matrix several times larger than the memory allowed.

Generates a genotype-like dosage matrix on disk (raw column-major float64,
readable by np.memmap and by R), then fits the same lasso path with

- pybiglasso, on a memmap, under a cgroup memory cap (systemd-run --user), and
- R biglasso, on a file-backed big.matrix, in a container with the same cap,

and checks that the two agree. Under cgroup v2 the page cache counts against
the cap, so the matrix really is re-read from disk rather than cached.

The Gaussian case prepares R's backing file before any measured fit. Each
solver's own backing file is then evicted, and residency is checked before
the process starts. --prepare-only generates and prepares storage without
fitting; --diagnostic runs correctness checks without publishing timings.

    uv run benchmarks/out_of_core.py run /scratch/biglasso-ooc \\
        --nrow 20000 --ncol 150000 --mem 6G
"""

import argparse
import hashlib
import json
import os
import pathlib
import platform
import struct
import subprocess
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "tests"))


def generate(workdir, nrow, ncol, seed=0, block=2_000, family="binomial"):
    """Write X.bin, y.bin, params.json a block of columns at a time."""
    import pybiglasso as bl

    workdir = pathlib.Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    # A failed regeneration must not leave the previous completion receipt.
    (workdir / "data.json").unlink(missing_ok=True)
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
    if family == "gaussian":
        y = eta + rng.normal(size=nrow)
    else:
        y = (rng.random(nrow) < bl.common.logit_to_prob(eta)).astype(np.float64)
    y.tofile(workdir / "y.bin")
    (workdir / "data.json").write_text(
        json.dumps(dict(nrow=nrow, ncol=ncol, family=family, seed=seed, block=block))
    )
    return y


def validate_data(workdir, nrow, ncol, family):
    """Do not silently reuse a response generated for another experiment."""
    workdir = pathlib.Path(workdir)
    metadata = workdir / "data.json"
    if metadata.exists():
        recorded = json.loads(metadata.read_text())
        for name, requested in (("nrow", nrow), ("ncol", ncol), ("family", family)):
            if recorded.get(name) != requested:
                raise ValueError(f"{metadata}: {name} does not match {requested!r}")
    elif family != "binomial":
        raise ValueError(
            f"{metadata} is missing; generate Gaussian data in a new directory"
        )
    for name, size in (("X.bin", nrow * ncol * 8), ("y.bin", nrow * 8)):
        path = workdir / name
        if path.stat().st_size != size:
            raise ValueError(f"{path} must contain exactly {size} bytes")


def write_params(
    workdir, nrow, ncol, nlambda, lambda_min, family="binomial", recording=False
):
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
        family=family,
        recording=recording,
    )
    (pathlib.Path(workdir) / "params.json").write_text(json.dumps(params))
    return params


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def path_digest(arrays):
    digest = hashlib.sha256(b"out-of-core-path-v1\0")
    for name, array in sorted(arrays.items()):
        frame = json.dumps([name, array.dtype.str, array.shape]).encode()
        payload = array.tobytes()
        digest.update(struct.pack("<QQ", len(frame), len(payload)))
        digest.update(frame)
        digest.update(payload)
    return digest.hexdigest()


def environment(build_info=None):
    import importlib.metadata

    import pybiglasso._rust as native

    cpu = platform.processor()
    for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            cpu = line.split(":", 1)[1].strip()
            break
    result = dict(
        cpu=cpu,
        python=platform.python_version(),
        kernel=platform.release(),
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        native_sha256=file_digest(pathlib.Path(native.__file__)),
        **{
            name: importlib.metadata.version(name)
            for name in ("pybiglasso", "numpy", "scipy")
        },
    )
    if build_info is not None:
        recorded = json.loads(build_info.read_text())
        if recorded.get("native_sha256") != result["native_sha256"]:
            raise ValueError(
                "build-info does not identify the installed native extension"
            )
        names = (
            "rustc",
            "llvm",
            "target",
            "opt_level",
            "lto",
            "codegen_units",
            "rustflags",
            "wheel_sha256",
            "maturin",
            "image_digest",
        )
        result["build"] = {name: recorded[name] for name in names}
        if "native_source_commit" in recorded:
            result["build"]["native_source_commit"] = recorded["native_source_commit"]
    return result


def storage_info(workdir):
    result = {}
    for name in ("X.bin", "X.bm", "y.bin"):
        path = workdir / name
        if not path.exists():
            continue
        stat = path.stat()
        result[name] = dict(
            logical_bytes=stat.st_size,
            allocated_bytes=stat.st_blocks * 512,
            sha256=file_digest(path),
        )
    if "X.bm" in result and result["X.bin"]["sha256"] != result["X.bm"]["sha256"]:
        raise RuntimeError("R's backing file no longer matches the input matrix")
    return result


def prepare_r(workdir, mem):
    """Prepare R's separate backing file before eviction and the timed run."""
    source_digest = file_digest(workdir / "X.bin")
    receipt = workdir / "r-prepared.json"
    if receipt.exists():
        recorded = json.loads(receipt.read_text())
        if (
            recorded["source_sha256"] == source_digest
            and (workdir / "X.desc").exists()
            and (workdir / "X.bm").exists()
            and file_digest(workdir / "X.bm") == source_digest
        ):
            return
    for name in ("X.bm", "X.desc"):
        (workdir / name).unlink(missing_ok=True)
    container = os.environ.get("CONTAINER", "docker")
    image = os.environ.get("R_IMAGE", "pybiglasso-r:r4.6.1")
    script = HERE / "r_filebacked.R"
    subprocess.run(
        [
            container,
            "run",
            "--rm",
            f"--memory={mem}",
            f"--memory-swap={mem}",
            "-e",
            "OPENBLAS_NUM_THREADS=1",
            "-e",
            "OMP_NUM_THREADS=1",
            "-e",
            "MKL_NUM_THREADS=1",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "-v",
            f"{workdir}:{workdir}",
            "-v",
            f"{HERE}:{HERE}:ro",
            image,
            "Rscript",
            str(script),
            str(workdir),
            "prepare",
        ],
        check=True,
    )
    if file_digest(workdir / "X.bm") != source_digest:
        raise ValueError("R's backing file does not contain the exact input bytes")
    receipt.write_text(json.dumps(dict(source_sha256=source_digest)))


def resident_bytes(path):
    """Inspect page residency without faulting the matrix into memory."""
    import ctypes

    page = os.sysconf("SC_PAGE_SIZE")
    data = np.memmap(path, dtype=np.uint8, mode="r")
    pages = (data.size + page - 1) // page
    status = (ctypes.c_ubyte * pages)()
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    libc.mincore.restype = ctypes.c_int
    if libc.mincore(data.ctypes.data, data.size, status):
        raise OSError(ctypes.get_errno(), "mincore")
    return min(data.size, sum(value & 1 for value in status) * page)


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


def cpu_use(seconds):
    def counters():
        fields = pathlib.Path("/proc/stat").read_text().splitlines()[0].split()[1:9]
        values = list(map(int, fields))
        return sum(values), sum(values[:3])

    total, busy = counters()
    time.sleep(seconds)
    next_total, next_busy = counters()
    return (next_busy - busy) / (next_total - total)


def wait_for_quiet():
    """The published-record protocol: 180 consecutive seconds below 10%."""
    quiet = 0
    samples = []
    while quiet < 180:
        sample = cpu_use(5)
        samples.append(sample)
        quiet = quiet + 5 if sample < 0.1 else 0
    return dict(samples=samples, threshold=0.1, consecutive_seconds=180)


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
            current_gb=int((base / "memory.current").read_text()) / 1e9,
            anon_gb=int(stat.get("anon", 0)) / 1e9,
            swap_bytes=int((base / "memory.swap.current").read_text()),
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
    import resource

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
    before = cgroup_stats()
    cached_before = resident_bytes(workdir / "X.bin")
    pre = cpu_use(1) if prm.get("recording") else None
    if pre is not None and pre > 0.15:
        raise RuntimeError(f"Busy pre-fit CPU bookend: {pre}")
    io0, t0 = read_bytes(), time.perf_counter()
    path = bl.biglasso(
        X,
        y,
        alpha=prm["alpha"],
        nlambda=prm["nlambda"],
        lambda_min=prm["lambda_min"],
        eps=prm["eps"],
        n_threads=threads,
        family=prm.get("family", "binomial"),
    )
    seconds = time.perf_counter() - t0
    io1 = read_bytes()
    post = cpu_use(1) if prm.get("recording") else None
    if post is not None and post > 0.15:
        raise RuntimeError(f"Busy post-fit CPU bookend: {post}")
    out = dict(
        seconds=seconds,
        read_gb=None if io0 is None else (io1 - io0) / 1e9,
        max_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e9,
        **cgroup_stats(),
        cached_before_bytes=cached_before,
        cpu_before=pre,
        cpu_after=post,
        threads=threads,
        n_nonzero=path.n_nonzero.tolist(),
    )
    if before:
        out["refaulted_gb"] -= before["refaulted_gb"]
        out["major_faults"] -= before["major_faults"]
    arrays = dict(
        lambdas=path.lambdas,
        intercept=path.intercept,
        coef=path.coef.toarray(),
        coef_data=path.coef.data,
        coef_indices=path.coef.indices,
        coef_indptr=path.coef.indptr,
        coef_shape=np.asarray(path.coef.shape),
        loss=path.deviance / 2,
        iter=path.iterations,
        center=path.center,
        scale=path.scale,
        rejections=path.n_rejected,
    )
    out["path_sha256"] = path_digest(arrays)
    np.savez(workdir / f"py-{threads}.npz", **arrays)
    (workdir / f"py-{threads}.json").write_text(json.dumps(out))
    print(json.dumps(out))


def run(args):
    workdir = pathlib.Path(args.workdir).absolute()
    gb = args.nrow * args.ncol * 8 / 1e9
    print(f"design: {args.nrow} x {args.ncol} float64 = {gb:.1f} GB; cap {args.mem}")
    if args.diagnostic:
        print("DIAGNOSTIC: correctness smoke only; timings are not a published record")
    if not (workdir / "X.bin").exists():
        t0 = time.perf_counter()
        generate(workdir, args.nrow, args.ncol, family=args.family)
        print(f"generated in {time.perf_counter() - t0:.0f}s")
    validate_data(workdir, args.nrow, args.ncol, args.family)
    write_params(
        workdir,
        args.nrow,
        args.ncol,
        args.nlambda,
        args.lambda_min,
        args.family,
        recording=args.family == "gaussian" and not args.diagnostic,
    )
    if args.family == "gaussian" and not args.skip_r:
        prepare_r(workdir, args.mem)

    if args.prepare_only:
        return

    metadata = None if args.diagnostic else environment(args.build_info)

    results = {}
    gates = {}
    for threads in args.threads:
        if args.family == "gaussian" and not args.diagnostic:
            gates[f"pybiglasso threads={threads}"] = wait_for_quiet()
        evict_from_page_cache(workdir / "X.bin")
        if resident_bytes(workdir / "X.bin"):
            raise RuntimeError("X.bin still has cached pages after eviction")
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
    if len({value["path_sha256"] for value in results.values()}) != 1:
        raise RuntimeError(
            "Python paths differ in their numeric outputs or recorded trajectory"
        )

    if not args.skip_r:
        image = os.environ.get(
            "R_IMAGE",
            "pybiglasso-r:r4.6.1"
            if args.family == "gaussian"
            else "pybiglasso-r:latest",
        )
        container = os.environ.get("CONTAINER", "docker")
        script = (
            HERE / "r_filebacked.R"
            if args.family == "gaussian"
            else REPO / "tests" / "r" / "fit_biglasso.R"
        )
        cmd = [
            container, "run", "--rm",
            f"--memory={args.mem}", f"--memory-swap={args.mem}",
            "-e", "OPENBLAS_NUM_THREADS=1", "-e", "OMP_NUM_THREADS=1",
            "-e", "MKL_NUM_THREADS=1",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-v", f"{workdir}:{workdir}", "-v", f"{script.parent}:{script.parent}",
            image, "Rscript", str(script), str(workdir),
        ]  # fmt: skip
        evict_from_page_cache(workdir / "X.bin")
        # The legacy binomial case still builds storage in the container;
        # Gaussian attaches the already prepared backing file after eviction.
        r_cached_before = None
        if args.family == "gaussian":
            if not args.diagnostic:
                gates["R biglasso"] = wait_for_quiet()
            evict_from_page_cache(workdir / "X.bm")
            r_cached_before = resident_bytes(workdir / "X.bm")
            if r_cached_before:
                raise RuntimeError("X.bm still has cached pages after eviction")
        else:
            for stale in ("X.bm", "X.desc"):
                (workdir / stale).unlink(missing_ok=True)
        t0 = time.perf_counter()
        subprocess.run(cmd, check=True)
        wall = time.perf_counter() - t0
        from support import rcompare

        r = rcompare.load_r_output(workdir / "out.json")
        if "error" in r:
            sys.exit(f"R failed: {r['error']}")
        raw = json.loads((workdir / "out.json").read_text())
        results["R biglasso"] = dict(
            seconds=r["seconds"],
            **{
                "wall_incl_process_setup"
                if args.family == "gaussian"
                else "wall_incl_filebacked_setup": wall
            },
            r_version=raw.get("r_version"),
            biglasso_version=raw.get("biglasso_version"),
            cpu_before=raw.get("cpu_before"),
            cpu_after=raw.get("cpu_after"),
            cached_before_bytes=r_cached_before,
            threads=1,
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
        # Compare the recorded arrays directly. The two losses have different
        # final-sweep timing, so they are deliberately excluded here.
        problems = []
        for key in ("center", "scale", "lambda_", "beta"):
            if np.shape(py[key]) != np.shape(r[key]) or not np.allclose(
                py[key], r[key], rtol=rcompare.RTOL, atol=rcompare.ATOL
            ):
                problems.append(f"{key} differs from R")
        for key in ("iter", "rejections"):
            if not np.array_equal(py[key], r[key]):
                problems.append(f"{key} differs from R")
        results["agreement"] = problems or (
            "agree (rtol=1e-9), identical iterations; loss not compared"
        )
        if problems and args.family == "gaussian":
            raise RuntimeError("; ".join(problems))

    print(json.dumps(results, indent=2))
    if args.diagnostic:
        return results
    out = args.out or HERE / "results" / (
        "out_of_core-gaussian.json" if args.family == "gaussian" else "out_of_core.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            dict(
                nrow=args.nrow,
                ncol=args.ncol,
                gb=gb,
                mem=args.mem,
                family=args.family,
                environment=metadata,
                quiet_gates=gates,
                storage=storage_info(workdir),
                **results,
            ),
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
    r.add_argument("--family", choices=["binomial", "gaussian"], default="binomial")
    r.add_argument("--prepare-only", action="store_true")
    r.add_argument(
        "--diagnostic", action="store_true", help="do not publish a timing record"
    )
    r.add_argument("--out", type=pathlib.Path)
    r.add_argument(
        "--build-info", type=pathlib.Path, help="compiler/wheel provenance JSON"
    )
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
