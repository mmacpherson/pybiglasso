"""Three-build whole-path experiment with a CPU gate before every setting."""

import datetime
import hashlib
import importlib.util
import itertools
import json
import os
import platform
import statistics
import subprocess
import time
import warnings
from collections import deque
from pathlib import Path

os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
import numpy as np

import pybiglasso as bl
from pybiglasso import _rust

ROOT = Path(__file__).resolve().parent
VARIANTS = {"baseline": "5065100", "ungated": "2b03702", "gated": "8128489"}


def utc():
    return datetime.datetime.now(datetime.UTC).isoformat()


def append(name, record):
    with (ROOT / name).open("a") as stream:
        stream.write(json.dumps(record) + "\n")


def cpu():
    values = list(map(int, Path("/proc/stat").read_text().splitlines()[0].split()[1:]))
    return sum(values[:8]), sum(values[:3]), values[4]


def workers():
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            comm = (entry / "comm").read_text().strip()
            # Read only to classify, never publish command arguments.
            args = (entry / "cmdline").read_bytes().decode(errors="replace")
            compiler = comm in {
                "rustc",
                "cargo",
                "cc1",
                "cc1plus",
                "clang",
                "clang++",
                "gcc",
                "g++",
                "gfortran",
                "clippy-driver",
                "maturin",
            }
            testing = comm in {"pytest", "R", "Rscript"} or (
                comm.startswith("python")
                and any(
                    s in args
                    for s in (
                        "pytest",
                        "benchmarks/",
                        "int8-threeway",
                        "gaussian-sum.py",
                    )
                )
            )
            if compiler or testing:
                found.append({"pid": int(entry.name), "comm": comm})
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return found


def power():
    governors = sorted(
        {
            p.read_text().strip()
            for p in Path("/sys/devices/system/cpu").glob(
                "cpu[0-9]*/cpufreq/scaling_governor"
            )
        }
    )
    # This desktop exposes no AC/battery sensor; record available sensors only.
    supplies = {
        path.parent.name: path.read_text().strip()
        for path in Path("/sys/class/power_supply").glob("*/online")
    }
    preferences = sorted(
        {
            path.read_text().strip()
            for path in Path("/sys/devices/system/cpu").glob(
                "cpu[0-9]*/cpufreq/energy_performance_preference"
            )
        }
    )
    assert governors == ["powersave"], governors
    assert preferences == ["balance_performance"], preferences
    return {
        "power_supply_online": supplies,
        "governors": governors,
        "energy_performance_preferences": preferences,
    }


def gate(label):
    window = deque(maxlen=37)
    previous = cpu()
    while True:
        time.sleep(5)
        now = cpu()
        total = now[0] - previous[0]
        busy = 100 * (now[1] - previous[1]) / total
        iowait = 100 * (now[2] - previous[2]) / total
        previous = now
        active = workers()
        tick = time.monotonic()
        window.append((tick, busy, bool(active)))
        duration = tick - window[0][0]
        median_busy = statistics.median(sample[1] for sample in window)
        workers_seen = any(sample[2] for sample in window)
        good = duration >= 180 and median_busy <= 15 and not workers_seen
        record = dict(
            label=label,
            utc=utc(),
            user_system_pct=busy,
            iowait_pct=iowait,
            load=list(os.getloadavg()),
            workers=active,
            qualifies=good,
            window_seconds=duration,
            window_samples=len(window),
            median_busy_pct=median_busy,
            window_workers_seen=workers_seen,
            **power(),
        )
        append("gate.jsonl", record)
        if good:
            append("gate-passed.jsonl", record)
            print("GATE PASSED " + label, flush=True)
            return record


def bookend(label):
    before = cpu()
    time.sleep(1)
    after = cpu()
    return dict(
        label=label,
        utc=utc(),
        user_system_pct=100 * (after[1] - before[1]) / (after[0] - before[0]),
        iowait_pct=100 * (after[2] - before[2]) / (after[0] - before[0]),
        load=list(os.getloadavg()),
        workers=workers(),
        **power(),
    )


def path_hash(path):
    digest = hashlib.sha256()
    coef = path.coef.tocsc()
    for value in (
        path.center,
        path.scale,
        path.lambdas,
        path.intercept,
        coef.data,
        coef.indices,
        coef.indptr,
        path.iterations,
        path.n_rejected,
        path.deviance,
    ):
        digest.update(str((value.dtype.str, value.shape)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def fit(module, X, y, **options):
    bl._rust = module
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        start_utc = utc()
        start = time.perf_counter()
        path = bl.biglasso(X, y, nlambda=100, backend="rust", **options)
        seconds = time.perf_counter() - start
        end_utc = utc()
    assert len(path) == 100
    return dict(
        utc_start=start_utc,
        utc_end=end_utc,
        seconds=seconds,
        path_hash=path_hash(path),
        nlambda=len(path),
        iterations=int(path.iterations.sum()),
        max_iterations=int(path.iterations.max()),
    )


def main():
    assert not (ROOT / "results.jsonl").exists(), "Fresh output required"
    modules, provenance = {}, {}
    for name, revision in VARIANTS.items():
        checkout = ROOT / name
        commit = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
        assert commit.startswith(revision)
        assert not subprocess.check_output(
            ["git", "-C", str(checkout), "status", "--porcelain"], text=True
        )
        extension = checkout / "src/pybiglasso/_rust.abi3.so"
        spec = importlib.util.spec_from_file_location(name + "._rust", extension)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name] = module
        provenance[name] = dict(
            source_commit=commit,
            extension_sha256=hashlib.sha256(extension.read_bytes()).hexdigest(),
        )
    cpu_name = next(
        line.split(":", 1)[1].strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines()
        if line.startswith("model name")
    )
    metadata = dict(
        cpu=cpu_name,
        python=platform.python_version(),
        numpy=np.__version__,
        rustc=subprocess.check_output(["rustc", "-vV"], text=True).strip(),
        variants=provenance,
        gate_rule=dict(
            statistic="median",
            cpu_components=["user", "nice", "system"],
            median_busy_pct_max=15,
            window_seconds=180,
            sample_seconds=5,
            window_samples=37,
            iowait_used=False,
            load_used=False,
            no_workers=True,
        ),
        allowed_background=["IB Gateway/JxBrowser", "clementine-web", "cadvisor"],
        **power(),
    )
    (ROOT / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    orders = list(itertools.permutations(modules))
    identities = {}
    for family, dtype, problems, alphas in [
        ("binomial", "int8", [(2000, 20000), (5000, 50000)], [1.0, 0.5]),
        ("binomial", "float64", [(5000, 1000), (2000, 20000)], [1.0, 0.5]),
        ("gaussian", "float64", [(5000, 1000), (2000, 20000)], [0.5]),
    ]:
        for nrow, ncol in problems:
            data = bl.common.make_biglasso_dataset(nrow=nrow, ncol=ncol, seed=1)
            y = (
                data["y"]
                if family == "binomial"
                else 0.5 + 2 * data["y"] + data["X"][:, 1] / 4
            )
            X = np.asarray(data["X"], dtype=dtype, order="F")
            X.flags.writeable = False
            for alpha in alphas:
                for threads in (1, 6):
                    label = (
                        f"{family}-{dtype}-{nrow}x{ncol}-alpha{alpha}-threads{threads}"
                    )
                    gate(label)
                    append("environment.jsonl", bookend(label + "-before"))
                    if not (ROOT / "timing-started.json").exists():
                        (ROOT / "timing-started.json").write_text(
                            json.dumps(dict(utc=utc(), label=label)) + "\n"
                        )
                    options = dict(family=family, alpha=alpha, n_threads=threads)
                    warmups = [
                        dict(variant=name, **fit(module, X, y, **options))
                        for name, module in modules.items()
                    ]
                    fits = []
                    for repeat, order in enumerate(orders):
                        for position, name in enumerate(order):
                            fits.append(
                                dict(
                                    variant=name,
                                    repeat=repeat,
                                    position=position,
                                    **fit(modules[name], X, y, **options),
                                )
                            )
                    hashes = {r["path_hash"] for r in warmups + fits}
                    assert len(hashes) == 1, label
                    key = family, dtype, nrow, ncol, alpha
                    identity = next(iter(hashes))
                    assert identities.setdefault(key, identity) == identity, label
                    times = {
                        name: [r["seconds"] for r in fits if r["variant"] == name]
                        for name in modules
                    }
                    medians = {name: statistics.median(v) for name, v in times.items()}
                    append(
                        "results.jsonl",
                        dict(
                            label=label,
                            family=family,
                            dtype=dtype,
                            nrow=nrow,
                            ncol=ncol,
                            alpha=alpha,
                            threads=threads,
                            seed=1,
                            orders=orders,
                            times=times,
                            median_seconds=medians,
                            minimum_seconds={name: min(v) for name, v in times.items()},
                            baseline_over_ungated=medians["baseline"]
                            / medians["ungated"],
                            baseline_over_gated=medians["baseline"] / medians["gated"],
                            path_hash=identity,
                            fits=fits,
                            warmups=warmups,
                        ),
                    )
                    append("environment.jsonl", bookend(label + "-after"))
                    print("COMPLETED " + label, flush=True)
    bl._rust = _rust
    (ROOT / "timing-finished.json").write_text(
        json.dumps(dict(utc=utc(), settings=20)) + "\n"
    )
    print("FINISHED", flush=True)


if __name__ == "__main__":
    main()
