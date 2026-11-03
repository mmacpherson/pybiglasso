"""Shadow then enabled paths, compared to unmodified stock solver source."""
# ruff: noqa: E402 -- environment and source path must precede native imports.

import argparse
import dataclasses
import hashlib
import importlib
import importlib.util
import itertools
import json
import os
import struct
import sys
import warnings
from pathlib import Path

os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src"))
import numpy as np
import pyarrow as pa

import pybiglasso as bl

bm = importlib.import_module("pybiglasso.binomial")
ap = argparse.ArgumentParser()
ap.add_argument("--library", type=Path, required=True)
ap.add_argument("--stock", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)


def load(p):
    spec = importlib.util.spec_from_file_location("pybiglasso._rust", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


stock = load(args.stock)
native = load(args.library)
rng = np.random.default_rng(51324)
base = np.asfortranarray(rng.normal(size=(97, 9)))
noise = rng.normal(scale=0.2, size=97)
response = base[:, 0] - 0.7 * base[:, 1] + 0.3 * base[:, 3] + noise
cases = []


def add(name, X, y=response, **kw):
    cases.append((name, X, y, kw))


for dtype in [
    "float64",
    "float32",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "bool",
]:
    if dtype.startswith("uint"):
        X = np.asfortranarray(rng.integers(0, 8, size=(97, 9)).astype(dtype))
    elif dtype == "bool":
        X = np.asfortranarray(base > 0)
    elif dtype.startswith("int"):
        X = np.asfortranarray(np.rint(base * 4).astype(dtype))
    else:
        X = np.asfortranarray(base.astype(dtype))
    add("dtype-" + dtype, X)
add("c-order", np.array(base, order="C"))
add("strided", np.asfortranarray(np.vstack([base, base]))[::2])
add("offset", np.asfortranarray(base + 1e12))
add(
    "wide-i64",
    np.asfortranarray(
        rng.integers(0, 10000, size=base.shape, dtype=np.int64) * 256 + 2**60
    ),
)
add(
    "wide-u64",
    np.asfortranarray(
        rng.integers(0, 10000, size=base.shape, dtype=np.uint64) * 256
        + np.uint64(2**64 - 2**24)
    ),
)
for factor in [np.array([0, 0.2, 1.0, 3.0, 0.5, 0.0, 2.0, 4.0, 0.1]), np.zeros(9)]:
    add("penalty-" + str(factor.sum()), base, penalty_factor=factor)
add(
    "prefit-auto",
    base,
    lambdas=None,
    nlambda=8,
    penalty_factor=np.array([0.0, 0.0, 1.0, 2.0, 0.5, 1.0, 3.0, 0.1, 1.0]),
)
add(
    "prefit-collinear",
    np.asfortranarray(base + base[:, [0]] * 1e-4),
    lambdas=None,
    nlambda=8,
    penalty_factor=np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
)
add(
    "prefit-budget",
    base,
    lambdas=None,
    nlambda=8,
    max_iter=1,
    penalty_factor=np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
)
add("auto-grid", base, lambdas=None, nlambda=9)
add("repeated-grid", base, lambdas=np.array([0.5, 0.5, 0.2, 0.2, 0.01, 0.01]))
add("sharp-grid", base, lambdas=np.array([3.0, 0.01, 1e-8, 0.0]))
add("short-budget-1", base, max_iter=1)
add("short-budget-2", base, max_iter=2)
add("dfmax", base, dfmax=1)
add("gather", base, row_indices=np.arange(0, 97, 4, dtype=np.int64), _row_path="gather")
add("mask", base, row_indices=np.arange(1, 95, dtype=np.int64), _row_path="mask")
add(
    "repeated-gather",
    base,
    row_indices=np.array([0, 3, 0, 7, 17, 3, 7, 20, 50, 0, 96, 12, 20], dtype=np.int64),
    _row_path="gather",
)
correlated = base.copy(order="F")
correlated[:, 1] = correlated[:, 0] + 1e-4 * correlated[:, 1]
add("near-collinear", correlated)
add("tiny-response", base, response * 1e-200, lambdas=np.array([5e-201, 1e-201, 0.0]))
add("overflow-response", base, response * 1e200)
add(
    "huge-penalty",
    base,
    penalty_factor=np.array([1e308, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    lambdas=np.array([3.0, 0.01, 0.001]),
)
nonfinite_y = response.copy()
nonfinite_y[0] = np.nan
add("unused-y", base, nonfinite_y, row_indices=np.arange(1, 97), _row_path="mask")
for bad in [np.nan, np.inf, 1e308]:
    X = base.copy(order="F")
    X[0, 0] = bad
    add(
        "mask-fallback-" + str(bad),
        X,
        row_indices=np.arange(1, 97, dtype=np.int64),
        _row_path="mask",
    )
add("subnormal-alpha", base, _alphas=[float.fromhex("0x0.0000000000001p-1022")])
add(
    "negative-zero-penalty",
    base,
    penalty_factor=np.array([-0.0, 0.2, 1.0, 3.0, 0.5, 0.0, 2.0, 4.0, 0.1]),
    _alphas=[1.0, 0.5],
)
# Independently partitioned primitives, odd bool chunks and mid-byte slices.
columns = {}
for j in range(9):
    v = (
        base[:, j]
        if j % 3 == 0
        else (
            base[:, j] > 0 if j % 3 == 1 else np.rint(base[:, j] * 4).astype(np.int64)
        )
    )
    chunks = []
    lo = 0
    for size in [0, 1 + j, 0, 7, 9, 13, 0, 67, 200]:
        hi = min(97, lo + size)
        prefix = np.zeros(3, dtype=v.dtype)
        a = pa.array(np.concatenate([prefix, v[lo:hi]]))
        chunks.append(a.slice(3, hi - lo))
        lo = hi
        if lo == 97:
            break
    columns[f"x{j}"] = pa.chunked_array(chunks)
table = pa.table(columns)
add("arrow-independent", table)
add("arrow-mask", table, row_indices=np.arange(1, 95, dtype=np.int64), _row_path="mask")
add(
    "arrow-repeated",
    table,
    row_indices=np.array([0, 3, 0, 7, 12, 96, 7]),
    _row_path="gather",
)
add(
    "arrow-one-row",
    pa.table(
        {
            f"x{j}": pa.chunked_array([pa.array([v]) for v in (base[:, j] > 0)])
            for j in range(9)
        }
    ),
)
file = args.out / "input.npy"
np.save(file, base)
add("memmap", np.load(file, mmap_mode="r"))
from sklearn.datasets import load_breast_cancer

bc = load_breast_cancer()
add(
    "breast-cancer",
    np.asfortranarray(bc.data),
    bc.target.astype(float),
    lambdas=np.array([0.1, 0.03, 0.01]),
    max_iter=300,
)


def fingerprint(p):
    h = hashlib.sha256(b"vep-path-digest-v2\0")
    fields = {
        "coef.dense": p.coef.toarray(),
        "coef.format": p.coef.format,
        "coef.shape": p.coef.shape,
        "coef.data": p.coef.data,
        "coef.indices": p.coef.indices,
        "coef.indptr": p.coef.indptr,
        "intercept": p.intercept,
        "lambdas": p.lambdas,
        "deviance": p.deviance,
        "iterations": p.iterations,
        "n_rejected": p.n_rejected,
        "center": p.center,
        "scale": p.scale,
    }
    for name, value in fields.items():
        a = np.asarray(value)
        frame = json.dumps([name, a.dtype.str, a.shape]).encode()
        payload = a.tobytes()
        h.update(struct.pack("<QQ", len(frame), len(payload)))
        h.update(frame)
        h.update(payload)
    return h.hexdigest()


def trace_frames(file):
    raw = file.read_bytes()
    off = 0
    out = []
    while off < len(raw):
        start = off
        h = np.frombuffer(raw, dtype="<u8", count=6, offset=off)
        _, _, n, p, q, _ = map(int, h)
        off += 48
        off += 40
        off += n * 8
        score_start = off
        scores = raw[off : off + p * 8]
        off += p * 8
        score_end = off
        off += p * 8
        off += q * 16 + p * 8
        active = raw[off : off + p]
        off += p
        off += p
        off += p + p * 8 + p * 8 + 5 * p * 8 + p
        # Compare every recorded field, including cache metadata and epochs.
        # Only inactive stock-score slots may deliberately retain old values.
        out.append(
            (
                raw[start:score_start],
                raw[score_end:off],
                b"".join(
                    scores[j * 8 : (j + 1) * 8] for j, flag in enumerate(active) if flag
                ),
            )
        )
    assert off == len(raw)
    return out


results = []
thread_references = {}
# VEP_SCALAR is set before native library loading in a separate process.
for name, X, y, opts in cases:
    for alpha in opts.get("_alphas", [1.0, 0.5, 1e-8]):
        for threads in [1, 2, 4, 6]:
            for compatible, bounds in itertools.product([False, True], [True, False]):
                bm.settings = dataclasses.replace(bm.settings, boundscheck=bounds)
                config = dict(
                    lambdas=np.array([0.5, 0.3, 0.1, 0.03, 0.01]),
                    max_iter=100,
                    nlambda=5,
                    stop_on_saturation=False,
                )
                config.update({k: v for k, v in opts.items() if not k.startswith("_")})
                config.update(
                    alpha=alpha,
                    n_threads=threads,
                    r_compatible=compatible,
                    family="gaussian",
                )
                tag = f"{name}-a{alpha}-t{threads}-r{int(compatible)}-b{int(bounds)}"
                rowpath = opts.get("_row_path", "auto")

                # Force row-path selection through the existing PRIVATE native argument.
                def fit(
                    module, mode, tag=tag, rowpath=rowpath, X=X, y=y, config=config
                ):
                    print("VEP_CASE", tag, mode, file=sys.stderr, flush=True)
                    os.environ["VEP_SKIP"] = mode
                    bl._rust = module
                    sys.modules["pybiglasso._rust"] = module
                    original = module.fit_path
                    module.fit_path = lambda *a, **kw: original(
                        *a, **dict(kw, row_path=rowpath)
                    )
                    try:
                        with warnings.catch_warnings(record=True) as observed:
                            warnings.simplefilter("always")
                            try:
                                p = bl.biglasso(X, y, **config)
                                return (
                                    "ok",
                                    fingerprint(p),
                                    [
                                        (w.category.__name__, str(w.message))
                                        for w in observed
                                    ],
                                )
                            except Exception as e:
                                return ("error", type(e).__name__, str(e))
                    finally:
                        module.fit_path = original

                os.environ.pop("VEP_TRACE", None)
                reference = fit(stock, "off")
                key = (name, alpha, compatible, bounds)
                assert thread_references.setdefault(key, reference) == reference, (
                    "stock thread invariance",
                    tag,
                )
                # Full traces for one thread per case; other thread counts still check all
                # returned bits, shadow predicates and native epoch/threshold invariants.
                record = threads == 1
                outputs = {}
                for mode in ["shadow", "enabled"]:
                    trace = args.out / f"{tag}-{mode}.bin"
                    trace.unlink(missing_ok=True)
                    if record:
                        os.environ["VEP_TRACE"] = str(trace)
                    else:
                        os.environ.pop("VEP_TRACE", None)
                    actual = fit(native, mode)
                    assert actual == reference, (tag, mode, reference, actual)
                    outputs[mode] = actual
                if record and reference[0] == "ok":
                    assert trace_frames(args.out / f"{tag}-shadow.bin") == trace_frames(
                        args.out / f"{tag}-enabled.bin"
                    ), ("activation/residual trace mismatch", tag)
                results.append(
                    dict(
                        case=tag,
                        result=reference[0],
                        digest=reference[1],
                        row_path=rowpath,
                        trace_checked=record and reference[0] == "ok",
                    )
                )
                print(json.dumps(results[-1]), flush=True)
(args.out / "summary.json").write_text(
    json.dumps(
        dict(
            cases=len(cases),
            fits_checked=len(results),
            successful=sum(r["result"] == "ok" for r in results),
            errors=sum(r["result"] == "error" for r in results),
            trace_pairs=sum(r["trace_checked"] for r in results),
        ),
        indent=2,
    )
)
