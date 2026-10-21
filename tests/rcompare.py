"""Compare pybiglasso fits against R biglasso.

A `Case` fully determines a synthetic dataset and the fit options. For each case
we write the data to a directory, fit it in Python, and fit it in R via
`tests/r/fit_biglasso.R`.

Where pybiglasso's default lambda grid intentionally differs from R's (any
non-unit penalty factor), both sides are fit on pybiglasso's grid passed as
user-supplied lambdas, so the two fits follow the same code path and must agree
iteration for iteration.

R is found as `Rscript` on PATH (override with RSCRIPT), or, if CONTAINER is set
to `docker` or `podman`, run inside the image named by R_IMAGE.
"""

import dataclasses
import gzip
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import time

import numpy as np

import pybiglasso as bl

TESTS_DIR = pathlib.Path(__file__).parent
R_SCRIPT = TESTS_DIR / "r" / "fit_biglasso.R"
FIXTURE_PATH = TESTS_DIR / "fixtures" / "r_biglasso.json.gz"

RTOL, ATOL = 1e-9, 1e-11
EPS, MAX_ITER = 1e-7, 1000


@dataclasses.dataclass(frozen=True)
class Case:
    nrow: int
    ncol: int
    seed: int = 123
    alpha: float = 1.0
    nlambda: int = 20
    lambda_min: float | None = None  # None: R's default rule, on nrow(X) vs ncol(X)
    lambda_log_scale: bool = True
    data: str = "normal"  # "normal" | "dosage"
    frac_mono: float = 0.0
    penalty: str = "none"  # "none" | "varied" | "zeros"
    rows: str = "all"  # "all" | "half"
    dtype: str = "float64"
    memmap: bool = False
    user_lambdas: bool = False
    dfmax: int | None = None
    filebacked: bool = False  # R side: file-backed big.matrix
    tag: str = ""

    @property
    def name(self):
        parts = [f"{self.data}-{self.nrow}x{self.ncol}", f"s{self.seed}"]
        parts.append(f"a{self.alpha}")
        parts.append(f"L{self.nlambda}")
        for field, default in [
            ("frac_mono", 0.0),
            ("penalty", "none"),
            ("rows", "all"),
            ("dtype", "float64"),
            ("dfmax", None),
        ]:
            value = getattr(self, field)
            if value != default:
                parts.append(f"{field}={value}")
        for flag in ("memmap", "user_lambdas", "filebacked"):
            if getattr(self, flag):
                parts.append(flag)
        if not self.lambda_log_scale:
            parts.append("linear")
        if self.tag:
            parts.append(self.tag)
        return "_".join(parts)

    @property
    def resolved_lambda_min(self):
        if self.lambda_min is not None:
            return self.lambda_min
        return 0.001 if self.nrow > self.ncol else 0.05

    @property
    def r_needs_lambdas(self):
        return self.user_lambdas or self.penalty != "none"


def fast_grid():
    """Cases small enough to run on every push, against stored R output."""
    cases = []

    # -- core grid: shape x regularization x monomorphic columns
    for seed in (123, 456, 789):
        for nrow, ncol in ((7, 3), (100, 10), (300, 40), (60, 200)):
            for alpha in (0.1, 0.5, 0.9, 1.0):
                for frac_mono in (0.0, 0.5):
                    cases.append(Case(nrow, ncol, seed, alpha, 20, frac_mono=frac_mono))
    for alpha in (0.1, 0.5, 0.9, 1.0):
        cases.append(Case(100, 10, 123, alpha, 5))

    # -- one feature at a time, on a tall and a wide problem
    for nrow, ncol in ((200, 30), (80, 150)):
        base = dict(nrow=nrow, ncol=ncol, alpha=0.8)
        cases += [
            Case(**base, data="dosage"),
            Case(nrow, ncol, seed=7, alpha=1.0, data="dosage"),
            Case(**base, penalty="varied"),
            Case(**base, penalty="zeros"),
            Case(**base, rows="half"),
            Case(**base, dtype="float32"),
            Case(**base, memmap=True),
            Case(**base, user_lambdas=True),
            Case(**base, lambda_log_scale=False),
            Case(**base, dfmax=5),
            Case(**base, data="dosage", penalty="zeros", rows="half", memmap=True),
        ]

    # -- strongly separable data, so the path saturates before the end
    cases.append(Case(30, 60, 11, 1.0, 50, lambda_min=1e-4, tag="saturating"))

    return cases


def soak_grid(scale=None):
    """Larger problems, meant to be run locally; scale rows with SOAK_SCALE."""
    scale = float(os.environ.get("SOAK_SCALE", 1.0) if scale is None else scale)
    cases = []
    for nrow, ncol in ((20_000, 500), (2_000, 20_000), (5_000, 5_000)):
        for seed in (1, 2):
            for alpha in (0.5, 1.0):
                cases.append(
                    Case(
                        round(nrow * scale),
                        ncol,
                        seed,
                        alpha,
                        50,
                        data="dosage",
                        memmap=True,
                        filebacked=True,
                    )
                )
    return cases


# -- data -----------------------------------------------------------------


def make_case_data(case, workdir):
    """Write the case's design to `workdir/X.bin` and return the fit inputs."""
    workdir = pathlib.Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(case.seed + 1)
    xpath = workdir / "X.bin"

    # -- X.bin is raw column-major float64, readable by R and by np.memmap
    Xbin = np.memmap(
        xpath, dtype=np.float64, mode="w+", shape=(case.nrow, case.ncol), order="F"
    )
    if case.data == "normal":
        d = bl.common.generate_test_data(case.nrow, case.ncol, seed=case.seed)
        Xbin[:] = d["X"]
        y = d["y"]
    elif case.data == "dosage":
        weights = bl.common.sample_biglasso_weights(
            np.random.default_rng(case.seed), case.ncol
        )
        gen = bl.common.biglasso_generator(
            np.random.default_rng(case.seed + 2), weights, nrow=min(case.nrow, 10_000)
        )
        ys, start = [], 0
        while start < case.nrow:
            batch = next(gen)
            stop = min(case.nrow, start + len(batch["y"]))
            Xbin[start:stop] = batch["X"][: stop - start]
            ys.append(batch["y"][: stop - start])
            start = stop
        y = np.concatenate(ys)
    else:
        raise ValueError(case.data)

    if case.frac_mono > 0:
        nmono = round(case.ncol * case.frac_mono)
        Xbin[:, np.sort(rng.choice(case.ncol, nmono, replace=False))] = 1.0

    if case.dtype == "float32":
        # -- round through float32 so that R sees exactly the values we fit
        Xbin[:] = Xbin.astype(np.float32)
    Xbin.flush()

    penalty_factor = None
    if case.penalty == "varied":
        penalty_factor = rng.uniform(0.5, 2.0, case.ncol)
    elif case.penalty == "zeros":
        penalty_factor = np.ones(case.ncol)
        penalty_factor[: min(3, case.ncol - 1)] = 0.0

    row_indices = None
    if case.rows == "half":
        row_indices = np.flip(np.arange(case.nrow)[::2])

    if case.memmap:
        X = np.memmap(
            xpath, dtype=np.float64, mode="r", shape=(case.nrow, case.ncol), order="F"
        )
    else:
        X = np.asfortranarray(np.array(Xbin, dtype=case.dtype))
    del Xbin

    return dict(
        X=X,
        y=y,
        penalty_factor=penalty_factor,
        row_indices=row_indices,
    )


def data_digest(data):
    """Hash of the generated inputs, to detect stale fixtures."""
    h = hashlib.sha256()
    h.update(np.asarray(data["X"], dtype=np.float64).tobytes(order="F"))
    h.update(np.asarray(data["y"], dtype=np.float64).tobytes())
    for extra in (data["penalty_factor"], data["row_indices"]):
        h.update(b"|" if extra is None else np.asarray(extra, np.float64).tobytes())
    return h.hexdigest()


# -- fitting --------------------------------------------------------------


def fit_py(case, data, lambdas=None):
    """Fit pybiglasso and return results in R biglasso's layout."""
    t0 = time.perf_counter()
    fit = bl.biglasso(
        data["X"],
        data["y"],
        alpha=case.alpha,
        nlambda=case.nlambda,
        lambdas=lambdas,
        eps=EPS,
        lambda_min=case.resolved_lambda_min,
        lambda_log_scale=case.lambda_log_scale,
        penalty_factor=data["penalty_factor"],
        row_indices=data["row_indices"],
        dfmax=case.dfmax,
        max_iter=MAX_ITER,
        r_compatible=True,
    )
    seconds = time.perf_counter() - t0
    return dict(
        center=fit.center,
        scale=fit.scale,
        lambda_=fit.lambdas,
        # -- R's `loss` is the negative log-likelihood, half the deviance
        loss=fit.deviance / 2,
        iter=fit.iterations,
        rejections=fit.n_rejected,
        beta=np.vstack([fit.intercept, fit.coef.toarray()]),
        seconds=seconds,
    )


def prepare_case(case, workdir, lambdas=None):
    """Write a case's inputs for R and fit it in Python.

    When the case gives R a lambda grid, it is derived from pybiglasso's default
    grid, unless `lambdas` (e.g. the grid a stored R fit was given) is passed.

    Returns (python results, data digest, lambdas given to R or None).
    """
    workdir = pathlib.Path(workdir)
    data = make_case_data(case, workdir)

    if not case.r_needs_lambdas:
        lambdas = None
    elif lambdas is not None:
        lambdas = np.asarray(lambdas, dtype=np.float64)
    else:
        grid = fit_py(case, data)["lambda_"]
        if case.user_lambdas:
            lambdas = grid * 0.9
        else:
            # -- grid[0] is lambda_max, where the leading column sits exactly on
            #    the KKT boundary; nudge it so last-bit differences in z between
            #    R and numba can't flip that check and shift iteration counts
            lambdas = grid.copy()
            lambdas[0] *= 1 + 1e-9
    py = fit_py(case, data, lambdas)

    y = data["y"]
    row_idx = None
    if data["row_indices"] is not None:
        row_idx = (data["row_indices"] + 1).tolist()  # R is 1-based
    np.asarray(y, dtype=np.float64).tofile(workdir / "y.bin")

    params = dict(
        nrow=case.nrow,
        ncol=case.ncol,
        ny=len(y),
        alpha=case.alpha,
        nlambda=case.nlambda,
        lambda_min=case.resolved_lambda_min,
        lambda_log_scale=case.lambda_log_scale,
        eps=EPS,
        max_iter=MAX_ITER,
        lambdas=None if lambdas is None else lambdas.tolist(),
        penalty_factor=(
            None if data["penalty_factor"] is None else data["penalty_factor"].tolist()
        ),
        row_idx=row_idx,
        dfmax=case.dfmax,
        filebacked=case.filebacked,
    )
    params = {k: v for k, v in params.items() if v is not None}
    (workdir / "params.json").write_text(json.dumps(params))

    digest = None if case.memmap and case.filebacked else data_digest(data)
    return py, digest, lambdas


def r_available():
    container = os.environ.get("CONTAINER")
    if container:
        return shutil.which(container) is not None
    return shutil.which(os.environ.get("RSCRIPT", "Rscript")) is not None


def run_r(case_dirs):
    """Fit every case directory in a single R session; return parsed outputs."""
    case_dirs = [pathlib.Path(d).absolute() for d in case_dirs]
    container = os.environ.get("CONTAINER")
    if container:
        image = os.environ.get("R_IMAGE", "pybiglasso-r:1.4.1")
        cmd = [container, "run", "--rm"]
        if pathlib.Path(container).name == "docker":
            cmd += ["--user", f"{os.getuid()}:{os.getgid()}"]
        for mount in sorted({R_SCRIPT.parent, *{d.parent for d in case_dirs}}):
            cmd += ["-v", f"{mount}:{mount}"]
        cmd += [image, "Rscript"]
    else:
        cmd = [os.environ.get("RSCRIPT", "Rscript")]
    subprocess.run([*cmd, str(R_SCRIPT), *map(str, case_dirs)], check=True)
    return [load_r_output(d / "out.json") for d in case_dirs]


def load_r_output(path):
    raw = json.loads(pathlib.Path(path).read_text())
    if "error" in raw:
        return {"error": raw["error"]}
    beta = np.array(raw["beta"], dtype=np.float64)
    out = {
        k: np.atleast_1d(np.array(raw[k], dtype=np.float64))
        for k in ("center", "scale", "lambda", "loss", "iter", "rejections")
    }
    out["lambda_"] = out.pop("lambda")
    # -- R truncates the path when it stops early, except for `rejections`
    out["rejections"] = out["rejections"][: len(out["lambda_"])]
    out["beta"] = beta.reshape(raw["beta_dim"], order="F")
    out["warnings"] = list(np.atleast_1d(raw.get("warnings") or []))
    out["biglasso_version"] = raw["biglasso_version"]
    out["r_version"] = raw["r_version"]
    out["seconds"] = raw.get("seconds")
    return out


# -- comparison -----------------------------------------------------------


def compare(py, r):
    """Return a list of human-readable disagreements (empty if they agree)."""
    if "error" in r:
        return [f"R failed: {r['error']}"]
    problems = []

    for key in ("center", "scale", "lambda_", "loss", "beta"):
        a, b = np.asarray(py[key]), np.asarray(r[key])
        if a.shape != b.shape:
            problems.append(f"{key}: shape {a.shape} (py) vs {b.shape} (R)")
        elif not np.allclose(a, b, rtol=RTOL, atol=ATOL):
            err = np.abs(a - b)
            worst = np.unravel_index(np.argmax(err - RTOL * np.abs(b)), err.shape)
            problems.append(
                f"{key}: max abs diff {err.max():.3g} "
                f"(at {tuple(int(i) for i in worst)}: py {a[worst]:.10g}, R {b[worst]:.10g})"
            )

    for key in ("iter", "rejections"):
        a, b = np.asarray(py[key]), np.asarray(r[key])
        if a.shape != b.shape or not np.array_equal(a, b):
            problems.append(f"{key}: py {a.tolist()} vs R {b.tolist()}")

    return problems


# -- fixtures -------------------------------------------------------------


def save_fixtures(entries, path=FIXTURE_PATH):
    """`entries` maps case name -> (digest, lambdas given to R, R output)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for name, (digest, lambdas, r) in sorted(entries.items()):
        payload[name] = dict(
            digest=digest,
            r_input_lambdas=None if lambdas is None else lambdas.tolist(),
            **{
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in r.items()
                if k != "seconds"
            },
        )
    with gzip.open(path, "wt") as fp:
        json.dump(payload, fp, separators=(",", ":"))


def load_fixtures(path=FIXTURE_PATH):
    with gzip.open(path, "rt") as fp:
        payload = json.load(fp)
    out = {}
    for name, entry in payload.items():
        r = {
            k: np.array(entry[k], dtype=np.float64)
            for k in ("center", "scale", "lambda_", "loss", "iter", "rejections")
        }
        r["beta"] = np.array(entry["beta"], dtype=np.float64)
        for k in ("warnings", "biglasso_version", "r_version"):
            r[k] = entry[k]
        out[name] = (entry["digest"], entry["r_input_lambdas"], r)
    return out
