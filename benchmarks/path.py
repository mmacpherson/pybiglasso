#!/usr/bin/env python3
"""Benchmark full lasso / elastic-net regression paths, in memory.

Every solver fits the same problem, (1/n) * NLL + lambda * (alpha * |b|_1 +
(1 - alpha) / 2 * |b|_2^2) on standardized features, along the same decreasing
lambda grid (R biglasso's default: 100 values), warm-starting from one lambda to
the next. Each solver runs at its default tolerance, as a user would run it,
and accuracy is reported alongside time: the worst relative suboptimality of
its objective, over the path, against the best solution any solver found.

Each (dataset, solver, threads) run happens in a fresh subprocess, with thread
counts set before numpy loads. Both families report native fit/path-call time;
the cap spends that same cumulative budget. Adapter setup and exports between
or after calls consume neither reported fit time nor the cap.

    uv run -p 3.14 --group bench python benchmarks/path.py --threads 1 --threads 12

With --family gaussian, compare Rust, R biglasso SSR, glmnet's native path,
scikit-learn coordinate descent, skglm, glum and statsmodels on RSS/(2n)
plus the same standardized penalty. Python competitors receive pre-standardized
Xs; pybiglasso and the R solvers standardize raw X themselves. The 100-lambda
grid is computed analytically; maximum KKT residuals and relative objective
differences versus glmnet accompany every completed path. An analytic
alpha/lambda map compensates for glmnet's internal response scaling.
Sklearn and skglm use their native path APIs; glum uses alpha_search, and
statsmodels OLS uses its supported start_params warm starts.

    uv run -p 3.14 --group bench python benchmarks/path.py --family gaussian

The Gaussian runner defaults to all seven solvers. Python competitors use a
10x relative cap (at least five seconds), with one measurement when a run is
slow. Override with --solver, --ratio-cap or --cap. Python 3.14 has wheels for
the bench group.
"""

import argparse
import functools
import hashlib
import json
import os
import pathlib
import platform
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time

HERE = pathlib.Path(__file__).parent

DATASETS = {
    # name: (nrow, ncol)
    "smoke-500-100": (500, 100),
    "tall-5k-1k": (5_000, 1_000),
    "tall-50k-2k": (50_000, 2_000),
    "wide-2k-20k": (2_000, 20_000),
    "wide-5k-50k": (5_000, 50_000),
}
SOLVERS = [
    "pybiglasso",
    "pybiglasso-tight",
    "pybiglasso-rcompat",
    "r-biglasso-ssr",
    "r-biglasso-adaptive",
    "r-glmnet",
    "glum",
    "skglm",
    "sklearn-liblinear",
    "sklearn-saga",
    "statsmodels",
]
TIMING_START = "--timing-start--"
TIMING_STOP = "--timing-stop--"


class FitClock:
    """One cumulative fit clock for both the reported time and the cap.

    Repeated fits may export snapshots between intervals. Neither those
    exports nor adapter setup consume fit time. Direct adapter calls are
    silent; subprocess runs emit boundaries for the parent's watchdog.
    """

    def __init__(self, signals=False):
        self.signals = signals
        self.seconds = 0.0

    def __enter__(self):
        if self.signals:
            print(TIMING_START, flush=True)
        self.started = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.seconds += time.perf_counter() - self.started
        if self.signals:
            print(f"{TIMING_STOP} {self.seconds:.17g}", flush=True)


# lasso-only solvers; R biglasso's Adaptive screening is lasso-only too
LASSO_ONLY = {"sklearn-liblinear", "statsmodels", "r-biglasso-adaptive"}
# R's elastic-net update; identical to pybiglasso's for the lasso
ENET_ONLY = {"pybiglasso-rcompat"}
# these run R in a container; their time is R's own system.time() of the fit
R_SOLVERS = {"r-biglasso-ssr", "r-biglasso-adaptive", "r-glmnet"}
# these don't use more than one thread; running them at more is redundant
SINGLE_THREADED = {
    "skglm",
    "sklearn-cd",
    "sklearn-liblinear",
    "sklearn-saga",
    "r-glmnet",
}
GAUSSIAN_SINGLE_THREADED = SINGLE_THREADED | {"statsmodels"}
GAUSSIAN_SOLVERS = [
    "pybiglasso",
    "r-biglasso-ssr",
    "r-glmnet",
    "glum",
    "skglm",
    "sklearn-cd",
    "statsmodels",
]
# statsmodels' L1 fit needs memory quadratic in the column count
STATSMODELS_MAX_GB = 8


# data -----------------------------------------------------------------


def make_data(name, seed=0, family="binomial"):
    import numpy as np

    import pybiglasso as bl

    nrow, ncol = DATASETS[name]
    d = bl.common.make_biglasso_dataset(
        nrow=nrow, ncol=ncol, seed=seed, marker_effect_sd=0.1
    )
    X, y = d["X"], d["y"]
    mu, sd = X.mean(axis=0), X.std(axis=0)
    keep = sd > 1e-6
    X = np.asfortranarray(X[:, keep])
    Xs = np.asfortranarray((X - mu[keep]) / sd[keep])
    if family == "gaussian":
        # Ordered arithmetic gives every child the same continuous response.
        y = np.full(nrow, 3.0)
        for j, weight in enumerate([1.5, -1.0, 0.75, 0.5, -0.25]):
            y += weight * Xs[:, j]
        y += np.random.default_rng(seed + 1).normal(scale=0.5, size=nrow)
    return X, Xs, y


def lambda_grid(X, y, alpha, family="binomial"):
    if family == "gaussian":
        import numpy as np

        # Mathematical input grid, not values generated by the port. All
        # solvers receive this grid explicitly, including R biglasso.
        Xs = (X - X.mean(axis=0)) / X.std(axis=0)
        maximum = np.max(np.abs(Xs.T @ (y - y.mean()))) / len(y) / alpha
        ratio = 0.001 if X.shape[0] > X.shape[1] else 0.05
        return np.exp(np.linspace(np.log(maximum), np.log(maximum * ratio), 100))
    import pybiglasso as bl

    nrow, ncol = X.shape
    path = bl.biglasso(
        X,
        y,
        alpha=alpha,
        nlambda=100,
        lambda_min=0.001 if nrow > ncol else 0.05,
        family="binomial",
    )
    return path.lambdas


# solvers: each returns (L, ncol) coefficients and (L,) intercepts, on the
# standardized scale ------------------------------------------------------


def fit_pybiglasso(
    X,
    Xs,
    y,
    lambdas,
    alpha,
    threads,
    eps=1e-7,
    r_compatible=False,
    family="binomial",
    *,
    clock=None,
):
    import numpy as np

    import pybiglasso as bl

    clock = clock or FitClock()
    with clock:
        path = bl.biglasso(
            X,
            y,
            alpha=alpha,
            lambdas=lambdas,
            n_threads=threads,
            eps=eps,
            r_compatible=r_compatible,
            family=family,
            backend="rust" if family == "gaussian" else None,
        )
    coef = path.coef.toarray().T * path.scale
    intercept = path.intercept + path.coef.toarray().T @ path.center
    L = len(lambdas)  # the path can stop early on saturation
    pad = np.full((L - len(path), coef.shape[1]), np.nan)
    out = (
        np.vstack([coef, pad]),
        np.concatenate([intercept, np.full(L - len(path), np.nan)]),
    )
    if family == "gaussian":
        return *out, clock.seconds, {"max_iterations": int(path.iterations.max())}
    return out


def fit_glum(X, Xs, y, lambdas, alpha, threads, family="binomial", *, clock=None):
    from glum import GeneralizedLinearRegressor

    # glum minimizes deviance/(2n) + lambda * the elastic-net penalty.
    # Normal deviance is RSS, so alpha=lambda and l1_ratio=alpha need no
    # response rescaling (glum's GeneralizedLinearRegressor objective).
    m = GeneralizedLinearRegressor(
        family="normal" if family == "gaussian" else family,
        alpha_search=True,
        # A list also works around glum 3.4.1's scalar alpha==0 check
        # during auto solver selection (an ndarray raises there).
        alpha=list(lambdas),
        l1_ratio=alpha,
        scale_predictors=False,
    )
    clock = clock or FitClock()
    with clock:
        m.fit(Xs, y)
    if family == "gaussian":
        return m.coef_path_, m.intercept_path_, clock.seconds
    return m.coef_path_, m.intercept_path_


def _warm_loop(model, Xs, y, lambdas, set_lambda, clock=None):
    import numpy as np

    coefs, intercepts = [], []
    clock = clock or FitClock()
    for lam in lambdas:
        set_lambda(model, lam)
        with clock:
            model.fit(Xs, y)
        coefs.append(np.ravel(model.coef_).copy())
        intercepts.append(float(np.ravel(model.intercept_)[0]))
    return np.array(coefs), np.array(intercepts)


def fit_skglm(X, Xs, y, lambdas, alpha, threads, family="binomial", *, clock=None):
    from skglm import ElasticNet, Lasso, SparseLogisticRegression

    # its penalty, alpha * (l1_ratio * |b|_1 + (1 - l1_ratio) / 2 * |b|_2^2),
    # is R's with lambda and alpha renamed
    # skglm's Gaussian loss is RSS/(2n), with the same penalty scaling.
    cls = SparseLogisticRegression if family == "binomial" else ElasticNet
    options = dict(l1_ratio=alpha)
    if family == "gaussian" and alpha == 1:
        cls, options = Lasso, {}
    m = cls(alpha=lambdas[0], fit_intercept=True, warm_start=True, **options)
    clock = clock or FitClock()
    if family == "gaussian":
        # The public path API validates and initializes the datafit once,
        # then warm-starts every lambda. With fit_intercept=True its last
        # coefficient row is the unpenalized intercept.
        with clock:
            _, coef, *_ = m.path(Xs, y, alphas=lambdas)
        return coef[:-1].T, coef[-1], clock.seconds
    return _warm_loop(
        m,
        Xs,
        y,
        lambdas,
        lambda m, lam: m.set_params(alpha=lam),
        clock=clock,
    )


def fit_sklearn_cd(X, Xs, y, lambdas, alpha, threads, family="gaussian", *, clock=None):
    import numpy as np
    from sklearn.linear_model import enet_path

    # RSS/(2n) + a*(l1_ratio*|b|_1 + (1-l1_ratio)*|b|_2^2/2):
    # a=lambda and l1_ratio=alpha, including the lasso at alpha=1.
    # enet_path does not fit an intercept. Prepare centered inputs once;
    # retain its default validation, tolerance and precompute='auto' so tall
    # data can use the native Gram path instead of 100 estimator fits.
    center, response_mean = Xs.mean(axis=0), y.mean()
    centered = np.asfortranarray(Xs - center)
    response = y - response_mean
    clock = clock or FitClock()
    with clock:
        _, coef, _ = enet_path(centered, response, alphas=lambdas, l1_ratio=alpha)
    intercept = response_mean - center @ coef
    return coef.T, intercept, clock.seconds


def _fit_sklearn(solver, Xs, y, lambdas, alpha, clock=None):
    from sklearn.linear_model import LogisticRegression

    n = len(y)
    kw = dict(solver=solver, warm_start=True, max_iter=10_000)
    if solver == "liblinear":
        # liblinear penalizes the intercept; a large scaling makes that negligible
        kw.update(penalty="l1", intercept_scaling=100.0)
    else:
        kw.update(l1_ratio=alpha)
    m = LogisticRegression(C=1.0, **kw)
    return _warm_loop(
        m, Xs, y, lambdas, lambda m, lam: m.set_params(C=1.0 / (n * lam)), clock
    )


def fit_sklearn_liblinear(X, Xs, y, lambdas, alpha, threads, *, clock=None):
    return _fit_sklearn("liblinear", Xs, y, lambdas, alpha, clock)


def fit_sklearn_saga(X, Xs, y, lambdas, alpha, threads, *, clock=None):
    return _fit_sklearn("saga", Xs, y, lambdas, alpha, clock)


def fit_statsmodels(
    X, Xs, y, lambdas, alpha, threads, family="binomial", *, clock=None
):
    import numpy as np
    import statsmodels.api as sm

    n, p = Xs.shape
    design = np.column_stack([np.ones(n), Xs])
    model = sm.OLS(y, design) if family == "gaussian" else sm.Logit(y, design)
    coefs, intercepts, start = [], [], None
    clock = clock or FitClock()
    # OLS has no public regularization-path API; start_params is its native
    # warm-start mechanism. Keep the previous full coefficient vector.
    for lam in lambdas:
        # statsmodels penalizes the summed log-likelihood, per coefficient;
        # n * lambda matches the mean-scaled objective. Intercept unpenalized.
        # OLS.fit_regularized instead uses RSS/(2n) + alpha * penalty:
        # weights=lambda, L1_wt=alpha (profile_scale=False, its default).
        # See statsmodels' OLS.fit_regularized objective; unlike Logit, no n.
        weights = np.full(p + 1, lam if family == "gaussian" else n * lam)
        weights[0] = 0.0
        options = (
            dict(method="elastic_net", L1_wt=alpha)
            if family == "gaussian"
            else dict(method="l1", disp=False)
        )
        with clock:
            res = model.fit_regularized(alpha=weights, start_params=start, **options)
        start = res.params
        coefs.append(res.params[1:].copy())
        intercepts.append(float(res.params[0]))
    out = np.array(coefs), np.array(intercepts)
    return (*out, clock.seconds) if family == "gaussian" else out


def statsmodels_memory_gb(ncol):
    """SLSQP, behind statsmodels' L1 fit, works with dense matrices over the
    2p + 1 split variables."""
    return (2 * ncol + 1) ** 2 * 8 / 1e9


def fit_pybiglasso_tight(X, Xs, y, lambdas, alpha, threads, *, clock=None):
    # R's default eps=1e-7 stops earlier than glum's default; this matches it
    return fit_pybiglasso(X, Xs, y, lambdas, alpha, threads, eps=1e-10, clock=clock)


def fit_pybiglasso_rcompat(X, Xs, y, lambdas, alpha, threads, *, clock=None):
    return fit_pybiglasso(
        X, Xs, y, lambdas, alpha, threads, r_compatible=True, clock=clock
    )


def _fit_r(X, Xs, y, lambdas, alpha, threads, screen, family="binomial", *, clock=None):
    """Fit in R biglasso; returns (coef, intercept, R's own fit seconds)."""
    import numpy as np

    sys.path.insert(0, str(HERE.parent / "tests"))
    from support import rcompare

    nrow, ncol = X.shape
    with tempfile.TemporaryDirectory(dir=os.environ.get("R_WORKDIR")) as d:
        d = pathlib.Path(d)
        np.asfortranarray(X, dtype=np.float64).ravel(order="F").tofile(d / "X.bin")
        np.asarray(y, dtype=np.float64).tofile(d / "y.bin")
        # R's default grid on unit penalty factors is exactly pybiglasso's
        params = dict(
            nrow=nrow,
            ncol=ncol,
            ny=nrow,
            alpha=alpha,
            nlambda=len(lambdas),
            lambda_min=float(lambdas[-1] / lambdas[0]),
            lambda_log_scale=True,
            eps=1e-7,
            max_iter=1000,
            screen=screen,
            ncores=threads,
            filebacked=False,
            family=family,
            timing_signals=bool(clock and clock.signals),
        )
        if family == "gaussian":
            params["lambdas"] = list(lambdas)
        (d / "params.json").write_text(json.dumps(params))
        (r,) = rcompare.run_r([d], env=r_thread_limits(threads))
    if "error" in r:
        raise RuntimeError(r["error"])
    beta, scale, center = r["beta"], r["scale"], r["center"]
    coef = (beta[1:] * scale[:, np.newaxis]).T
    intercept = beta[0] + center @ beta[1:]
    pad = len(lambdas) - coef.shape[0]
    coef = np.vstack([coef, np.full((pad, ncol), np.nan)])
    intercept = np.concatenate([intercept, np.full(pad, np.nan)])
    if family == "gaussian":
        return (
            coef,
            intercept,
            float(r["seconds"]),
            {
                "biglasso": r["biglasso_version"],
                "r": r["r_version"],
                "warnings": r["warnings"],
                "max_iterations": int(r["iter"].max()),
            },
        )
    return coef, intercept, float(r["seconds"])


def fit_r_ssr(X, Xs, y, lambdas, alpha, threads, family="binomial", *, clock=None):
    return _fit_r(X, Xs, y, lambdas, alpha, threads, "SSR", family=family, clock=clock)


def fit_r_adaptive(X, Xs, y, lambdas, alpha, threads, *, clock=None):
    return _fit_r(X, Xs, y, lambdas, alpha, threads, "Adaptive", clock=clock)


def r_thread_limits(threads):
    # Docker does not inherit the parent's limits. Keep BLAS sequential while
    # allowing biglasso's explicit ncores to control its OpenMP work.
    # Pin both families: even binomial glmnet's lognet wrapper wakes BLAS
    # workers through y %*% rep(1, nc) when preparing response weights.
    return {
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OMP_NUM_THREADS": str(threads),
    }


def fit_r_glmnet(
    X,
    Xs,
    y,
    lambdas,
    alpha,
    threads,
    family="binomial",
    *,
    generic=False,
    thresh=None,
    clock=None,
):
    """Fit in R's glmnet; returns (coef, intercept, R's own fit seconds,
    glmnet's version). glmnet gets raw X and standardizes it itself, as its
    users run it and as pybiglasso does, so its time includes that."""
    import numpy as np

    sys.path.insert(0, str(HERE.parent / "tests"))
    from support import rcompare

    def load(path):
        return json.loads(pathlib.Path(path).read_text())

    nrow, ncol = X.shape
    with tempfile.TemporaryDirectory(dir=os.environ.get("R_WORKDIR")) as d:
        d = pathlib.Path(d)
        np.asfortranarray(X, dtype=np.float64).ravel(order="F").tofile(d / "X.bin")
        np.asarray(y, dtype=np.float64).tofile(d / "y.bin")
        params = dict(
            nrow=nrow,
            ncol=ncol,
            alpha=alpha,
            lambdas=list(lambdas),
            family=family,
            generic=generic,
            thresh=thresh,
            timing_signals=bool(clock and clock.signals),
        )
        (d / "params.json").write_text(json.dumps(params))
        (r,) = rcompare.run_r(
            [d],
            script=HERE / "r" / "fit_glmnet.R",
            load=load,
            env=r_thread_limits(threads),
        )
    if "error" in r:
        raise RuntimeError(r["error"])
    coef = np.array(r["beta"], dtype=np.float64).reshape(r["beta_dim"], order="F").T
    intercept = np.atleast_1d(np.array(r["intercept"], dtype=np.float64))
    # glmnet reports X's own scale; the standardized one is the benchmark's
    intercept = intercept + coef @ X.mean(axis=0)
    coef = coef * X.std(axis=0)
    # glmnet fits every lambda of a supplied grid, but pad a short path (a
    # fit that stopped partway) so its objectives line up by lambda
    pad = len(lambdas) - coef.shape[0]
    coef = np.vstack([coef, np.full((pad, ncol), np.nan)])
    intercept = np.concatenate([intercept, np.full(pad, np.nan)])
    return (
        coef,
        intercept,
        float(r["seconds"]),
        {
            "glmnet": r["glmnet_version"],
            "r": r["r_version"],
            "glmnet_family": r["glmnet_family"],
            "glmnet_alpha": r["glmnet_alpha"],
            "glmnet_lambda_factor": r["glmnet_lambda_factor"],
            "response_scale": r["response_scale"],
            "warnings": list(np.atleast_1d(r.get("warnings") or [])),
            "jerr": r.get("jerr"),
        },
    )


FITTERS = {
    "pybiglasso": fit_pybiglasso,
    "pybiglasso-rcompat": fit_pybiglasso_rcompat,
    "r-biglasso-ssr": fit_r_ssr,
    "r-biglasso-adaptive": fit_r_adaptive,
    "r-glmnet": fit_r_glmnet,
    "pybiglasso-tight": fit_pybiglasso_tight,
    "glum": fit_glum,
    "skglm": fit_skglm,
    "sklearn-liblinear": fit_sklearn_liblinear,
    "sklearn-saga": fit_sklearn_saga,
    "sklearn-cd": fit_sklearn_cd,
    "statsmodels": fit_statsmodels,
}


# one run, in a subprocess ---------------------------------------------


def run_one(args):
    import warnings

    import numpy as np

    warnings.filterwarnings("ignore")
    X, Xs, y = make_data(args.dataset, family=args.family)
    lambdas = np.load(args.lambdas)
    fitter = FITTERS[args.solver]
    if args.family == "gaussian":
        fitter = functools.partial(fitter, family=args.family)
    if args.solver == "skglm":
        # This competitor still uses numba; pybiglasso has no JIT warm-up.
        fitter(X[:50, :20], Xs[:50, :20], y[:50], lambdas[:2], args.alpha, 1)
    clock = FitClock(signals=True)
    with warnings.catch_warnings(record=True) as observed:
        if args.family == "gaussian":
            warnings.simplefilter("always")
        out = fitter(X, Xs, y, lambdas, args.alpha, args.threads, clock=clock)
    # Both families use only their native fit intervals. In particular,
    # binomial Python timings no longer include setup and coefficient export.
    seconds = clock.seconds
    versions = {}
    if len(out) == 4:  # R's glmnet, which also reports its version
        coef, intercept, seconds, versions = out
    elif len(out) == 3:  # the solver timed its own fit (R)
        coef, intercept, seconds = out
    else:
        coef, intercept = out
    if args.family == "gaussian":
        versions["warnings"] = [
            *versions.get("warnings", []),
            *(str(w.message) for w in observed),
        ]
    np.savez(
        args.out,
        coef=coef,
        intercept=intercept,
        seconds=seconds,
        versions=json.dumps(versions),
        input_digest=input_digest(X, y),
    )


# orchestration ----------------------------------------------------------


def run_capped(cmd, env, cap):
    """Cap the same cumulative native-fit intervals the child reports.

    A stdout reader works on Windows pipes too. Setup, snapshot export and
    final serialization have no fit deadline. STOP carries the child's
    cumulative time, so repeated-fit paths spend one shared budget.
    """
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=os.name == "posix",
    )
    events = queue.Queue()

    def read_stdout():
        for line in proc.stdout:
            events.put(line.strip())
        events.put(None)

    threading.Thread(target=read_stdout, daemon=True).start()
    seconds, deadline, active, intervals = 0.0, None, False, 0

    def timed_out():
        try:
            if os.name == "posix":
                # Include local R/container clients in the termination signal.
                os.killpg(proc.pid, signal.SIGTERM)
            else:
                proc.kill()
        except ProcessLookupError:
            pass
        proc.wait()
        return False

    try:
        while True:
            try:
                line = events.get(
                    timeout=max(0, deadline - time.perf_counter()) if active else None
                )
            except queue.Empty:
                return timed_out()
            if line is None:
                break
            if line == TIMING_START:
                if active:
                    raise RuntimeError("nested fit timing intervals")
                active, deadline = True, time.perf_counter() + cap - seconds
                intervals += 1
            elif line.startswith(TIMING_STOP + " "):
                if not active:
                    raise RuntimeError("fit timing stop without start")
                elapsed = float(line.split()[1])
                if not seconds <= elapsed < float("inf"):
                    raise RuntimeError("invalid cumulative fit time")
                seconds, active, deadline = elapsed, False, None
                if seconds > cap:
                    return timed_out()
        proc.wait()
    finally:
        if proc.poll() is None:
            timed_out()
        proc.stdout.close()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    if active or not intervals:
        raise RuntimeError("child did not finish its fit timing intervals")
    return True


def objective(Xs, y, lam, alpha, coef, intercept, family="binomial"):
    import numpy as np

    eta = intercept + Xs @ coef
    loss = (
        np.mean((eta - y) ** 2) / 2
        if family == "gaussian"
        else (np.logaddexp(0.0, eta) - y * eta).mean()
    )
    return loss + lam * (alpha * np.abs(coef).sum() + (1 - alpha) / 2 * coef @ coef)


def gaussian_kkt(Xs, y, lambdas, alpha, coef, intercept):
    """First-principles mean-squared-loss subgradient, outside the timer."""
    import numpy as np

    residual = intercept + Xs @ coef.T - y[:, None]
    gradient = Xs.T @ residual / len(y)
    beta = coef.T
    l1 = lambdas * alpha
    gradient += lambdas * (1 - alpha) * beta
    gradient += np.where(beta != 0, l1 * np.sign(beta), np.clip(-gradient, -l1, l1))
    return float(max(np.max(np.abs(gradient)), np.max(np.abs(residual.mean(axis=0)))))


def input_digest(X, y):
    digest = hashlib.sha256(str((X.shape, str(X.dtype), str(y.dtype))).encode())
    digest.update(memoryview(X.ravel(order="F")).cast("B"))
    digest.update(memoryview(y).cast("B"))
    return digest.hexdigest()


def environment():
    import importlib.metadata as md

    versions = {}
    for pkg in (
        "pybiglasso",
        "numpy",
        "numba",
        "glum",
        "skglm",
        "scikit-learn",
        "statsmodels",
    ):
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    cpu = platform.processor()
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return dict(
        cpu=cpu,
        cores=os.cpu_count(),
        python=platform.python_version(),
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
        ).strip(),
        **versions,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", action="append", choices=list(DATASETS))
    ap.add_argument(
        "--solver",
        action="append",
        choices=list(dict.fromkeys(SOLVERS + GAUSSIAN_SOLVERS)),
    )
    ap.add_argument("--family", choices=["binomial", "gaussian"], default="binomial")
    ap.add_argument("--alpha", type=float, action="append", help="L1/L2 mix (R's)")
    ap.add_argument("--threads", type=int, action="append")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cap", type=float, default=900, help="seconds per run")
    ap.add_argument(
        "--ratio-cap",
        type=float,
        default=10,
        help="also cap each run at this multiple of pybiglasso's single-threaded "
        "time on the same problem; a timeout then means 'over this many times "
        "slower'",
    )
    ap.add_argument("--out")
    ap.add_argument("--run-one", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--lambdas", help=argparse.SUPPRESS)
    args, _ = ap.parse_known_args()

    if args.run_one:
        args.dataset, args.solver = args.dataset[0], args.solver[0]
        args.alpha, args.threads = args.alpha[0], args.threads[0]
        args.out = ap.parse_known_args()[1][0]
        return run_one(args)

    import numpy as np

    datasets = args.dataset or list(DATASETS)
    solvers = args.solver or (
        GAUSSIAN_SOLVERS if args.family == "gaussian" else SOLVERS
    )
    supported = GAUSSIAN_SOLVERS if args.family == "gaussian" else SOLVERS
    if any(s not in supported for s in solvers):
        ap.error(f"{args.family} benchmarks support {', '.join(supported)}.")
    single_threaded = (
        GAUSSIAN_SINGLE_THREADED if args.family == "gaussian" else SINGLE_THREADED
    )
    alphas = args.alpha or [1.0]
    threads = args.threads or [1]
    filename = "path-gaussian.jsonl" if args.family == "gaussian" else "path.jsonl"
    out = pathlib.Path(args.out or HERE / "results" / filename)
    out.parent.mkdir(parents=True, exist_ok=True)
    env_info = environment()

    for dataset in datasets:
        X, Xs, y = make_data(dataset, family=args.family)
        digest = input_digest(X, y)
        for alpha in alphas:
            lambdas = lambda_grid(X, y, alpha, family=args.family)
            runs = {}
            run_cap = args.cap
            with tempfile.TemporaryDirectory() as tmp:
                np.save(f"{tmp}/lambdas.npy", lambdas)
                for solver in solvers:
                    if args.family == "binomial" and alpha < 1 and solver in LASSO_ONLY:
                        continue
                    if alpha == 1 and solver in ENET_ONLY:
                        continue
                    if (
                        solver == "statsmodels"
                        # Only Logit's SLSQP needs quadratic workspace;
                        # Gaussian OLS uses coordinate descent.
                        and args.family == "binomial"
                        and statsmodels_memory_gb(X.shape[1]) > STATSMODELS_MAX_GB
                    ):
                        print(
                            f"{dataset} alpha={alpha} statsmodels: skipped, needs "
                            f"~{statsmodels_memory_gb(X.shape[1]):.0f} GB",
                            flush=True,
                        )
                        continue
                    for t in threads:
                        if solver in single_threaded and t != min(threads):
                            continue
                        times, result, versions = [], None, {}
                        for rep in range(args.repeats):
                            res = f"{tmp}/{solver}-{t}-{rep}.npz"
                            env = dict(os.environ)
                            for var in (
                                "OMP_NUM_THREADS",
                                "OPENBLAS_NUM_THREADS",
                                "MKL_NUM_THREADS",
                                "NUMBA_NUM_THREADS",
                            ):
                                env[var] = str(t)
                            cmd = [
                                sys.executable,
                                __file__,
                                "--run-one",
                                "--dataset", dataset,
                                "--solver", solver,
                                "--family", args.family,
                                "--alpha", str(alpha),
                                "--threads", str(t),
                                "--lambdas", f"{tmp}/lambdas.npy",
                                res,
                            ]  # fmt: skip
                            # R's clock covers container start and data
                            # loading, which its reported time excludes
                            cap = args.cap if solver in R_SOLVERS else run_cap
                            if not run_capped(cmd, env, cap):
                                times.append(float("inf"))
                                break
                            # Windows cannot remove the temporary directory
                            # while a result archive is still open.
                            with np.load(res) as r:
                                assert str(r["input_digest"]) == digest, (
                                    "child fit inputs differ"
                                )
                                times.append(float(r["seconds"]))
                                result = (r["coef"], r["intercept"])
                                versions = json.loads(str(r["versions"]))
                            if times[-1] > run_cap / 3:
                                break  # slow; one measurement will do
                        runs[(solver, t)] = (times, result, versions)
                        if solver == "pybiglasso" and t == 1 and np.isfinite(times[0]):
                            # a floor only lengthens the cap, so a timeout still
                            # means "over ratio_cap times slower"
                            run_cap = min(
                                args.cap,
                                max(5.0, args.ratio_cap * float(np.median(times))),
                            )
                        print(
                            f"{dataset} alpha={alpha} {solver} threads={t}: "
                            + ", ".join(f"{s:.2f}s" for s in times),
                            flush=True,
                        )

            # accuracy: objective at each lambda vs the best any solver found
            objs = {}
            for key, (_times, result, _versions) in runs.items():
                if result is None:
                    continue
                coef, intercept = result
                objs[key] = np.array(
                    [
                        objective(Xs, y, lam, alpha, coef[k], intercept[k], args.family)
                        if np.isfinite(intercept[k])
                        else np.nan
                        for k, lam in enumerate(lambdas)
                    ]
                )
            best = (
                np.nanmin(np.vstack(list(objs.values())), axis=0)
                if args.family == "binomial"
                else None
            )
            reference = objs.get(("r-glmnet", min(threads)))

            with open(out, "a") as fp:
                for (solver, t), (times, result, versions) in runs.items():
                    subopt = fitted = None
                    kkt = difference = None
                    if (solver, t) in objs:
                        fitted = int(np.isfinite(objs[(solver, t)]).sum())
                        if best is not None:
                            rel = (objs[(solver, t)] - best) / np.abs(best)
                            subopt = float(np.nanmax(rel))
                        if args.family == "gaussian":
                            kkt = gaussian_kkt(Xs, y, lambdas, alpha, *result)
                            if reference is not None:
                                difference = float(
                                    np.nanmax(
                                        np.abs(objs[(solver, t)] - reference)
                                        / np.abs(reference)
                                    )
                                )
                    record = dict(
                        dataset=dataset,
                        family=args.family,
                        input_digest=digest,
                        grid_source="analytic standardized score"
                        if args.family == "gaussian"
                        else "pybiglasso",
                        lambda_max=float(lambdas[0]),
                        lambda_min=float(lambdas[-1]),
                        ncol_used=X.shape[1],
                        nrow=DATASETS[dataset][0],
                        ncol=DATASETS[dataset][1],
                        alpha=alpha,
                        nlambda=len(lambdas),
                        solver=solver,
                        threads=t,
                        seconds=float(np.median(times)) if times else None,
                        times=times,
                        timed_out=bool(times) and not np.isfinite(times[-1]),
                        cap=args.cap
                        if solver == "pybiglasso" or solver in R_SOLVERS
                        else run_cap,
                        timing_scope="native fit intervals",
                        ratio_cap=args.ratio_cap,
                        max_rel_suboptimality=subopt,
                        max_kkt_residual=kkt,
                        max_kkt_residual_over_min_l1=kkt / (lambdas[-1] * alpha)
                        if kkt is not None
                        else None,
                        max_rel_objective_difference_vs_glmnet=difference,
                        fitted_lambdas=fitted,
                        **env_info,
                        **versions,
                    )
                    if solver in R_SOLVERS:
                        record["r_thread_limits"] = r_thread_limits(t)
                    fp.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
