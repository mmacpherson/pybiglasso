#!/usr/bin/env python3
"""Benchmark full lasso / elastic-net logistic regression paths, in memory.

Every solver fits the same problem, (1/n) * NLL + lambda * (alpha * |b|_1 +
(1 - alpha) / 2 * |b|_2^2) on standardized features, along the same decreasing
lambda grid (R biglasso's default: 100 values), warm-starting from one lambda to
the next. Each solver runs at its default tolerance, as a user would run it,
and accuracy is reported alongside time: the worst relative suboptimality of
its objective, over the path, against the best solution any solver found.

Each (dataset, solver, threads) run happens in a fresh subprocess, with thread
counts set before numpy loads and a hard time cap.

    uv run --group bench benchmarks/path.py --threads 1 --threads 12
"""

import argparse
import json
import os
import pathlib
import platform
import subprocess
import sys
import tempfile
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
    "glum",
    "skglm",
    "sklearn-liblinear",
    "sklearn-saga",
    "statsmodels",
]
TIMING_START = "--timing-start--"
# lasso-only solvers; R biglasso's Adaptive screening is lasso-only too
LASSO_ONLY = {"skglm", "sklearn-liblinear", "statsmodels", "r-biglasso-adaptive"}
# R's elastic-net update; identical to pybiglasso's for the lasso
ENET_ONLY = {"pybiglasso-rcompat"}
# these run R in a container; their time is R's own system.time() of the fit
R_SOLVERS = {"r-biglasso-ssr", "r-biglasso-adaptive"}
# these don't use more than one thread; running them at more is redundant
SINGLE_THREADED = {"skglm", "sklearn-liblinear", "sklearn-saga"}
# statsmodels' L1 fit needs memory quadratic in the column count
STATSMODELS_MAX_GB = 8


# data -----------------------------------------------------------------


def make_data(name, seed=0):
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
    return X, Xs, y


def lambda_grid(X, y, alpha):
    import pybiglasso as bl

    nrow, ncol = X.shape
    path = bl.biglasso(
        X, y, alpha=alpha, nlambda=100, lambda_min=0.001 if nrow > ncol else 0.05
    )
    return path.lambdas


# solvers: each returns (L, ncol) coefficients and (L,) intercepts, on the
# standardized scale ------------------------------------------------------


def fit_pybiglasso(X, Xs, y, lambdas, alpha, threads, eps=1e-7, r_compatible=False):
    import numpy as np

    import pybiglasso as bl

    path = bl.biglasso(
        X,
        y,
        alpha=alpha,
        lambdas=lambdas,
        n_threads=threads,
        eps=eps,
        r_compatible=r_compatible,
    )
    coef = path.coef.toarray().T * path.scale
    intercept = path.intercept + path.coef.toarray().T @ path.center
    L = len(lambdas)  # the path can stop early on saturation
    pad = np.full((L - len(path), coef.shape[1]), np.nan)
    return np.vstack([coef, pad]), np.concatenate(
        [intercept, np.full(L - len(path), np.nan)]
    )


def fit_glum(X, Xs, y, lambdas, alpha, threads):
    from glum import GeneralizedLinearRegressor

    m = GeneralizedLinearRegressor(
        family="binomial",
        alpha_search=True,
        alphas=lambdas,
        l1_ratio=alpha,
        scale_predictors=False,
    ).fit(Xs, y)
    return m.coef_path_, m.intercept_path_


def _warm_loop(model, Xs, y, lambdas, set_lambda):
    import numpy as np

    coefs, intercepts = [], []
    for lam in lambdas:
        set_lambda(model, lam)
        model.fit(Xs, y)
        coefs.append(np.ravel(model.coef_).copy())
        intercepts.append(float(np.ravel(model.intercept_)[0]))
    return np.array(coefs), np.array(intercepts)


def fit_skglm(X, Xs, y, lambdas, alpha, threads):
    from skglm import SparseLogisticRegression

    m = SparseLogisticRegression(alpha=lambdas[0], fit_intercept=True, warm_start=True)
    return _warm_loop(m, Xs, y, lambdas, lambda m, lam: m.set_params(alpha=lam))


def _fit_sklearn(solver, Xs, y, lambdas, alpha):
    from sklearn.linear_model import LogisticRegression

    n = len(y)
    kw = dict(solver=solver, warm_start=True, max_iter=10_000)
    if solver == "liblinear":
        # liblinear penalizes the intercept; a large scaling makes that negligible
        kw.update(penalty="l1", intercept_scaling=100.0)
    else:
        kw.update(l1_ratio=alpha)
    m = LogisticRegression(C=1.0, **kw)
    return _warm_loop(m, Xs, y, lambdas, lambda m, lam: m.set_params(C=1.0 / (n * lam)))


def fit_sklearn_liblinear(X, Xs, y, lambdas, alpha, threads):
    return _fit_sklearn("liblinear", Xs, y, lambdas, alpha)


def fit_sklearn_saga(X, Xs, y, lambdas, alpha, threads):
    return _fit_sklearn("saga", Xs, y, lambdas, alpha)


def fit_statsmodels(X, Xs, y, lambdas, alpha, threads):
    import numpy as np
    import statsmodels.api as sm

    n, p = Xs.shape
    model = sm.Logit(y, np.column_stack([np.ones(n), Xs]))
    coefs, intercepts, start = [], [], None
    for lam in lambdas:
        # statsmodels penalizes the summed log-likelihood, per coefficient;
        # n * lambda matches the mean-scaled objective. Intercept unpenalized.
        weights = np.full(p + 1, n * lam)
        weights[0] = 0.0
        res = model.fit_regularized(
            method="l1", alpha=weights, start_params=start, disp=False
        )
        start = res.params
        coefs.append(res.params[1:].copy())
        intercepts.append(float(res.params[0]))
    return np.array(coefs), np.array(intercepts)


def statsmodels_memory_gb(ncol):
    """SLSQP, behind statsmodels' L1 fit, works with dense matrices over the
    2p + 1 split variables."""
    return (2 * ncol + 1) ** 2 * 8 / 1e9


def fit_pybiglasso_tight(X, Xs, y, lambdas, alpha, threads):
    # R's default eps=1e-7 stops earlier than glum's default; this matches it
    return fit_pybiglasso(X, Xs, y, lambdas, alpha, threads, eps=1e-10)


def fit_pybiglasso_rcompat(X, Xs, y, lambdas, alpha, threads):
    return fit_pybiglasso(X, Xs, y, lambdas, alpha, threads, r_compatible=True)


def _fit_r(X, Xs, y, lambdas, alpha, threads, screen):
    """Fit in R biglasso; returns (coef, intercept, R's own fit seconds)."""
    import numpy as np

    sys.path.insert(0, str(HERE.parent / "tests"))
    import rcompare

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
        )
        (d / "params.json").write_text(json.dumps(params))
        (r,) = rcompare.run_r([d])
    if "error" in r:
        raise RuntimeError(r["error"])
    beta, scale, center = r["beta"], r["scale"], r["center"]
    coef = (beta[1:] * scale[:, np.newaxis]).T
    intercept = beta[0] + center @ beta[1:]
    pad = len(lambdas) - coef.shape[0]
    coef = np.vstack([coef, np.full((pad, ncol), np.nan)])
    intercept = np.concatenate([intercept, np.full(pad, np.nan)])
    return coef, intercept, float(r["seconds"])


def fit_r_ssr(X, Xs, y, lambdas, alpha, threads):
    return _fit_r(X, Xs, y, lambdas, alpha, threads, "SSR")


def fit_r_adaptive(X, Xs, y, lambdas, alpha, threads):
    return _fit_r(X, Xs, y, lambdas, alpha, threads, "Adaptive")


FITTERS = {
    "pybiglasso": fit_pybiglasso,
    "pybiglasso-rcompat": fit_pybiglasso_rcompat,
    "r-biglasso-ssr": fit_r_ssr,
    "r-biglasso-adaptive": fit_r_adaptive,
    "pybiglasso-tight": fit_pybiglasso_tight,
    "glum": fit_glum,
    "skglm": fit_skglm,
    "sklearn-liblinear": fit_sklearn_liblinear,
    "sklearn-saga": fit_sklearn_saga,
    "statsmodels": fit_statsmodels,
}


# one run, in a subprocess ---------------------------------------------


def run_one(args):
    import warnings

    import numpy as np

    warnings.filterwarnings("ignore")
    X, Xs, y = make_data(args.dataset)
    lambdas = np.load(args.lambdas)
    if args.solver in ("pybiglasso", "pybiglasso-tight", "skglm"):
        # numba-based solvers: compile (or load cached kernels) before timing
        FITTERS[args.solver](
            X[:50, :20], Xs[:50, :20], y[:50], lambdas[:2], args.alpha, 1
        )
    print(TIMING_START, flush=True)  # the parent's time cap starts here
    t0 = time.perf_counter()
    out = FITTERS[args.solver](X, Xs, y, lambdas, args.alpha, args.threads)
    seconds = time.perf_counter() - t0
    if len(out) == 3:  # the solver timed its own fit (R)
        coef, intercept, seconds = out
    else:
        coef, intercept = out
    np.savez(args.out, coef=coef, intercept=intercept, seconds=seconds)


# orchestration ----------------------------------------------------------


def run_capped(cmd, env, cap):
    """Run a child; the cap counts from when it starts timing its fit, so data
    generation and imports are excluded. Returns False on timeout."""
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, text=True)
    for line in proc.stdout:
        if line.strip() == TIMING_START:
            break
    try:
        proc.wait(timeout=cap)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return False
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return True


def objective(Xs, y, lam, alpha, coef, intercept):
    import numpy as np

    eta = intercept + Xs @ coef
    nll = (np.logaddexp(0.0, eta) - y * eta).mean()
    return nll + lam * (alpha * np.abs(coef).sum() + (1 - alpha) / 2 * coef @ coef)


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
        **versions,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", action="append", choices=list(DATASETS))
    ap.add_argument("--solver", action="append", choices=SOLVERS)
    ap.add_argument("--alpha", type=float, action="append", help="L1/L2 mix (R's)")
    ap.add_argument("--threads", type=int, action="append")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cap", type=float, default=900, help="seconds per run")
    ap.add_argument(
        "--ratio-cap",
        type=float,
        default=100,
        help="also cap each run at this multiple of pybiglasso's single-threaded "
        "time on the same problem; a timeout then means 'over this many times "
        "slower'",
    )
    ap.add_argument("--out", default=HERE / "results" / "path.jsonl")
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
    solvers = args.solver or SOLVERS
    alphas = args.alpha or [1.0]
    threads = args.threads or [1]
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    env_info = environment()

    for dataset in datasets:
        X, Xs, y = make_data(dataset)
        for alpha in alphas:
            lambdas = lambda_grid(X, y, alpha)
            runs = {}
            run_cap = args.cap
            with tempfile.TemporaryDirectory() as tmp:
                np.save(f"{tmp}/lambdas.npy", lambdas)
                for solver in solvers:
                    if alpha < 1 and solver in LASSO_ONLY:
                        continue
                    if alpha == 1 and solver in ENET_ONLY:
                        continue
                    if (
                        solver == "statsmodels"
                        and statsmodels_memory_gb(X.shape[1]) > STATSMODELS_MAX_GB
                    ):
                        print(
                            f"{dataset} alpha={alpha} statsmodels: skipped, needs "
                            f"~{statsmodels_memory_gb(X.shape[1]):.0f} GB",
                            flush=True,
                        )
                        continue
                    for t in threads:
                        if solver in SINGLE_THREADED and t != min(threads):
                            continue
                        times, result = [], None
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
                            r = np.load(res)
                            times.append(float(r["seconds"]))
                            result = (r["coef"], r["intercept"])
                            if times[-1] > run_cap / 3:
                                break  # slow; one measurement will do
                        runs[(solver, t)] = (times, result)
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
            for key, (_times, result) in runs.items():
                if result is None:
                    continue
                coef, intercept = result
                objs[key] = np.array(
                    [
                        objective(Xs, y, lam, alpha, coef[k], intercept[k])
                        if np.isfinite(intercept[k])
                        else np.nan
                        for k, lam in enumerate(lambdas)
                    ]
                )
            best = np.nanmin(np.vstack(list(objs.values())), axis=0)

            with open(out, "a") as fp:
                for (solver, t), (times, _result) in runs.items():
                    subopt = None
                    if (solver, t) in objs:
                        rel = (objs[(solver, t)] - best) / np.abs(best)
                        subopt = float(np.nanmax(rel))
                    record = dict(
                        dataset=dataset,
                        nrow=DATASETS[dataset][0],
                        ncol=DATASETS[dataset][1],
                        alpha=alpha,
                        nlambda=len(lambdas),
                        solver=solver,
                        threads=t,
                        seconds=float(np.median(times)) if times else None,
                        times=times,
                        timed_out=bool(times) and not np.isfinite(times[-1]),
                        cap=run_cap if solver != "pybiglasso" else args.cap,
                        ratio_cap=args.ratio_cap,
                        max_rel_suboptimality=subopt,
                        **env_info,
                    )
                    fp.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
