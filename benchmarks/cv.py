#!/usr/bin/env python3
"""File-backed binomial CV versus bigstatsr CMSA, with/without early stopping.

Phase 1 defaults to smoke: these are functional runs, not reportable timings.
The selected models differ: pybiglasso refits on all rows; CMSA averages folds.
Data preparation, imports, warm-up, export and scoring are outside fit clocks.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import path as bench
from competitors import run_r
from scipy.special import expit
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

HERE = Path(__file__).resolve().parent
SOLVERS = ("pybiglasso", "r-bigstatsr")
MODES = ("on", "off")


def data(dataset, seed):
    if dataset != "smoke":
        X, _, y = bench.make_data(dataset, seed=seed, family="binomial")
        return X, y
    # Mixed, modest signal; never used for a published performance claim.
    rng = np.random.default_rng(seed)
    X = np.asfortranarray(rng.binomial(2, 0.35, size=(160, 24)).astype(float))
    logits = 0.4 * (X[:, 0] - X[:, 1]) - 0.2
    return X, (rng.random(len(X)) < expit(logits)).astype(float)


def prepare_case(dataset, directory, folds, seed):
    X, y = data(dataset, seed)
    train, test = train_test_split(
        np.arange(len(y)),
        test_size=0.2,
        stratify=y,
        random_state=seed,
    )
    train, test = np.sort(train), np.sort(test)
    training = X[train]
    splitter = StratifiedKFold(folds, shuffle=True, random_state=seed)
    indices = list(splitter.split(training, y[train]))
    keep = training.std(axis=0) > 1e-6
    for fit_rows, _ in indices:
        keep &= training[fit_rows].std(axis=0) > 1e-6
    assert keep.any(), "no columns vary in every fold's training rows"
    # Common untimed filter: bigstatsr otherwise drops fold-constant columns.
    training = np.asfortranarray(training[:, keep], dtype=np.float64)
    training.ravel(order="F").tofile(directory / "X_train.bk")
    y[train].astype(np.float64).tofile(directory / "y_train.bin")
    sets = np.empty(len(train), dtype=int)
    for k, (_, held_out) in enumerate(indices, 1):
        sets[held_out] = k
    np.savetxt(directory / "folds.txt", sets, fmt="%d")
    shape = dict(nrow=len(train), ncol=int(keep.sum()), ncol_original=X.shape[1])
    (directory / "shape.json").write_text(json.dumps(shape))
    return X[test][:, keep], y[test]


def open_train(directory):
    shape = json.loads((directory / "shape.json").read_text())
    X = np.memmap(
        directory / "X_train.bk",
        dtype=np.float64,
        mode="r",
        shape=(shape["nrow"], shape["ncol"]),
        order="F",
    )
    y = np.fromfile(directory / "y_train.bin", dtype=np.float64)
    sets = np.atleast_1d(np.loadtxt(directory / "folds.txt", dtype=int))
    return X, y, sets


def fold_indices(sets):
    return [
        (np.flatnonzero(sets != k), np.flatnonzero(sets == k)) for k in np.unique(sets)
    ]


def validate_model(coef, intercept, shape, seconds):
    if coef.shape != (shape["ncol"],) or not np.isfinite(coef).all():
        raise ValueError("invalid exported coefficient vector")
    if not np.isfinite(intercept) or not np.isfinite(seconds) or seconds < 0:
        raise ValueError("invalid intercept or fit clock")


def fit_python(directory, options):
    from pybiglasso.sklearn import BiglassoClassifierCV

    X, y, sets = open_train(directory)
    # Classifier's _fit_kwargs explicitly passes family='binomial'; no family
    # constructor argument exists. Check the returned family as well.
    model = BiglassoClassifierCV(
        l1_ratio=options.alpha,
        cv=fold_indices(sets),
        n_jobs=options.threads,
        n_alphas=options.nlambda,
        alpha_min_ratio=options.lambda_min,
        early_stopping=10 if options.early == "on" else None,
        backend="rust",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        started = time.perf_counter()
        model.fit(X, y)
        seconds = time.perf_counter() - started
    if model.path_.family != "binomial":
        raise ValueError("CV benchmark requires family='binomial'")
    counts = np.isfinite(model.cv_scores_).sum(axis=1)
    if options.early == "off" and not np.all(counts == options.nlambda):
        raise ValueError("a non-early-stopped fold did not fit the requested path")
    return (
        model.coef_.ravel(),
        float(model.intercept_[0]),
        seconds,
        dict(
            model_kind="selected full-data refit",
            family="binomial",
            fold_lambdas=counts.tolist(),
            full_fit_lambdas=len(model.path_),
            selected_lambda=float(model.alpha_),
            eps=model.tol,
        ),
    )


def fit_bigstatsr(directory, options):
    shape = json.loads((directory / "shape.json").read_text())
    params = dict(
        **shape,
        family="binomial",
        folds=options.folds,
        alpha=options.alpha,
        ncores=options.threads,
        seed=options.seed,
        nlambda=options.nlambda,
        lambda_min=options.lambda_min,
        nlam_min=min(50, options.nlambda),
        early_stopping=options.early == "on",
    )
    (directory / "params.json").write_text(json.dumps(params))
    # This summary advisory describes validation-loss stopping, not coordinate
    # convergence. Retain it; reject actual max-iteration failures below.
    result = run_r(
        directory,
        HERE / "r" / "fit_bigstatsr.R",
        options.timeout,
        advisory_warnings=(
            "Some models may not have reached a minimum; check with summary() and plot().",
        ),
    )
    iterations = np.atleast_1d(np.asarray(result["fold_max_iterations"], dtype=float))
    if (
        iterations.shape != (options.folds,)
        or not np.isfinite(iterations).all()
        or np.any(iterations >= result["max_iter"])
        or np.any(iterations < 0)
    ):
        raise ValueError("bigstatsr did not converge")
    columns = np.atleast_1d(np.asarray(result["ind_col"], dtype=int)) - 1
    beta = np.atleast_1d(np.asarray(result["beta"], dtype=float))
    if len(columns) != len(beta) or len(np.unique(columns)) != len(columns):
        raise ValueError("invalid bigstatsr column mapping")
    if np.any(columns < 0) or np.any(columns >= shape["ncol"]):
        raise ValueError("bigstatsr exported an out-of-range column")
    coef = np.zeros(shape["ncol"])
    coef[columns] = beta
    intercept, seconds = float(result["intercept"]), float(result["seconds"])
    X, _, _ = open_train(directory)
    probe = np.atleast_1d(np.asarray(result["probe_logits"], dtype=float))
    if probe.shape != (min(8, len(X)),):
        raise ValueError("bigstatsr prediction probe is incomplete")
    # An export/summary check against R's own predict method, not a claim that
    # CMSA minimizes one common penalized objective. gamma_(4p+8) includes
    # both dot-product orders and averaging/export on this small probe.
    operations = 4 * shape["ncol"] + 8
    unit = np.finfo(float).eps
    magnitude = 1 + abs(intercept) + abs(X[: len(probe)]) @ abs(coef)
    allowance = operations * unit / (1 - operations * unit) * magnitude
    if not np.all(abs(X[: len(probe)] @ coef + intercept - probe) <= allowance):
        raise ValueError("bigstatsr exported model disagrees with predict()")
    counts = np.atleast_1d(result["fold_lambdas"])
    if (
        counts.shape != (options.folds,)
        or not np.isfinite(counts).all()
        or np.any(counts < 1)
        or np.any(counts > options.nlambda)
    ):
        raise ValueError("invalid bigstatsr fold lengths")
    if options.early == "off" and not np.all(counts == options.nlambda):
        raise ValueError("bigstatsr non-early-stopped fold path is incomplete")
    metadata = {
        key: result[key]
        for key in (
            "bigstatsr_version",
            "r_version",
            "model_kind",
            "fold_lambdas",
            "n_abort",
            "nlam_min",
            "nlambda",
            "lambda_min",
            "eps",
            "all_early_stopped",
            "fold_max_iterations",
            "fold_messages",
            "max_iter",
            "warnings",
        )
    }
    return coef, intercept, seconds, metadata


def run_one(args):
    directory = Path(args.case)
    shape = json.loads((directory / "shape.json").read_text())
    fitter = fit_python if args.solver == "pybiglasso" else fit_bigstatsr
    # Tiny type-matched Python warm-up, untimed; no compilation is timed.
    if args.solver == "pybiglasso":
        with tempfile.TemporaryDirectory(dir=directory.parent) as warm:
            warm = Path(warm)
            prepare_case("smoke", warm, args.folds, args.seed)
            fitter(warm, args)
    coef, intercept, seconds, metadata = fitter(directory, args)
    validate_model(coef, intercept, shape, seconds)
    np.savez(
        args.result,
        coef=coef,
        intercept=intercept,
        seconds=seconds,
        metadata=json.dumps(metadata),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", action="append", choices=["smoke", *bench.DATASETS]
    )
    parser.add_argument("--solver", action="append", choices=SOLVERS)
    parser.add_argument("--early-stopping", action="append", choices=MODES)
    parser.add_argument("--alpha", type=float, action="append")
    parser.add_argument("--threads", type=int, action="append")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--nlambda", type=int, default=100)
    parser.add_argument("--lambda-min", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--timeout", type=float, default=1800, help="wall safety cap per fit"
    )
    parser.add_argument("--phase", choices=["smoke", "reportable"], default="smoke")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-one", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case", help=argparse.SUPPRESS)
    parser.add_argument("result", nargs="?", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (
        args.folds < 2
        or args.nlambda < 2
        or args.repeats < 1
        or args.timeout <= 0
        or not 0 < args.lambda_min <= 1
        or any(t < 1 for t in args.threads or [1])
        or any(not 0 < a <= 1 for a in args.alpha or [1.0])
    ):
        parser.error("invalid folds/grid/repeats/threads/timeout/alpha")
    if args.phase == "reportable" and args.repeats < 3:
        parser.error("reportable measurements require at least three repetitions")
    if args.phase == "reportable" and "smoke" in (args.dataset or ["smoke"]):
        parser.error("choose a real dataset for reportable measurements")
    if args.run_one:
        args.solver, args.alpha, args.threads = (
            args.solver[0],
            args.alpha[0],
            args.threads[0],
        )
        args.early = args.early_stopping[0]
        run_one(args)
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for dataset in args.dataset or ["smoke"]:
        with tempfile.TemporaryDirectory(dir=os.environ.get("R_WORKDIR")) as tmp:
            directory = Path(tmp)
            X_test, y_test = prepare_case(dataset, directory, args.folds, args.seed)
            X, y, sets = open_train(directory)
            digest = bench.input_digest(X, y)
            del X  # close the map before TemporaryDirectory cleanup (also on Windows)
            folds_digest = hashlib.sha256(
                (directory / "folds.txt").read_bytes()
            ).hexdigest()
            shape = json.loads((directory / "shape.json").read_text())
            for alpha in args.alpha or [1.0, 0.5]:
                for solver in args.solver or SOLVERS:
                    for early in args.early_stopping or MODES:
                        for threads in args.threads or [1]:
                            samples, scores, diagnostics = [], [], []
                            for _repeat in range(args.repeats):
                                result_path = directory / "result.npz"
                                limits = bench.r_thread_limits(1)
                                env = dict(
                                    os.environ, **limits, RAYON_NUM_THREADS=str(threads)
                                )
                                command = [
                                    sys.executable,
                                    str(Path(__file__).resolve()),
                                    "--run-one",
                                    "--case",
                                    str(directory),
                                    "--out",
                                    str(args.out),
                                    "--solver",
                                    solver,
                                    "--alpha",
                                    str(alpha),
                                    "--threads",
                                    str(threads),
                                    "--early-stopping",
                                    early,
                                    "--folds",
                                    str(args.folds),
                                    "--seed",
                                    str(args.seed),
                                    "--nlambda",
                                    str(args.nlambda),
                                    "--lambda-min",
                                    str(args.lambda_min),
                                    "--timeout",
                                    str(args.timeout),
                                    str(result_path),
                                ]
                                subprocess.run(
                                    command,
                                    env=env,
                                    check=True,
                                    timeout=2 * args.timeout + 120,
                                )
                                with np.load(result_path) as result:
                                    coef, intercept = (
                                        result["coef"],
                                        float(result["intercept"]),
                                    )
                                    probability = expit(X_test @ coef + intercept)
                                    scores.append(
                                        dict(
                                            auc=float(
                                                roc_auc_score(y_test, probability)
                                            ),
                                            log_loss=float(
                                                log_loss(y_test, probability)
                                            ),
                                        )
                                    )
                                    samples.append(float(result["seconds"]))
                                    diagnostics.append(
                                        json.loads(str(result["metadata"]))
                                    )
                            row = dict(
                                dataset=dataset,
                                family="binomial",
                                solver=solver,
                                early_stopping=early,
                                threads=threads,
                                folds=args.folds,
                                alpha=alpha,
                                nlambda=args.nlambda,
                                lambda_min=args.lambda_min,
                                phase=args.phase,
                                reportable=args.phase == "reportable",
                                times=samples,
                                seconds=float(np.median(samples)),
                                held_out=scores,
                                diagnostics=diagnostics,
                                input_digest=digest,
                                folds_digest=folds_digest,
                                nrow_test=len(y_test),
                                **shape,
                                **bench.environment(),
                                thread_limits=dict(
                                    limits, RAYON_NUM_THREADS=str(threads)
                                ),
                            )
                            with args.out.open("a") as output:
                                output.write(json.dumps(row, allow_nan=False) + "\n")
                            print(
                                f"{dataset} {solver} early={early} threads={threads}: {row['seconds']:.4g}s ({args.phase})",
                                flush=True,
                            )


if __name__ == "__main__":
    main()
