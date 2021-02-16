#!/usr/bin/env python3
import logging
import pathlib
import timeit

import click
import numpy as np
import sh
import sklearn.linear_model
from jax import random

import pybiglasso as bl

logging.getLogger("pybiglasso").setLevel(level=logging.WARNING)
logging.getLogger("sh").setLevel(level=logging.WARNING)


def synthesize_logistic_dataset(
    nrow, ncol, batchsize=1000, Xfile="X.csv", yfile="y.csv"
):
    assert (
        nrow % batchsize == 0
    ), f"nrow [{nrow}] is not multiple of batchsize [{batchsize}]"
    num_batches = nrow // batchsize

    rng_key = random.PRNGKey(bl.common.DEFAULT_SEED)
    rng_key, *subkeys = random.split(rng_key, 3)

    weights = bl.common.sample_biglasso_weights(subkeys[0], ncol)

    bg = bl.common.biglasso_generator(subkeys[1], weights, batchsize)

    y_batches = []
    with open(Xfile, "wt") as fp:
        for _ in range(num_batches):
            a_batch = next(bg)
            np.savetxt(fp, a_batch["X"], fmt="%.3f", delimiter=",")
            y_batches.append(a_batch["y"])

    np.savetxt(yfile, np.concatenate(y_batches), fmt="%d")


@click.command()
@click.option("--nrow", default=10000, type=int)
@click.option("--ncol", default=100, type=int)
@click.option("--nreps", default=5, type=int)
@click.option("--nlambda", default=51, type=int)
@click.option("--alpha", default=0.8, type=float)
@click.option("--mmap/--no-mmap", default=True)
@click.option("--dtype", default="float64", type=click.Choice(["float64", "float32"]))
@click.option("--mmap-file", default="py-mmap.npy", type=str)
@click.option("--seed", default=13579, type=int)
@click.option("--workdir", default="py-vs-r", type=click.Path(dir_okay=True))
@click.option("--r-docker-image", default="rbox", type=str)
@click.option("--sgd/--no-sgd", default=True)
def main(
    nrow,
    ncol,
    nreps,
    nlambda,
    alpha,
    mmap,
    dtype,
    mmap_file,
    seed,
    workdir,
    r_docker_image,
    sgd,
):

    # -- look up directory this script lives in
    this_dir = pathlib.Path(__file__).parent
    repo_root_dir = this_dir.parent.absolute()

    # -- make directory to hold synthetic input data
    pathlib.Path(workdir).mkdir(parents=True, exist_ok=True)

    Xf = f"{workdir}/X.csv"
    yf = f"{workdir}/y.csv"
    mf = f"{workdir}/{mmap_file}"

    # -- we always generate the data via csv files so this works when the size exceeds available RAM
    synthesize_logistic_dataset(nrow, ncol, Xfile=Xf, yfile=yf)

    # -- load data for pybiglasso
    if mmap:
        X = np.lib.format.open_memmap(
            mf, mode="w+", dtype=dtype, shape=(nrow, ncol), fortran_order=True
        )
        with open(Xf, "rt") as fp:
            for ix in range(nrow):
                row = np.loadtxt(fp, dtype=dtype, delimiter=",", max_rows=1)
                X[ix, :] = row

        del X
        X = np.lib.format.open_memmap(mf, mode="r")
    else:
        X = np.loadtxt(Xf, dtype=dtype, delimiter=",")
        X = np.asfortranarray(X)

    y = np.loadtxt(yf)
    # print("synthetic data loaded")

    # -- run a small problem to incur the jit cost, to keep it out of the
    #    benchmark
    _ = bl.biglasso(
        np.asfortranarray(np.random.normal(size=(10, 2))),
        (np.random.normal(size=10) > 0).astype(np.float64),
        alpha=0.8,
        nlambda=10,
    )

    # -- benchmark pybiglasso under given config
    bl_time = timeit.timeit(
        "bl.biglasso(X, y, family='binomial', alpha=alpha, nlambda=nlambda)",
        number=nreps,
        globals=dict(bl=bl, X=X, y=y, alpha=alpha, nlambda=nlambda),
    )
    print(
        f"pybiglasso {round(1000*bl_time/nreps):d} {nrow} {ncol} {nlambda} {alpha} {int(mmap)} {dtype}"
    )

    if sgd:
        blfit = bl.biglasso(X, y, family="binomial", alpha=alpha, nlambda=nlambda)
        lambdas = blfit["df_nlambda"].lambda_.to_numpy()

        # -- do sklearn sgd
        def fit_enet_path_via_sgd(X, y, alpha, nlambda):
            for ix, lam in enumerate(lambdas):
                sgder = sklearn.linear_model.SGDClassifier(
                    loss="log",
                    penalty="elasticnet",
                    alpha=lam,
                    l1_ratio=alpha,
                    tol=1e-7,
                )
                sgder.fit(X, y)
            return sgder

        sgd_time = timeit.timeit(
            "fit_enet_path_via_sgd(X, y, alpha, nlambda)",
            number=nreps,
            globals=dict(
                fit_enet_path_via_sgd=fit_enet_path_via_sgd,
                bl=bl,
                X=X,
                y=y,
                alpha=alpha,
                nlambda=nlambda,
            ),
        )
        print(
            f"sklearn.sgd {round(1000*sgd_time/nreps):d} {nrow} {ncol} {nlambda} {alpha} {int(mmap)} {dtype}"
        )

    if dtype != "float64":
        # -- R's biglasso does not have meaningful non-float64 mode
        return

    # -- preemptively delete R bigmemory backing files; if we don't it'll throw
    sh.rm("-f", Xf.replace(".csv", ".bin"))
    sh.rm("-f", Xf.replace(".csv", ".desc"))

    # -- (call script to) benchmark R biglasso on exact same input data
    r_result = sh.docker(
        "run",
        "--rm",
        "-v",
        f"{repo_root_dir}:/app",
        "-w",
        "/app",
        r_docker_image,
        "Rscript",
        f"{this_dir}/benchmark_biglasso.R",
        Xf,
        yf,
        nreps,
        nlambda,
        alpha,
        int(mmap),
    )
    print(r_result.stdout.decode("utf8").strip())


if __name__ == "__main__":
    main()
