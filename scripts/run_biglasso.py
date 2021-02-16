#!/usr/bin/env python3
import logging
import pathlib
import sys

import click
import numpy as np
from jax import random
from tqdm import tqdm

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

    weights = bl.common.sample_biglasso_weights(subkeys[0], ncol, marker_effect_sd=0.05)

    bg = bl.common.biglasso_generator(subkeys[1], weights, batchsize)

    y_batches = []
    with open(Xfile, "wt") as fp:
        for _ in tqdm(range(num_batches)):
            a_batch = next(bg)
            np.savetxt(fp, a_batch["X"], fmt="%.3f", delimiter=",")
            y_batches.append(a_batch["y"])

    y = np.concatenate(y_batches)

    np.savetxt(yfile, y, fmt="%d")


@click.command()
@click.option("--nrow", default=10000, type=int)
@click.option("--ncol", default=100, type=int)
@click.option("--nlambda", default=51, type=int)
@click.option("--alpha", default=0.8, type=float)
@click.option("--mmap/--no-mmap", default=True)
@click.option("--dtype", default="float64", type=click.Choice(["float64", "float32"]))
@click.option("--mmap-file", default="py-mmap.npy", type=str)
@click.option("--seed", default=13579, type=int)
@click.option("--workdir", default="py-vs-r", type=click.Path(dir_okay=True))
def main(
    nrow,
    ncol,
    nlambda,
    alpha,
    mmap,
    dtype,
    mmap_file,
    seed,
    workdir,
):

    # -- make directory to hold synthetic input data
    pathlib.Path(workdir).mkdir(parents=True, exist_ok=True)

    Xf = f"{workdir}/X.csv"
    yf = f"{workdir}/y.csv"
    mf = f"{workdir}/{mmap_file}"

    # -- we always generate the data via csv files so this works when the size exceeds available RAM
    print("synthesizing data", file=sys.stderr)
    synthesize_logistic_dataset(nrow, ncol, Xfile=Xf, yfile=yf)

    # -- load data for pybiglasso
    if mmap:
        print("building mmap", file=sys.stderr)
        X = np.lib.format.open_memmap(
            mf, mode="w+", dtype=dtype, shape=(nrow, ncol), fortran_order=True
        )
        with open(Xf, "rt") as fp:
            for ix in tqdm(range(nrow)):
                row = np.loadtxt(fp, dtype=dtype, delimiter=",", max_rows=1)
                X[ix, :] = row

        del X
        X = np.lib.format.open_memmap(mf, mode="r")
    else:
        X = np.loadtxt(Xf, dtype=dtype, delimiter=",")
        X = np.asfortranarray(X)

    X = np.lib.format.open_memmap(mf, mode="r")
    y = np.loadtxt(yf)
    assert y.var() > 0, "Uh oh, monomorphic response!"

    # -- benchmark pybiglasso under given config
    print("fitting biglasso model", file=sys.stderr)
    bl.biglasso(X, y, family="binomial", alpha=alpha, nlambda=nlambda)


if __name__ == "__main__":
    main()
