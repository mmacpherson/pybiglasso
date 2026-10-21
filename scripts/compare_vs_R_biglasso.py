#!/usr/bin/env python3
import itertools
import pathlib

import click
import numpy as np
import pandas as pd
import sh

import pybiglasso as bl

# import biglasso_ref as biglasso


@click.command()
@click.option("--workdir", default="py-vs-r", type=click.Path(dir_okay=True))
@click.option("--r-docker-image", default="rbox", type=str)
def main(workdir, r_docker_image):

    # -- look up directory this script lives in
    this_dir = pathlib.Path(__file__).parent
    repo_root_dir = this_dir.parent.absolute()

    # -- make directory to hold synthetic input data
    pathlib.Path(workdir).mkdir(parents=True, exist_ok=True)

    seeds = (123, 456, 789)
    sizes = (
        (7, 3),
        (100, 10),
    )
    nlambdas = (5, 20)
    alphas = (0.1, 0.5, 0.9, 1.0)
    fracs_present = (1.0, 0.5)

    for seed, size, alpha, nlambda, frac_present in itertools.product(
        seeds, sizes, alphas, nlambdas, fracs_present
    ):

        print(
            f"config: seed={seed}, design_matrix_size={size}, alpha={alpha}, nlambda={nlambda}, frac_cols_present={frac_present}"
        )

        np.random.seed(seed)

        # -- generate test data set and save to disk
        nrow, ncol = size
        test_data = bl.common.generate_test_data(nrow, ncol, seed=seed)
        X, y = test_data["X"], test_data["y"]
        X = np.asfortranarray(X)

        if frac_present < 1:
            ncol_mono = round(ncol * (1 - frac_present))
            mono_columns = np.sort(
                np.random.choice(ncol, size=ncol_mono, replace=False)
            )
            X[:, mono_columns] = 1
        Xdf = pd.DataFrame(X, columns=[f"col{e}" for e in range(ncol)])
        Xdf["y"] = y
        if_base = f"{workdir}/eg-{nrow}-{ncol}_alpha-{alpha}_nlambda-{nlambda}"
        infile = f"{if_base}.feather"
        Xdf.to_feather(infile)

        # -- fit our python-clone biglasso
        print("Running python biglasso")
        bl_fit = bl.biglasso(X, y, family="binomial", alpha=alpha, nlambda=nlambda)

        # -- perform biglasso fit inside docker container, writing outputs to some
        #    feather files
        # -- note that params, e.g. alpha and nlambda, are getting passed in via filename
        print("Running R biglasso")
        sh.docker(
            "run",
            "--rm",
            "-v",
            f"{repo_root_dir}:/app",
            "-w",
            "/app",
            r_docker_image,
            "Rscript",
            "/app/scripts/run_biglasso.R",
            infile,
        )

        # -- when successfully run, the above command should produce these three files
        df_r_beta = pd.read_feather(f"{if_base}.r-biglasso.beta.feather")
        df_r_ncol = pd.read_feather(f"{if_base}.r-biglasso.ncol.feather")
        df_r_nlam = pd.read_feather(f"{if_base}.r-biglasso.nlam.feather")

        # -- now compare the fits

        # -- center, scale, and lambda grid must agree
        assert np.allclose(
            df_r_ncol.center.to_numpy(), bl_fit["df_ncol"].center.to_numpy()
        )
        assert np.allclose(
            df_r_ncol.scale.to_numpy(), bl_fit["df_ncol"].scale.to_numpy()
        )
        assert np.allclose(
            df_r_nlam["lambda"].to_numpy(), bl_fit["df_nlambda"].lambda_.to_numpy()
        )
        assert np.allclose(
            df_r_nlam["iter"].to_numpy(), bl_fit["df_nlambda"].iterations.to_numpy()
        )

        # -- betas must agree
        assert np.allclose(
            df_r_beta.iloc[0, :].to_numpy(), bl_fit["df_weights"].iloc[:, 0].to_numpy()
        )
        assert np.allclose(
            df_r_beta.iloc[1:, :].to_numpy(),
            bl_fit["df_weights"].iloc[:, 1:].to_numpy().transpose(),
        )

        print("python and R biglasso routines agree.")
        print("--")


if __name__ == "__main__":
    main()
