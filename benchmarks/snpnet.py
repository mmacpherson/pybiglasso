"""Optional snpnet path adapter and a correctness-only, non-reportable smoke."""

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import path as bench
from competitors import run_r
from scipy.special import expit

HERE = Path(__file__).resolve().parent
# Pinned snpnet passes thresh through the older glmnet API and supplies a
# zero-column covariate after its deliberately constant dummy is removed.
# Retain these advisories in metadata; every other warning remains fatal.
ADVISORY_WARNINGS = (
    "Passing 'thresh' to glmnet() is deprecated. Use control = list(thresh = ...) instead.",
    "non-empty data for zero-extent matrix",
)


def write_plink1(X, y, prefix):
    """Encode A-allele dosages as variant-major BED, without rounding data."""
    if X.ndim != 2 or len(y) != len(X) or not np.isin(X, [0, 1, 2]).all():
        raise ValueError("PLINK benchmark inputs must be complete 0/1/2 dosages")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("binomial response must contain only 0 and 1")
    nrow, ncol = X.shape
    codes = np.array([0b11, 0b10, 0b00], dtype=np.uint8)[X.astype(np.intp)]
    padded = np.zeros(((nrow + 3) // 4 * 4, ncol), dtype=np.uint8)
    padded[:nrow] = codes
    quads = padded.reshape(-1, 4, ncol)
    packed = quads[:, 0] | quads[:, 1] << 2 | quads[:, 2] << 4 | quads[:, 3] << 6
    prefix = Path(prefix)
    with prefix.with_suffix(".bed").open("wb") as output:
        output.write(bytes([0x6C, 0x1B, 0x01]))
        output.write(np.ascontiguousarray(packed.T).tobytes())
    prefix.with_suffix(".bim").write_text(
        "".join(f"1\tv{j + 1}\t0\t{j + 1}\tA\tC\n" for j in range(ncol))
    )
    prefix.with_suffix(".fam").write_text(
        "".join(f"s{i + 1}\ts{i + 1}\t0\t0\t0\t-9\n" for i in range(nrow))
    )
    (prefix.parent / "pheno.tsv").write_text(
        "FID\tIID\ty\tzero\n"
        + "".join(
            f"s{i + 1}\ts{i + 1}\t{int(value) + 1}\t0\n" for i, value in enumerate(y)
        )
    )


def fit_r_snpnet(X, Xs, y, lambdas, alpha, threads, *, eps=1e-7, timeout=1800):
    if alpha != 1:
        raise ValueError("snpnet's raw-dosage penalty mapping is lasso-only")
    mean, scale = X.mean(axis=0), X.std(axis=0)
    if np.any(scale <= 1e-6):
        raise ValueError("snpnet path needs the common nonconstant-column filter")
    with tempfile.TemporaryDirectory(dir=os.environ.get("R_WORKDIR")) as tmp:
        directory = Path(tmp)
        write_plink1(X, y, directory / "X")
        params = dict(
            nrow=len(y),
            ncol=X.shape[1],
            family="binomial",
            lambdas=list(lambdas),
            sds=list(scale),
            ncores=threads,
            eps=eps,
        )
        (directory / "params.json").write_text(json.dumps(params))
        result = run_r(
            directory,
            HERE / "r" / "fit_snpnet.R",
            timeout,
            advisory_warnings=ADVISORY_WARNINGS,
        )
    coef = (
        np.array(result["beta"], dtype=float).reshape(result["beta_dim"], order="F").T
    )
    intercept = np.atleast_1d(np.asarray(result["intercept"], dtype=float))
    if coef.shape != (len(lambdas), X.shape[1]) or intercept.shape != (len(lambdas),):
        raise ValueError("snpnet did not complete the requested path")
    if not np.isfinite(coef).all() or not np.isfinite(intercept).all():
        raise ValueError("nonfinite snpnet model")
    np.testing.assert_array_equal(np.asarray(result["lambdas"], dtype=float), lambdas)
    seconds = float(result["seconds"])
    if not np.isfinite(seconds) or seconds < 0:
        raise ValueError("invalid snpnet fit clock")
    intercept = intercept + coef @ mean
    coef *= scale
    return (
        coef,
        intercept,
        seconds,
        dict(
            snpnet=result["snpnet_version"],
            r=result["r_version"],
            warnings=result.get("warnings", []),
            fitted_lambdas=len(intercept),
            eps=eps,
            genotype_conversion="untimed BED to PGEN",
            family="binomial",
        ),
    )


def max_kkt(Xs, y, lambdas, coef, intercept):
    errors = expit(Xs @ coef.T + intercept) - y[:, None]
    gradient = Xs.T @ errors / len(y)
    l1 = lambdas[None, :]
    gradient += np.where(coef.T != 0, l1 * np.sign(coef.T), np.clip(-gradient, -l1, l1))
    return float(max(abs(gradient).max(), abs(errors.mean(axis=0)).max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    from cv import data

    import pybiglasso as bl

    X, y = data("smoke", 0)
    Xs = (X - X.mean(axis=0)) / X.std(axis=0)
    maximum = np.max(abs(Xs.T @ (y - y.mean()))) / len(y)
    # <=30 points also exercises small inputs without the upstream variant-
    # exhaustion truncation of a partially extended 100-point path.
    lambdas = np.geomspace(maximum * (1 + 1e-9), maximum * 0.5, 20)
    coef, intercept, _, metadata = fit_r_snpnet(
        X, Xs, y, lambdas, 1.0, 1, eps=1e-14, timeout=args.timeout
    )
    kkt = max_kkt(Xs, y, lambdas, coef, intercept)
    # Same independently established mean-NLL KKT ceiling at eps=1e-14 as
    # the numerical test suite, not an observed-output-derived tolerance.
    if kkt > 1e-6:
        raise ValueError(f"snpnet smoke fails independent KKT: {kkt}")
    fitted = bl.biglasso(
        X,
        y,
        family="binomial",
        backend="rust",
        lambdas=lambdas,
        eps=1e-14,
        n_threads=1,
        max_iter=10_000,
    )
    if len(fitted) != len(lambdas):
        raise ValueError("Python smoke path is incomplete")
    raw_coef = fitted.coef.toarray().T
    python_kkt = max_kkt(
        Xs,
        y,
        lambdas,
        raw_coef * X.std(axis=0),
        fitted.intercept + raw_coef @ X.mean(axis=0),
    )
    if python_kkt > 1e-6:
        raise ValueError(f"Python smoke fails independent KKT: {python_kkt}")
    row = dict(
        phase="smoke",
        reportable=False,
        max_kkt=kkt,
        python_max_kkt=python_kkt,
        input_digest=bench.input_digest(X, y),
        **metadata,
    )
    args.out.write_text(json.dumps(row, allow_nan=False) + "\n")
    print("snpnet smoke: full path and independent KKT passed; not reportable timings")


if __name__ == "__main__":
    main()
