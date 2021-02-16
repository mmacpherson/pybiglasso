#!/usr/bin/env python3

import logging

import click
import numpy as np

logFormatter = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(format=logFormatter, level=logging.INFO)
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


# -- UNUSED!
def standardize_and_get_residuals(X, y):

    zmax = 0.0

    n, p = X.shape
    assert len(y) == n

    center = np.zeros(p, dtype=float)
    scale = np.zeros(p, dtype=float)

    for j in range(p):

        sum_xy = 0.0
        sum_y = 0.0

        for i in range(n):
            center[j] += X[i, j]
            scale[j] += X[i, j] ** 2

            sum_xy += X[i, j] * y[i]
            sum_y += y[i]

        center[j] = center[j] / n  # center
        scale[j] = np.sqrt(scale[j] / n - center[j] ** 2)  # scale

        if scale[j] < 1e-6:
            continue

        zj = (sum_xy - center[j] * sum_y) / (scale[j] * n)  # residual
        if abs(zj) > zmax:
            zmax = abs(zj)
            xmax_ptr = j  # xmax_ptr is the index in the raw xMat, not index in col_idx!

    # *p_keep_ptr = col_idx.size();
    # `*lambda_max_ptr = zmax / alpha;

    return center, scale, zmax, xmax_ptr


def print_vector(v, label, _round=6):
    return f"{label}: {np.asarray(v).round(_round)}"


def biglasso(
    X, y, alpha=1.0, family="binomial", nlambda=100, eps=1e-7, lambda_min=None
):

    if family != "binomial":
        raise NotImplementedError("Sorry, we only do binomial endpoints now.")

    results = binomial_hsr(
        X, y, nlamda=nlambda, eps=eps, alpha=alpha, lamda_min=lambda_min
    )

    # -- postprocess
    #    2. untransform coefficients

    results["weights"] /= results["scale"][results["col_map"], np.newaxis]
    results["intercept"] -= results["center"][results["col_map"]] @ results["weights"]
    _weights = np.zeros(
        (len(results["center"]), len(results["lambda_"])),
        dtype=results["weights"].dtype,
    )
    _weights[results["col_map"], :] = results["weights"]
    results["weights"] = _weights

    return results


def binomial_hsr(
    X,
    y,
    row_idx=None,
    lamda=None,
    nlamda=None,
    lamda_log_scale=True,
    lamda_min=None,
    alpha=None,
    # user=None,
    eps=None,
    max_iter=1000,
    multiplier=None,
    dfmax=None,
):
    # p_keep = 0
    nrow, ncol = X.shape
    logger.debug(f"Shape: [{nrow}, {ncol}]")

    if lamda_min is None:
        lamda_min = 0.001 if (nrow > ncol) else 0.05

    iters = np.zeros(nlamda, dtype=int)
    deviance = np.zeros(nlamda, dtype=float)
    n_reject = np.zeros(nlamda, dtype=int)

    # -- FIXFIX: do online
    center = X.mean(0)
    scale = X.std(0)

    # -- form map of original X column index to reduced column index
    col_map = np.where(scale > 1e-6)[0]
    ncol_used = len(col_map)
    logger.debug(f"Using [{ncol_used}] columns of [{len(center)}].")
    if multiplier is None:
        multiplier = np.ones_like(center)  # ie, penalty factor
    else:
        assert len(multiplier) == len(
            center
        ), "Multiplier has [{len(multiplier)}] elements, should have [{len(center)}] elements."

    # -- init remaining structures, based on ncol_used
    beta0 = np.zeros(nlamda, dtype=float)
    beta = np.zeros((ncol_used, nlamda), dtype=float)  # FIXFIX: this can be sparse

    a = np.zeros(ncol_used, dtype=float)
    w = np.zeros(ncol_used, dtype=float)
    e1 = np.zeros(ncol_used, dtype=int)

    # -- FIXFIX: do online
    sum_xy = (X[:, col_map] * y[:, np.newaxis]).sum(0)
    z = (sum_xy - center[col_map] * y.sum()) / (scale[col_map] * nrow)  # residuals
    zmax = np.abs(z).max()
    # xmax_ptr = np.abs(z).argmax()
    lamda_max = zmax / alpha  # FIXFIX: alpha zero case?

    ybar = y.mean()
    a0 = beta0[0] = np.log(ybar / (1 - ybar))

    logger.debug(print_vector(center, "center"))
    logger.debug(print_vector(scale, "scale"))

    # r = y.copy()
    # s = y - ybar
    eta = np.repeat(a0, len(y))
    null_deviance = -1.0 * (y * np.log(ybar) + (1 - y) * np.log(1 - ybar)).sum()
    thresh = eps * null_deviance / nrow

    # sumS = s.sum()  # sum of s
    # sum_w_residuals = 0.0  # sum of w * r

    # -- set up lambda grid if not provided
    if lamda_log_scale:
        lamda = np.flip(
            np.logspace(
                np.log(lamda_min * lamda_max), np.log(lamda_max), num=nlamda, base=np.e
            )
        )
    else:
        lamda = np.linspace(lamda_min, lamda_max, num=nlamda)
    logger.debug(print_vector(lamda, "lamda", _round=8))

    # -- init; FIXFIX - user-provided case
    deviance[0] = null_deviance
    lstart = 1
    n_reject[0] = ncol_used

    # -- loop over lambda values
    for il in range(lstart, len(lamda)):

        lam = lamda[il]
        logger.debug(f"lambda-index: {il} lambda: {lam}")

        if il > 0:
            if dfmax is not None:
                nv = (a != 0).sum()
                if nv > dfmax:
                    raise ValueError("Exceeded max coefficients.")

            cutoff = 2 * lam - lamda[il - 1]
        else:
            cutoff = 2 * lam - lamda_max

        e2 = (np.abs(z) > cutoff * alpha * multiplier[col_map]).astype(int)
        n_reject[il] = ncol_used - e2.sum()
        while iters[il] < max_iter:
            while iters[il] < max_iter:
                while iters[il] < max_iter:
                    iters[il] += 1

                    logger.debug("🐶")
                    logger.debug(f"For lamda ix [{il}], on iteration [{iters[il]}].")
                    # logger.debug(print_vector(eta, "eta"))
                    logger.debug(print_vector(e1, "e1"))
                    p = np.where(
                        eta > 10,
                        1,
                        np.where(eta < -10, 0, np.exp(eta) / (1 + np.exp(eta))),
                    )
                    w = np.where(
                        eta > 10, 0.0001, np.where(eta < -10, 0.0001, p * (1 - p))
                    )
                    s = y - p
                    r = s / w
                    deviance[il] = -np.log(np.where(y == 1, p, 1 - p)).sum()

                    if deviance[il] / null_deviance < 0.01:
                        logger.debug(
                            f"Model saturated. {deviance[il]} {null_deviance} Exiting!"
                        )

                        # -- truncate values at this lambda
                        return dict(
                            # -- length ncol
                            center=center,
                            scale=scale,
                            col_map=col_map,
                            # -- length nlambda
                            iters=iters[:il],
                            intercept=beta0[:il],
                            weights=beta[:, :il],
                            lambda_=lamda[:il],
                            deviance=deviance[:il],
                            num_rejected=n_reject[:il],
                        )

                    # -- update intercept
                    xwr = w @ r
                    _si = xwr / w.sum()
                    beta0[il] = _si + a0
                    if _si != 0:
                        a0 = beta0[il]
                        r -= _si
                        eta += _si
                    sum_w_residuals = w @ r
                    logger.debug(
                        f"b0 xwr {xwr:.6f} swr {sum_w_residuals:.6f} a0 {a0:.6f}"
                    )
                    # -- update betas
                    max_update = 0.0
                    for j in range(ncol_used):
                        jj = col_map[j]
                        if not e1[j]:
                            continue

                        xwr = (
                            (X[:, jj] * r * w).sum() - center[jj] * sum_w_residuals
                        ) / scale[jj]
                        v0 = w @ (X[:, jj] ** 2)
                        v1 = w @ X[:, jj]
                        v2 = w.sum()
                        v = (
                            (v0 - 2 * center[jj] * v1 + (center[jj] ** 2) * v2)
                            / (scale[jj] ** 2)
                            / nrow
                        )

                        u = xwr / nrow + v * a[j]
                        l1 = lam * multiplier[jj] * alpha
                        l2 = lam * multiplier[jj] * (1 - alpha)

                        if abs(u) <= l1:
                            beta[j, il] = 0
                        else:
                            sign = 1 if (u > 0) else -1
                            beta[j, il] = sign * (abs(u) - l1) / v / (1 + l2)

                        shift = beta[j, il] - a[j]
                        if shift != 0:
                            update = ((beta[j, il] - a[j]) ** 2) * v
                            max_update = max(update, max_update)
                            _si = shift * (X[:, jj] - center[jj]) / scale[jj]
                            r -= _si
                            eta += _si
                            sum_w_residuals = r @ w
                            a[j] = beta[j, il]

                        logger.debug(
                            f"b[{j}] xwr {xwr:.6f} v {v:.6f} u {u:.6f} l1 {l1:.6g} l2 {l2:.6g}"
                        )
                        logger.debug(
                            f"b[{j}] b {(shift + a[j]):.6f} a {a[j]:.6f} shift {shift:.6f} update {update:.6f} swr {sum_w_residuals:.6f}"
                        )
                        logger.debug(f"b[{j}] r {r}")
                        logger.debug(f"b[{j}] w {w}")

                    logger.debug(print_vector(deviance, "deviance"))
                    logger.debug(print_vector(beta0, "intercept"))
                    logger.debug(print_vector(beta[:, il], "beta"))

                    if max_update < thresh:
                        logger.debug(
                            f"max_update {max_update:.3g} l.t. thresh {thresh:.3g}, breaking"
                        )
                        break

                # -- scan for violations in strong set
                violations = 0
                logger.debug(f"SSC sumS: {s.sum()}")
                # logger.debug(print_vector(s, "s"))
                for j in range(ncol_used):
                    jj = col_map[j]
                    if (e1[j] != 0) or (e2[j] != 1):
                        continue
                    _sum = X[:, jj] @ s
                    z[j] = (_sum - center[jj] * s.sum()) / (scale[jj] * nrow)
                    l1 = lam * multiplier[jj] * alpha
                    l2 = lam * multiplier[jj] * (1 - alpha)
                    logger.debug(f"ssc[{j}] _s {_sum:0.3f} z {z[j]:.3f} a {a[j]:.3f}")
                    if abs(z[j] - a[j] * l2) > l1:
                        e1[j] = 1
                        violations += 1
                if violations == 0:
                    logger.debug("no strong-set violations, breaking")
                    break

            # -- scan for violation in rest
            violations = 0
            for j in range(ncol_used):
                jj = col_map[j]
                if e2[j] != 0:
                    continue
                _sum = X[:, jj] @ s
                z[j] = (_sum - center[jj] * s.sum()) / (scale[jj] * nrow)
                l1 = lam * multiplier[jj] * alpha
                l2 = lam * multiplier[jj] * (1 - alpha)
                if abs(z[j] - a[j] * l2) > l1:
                    e1[j] = e2[j] = 1
                    violations += 1
            if violations == 0:
                logger.debug("no rest-of-set violations, breaking")
                break

    return dict(
        # -- length ncol
        center=center,
        scale=scale,
        col_map=col_map,
        # -- length nlambda
        iters=iters,
        intercept=beta0,
        weights=beta,
        lambda_=lamda,
        deviance=deviance,
        num_rejected=n_reject,
    )


@click.command()
@click.argument("arg", default=1, type=int)
def main(arg):
    pass


if __name__ == "__main__":
    main()
