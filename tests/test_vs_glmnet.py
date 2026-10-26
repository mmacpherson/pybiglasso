"""The default objective against R glmnet, which minimizes the same one.

- default: stored glmnet output (tests/fixtures/r_glmnet.json.gz; no R needed)
- `-m r`: the same cases live in R (Rscript, or CONTAINER + R_IMAGE)

pybiglasso fits glmnet's lambda grid. The two paths take different routes,
so each lambda's pair is compared as two solutions of one problem: both must
pass its KKT conditions, and convexity then bounds their distance
(support.certificates.assert_same_optimum).

The solver clips |eta| at ETA_CLIP, as R biglasso does, and solves the
clipped problem there, so the comparison holds where glmnet's own fit keeps
every |eta| within it. Beyond, pybiglasso-h7i: a strict xfail.
"""

import numpy as np
import pytest
from support import rglmnet
from support.certificates import (
    assert_same_optimum,
    standardize,
    standardized_theta,
    with_ones,
)
from test_vs_r import needs_r

import pybiglasso as bl

CASES = rglmnet.glmnet_grid()
# update_irls's eta_max: beyond it p is exactly 0 or 1
ETA_CLIP = 10.0
# the response with 3% ones, whose small lambdas pass ETA_CLIP
RARE = next(c for c in CASES if c.response == "rare")


def clipped_lambdas(X, glmnet):
    """Which of glmnet's lambdas have a row with |eta| > ETA_CLIP."""
    eta = glmnet["intercept"] + X @ glmnet["beta"]
    return np.abs(eta).max(axis=0) > ETA_CLIP


def assert_matches_glmnet(case, data, glmnet, *, clipped=False):
    """Both fits are certified close at each lambda unclipped (or clipped)."""
    assert glmnet["jerr"] == 0 and not glmnet["warnings"], glmnet["warnings"]
    X, y, pf = data["X"], data["y"], data["penalty_factor"]
    lambdas = glmnet["lambda_"]
    assert len(lambdas) == case.nlambda
    fit = bl.biglasso(
        X,
        y,
        alpha=case.alpha,
        lambdas=lambdas,
        penalty_factor=pf,
        eps=1e-14,
        stop_on_saturation=False,
    )
    np.testing.assert_array_equal(fit.lambdas, lambdas)

    # constant columns (monomorphic dosages) are no part of the problem
    varying = X.std(axis=0) > 0
    coef = fit.coef.toarray()
    assert not coef[~varying].any() and not glmnet["beta"][~varying].any()
    Xs, center, scale = standardize(X[:, varying])
    Z = with_ones(Xs)
    penalty = 1.0 if pf is None else pf[varying]

    selected = np.flatnonzero(clipped_lambdas(X, glmnet) == clipped)
    # never vacuous: something beyond the null model is compared
    assert glmnet["beta"][:, selected].any()
    for k in selected:
        ours = standardized_theta(fit.intercept[k], coef[varying, k], center, scale)
        theirs = standardized_theta(
            glmnet["intercept"][k], glmnet["beta"][varying, k], center, scale
        )
        assert_same_optimum(Z, y, ours, theirs, lambdas[k], case.alpha, penalty)


@pytest.fixture(scope="session")
def fixtures():
    return rglmnet.load_fixtures()


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_matches_stored_glmnet(case, fixtures, tmp_path):
    digest, glmnet = fixtures[case.name]
    data, new_digest = rglmnet.prepare_case(case, tmp_path)
    assert new_digest == digest, (
        "Generated data differs from what the stored glmnet output was fit on; "
        "regenerate with `make fixtures`"
    )
    assert_matches_glmnet(case, data, glmnet)


def test_rare_response_reaches_the_clip(fixtures, tmp_path):
    """The xfail below has clipped lambdas to compare; its other
    preconditions are test_matches_stored_glmnet's for the same case."""
    _, glmnet = fixtures[RARE.name]
    data, _ = rglmnet.prepare_case(RARE, tmp_path)
    assert clipped_lambdas(data["X"], glmnet).any()


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="pybiglasso-h7i: the |eta| > 10 clip biases the default objective",
)
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_clipped_lambdas_match_stored_glmnet(fixtures, tmp_path):
    _, glmnet = fixtures[RARE.name]
    data, _ = rglmnet.prepare_case(RARE, tmp_path)
    assert_matches_glmnet(RARE, data, glmnet, clipped=True)


@pytest.fixture(scope="session")
def live(tmp_path_factory):
    """Fit every case in one R session."""
    root = tmp_path_factory.mktemp("glmnet-live")
    data = {c.name: rglmnet.prepare_case(c, root / c.name)[0] for c in CASES}
    outputs = rglmnet.run_glmnet([root / c.name for c in CASES])
    return {c.name: (data[c.name], out) for c, out in zip(CASES, outputs, strict=True)}


@pytest.mark.r
@needs_r
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_matches_live_glmnet(case, live):
    data, glmnet = live[case.name]
    assert "error" not in glmnet, glmnet.get("error")
    assert_matches_glmnet(case, data, glmnet)
