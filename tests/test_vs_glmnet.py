"""The default objective against R glmnet, which minimizes the same one.

- default: stored glmnet output (tests/fixtures/r_glmnet.json.gz; no R needed)
- `-m r`: the same cases live in R (Rscript, or CONTAINER + R_IMAGE)

pybiglasso fits glmnet's lambda grid. The two paths take different routes,
so each lambda's pair is compared as two solutions of one problem: both must
pass its KKT conditions, and then convexity bounds their distance: through
the curvature of tall data (support.certificates.assert_same_optimum), or
through the duality gap of wide data (assert_gap_close).

Every lambda is checked, including rare-response fits beyond R biglasso's
|eta| > 10 cutoff. Default mode uses glmnet's later pmin=1e-9 boundary.
"""

import numpy as np
import pytest
from support import rglmnet
from support.certificates import (
    KKT_TOL,
    assert_same_optimum,
    duality_gap,
    gap_coefficient_distance,
    standardize,
    standardized_kkt,
    standardized_theta,
    with_ones,
)
from test_vs_r import needs_r

import pybiglasso as bl

CASES = rglmnet.glmnet_grid()
# R biglasso's boundary, retained only in compatibility mode.
ETA_CLIP = 10.0
# the response with 3% ones, whose small lambdas pass ETA_CLIP
RARE = next(c for c in CASES if c.response == "rare")


def clipped_lambdas(X, glmnet):
    """Which of glmnet's lambdas have a row with |eta| > ETA_CLIP."""
    eta = glmnet["intercept"] + X @ glmnet["beta"]
    return np.abs(eta).max(axis=0) > ETA_CLIP


def fit_glmnet_grid(case, data, glmnet):
    """pybiglasso on glmnet's grid, and the standardized problem both solve.

    Checks what every comparison assumes: glmnet's whole path, without errors
    or warnings, and constant columns (monomorphic dosages) zero in both fits,
    which leaves them out of the problem.
    """
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
        family="binomial",
    )
    np.testing.assert_array_equal(fit.lambdas, lambdas)
    varying = X.std(axis=0) > 0
    coef = fit.coef.toarray()
    assert not coef[~varying].any() and not glmnet["beta"][~varying].any()
    Xs, center, scale = standardize(X[:, varying])

    def theta(intercept, beta):
        return standardized_theta(intercept, beta[varying], center, scale)

    return dict(
        Z=with_ones(Xs),
        y=y,
        lambdas=lambdas,
        penalty=1.0 if pf is None else pf[varying],
        ours=[theta(fit.intercept[k], coef[:, k]) for k in range(len(lambdas))],
        theirs=[
            theta(glmnet["intercept"][k], glmnet["beta"][:, k])
            for k in range(len(lambdas))
        ],
    )


def assert_matches_glmnet(case, data, glmnet):
    """Both paths pass KKT and the distance certificate at every lambda."""
    p = fit_glmnet_grid(case, data, glmnet)
    for k, lam in enumerate(p["lambdas"]):
        residual = standardized_kkt(
            p["Z"], p["y"], p["theirs"][k], lam, case.alpha, p["penalty"]
        )
        assert residual < KKT_TOL, f"glmnet at lambda {k}: {residual:.2e}"
    # never vacuous: something beyond the null model is compared
    assert glmnet["beta"].any()
    for k in range(len(p["lambdas"])):
        args = (p["lambdas"][k], case.alpha, p["penalty"])
        if case.nrow > case.ncol:
            assert_same_optimum(p["Z"], p["y"], p["ours"][k], p["theirs"][k], *args)
        else:
            assert_gap_close(p["Z"], p["y"], p["ours"][k], p["theirs"][k], *args)


def assert_gap_close(Z, y, ours, theirs, lam, alpha, penalty):
    """Wide data: no curvature in every direction, so the duality gap
    certifies instead. Both fits pass KKT (the caps carry the teeth); for
    alpha < 1, ridge curvature bounds each fit's coefficients' distance from
    the optimum by its gap, so the two lie within the sum. The lasso has no
    such bound when p > n: its optimum need not be unique."""
    for theta in (ours, theirs):
        assert standardized_kkt(Z, y, theta, lam, alpha, penalty) < KKT_TOL
    gaps = [duality_gap(Z, y, theta, lam, alpha, penalty) for theta in (ours, theirs)]
    assert all(np.isfinite(gaps))
    if alpha < 1:
        radii = [gap_coefficient_distance(g, lam, alpha, penalty) for g in gaps]
        assert np.linalg.norm(ours[1:] - theirs[1:]) <= sum(radii)


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
    """The regression covers both the extended range and the later clip."""
    _, glmnet = fixtures[RARE.name]
    data, _ = rglmnet.prepare_case(RARE, tmp_path)
    assert clipped_lambdas(data["X"], glmnet).any()
    eta = glmnet["intercept"] + data["X"] @ glmnet["beta"]
    maximum = np.abs(eta).max(axis=0)
    new_cutoff = np.log(1 / 1e-9 - 1)
    assert ((maximum > ETA_CLIP) & (maximum < new_cutoff)).any()
    assert (maximum > new_cutoff).any()


def test_clipped_lambdas_pass_kkt(fixtures, tmp_path):
    """Default fits pass KKT even past R biglasso's |eta| > 10 cutoff."""
    _, glmnet = fixtures[RARE.name]
    data, _ = rglmnet.prepare_case(RARE, tmp_path)
    p = fit_glmnet_grid(RARE, data, glmnet)
    clipped = np.flatnonzero(clipped_lambdas(data["X"], glmnet))
    residuals = {
        int(k): standardized_kkt(
            p["Z"], p["y"], p["ours"][k], p["lambdas"][k], RARE.alpha, p["penalty"]
        )
        for k in clipped
    }
    failing = {k: f"{r:.1e}" for k, r in residuals.items() if not r < KKT_TOL}
    assert not failing, f"KKT residual above KKT_TOL at clipped lambdas {failing}"


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
