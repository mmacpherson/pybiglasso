"""Independent glmnet solutions of the Gaussian default objective."""

import numpy as np
import pytest
from support import rcompare, rglmnet
from support.certificates import (
    KKT_TOL,
    gap_coefficient_distance,
    gaussian_curvature,
    gaussian_duality_gap,
    gaussian_subgradient_residual,
    standardize,
    standardized_theta,
    with_ones,
)

import pybiglasso as bl

CASES = rglmnet.gaussian_grid()


def assert_matches(case, data, expected):
    assert expected["jerr"] == 0 and not expected["warnings"], expected["warnings"]
    X, y, pf = data["X"], data["y"], data["penalty_factor"]
    pf = np.ones(X.shape[1]) if pf is None else pf
    lambdas = expected["lambda_"]
    assert len(lambdas) == case.nlambda
    fit = bl.biglasso(
        X,
        y,
        family="gaussian",
        backend="rust",
        alpha=case.alpha,
        lambdas=lambdas,
        penalty_factor=pf,
        eps=1e-18,
        max_iter=10000,
    )
    Xs, center, scale = standardize(X)
    Z = with_ones(Xs)
    for k, lam in enumerate(lambdas):
        a = standardized_theta(
            fit.intercept[k], fit.coef[:, [k]].toarray().ravel(), center, scale
        )
        b = standardized_theta(
            expected["intercept"][k], expected["beta"][:, k], center, scale
        )
        residuals = [
            gaussian_subgradient_residual(Z, y, t, lam, case.alpha, pf) for t in (a, b)
        ]
        assert max(np.max(np.abs(r)) for r in residuals) <= KKT_TOL
        if case.nrow > case.ncol:
            curvature = gaussian_curvature(Z, lam, case.alpha, pf)
            distance = sum(np.linalg.norm(r) for r in residuals) / curvature
            # Account for independent matrix-product rounding near an exact
            # optimum, using the scale and operation count, not fit differences.
            rounding = (
                64
                * len(y)
                * np.finfo(float).eps
                * (1 + np.linalg.norm(y) + np.linalg.norm(a) + np.linalg.norm(b))
            )
            assert np.linalg.norm(a - b) <= distance + rounding / curvature
        else:
            gaps = [gaussian_duality_gap(Z, y, t, lam, case.alpha, pf) for t in (a, b)]
            if case.alpha < 1:
                assert np.linalg.norm(a[1:] - b[1:]) <= sum(
                    gap_coefficient_distance(g, lam, case.alpha, pf) for g in gaps
                )


@pytest.fixture(scope="module")
def fixtures():
    return rglmnet.load_fixtures()


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_gaussian_matches_stored_glmnet(case, fixtures, tmp_path):
    data, digest = rglmnet.prepare_case(case, tmp_path)
    expected_digest, expected = fixtures[case.name]
    assert digest == expected_digest
    assert_matches(case, data, expected)


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    root = tmp_path_factory.mktemp("gaussian-glmnet-live")
    dirs = [root / c.name for c in CASES]
    data = [rglmnet.prepare_case(c, d)[0] for c, d in zip(CASES, dirs, strict=True)]
    outputs = rglmnet.run_glmnet(dirs)
    return {
        c.name: (inputs, out)
        for c, inputs, out in zip(CASES, data, outputs, strict=True)
    }


@pytest.mark.r
@pytest.mark.skipif(not rcompare.r_available(), reason="R not available")
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_gaussian_matches_live_glmnet(case, live):
    data, expected = live[case.name]
    assert_matches(case, data, expected)
