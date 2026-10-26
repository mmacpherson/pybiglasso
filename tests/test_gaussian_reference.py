"""Gaussian's outside oracle, before porting the production kernel."""

import numpy as np
import pytest
from support import rcompare, reference
from support.certificates import (
    KKT_TOL,
    gaussian_subgradient_residual,
    standardize,
    standardized_theta,
    with_ones,
)

CASES = rcompare.gaussian_grid()


@pytest.fixture(scope="session")
def fixtures():
    return rcompare.load_fixtures()


def options(case, data, lambdas):
    return dict(
        family="gaussian",
        alpha=case.alpha,
        nlambda=case.nlambda,
        lambdas=lambdas,
        eps=rcompare.EPS,
        lambda_min=case.resolved_lambda_min,
        lambda_log_scale=case.lambda_log_scale,
        penalty_factor=data["penalty_factor"],
        row_indices=data["row_indices"],
        dfmax=case.dfmax,
        max_iter=case.max_iter,
        r_compatible=True,
    )


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_gaussian_reference_matches_stored_r(case, fixtures, tmp_path):
    digest, lambdas, expected = fixtures[case.name]
    data = rcompare.make_case_data(case, tmp_path)
    assert rcompare.data_digest(data) == digest
    got = reference.fit_path(data["X"], data["y"], **options(case, data, lambdas))
    assert not (problems := rcompare.compare(got, expected)), "\n".join(problems)


@pytest.mark.r
@pytest.mark.skipif(not rcompare.r_available(), reason="R not available")
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_gaussian_reference_matches_live_r(case, live_gaussian, tmp_path):
    lambdas, expected = live_gaussian[case.name]
    data = rcompare.make_case_data(case, tmp_path)
    got = reference.fit_path(data["X"], data["y"], **options(case, data, lambdas))
    assert not (problems := rcompare.compare(got, expected)), "\n".join(problems)


@pytest.fixture(scope="session")
def live_gaussian(tmp_path_factory):
    root = tmp_path_factory.mktemp("gaussian-r-live")
    dirs = [root / case.name for case in CASES]
    grids = [
        rcompare.prepare_case(case, d, fit_python=False)[2]
        for case, d in zip(CASES, dirs, strict=True)
    ]
    outputs = rcompare.run_r(dirs)
    return {
        case.name: (grid, output)
        for case, grid, output in zip(CASES, grids, outputs, strict=True)
    }


@pytest.mark.parametrize("alpha", [0.1, 0.5, 1.0])
def test_gaussian_one_feature_closed_form(alpha):
    # Centered, population variance one. The optimum is soft-threshold(z, l1)/(1+l2).
    X = np.array([[-1.0], [1.0], [-1.0], [1.0]])
    y = 3 + 2 * X[:, 0]
    lambdas = np.array([3.0, 0.5, 0.0])
    fit = reference.fit_path(X, y, family="gaussian", alpha=alpha, lambdas=lambdas)
    expected = np.maximum(2 - lambdas * alpha, 0) / (1 + lambdas * (1 - alpha))
    np.testing.assert_allclose(fit["beta"][0], 3)
    np.testing.assert_allclose(fit["beta"][1], expected)
    np.testing.assert_allclose(fit["loss"], 4 * (2 - expected) ** 2)


def test_gaussian_comparison_rejects_wrong_objective(tmp_path, fixtures):
    case = next(c for c in CASES if c.alpha == 0.5 and c.frac_mono == 0)
    _, lambdas, expected = fixtures[case.name]
    data = rcompare.make_case_data(case, tmp_path)
    opts = options(case, data, lambdas)
    opts["alpha"] = 1.0
    wrong = reference.fit_path(data["X"], data["y"], **opts)
    assert any(p.startswith("beta:") for p in rcompare.compare(wrong, expected))


@pytest.mark.parametrize(
    "case",
    [
        rcompare.Case(100, 10, family="gaussian", alpha=0.5, penalty="varied"),
        rcompare.Case(60, 200, family="gaussian", alpha=0.5),
        rcompare.Case(100, 10, family="gaussian", alpha=0.5, penalty="zeros"),
    ],
    ids=lambda c: c.name,
)
def test_gaussian_reference_default_satisfies_independent_kkt(case, tmp_path):
    data = rcompare.make_case_data(case, tmp_path)
    # Tighten convergence to check the objective independently of the R
    # trajectory; keep the existing KKT cap, not a cap fitted to the output.
    fit = reference.fit_path(
        data["X"],
        data["y"],
        family="gaussian",
        alpha=case.alpha,
        penalty_factor=data["penalty_factor"],
        nlambda=5,
        eps=1e-18,
        max_iter=10_000,
    )
    assert not fit["events"]
    Xs, center, scale = standardize(data["X"])
    theta = np.column_stack(
        [standardized_theta(b[0], b[1:], center, scale) for b in fit["beta"].T]
    )
    pf = 1.0 if data["penalty_factor"] is None else data["penalty_factor"]
    residual = gaussian_subgradient_residual(
        with_ones(Xs), data["y"], theta, fit["lambda_"], case.alpha, pf
    )
    assert np.max(np.abs(residual)) <= KKT_TOL
