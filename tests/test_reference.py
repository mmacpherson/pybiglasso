"""The solver against the plain numpy reference, `tests/reference.py`.

The reference is first held to R itself, on the stored output of the whole
comparison grid. It then checks what R can't: the default (not r_compatible)
updates and loss, the default grid with unpenalized columns, and fitting past
saturation, to the R comparison's tolerances and with equal
iteration counts and rejections.
"""

import warnings

import numpy as np
import pytest
import rcompare
import reference

import pybiglasso as bl

FAST = rcompare.fast_grid()


@pytest.fixture(scope="session")
def fixtures():
    return rcompare.load_fixtures()


def fit_options(case, data, lambdas=None, **extra):
    return dict(
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
        **extra,
    )


def fit_reference(data, options):
    return reference.fit_path(np.asarray(data["X"]), data["y"], **options)


def fit_solver(data, options):
    """A pybiglasso fit, in R biglasso's layout."""
    fit = bl.biglasso(data["X"], data["y"], **options)
    return dict(
        center=fit.center,
        scale=fit.scale,
        lambda_=fit.lambdas,
        loss=fit.deviance / 2,
        iter=fit.iterations,
        rejections=fit.n_rejected,
        beta=np.vstack([fit.intercept, fit.coef.toarray()]),
    )


def _assert_agree(got, want):
    problems = rcompare.compare(got, want)
    assert not problems, "\n".join(problems)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("case", FAST, ids=lambda c: c.name)
def test_reference_matches_stored_r_output(case, fixtures, tmp_path):
    digest, lambdas, r = fixtures[case.name]
    data = rcompare.make_case_data(case, tmp_path)
    assert rcompare.data_digest(data) == digest
    lambdas = lambdas if case.r_needs_lambdas else None
    options = fit_options(case, data, lambdas, r_compatible=True)
    _assert_agree(fit_reference(data, options), r)


# the default grid with unpenalized columns, which R fits differently, and
# past saturation, which R doesn't; the rest of the cases are from the R grid
BEYOND_R = [
    rcompare.Case(200, 30, alpha=0.8, penalty="zeros", tag="prefit"),
    rcompare.Case(80, 150, alpha=1.0, penalty="zeros", rows="half", tag="prefit"),
    rcompare.Case(30, 60, 11, 1.0, 50, lambda_min=1e-4, tag="past-saturation"),
]
DEFAULT_CASES = [c for c in FAST if c.nrow * c.ncol >= 3000 or c.tag] + BEYOND_R


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("r_compatible", [False, True], ids=["default", "r_compatible"])
@pytest.mark.parametrize("case", DEFAULT_CASES, ids=lambda c: c.name)
def test_solver_matches_reference(case, r_compatible, fixtures, tmp_path):
    data = rcompare.make_case_data(case, tmp_path)
    # supplied-lambda cases get the grid their stored R fit was given
    lambdas = fixtures[case.name][1] if case.user_lambdas else None
    options = fit_options(
        case,
        data,
        lambdas,
        r_compatible=r_compatible,
        stop_on_saturation=case.tag != "past-saturation",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        want = fit_reference(data, options)
    _assert_agree(fit_solver(data, options), want)


def test_solver_matches_reference_short_of_convergence(tmp_path):
    # stopped at max_iter, every lambda's iterate must still agree
    case = rcompare.Case(300, 40, alpha=0.5)
    data = rcompare.make_case_data(case, tmp_path)
    options = dict(fit_options(case, data), max_iter=3)
    with pytest.warns(UserWarning, match="max_iter"):
        got = fit_solver(data, options)
    want = fit_reference(data, options)
    assert any(kind == "max_iter" for kind, _ in want["events"])
    _assert_agree(got, want)


def test_comparison_detects_a_wrong_update(tmp_path):
    # the comparison can fail: R's elastic-net update is close to the default
    # one, and the reference must tell them apart
    case = rcompare.Case(300, 40, alpha=0.5)
    data = rcompare.make_case_data(case, tmp_path)
    got = fit_solver(data, fit_options(case, data))
    want = fit_reference(data, fit_options(case, data, r_compatible=True))
    assert rcompare.compare(got, want)


def test_comparison_detects_a_wrong_kkt_scan(tmp_path, monkeypatch):
    # the KKT scans use the residual from before the sweep that converged, as
    # R's do; scanning with the residual after it changes the rejections
    case = rcompare.Case(300, 40, alpha=0.9)
    data = rcompare.make_case_data(case, tmp_path)
    options = fit_options(case, data, r_compatible=True)
    got = fit_solver(data, options)
    _assert_agree(got, fit_reference(data, options))

    scan = reference.kkt_violations

    def after_the_sweep(problem, state, lam, columns):
        state.residual = problem.y - 1 / (1 + np.exp(-state.eta))
        return scan(problem, state, lam, columns)

    monkeypatch.setattr(reference, "kkt_violations", after_the_sweep)
    problems = rcompare.compare(got, fit_reference(data, options))
    assert any(p.startswith("rejections") for p in problems), problems
