"""When a fit warns that it failed to converge within max_iter.

A cap the fit never needed can't change it: fitting again with max_iter equal
to the most iterations any lambda took must reproduce the uncapped path
exactly. Default mode then warns only if some lambda stopped unconverged;
r_compatible keeps R biglasso's rule, a warning whenever any lambda used all
max_iter iterations (checked against R itself in tests/support/rcompare.py).
"""

import warnings

import numpy as np
import pytest
from support import rcompare

import pybiglasso as bl

MESSAGE = "Failed to converge within max_iter"


def data(family):
    rng = np.random.default_rng(31)
    X = np.asfortranarray(rng.standard_normal((200, 30)))
    eta = X[:, :4] @ np.array([1.5, -1.0, 0.8, 0.5])
    if family == "gaussian":
        y = eta + rng.standard_normal(200)
    else:
        y = (rng.random(200) < 1 / (1 + np.exp(-eta))).astype(float)
    return X, y


def fit(X, y, **options):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        path = bl.biglasso(X, y, nlambda=20, n_threads=1, **options)
    return path, [str(w.message) for w in caught if MESSAGE in str(w.message)]


def same_path(a, b):
    assert np.array_equal(a.lambdas, b.lambdas)
    assert np.array_equal(a.iterations, b.iterations)
    assert np.array_equal(a.intercept, b.intercept)
    assert (a.coef != b.coef).nnz == 0
    assert np.array_equal(a.deviance, b.deviance)


@pytest.mark.parametrize("family", ["gaussian", "binomial"])
@pytest.mark.parametrize("r_compatible", [False, True])
def test_a_cap_at_the_most_iterations_used_changes_nothing(family, r_compatible):
    X, y = data(family)
    options = dict(family=family, r_compatible=r_compatible)
    uncapped, uncapped_warnings = fit(X, y, **options)
    assert uncapped_warnings == []
    most = int(uncapped.iterations.max())
    # the case this test is about: a lambda that converged on its last
    # allowed iteration
    assert most >= 2 and (uncapped.iterations == most).sum() >= 1

    capped, capped_warnings = fit(X, y, max_iter=most, **options)

    same_path(capped, uncapped)
    if r_compatible:
        # R's rule: any(iter == max.iter), converged or not
        assert len(capped_warnings) == (uncapped.iterations == most).sum()
    else:
        assert capped_warnings == []


@pytest.mark.parametrize("family", ["gaussian", "binomial"])
@pytest.mark.parametrize("r_compatible", [False, True])
def test_a_cap_below_that_still_warns(family, r_compatible):
    X, y = data(family)
    options = dict(family=family, r_compatible=r_compatible)
    uncapped, _ = fit(X, y, **options)
    most = int(uncapped.iterations.max())

    capped, capped_warnings = fit(X, y, max_iter=most - 1, **options)

    # the lambdas that needed `most` iterations can't have converged
    assert capped.iterations.max() == most - 1
    assert len(capped_warnings) >= 1


@pytest.mark.parametrize("capped", sorted(rcompare.CONVERGED_AT_MAX_ITER))
def test_r_converges_on_the_last_allowed_iteration_of_these_cases(capped):
    # From R's stored output alone: the capped fit takes the same iterations
    # as R's uncapped fit of the same problem, whose most is the cap, and R
    # still warns
    stored = rcompare.load_fixtures()
    r_capped = stored[capped][2]
    r_uncapped = stored[rcompare.CONVERGED_AT_MAX_ITER[capped]][2]
    case = next(
        c for c in rcompare.fast_grid() + rcompare.gaussian_grid() if c.name == capped
    )
    assert np.array_equal(r_capped["iter"], r_uncapped["iter"])
    assert max(r_uncapped["iter"]) == case.max_iter
    assert rcompare.warning_kinds(r_capped["warnings"]) == ["max_iter"]
    assert rcompare.warning_kinds(r_uncapped["warnings"]) == []
