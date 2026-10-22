"""Agreement with R biglasso, in three tiers.

- default: compare against stored R biglasso 1.7.2 output (no R needed)
- `-m r`: run the same grid live in R (Rscript, or CONTAINER + R_IMAGE)
- `-m soak`: larger memory-mapped problems, live in R; meant for a beefy box
"""

import warnings

import pytest
import rcompare

FAST = rcompare.fast_grid()
SOAK = rcompare.soak_grid()

needs_r = pytest.mark.skipif(
    not rcompare.r_available(), reason="R not found (set RSCRIPT, or CONTAINER)"
)


def _assert_agree(py, r):
    problems = rcompare.compare(py, r)
    assert not problems, "\n".join(problems)


@pytest.fixture(scope="session")
def fixtures():
    return rcompare.load_fixtures()


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("case", FAST, ids=lambda c: c.name)
def test_matches_stored_r_output(case, fixtures, tmp_path):
    digest, lambdas, r = fixtures[case.name]
    py, new_digest, _ = rcompare.prepare_case(case, tmp_path, lambdas)
    assert new_digest == digest, (
        "Generated data differs from what the stored R output was fit on; "
        "regenerate with scripts/make_r_fixtures.py"
    )
    _assert_agree(py, r)


@pytest.fixture(scope="session")
def live_fast(tmp_path_factory):
    """Fit the whole fast grid in one R session."""
    root = tmp_path_factory.mktemp("r-live")
    prepared = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for case in FAST:
            prepared[case.name] = rcompare.prepare_case(case, root / case.name)[0]
    outputs = rcompare.run_r([root / c.name for c in FAST])
    return {c.name: (prepared[c.name], r) for c, r in zip(FAST, outputs, strict=True)}


@pytest.mark.r
@needs_r
@pytest.mark.parametrize("case", FAST, ids=lambda c: c.name)
def test_matches_live_r(case, live_fast):
    py, r = live_fast[case.name]
    _assert_agree(py, r)


@pytest.mark.soak
@needs_r
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("case", SOAK, ids=lambda c: c.name)
def test_soak_matches_live_r(case, tmp_path):
    py, _, _ = rcompare.prepare_case(case, tmp_path)
    (r,) = rcompare.run_r([tmp_path])
    print(
        f"\n{case.name}: python {py['seconds']:.1f}s, R {r.get('seconds', float('nan')):.1f}s"
    )
    _assert_agree(py, r)
