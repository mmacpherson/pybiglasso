"""The Rust backend's two ways to fit row_indices: gathering the rows, or a 0/1
mask over every row. Each is checked against a fit on a copy of the rows."""

import numpy as np
import pytest
from scipy.special import expit
from support.paths import assert_bitwise, assert_same_trajectory

import pybiglasso as bl

_rust = pytest.importorskip("pybiglasso._rust")

# the C-order cases are deliberate
pytestmark = pytest.mark.filterwarnings("ignore:The design matrix is not in Fortran")


@pytest.fixture
def fit(monkeypatch):
    """biglasso() on the Rust backend, through a forced row path; returns the
    fit and the path that ran."""
    direct = _rust.fit_path
    ran = []

    def fit(X, y, row_path="auto", **kwargs):
        def recording(*args, **kw):
            path = direct(*args, **kw, row_path=row_path)
            ran.append(path["row_path"])
            return path

        monkeypatch.setattr(_rust, "fit_path", recording)
        result = bl.biglasso(X, y, backend="rust", family="binomial", **kwargs)
        return result, ran.pop()

    return fit


def data(n, p, seed=0, order="F", dtype=np.float64):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = rng.binomial(1, expit(X[:, :3].sum(axis=1))).astype(np.float64)
    return np.asarray(X, dtype=dtype, order=order), y


SHAPES = dict(wide=(300, 500), tall=(20_000, 8))


@pytest.mark.parametrize("row_path", ["gather", "mask"])
@pytest.mark.parametrize("sort", [True, False], ids=["sorted", "shuffled"])
@pytest.mark.parametrize("share", [0.6, 0.9])
@pytest.mark.parametrize("order", ["F", "C"])
@pytest.mark.parametrize("shape", SHAPES)
def test_rows_match_a_copy(fit, shape, order, share, sort, row_path):
    X, y = data(*SHAPES[shape], order=order)
    rng = np.random.default_rng(1)
    rows = rng.choice(len(y), int(share * len(y)), replace=False)
    if sort:
        rows.sort()
    subset, ran = fit(X, y, row_path, row_indices=rows, nlambda=20)
    copy, _ = fit(np.asarray(X[rows], order=order), y[rows], nlambda=20)
    assert ran == row_path
    assert_same_trajectory(subset, copy)


def test_mask_matches_a_copy_in_float32(fit):
    X, y = data(300, 500, dtype=np.float32)
    rows = np.arange(len(y))[np.arange(len(y)) % 10 != 3]
    subset, ran = fit(X, y, "mask", row_indices=rows, nlambda=20)
    copy, _ = fit(np.asfortranarray(X[rows]), y[rows], nlambda=20)
    assert ran == "mask"
    assert_same_trajectory(subset, copy)


@pytest.mark.parametrize(
    "rows,expected",
    [
        (np.arange(50), "mask"),  # half of X: MASK_SHARE
        (np.arange(100)[::-1], "mask"),  # every row, in reverse
        (np.arange(49), "gather"),
        (np.r_[np.arange(60), 0], "gather"),  # a repeated row
    ],
    ids=["half", "all-reversed", "under-half", "repeat"],
)
def test_auto_chooses_by_share_and_repeats(fit, rows, expected):
    X, y = data(100, 5)
    assert fit(X, y, row_indices=rows, nlambda=5)[1] == expected


@pytest.mark.parametrize("n_threads", [2, 3, 8])
def test_mask_is_bit_identical_across_threads(fit, n_threads):
    # tall enough that threads share each column's chunks
    X, y = data(40_000, 6)
    rows = np.arange(len(y))[np.arange(len(y)) % 5 != 0]
    one, _ = fit(X, y, "mask", row_indices=rows, nlambda=10, n_threads=1)
    many, ran = fit(X, y, "mask", row_indices=rows, nlambda=10, n_threads=n_threads)
    assert ran == "mask"
    assert_bitwise(one, many)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 1e300, -1e300], ids=str)
def test_unsafe_left_out_rows_gather_instead(fit, value):
    # the mask would sweep the left-out rows with weight 0, and 0 * NaN is NaN
    X, y = data(200, 30)
    rows = np.arange(len(y))[np.arange(len(y)) % 5 != 0]
    X[0, 7] = value  # row 0 is left out
    masked, ran = fit(X, y, "mask", row_indices=rows, nlambda=10)
    gathered, _ = fit(X, y, "gather", row_indices=rows, nlambda=10)
    assert ran == "gather"
    assert_bitwise(masked, gathered)


@pytest.mark.parametrize("value", [10.0, -1e100], ids=str)
def test_large_finite_left_out_rows_keep_the_mask(fit, value):
    # far outside the fitted rows' range, but z^2 stays finite (LEFT_OUT_Z_MAX)
    X, y = data(200, 30)
    rows = np.arange(len(y))[np.arange(len(y)) % 5 != 0]
    copy, _ = fit(np.asfortranarray(X[rows]), y[rows], nlambda=10)
    X[0, 7] = value
    masked, ran = fit(X, y, "mask", row_indices=rows, nlambda=10)
    assert ran == "mask"
    assert_same_trajectory(masked, copy)


def test_left_out_rows_of_unused_columns_are_ignored(fit):
    # a column constant in the fitted rows is never swept
    X, y = data(200, 30)
    rows = np.arange(len(y))[np.arange(len(y)) % 5 != 0]
    X[rows, 7] = 1.0
    X[0, 7] = np.nan
    assert fit(X, y, "mask", row_indices=rows, nlambda=10)[1] == "mask"


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf], ids=str)
def test_left_out_responses_may_be_anything(fit, value):
    # y need only be finite in the rows being fit, and NaN * 0 is NaN
    X, y = data(200, 30)
    rows = np.arange(len(y))[np.arange(len(y)) % 5 != 0]
    copy, _ = fit(np.asfortranarray(X[rows]), y[rows], nlambda=10)
    y[0] = value
    masked, ran = fit(X, y, "mask", row_indices=rows, nlambda=10)
    assert ran == "mask"
    assert_same_trajectory(masked, copy)


def test_mask_centers_shuffled_rows_as_gathering_does(fit):
    # far from zero, the center's rounding moves every centered value, so it
    # must sum the rows in row_indices' order, as the gather and a copy do
    rng = np.random.default_rng(3)
    X = np.asfortranarray(1e12 + 0.01 * rng.normal(size=(300, 3)))
    y = rng.binomial(1, expit((X[:, 0] - 1e12) / 0.01)).astype(np.float64)
    rows = rng.permutation(300)[:240]
    masked, ran = fit(X, y, "mask", row_indices=rows, nlambda=20)
    copy, _ = fit(np.asfortranarray(X[rows]), y[rows], nlambda=20)
    assert ran == "mask"
    np.testing.assert_array_equal(masked.center, copy.center)
    np.testing.assert_array_equal(masked.scale, copy.scale)
    assert_same_trajectory(masked, copy)
