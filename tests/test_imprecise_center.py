import warnings

import numpy as np
import pytest

import pybiglasso as bl


def _fit(X, y, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bl.biglasso(X, y, family="gaussian", nlambda=5, **kwargs)
    return [str(w.message) for w in caught if "too far from zero" in str(w.message)]


@pytest.fixture
def near_2_54():
    # Values near 2**54 that vary by a few units. Adjacent float64 values are
    # 4 apart there, and a 257-row sum is near 2**62 (spacing 1024), so no
    # binary64 center is within the needed fraction of the columns' spread.
    rng = np.random.default_rng(5)
    n = 257
    X = np.asfortranarray(
        np.column_stack(
            [
                2**54 + 4 * rng.integers(0, 4, n),  # far from zero
                rng.normal(size=n),  # ordinary
                1e8 + rng.normal(size=n),  # far, but centered precisely
            ]
        ).astype(np.float64)
    )
    y = X[:, 1] + rng.normal(size=n)
    return X, y


def test_warns_and_names_columns_too_far_from_zero(near_2_54):
    X, y = near_2_54
    messages = _fit(X, y, feature_names=["counter", "signal", "offset"])
    assert len(messages) == 1
    assert "counter" in messages[0]
    assert "signal" not in messages[0] and "offset" not in messages[0]


def test_warns_on_uint64_and_with_row_indices(near_2_54):
    X, y = near_2_54
    assert _fit(X.astype(np.uint64)[:, :1].copy(order="F"), y)
    assert _fit(X, y, row_indices=np.arange(0, len(y), 2))


def test_no_warning_once_shifted(near_2_54):
    X, y = near_2_54
    shifted = X.copy(order="F")
    shifted[:, 0] -= 2**54
    assert _fit(shifted, y) == []


def test_no_warning_on_ordinary_data():
    rng = np.random.default_rng(6)
    X = np.asfortranarray(rng.normal(size=(500, 20)) * 1e3 + 1e5)
    assert _fit(X, rng.normal(size=500)) == []
