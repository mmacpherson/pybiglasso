"""X is never copied: fitting, cross-validation, and scoring read a memmap in place.

Peak memory is measured with tracemalloc, which sees numpy's allocations; a
float64 copy of X, or of all its rows for the active columns, would exceed the
bounds below several times over.
"""

import tracemalloc

import numpy as np
import pytest

import pybiglasso as bl
from pybiglasso import path as path_module

NROW, NCOL = 4_000, 2_500


@pytest.fixture(scope="module")
def genotypes(tmp_path_factory):
    """An int8 memmap, 10 MB on disk; 80 MB as float64."""
    data = bl.common.make_biglasso_dataset(nrow=NROW, ncol=NCOL, seed=3)
    file = tmp_path_factory.mktemp("ooc") / "X.npy"
    X = np.lib.format.open_memmap(
        file, mode="w+", dtype=np.int8, shape=(NROW, NCOL), fortran_order=True
    )
    X[:] = data["X"]
    X.flush()
    del X
    X, y = np.load(file, mmap_mode="r"), data["y"].astype(np.float64)
    # compile the kernels for read-only int8 input outside the measurements
    bl.biglasso(X, y, nlambda=2)
    bl.biglasso(X, y, nlambda=2, row_indices=np.arange(0, NROW, 2), n_threads=1)
    return X, y


@pytest.fixture
def small_chunks(monkeypatch):
    monkeypatch.setattr(path_module, "CHUNK_ELEMENTS", 1 << 16)


def peak_bytes(fn):
    tracemalloc.start()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def float64_size(X):
    return X.shape[0] * X.shape[1] * 8


def test_cv_reads_the_memmap_in_place(genotypes, small_chunks, monkeypatch):
    pytest.importorskip("sklearn")
    import pybiglasso.sklearn as bsk

    X, y = genotypes
    seen = []

    def spy(X_, *args, **kwargs):
        seen.append(X_)
        return bl.biglasso(X_, *args, **kwargs)

    monkeypatch.setattr(bsk, "biglasso", spy)
    cv = bsk.BiglassoClassifierCV(l1_ratio=0.5, n_alphas=5, cv=3, n_jobs=1)
    peak = peak_bytes(lambda: cv.fit(X, y))

    assert len(seen) == 4  # the full path, then one per fold
    for X_ in seen:
        assert X_.dtype == np.int8
        assert np.shares_memory(X_, X)
    assert peak < 0.1 * float64_size(X), f"peak {peak / 1e6:.1f} MB"


@pytest.mark.parametrize("subset", [False, True])
def test_scoring_is_bounded_and_exact(genotypes, small_chunks, subset):
    X, y = genotypes
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(NROW, NROW // 2, replace=False)) if subset else None
    path = bl.biglasso(X, y, alpha=0.5, nlambda=20, lambda_min=0.05)
    assert path.n_nonzero[-1] > 100  # enough active columns to matter

    eta = []
    peak = peak_bytes(lambda: eta.append(path.decision_function(X, row_indices=rows)))
    assert peak < 0.1 * float64_size(X), f"peak {peak / 1e6:.1f} MB"

    dense = np.asarray(X, dtype=np.float64)
    if rows is not None:
        dense = dense[rows]
    expected = path.intercept + dense @ path.coef.toarray()
    np.testing.assert_allclose(eta[0], expected, rtol=1e-12, atol=1e-12)


def test_prediction_rejects_nonfinite_values(genotypes):
    X, y = genotypes
    path = bl.biglasso(X, y, nlambda=5)
    bad = np.asfortranarray(X[:50], dtype=np.float64)
    bad[3, path.coef.indices[0]] = np.nan
    with pytest.raises(ValueError, match="NaN or infinity"):
        path.decision_function(bad)


def test_sklearn_converts_only_what_the_kernels_cannot_read():
    pytest.importorskip("sklearn")
    from pybiglasso.sklearn import BiglassoClassifier, _as_design

    rng = np.random.default_rng(1)
    X = rng.integers(0, 3, size=(200, 5)).astype(np.int8)
    for dtype in (np.int8, np.uint8, np.int64, np.float32, np.float64):
        Xd = np.asfortranarray(X.astype(dtype))
        assert _as_design(Xd) is Xd
    assert _as_design(X.astype(bool)).dtype == np.uint8
    assert _as_design(X.astype(np.float16)).dtype == np.float32

    y = (X[:, 0] + rng.normal(size=200) > 1).astype(int)
    ref = BiglassoClassifier(alpha=0.01).fit(X.astype(np.float64), y)
    for dtype in (np.int8, np.uint8, np.float16):
        est = BiglassoClassifier(alpha=0.01).fit(X.astype(dtype), y)
        np.testing.assert_allclose(est.coef_, ref.coef_, rtol=1e-6, atol=1e-9)
        np.testing.assert_array_equal(est.predict(X.astype(dtype)), ref.predict(X))


def test_auc_cv_rejects_a_one_class_held_out_fold():
    pytest.importorskip("sklearn")
    from sklearn.model_selection import KFold

    from pybiglasso.sklearn import BiglassoClassifierCV

    rng = np.random.default_rng(2)
    X = np.asfortranarray(rng.normal(size=(60, 4)))
    y = np.r_[np.zeros(30), np.ones(30)]  # unshuffled: KFold(2) splits by class
    cv = BiglassoClassifierCV(cv=KFold(3), scoring="auc", n_alphas=5)
    with pytest.raises(ValueError, match="all one class"):
        cv.fit(X, y)


@pytest.mark.parametrize("chunk", [50, 7, 1 << 22])
def test_linear_predictor_blocks_rows_and_columns(monkeypatch, chunk):
    """More active columns than fit in one block: X is read in column blocks too."""
    import scipy.sparse as sps

    monkeypatch.setattr(path_module, "CHUNK_ELEMENTS", chunk)
    rng = np.random.default_rng(4)
    X = np.asfortranarray(rng.integers(0, 3, size=(300, 200)).astype(np.int8))
    coef = sps.random_array((200, 6), density=0.6, random_state=rng, format="csc")
    intercept = rng.normal(size=6)
    rows = rng.permutation(300)[:150]
    for r in (None, rows):
        eta = path_module.linear_predictor(X, coef, intercept, r)
        Xr = X if r is None else X[r]
        expected = intercept + Xr.astype(np.float64) @ coef.toarray()
        np.testing.assert_allclose(eta, expected, rtol=1e-12, atol=1e-12)


def test_linear_predictor_with_no_active_columns():
    X = np.ones((10, 3), dtype=np.int8)
    eta = path_module.linear_predictor(X, np.zeros((3, 2)), np.array([0.5, -1.0]))
    np.testing.assert_array_equal(eta, np.tile([0.5, -1.0], (10, 1)))
