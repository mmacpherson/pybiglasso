"""Public storage/thread invariances complement the native shadow oracle."""

import dataclasses
import warnings

import numpy as np
import pyarrow as pa
import pyarrow.feather as pf
import pytest
from support.path_identity import path_digest

import pybiglasso as bl
from pybiglasso.arrow import _open_mapped


@pytest.mark.parametrize("alpha", [1.0, 0.5])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("rows", ["all", "mask", "repeated"])
def test_gaussian_numeric_outputs_and_warnings_across_threads_and_storage(
    alpha, packed, rows, tmp_path
):
    rng = np.random.default_rng(1853)
    X = np.asfortranarray(rng.integers(0, 2, (129, 9)))
    if not packed:
        X = np.asfortranarray(X + rng.normal(scale=0.25, size=X.shape))
    y = X[:, 0] - 0.75 * X[:, 1] + rng.normal(scale=0.1, size=len(X))
    indices = {
        "all": None,
        "mask": np.arange(1, len(X) - 1),
        "repeated": np.array([1, 7, 1, 99, 3, 128, 20, 17, 7, 3, 55]),
    }[rows]
    columns = {}
    for j in range(X.shape[1]):
        column = X[:, j].astype(bool) if packed else X[:, j]
        # Per-column partitions differ; each slice starts inside its own buffer.
        parts = []
        size = 7 + 2 * j
        for start in range(0, len(X), size):
            values = column[start : start + size]
            padded = np.concatenate([np.zeros(3, dtype=column.dtype), values])
            parts.append(pa.array(padded).slice(3))
        columns[f"x{j}"] = pa.chunked_array(parts)
    table = pa.table(columns)
    file = tmp_path / "X.feather"
    pf.write_feather(table, file, compression="uncompressed", chunksize=13)
    mapped = _open_mapped(file)
    expected = None
    for storage in [X, table, mapped]:
        for threads in [1, 2, 4, 6]:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                path = bl.biglasso(
                    storage,
                    y,
                    alpha=alpha,
                    nlambda=25,
                    row_indices=indices,
                    n_threads=threads,
                )
            actual = (
                path_digest(path),
                [(w.category.__name__, str(w.message)) for w in caught],
            )
            if expected is None:
                expected = actual
            assert actual == expected


def test_path_digest_frames_dtype_shape_and_sparse_storage():
    # Pure hand-built path: these checks never generate expected solver values.
    import scipy.sparse as sps

    path = bl.BiglassoPath(
        lambdas=np.array([1.0, 0.5]),
        intercept=np.zeros(2),
        coef=sps.csc_array([[0.0, 1.0], [0.0, 0.0]]),
        deviance=np.array([2.0, 1.0]),
        iterations=np.array([0, 2]),
        n_rejected=np.array([2, 1]),
        center=np.zeros(2),
        scale=np.ones(2),
        penalty_factor=np.ones(2),
        feature_names=("a", "b"),
        alpha=1.0,
        family="gaussian",
    )
    digest = path_digest(path)
    assert path_digest(dataclasses.replace(path, lambdas=path.lambdas.copy())) == digest
    # Identical raw payload, different dtype or shape.
    assert (
        path_digest(dataclasses.replace(path, lambdas=path.lambdas.view("i8")))
        != digest
    )
    assert (
        path_digest(dataclasses.replace(path, lambdas=path.lambdas.reshape(1, 2)))
        != digest
    )
    # Same dense values, different sparse representation.
    assert path_digest(dataclasses.replace(path, coef=path.coef.tocsr())) != digest
    sparse = path.coef.copy()
    dtype = "i4" if sparse.indices.dtype.itemsize == 8 else "i8"
    sparse.indices = sparse.indices.astype(dtype)
    assert path_digest(dataclasses.replace(path, coef=sparse)) != digest
