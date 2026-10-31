"""Mapped IPC buffers retain their original physical representation."""

import gc
import math
import re
import struct
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from packaging.version import Version

from pybiglasso import biglasso
from pybiglasso.arrow import _ArrowMatrix, _open_mapped

pa = pytest.importorskip("pyarrow")


def write_ipc(path, arrays, *, compression=None, batches=1):
    batch = pa.record_batch(arrays, names=[f"x{i}" for i in range(len(arrays))])
    options = pa.ipc.IpcWriteOptions(compression=compression)
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, batch.schema, options=options) as writer:
            for _ in range(batches):
                writer.write_batch(batch)


def test_explicit_table_conversion_from_arrow_stream():
    values = np.asfortranarray(np.arange(123 * 2).reshape(123, 2) % 7, dtype=float)
    source = pa.table({"x0": values[:, 0], "x1": values[:, 1]})
    reader = pa.RecordBatchReader.from_stream(source)
    table = reader.read_all()
    target = np.arange(123) % 2
    same_path(biglasso(table, target, nlambda=5), biglasso(values, target, nlambda=5))


def _arrow_recipe(marker, document="docs/arrow.md"):
    documentation = Path(__file__).resolve().parents[1] / document
    recipes = re.findall(r"```python\n(.*?)\n```", documentation.read_text(), re.S)
    [recipe] = [block for block in recipes if marker in block]
    return recipe


def test_readme_arrow_example_verbatim(tmp_path, monkeypatch):
    import pyarrow.feather as pf

    rows = np.arange(123)
    rng = np.random.default_rng(28)
    x1 = rng.normal(size=len(rows))
    x2 = rows % 3 == 0
    response = x1 / 2 + rng.normal(size=len(rows))
    pf.write_feather(
        pa.table({"x1": x1, "x2": x2, "y": response, "id": ["unused"] * len(rows)}),
        tmp_path / "data.arrow",
        compression="uncompressed",
        chunksize=41,
    )
    recipe = _arrow_recipe("bl.biglasso(table.drop_columns", document="README.md")
    monkeypatch.chdir(tmp_path)
    namespace = {}
    exec(compile(recipe, "README.md", "exec"), namespace)
    np.save(tmp_path / "features.npy", np.asfortranarray(np.column_stack([x1, x2])))
    same_path(
        namespace["path"],
        biglasso(np.load(tmp_path / "features.npy", mmap_mode="r"), response),
    )
    assert list(namespace["path"].feature_names) == ["x1", "x2"]


def _mapped_read_allocations(recipe, directory):
    probe = (
        "import pyarrow as pa\nbefore = pa.total_allocated_bytes()\n"
        + recipe
        + "\nprint(pa.total_allocated_bytes() - before, pa.default_memory_pool().max_memory())"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=directory,
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(map(int, result.stdout.split()))


def _write_uncompressed_recipe_file(directory, extra_columns):
    import pyarrow.feather as pf

    rows = np.arange(12288)
    columns = {
        "x1": rows.astype(np.float64),
        "x2": rows % 2 == 0,
        "y": (rows % 3).astype(np.float64),
        "id": ["unused"] * len(rows),
        **{f"extra_{i}": rows.astype(np.float64) for i in range(extra_columns)},
    }
    pf.write_feather(
        pa.table(columns),
        directory / "data.feather",
        compression="uncompressed",
        chunksize=4096,
    )


@pytest.mark.parametrize("extra_columns", [0, 62])
def test_documented_mapped_read_then_select_allocates_zero(
    tmp_path, monkeypatch, extra_columns
):
    _write_uncompressed_recipe_file(tmp_path, extra_columns)
    recipe = _arrow_recipe("table = pf.read_table")
    assert _mapped_read_allocations(recipe, tmp_path) == (0, 0)
    namespace = {}
    monkeypatch.chdir(tmp_path)
    exec(compile(recipe, "docs/arrow.md", "exec"), namespace)
    assert namespace["X"].column_names == ["x1", "x2"]
    assert namespace["X"].column(0).num_chunks == 3
    np.testing.assert_array_equal(namespace["y"].to_numpy(), np.arange(12288) % 3)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="PyArrow 26.0.0 copies uncompressed batch bodies with columns=; remove this marker when fixed",
)
def test_feather_columns_projection_allocates_zero(tmp_path):
    _write_uncompressed_recipe_file(tmp_path, extra_columns=62)
    recipe = (
        "import pyarrow.feather as pf\n"
        "table = pf.read_table('data.feather', memory_map=True, columns=['x1', 'x2', 'y'])"
    )
    assert _mapped_read_allocations(recipe, tmp_path) == (0, 0)


@pytest.mark.parametrize("compression", ["lz4", "zstd"])
@pytest.mark.parametrize("family", ["binomial", "gaussian"])
def test_documented_batch_rewrite_preserves_values_and_complete_path(
    tmp_path, monkeypatch, compression, family
):
    import pyarrow.feather as pf

    rows = np.arange(123)
    rng = np.random.default_rng(57)
    x1 = rng.integers(-3, 4, len(rows)).astype(float) / 4
    x2 = rows % 3 == 0
    response = rng.integers(0, 2, len(rows)).astype(float)
    original = pa.table(
        {"x1": x1, "x2": x2, "y": response, "id": ["unused"] * len(rows)}
    )
    pf.write_feather(
        original, tmp_path / "data.feather", compression=compression, chunksize=41
    )
    monkeypatch.chdir(tmp_path)
    recipe = _arrow_recipe("pa.ipc.IpcWriteOptions(compression=None)")
    exec(compile(recipe, "docs/arrow.md", "exec"), {})
    rewritten = _open_mapped(tmp_path / "data-uncompressed.feather")
    assert rewritten.equals(original)
    assert rewritten.column(0).num_chunks == 3
    np.save(tmp_path / "features.npy", np.asfortranarray(np.column_stack([x1, x2])))
    same_path(
        biglasso(
            rewritten.select(["x1", "x2"]), rewritten["y"], family=family, nlambda=5
        ),
        biglasso(
            np.load(tmp_path / "features.npy", mmap_mode="r"),
            response,
            family=family,
            nlambda=5,
        ),
    )


@pytest.mark.parametrize(
    "kind", ["bool", "int8", "uint8", "int16", "float32", "float64"]
)
def test_values_are_readonly_views_of_the_mapping(tmp_path, monkeypatch, kind):
    values = np.arange(35) % 7
    values = (values % 2).astype(bool) if kind == "bool" else values.astype(kind)
    path = tmp_path / "design.arrow"
    write_ipc(path, [pa.array(values)])
    real_map = pa.memory_map
    mappings = []

    def capture(*args):
        source = real_map(*args)
        mappings.append(source.read_buffer(source.size()))
        source.seek(0)
        return source

    monkeypatch.setattr(pa, "memory_map", capture)
    X = _open_mapped(path)
    view, bit_offset, _ = _ArrowMatrix(X)._columns[0][0]
    assert X.shape == (35, 1)
    assert X.column_names == ["x0"]
    assert isinstance(X, pa.Table)
    assert not view.flags.writeable and view.flags.aligned
    assert mappings[0].address <= view.ctypes.data
    assert view.ctypes.data + view.nbytes <= mappings[0].address + mappings[0].size
    assert view.ctypes.data == X.column(0).chunk(0).buffers()[1].address
    if kind == "bool":
        assert view.dtype == np.uint8
        assert bit_offset == 0
        assert view.nbytes == math.ceil(len(values) / 8)
        actual = np.unpackbits(view, bitorder="little")[: len(values)]
        np.testing.assert_array_equal(actual, values)
    else:
        assert view.dtype == values.dtype
        assert bit_offset is None
        np.testing.assert_array_equal(view, values)
    assert not view.flags.writeable


@pytest.mark.parametrize("compression", ["lz4", "zstd"])
def test_compression_is_rejected_before_decoding(tmp_path, monkeypatch, compression):
    path = tmp_path / "compressed.arrow"
    write_ipc(
        path, [pa.array(np.arange(1000, dtype=np.int16))], compression=compression
    )
    open_file = pa.ipc.open_file

    def inspect(source):
        reader = open_file(source)

        def forbidden(*args):
            raise AssertionError("Compressed batch was decoded before rejection")

        return SimpleNamespace(
            num_record_batches=reader.num_record_batches,
            schema=reader.schema,
            get_batch=forbidden,
        )

    monkeypatch.setattr(pa.ipc, "open_file", inspect)
    with pytest.raises(ValueError, match="uncompressed"):
        _open_mapped(path)


@pytest.mark.parametrize("compression", [None, "lz4", "zstd"])
def test_footer_must_select_preflight_batch(tmp_path, monkeypatch, compression):
    from pybiglasso.arrow import _field

    arrays = [pa.array(np.arange(107), type=pa.int16())]
    plain = tmp_path / "plain.arrow"
    target = tmp_path / "target.arrow"
    write_ipc(plain, arrays)
    write_ipc(target, arrays, compression=compression)
    raw = plain.read_bytes()
    source = pa.BufferReader(raw)
    source.seek(8)
    pa.ipc.read_message(source)
    start = source.tell()
    pa.ipc.read_message(source)
    extra = raw[start : source.tell()]
    indexed = target.read_bytes()
    patched = bytearray(indexed[:start] + extra + indexed[start:])
    footer_length = struct.unpack_from("<I", patched, len(patched) - 10)[0]
    footer = memoryview(patched)[-10 - footer_length : -10]
    root = struct.unpack_from("<I", footer)[0]
    field = _field(footer, root, 3)
    vector = field + struct.unpack_from("<I", footer, field)[0]
    block = vector + 4
    offset = struct.unpack_from("<q", footer, block)[0]
    struct.pack_into("<q", footer, block, offset + len(extra))
    target.write_bytes(patched)
    # The file reader genuinely follows the adjusted footer, not the extra batch.
    assert (
        pa.ipc.open_file(target)
        .get_batch(0)
        .equals(pa.record_batch(arrays, names=["x0"]))
    )
    open_file = pa.ipc.open_file

    def inspect(source):
        reader = open_file(source)

        def forbidden(*args):
            pytest.fail("Footer-selected batch decoded before offset validation")

        return SimpleNamespace(
            num_record_batches=reader.num_record_batches,
            schema=reader.schema,
            get_batch=forbidden,
        )

    if compression is not None:
        monkeypatch.setattr(pa.ipc, "open_file", inspect)
    if compression is None:
        # Unindexed messages are irrelevant: inspect the footer-selected batch.
        assert _open_mapped(target).equals(
            pa.Table.from_batches([pa.record_batch(arrays, names=["x0"])])
        )
    else:
        with pytest.raises(ValueError, match="uncompressed"):
            _open_mapped(target)


@pytest.mark.parametrize("kind", ["bool", "int16"])
@pytest.mark.parametrize("nrow", [9, 17])
@pytest.mark.parametrize(
    "case",
    [
        "list",
        "negative",
        "negative list",
        "first",
        "repeated",
        "empty",
        "unsigned",
        "past end",
        "before start",
        "huge unsigned",
        "float",
    ],
)
def test_public_prediction_row_indices_match_numpy(tmp_path, kind, nrow, case):
    import scipy.sparse as sps

    from pybiglasso.path import BiglassoPath

    values = np.zeros(nrow, dtype=kind)
    values[-1] = 1
    path = tmp_path / "rows.arrow"
    write_ipc(path, [pa.array(values)])
    design = _open_mapped(path)
    reference = values[:, None]
    model = BiglassoPath(
        lambdas=np.ones(1),
        intercept=np.zeros(1),
        coef=sps.csc_array([[1.0]]),
        deviance=np.zeros(1),
        iterations=np.zeros(1, dtype=int),
        n_rejected=np.zeros(1, dtype=int),
        center=np.zeros(1),
        scale=np.ones(1),
        penalty_factor=np.ones(1),
        feature_names=("x0",),
        alpha=1.0,
        family="gaussian",
    )
    rows = {
        "list": [nrow - 1],
        "negative": np.array([-1]),
        "negative list": [-1],
        "first": np.array([-nrow]),
        "repeated": [0, -1, nrow - 1, -nrow],
        "empty": [],
        "unsigned": np.array([nrow - 1], dtype=np.uint64),
        "past end": np.array([nrow]),
        "before start": np.array([-nrow - 1]),
        "huge unsigned": np.array([2**64 - 1], dtype=np.uint64),
        "float": np.array([1.5]),
    }[case]
    if case in ("past end", "before start", "float"):
        with pytest.raises(IndexError):
            model.predict(reference, row_indices=rows)
        with pytest.raises(IndexError):
            model.predict(design, row_indices=rows)
    else:
        expected = values[np.asarray(rows, dtype=np.intp)][:, None].astype(float)
        np.testing.assert_array_equal(
            model.predict(reference, row_indices=rows), expected
        )
        np.testing.assert_array_equal(model.predict(design, row_indices=rows), expected)


@pytest.mark.parametrize("batches", [0, 2])
def test_loads_zero_or_multiple_batches(tmp_path, batches):
    path = tmp_path / "batches.arrow"
    write_ipc(path, [pa.array([1, 2], type=pa.int8())], batches=batches)
    table = _open_mapped(path)
    assert table.num_rows == 2 * batches
    assert table.column(0).num_chunks == batches


@pytest.mark.parametrize("values", [[None, 1], [1, None]])
def test_rejects_nulls(tmp_path, values):
    path = tmp_path / "nulls.arrow"
    write_ipc(path, [pa.array(values, type=pa.int8())])
    with pytest.raises(ValueError, match="Column.*chunk.*null"):
        biglasso(_open_mapped(path), [0, 1])


@pytest.mark.parametrize(
    "kind", [pa.float16(), pa.string(), pa.timestamp("s"), pa.decimal128(4)]
)
def test_rejects_unsupported_types(tmp_path, kind):
    path = tmp_path / "unsupported.arrow"
    values = ["a", "b"] if kind == pa.string() else [1, 2]
    if kind == pa.float16():
        values = np.array(values, dtype=np.float16)
    write_ipc(path, [pa.array(values, type=kind)])
    with pytest.raises(ValueError, match="Column.*chunk.*unsupported Arrow type"):
        biglasso(_open_mapped(path), [0, 1])


def test_rejects_empty_rows(tmp_path):
    path = tmp_path / "empty.arrow"
    write_ipc(path, [pa.array([], type=pa.int8())])
    with pytest.raises(ValueError, match="nonempty"):
        biglasso(_open_mapped(path), [])


def test_unique_feature_names(tmp_path):
    batch = pa.record_batch([pa.array([0, 1], type=pa.int8())] * 2, names=["x", "x"])
    path = tmp_path / "duplicate.arrow"
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, batch.schema) as writer:
            writer.write_batch(batch)
    with pytest.raises(ValueError, match="unique"):
        biglasso(_open_mapped(path), [0, 1])


KINDS = [
    "bool",
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "float32",
    "float64",
]


def designs(tmp_path, kind="mixed", nrow=2057, ncol=9):
    rng = np.random.default_rng(124)
    arrays = []
    for j in range(ncol):
        dtype = KINDS[j % len(KINDS)] if kind == "mixed" else kind
        values = rng.integers(0, 2 if dtype == "bool" else 7, nrow).astype(dtype)
        if dtype in ("float32", "float64"):
            values /= 4
        arrays.append(values)
    path = tmp_path / "design.arrow"
    write_ipc(path, [pa.array(a) for a in arrays])
    npy = tmp_path / "design.npy"
    np.save(npy, np.asfortranarray(np.column_stack(arrays)))
    return _open_mapped(path), np.load(npy, mmap_mode="r")


def same_path(actual, expected):
    for field in (
        "lambdas",
        "intercept",
        "iterations",
        "n_rejected",
        "center",
        "scale",
        "deviance",
    ):
        a, b = getattr(actual, field), getattr(expected, field)
        assert a.dtype == b.dtype and a.shape == b.shape
        assert a.tobytes() == b.tobytes(), field
    for field in ("data", "indices", "indptr"):
        assert (
            getattr(actual.coef, field).tobytes()
            == getattr(expected.coef, field).tobytes()
        )


@pytest.mark.parametrize("kind", KINDS + ["mixed"])
@pytest.mark.parametrize("family", ["binomial", "gaussian"])
@pytest.mark.parametrize("subset", ["all", "mask", "gather", "repeated"])
@pytest.mark.parametrize("checked", [False, True])
def test_paths_are_bit_identical_to_fortran_npy(
    tmp_path, monkeypatch, kind, family, subset, checked
):
    from pybiglasso.config import Settings

    monkeypatch.setattr("pybiglasso.binomial.settings", Settings(boundscheck=checked))
    X, reference = designs(tmp_path, kind)
    n = X.shape[0]
    rng = np.random.default_rng(6)
    y = rng.integers(0, 2, n).astype(float)
    if family == "gaussian":
        y += reference[:, 0] / 4 + 0.5
    rows = {
        "all": None,
        "mask": np.arange(n)[np.arange(n) % 5 != 0],
        "gather": np.arange(0, n, 5),
        "repeated": np.arange(0, n, 3).repeat(2),
    }[subset]
    for threads, alpha in ((1, 1.0), (1, 0.5), (2, 1.0), (2, 0.5)):
        options = dict(
            family=family,
            alpha=alpha,
            row_indices=rows,
            n_threads=threads,
            nlambda=8,
            lambda_min=0.2,
        )
        expected = biglasso(reference, y, **options)
        actual = biglasso(X, y, **options)
        same_path(actual, expected)
        np.testing.assert_array_equal(
            actual.predict(X, row_indices=rows),
            expected.predict(reference, row_indices=rows),
        )
        np.testing.assert_array_equal(
            actual.score(X, y, row_indices=rows),
            expected.score(reference, y, row_indices=rows),
        )
        same_path(
            biglasso(X, y, lambdas=expected.lambdas, **options),
            biglasso(reference, y, lambdas=expected.lambdas, **options),
        )


@pytest.mark.parametrize("family", ["binomial", "gaussian"])
@pytest.mark.filterwarnings("error::UserWarning")
def test_cv_and_estimators_do_not_convert_design(tmp_path, family):
    from pybiglasso import sklearn as estimators

    X, reference = designs(tmp_path, nrow=107)
    y = np.random.default_rng(8).integers(0, 2, X.shape[0]).astype(float)
    name = "Classifier" if family == "binomial" else "Regressor"
    for suffix, options in (
        ("", dict(alpha=0.1)),
        ("CV", dict(n_alphas=6, alpha_min_ratio=0.3, cv=3, n_jobs=2)),
    ):
        cls = getattr(estimators, "Biglasso" + name + suffix)
        with pytest.raises(ValueError, match="requires y.*None"):
            cls(**options).fit(X, None)
        actual, expected = cls(**options).fit(X, y), cls(**options).fit(reference, y)
        assert tuple(actual.feature_names_in_) == tuple(X.column_names)
        np.testing.assert_array_equal(actual.coef_, expected.coef_)
        np.testing.assert_array_equal(actual.predict(X), expected.predict(reference))
        assert actual.score(X, y) == expected.score(reference, y)
        with pytest.warns(UserWarning, match="does not have valid feature names"):
            with pytest.raises(ValueError):
                actual.predict(reference[:, :-1])

        renamed = pa.record_batch(
            [pa.array(reference[:, j]) for j in range(reference.shape[1])],
            names=[f"renamed{j}" for j in range(reference.shape[1])],
        )
        path = tmp_path / "renamed.arrow"
        with pa.OSFile(str(path), "wb") as sink:
            with pa.ipc.new_file(sink, renamed.schema) as writer:
                writer.write_batch(renamed)
        with pytest.raises(ValueError, match="feature names"):
            actual.predict(_open_mapped(path))


def test_auto_threads_and_callback_lifetime(tmp_path, monkeypatch):
    from pybiglasso import _threads

    X, reference = designs(tmp_path, nrow=107)
    monkeypatch.setattr(_threads, "physical_cores", lambda: 99)
    assert _threads.resolve_threads("auto", X, 10**9) == 1
    expected = biglasso(reference, np.arange(107) % 2, nlambda=5)
    held = [X]
    del X

    def callback(*args):
        # Drop the last outside reference while native readers are detached.
        held.clear()
        gc.collect()
        return False

    actual = biglasso(held[0], np.arange(107) % 2, nlambda=5, _stop=callback)
    same_path(actual, expected)


def native_arguments(X, y):
    return dict(
        x=(X.shape[0], _ArrowMatrix(X)._columns),
        y=y,
        row_indices=None,
        alpha=0.5,
        lambdas=np.array([0.1, 0.05]),
        nlambda=2,
        eps=1e-7,
        lambda_min=0.2,
        lambda_log_scale=True,
        penalty_factor=np.ones(X.shape[1]),
        dfmax=X.shape[1] + 1,
        max_iter=1000,
        stop_on_saturation=False,
        r_compatible=False,
        n_threads=2,
        boundscheck=True,
        family="gaussian",
    )


@pytest.mark.parametrize("offset", range(8))
@pytest.mark.parametrize("checked", [False, True])
def test_native_packed_bit_offsets_and_owner_lifetime(tmp_path, offset, checked):
    from pybiglasso import _rust

    X, reference = designs(tmp_path, kind="bool", nrow=2057, ncol=3)
    y = (np.arange(2057) % 7).astype(float)
    args = native_arguments(X, y)
    owners = []
    for j in range(3):
        logical = np.r_[np.ones(offset, dtype=bool), reference[:, j]]
        bitmap = np.packbits(logical, bitorder="little")
        bitmap.flags.writeable = False
        owners.append([(bitmap, offset, 2057)])
    held = [X]
    del X, bitmap
    args.update(x=(2057, owners), boundscheck=checked)
    expected = _rust.fit_path(**{**args, "x": reference})

    def callback(*unused):
        owners.clear()
        held.clear()
        gc.collect()
        return False

    actual = _rust.fit_path(**args, on_lambda=callback)
    for field in ("center", "scale", "iters", "deviance", "intercept", "lambda_"):
        assert actual[field].tobytes() == expected[field].tobytes(), field


@pytest.mark.parametrize(
    "bad",
    [
        "empty",
        "zero rows",
        "short",
        "bitmap",
        "overflow",
        "negative",
        "wrong bit type",
        "unsupported",
        "unaligned",
        "strided",
        "writeable",
        "two dimensional",
    ],
)
def test_native_rejects_malformed_descriptors(tmp_path, bad):
    from pybiglasso import _rust

    X, _ = designs(tmp_path, "float64", nrow=41, ncol=1)
    args = native_arguments(X, (np.arange(41) % 3).astype(float))
    array = np.arange(41, dtype=float)
    offset = None
    nrow = 41
    if bad == "empty":
        args["x"] = (41, [])
    else:
        if bad == "zero rows":
            nrow = 0
        elif bad == "short":
            array = array[:-1]
        elif bad in ("bitmap", "overflow", "negative"):
            array = np.zeros(6, dtype=np.uint8)
            offset = {"bitmap": 8, "overflow": 2**64 - 1, "negative": -1}[bad]
        elif bad == "wrong bit type":
            offset = 0
        elif bad == "unsupported":
            array = array.astype(np.float16)
        elif bad == "unaligned":
            array = np.ndarray(
                41, dtype=np.float64, buffer=bytearray(41 * 8 + 1), offset=1
            )
        elif bad == "strided":
            array = np.arange(82, dtype=float)[::2]
        elif bad == "two dimensional":
            array = array[:, None]
        if bad != "writeable":
            array.flags.writeable = False
        args["x"] = (nrow, [[(array, offset, 41)]])
    with pytest.raises((ValueError, TypeError, OverflowError)):
        _rust.fit_path(**args)


def test_float_selected_rows_and_mask_fallback(tmp_path):
    from pybiglasso import _rust

    X, reference = designs(tmp_path, "float64", nrow=107, ncol=3)
    values = np.array(reference, order="F")
    values[0, 0] = np.nan
    write_ipc(tmp_path / "nan.arrow", [pa.array(values[:, j]) for j in range(3)])
    X = _open_mapped(tmp_path / "nan.arrow")
    y = (np.arange(107) % 5).astype(float)
    args = native_arguments(X, y)
    args.update(row_indices=np.arange(1, 107, dtype=np.int64), row_path="mask")
    actual = _rust.fit_path(**args)
    expected = _rust.fit_path(**{**args, "x": values})
    assert actual["row_path"] == expected["row_path"] == "gather"
    assert actual["deviance"].tobytes() == expected["deviance"].tobytes()
    with pytest.raises(ValueError, match="finite|NaN"):
        biglasso(X, y)


def partitioned_table(partition, nrow):
    """Same dyadic values; only physical type, chunks and offsets differ."""
    rng = np.random.default_rng(228)
    kinds = KINDS if partition != "work" else ["bool", "float64", "int64", "float32"]
    arrays, columns = [], []
    for j, kind in enumerate(kinds):
        values = rng.integers(0, 2 if kind == "bool" else 7, nrow).astype(kind)
        if kind.startswith("float"):
            values /= 4
        arrays.append(values)
        if partition == "one-row":
            ends = list(range(nrow + 1))
        elif partition == "independent":
            ends = list(range(0, nrow, (j + 1) * 19)) + [nrow]
        else:
            ends = sorted(
                {
                    0,
                    nrow,
                    *[
                        i + j % 3
                        for i in (
                            1,
                            7,
                            8,
                            9,
                            15,
                            16,
                            17,
                            2047,
                            2048,
                            2049,
                            16383,
                            16384,
                            16385,
                        )
                        if i + j % 3 < nrow
                    ],
                }
            )
        chunks = [pa.array([], type=pa.from_numpy_dtype(values.dtype))]
        for k, (start, stop) in enumerate(zip(ends[:-1], ends[1:], strict=True)):
            offset = (k + j) % 8 + 1
            # Each chunk is a slice of its own allocated backing buffer; bool
            # starts at every bit offset, including slices beyond a full byte.
            backing = pa.array(
                np.r_[np.zeros(offset, dtype=values.dtype), values[start:stop]]
            )
            chunks.append(backing.slice(offset, stop - start))
            if k % 3 == 0:
                chunks.append(chunks[0])
        chunks.append(chunks[0])
        columns.append(pa.chunked_array(chunks))
    table = pa.Table.from_arrays(columns, names=[f"x{j}" for j in range(len(columns))])
    return table, np.asfortranarray(np.column_stack(arrays))


@pytest.mark.parametrize("partition", ["lanes", "independent", "one-row", "work"])
@pytest.mark.parametrize("family", ["binomial", "gaussian"])
@pytest.mark.parametrize("subset", ["all", "mask", "gather", "repeated"])
@pytest.mark.parametrize("checked", [False, True])
def test_partition_only_changes_addressing(
    tmp_path, monkeypatch, partition, family, subset, checked
):
    from pybiglasso.config import Settings

    nrow = 73 if partition == "one-row" else 16391 if partition == "work" else 2057
    table, values = partitioned_table(partition, nrow)
    np.save(tmp_path / "reference.npy", values)
    reference = np.load(tmp_path / "reference.npy", mmap_mode="r")
    y = (np.arange(nrow) % 2).astype(float)
    if family == "gaussian":
        y += values[:, 1] / 4 + 0.5
    rows = {
        "all": None,
        "mask": np.arange(1, nrow),
        "gather": np.arange(0, nrow, 5),
        "repeated": np.arange(0, nrow, 3).repeat(2),
    }[subset]
    monkeypatch.setattr("pybiglasso.binomial.settings", Settings(boundscheck=checked))
    for threads, alpha in ((1, 1.0), (2, 1.0), (1, 0.5), (2, 0.5)):
        options = dict(
            family=family,
            alpha=alpha,
            n_threads=threads,
            row_indices=rows,
            nlambda=5,
            lambda_min=0.4,
        )
        npy_fit = biglasso(reference, y, **options)
        arrow_fit = biglasso(
            table, pa.chunked_array([pa.array(y[:7]), pa.array(y[7:])]), **options
        )
        same_path(arrow_fit, npy_fit)
        same_path(
            biglasso(table, y, lambdas=npy_fit.lambdas, **options),
            biglasso(reference, y, lambdas=npy_fit.lambdas, **options),
        )
        np.testing.assert_array_equal(
            arrow_fit.predict(table, row_indices=rows),
            npy_fit.predict(reference, row_indices=rows),
        )
        np.testing.assert_array_equal(
            arrow_fit.score(table, pa.array(y), row_indices=rows),
            npy_fit.score(reference, y, row_indices=rows),
        )


@pytest.mark.parametrize("compression", [None, "lz4", "zstd"])
def test_all_footer_batches_checked_before_any_decode(
    tmp_path, monkeypatch, compression
):
    # Start with two compressed batches, then transplant a complete uncompressed
    # first block of exactly the same byte length by composing independent files.
    from pybiglasso.arrow import _field

    batch = pa.record_batch([pa.array(np.arange(107), type=pa.int16())], names=["x0"])
    plain = pa.BufferOutputStream()
    with pa.ipc.new_file(plain, batch.schema) as writer:
        writer.write_batch(batch)
    reader = pa.BufferReader(plain.getvalue())
    reader.seek(8)
    pa.ipc.read_message(reader)
    start = reader.tell()
    pa.ipc.read_message(reader)
    plain_block = plain.getvalue().to_pybytes()[start : reader.tell()]
    target = tmp_path / "multi.arrow"
    write_ipc(target, batch.columns, compression=compression, batches=2)
    raw = bytearray(target.read_bytes())
    footer_size = struct.unpack_from("<I", raw, len(raw) - 10)[0]
    footer = memoryview(raw)[-10 - footer_size : -10]
    root = struct.unpack_from("<I", footer)[0]
    field = _field(footer, root, 3)
    vector = field + struct.unpack_from("<I", footer, field)[0]
    block0, block1 = vector + 4, vector + 28
    offset0, offset1 = (
        struct.unpack_from("<q", footer, k)[0] for k in (block0, block1)
    )
    # Point the first footer entry at the new plain block and shift the second.
    struct.pack_into("<q", footer, block1, offset1 + len(plain_block))
    metadata_len = struct.unpack_from("<i", plain_block, 4)[0] + 8
    body_len = len(plain_block) - metadata_len
    struct.pack_into("<qi", footer, block0, offset0, metadata_len)
    struct.pack_into("<q", footer, block0 + 16, body_len)
    del footer
    raw = raw[:offset0] + plain_block + raw[offset0:]
    target.write_bytes(raw)
    real_open = pa.ipc.open_file
    if compression:

        def inspect(source):
            actual = real_open(source)

            def forbidden(*args):
                pytest.fail("An earlier batch decoded before checking all batches")

            return SimpleNamespace(
                schema=actual.schema,
                num_record_batches=actual.num_record_batches,
                get_batch=forbidden,
            )

        monkeypatch.setattr(pa.ipc, "open_file", inspect)
        with pytest.raises(ValueError, match="uncompressed"):
            _open_mapped(target)
    else:
        table = _open_mapped(target)
        assert table.num_rows == 214
        assert table.column(0).num_chunks == 2


def test_unaligned_arrow_is_rejected_without_copying():
    values = np.ndarray(41, dtype=np.float64, buffer=bytearray(41 * 8 + 1), offset=1)
    values[:] = np.arange(41)
    assert not values.flags.aligned
    array = pa.Array.from_buffers(pa.float64(), 41, [None, pa.py_buffer(values)])
    with pytest.raises(
        ValueError, match="Column 'bad', chunk 0: unaligned.*aligned Arrow buffers"
    ):
        biglasso(pa.table({"bad": array}), np.arange(41) % 2)


def test_mutable_numpy_backed_table_borrows_readonly_views():
    values = np.arange(73, dtype=float)
    table = pa.table({"x": pa.array(values)})
    descriptor = _ArrowMatrix(table)._columns[0][0]
    assert descriptor[0].ctypes.data == values.ctypes.data
    assert not descriptor[0].flags.writeable
    assert values.flags.writeable
    fit = biglasso(table, values / 4, nlambda=5)
    assert values.flags.writeable
    np.testing.assert_array_equal(values, np.arange(73))
    assert fit.feature_names == ("x",)


@pytest.mark.parametrize("form", ["table", "batch"])
@pytest.mark.parametrize("labels", [False, True])
def test_arrow_response_preserves_estimator_target_contract(form, labels):
    from pybiglasso.sklearn import (
        BiglassoClassifier,
        BiglassoClassifierCV,
        BiglassoRegressor,
        BiglassoRegressorCV,
    )

    values = np.asfortranarray(np.arange(123).reshape(41, 3) % 7)
    batch = pa.record_batch(
        [pa.array(values[:, j]) for j in range(3)], names=["a", "b", "c"]
    )
    X = batch if form == "batch" else pa.Table.from_batches([batch])
    y = (
        np.where(np.arange(41) % 2, "case", "control")
        if labels
        else (np.arange(41) % 2).astype(float)
    )
    target = pa.chunked_array([pa.array(y[:3]), pa.array(y[3:])])
    classes = (
        [BiglassoClassifier, BiglassoClassifierCV]
        if labels
        else [BiglassoRegressor, BiglassoRegressorCV]
    )
    for cls in classes:
        options = (
            dict(n_alphas=5, cv=3, n_jobs=2)
            if cls.__name__.endswith("CV")
            else dict(alpha=0.1)
        )
        actual, reference = cls(**options).fit(X, target), cls(**options).fit(values, y)
        np.testing.assert_array_equal(actual.coef_, reference.coef_)
        np.testing.assert_array_equal(actual.predict(X), reference.predict(values))
        if labels:
            np.testing.assert_array_equal(actual.classes_, ["case", "control"])


@pytest.mark.parametrize("framework", ["pandas", "polars"])
@pytest.mark.parametrize(
    "model",
    [
        "binomial",
        "gaussian",
        "BiglassoClassifier",
        "BiglassoClassifierCV",
        "BiglassoRegressor",
        "BiglassoRegressorCV",
        "BiglassoClassifier_strings",
        "BiglassoClassifierCV_strings",
    ],
)
def test_arraylike_frame_fit_matches_fortran_ndarray(framework, model, request):
    import pybiglasso as bl

    module = pytest.importorskip(framework)
    rng = np.random.default_rng(33)
    values = rng.normal(size=(123, 5))
    names = [f"feature_{j}" for j in range(values.shape[1])]
    signal = values[:, 0] - values[:, 1] / 2
    binary = (rng.random(len(values)) < 1 / (1 + np.exp(-signal))).astype(float)
    target = (
        signal + rng.normal(size=len(values))
        if model in ("gaussian", "BiglassoRegressor", "BiglassoRegressorCV")
        else binary
    )
    if model.endswith("_strings"):
        target = np.where(binary == 1, "case", "control")
    if framework == "pandas":
        frame = module.DataFrame(values, columns=names)
        response = module.Series(target, name="y")
    else:
        frame = module.DataFrame({name: values[:, j] for j, name in enumerate(names)})
        response = module.Series("y", target)
    reference_values = np.asfortranarray(values)
    expected_warning = not np.asarray(frame).flags.f_contiguous
    if model in ("binomial", "gaussian"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            actual = biglasso(frame, response, family=model, alpha=0.7, nlambda=7)
        reference = biglasso(
            reference_values, target, family=model, alpha=0.7, nlambda=7
        )
        same_path(actual, reference)
        assert list(actual.feature_names) == names
    else:
        sklearn = pytest.importorskip("sklearn")
        if framework == "polars":
            request.applymarker(
                pytest.mark.xfail(
                    condition=Version(sklearn.__version__) < Version("1.9"),
                    strict=True,
                    reason="scikit-learn < 1.9 drops feature names from Polars DataFrames",
                )
            )
        cls = getattr(bl, model.removesuffix("_strings"))
        options = (
            dict(n_alphas=7, cv=3, n_jobs=1)
            if cls.__name__.endswith("CV")
            else dict(alpha=0.1)
        )
        options["l1_ratio"] = 0.7
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            actual = cls(**options).fit(frame, response)
        reference = cls(**options).fit(reference_values, target)
        assert list(actual.feature_names_in_) == names
        for attribute in (
            "coef_",
            "intercept_",
            "n_iter_",
            "cv_scores_",
            "alphas_",
            "alpha_",
        ):
            if hasattr(actual, attribute):
                left, right = (
                    np.asarray(getattr(actual, attribute)),
                    np.asarray(getattr(reference, attribute)),
                )
                assert left.shape == right.shape and left.dtype == right.dtype
                assert left.tobytes() == right.tobytes(), attribute
        if hasattr(actual, "path_"):
            same_path(actual.path_, reference.path_)
            assert list(actual.path_.feature_names) == names
        if hasattr(actual, "classes_"):
            np.testing.assert_array_equal(actual.classes_, reference.classes_)
        np.testing.assert_array_equal(
            actual.predict(frame), reference.predict(reference_values)
        )
    assert (
        any("not in Fortran" in str(item.message) for item in caught)
        == expected_warning
    )


@pytest.mark.parametrize(
    "model",
    [
        "biglasso",
        "BiglassoClassifier",
        "BiglassoClassifierCV",
        "BiglassoRegressor",
        "BiglassoRegressorCV",
    ],
)
def test_arraylike_stream_reader_rejected_without_consumption(model):
    import pybiglasso as bl

    batch = pa.record_batch({"x": np.arange(41, dtype=float)})
    consumed = []

    def batches():
        consumed.append(True)
        yield batch

    reader = pa.RecordBatchReader.from_batches(batch.schema, batches())
    target = np.arange(41) % 2
    if model == "biglasso":
        fit = biglasso
    else:
        pytest.importorskip("sklearn")
        fit = getattr(bl, model)().fit
    with pytest.raises(TypeError, match="Convert explicitly to a pyarrow.Table"):
        fit(reader, target)
    assert consumed == []
    assert reader.read_next_batch().equals(batch)
    assert consumed == [True]


@pytest.mark.parametrize("protocol", ["__arrow_c_stream__", "__arrow_c_array__"])
def test_numpy_array_protocol_precedes_arrow_exporter(protocol):
    values = np.asfortranarray(np.arange(123 * 3).reshape(123, 3) % 7, dtype=float)
    target = np.arange(123) % 2

    def forbidden(*args, **kwargs):
        pytest.fail("Arrow exporter was called instead of __array__")

    class Matrix:
        columns = ["one", "two", "three"]

        def __array__(self, dtype=None, copy=None):
            return np.asarray(values, dtype=dtype)

    setattr(Matrix, protocol, forbidden)
    actual = biglasso(Matrix(), target, nlambda=5)
    same_path(actual, biglasso(values, target, nlambda=5))
    assert list(actual.feature_names) == Matrix.columns


@pytest.mark.parametrize("framework", ["pandas", "polars"])
def test_arraylike_frame_keeps_numpy_dtype_validation(framework):
    module = pytest.importorskip(framework)
    frame = module.DataFrame({"feature": ["a", "b"]})
    with pytest.raises(ValueError, match="native real numeric dtype"):
        biglasso(frame, [0, 1])


@pytest.mark.parametrize("protocol", ["__arrow_c_stream__", "__arrow_c_array__"])
def test_implicit_exporters_are_rejected_before_conversion(protocol):
    def forbidden(*args):
        pytest.fail("Implicit exporter was invoked")

    cls = type(
        "Exporter",
        (),
        {protocol: forbidden},
    )
    with pytest.raises(TypeError, match="Convert explicitly to a pyarrow.Table"):
        biglasso(cls(), [0, 1])


@pytest.mark.parametrize(
    "kind", ["dictionary", "list", "struct", "extension", "timestamp", "string", "null"]
)
def test_column_chunk_validation_always_errors(kind):
    if kind == "dictionary":
        array = pa.array(["x", "y"]).dictionary_encode()
    elif kind == "list":
        array = pa.array([[1], [2]])
    elif kind == "struct":
        array = pa.array([{"x": 1}, {"x": 2}])
    elif kind == "extension":

        class Extension(pa.ExtensionType):
            def __init__(self):
                super().__init__(pa.int8(), "test-only")

            def __arrow_ext_serialize__(self):
                return b""

        array = pa.ExtensionArray.from_storage(
            Extension(), pa.array([1, 2], type=pa.int8())
        )
    elif kind == "timestamp":
        array = pa.array([1, 2], type=pa.timestamp("s"))
    elif kind == "null":
        array = pa.array([1.0, None])
    else:
        array = pa.array(["x", "y"])
    table = pa.table({"bad": pa.chunked_array([array.slice(0, 0), array])})
    with pytest.raises(ValueError, match="Column 'bad', chunk [01]:"):
        biglasso(table, [0, 1])


@pytest.mark.parametrize("family", ["binomial", "gaussian"])
def test_mapped_multibatch_and_allocated_tables_share_complete_paths(tmp_path, family):
    table, reference = partitioned_table("lanes", 2057)
    path = tmp_path / "partitioned.arrow"
    with pa.ipc.new_file(path, table.schema) as writer:
        writer.write_table(table, max_chunksize=13)
    mapped = _open_mapped(path)
    assert isinstance(mapped, pa.Table)
    assert mapped.column(0).num_chunks > 1
    y = (np.arange(2057) % 2).astype(float)
    options = dict(family=family, alpha=0.5, nlambda=5, n_threads=2)
    same_path(
        biglasso(mapped, pa.array(y), **options), biglasso(reference, y, **options)
    )
    same_path(biglasso(table, y, **options), biglasso(reference, y, **options))
    gc.collect()
    np.testing.assert_array_equal(mapped.column(0).to_numpy(), reference[:, 0])


def test_chunked_nonfinite_excluded_rows_use_mask_fallback():
    from pybiglasso import _rust

    values = np.asfortranarray(np.random.default_rng(43).normal(size=(107, 3)))
    values[0, 0] = np.nan
    table = pa.Table.from_arrays(
        [
            pa.chunked_array(
                [pa.array(values[: j + 1, j]), pa.array(values[j + 1 :, j])]
            )
            for j in range(3)
        ],
        names=[f"x{j}" for j in range(3)],
    )
    y = (np.arange(107) % 5).astype(float)
    args = native_arguments(table, y)
    args.update(row_indices=np.arange(1, 107, dtype=np.int64), row_path="mask")
    actual, reference = _rust.fit_path(**args), _rust.fit_path(**{**args, "x": values})
    assert actual["row_path"] == reference["row_path"] == "gather"
    assert actual["deviance"].tobytes() == reference["deviance"].tobytes()
    with pytest.raises(ValueError, match="finite|NaN"):
        biglasso(table, y)


@pytest.mark.parametrize("kind", ["int64", "uint64"])
def test_wide_integer_conversion_matches_ndarray(kind):
    values = ((1 << 53) + 1 + np.arange(41, dtype=kind) * 3).astype(kind)
    table = pa.table({"x": pa.array(values)})
    reference = np.asfortranarray(values[:, None])
    y = (np.arange(41) % 7).astype(float)
    same_path(biglasso(table, y, nlambda=5), biglasso(reference, y, nlambda=5))


@pytest.mark.parametrize("compression", [None, "zstd"])
def test_dictionary_compression_is_preflighted_and_projection_is_explicit(
    tmp_path, compression, monkeypatch
):
    table = pa.table(
        {
            "x": np.arange(41) % 7,
            "label": pa.array(
                ["case" if i % 2 else "control" for i in range(41)]
            ).dictionary_encode(),
        }
    )
    path = tmp_path / "dictionary.arrow"
    with pa.ipc.new_file(
        path, table.schema, options=pa.ipc.IpcWriteOptions(compression=compression)
    ) as writer:
        writer.write_table(table)
    if compression:
        monkeypatch.setattr(
            pa.ipc,
            "open_file",
            lambda *_: pytest.fail("Reader opened before dictionary preflight"),
        )
        with pytest.raises(ValueError, match="uncompressed"):
            _open_mapped(path)
    else:
        mapped = _open_mapped(path)
        assert mapped.equals(table)
        with pytest.raises(ValueError, match="Column 'label', chunk 0: unsupported"):
            biglasso(mapped, np.arange(41) % 2)
        from pybiglasso.sklearn import BiglassoClassifier

        fit = BiglassoClassifier().fit(mapped.select(["x"]), mapped.column("label"))
        np.testing.assert_array_equal(fit.classes_, ["case", "control"])


def test_dictionary_delta_rejected_before_reader_can_concatenate(tmp_path, monkeypatch):
    path = tmp_path / "delta.arrow"
    first = pa.record_batch(
        [pa.DictionaryArray.from_arrays(pa.array([0, 1]), pa.array(["a", "b"]))],
        names=["label"],
    )
    second = pa.record_batch(
        [pa.DictionaryArray.from_arrays(pa.array([1, 2]), pa.array(["a", "b", "c"]))],
        names=["label"],
    )
    with pa.ipc.new_file(
        path, first.schema, options=pa.ipc.IpcWriteOptions(emit_dictionary_deltas=True)
    ) as writer:
        writer.write_batch(first)
        writer.write_batch(second)
    monkeypatch.setattr(
        pa.ipc,
        "open_file",
        lambda *_: pytest.fail("Dictionary concatenated before preflight"),
    )
    with pytest.raises(ValueError, match="dictionary deltas"):
        _open_mapped(path)


@pytest.mark.parametrize("storage", ["mapped", "allocated", "numpy"])
@pytest.mark.parametrize("family", ["binomial", "gaussian"])
def test_every_chunk_owner_survives_dropped_outer_references(tmp_path, storage, family):
    from pybiglasso import _rust

    table, reference = designs(tmp_path, nrow=107)
    arrays = None
    if storage == "allocated":
        table = pa.Table.from_arrays(
            [
                pa.array(column.to_pylist(), type=column.type)
                for column in table.itercolumns()
            ],
            names=table.column_names,
        )
    elif storage == "numpy":
        arrays = [
            np.array(reference[:, j], copy=True) for j in range(reference.shape[1])
        ]
        table = pa.Table.from_arrays(
            [pa.array(values) for values in arrays], names=table.column_names
        )
    # Each column's independent boundaries also cross the kernel's fixed lanes.
    table = pa.Table.from_arrays(
        [
            pa.chunked_array(
                [column.chunk(0).slice(0, j + 1), column.chunk(0).slice(j + 1)]
            )
            for j, column in enumerate(table.itercolumns())
        ],
        names=table.column_names,
    )
    y = (np.arange(107) % 2).astype(float)
    args = native_arguments(table, y)
    columns = [list(column) for column in args["x"][1]]
    args.update(x=(107, columns), family=family)
    reference_fit = _rust.fit_path(**{**args, "x": reference})
    held = [table, arrays]
    del table, arrays

    def callback(*unused):
        columns.clear()
        held.clear()
        gc.collect()
        return False

    actual = _rust.fit_path(**args, on_lambda=callback)
    for field in (
        "center",
        "scale",
        "col_map",
        "iters",
        "n_reject",
        "deviance",
        "intercept",
        "lambda_",
    ):
        assert actual[field].tobytes() == reference_fit[field].tobytes(), field
    for field in ("coef_rows", "coef_vals"):
        for a, b in zip(actual[field], reference_fit[field], strict=True):
            if a is None or b is None:
                assert a is b
            else:
                assert a.tobytes() == b.tobytes(), field
