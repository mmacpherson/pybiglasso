"""Mapped IPC buffers retain their original physical representation."""

import gc
import math
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from pybiglasso import biglasso
from pybiglasso.arrow import ArrowDesign

pa = pytest.importorskip("pyarrow")


def write_ipc(path, arrays, *, compression=None, batches=1):
    batch = pa.record_batch(arrays, names=[f"x{i}" for i in range(len(arrays))])
    options = pa.ipc.IpcWriteOptions(compression=compression)
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, batch.schema, options=options) as writer:
            for _ in range(batches):
                writer.write_batch(batch)


@pytest.mark.parametrize(
    "kind", ["bool", "int8", "uint8", "int16", "float32", "float64"]
)
def test_values_are_readonly_views_of_the_mapping(tmp_path, kind):
    values = np.arange(35) % 7
    values = (values % 2).astype(bool) if kind == "bool" else values.astype(kind)
    path = tmp_path / "design.arrow"
    write_ipc(path, [pa.array(values)])
    X = ArrowDesign(path)
    view, bit_offset = X._columns[0]
    assert X.shape == (35, 1)
    assert X.columns == ("x0",)
    assert not view.flags.writeable and view.flags.aligned
    assert X._mapping.address <= view.ctypes.data
    assert view.ctypes.data + view.nbytes <= X._mapping.address + X._mapping.size
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
    with pytest.raises(TypeError, match="implicitly converted"):
        np.asarray(X)


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
        ArrowDesign(path)


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

    monkeypatch.setattr(pa.ipc, "open_file", inspect)
    with pytest.raises(ValueError, match="footer|offset"):
        ArrowDesign(target)


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
    design = ArrowDesign(path)
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
def test_requires_one_batch(tmp_path, batches):
    path = tmp_path / "batches.arrow"
    write_ipc(path, [pa.array([1, 2], type=pa.int8())], batches=batches)
    with pytest.raises(ValueError, match="exactly one record batch"):
        ArrowDesign(path)


@pytest.mark.parametrize("values", [[None, 1], [1, None]])
def test_rejects_nulls(tmp_path, values):
    path = tmp_path / "nulls.arrow"
    write_ipc(path, [pa.array(values, type=pa.int8())])
    with pytest.raises(ValueError, match="null"):
        ArrowDesign(path)


@pytest.mark.parametrize("kind", [pa.int32(), pa.uint16(), pa.float16(), pa.string()])
def test_rejects_unsupported_types(tmp_path, kind):
    path = tmp_path / "unsupported.arrow"
    values = ["a", "b"] if kind == pa.string() else [1, 2]
    write_ipc(path, [pa.array(values, type=kind)])
    with pytest.raises(ValueError, match="Arrow columns must"):
        ArrowDesign(path)


def test_rejects_empty_rows(tmp_path):
    path = tmp_path / "empty.arrow"
    write_ipc(path, [pa.array([], type=pa.int8())])
    with pytest.raises(ValueError, match="at least one row"):
        ArrowDesign(path)


def test_unique_feature_names(tmp_path):
    batch = pa.record_batch([pa.array([0, 1], type=pa.int8())] * 2, names=["x", "x"])
    path = tmp_path / "duplicate.arrow"
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, batch.schema) as writer:
            writer.write_batch(batch)
    with pytest.raises(ValueError, match="unique"):
        ArrowDesign(path)


KINDS = ["bool", "int8", "uint8", "int16", "float32", "float64"]


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
    return ArrowDesign(path), np.load(npy, mmap_mode="r")


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
        assert tuple(actual.feature_names_in_) == X.columns
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
            actual.predict(ArrowDesign(path))


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

    actual = biglasso(held.pop(), np.arange(107) % 2, nlambda=5, _stop=callback)
    same_path(actual, expected)


def native_arguments(X, y):
    return dict(
        x=(X.shape[0], X._columns),
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
        owners.append((bitmap, offset))
    args.update(x=(2057, owners), boundscheck=checked)
    expected = _rust.fit_path(**{**args, "x": reference})

    def callback(*unused):
        owners.clear()
        X._columns = ()
        X._source.close()
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
            array = array.astype(np.int32)
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
        args["x"] = (nrow, [(array, offset)])
    with pytest.raises((ValueError, TypeError, OverflowError)):
        _rust.fit_path(**args)


def test_float_selected_rows_and_mask_fallback(tmp_path):
    from pybiglasso import _rust

    X, reference = designs(tmp_path, "float64", nrow=107, ncol=3)
    values = np.array(reference, order="F")
    values[0, 0] = np.nan
    write_ipc(tmp_path / "nan.arrow", [pa.array(values[:, j]) for j in range(3)])
    X = ArrowDesign(tmp_path / "nan.arrow")
    y = (np.arange(107) % 5).astype(float)
    args = native_arguments(X, y)
    args.update(row_indices=np.arange(1, 107, dtype=np.int64), row_path="mask")
    actual = _rust.fit_path(**args)
    expected = _rust.fit_path(**{**args, "x": values})
    assert actual["row_path"] == expected["row_path"] == "gather"
    assert actual["deviance"].tobytes() == expected["deviance"].tobytes()
    with pytest.raises(ValueError, match="finite|NaN"):
        biglasso(X, y)
