"""Mapped IPC buffers retain their original physical representation."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

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
