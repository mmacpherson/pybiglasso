"""One-shot producers and exact, bounded Arrow-stream materialization."""

import importlib
from decimal import Decimal

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
from pybiglasso.ingest import ingest, load  # noqa: E402


class Once:
    def __init__(self, schema, batches):
        self.schema = schema
        self.batches = batches
        self.acquisitions = 0
        self.reads = 0

    def __arrow_c_stream__(self, requested_schema=None):
        self.acquisitions += 1
        assert self.acquisitions == 1, "source reacquired"

        def batches():
            for batch in self.batches:
                self.reads += 1
                yield batch

        return pa.RecordBatchReader.from_batches(
            self.schema, batches()
        ).__arrow_c_stream__()


class ScaledScalar(pa.ExtensionScalar):
    def as_py(self, **kwargs):
        return None if not self.is_valid else 2 * self.value.as_py(**kwargs)


class ScaledType(pa.ExtensionType):
    def __init__(self):
        super().__init__(pa.float64(), "pybiglasso.test.scaled")

    def __arrow_ext_serialize__(self):
        return b""

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        return cls()

    def __arrow_ext_scalar_class__(self):
        return ScaledScalar

    def to_pandas_dtype(self):
        return np.dtype("float64")


@pytest.fixture
def scaled_type():
    extension = ScaledType()
    pa.register_extension_type(extension)
    try:
        yield extension
    finally:
        pa.unregister_extension_type(extension.extension_name)


def source():
    schema = pa.schema(
        [
            pa.field("wide", pa.int64(), metadata={b"unit": b"count"}),
            pa.field("bit", pa.bool_()),
            pa.field("y", pa.float64()),
        ],
        metadata={b"origin": b"example"},
    )
    rows = [(2**53 + 1, True, 1.5), (2**53 + 3, False, -0.0), (7, True, 3.0)]

    def batch(values):
        return pa.RecordBatch.from_arrays(
            [
                pa.array([r[i] for r in values], type=f.type)
                for i, f in enumerate(schema)
            ],
            schema=schema,
        )

    return Once(schema, [batch(rows[:1]), batch([]), batch(rows[1:])])


def test_preserve_one_acquisition_empty_batches_and_schema(tmp_path):
    producer = source()
    result = ingest(
        producer,
        tmp_path / "data.feather",
        response="y",
        dtypes="preserve",
        block_size=1,
    )
    assert producer.acquisitions == 1 and producer.reads == 3
    assert result.X.column_names == ["wide", "bit"]
    assert result.X["wide"].to_pylist() == [2**53 + 1, 2**53 + 3, 7]
    assert result.X["bit"].to_pylist() == [True, False, True]
    assert result.X.schema.field("wide").equals(
        producer.schema.field("wide"), check_metadata=True
    )
    assert result.X.schema.metadata[b"origin"] == b"example"
    assert result.y.tobytes() == np.asarray([1.5, -0.0, 3.0], dtype="<f8").tobytes()
    reopened = load(result.path)
    assert reopened.X.equals(result.X)
    assert not reopened.y.flags.writeable


@pytest.mark.parametrize("kind", ["table", "reader", "polars"])
def test_auto_stream_matches_first_principles(tmp_path, kind):
    table = pa.table({"x": [0, 1, 0], "f": [0.25, -0.0, 0.5], "bad": ["text"] * 3})
    if kind == "table":
        producer = table
    elif kind == "reader":
        producer = table.to_reader(max_chunksize=1)
    else:
        producer = pytest.importorskip("polars").from_arrow(table)
    result = ingest(
        producer, tmp_path / "data.feather", features=["f", "x"], block_size=1
    )
    assert result.X.schema.types == [pa.float32(), pa.bool_()]
    assert result.X.to_pydict() == {"f": [0.25, -0.0, 0.5], "x": [False, True, False]}
    assert np.signbit(result.X["f"][1].as_py())


def test_decimal_stream_rounding_only_for_explicit_float64_feature(tmp_path):
    table = pa.table(
        {
            "x": [Decimal("0.1"), Decimal("0.2")],
            "y": [Decimal("9007199254740993"), Decimal("9007199254740995")],
        }
    )
    with pytest.raises(ValueError, match="Column 'y'.*no supported exact dtype"):
        ingest(
            table,
            tmp_path / "response.arrow",
            response="y",
            dtypes={"x": "float64"},
        )
    inexact_float32 = pa.table({"x": [Decimal("1.00000000000000000001")]})
    with pytest.raises(ValueError, match="Column 'x'.*no supported exact dtype"):
        ingest(
            inexact_float32,
            tmp_path / "float32.arrow",
            dtypes={"x": "float32"},
        )
    rounded = ingest(
        table.select(["x"]),
        tmp_path / "float64.arrow",
        dtypes={"x": "float64"},
    )
    assert rounded.X["x"].to_pylist() == [0.1, 0.2]


@pytest.mark.parametrize("options", [{}, {"cache": True}, {"cache_dir": "unused"}])
def test_stream_options_rejected_before_acquisition(tmp_path, options):
    producer = source()
    kwargs = {"out": tmp_path / "data.feather", **options}
    if not options:
        kwargs.pop("out")
    with pytest.raises(ValueError, match="stream.*(out|cache)"):
        ingest(producer, **kwargs)
    assert producer.acquisitions == producer.reads == 0


@pytest.mark.parametrize("mode", ["auto", "preserve"])
@pytest.mark.parametrize("bad", [None, float("nan"), float("inf")])
def test_stream_bad_values_have_cumulative_rows_and_no_publication(tmp_path, mode, bad):
    schema = pa.schema([("x", pa.float64()), ("y", pa.float64())])
    batches = [
        pa.record_batch([[1.0, 2.0], [1.0, 2.0]], schema=schema),
        pa.record_batch([[], []], schema=schema),
        pa.record_batch([[3.0, bad], [3.0, 4.0]], schema=schema),
    ]
    producer = Once(schema, batches)
    path = tmp_path / "data.feather"
    with pytest.raises(ValueError, match="Column 'x', row 4"):
        ingest(producer, path, response="y", dtypes=mode, block_size=1)
    assert not path.exists() and not path.with_suffix(".feather.response").exists()
    assert not list(tmp_path.glob(".ingest-*"))


@pytest.mark.parametrize(
    "dtype",
    [
        pa.int8(),
        pa.uint8(),
        pa.int16(),
        pa.uint16(),
        pa.int32(),
        pa.uint32(),
        pa.int64(),
        pa.uint64(),
        pa.float32(),
        pa.float64(),
        pa.bool_(),
    ],
)
def test_preserve_all_solver_types(tmp_path, dtype, monkeypatch):
    value = (
        True
        if pa.types.is_boolean(dtype)
        else (2**64 - 1 if dtype == pa.uint64() else 1)
    )
    table = pa.table({"x": pa.array([value, value], type=dtype)})
    module = importlib.import_module("pybiglasso.ingest")
    monkeypatch.setattr(
        module,
        "_numeric",
        lambda *_args, **_kwargs: pytest.fail(
            "primitive preserve used scalar validation"
        ),
    )
    result = ingest(table, tmp_path / "data.feather", dtypes="preserve", block_size=1)
    assert result.X.schema.types == [dtype]
    assert result.X["x"].to_pylist() == [value, value]


def test_preserve_primitive_response_is_vectorized_and_exact(tmp_path, monkeypatch):
    table = pa.table({"x": [1, 2], "y": [2**53, 2**54]})
    module = importlib.import_module("pybiglasso.ingest")
    monkeypatch.setattr(
        module,
        "_numeric",
        lambda *_args, **_kwargs: pytest.fail(
            "primitive response used scalar validation"
        ),
    )
    result = ingest(
        table,
        tmp_path / "data.feather",
        response="y",
        dtypes="preserve",
    )
    np.testing.assert_array_equal(result.y, [2**53, 2**54])


@pytest.mark.parametrize("values", [[Decimal("0.25"), Decimal("0.5")], ["0.25", "0.5"]])
def test_preserve_decimal_and_string_responses_keep_scalar_validation(
    tmp_path, monkeypatch, values
):
    table = pa.table({"x": [1, 2], "y": values})
    module = importlib.import_module("pybiglasso.ingest")
    original = module._numeric
    validated = []

    def numeric(value, *args, **kwargs):
        validated.append(value)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(module, "_numeric", numeric)
    result = ingest(
        table,
        tmp_path / "data.feather",
        response="y",
        dtypes="preserve",
    )
    assert validated == values
    np.testing.assert_array_equal(result.y, [0.25, 0.5])


def test_preserve_extension_response_uses_logical_values(tmp_path, scaled_type):
    response = pa.ExtensionArray.from_storage(scaled_type, pa.array([1.0, 2.0, 3.0]))
    result = ingest(
        pa.table({"x": [1, 2, 3], "y": response}),
        tmp_path / "data.feather",
        response="y",
        dtypes="preserve",
    )
    np.testing.assert_array_equal(result.y, [2.0, 4.0, 6.0])


def test_preserve_extension_feature_is_rejected(tmp_path, scaled_type):
    feature = pa.ExtensionArray.from_storage(scaled_type, pa.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="Column 'x'.*preserve"):
        ingest(
            pa.table({"x": feature}),
            tmp_path / "data.feather",
            dtypes="preserve",
        )


@pytest.mark.parametrize(
    "values, expected",
    [([1, 2, 3], [1.0, 2.0, 3.0]), (["1.5", "2.5", "3.5"], [1.5, 2.5, 3.5])],
)
def test_preserve_dictionary_response_uses_logical_values(tmp_path, values, expected):
    response = pa.array(values).dictionary_encode()
    result = ingest(
        pa.table({"x": [1, 2, 3], "y": response}),
        tmp_path / "data.feather",
        response="y",
        dtypes="preserve",
    )
    np.testing.assert_array_equal(result.y, expected)


def test_preserve_dictionary_response_error_has_cumulative_row(tmp_path):
    first = pa.array(["1", "2"]).dictionary_encode()
    second = pa.array(["3", "bad"]).dictionary_encode()
    schema = pa.schema([("x", pa.int64()), ("y", first.type)])
    producer = Once(
        schema,
        [
            pa.record_batch([[1, 2], first], schema=schema),
            pa.record_batch([[3, 4], second], schema=schema),
        ],
    )
    with pytest.raises(
        ValueError, match="Column 'y', row 4: missing or nonnumeric value 'bad'"
    ):
        ingest(
            producer,
            tmp_path / "data.feather",
            response="y",
            dtypes="preserve",
        )


def test_preserve_dictionary_response_keeps_integer_exactness(tmp_path):
    response = pa.array([2**53, 2**53 + 1]).dictionary_encode()
    with pytest.raises(ValueError, match=f"Column 'y', row 2.*value {2**53 + 1}"):
        ingest(
            pa.table({"x": [1, 2], "y": response}),
            tmp_path / "data.feather",
            response="y",
            dtypes="preserve",
        )


@pytest.mark.parametrize(
    "batches, expected",
    [
        ([pa.array([99.0, float("nan"), None]).slice(1)], "row 1.*nan"),
        ([pa.array([1.0, 2.0]), pa.array([float("nan"), None])], "row 3.*nan"),
        ([pa.array([1.0, float("nan")]), pa.array([None])], "row 2.*nan"),
    ],
)
def test_preserve_reports_first_mixed_invalid_in_row_order(tmp_path, batches, expected):
    schema = pa.schema([("x", pa.float64())])
    producer = Once(
        schema, [pa.record_batch([array], schema=schema) for array in batches]
    )
    with pytest.raises(ValueError, match=expected):
        ingest(producer, tmp_path / "data.feather", dtypes="preserve")


@pytest.mark.parametrize(
    "array",
    [
        pa.array(["1"]),
        pa.array([1], type=pa.date32()),
        pa.array([[1]]),
        pa.array(["a"]).dictionary_encode(),
    ],
)
def test_preserve_rejects_unsupported_schema_before_reading(tmp_path, array):
    table = pa.table({"x": array})
    producer = Once(table.schema, table.to_batches())
    with pytest.raises(ValueError, match="Column 'x'.*preserve"):
        ingest(producer, tmp_path / "data.feather", dtypes="preserve")
    assert producer.reads == 0


def test_auto_spool_keeps_late_integer_exactness(tmp_path):
    producer = source()
    with pytest.raises(
        ValueError, match="Column 'wide', row 1: no supported exact dtype"
    ):
        ingest(producer, tmp_path / "data.feather")
    assert producer.acquisitions == 1


def test_auto_never_reacquires_and_uses_lossless_spool(tmp_path, monkeypatch):
    schema = pa.schema([("x", pa.int64())])
    producer = Once(
        schema,
        [
            pa.record_batch([[0, 1]], schema=schema),
            pa.record_batch([[256]], schema=schema),
        ],
    )
    module = importlib.import_module("pybiglasso.ingest")
    original = module._stage

    def stage(blocks, columns, types, directory, count, decimal_rounding=()):
        assert producer.reads == 2
        assert (directory / "source.arrow").exists()
        return original(blocks, columns, types, directory, count, decimal_rounding)

    monkeypatch.setattr(module, "_stage", stage)
    result = ingest(producer, tmp_path / "data.feather")
    assert result.X.schema.types == [pa.int16()]
    assert result.X["x"].to_pylist() == [0, 1, 256]
    assert producer.acquisitions == 1


def test_midstream_failure_does_not_publish(tmp_path):
    schema = pa.schema([("x", pa.int8())])

    def batches():
        yield pa.record_batch([pa.array([1], type=pa.int8())], schema=schema)
        raise ValueError("producer failed")

    for mode in ("auto", "preserve"):
        with pytest.raises((ValueError, pa.ArrowInvalid), match="producer failed"):
            ingest(Once(schema, batches()), tmp_path / f"{mode}.feather", dtypes=mode)
        assert not (tmp_path / f"{mode}.feather").exists()


@pytest.mark.parametrize("suffix", ["feather", "parquet"])
def test_preserve_paths_and_cache_keep_field_metadata(tmp_path, suffix):
    import pyarrow.feather as feather
    import pyarrow.parquet as parquet

    schema = pa.schema(
        [pa.field("x", pa.uint64(), nullable=False, metadata={b"unit": b"count"})]
    )
    table = pa.Table.from_arrays(
        [pa.array([2**64 - 1, 2**63 + 1], type=pa.uint64())], schema=schema
    )
    path = tmp_path / f"source.{suffix}"
    (feather.write_feather if suffix == "feather" else parquet.write_table)(table, path)
    first = ingest(path, cache=True, cache_dir=tmp_path / "cache", dtypes="preserve")
    second = ingest(path, cache=True, cache_dir=tmp_path / "cache", dtypes="preserve")
    assert second.path == first.path
    assert first.X.schema.field("x") == schema.field("x")
    assert first.X["x"].to_pylist() == [2**64 - 1, 2**63 + 1]


@pytest.mark.parametrize("mode", ["auto", "preserve"])
def test_empty_streams_and_duplicate_names_are_rejected(tmp_path, mode):
    schema = pa.schema([("x", pa.int64())])
    with pytest.raises(ValueError, match="at least one data row"):
        ingest(Once(schema, []), tmp_path / "empty.feather", dtypes=mode)
    schema = pa.schema([("x", pa.int64()), ("x", pa.int64())])
    producer = Once(schema, [])
    with pytest.raises(ValueError, match="unique"):
        ingest(producer, tmp_path / "duplicate.feather", dtypes=mode)
    assert producer.reads == 0


def test_stream_selection_and_exact_overrides(tmp_path):
    table = pa.table(
        {
            "x1": [0, 1],
            "x2": [0.25, 0.5],
            "x3": [1, 2],
            "bad": [None, None],
            "y": [1, 0],
        }
    )
    result = ingest(
        table,
        tmp_path / "data.feather",
        response="y",
        features=["x2", "x3", "x1", "bad"],
        match="x*",
        exclude=["x3"],
        dtypes={"x1": "float64"},
    )
    assert result.X.column_names == ["x2", "x1"]
    assert result.X.schema.types == [pa.float32(), pa.float64()]
    assert result.y.tolist() == [1.0, 0.0]
    with pytest.raises(ValueError, match="Column 'x', row 2"):
        ingest(
            pa.table({"x": [1.0, 0.1]}),
            tmp_path / "bad.feather",
            dtypes={"x": "float32"},
        )
    with pytest.raises(ValueError, match="Column 'y', row 1"):
        ingest(
            pa.table({"x": [1], "y": [2**53 + 1]}),
            tmp_path / "bad-y.feather",
            response="y",
            dtypes="preserve",
        )


def test_existing_output_does_not_consume_stream(tmp_path):
    path = tmp_path / "data.feather"
    path.write_bytes(b"existing")
    producer = source()
    with pytest.raises(FileExistsError):
        ingest(producer, path)
    assert producer.acquisitions == 0 and path.read_bytes() == b"existing"


def test_preserve_midbyte_bool_slices_across_odd_batches(tmp_path):
    values = [
        False,
        True,
        True,
        False,
        True,
        False,
        False,
        True,
        True,
        False,
        True,
        False,
        True,
    ]
    array = pa.array([True] * 3 + values).slice(3)
    schema = pa.schema([("x", pa.bool_())])
    batches = [
        pa.record_batch([array.slice(start, length)], schema=schema)
        for start, length in [(0, 5), (5, 3), (8, 5)]
    ]
    result = ingest(
        Once(schema, batches),
        tmp_path / "data.feather",
        dtypes="preserve",
        block_size=3,
    )
    assert result.X["x"].to_pylist() == values


def test_response_first_publication_rolls_back_on_stream_failure(tmp_path, monkeypatch):
    module = importlib.import_module("pybiglasso.ingest")
    original = module.os.link
    path = tmp_path / "data.feather"

    def link(origin, destination):
        if destination == path:
            assert path.with_suffix(".feather.response").exists()
            raise OSError("publication failed")
        return original(origin, destination)

    monkeypatch.setattr(module.os, "link", link)
    with pytest.raises(OSError, match="publication failed"):
        ingest(source(), path, response="y", dtypes="preserve")
    assert not path.exists() and not path.with_suffix(".feather.response").exists()


def test_schema_drift_check_without_c_interface_normalization(tmp_path):
    # Exercises exposed batch types only. C-stream import may conceal an
    # invalid producer's original types; matching imported schemas prove none.
    module = importlib.import_module("pybiglasso.ingest")

    class Reader:
        schema = pa.schema([("x", pa.int64())])

        def __iter__(self):
            yield pa.record_batch([pa.array([1], type=pa.int64())], names=["x"])
            yield pa.record_batch([pa.array([2], type=pa.int32())], names=["x"])

    with pytest.raises(ValueError, match="schema changed"):
        list(module._stream_blocks(Reader(), ("x",), 4))


def test_stream_documentation_example_verbatim(tmp_path, monkeypatch):
    from pathlib import Path

    doc = (Path(__file__).parents[1] / "docs" / "ingest.md").read_text()
    snippet = (
        doc.split("## Arrow streams", 1)[1]
        .split("```python\n", 1)[1]
        .split("```", 1)[0]
    )
    monkeypatch.chdir(tmp_path)
    namespace = {}
    exec(snippet, namespace)
    result = namespace["data"]
    assert result.X.to_pydict() == {"x": [1, 2, 3]}
    assert result.X.schema.types == [pa.int64()]
    np.testing.assert_array_equal(result.y, [0.5, 1.5, 2.5])


@pytest.mark.recipes
def test_duckdb_relation_arrow_reader(tmp_path):
    import duckdb

    with duckdb.connect() as connection:
        reader = connection.sql(
            "SELECT i::BIGINT AS x, i/2.0 AS y FROM range(3) t(i)"
        ).fetch_record_batch(1)
        result = ingest(
            reader, tmp_path / "data.feather", response="y", dtypes="preserve"
        )
    assert result.X["x"].to_pylist() == [0, 1, 2]
    assert result.y.tolist() == [0.0, 0.5, 1.0]


def test_preserve_needs_a_typed_source(tmp_path):
    path = tmp_path / "source.csv"
    path.write_text("x\n1\n2\n")
    with pytest.raises(ValueError, match="use 'auto' or a mapping for CSV/Excel"):
        ingest(path, tmp_path / "data.feather", dtypes="preserve")
    assert not (tmp_path / "data.feather").exists()


def test_preserve_parquet_projection_keeps_unselected_columns_unread(
    tmp_path, monkeypatch
):
    import pyarrow.parquet as pq

    path = tmp_path / "source.parquet"
    pq.write_table(pa.table({"x": [1, 2], "unused": ["a", "b"], "y": [0, 1]}), path)
    original = pq.ParquetFile.iter_batches

    def batches(self, *args, **kwargs):
        assert kwargs["columns"] == ["x", "y"]
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", batches)
    result = ingest(
        path,
        tmp_path / "data.feather",
        features=["x", "unused"],
        exclude=["unused"],
        response="y",
        dtypes="preserve",
    )
    assert result.X.to_pydict() == {"x": [1, 2]}


def test_preserve_path_detects_changes_before_publication(tmp_path, monkeypatch):
    import os

    import pyarrow.parquet as pq

    path = tmp_path / "source.parquet"
    pq.write_table(pa.table({"x": [1, 2]}), path)
    module = importlib.import_module("pybiglasso.ingest")
    original = module._stage

    def stage(*args, **kwargs):
        result = original(*args, **kwargs)
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        return result

    monkeypatch.setattr(module, "_stage", stage)
    with pytest.raises(ValueError, match="Source changed"):
        ingest(path, tmp_path / "data.feather", dtypes="preserve")
    assert not (tmp_path / "data.feather").exists()


@pytest.mark.parametrize("dtypes", ["auto", {"x": "int16"}])
def test_spool_accepts_fixed_schema_replacement_dictionaries(tmp_path, dtypes):
    # Each batch has the same physical type but a different valid dictionary.
    arrays = [pa.array(values).dictionary_encode() for values in ([1, 2], [3, 4])]
    schema = pa.schema([("x", arrays[0].type)])
    producer = Once(
        schema, [pa.record_batch([array], schema=schema) for array in arrays]
    )
    result = ingest(producer, tmp_path / "data.feather", dtypes=dtypes, block_size=1)
    assert result.X["x"].to_pylist() == [1, 2, 3, 4]
    assert result.X.schema.types == [pa.int8() if dtypes == "auto" else pa.int16()]
    assert producer.acquisitions == 1 and producer.reads == 2
