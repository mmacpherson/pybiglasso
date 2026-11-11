"""Source-reader expectations, independent of pybiglasso's converter."""

import csv
import json
import re
from decimal import Decimal

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
import pyarrow.csv as pc  # noqa: E402
import pyarrow.feather as feather  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from pybiglasso.ingest import ingest  # noqa: E402


@pytest.fixture
def source_table():
    return pa.table(
        {
            "bit": [True, False, True, False, True, False, True, False, True],
            "signed": [-128, 127, 0, 1, -1, 2, 3, 4, 5],
            "unsigned": [128, 255, 200, 201, 202, 203, 204, 205, 206],
            "wide": [-129, 256, 1000, 2000, 3000, 4000, 5000, 6000, 7000],
            "single": [0.25, 0.5, 0.75, 1.25, 1.5, 1.75, 2.25, 2.5, 2.75],
            "double": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "outcome": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        }
    )


@pytest.mark.parametrize("format", ["csv", "parquet", "feather"])
def test_source_round_trip_and_exact_types(tmp_path, source_table, format):
    source = tmp_path / f"source.{format}"
    if format == "csv":
        pc.write_csv(source_table, source)
        expected = pc.read_csv(source)
    elif format == "parquet":
        pq.write_table(source_table, source, row_group_size=2)
        expected = pq.read_table(source)
    else:
        feather.write_feather(source_table, source, compression="lz4", chunksize=2)
        expected = feather.read_table(source)
    result = ingest(source, tmp_path / "X.arrow", response="outcome", block_size=128)
    with pa.memory_map(str(result.path)) as mapping:
        reader = pa.ipc.open_file(mapping)
        assert reader.num_record_batches == 1
        actual = reader.get_batch(0)
        assert actual.schema.types == [
            pa.bool_(),
            pa.int8(),
            pa.uint8(),
            pa.int16(),
            pa.float32(),
            pa.float64(),
        ]
        assert actual.to_pydict() == expected.drop(["outcome"]).to_pydict()
        assert all(array.null_count == 0 for array in actual.columns)
        # Bool is nine bits, not nine unpacked bytes.
        assert actual.column("bit").buffers()[1].size == 2
    assert result.feature_names == tuple(expected.column_names[:-1])
    np.testing.assert_array_equal(result.y, expected["outcome"].to_numpy())
    assert not result.y.flags.writeable
    assert result.X.shape == (9, 6)


def test_default_csv_block_handles_very_wide_rows_and_headers(tmp_path):
    wide = tmp_path / "wide.csv"
    names = [f"feature_{index:05d}" for index in range(20_000)]
    wide.write_text(",".join(names) + "\n" + ",".join(["1"] * len(names)) + "\n")
    result = ingest(wide, tmp_path / "wide.arrow")
    assert result.feature_names == tuple(names)
    assert result.X.shape == (1, 20_000)
    assert result.X.column(19_999)[0].as_py() == 1

    long_name = "x" * 200_000
    long_header = tmp_path / "long-header.csv"
    long_header.write_text(f"{long_name},y\n2,0\n3,1\n")
    result = ingest(long_header, tmp_path / "long-header.arrow", response="y")
    assert result.feature_names == (long_name,)
    assert result.X.column(0).to_pylist() == [2, 3]
    np.testing.assert_array_equal(result.y, [0.0, 1.0])


@pytest.mark.parametrize("header,row", [("x", "1"), ("x,y", "1,0")])
def test_csv_rejects_blank_data_lines(tmp_path, header, row):
    source = tmp_path / "blank.csv"
    source.write_text(f"{header}\n{row}\n\n{row}\n")
    with pytest.raises(ValueError, match=r"Column 'x', row 2"):
        ingest(source, tmp_path / "blank.arrow")


def test_csv_ragged_row_reports_parse_error_not_block_size(tmp_path):
    source = tmp_path / "ragged.csv"
    source.write_text("x,y,z\n1,2,3\n4,5\n")
    with pytest.raises(ValueError, match=r"Expected 3 columns, got 2") as error:
        ingest(source, tmp_path / "ragged.arrow")
    assert "block_size" not in str(error.value)


@pytest.mark.parametrize(
    "content", [b"x,y\n1,0\n", b"x,y\r\n1,0\r\n", b"\xef\xbb\xbfx,y\n1,0\n"]
)
def test_csv_trailing_newline_crlf_and_bom(tmp_path, content):
    source = tmp_path / "ordinary.csv"
    source.write_bytes(content)
    result = ingest(source, tmp_path / "ordinary.arrow", response="y")
    assert result.X.column("x").to_pylist() == [1]
    np.testing.assert_array_equal(result.y, [0.0])


def test_selection_order_glob_regex_and_unselected_bad_values(tmp_path):
    source = tmp_path / "source.parquet"
    pq.write_table(
        pa.table({"x1": [2, 3], "x2": [4, 5], "bad": [None, None], "y": [0, 1]}), source
    )
    result = ingest(
        source,
        tmp_path / "glob.arrow",
        features=["x2", "x1"],
        match="x*",
        exclude=["x1"],
        response="y",
    )
    assert result.feature_names == ("x2",)
    result = ingest(source, tmp_path / "regex.arrow", match=re.compile(r"^x[12]$"))
    assert result.feature_names == ("x1", "x2")
    assert result.y is None


@pytest.mark.parametrize(
    "bad", ["", "NA", "text", "nan", "inf", "-inf", "9007199254740993"]
)
def test_rejection_names_column_and_data_row(tmp_path, bad):
    source = tmp_path / "source.csv"
    with source.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerows([["x", "y"], [1, 0], [bad, 1]])
    with pytest.raises(ValueError, match=r"Column 'x', row 2"):
        ingest(source, tmp_path / "X.arrow", response="y")
    assert not (tmp_path / "X.arrow").exists()


def test_explicit_dtype_requires_exactness_and_preserves_signed_zero(tmp_path):
    source = tmp_path / "source.parquet"
    pq.write_table(pa.table({"x": [0.1, -0.0]}), source)
    with pytest.raises(ValueError, match="no supported exact dtype"):
        ingest(source, tmp_path / "bad.arrow", dtypes={"x": "float32"})
    result = ingest(source, tmp_path / "good.arrow", dtypes={"x": "float64"})
    with pa.memory_map(str(result.path)) as mapping:
        values = pa.ipc.open_file(mapping).get_batch(0).column(0).to_numpy()
        assert values[0] == 0.1
        assert np.signbit(values[1])


def test_cache_reuse_key_and_persistent_artifacts(tmp_path, monkeypatch):
    source = tmp_path / "source.csv"
    source.write_text("x,y\n2,0\n3,1\n")
    first = ingest(source, response="y", cache=True, cache_dir=tmp_path / "cache")
    import importlib

    module = importlib.import_module("pybiglasso.ingest")
    original = module._blocks
    monkeypatch.setattr(
        module, "_blocks", lambda *_: pytest.fail("Cache hit read source data")
    )
    second = ingest(source, response="y", cache=True, cache_dir=tmp_path / "cache")
    assert second.path == first.path
    assert second.feature_names == ("x",)
    np.testing.assert_array_equal(second.y, [0.0, 1.0])
    monkeypatch.setattr(module, "_blocks", original)
    changed = ingest(source, cache=True, cache_dir=tmp_path / "cache")
    assert changed.path != first.path
    source.write_text("x,y\n2,0\n4,1\n")
    changed_source = ingest(
        source, response="y", cache=True, cache_dir=tmp_path / "cache"
    )
    assert changed_source.path != first.path
    assert first.path.exists()


def test_duplicate_names_empty_input_and_existing_output(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("x,x\n1,2\n")
    with pytest.raises(ValueError, match="unique"):
        ingest(source, tmp_path / "X.arrow")
    source.write_text("x\n")
    with pytest.raises(ValueError, match="at least one data row"):
        ingest(source, tmp_path / "X.arrow")
    source.write_text("x\n1\n")
    output = tmp_path / "X.arrow"
    output.write_bytes(b"untouched")
    with pytest.raises(FileExistsError):
        ingest(source, output)
    assert output.read_bytes() == b"untouched"


def test_source_change_between_passes_rejected(tmp_path, monkeypatch):
    source = tmp_path / "source.csv"
    source.write_text("x\n1\n2\n")
    import importlib

    module = importlib.import_module("pybiglasso.ingest")
    original = module._infer

    def change_source(*args):
        result = original(*args)
        source.write_text("x\n3\n4\n5\n")
        return result

    monkeypatch.setattr(module, "_infer", change_source)
    with pytest.raises(ValueError, match="Source changed"):
        ingest(source, tmp_path / "X.arrow")


def test_cache_metadata_matches_features(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("x\n1\n2\n")
    result = ingest(source, cache=True, cache_dir=tmp_path / "cache")
    path = result.path
    del result  # release the feature mapping before rewriting, including Windows
    table = feather.read_table(path, memory_map=False)
    metadata = json.loads(table.schema.metadata[b"pybiglasso.ingest"])
    metadata["rows"] = 3
    table = table.replace_schema_metadata(
        {b"pybiglasso.ingest": json.dumps(metadata).encode()}
    )
    feather.write_feather(table, path, compression="uncompressed")
    with pytest.raises(ValueError, match="does not match"):
        ingest(source, cache=True, cache_dir=tmp_path / "cache")


@pytest.mark.parametrize("cache", [False, True])
def test_explicit_persistent_destination_required(tmp_path, cache):
    source = tmp_path / "source.csv"
    source.write_text("x\n1\n2\n")
    with pytest.raises(ValueError, match="persistent out or cache_dir"):
        ingest(source, cache=cache)


def test_explicit_cache_directory_without_reuse_is_persistent(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("x\n1\n2\n")
    directory = tmp_path / "artifacts"
    first = ingest(source, cache_dir=directory)
    second = ingest(source, cache_dir=directory)
    assert first.path != second.path
    assert first.path.is_absolute() and second.path.is_absolute()
    assert first.path.is_relative_to(directory)
    assert first.path.exists() and second.path.exists()


@pytest.mark.parametrize("dtype", ["uint16", "int32", "uint32", "int64", "uint64"])
def test_unsupported_explicit_storage_dtype_rejected(tmp_path, dtype):
    source = tmp_path / "source.csv"
    source.write_text("x\n1\n2\n")
    with pytest.raises(ValueError, match="supported types"):
        ingest(source, tmp_path / "X.arrow", dtypes={"x": dtype})


@pytest.mark.parametrize("value", [32768, 65535, 2**40 + 1, 2**53, 2**54])
def test_wider_source_integer_uses_exact_supported_representation(tmp_path, value):
    source = tmp_path / "source.parquet"
    pq.write_table(pa.table({"x": pa.array([value], type=pa.uint64())}), source)
    result = ingest(source, tmp_path / "X.arrow")
    with pa.memory_map(str(result.path)) as mapping:
        column = pa.ipc.open_file(mapping).get_batch(0).column(0)
        assert column.type == pa.float64()
        assert column[0].as_py() == value


@pytest.mark.parametrize("value", [-(2**53) - 1, 2**53 + 1, 2**64 - 1])
def test_wide_integer_text_never_rounds_into_float64(tmp_path, value):
    source = tmp_path / "source.csv"
    source.write_text(f"x\n{value}\n")
    with pytest.raises(ValueError, match="Column 'x', row 1: no supported exact dtype"):
        ingest(source, tmp_path / "X.arrow")


def test_csv_preserves_large_integer_and_signed_zero_spelling(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("x\n-0.0\n0.25\n")
    result = ingest(source, tmp_path / "zero.arrow")
    assert np.signbit(result.X.column(0).chunk(0)[0].as_py())
    source.write_text("x\n9007199254740993.0\n")
    with pytest.raises(ValueError, match="no supported exact dtype"):
        ingest(source, tmp_path / "integer.arrow")


def test_decimal_source_requires_exact_binary_representation(tmp_path):
    source = tmp_path / "source.parquet"
    pq.write_table(pa.table({"x": [Decimal("0.25"), Decimal("0.50")]}), source)
    result = ingest(source, tmp_path / "exact.arrow")
    np.testing.assert_array_equal(result.X.column(0).to_numpy(), [0.25, 0.5])
    pq.write_table(pa.table({"x": [Decimal("0.1")]}), source)
    with pytest.raises(ValueError, match="Column 'x', row 1: no supported exact dtype"):
        ingest(source, tmp_path / "inexact.arrow")
    result = ingest(source, tmp_path / "rounded.arrow", dtypes={"x": "float64"})
    assert result.X.column(0)[0].as_py() == float(Decimal("0.1"))

    pq.write_table(
        pa.table(
            {
                "x": [Decimal("0.25"), Decimal("0.50")],
                "y": [Decimal("9007199254740993"), Decimal("9007199254740995")],
            }
        ),
        source,
    )
    with pytest.raises(ValueError, match="Column 'y'.*no supported exact dtype"):
        ingest(
            source,
            tmp_path / "response.arrow",
            response="y",
            dtypes={"x": "float64"},
        )
    pq.write_table(pa.table({"x": [Decimal("1.00000000000000000001")]}), source)
    with pytest.raises(ValueError, match="Column 'x'.*no supported exact dtype"):
        ingest(source, tmp_path / "float32.arrow", dtypes={"x": "float32"})


@pytest.mark.parametrize("format", ["stream", "parquet", "feather"])
@pytest.mark.parametrize("dtypes", ["auto", {"x": "int8"}, "preserve"])
def test_pandas_physical_index_requires_explicit_selection(tmp_path, format, dtypes):
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"x": [1, 2]}, index=pd.Index([10, 20], name="row"))
    table = pa.Table.from_pandas(frame)
    source = table
    if format == "parquet":
        source = tmp_path / "indexed.parquet"
        pq.write_table(table, source)
    elif format == "feather":
        source = tmp_path / "indexed.feather"
        feather.write_feather(table, source)
    with pytest.raises(ValueError, match="Pandas index columns.*row"):
        ingest(source, tmp_path / f"implicit-{format}.arrow", dtypes=dtypes)
    selected = ingest(
        source,
        tmp_path / f"selected-{format}.arrow",
        features=["x"],
        dtypes=dtypes,
    )
    assert selected.feature_names == ("x",)
    explicit = ingest(
        source,
        tmp_path / f"explicit-{format}.arrow",
        features=["x", "row"],
        dtypes=dtypes,
    )
    assert explicit.feature_names == ("x", "row")


def test_failed_ingest_removes_created_directories(tmp_path):
    source = tmp_path / "bad.csv"
    source.write_text("x\nnot-a-number\n")
    cache = tmp_path / "cache"
    with pytest.raises(ValueError):
        ingest(source, cache_dir=cache)
    assert list(cache.iterdir()) == []

    output = tmp_path / "new" / "nested" / "data.arrow"
    with pytest.raises(ValueError):
        ingest(source, output)
    assert not (tmp_path / "new").exists()

    stream_output = tmp_path / "stream" / "nested" / "data.arrow"
    with pytest.raises(ValueError):
        ingest(pa.table({"x": [1, None]}), stream_output)
    assert not (tmp_path / "stream").exists()


def test_failed_ingest_does_not_remove_concurrent_output(tmp_path, monkeypatch):
    import importlib

    module = importlib.import_module("pybiglasso.ingest")
    bad = tmp_path / "bad.csv"
    good = tmp_path / "good.csv"
    bad.write_text("x\nnot-a-number\n")
    good.write_text("x\n1\n")
    original = module._infer

    def interleave(*args, **kwargs):
        monkeypatch.setattr(module, "_infer", original)
        ingest(good, tmp_path / "shared" / "successful.arrow")
        raise ValueError("first ingest failed")

    monkeypatch.setattr(module, "_infer", interleave)
    with pytest.raises(ValueError, match="first ingest failed"):
        ingest(bad, tmp_path / "shared" / "failed.arrow")
    assert (tmp_path / "shared" / "successful.arrow").exists()


def test_ingest_option_validation_is_specific(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("x\n1\n")
    with pytest.raises(ValueError, match="unselected or unknown.*missing"):
        ingest(source, tmp_path / "unknown.arrow", dtypes={"missing": "float64"})
    with pytest.raises(ValueError, match="Unsupported dtype.*int64"):
        ingest(source, tmp_path / "dtype.arrow", dtypes={"x": "int64"})
    with pytest.raises(ValueError, match="sheet is only valid"):
        ingest(source, tmp_path / "sheet.arrow", sheet="Data")
    with pytest.raises(ValueError, match="cache_dir cannot be used"):
        ingest(
            source,
            tmp_path / "out.arrow",
            cache_dir=tmp_path / "cache",
        )
    result = ingest(source, tmp_path / "numpy-int.arrow", block_size=np.int64(1))
    assert result.X.shape == (1, 1)


def test_csv_text_parsing_edge_cases(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("x\n1_000\n")
    with pytest.raises(ValueError, match="1_000"):
        ingest(source, tmp_path / "underscore.arrow")
    source.write_text("x\n1e-400\n")
    with pytest.raises(ValueError, match="1e-400"):
        ingest(source, tmp_path / "underflow.arrow")
    source.write_text("x\n1e400\n")
    with pytest.raises(ValueError, match="1e400"):
        ingest(source, tmp_path / "overflow.arrow")
    source.write_text("x\ntrue\n2\n")
    assert ingest(source, tmp_path / "mixed.arrow").X.column(0).to_pylist() == [1, 2]
    source.write_text("x\n-0\n")
    assert ingest(source, tmp_path / "minus-zero.arrow").X.column(0).to_pylist() == [
        False
    ]


def test_excel_round_trip_against_independent_reader(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    source = tmp_path / "source.xlsx"
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.title = "Data"
    worksheet.append(["x", "bit", "outcome"])
    for row in [(2, True, 0.25), (3, False, 0.5), (4, True, 0.75)]:
        worksheet.append(row)
    workbook.save(source)
    workbook.close()
    independent = openpyxl.load_workbook(source, read_only=True, data_only=True)
    try:
        rows = list(independent["Data"].values)[1:]
    finally:
        independent.close()
    result = ingest(
        source, tmp_path / "X.arrow", response="outcome", sheet="Data", block_size=2
    )
    with pa.memory_map(str(result.path)) as mapping:
        actual = pa.ipc.open_file(mapping).get_batch(0)
        assert actual.column("x").to_pylist() == [row[0] for row in rows]
        assert actual.column("bit").to_pylist() == [row[1] for row in rows]
    np.testing.assert_array_equal(result.y, [row[2] for row in rows])
    with pytest.raises(ValueError, match="Unknown Excel sheet.*Missing"):
        ingest(source, tmp_path / "missing-sheet.arrow", sheet="Missing")


def test_receipt_contains_standard_table_and_file_reopens_without_sidecar(tmp_path):
    import pyarrow.feather as pf

    from pybiglasso import BiglassoRegressor, biglasso

    source = tmp_path / "source.csv"
    source.write_text(
        "x,z,y\n" + "\n".join(f"{i % 7},{i % 3},{i % 2}" for i in range(41)) + "\n"
    )
    result = ingest(source, tmp_path / "features.arrow", response="y")
    assert isinstance(result.X, pa.Table)
    assert result.feature_names == tuple(result.X.column_names)
    assert result.X.column_names == ["x", "z"]
    with pytest.raises((TypeError, ValueError)):
        biglasso(result, result.y, nlambda=3)
    with pytest.raises((TypeError, ValueError)):
        BiglassoRegressor().fit(result, result.y)
    assert not result.path.with_suffix(result.path.suffix + ".json").exists()
    reopened = pf.read_table(result.path, memory_map=True)
    with pa.memory_map(str(result.path), "r") as stream:
        ordinary = pa.ipc.open_file(stream).read_all()
    assert reopened.equals(ordinary)
    assert reopened.equals(result.X)
    actual = biglasso(reopened, result.y, nlambda=3)
    reference = biglasso(result.X, result.y, nlambda=3)
    for field in ("lambdas", "intercept", "iterations", "deviance"):
        assert getattr(actual, field).tobytes() == getattr(reference, field).tobytes()


def test_ingest_rejects_plain_file_objects(tmp_path):
    import io

    with pytest.raises(TypeError, match="filesystem path or an Arrow stream"):
        ingest(io.BytesIO(b"x\n1\n2\n"), tmp_path / "out.arrow")
    assert not (tmp_path / "out.arrow").exists()
