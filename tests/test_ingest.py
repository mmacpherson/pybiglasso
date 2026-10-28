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
    np.testing.assert_array_equal(result.response, expected["outcome"].to_numpy())
    assert not result.response.flags.writeable
    assert result.design.shape == (9, 6)


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
    assert result.response is None


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
    np.testing.assert_array_equal(second.response, [0.0, 1.0])
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


def test_cache_metadata_matches_design(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("x\n1\n2\n")
    result = ingest(source, cache=True, cache_dir=tmp_path / "cache")
    metadata_path = result.path.with_suffix(".arrow.json")
    metadata = json.loads(metadata_path.read_text())
    metadata["rows"] = 3
    metadata_path.write_text(json.dumps(metadata))
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
    assert np.signbit(result.design._columns[0][0][0])
    source.write_text("x\n9007199254740993.0\n")
    with pytest.raises(ValueError, match="no supported exact dtype"):
        ingest(source, tmp_path / "integer.arrow")


def test_decimal_source_requires_exact_binary_representation(tmp_path):
    source = tmp_path / "source.parquet"
    pq.write_table(pa.table({"x": [Decimal("0.25"), Decimal("0.50")]}), source)
    result = ingest(source, tmp_path / "exact.arrow")
    np.testing.assert_array_equal(result.design._columns[0][0], [0.25, 0.5])
    pq.write_table(pa.table({"x": [Decimal("0.1")]}), source)
    with pytest.raises(ValueError, match="Column 'x', row 1: no supported exact dtype"):
        ingest(source, tmp_path / "inexact.arrow")


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
    np.testing.assert_array_equal(result.response, [row[2] for row in rows])
