"""Ingest artifacts describe and reopen themselves without the source file."""

import hashlib
import importlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as pf
import pytest

from pybiglasso.ingest import ingest

KEY = b"pybiglasso.ingest"


def loader(path):
    return importlib.import_module("pybiglasso.ingest").load(path)


def metadata(**changes):
    return (
        dict(version=1, rows=2, features=["x", "z"], response="y", key="0" * 64)
        | changes
    )


def write_artifact(path, *, meta=None, columns=None):
    table = pa.table({"x": [2, 3], "z": [0.25, 0.5]} if columns is None else columns)
    if meta is not None:
        table = table.replace_schema_metadata({KEY: json.dumps(meta).encode()})
    pf.write_feather(table, path, compression="uncompressed")


@pytest.mark.parametrize("response", ["y", None])
def test_load_round_trip_after_source_deleted_and_files_moved(tmp_path, response):
    source = tmp_path / "source.csv"
    source.write_text("id,z,x,y\n10,0.25,2,-0.0\n11,0.5,3,1.25\n")
    out = tmp_path / "data.feather"
    result = ingest(source, out, response=response, features=["x", "z"])
    assert result.X.to_pydict() == {"x": [2, 3], "z": [0.25, 0.5]}
    del result
    source.unlink()
    moved = tmp_path / "kept" / "renamed.feather"
    moved.parent.mkdir()
    out.rename(moved)
    if response:
        out.with_suffix(".feather.response").rename(
            moved.with_suffix(".feather.response")
        )
    data = loader(moved)
    assert data.path == moved.absolute()
    assert data.feature_names == ("x", "z")
    assert data.X.to_pydict() == {"x": [2, 3], "z": [0.25, 0.5]}
    if response:
        assert isinstance(data.y, np.memmap) and data.y.mode == "r"
        assert not data.y.flags.writeable
        assert data.y.dtype == np.dtype("float64")
        assert data.y.tobytes() == np.array([-0.0, 1.25], dtype=np.float64).tobytes()
    else:
        assert data.y is None


def test_schema_metadata_and_standard_reader_column_order(tmp_path):
    import polars as pl

    source = tmp_path / "source.csv"
    source.write_text("id,z,y,x\n10,0.25,0,2\n11,0.5,1,3\n")
    out = tmp_path / "data.feather"
    data = ingest(source, out, response="y", exclude=["id"])
    assert {p.name for p in tmp_path.glob("data.feather*")} == {
        "data.feather",
        "data.feather.response",
    }
    schema = pa.ipc.open_file(out).schema
    assert set(schema.metadata) == {KEY}
    recorded = json.loads(schema.metadata[KEY])
    stat = source.stat()
    options = dict(
        version=5,
        source=dict(
            path=str(source.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns
        ),
        features=["z", "x"],
        response="y",
        dtypes={},
        sheet=None,
        block_size=65536,
        excel_shared_strings_limit=1 << 30,
    )
    expected_key = hashlib.sha256(
        json.dumps(options, sort_keys=True).encode()
    ).hexdigest()
    assert recorded == metadata(features=["z", "x"], key=expected_key)
    assert schema.names == ["z", "x"]
    assert pf.read_table(out).to_pydict() == {"z": [0.25, 0.5], "x": [2, 3]}
    assert pl.read_ipc(out).columns == ["z", "x"]
    assert pl.read_ipc(out).to_dict(as_series=False) == {"z": [0.25, 0.5], "x": [2, 3]}
    assert data.feature_names == ("z", "x")


@pytest.mark.parametrize("size", [0, 8, 15, 17, 24])
def test_load_rejects_response_wrong_size(tmp_path, size):
    out = tmp_path / "data.feather"
    write_artifact(out, meta=metadata())
    out.with_suffix(".feather.response").write_bytes(bytes(size))
    with pytest.raises(ValueError, match="response.*size|size.*response"):
        loader(out)


def test_load_rejects_missing_response(tmp_path):
    out = tmp_path / "data.feather"
    write_artifact(out, meta=metadata())
    with pytest.raises(FileNotFoundError, match="response"):
        loader(out)


@pytest.mark.parametrize(
    "version, compression", [(2, "uncompressed"), (2, "lz4"), (1, None)]
)
def test_plain_feather_has_actionable_error(tmp_path, version, compression):
    out = tmp_path / "plain.feather"
    pf.write_feather(
        pa.table({"x": [1, 2]}), out, version=version, compression=compression
    )
    with pytest.raises(
        ValueError, match=r"ingest metadata.*pyarrow.feather.read_table"
    ):
        loader(out)


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"rows": 3}, "rows"),
        ({"features": ["z", "x"]}, "features"),
        ({"features": ["x"]}, "features"),
    ],
)
def test_load_validates_rows_and_features(tmp_path, changes, match):
    out = tmp_path / "data.feather"
    write_artifact(out, meta=metadata(response=None, **changes))
    with pytest.raises(ValueError, match=match):
        loader(out)


@pytest.mark.parametrize(
    "meta",
    [
        [],
        {},
        metadata(version=2),
        metadata(version=True),
        metadata(rows=0),
        metadata(rows=True),
        metadata(features=["x", "x"]),
        metadata(response=7),
        metadata(response="x"),
        metadata(key=5),
    ],
)
def test_load_rejects_invalid_metadata(tmp_path, meta):
    out = tmp_path / "data.feather"
    write_artifact(out, meta=meta)
    with pytest.raises(ValueError, match="ingest.*metadata|ingest.*version"):
        loader(out)


def test_load_rejects_invalid_metadata_json(tmp_path):
    out = tmp_path / "data.feather"
    table = pa.table({"x": [2, 3]}).replace_schema_metadata({KEY: b"not json"})
    pf.write_feather(table, out, compression="uncompressed")
    with pytest.raises(ValueError, match="ingest metadata"):
        loader(out)


def test_publication_exposes_complete_metadata_and_rolls_back_own_links(
    tmp_path, monkeypatch
):
    module = importlib.import_module("pybiglasso.ingest")
    source = tmp_path / "source.csv"
    source.write_text("x,y\n2,0.25\n3,0.5\n")
    out = tmp_path / "data.feather"
    link = os.link
    published = []

    def publish(origin, destination):
        destination = Path(destination)
        if destination == out:
            assert (
                out.with_suffix(".feather.response").read_bytes()
                == np.array([0.25, 0.5], dtype=np.float64).tobytes()
            )
            assert (
                json.loads(pa.ipc.open_file(origin).schema.metadata[KEY])["response"]
                == "y"
            )
        link(origin, destination)
        published.append(destination)

    monkeypatch.setattr(module.os, "link", publish)
    ingest(source, out, response="y")
    assert published == [out.with_suffix(".feather.response"), out]

    failed = tmp_path / "failed.feather"

    def collision(origin, destination):
        if Path(destination) == failed:
            failed.write_bytes(b"another writer's file")
            raise FileExistsError("publication collision")
        link(origin, destination)

    monkeypatch.setattr(module.os, "link", collision)
    with pytest.raises(FileExistsError, match="collision"):
        ingest(source, failed, response="y")
    assert failed.read_bytes() == b"another writer's file"
    assert not failed.with_suffix(".feather.response").exists()


def test_cache_version_bump_avoids_legacy_output(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("x\n2\n3\n")
    stat = source.stat()
    options = dict(
        version=4,
        source=dict(
            path=str(source.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns
        ),
        features=("x",),
        response=None,
        dtypes={},
        sheet=None,
        block_size=65536,
        excel_shared_strings_limit=1 << 30,
    )
    old_key = hashlib.sha256(json.dumps(options, sort_keys=True).encode()).hexdigest()
    cache = tmp_path / "cache"
    legacy = cache / old_key / "data.feather"
    legacy.parent.mkdir(parents=True)
    pf.write_feather(pa.table({"x": [2, 3]}), legacy, compression="uncompressed")
    legacy.with_suffix(".feather.json").write_text(
        json.dumps(dict(key=old_key, rows=2, features=["x"], response=None))
    )
    data = ingest(source, cache=True, cache_dir=cache)
    assert data.path != legacy
    assert data.X.to_pydict() == {"x": [2, 3]}
    assert legacy.exists()


@pytest.mark.parametrize("damage", ["missing", "truncated"])
def test_cache_reuse_validates_response_companion(tmp_path, damage):
    source = tmp_path / "source.csv"
    source.write_text("x,y\n2,0.25\n3,0.5\n")
    cache = tmp_path / "cache"
    result = ingest(source, response="y", cache=True, cache_dir=cache)
    response = result.path.with_suffix(".feather.response")
    del result
    if damage == "missing":
        response.unlink()
        error = FileNotFoundError
    else:
        response.write_bytes(bytes(8))
        error = ValueError
    with pytest.raises(error, match="response"):
        ingest(source, response="y", cache=True, cache_dir=cache)


def test_cache_reuses_embedded_key_without_reading_source_data(tmp_path, monkeypatch):
    source = tmp_path / "source.csv"
    source.write_text("x,y\n2,0.25\n3,0.5\n")
    cache = tmp_path / "cache"
    first = ingest(source, response="y", cache=True, cache_dir=cache)
    module = importlib.import_module("pybiglasso.ingest")
    monkeypatch.setattr(
        module, "_blocks", lambda *a, **kw: pytest.fail("cache hit converted source")
    )
    second = ingest(source, response="y", cache=True, cache_dir=cache)
    assert second.path == first.path
    assert not second.path.with_suffix(".feather.json").exists()
    assert second.X.to_pydict() == {"x": [2, 3]}
    np.testing.assert_array_equal(second.y, [0.25, 0.5])


def test_load_borrows_readonly_feature_buffers(tmp_path, monkeypatch):
    source = tmp_path / "source.csv"
    source.write_text("x,y\n0.25,0.5\n0.75,1.5\n")
    ingest(source, tmp_path / "data.feather", response="y")
    module = importlib.import_module("pybiglasso.ingest")
    original = module._open_mapped
    mapped = []

    def open_mapped(path):
        table = original(path)
        mapped.append(table)
        return table

    monkeypatch.setattr(module, "_open_mapped", open_mapped)
    data = loader(tmp_path / "data.feather")
    assert data.X is mapped[0]
    assert data.X.to_pydict() == {"x": [0.25, 0.75]}
    assert all(
        not buffer.is_mutable
        for column in data.X.columns
        for chunk in column.chunks
        for buffer in chunk.buffers()
        if buffer is not None
    )
