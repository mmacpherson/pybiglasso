"""Blockwise conversion from paths and Arrow streams to persistent Arrow files."""

from __future__ import annotations

import csv
import fnmatch
import hashlib
import json
import math
import numbers
import os
import re
import struct
import tempfile
import zipfile
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TYPE_CHECKING
from xml.parsers import expat

import numpy as np

from .arrow import _open_mapped

__all__ = ["IngestResult", "ingest", "load"]

if TYPE_CHECKING:
    import pyarrow as pa

# Inference candidates deliberately narrower than the fitting input contract.
_TYPES = ("bool", "int8", "uint8", "int16", "float32", "float64")
_INTEGER = re.compile(r"[+-]?\d+\Z")
_METADATA_KEY = b"pybiglasso.ingest"
_FORMAT_VERSION = 1
_RESPONSE_DTYPE = "<f8"
_MAX_BLOCK_CELLS = 1 << 18


@dataclass(frozen=True)
class IngestResult:
    """Persistent output: read-only features X and an optional response y.

    Files are not deleted when this object is collected. This lets a retained
    X or y outlive the result, including on Windows.
    """

    X: pa.Table
    y: np.ndarray | None
    feature_names: tuple[str, ...]
    path: Path


def _arrow():
    try:
        import pyarrow as pa
    except ImportError as error:
        raise ImportError("ingest needs pip install 'pybiglasso[arrow]'") from error
    return pa


def _names(values):
    names = tuple(values)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Source columns must have nonempty string names.")
    if len(set(names)) != len(names):
        raise ValueError("Source column names must be unique.")
    return names


def _selection(names, response, features, exclude, match):
    if response is not None and response not in names:
        raise ValueError(f"Unknown response column: {response!r}")
    if isinstance(features, str) or isinstance(exclude, str):
        raise ValueError("features and exclude must be sequences of column names.")
    selected = list(names if features is None else features)
    excluded = set(() if exclude is None else exclude)
    if len(set(selected)) != len(selected):
        raise ValueError("Feature names must be unique.")
    unknown = (set(selected) | excluded) - set(names)
    if unknown:
        raise ValueError(f"Unknown column: {sorted(unknown)!r}")
    if match is not None and not isinstance(match, (str, re.Pattern)):
        raise ValueError("match must be a glob string or compiled regular expression.")
    selected = [
        name
        for name in selected
        if name != response
        and name not in excluded
        and (
            match is None
            or (
                bool(match.search(name))
                if isinstance(match, re.Pattern)
                else fnmatch.fnmatchcase(name, match)
            )
        )
    ]
    if not selected:
        raise ValueError("Select at least one feature column.")
    return tuple(selected)


def _numeric(value, column, row, dtype=None):
    """Preserve integer precision before testing any floating representation."""
    if isinstance(value, Decimal):
        if not value.is_finite() or value.adjusted() > 308:
            raise ValueError(f"Column {column!r}, row {row}: nonfinite numeric value.")
        if dtype is not None and np.dtype(dtype).kind == "f":
            value = float(value)
        elif value == value.to_integral_value():
            value = int(value)
        else:
            converted = float(value)
            if Decimal.from_float(converted) != value:
                raise ValueError(
                    f"Column {column!r}, row {row}: no supported exact dtype."
                )
            value = converted
    if isinstance(value, str):
        text = value.strip()
        try:
            if text.lower() in ("true", "false"):
                value = text.lower() == "true"
            elif _INTEGER.fullmatch(text):
                value = int(text)
            else:
                number = Decimal(text)
                if not number.is_finite() or number.adjusted() > 308:
                    raise ValueError
                # Integral decimal/exponent spellings must not round via f64.
                value = (
                    -0.0
                    if number.is_zero() and number.is_signed()
                    else (
                        int(number)
                        if number == number.to_integral_value()
                        else float(number)
                    )
                )
        except (ValueError, InvalidOperation, OverflowError):
            raise ValueError(
                f"Column {column!r}, row {row}: missing or nonnumeric value."
            ) from None
    if not isinstance(value, (bool, int, float)):
        raise ValueError(f"Column {column!r}, row {row}: missing or nonnumeric value.")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"Column {column!r}, row {row}: nonfinite numeric value.")
    return value


def _exact(value, dtype):
    if np.dtype(dtype).kind in "biu":
        if isinstance(value, float) and value == 0 and math.copysign(1, value) < 0:
            return False
        limits = None if dtype == "bool" else np.iinfo(dtype)
        lower, upper = (0, 1) if limits is None else (limits.min, limits.max)
        return lower <= value <= upper and int(value) == value
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            cast = float(np.dtype(dtype).type(value))
        return math.isfinite(cast) and cast == value
    except (OverflowError, ValueError):
        return False


def _headers(source, pa, sheet, excel_shared_strings_limit):
    suffix = source.suffix.lower()
    if suffix == ".csv":
        previous_limit = csv.field_size_limit()
        csv.field_size_limit(max(previous_limit, _csv_block_size(source, 65536)))
        try:
            with source.open(newline="", encoding="utf-8-sig") as stream:
                return _names(next(csv.reader(stream), ()))
        finally:
            csv.field_size_limit(previous_limit)
    if suffix in (".parquet", ".pq"):
        import pyarrow.parquet as pq

        with pq.ParquetFile(source) as reader:
            return _names(reader.schema_arrow.names)
    if suffix in (".feather", ".arrow", ".ipc"):
        with pa.memory_map(str(source), "r") as mapping:
            return _names(pa.ipc.open_file(mapping).schema.names)
    if suffix == ".xlsx":
        workbook = _workbook(source, excel_shared_strings_limit)
        try:
            worksheet = workbook.worksheets[0] if sheet is None else workbook[sheet]
            return _names(next(worksheet.iter_rows(values_only=True), ()))
        finally:
            workbook.close()
    raise ValueError("Source must be CSV, Parquet, Arrow file (Feather v2), or .xlsx.")


def _check_zip_directory(source):
    # ZipFile allocates the directory before it can expose ZipInfo sizes.
    # Bound the EOCD fields first, choosing the same last signature it uses.
    with source.open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - (65535 + 22 + 20)))
        tail = stream.read()
    offset = tail.rfind(b"PK\x05\x06")
    if offset < 0 or offset + 22 > len(tail):
        raise ValueError("Invalid XLSX ZIP directory.")
    _, disk, start_disk, disk_entries, entries, size, _, comment = struct.unpack_from(
        "<4s4H2LH", tail, offset
    )
    if offset + 22 + comment != len(tail):
        raise ValueError("Invalid XLSX ZIP directory.")
    if (
        disk
        or start_disk
        or disk_entries != entries
        or entries > 1024
        or size > 1 << 20
        or tail[max(0, offset - 20) : offset - 16] == b"PK\x06\x07"
    ):
        raise ValueError("XLSX ZIP directory exceeds the v1 size/entry limits.")


def _shared_string_parts(archive):
    # The reader selects by content type, not by the usual part filename.
    manifest_limit = 1 << 20
    manifest = archive.getinfo("[Content_Types].xml")
    if manifest.file_size > manifest_limit:
        raise ValueError("XLSX content-types manifest exceeds the 1 MiB limit.")
    with archive.open(manifest) as stream:
        data = stream.read(manifest_limit + 1)
    if len(data) > manifest_limit:
        raise ValueError("XLSX content-types manifest exceeds the 1 MiB limit.")
    parts = set()
    parser = expat.ParserCreate(namespace_separator="}")

    def start(name, attributes):
        if (
            name.rsplit("}", 1)[-1] == "Override"
            and attributes.get("ContentType")
            == "application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"
        ):
            part = attributes.get("PartName", "")
            if not part.startswith("/"):
                raise ValueError("Invalid XLSX shared-string part name.")
            parts.add(part[1:])

    def reject_doctype(*args):
        raise ValueError("XLSX content-types manifest must not contain a DTD.")

    parser.StartElementHandler = start
    parser.StartDoctypeDeclHandler = reject_doctype
    try:
        parser.Parse(data, True)
    except expat.ExpatError as error:
        raise ValueError("Invalid XLSX content-types manifest.") from error
    return parts


def _check_shared_strings(source, limit):
    # ZipInfo.file_size is the uncompressed size recorded in the directory.
    # Do not open/decompress the table merely to decide whether it is too big.
    _check_zip_directory(source)
    with zipfile.ZipFile(source) as archive:
        members = archive.infolist()
        if len(members) > 1024:
            raise ValueError("XLSX ZIP directory exceeds the v1 size/entry limits.")
        by_name = {member.filename: member for member in members}
        if len(by_name) != len(members):
            raise ValueError("XLSX ZIP contains duplicate part names.")
        parts = {
            name for name in by_name if name.rsplit("/", 1)[-1] == "sharedStrings.xml"
        }
        # Keep the cheap conventional-name rejection before manifest decoding.
        size = sum(by_name[name].file_size for name in parts)
        if size <= limit:
            parts.update(_shared_string_parts(archive))
            try:
                size = sum(by_name[name].file_size for name in parts)
            except KeyError as error:
                raise ValueError("XLSX shared-string part is missing.") from error
    if size > limit:
        raise ValueError(
            "XLSX sharedStrings.xml exceeds excel_shared_strings_limit; "
            "convert the workbook to CSV or Parquet, or raise the ceiling explicitly."
        )


def _workbook(source, excel_shared_strings_limit):
    _check_shared_strings(source, excel_shared_strings_limit)
    try:
        import openpyxl
    except ImportError as error:
        raise ImportError(
            "Excel ingest needs pip install 'pybiglasso[ingest]'"
        ) from error
    return openpyxl.load_workbook(
        source, read_only=True, data_only=True, keep_links=False
    )


def _row_block_size(block_size, columns):
    return min(block_size, max(1, _MAX_BLOCK_CELLS // len(columns)))


def _csv_block_size(source, block_size):
    with source.open("rb") as stream:
        longest_line = max((len(line) for line in stream), default=0)
    return max(block_size, longest_line + 1)


def _blocks(
    source, columns, all_names, block_size, sheet, pa, excel_shared_strings_limit
):
    """Yield selected columns; only a source batch or block is decoded at once."""
    suffix = source.suffix.lower()
    if suffix == ".csv":
        import pyarrow.csv as pc

        options = pc.ConvertOptions(
            column_types={name: pa.string() for name in all_names},
            include_columns=list(columns),
            strings_can_be_null=False,
        )
        csv_block_size = _csv_block_size(source, block_size)
        try:
            with pc.open_csv(
                source,
                read_options=pc.ReadOptions(block_size=csv_block_size),
                parse_options=pc.ParseOptions(ignore_empty_lines=False),
                convert_options=options,
            ) as reader:
                for batch in reader:
                    yield dict(
                        zip(
                            columns,
                            (column.to_pylist() for column in batch.columns),
                            strict=True,
                        )
                    )
        except pa.ArrowInvalid as error:
            raise ValueError(
                f"Cannot parse CSV with ingest block_size={csv_block_size} bytes: "
                f"{error}"
            ) from error
    elif suffix in (".parquet", ".pq"):
        import pyarrow.parquet as pq

        with pq.ParquetFile(source) as reader:
            for batch in reader.iter_batches(
                batch_size=_row_block_size(block_size, columns),
                columns=list(columns),
                use_threads=False,
            ):
                yield {name: batch.column(name).to_pylist() for name in columns}
    elif suffix in (".feather", ".arrow", ".ipc"):
        with pa.memory_map(str(source), "r") as mapping:
            reader = pa.ipc.open_file(mapping)
            for index in range(reader.num_record_batches):
                batch = reader.get_batch(index)
                row_block_size = _row_block_size(block_size, columns)
                for start in range(0, batch.num_rows, row_block_size):
                    yield {
                        name: batch.column(name)
                        .slice(start, row_block_size)
                        .to_pylist()
                        for name in columns
                    }
    else:
        workbook = _workbook(source, excel_shared_strings_limit)
        try:
            worksheet = workbook.worksheets[0] if sheet is None else workbook[sheet]
            rows = worksheet.iter_rows(values_only=True)
            next(rows, None)
            indices = [all_names.index(name) for name in columns]
            block = {name: [] for name in columns}
            row_block_size = _row_block_size(block_size, columns)
            for row in rows:
                for name, index in zip(columns, indices, strict=True):
                    block[name].append(row[index] if index < len(row) else None)
                if len(block[columns[0]]) == row_block_size:
                    yield block
                    block = {name: [] for name in columns}
            if block[columns[0]]:
                yield block
        finally:
            workbook.close()


def _fingerprint(source):
    stat = source.stat()
    return dict(path=str(source), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def _check_source(source, fingerprint):
    if _fingerprint(source) != fingerprint:
        raise ValueError("Source changed during ingest.")


def _infer(blocks, columns, overrides):
    candidates = {
        name: ([overrides[name]] if name in overrides else list(_TYPES))
        for name in columns
    }
    count = 0
    for block in blocks:
        length = len(block[columns[0]])
        for name in columns:
            for index, raw in enumerate(block[name], count + 1):
                value = _numeric(raw, name, index, overrides.get(name))
                # Counts/IDs beyond the supported integer widths use f64,
                # not f32, even when a particular integer happens to fit f32.
                wide_integer = isinstance(value, int) and not -32768 <= value <= 32767
                candidates[name] = [
                    dtype
                    for dtype in candidates[name]
                    if _exact(value, dtype)
                    and not (
                        name not in overrides and wide_integer and dtype == "float32"
                    )
                ]
                if not candidates[name]:
                    raise ValueError(
                        f"Column {name!r}, row {index}: no supported exact dtype."
                    )
        count += length
    if not count:
        raise ValueError("Source must contain at least one data row.")
    return {name: candidates[name][0] for name in columns}, count


def _stage(blocks, columns, types, directory, count):
    tails = {
        name: np.empty(0, dtype=np.bool_) for name in columns if types[name] == "bool"
    }
    paths = {
        name: directory / f"column-{index}.bin" for index, name in enumerate(columns)
    }
    seen = 0
    for block in blocks:
        for name in columns:
            values = [
                _numeric(raw, name, seen + index + 1, types[name])
                for index, raw in enumerate(block[name])
            ]
            for index, value in enumerate(values, seen + 1):
                if not _exact(value, types[name]):
                    raise ValueError(
                        f"Column {name!r}, row {index}: no supported exact dtype."
                    )
            array = np.asarray(values, dtype=types[name])
            if types[name] == "bool":
                bits = np.concatenate((tails[name], array))
                complete = len(bits) // 8 * 8
                array = np.packbits(bits[:complete], bitorder="little")
                tails[name] = bits[complete:].copy()
            with paths[name].open("ab") as stream:
                stream.write(array.tobytes())
        seen += len(block[columns[0]])
    if count is not None and seen != count:
        raise ValueError("Source row count changed during ingest.")
    for name, tail in tails.items():
        if len(tail):
            with paths[name].open("ab") as stream:
                stream.write(np.packbits(tail, bitorder="little").tobytes())
    return paths, seen


def _read_metadata(path):
    pa = _arrow()
    # Inspect only the schema, even for an ordinary compressed Feather file:
    # deciding whether this is an ingest output must never decode its data.
    try:
        with pa.memory_map(str(path), "r") as mapping:
            raw = (pa.ipc.open_file(mapping).schema.metadata or {}).get(_METADATA_KEY)
    except pa.ArrowInvalid as error:
        raise ValueError(
            "Cannot read ingest metadata. For a plain Feather file use "
            "pyarrow.feather.read_table(path, memory_map=True)."
        ) from error
    if raw is None:
        raise ValueError(
            "Feather file has no ingest metadata. For a plain Feather file use "
            "pyarrow.feather.read_table(path, memory_map=True)."
        )
    try:
        metadata = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("Invalid ingest metadata in the Feather schema.") from error
    if (
        not isinstance(metadata, dict)
        or not {"version", "rows", "features", "response", "response_sha256", "key"}
        <= metadata.keys()
    ):
        raise ValueError("Invalid ingest metadata: missing required fields.")
    if type(metadata["version"]) is not int or metadata["version"] != _FORMAT_VERSION:
        raise ValueError("Unsupported ingest metadata format version.")
    rows, features, response, response_sha256, key = (
        metadata[name]
        for name in ("rows", "features", "response", "response_sha256", "key")
    )
    if (
        type(rows) is not int
        or rows <= 0
        or not isinstance(features, list)
        or not features
        or any(not isinstance(name, str) or not name for name in features)
        or len(set(features)) != len(features)
        or (
            response is not None
            and (not isinstance(response, str) or not response or response in features)
        )
        or (response is None and response_sha256 is not None)
        or (
            response is not None
            and (
                not isinstance(response_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", response_sha256) is None
            )
        )
        or not isinstance(key, str)
        or re.fullmatch(r"[0-9a-f]{64}", key) is None
    ):
        raise ValueError("Invalid ingest metadata in the Feather schema.")
    return metadata


def _open_result(path, metadata):
    X = _open_mapped(path)
    if X.num_rows != metadata["rows"]:
        raise ValueError("Ingest metadata does not match the feature file's rows.")
    if tuple(X.column_names) != tuple(metadata["features"]):
        raise ValueError("Ingest metadata does not match its features or their order.")
    response = None
    if metadata["response"] is not None:
        response_path = path.with_suffix(path.suffix + ".response")
        try:
            with response_path.open("rb") as stream:
                size = os.fstat(stream.fileno()).st_size
                expected = metadata["rows"] * np.dtype(_RESPONSE_DTYPE).itemsize
                if size != expected:
                    raise ValueError(
                        f"Ingest response sidecar size does not match metadata: "
                        f"expected {expected} bytes, found {size}."
                    )
                if (
                    hashlib.file_digest(stream, "sha256").hexdigest()
                    != metadata["response_sha256"]
                ):
                    raise ValueError(
                        f"Ingest response file {response_path} does not match "
                        "the feature file it was written with (SHA256 mismatch)."
                    )
                stream.seek(0)
                response = np.memmap(
                    stream, mode="r", dtype=_RESPONSE_DTYPE, shape=(metadata["rows"],)
                )
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"Missing ingest response sidecar: {response_path}. "
                "Keep the Feather file and its .response file together."
            ) from error
    return IngestResult(X, response, tuple(X.column_names), path)


def load(path: str | os.PathLike[str]) -> IngestResult:
    """Reopen an ingest output without its original source.

    X is an ordinary memory-mapped pyarrow.Table. If the schema records a
    response, y is its read-only little-endian float64 memmap, otherwise None.
    Keep the Feather file and its optional .response file together. Their metadata,
    feature names/order, row count, response byte size and SHA256 are validated.
    Verifying the checksum reads all response bytes with bounded memory.
    Do not change either file while any mapping is in use.

    Plain Feather files without ingest metadata should instead be opened
    with pyarrow.feather.read_table(path, memory_map=True).
    """
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("load accepts a filesystem path to an ingest output.")
    path = Path(path).absolute()
    return _open_result(path, _read_metadata(path))


def _write_ipc(path, raw, selected, types, count, pa, metadata, schema=None):
    # Keep mmap-backed arrays local to this call: release every staging view
    # before TemporaryDirectory removes its files, including on Windows.
    arrays = []
    for name in selected:
        buffer = pa.memory_map(str(raw[name]), "r").read_buffer()
        arrays.append(
            pa.Array.from_buffers(
                pa.from_numpy_dtype(types[name]), count, [None, buffer]
            )
        )
    batch = (
        pa.record_batch(arrays, names=list(selected))
        if schema is None
        else pa.RecordBatch.from_arrays(arrays, schema=schema)
    )
    batch = batch.replace_schema_metadata(
        {
            **(batch.schema.metadata or {}),
            _METADATA_KEY: json.dumps(metadata, sort_keys=True).encode(),
        }
    )
    with (
        pa.OSFile(str(path), "wb") as stream,
        pa.ipc.new_file(
            stream, batch.schema, options=pa.ipc.IpcWriteOptions(compression=None)
        ) as writer,
    ):
        writer.write_batch(batch)


def _publish(
    staging, raw, selected, types, count, path, response, key, pa, schema=None
):
    response_path = path.with_suffix(path.suffix + ".response")
    output = staging / "data.feather"
    response_sha256 = None
    if response is not None:
        # Hash the staged bytes that will be hard-linked into place,
        # including their exact floating representations and signed zeros.
        with raw[response].open("rb") as stream:
            response_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    metadata = dict(
        version=_FORMAT_VERSION,
        key=key,
        rows=count,
        features=selected,
        response=response,
        response_sha256=response_sha256,
    )
    _write_ipc(output, raw, selected, types, count, pa, metadata, schema)
    # Hard links publish without overwriting an existing destination; all
    # staging files are on this filesystem. Roll back only our own links.
    published = []
    try:
        for origin, destination in [
            *(([(raw[response], response_path)]) if response is not None else []),
            (output, path),
        ]:
            os.link(origin, destination)
            published.append(destination)
    except OSError:
        for destination in published:
            destination.unlink()
        raise
    return metadata


def _stream_blocks(reader, columns, block_size, spool=None):
    schema = reader.schema
    row_block_size = _row_block_size(block_size, columns)
    for batch in reader:
        if not batch.schema.equals(schema, check_metadata=True):
            raise ValueError("Arrow stream schema changed during ingest.")
        # Empty batches carry a schema, not an end-of-stream indication.
        for start in range(0, batch.num_rows, row_block_size):
            selected = batch.select(list(columns)).slice(start, row_block_size)
            if spool is not None:
                spool.write_batch(selected)
            block = {name: selected.column(name).to_pylist() for name in columns}
            del selected
            yield block
            del block
        # Do not keep the preceding producer batch alive while requesting
        # its successor; even a small slice owns the full producer buffers.
        del batch


def _spooled_blocks(path, columns, block_size, pa):
    with (
        pa.memory_map(str(path), "r") as mapping,
        pa.ipc.open_stream(mapping) as reader,
    ):
        yield from _stream_blocks(reader, columns, block_size)


def _preserved_types(schema, selected, pa):
    types = {}
    for name in selected:
        dtype = schema.field(name).type
        if not (
            pa.types.is_boolean(dtype)
            or pa.types.is_integer(dtype)
            or dtype in (pa.float32(), pa.float64())
        ):
            raise ValueError(
                f"Column {name!r}: dtypes='preserve' needs bool, integer, "
                f"float32 or float64 storage, found {dtype}."
            )
        types[name] = np.dtype(dtype.to_pandas_dtype()).name
    return types


def _ingest_stream(
    reader,
    path,
    response,
    features,
    exclude,
    match,
    dtypes,
    block_size,
    pa,
    fingerprint=None,
    key=None,
):
    names = _names(reader.schema.names)
    selected = _selection(names, response, features, exclude, match)
    preserve = isinstance(dtypes, str) and dtypes == "preserve"
    overrides = _dtype_overrides(dtypes, selected)
    columns = (*selected, *((response,) if response is not None else ()))
    schema = (
        pa.schema(
            [reader.schema.field(name) for name in selected],
            metadata=reader.schema.metadata,
        )
        if preserve
        else None
    )
    types = _preserved_types(reader.schema, selected, pa) if preserve else None
    path.parent.mkdir(parents=True, exist_ok=True)
    response_path = path.with_suffix(path.suffix + ".response")
    if any(p.exists() for p in (path, response_path)):
        raise FileExistsError(f"Ingest output already exists: {path}")
    if key is None:
        key = hashlib.sha256(os.urandom(32)).hexdigest()
    with tempfile.TemporaryDirectory(prefix=".ingest-", dir=path.parent) as temporary:
        staging = Path(temporary)
        if preserve:
            if response is not None:
                types[response] = _RESPONSE_DTYPE
            raw, count = _stage(
                _stream_blocks(reader, columns, block_size),
                columns,
                types,
                staging,
                None,
            )
            if not count:
                raise ValueError("Source must contain at least one data row.")
        else:
            spool_path = staging / "source.arrow"
            spool_schema = pa.schema([reader.schema.field(name) for name in columns])
            # A file-format spool retains an index entry for every batch and
            # rejects replacement dictionaries. A sequential stream needs no
            # footer index and permits each fixed-schema batch's dictionary.
            with (
                pa.OSFile(str(spool_path), "wb") as stream,
                pa.ipc.new_stream(
                    stream,
                    spool_schema,
                    options=pa.ipc.IpcWriteOptions(compression=None),
                ) as spool,
            ):
                types, count = _infer(
                    _stream_blocks(reader, columns, block_size, spool),
                    columns,
                    {
                        **overrides,
                        **({response: _RESPONSE_DTYPE} if response is not None else {}),
                    },
                )
            raw, _ = _stage(
                _spooled_blocks(spool_path, columns, block_size, pa),
                columns,
                types,
                staging,
                count,
            )
        if fingerprint is not None:
            _check_source(*fingerprint)
        metadata = _publish(
            staging, raw, selected, types, count, path, response, key, pa, schema
        )
    return _open_result(path, metadata)


@contextmanager
def _preserve_reader(source, columns, block_size, pa):
    if source.suffix.lower() in (".parquet", ".pq"):
        import pyarrow.parquet as pq

        with pq.ParquetFile(source) as file:
            with pa.RecordBatchReader.from_batches(
                pa.schema(
                    [file.schema_arrow.field(name) for name in columns],
                    metadata=file.schema_arrow.metadata,
                ),
                file.iter_batches(
                    batch_size=_row_block_size(block_size, columns),
                    columns=list(columns),
                    use_threads=False,
                ),
            ) as reader:
                yield reader
    elif source.suffix.lower() in (".feather", ".arrow", ".ipc"):
        with pa.memory_map(str(source), "r") as mapping:
            file = pa.ipc.open_file(mapping)
            with pa.RecordBatchReader.from_batches(
                file.schema, (file.get_batch(i) for i in range(file.num_record_batches))
            ) as reader:
                yield reader
    else:
        raise ValueError(
            "dtypes='preserve' requires an Arrow stream, Feather or "
            "Parquet schema; use 'auto' or a mapping for CSV/Excel."
        )


def _dtype_overrides(dtypes, selected):
    if isinstance(dtypes, str):
        if dtypes not in ("auto", "preserve"):
            raise ValueError("dtypes must be 'auto', 'preserve' or a mapping.")
        overrides = {}
    elif isinstance(dtypes, Mapping):
        overrides = {name: np.dtype(dtype).name for name, dtype in dtypes.items()}
    else:
        raise ValueError("dtypes must be 'auto', 'preserve' or a mapping.")
    if set(overrides) - set(selected) or any(
        dtype not in _TYPES for dtype in overrides.values()
    ):
        raise ValueError(
            "dtype overrides must name selected features and supported types."
        )
    return overrides


def ingest(
    source,
    out=None,
    *,
    response=None,
    features=None,
    exclude=None,
    match=None,
    dtypes="auto",
    block_size=65536,
    sheet=None,
    excel_shared_strings_limit=1 << 30,
    cache=False,
    cache_dir=None,
):
    """Convert a numeric source to persistent, feature-only Arrow file storage.

    Paths and Arrow ``__arrow_c_stream__`` producers are accepted. Streams
    require an explicit ``out`` and do not support cache reuse. A stream is
    acquired once, never rewound; memory is its batch plus bounded consumer work.

    ``match`` is a glob string or compiled regex. ``dtypes='auto'`` infers exact
    types from every value; a partial mapping fixes selected physical types.
    Streams spool losslessly to disk for this second pass. ``'preserve'`` keeps
    supported Arrow field types and metadata in one pass (streams, Parquet and
    Feather); selected values are still checked for nulls and finiteness.
    Response selection is explicit and produces a separate read-only f64
    mapping. No response/family is inferred from names. Error row numbers are
    one-based data rows (excluding the header).

    ``block_size`` means bytes for CSV and rows for other readers. Compressed
    Feather decoding is bounded by its largest existing record batch. Excel
    is an exception to block-bounded memory: its complete workbook shared-string
    table may be held in memory, including unselected columns. Before each open,
    ``excel_shared_strings_limit`` caps its declared uncompressed XML size
    (1 GiB by default), without decompressing it. This is not an RSS ceiling.

    Specify an explicit persistent ``out`` or ``cache_dir``. There is no
    implicit temporary destination or automatic deletion. ``cache=True``
    reuses matching source path/size/mtime/options; otherwise a fresh artifact
    is written. Explicit existing outputs are never overwritten. Callers own
    the returned Feather output and its optional .response file. Use load()
    to reopen them later, even when the source has been removed.
    """
    is_path = isinstance(source, (str, os.PathLike))
    if not is_path:
        if out is None:
            raise ValueError("Arrow streams require an explicit persistent out.")
        if cache or cache_dir is not None:
            raise ValueError("Arrow streams do not support cache or cache_dir.")
        if not hasattr(source, "__arrow_c_stream__"):
            raise TypeError(
                "ingest accepts a filesystem path or an Arrow stream "
                "with __arrow_c_stream__."
            )
    elif out is None and cache_dir is None:
        raise ValueError("Specify a persistent out or cache_dir for ingest.")
    pa = _arrow()
    if (
        not isinstance(block_size, numbers.Integral)
        or isinstance(block_size, bool)
        or block_size <= 0
    ):
        raise ValueError("block_size must be a positive integer.")
    if (
        not isinstance(excel_shared_strings_limit, numbers.Integral)
        or isinstance(excel_shared_strings_limit, bool)
        or excel_shared_strings_limit <= 0
    ):
        raise ValueError("excel_shared_strings_limit must be a positive integer.")
    if not is_path:
        path = Path(out).absolute()
        if path.exists() or path.with_suffix(path.suffix + ".response").exists():
            raise FileExistsError(f"Ingest output already exists: {path}")
        with pa.RecordBatchReader.from_stream(source) as reader:
            return _ingest_stream(
                reader, path, response, features, exclude, match, dtypes, block_size, pa
            )
    source = Path(source).resolve(strict=True)
    names = _headers(source, pa, sheet, excel_shared_strings_limit)
    selected = _selection(names, response, features, exclude, match)
    overrides = _dtype_overrides(dtypes, selected)
    columns = (*selected, *((response,) if response is not None else ()))
    fingerprint = _fingerprint(source)
    options = dict(
        version=5,
        source=fingerprint,
        features=selected,
        response=response,
        dtypes="preserve" if dtypes == "preserve" else overrides,
        sheet=sheet,
        block_size=block_size,
        excel_shared_strings_limit=excel_shared_strings_limit,
    )
    key = hashlib.sha256(json.dumps(options, sort_keys=True).encode()).hexdigest()
    if out is None:
        root = Path(cache_dir).absolute()
        root.mkdir(parents=True, exist_ok=True)
        directory = (
            root / key if cache else Path(tempfile.mkdtemp(prefix="ingest-", dir=root))
        )
        directory.mkdir(exist_ok=True)
        path = directory / "data.feather"
    else:
        path = Path(out).absolute()
        path.parent.mkdir(parents=True, exist_ok=True)
    response_path = path.with_suffix(path.suffix + ".response")
    if cache and path.exists():
        metadata = _read_metadata(path)
        if metadata["key"] == key:
            return _open_result(path, metadata)
    if any(p.exists() for p in (path, response_path)):
        raise FileExistsError(f"Ingest output already exists: {path}")

    if isinstance(dtypes, str) and dtypes == "preserve":
        with _preserve_reader(source, columns, block_size, pa) as reader:
            return _ingest_stream(
                reader,
                path,
                response,
                selected,
                None,
                None,
                dtypes,
                block_size,
                pa,
                (source, fingerprint),
                key,
            )

    def blocks():
        return _blocks(
            source, columns, names, block_size, sheet, pa, excel_shared_strings_limit
        )

    # The response is separate solver input, always exactly representable f64.
    types, count = _infer(
        blocks(),
        columns,
        {**overrides, **({response: _RESPONSE_DTYPE} if response is not None else {})},
    )
    _check_source(source, fingerprint)
    with tempfile.TemporaryDirectory(prefix=".ingest-", dir=path.parent) as temporary:
        staging = Path(temporary)
        raw, _ = _stage(blocks(), columns, types, staging, count)
        _check_source(source, fingerprint)
        metadata = _publish(
            staging, raw, selected, types, count, path, response, key, pa
        )
    return _open_result(path, metadata)
