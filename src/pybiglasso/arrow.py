"""Private Arrow input adaptation and mapped-file preflight."""

import struct
import sys
from pathlib import Path

import numpy as np

__all__ = []


def _is_arrow(value, names=("Table", "RecordBatch")):
    # A caller with an Arrow object has already imported PyArrow. Keep it optional
    # and avoid importing it on NumPy-only package/fit/prediction paths.
    pa = sys.modules.get("pyarrow")
    return pa is not None and isinstance(value, tuple(getattr(pa, n) for n in names))


def _response(y):
    if _is_arrow(y, ("Array", "ChunkedArray")):
        if y.null_count:
            raise ValueError("y must not contain Arrow null values.")
        # Only the response may be materialized, never the feature matrix.
        return np.ascontiguousarray(y.to_numpy(zero_copy_only=False))
    return y


def _as_arrow_matrix(X):
    if isinstance(X, _ArrowMatrix):
        return X
    if _is_arrow(X):
        return _ArrowMatrix(X)
    if (
        type(X).__module__.split(".")[0] in ("pandas", "polars")
        or hasattr(X, "__arrow_c_stream__")
        or hasattr(X, "__arrow_c_array__")
    ):
        raise TypeError(
            "X must be a NumPy array, pyarrow.Table or pyarrow.RecordBatch. "
            "Convert explicitly to a pyarrow.Table, or use ingest on a file."
        )
    return X


def _field(metadata, table, index):
    """A checked FlatBuffers table field for IPC preflight."""

    def read(fmt, offset):
        if offset < 0 or offset + struct.calcsize(fmt) > len(metadata):
            raise ValueError("Invalid Arrow file metadata offsets.")
        return struct.unpack_from(fmt, metadata, offset)[0]

    vtable = table - read("<i", table)
    length = read("<H", vtable)
    entry = 4 + 2 * index
    if entry + 2 > length:
        return None
    offset = read("<H", vtable + entry)
    return table + offset if offset else None


def _check_uncompressed(source, pa):
    source.seek(0)
    if source.read(8)[:6] != b"ARROW1":
        raise ValueError("Mapped storage requires an Arrow file (Feather v2).")
    source.seek(source.size() - 10)
    trailer = source.read(10)
    if trailer[4:] != b"ARROW1":
        raise ValueError("Invalid Arrow file footer.")
    length = struct.unpack_from("<I", trailer)[0]
    if length > source.size() - 10:
        raise ValueError("Invalid Arrow file footer length.")
    source.seek(source.size() - 10 - length)
    footer = memoryview(source.read_buffer(length))
    root = struct.unpack_from("<I", footer)[0]
    schema = _field(footer, root, 1)
    if schema is None:
        raise ValueError("Missing Arrow file schema.")
    schema += struct.unpack_from("<I", footer, schema)[0]
    endian = _field(footer, schema, 0)
    endian = struct.unpack_from("<h", footer, endian)[0] if endian else 0
    if endian != (sys.byteorder == "big"):
        raise ValueError(
            "Mapped Arrow files must have native byte order; convert explicitly first."
        )
    # The footer, not the sequential message stream, determines what the reader
    # decodes. Check *all* indexed dictionary and record batches before any decode.
    for index in (2, 3):
        field = _field(footer, root, index)
        if field is None:
            continue
        vector = field + struct.unpack_from("<I", footer, field)[0]
        count = struct.unpack_from("<I", footer, vector)[0]
        if vector + 4 + 24 * count > len(footer):
            raise ValueError("Invalid Arrow file footer blocks.")
        for k in range(count):
            offset = struct.unpack_from("<q", footer, vector + 4 + 24 * k)[0]
            if offset < 8 or offset >= source.size() - 10 - length:
                raise ValueError("Invalid Arrow file footer batch offset.")
            source.seek(offset)
            message = pa.ipc.read_message(source)
            metadata = memoryview(message.metadata)
            message_root = struct.unpack_from("<I", metadata)[0]
            header = _field(metadata, message_root, 2)
            if header is None:
                raise ValueError("Missing Arrow batch metadata.")
            table = header + struct.unpack_from("<I", metadata, header)[0]
            if index == 2:
                if message.type != "dictionary":
                    raise ValueError("Expected an Arrow dictionary batch.")
                delta = _field(metadata, table, 2)
                if delta is not None and struct.unpack_from("<?", metadata, delta)[0]:
                    raise ValueError(
                        "Mapped storage cannot borrow dictionary deltas; use ingest to convert the file."
                    )
                data = _field(metadata, table, 1)
                if data is None:
                    raise ValueError("Missing Arrow dictionary data.")
                table = data + struct.unpack_from("<I", metadata, data)[0]
            elif message.type != "record batch":
                raise ValueError("Expected an Arrow record batch.")
            if _field(metadata, table, 3) is not None:
                raise ValueError(
                    "Mapped storage requires an uncompressed Arrow file; use ingest to convert the file."
                )


def _open_mapped(path):
    """Map an uncompressed Arrow file (Feather v2) as an ordinary pyarrow.Table.

    All footer-selected batches are preflighted before decoding. The Table's
    buffers retain the mapping; no sidecar or custom X wrapper is needed.
    Select/drop columns with Arrow before fitting. Do not modify the file while
    any Table, slice or fitted operation still refers to its buffers.
    """
    try:
        import pyarrow as pa
    except ImportError as error:
        raise ImportError(
            "Arrow support needs pip install 'pybiglasso[arrow]'"
        ) from error
    source = pa.memory_map(str(Path(path)), "r")
    try:
        _check_uncompressed(source, pa)
        reader = pa.ipc.open_file(source)
        source.seek(0)
        mapping = source.read_buffer(source.size())
        batches = []
        for k in range(reader.num_record_batches):
            batch = reader.get_batch(k)
            for array in batch.columns:
                for buffer in _buffers(array, pa):
                    if (
                        buffer is not None
                        and buffer.size
                        and not (
                            mapping.address <= buffer.address
                            and buffer.address + buffer.size
                            <= mapping.address + mapping.size
                        )
                    ):
                        raise ValueError(
                            "Arrow buffers must be borrowed from the mapping."
                        )
            batches.append(batch)
        return pa.Table.from_batches(batches, schema=reader.schema)
    except (pa.ArrowException, struct.error) as error:
        raise ValueError(f"Invalid Arrow file: {error}") from error
    finally:
        source.close()


def _buffers(array, pa):
    yield from array.buffers()
    if pa.types.is_dictionary(array.type):
        yield from _buffers(array.dictionary, pa)


def _chunk_descriptor(chunk, where, pa):
    try:
        if chunk.type == pa.bool_():
            dtype = None
        elif pa.types.is_integer(chunk.type) or chunk.type in (
            pa.float32(),
            pa.float64(),
        ):
            dtype = np.dtype(chunk.type.to_pandas_dtype())
        else:
            raise ValueError(
                f"{where}: unsupported Arrow type {chunk.type}; use primitive numeric columns."
            )
        if chunk.null_count:
            raise ValueError(f"{where}: Arrow nulls are not supported.")
        # Validate buffer layout metadata, without scanning feature values.
        chunk.validate(full=False)
        values = chunk.buffers()[1]
        if not len(chunk):
            view, offset = (
                np.empty(0, dtype=dtype or np.uint8),
                None if dtype else 0,
            )
        elif dtype is None:
            offset = chunk.offset % 8
            view = np.frombuffer(
                values,
                dtype=np.uint8,
                count=(offset + len(chunk) + 7) // 8,
                offset=chunk.offset // 8,
            )
        else:
            offset = None
            view = np.frombuffer(
                values,
                dtype=dtype,
                count=len(chunk),
                offset=chunk.offset * dtype.itemsize,
            )
        if not view.flags.aligned:
            raise ValueError(
                f"{where}: unaligned buffer; construct aligned Arrow buffers explicitly, or ingest a file."
            )
        view.flags.writeable = False
        return view, offset, len(chunk)
    except (pa.ArrowException, ValueError, TypeError, OverflowError) as error:
        if str(error).startswith(where + ":"):
            raise
        raise ValueError(f"{where}: invalid Arrow buffer: {error}") from error


class _ArrowMatrix:
    """Private borrowed chunk descriptors, not an accepted public data format."""

    def __init__(self, table):
        import pyarrow as pa

        self._table = table
        self.shape = (table.num_rows, table.num_columns)
        self.columns = tuple(table.column_names)
        if not all(self.shape):
            raise ValueError("Arrow X must have nonempty rows and columns.")
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("Arrow X requires unique column names.")
        self._columns = []
        for j, name in enumerate(self.columns):
            column = table.column(j)
            chunks = column.chunks if isinstance(column, pa.ChunkedArray) else [column]
            descriptors = []
            for k, chunk in enumerate(chunks):
                descriptors.append(
                    _chunk_descriptor(chunk, f"Column {name!r}, chunk {k}", pa)
                )
            if sum(n for _, _, n in descriptors) != self.shape[0]:
                raise ValueError(f"Column {name!r}: chunk lengths must match X rows.")
            self._columns.append(tuple(descriptors))
        self._columns = tuple(self._columns)

    def _block(self, rows, columns):
        indices = (
            np.arange(*rows.indices(self.shape[0]))
            if isinstance(rows, slice)
            else np.asarray(rows)
        )
        if indices.size == 0 and not isinstance(rows, np.ndarray):
            indices = indices.astype(np.intp)
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise IndexError("row_indices must be a one-dimensional integer array.")
        indices = indices.astype(np.intp, copy=False)
        nrow = self.shape[0]
        if np.any(indices < -nrow) or np.any(indices >= nrow):
            raise IndexError("row_indices has indices outside the design.")
        indices = np.where(indices < 0, indices + nrow, indices)
        block = np.empty((len(indices), len(columns)), dtype=np.float64)
        for k, j in enumerate(columns):
            start = 0
            for values, offset, n in self._columns[j]:
                selected = (indices >= start) & (indices < start + n)
                local = indices[selected] - start
                block[selected, k] = (
                    values[local]
                    if offset is None
                    else (values[(local + offset) // 8] >> ((local + offset) % 8)) & 1
                )
                start += n
        return block

    def _check_finite(self):
        # Preserve sklearn's prediction check without materializing columns.
        for column in self._columns:
            for values, offset, _ in column:
                if offset is None and values.dtype.kind == "f":
                    for start in range(0, len(values), 1 << 20):
                        if not np.isfinite(values[start : start + (1 << 20)]).all():
                            raise ValueError("X contains NaN or infinity.")
