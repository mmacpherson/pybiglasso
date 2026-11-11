# Blockwise ingest

Install `pybiglasso[arrow]` for CSV, Parquet and Feather files;
`pybiglasso[ingest]` includes Arrow and the optional `.xlsx` reader.
Importing pybiglasso does not import either reader dependency.

```python
from pybiglasso import biglasso
from pybiglasso.ingest import ingest

data = ingest(
    "observations.parquet", "data.feather",
    response="outcome", match="predictor_*",
)
X, y, names = data.X, data.y, data.feature_names  # X is a pyarrow.Table
fit = biglasso(X, y, family="gaussian")
```

The converter accepts CSV, Parquet, Feather and `.xlsx` paths, and
[Arrow streams](#arrow-streams) from readers, Tables and compatible data tools.
Existing NumPy arrays/memmaps and compatible Tables/RecordBatches can be fitted
directly; ingest writes a persistent file when you need one. See [Feather files and Arrow Tables](arrow.md).

It writes **features only** to an
uncompressed **Feather file**, containing exactly one nonempty record batch.
The response is selected explicitly and stored separately as a read-only
float64 mapping; no target or family is inferred from column names. The receipt
is not X: pass `.X` (an ordinary Table) and `.y` explicitly.

With a response selected, the output is **two files that travel together**:
`data.feather` holds X, and `data.feather.response` holds y as little-endian
float64: rows × 8 bytes, with no header. Keep both when
copying or renaming the output. To reopen them next time, without the source:

```python
from pybiglasso import biglasso
from pybiglasso.ingest import load

data = load("data.feather")
fit = biglasso(data.X, data.y, family="gaussian")
```

`load` memory-maps X and y read-only. Without a selected response there is
only the Feather file, and `data.y` is `None`.
The solver's existing conversion of y to native float64 handles byte order
on big-endian platforms; the returned mapping keeps its little-endian dtype.

## Arrow streams

Pass a reader or another object with `__arrow_c_stream__` and choose where to
keep the result. For example, a producer can supply batches through a reader:

```python
import pyarrow as pa
from pybiglasso.ingest import ingest

schema = pa.schema([("x", pa.int64()), ("y", pa.float64())])
reader = pa.RecordBatchReader.from_batches(schema, [
    pa.record_batch([[1, 2], [0.5, 1.5]], schema=schema),
    pa.record_batch([[3], [2.5]], schema=schema),
])
data = ingest(reader, "stream.feather", response="y", dtypes="preserve")
```

The same call accepts a PyArrow Table, an eager Polars DataFrame, a DuckDB
relation's Arrow reader or an ADBC Arrow reader. It uses Arrow's standard
`RecordBatchReader.from_stream` interface, without adapters for each tool.
It acquires and consumes the stream once; empty batches do not end the stream.
Pass the Arrow reader exported by a cursor, rather than the database connection.
Pandas index columns materialized in Arrow are not selected implicitly. Choose
model columns with `features=` or `exclude=`, or call
`frame.reset_index(drop=True)` before exporting a stream. An index named
explicitly in `features=` is treated as an intentional feature. The same rule
applies to Parquet and Feather paths carrying pandas schema metadata.

**Stream producers must supply every batch using the declared schema.**
Ingest preserves and validates the schema exposed by the stream; it cannot
recover original types if a producer mislabels its buffers. Such malformed
streams can yield **incorrect values without an error**. Producers should
compare each batch's schema with the intended schema before exporting it.
After import, a matching schema or a successful Arrow `validate(full=True)`
check does not establish what the producer's original types were.

`dtypes="preserve"` keeps the selected fields' types, nullability and field
metadata, together with the schema metadata (ingest replaces its own reserved
`pybiglasso.ingest` key). Supported feature storage is bool, any 8/16/32/64-bit
integer, float32 or float64. Selected values must still be finite and non-null.
The response is always converted exactly to the separate little-endian float64
file. Wider integers can stay exact in X, but a response integer that cannot be
represented exactly as float64 is rejected. Preserve mode also works for
Parquet and Feather paths; CSV and Excel need `auto` or a dtype mapping.

The default `dtypes="auto"` and partial dtype mappings inspect **every selected
value**, including late batches. The stream is written losslessly to a temporary
uncompressed Arrow stream spool while types are inferred, then converted from
that spool. The spool is sequential: it keeps no growing batch index and accepts
replacement dictionaries with a fixed schema. There is no sampling or second
request to the producer. Preserve mode validates and stages in one pass, without
this spool. Both modes reject schema differences exposed by imported batches,
report value errors with cumulative one-based row numbers and publish only after
the whole stream succeeds. These checks do not recover types concealed by a
malformed producer.

Streams require an explicit `out`; `cache=True` and `cache_dir` are rejected
before acquisition. A stream has no stable source path to use as a cache key.
Reopen the finished output with `load`, as above. Conversion memory is the
**producer's batch plus bounded consumer work**: `block_size` limits the rows
converted at once, but cannot shrink a batch the producer already allocated.
For dictionary columns, this includes the producer's dictionary values; a small
batch may still refer to a large dictionary. Nothing concatenates all batches
into RAM. Disk space must also cover column staging, the final output and, in
auto mode, the lossless selected-column spool.

## Selection and exact representation

Names must be nonempty, unique strings. `features` chooses an ordered list;
otherwise source order is used. `exclude` removes names. `match` accepts a glob
string or a compiled regular expression. The response is always excluded from
features, even if explicitly listed. At least one feature must remain.

`dtypes="auto"` chooses the first exact representation in this order:
bool, int8, uint8, int16, float32, float64. These are the converter's candidate output types; direct Arrow inputs also accept wider integer
storage. In auto mode, parsed integers outside the int16
range use float64 rather than float32, provided conversion is exact. Every
integer with magnitude at most 2^53 fits float64 exactly; larger integers are
accepted only when they too are exactly representable. Integer spellings in
CSV are parsed without an intermediate float, including integral
decimal/exponent spellings. Fractional
CSV values use float64 parsing. Float32 is used only when conversion preserves
every parsed value; signed floating zero is retained. A partial dtype mapping
overrides selected feature columns. Integer and float32 overrides keep the
exactness requirement; an explicit float64 override accepts finite Arrow
decimal values by the usual float64 rounding:

```python
data = ingest("data.csv", dtypes={"measurement": "float64"},
              cache=True, cache_dir="converted-data")
```

Unsupported exact types are not silently rounded: for example, an integer
`9007199254740993` is rejected. Missing, nonnumeric and nonfinite selected
values are rejected with the column name and one-based data-row number
(excluding the header). Unselected columns are not numerically validated.
CSV numeric text follows ordinary decimal syntax: underscores, overflow and
nonzero values that underflow float64 are rejected. Boolean text may share a
column with integers (`true` is 1), and integer `-0` is boolean false.

## Bounded conversion

For paths in auto mode, the first pass infers types and counts rows. The second stages typed columns
on disk. CSV, Parquet and Feather are read by block or source record batch,
with per-column state. Excel is an explicit exception to block-bounded memory.
The writer maps those raw columns read-only and writes one Arrow record batch;
it does not concatenate the full input into an in-memory table. Bool staging carries
at most seven bits between blocks, and writes packed bits, not byte-per-row
bool columns. Source size and modification time are checked between passes.

- CSV uses Arrow's incremental reader, with string columns to avoid first-block
  type inference rounding later integers. `block_size` is bytes for CSV and is
  automatically raised when a header or physical record is longer, so wide
  records work at the default. Blank data lines are rejected; an ordinary
  trailing newline, CRLF and a UTF-8 BOM are accepted.
- Parquet uses bounded `iter_batches`, respecting column selection.
  Its decoder may retain column pages for a selected row group even when the
  yielded batch is smaller, so source decoding is bounded by that row group's
  selected, uncompressed data rather than by `block_size` alone.
- Feather files are read one existing record batch at a time, then sliced into
  blocks. Compressed source batches must be decoded by Arrow; their original
  batch size therefore bounds the decoding allocation, not `block_size`.
- Excel uses openpyxl's read-only worksheet iterator. `sheet` selects a named
  worksheet and an unknown name is an error; otherwise the first is used.
  Formula cells need stored values;
  external workbook links are not loaded. **The workbook's complete
  shared-string table may be held in memory, including strings in unselected
  columns or sheets.** `read_only=True` does not make this table block-bounded.
  Excel's per-sheet limits are 1,048,576 rows × 16,384 columns; these dimension
  limits are not a practical RAM budget for workbook-wide shared strings.

Before each workbook open, a preflight checks the declared uncompressed sizes
of all shared-string parts, including parts with nonstandard filenames. It
reads `[Content_Types].xml` within a 1 MiB limit to discover those parts by
content type, then checks their ZIP-directory sizes without decompressing the
shared strings. Malformed manifests, DTDs, missing declared parts and duplicate
ZIP part names are rejected.
`excel_shared_strings_limit` defaults to **1 GiB (1,073,741,824 bytes)**. Above
that ceiling, ingestion refuses with guidance to convert to CSV or Parquet.
The ceiling can be explicitly raised, accepting the associated memory cost:

```python
data = ingest("data.xlsx", "data.feather", excel_shared_strings_limit=2 << 30)
```

Separately, the ZIP directory is capped at 1 MiB and 1,024 entries before
constructing `ZipFile`, so checking the shared-string size does not itself
allocate an unbounded directory. ZIP64 archive directories are unsupported
in v1. These directory limits are independent of the shared-string ceiling.

This ceiling is shared-string XML bytes, not total workbook memory or an RSS
ceiling; Python string objects and other workbook metadata add overhead.
Workbooks below the ceiling may still require substantial RAM. Prefer CSV or
Parquet when a strict blockwise-memory workflow is needed.

For non-CSV readers `block_size` counts rows, but each consumer block is also
capped at 262,144 selected cells. Wide inputs therefore use fewer rows per
block instead of multiplying Python conversion objects by their column count.
Arrow-backed batches convert one column to Python values at a time. During the
final IPC write, the writer maps every staged feature column; those file-backed
pages are bounded by the logical output size, not by one consumer block. Peak
RSS is therefore bounded by the selected source row group or record batch, the
logical output size, the 262,144-cell consumer block and reader/writer metadata,
not by `block_size` alone. Output disk space must cover both column staging and
the final Feather file during conversion.

## Persistent outputs and reuse

`IngestResult` exposes `.X` (a standard `pyarrow.Table`), `.y`,
`.feature_names` and `.path`. The feature file is an ordinary uncompressed
Feather file. Its schema records the format version, rows, feature names and
order, response name (or null), response SHA256 (or null) and cache key under
`pybiglasso.ingest`.
`load(data.path)` validates that metadata against the feature file and checks
the response file's exact byte size before mapping y. It detects a missing
response file, one of the wrong size, or contents that do not match the recorded
checksum. Both `load` and cache reuse read all response bytes to verify SHA256,
using bounded memory; X remains memory-mapped.

Ordinary Arrow or Polars readers see the feature columns in their selected
order. PyArrow preserves the schema metadata; Polars drops it on `to_arrow`
or `write_ipc`. A file rewritten that way gets the plain-Feather error from
`load`, so reopen the original ingest output instead.
For a plain Feather file created elsewhere, use
`pyarrow.feather.read_table(path, memory_map=True)`; `load` requires ingest
metadata. The receipt itself is not accepted
as a fitting input. Mappings are read-only. Files persist independently of the Python result, so
retained X and y mappings do not depend on a temporary-directory owner.

For paths, an explicit persistent `out` or `cache_dir` is required. With `out=None`,
output lives in a fresh subdirectory of the supplied `cache_dir`. `cache=True`
instead reuses an artifact keyed by resolved source path, byte size, nanosecond
mtime and conversion options. It does not hash all source contents. Explicit
output paths and their response files are never overwritten. The caller owns
the Feather file and optional `.response` little-endian float64 file;
remove them only after all mappings are no longer in use.
There is no implicit temporary destination or automatic artifact deletion.
Publication links the response first and the metadata-bearing Feather file
last, so a visible completed Feather output has its companion ready.
Output storage must support hard links; this preserves atomic, no-overwrite
publication and is not available on filesystems such as exFAT. If conversion
fails, ingest removes a directory created by the call only while it remains
empty; an existing cache root, pre-existing output parents and concurrently
published contents are left untouched.

Tests compare values and types against independent Arrow/openpyxl source
reads, check row errors, selection and caching, and on Linux put the first
large CSV conversion under an RLIMIT_DATA allowance smaller than the numeric
table. A whole-table reader is the negative control.
An independent valid XLSX fixture has 4,096 unique strings in an unselected
column: it ingests below the ceiling, while an explicitly smaller ceiling
rejects it before any workbook-reader or decompression call. Directory-size
probes check the default 1 GiB boundary without allocating a giant workbook.
