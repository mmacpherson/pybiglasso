# Blockwise ingest

Install `pybiglasso[arrow]` for CSV, Parquet and Feather/Arrow IPC files;
`pybiglasso[ingest]` includes Arrow and the optional `.xlsx` reader.
Importing pybiglasso does not import either reader dependency.

```python
from pybiglasso import biglasso
from pybiglasso.ingest import ingest

data = ingest(
    "observations.parquet", "design.arrow",
    response="outcome", match="predictor_*",
)
X, y, names = data.design, data.response, data.feature_names  # X is a pyarrow.Table
fit = biglasso(X, y, family="gaussian")
```

The converter accepts filesystem paths only: CSV, Parquet, Feather/IPC files
and `.xlsx`. Tables, streams and file objects are not ingest sources. Existing
NumPy arrays/memmaps and compatible Tables/RecordBatches can be fitted directly;
ingest is an optional explicit materializer. See [Arrow inputs](arrow.md).

It writes **features only** to an
uncompressed Arrow IPC **file**, containing exactly one nonempty record batch.
The response is selected explicitly and stored separately as a read-only
float64 mapping; no target or family is inferred from column names. The receipt
is not X: pass `.design` (an ordinary Table) and `.response` explicitly.

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
overrides selected feature columns, with the same exactness requirement:

```python
data = ingest("data.csv", dtypes={"measurement": "float64"},
              cache=True, cache_dir="converted-data")
```

Unsupported exact types are not silently rounded: for example, an integer
`9007199254740993` is rejected. Missing, nonnumeric and nonfinite selected
values are rejected with the column name and one-based data-row number
(excluding the header). Unselected columns are not numerically validated.

## Bounded conversion

The first pass infers types and counts rows. The second stages typed columns
on disk. CSV, Parquet and Feather are read by block or source record batch,
with per-column state. Excel is an explicit exception to block-bounded memory.
The writer maps those raw columns read-only and writes one IPC batch; it does
not concatenate the full input into an in-memory table. Bool staging carries
at most seven bits between blocks, and writes packed bits, not byte-per-row
bool columns. Source size and modification time are checked between passes.

- CSV uses Arrow's incremental reader, with string columns to avoid first-block
  type inference rounding later integers. `block_size` is bytes for CSV.
- Parquet uses bounded `iter_batches`, respecting column selection.
- Feather/IPC reads one existing record batch at a time, then slices it into
  blocks. Compressed source batches must be decoded by Arrow; their original
  batch size therefore bounds the decoding allocation, not `block_size`.
- Excel uses openpyxl's read-only worksheet iterator. `sheet` selects a named
  worksheet; otherwise the first is used. Formula cells need stored values;
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
data = ingest("data.xlsx", "design.arrow", excel_shared_strings_limit=2 << 30)
```

Separately, the ZIP directory is capped at 1 MiB and 1,024 entries before
constructing `ZipFile`, so checking the shared-string size does not itself
allocate an unbounded directory. ZIP64 archive directories are unsupported
in v1. These directory limits are independent of the shared-string ceiling.

This ceiling is shared-string XML bytes, not total workbook memory or an RSS
ceiling; Python string objects and other workbook metadata add overhead.
Workbooks below the ceiling may still require substantial RAM. Prefer CSV or
Parquet when a strict blockwise-memory workflow is needed.

For non-CSV readers `block_size` counts rows. Output disk space must cover
both column staging and the final IPC file during conversion.

## Persistent outputs and reuse

`IngestResult` exposes `.design` (a standard `pyarrow.Table`), `.response`,
`.feature_names` and `.path`. The feature IPC file is ordinary uncompressed
Arrow storage: `pybiglasso.arrow.mmap_ipc(data.path)` or PyArrow's own file
reader reopens it without any JSON sidecar. The receipt itself is not accepted
as a fitting input. Mappings are read-only. Files persist independently of the Python result, so
retained designs and responses do not depend on a temporary-directory owner.

An explicit persistent `out` or `cache_dir` is required. With `out=None`,
output lives in a fresh subdirectory of the supplied `cache_dir`. `cache=True`
instead reuses an artifact keyed by resolved source path, byte size, nanosecond
mtime and conversion options. It does not hash all source contents. Explicit
output paths and their sidecars are never overwritten. The caller owns the IPC
file, optional `.response` raw-float64 sidecar and `.json` metadata sidecar;
remove them only after all mappings are no longer in use.
There is no implicit temporary destination or automatic artifact deletion.

Tests compare values and types against independent Arrow/openpyxl source
reads, check row errors, selection and caching, and on Linux put the first
large CSV conversion under an RLIMIT_DATA allowance smaller than the numeric
table. A whole-table reader is the negative control.
An independent valid XLSX fixture has 4,096 unique strings in an unselected
column: it ingests below the ceiling, while an explicitly smaller ceiling
rejects it before any workbook-reader or decompression call. Directory-size
probes check the default 1 GiB boundary without allocating a giant workbook.
