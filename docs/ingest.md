# Blockwise ingest (draft, post-0.4.0)

Install `pybiglasso[arrow]` for CSV, Parquet and Feather/Arrow IPC files;
`pybiglasso[ingest]` includes Arrow and the optional `.xlsx` reader.
Importing pybiglasso does not import either reader dependency.

```python
from pybiglasso.ingest import ingest

data = ingest(
    "observations.parquet", "design.arrow",
    response="outcome", match="predictor_*",
)
X, y, names = data.design, data.response, data.feature_names
```

This companion to the draft Arrow loader writes **features only** to an
uncompressed Arrow IPC **file**, containing exactly one nonempty record batch.
The response is selected explicitly and stored separately as a read-only
float64 mapping; no target or family is inferred from column names. The loader
and solver integration are developed in the companion Arrow IPC change.

## Selection and exact representation

Names must be nonempty, unique strings. `features` chooses an ordered list;
otherwise source order is used. `exclude` removes names. `match` accepts a glob
string or a compiled regular expression. The response is always excluded from
features, even if explicitly listed. At least one feature must remain.

`dtypes="auto"` chooses the first exact representation in this order:
bool, int8, uint8, int16, float32, float64. These are the physical types accepted
by the companion loader. In auto mode, parsed integers outside the int16
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
on disk, retaining at most one decoded block and per-column state in memory.
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
  worksheet; otherwise the first is used. Formula cells need stored values.

For non-CSV readers `block_size` counts rows. Output disk space must cover
both column staging and the final IPC file during conversion.

## Persistent outputs and reuse

`IngestResult` exposes `.design`, `.response`, `.feature_names` and `.path`.
Mappings are read-only. Files persist independently of the Python result, so
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
