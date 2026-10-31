# Arrow inputs

Open an uncompressed Arrow (Feather) file, choose the feature columns, and
keep the response separate:

```python
import pyarrow.feather as pf

table = pf.read_table("data.feather", memory_map=True)
features = ["x1", "x2"]
X = table.select(features)
y = table["y"]
```

Then fit as you would with a NumPy array:

```python
from pybiglasso import biglasso

fit = biglasso(X, y, family="gaussian")
prediction = fit.predict(X)
```

For an uncompressed file, the columns stay on disk and are read as the fit
needs them. **PyArrow's `write_feather` compresses by default. Compressed files
are decompressed into RAM when opened, even with `memory_map=True`.** If you
did not write the file uncompressed yourself,
[rewrite it uncompressed](#rewrite-an-arrow-file-uncompressed) once.

Choose features with `select` or `drop_columns` on the Table after opening.
**Avoid `read_table(..., columns=...)`: in PyArrow 26.0.0 this copies all the
file's uncompressed data, including unselected columns, into RAM.** Reading
without `columns=` and choosing columns afterward uses zero Arrow-pool bytes
on the tested uncompressed files. Future PyArrow versions may behave differently.

The NumPy counterpart is:

```python
import numpy as np

X = np.load("features.npy", mmap_mode="r")
y = np.load("response.npy", mmap_mode="r")
```

## Rewrite an Arrow file uncompressed

Write a new file one batch at a time with standard PyArrow APIs:

```python
import pyarrow as pa

with pa.memory_map("data.feather", "r") as source:
    reader = pa.ipc.open_file(source)
    with pa.OSFile("data-uncompressed.feather", "wb") as target:
        options = pa.ipc.IpcWriteOptions(compression=None)
        with pa.ipc.new_file(target, reader.schema, options=options) as writer:
            for index in range(reader.num_record_batches):
                writer.write_batch(reader.get_batch(index))
```

This reads and writes one batch at a time. For the supported numeric/bool
columns, memory use depends on the largest batch, plus the writer's and
compression codec's workspaces. A file with one very large batch can still
need substantial RAM. The result keeps the same batches and opens directly
with `pf.read_table("data-uncompressed.feather", memory_map=True)`.

When writing a new Feather file from a Table, pass `compression="uncompressed"`
to `pf.write_feather`. Ingest also converts supported source files and always
writes uncompressed feature files; its conversion-memory limits still apply.

## Accepted inputs

X can be a `pyarrow.Table` or `pyarrow.RecordBatch` with non-null bool,
int8/uint8/int16/uint16/int32/uint32/int64/uint64, float32 or float64 columns.
String, dictionary/categorical, nested, extension, decimal and temporal columns
are rejected. Buffers must be aligned. For mapped storage, use an uncompressed,
native-endian **Arrow file** (Feather v2), with one or multiple record batches.
PyArrow is optional (`pip install 'pybiglasso[arrow]'`). NumPy arrays and
Fortran `.npy` memmaps remain direct inputs; no conversion to Arrow is required.

**For Table/RecordBatch inputs, pybiglasso never copies X.** It borrows the chunk
buffers, whether they came from a mapping, an allocation or NumPy. This is a
statement about what pybiglasso does, not about the provenance of a Table or the
memory used by its producer. Bool remains bit-packed. The small response can
be materialized as a contiguous array using O(n) memory.

Table/RecordBatch feature names come from `column_names` and must be unique.
Choosing features, extracting the response and keeping rows aligned are the
caller's responsibility.
Pandas and Polars DataFrames keep the existing array-like path: feature names
come from `.columns`, and `np.asarray` converts the values before fitting.
Polars feature names in the estimators need scikit-learn 1.9 or newer.
Their Series work as responses too, including string labels for sklearn
classifiers. This conversion can allocate the full feature matrix; pass a
Table or RecordBatch to borrow Arrow buffers directly.

An Arrow `__arrow_c_stream__`/`__arrow_c_array__` exporter without `__array__`
(such as a RecordBatchReader) is not a matrix input and is rejected without
consuming it. Convert it explicitly to a Table, or use ingest on a file.
Exporting or building a Table upstream may itself allocate.

All classifier/regressor estimators, including CV, accept these Arrow X types.
Arrow `Array` and `ChunkedArray` responses use the existing target rules;
sklearn's binary string class labels retain their normal meaning. The functional
`biglasso` response is numeric. Null targets are rejected.

## Files split into batches

Files can contain several batches, and columns can be split into different
chunks. Fit the Table directly: there is no need to join the batches first.
The reader follows the original row order, so splitting the same values into
batches leaves the complete fitted path unchanged.

Columns can have different chunk boundaries, including empty chunks, one-row
chunks and sliced arrays. Primitive views start at `array.offset * itemsize` and
cover exactly the logical length; bool readers retain the logical bit offset.
Every column/chunk is checked, with contextual errors for unsupported types,
nulls, lengths or alignment. Unaligned buffers are rejected with guidance to
construct aligned buffers explicitly or ingest a file; there is no copying
fallback and no validation bypass.

Rust retains typed read guards for every chunk across GIL release, worker
threads and callbacks. Cumulative chunk ends give checked binary lookup, with a
single-chunk fast path. A block inside a chunk uses the existing reader; a
crossing assembles exactly the same fixed lanes as contiguous storage. The
logical work boundaries, accumulator lanes and reduction order never restart
at a storage boundary. Integers are read as float64 with the same precision
semantics as the ndarray solver; ingest's exact storage-conversion rules are
separate. There is no shared mutable cursor, whole-column
concatenation, `combine_chunks`, or row-sized chunk index.

`n_threads="auto"` conservatively uses one worker for **all** Tables and
RecordBatches. It does not guess their provenance or locality. Explicit positive
thread counts override this choice. No Arrow performance warnings or speedup
claims are made; representative benchmarks are separate work.

Arrow readers always use checked indexing, even under
`PYBIGLASSO_BOUNDSCHECK=False`. Tests under both settings change only the NumPy
reference's indexing mode. Prediction/scoring read bounded blocks of used
columns, normalize signed integer row selections and reject out-of-range rows
before primitive or bit addressing. No full X matrix is materialized.

NaN/infinity follow the existing selected-row fitting policy: excluded rows do
not trigger a new full-X finiteness scan, and unsafe mask arithmetic falls back
to gathering. sklearn prediction retains its existing finite-value check.

## Keep the data unchanged during a fit

`pf.read_table(path, memory_map=True)` reads an Arrow file (Feather v2) as an
ordinary Table, including files with multiple record batches. Select/drop
columns on that Table; the `columns=` reader behavior is described above.
No JSON sidecar is needed, including for feature files written by ingest.

`memory_map=True` does not promise that the reader allocates nothing: PyArrow
silently decompresses compressed files, and other storage conversions can
allocate too. For mapped storage, use native-endian uncompressed files; request
`compression="uncompressed"` when writing with `pf.write_feather`. File loading
and its allocations belong to PyArrow. Once a Table is passed to pybiglasso,
pybiglasso borrows its existing buffers without copying X. Multiple batches
need not be concatenated or rewritten to fit.

In-memory and mutable Arrow buffers are accepted through private read-only
views. As with ndarray inputs, **do not mutate the underlying buffers during a
fit, prediction or score**, including from another alias, thread or callback.
Do not overwrite or truncate a mapped file while any Table/slice refers to it.
Read-only views prevent writes through our readers, not writes from other owners.
Mappings and allocated buffers remain owned as long as any retained Arrow/NumPy
view uses them. No automatic file deletion or lifetime management is imposed.

## Verification

- Complete paths match Fortran `.npy` byte-for-byte while varying only chunk
  partitioning, including lane/work boundaries, independent columns, empty and
  one-row chunks, sliced primitives and non-byte-aligned bool slices. Both
  families/penalties, row masks/gathers/repeats/fallbacks, thread counts and
  bounds settings are covered.
- Direct descriptors, offset/tail checks and callback owner drops exercise the
  native borrowing boundary; mutable NumPy-backed Tables remain uncopied and
  their original writeability is unchanged.
- Native-memory caps cover mapped loading plus first fit, and separately fitting
  preallocated Tables. Primitive-copy, bool-expansion and native-allocation
  controls must fail at allocation, with thread-stack caching enabled/disabled.
- Existing native, fast, stored/live-R and hook gates retain their strength. No
  numerical expectations, fixtures or tolerances are generated from fits.

References: [Arrow IPC](https://arrow.apache.org/docs/python/ipc.html),
[columnar layout](https://arrow.apache.org/docs/format/Columnar.html),
[IPC messages](https://github.com/apache/arrow/blob/main/format/Message.fbs).
