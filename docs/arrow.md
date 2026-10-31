# Arrow inputs

X can be a `pyarrow.Table` or `pyarrow.RecordBatch` with non-null bool,
int8/uint8/int16/uint16/int32/uint32/int64/uint64, float32 or float64 columns.
String, dictionary/categorical, nested, extension, decimal and temporal columns
are rejected. Buffers must be aligned. For mapped storage, use an uncompressed,
native-endian Arrow IPC/Feather v2 **file**, with one or multiple record batches.
PyArrow is optional (`pip install 'pybiglasso[arrow]'`). NumPy arrays and
Fortran `.npy` memmaps remain direct inputs; no conversion to Arrow is required.

**pybiglasso never copies X.** For Arrow inputs it borrows the existing chunk
buffers, whether they came from a mapping, an allocation or NumPy. This is a
statement about what pybiglasso does, not about the provenance of a Table or the
memory used by its producer. Bool remains bit-packed. The small response can
be materialized as a contiguous array using O(n) memory.

```python
import numpy as np
import pyarrow as pa
from pybiglasso import biglasso
from pybiglasso.arrow import mmap_ipc

# Existing NumPy storage needs no Arrow step.
X_numpy = np.load("features.npy", mmap_mode="r")
y = np.load("response.npy", mmap_mode="r")
fit_numpy = biglasso(X_numpy, y, family="gaussian")

# Map an existing uncompressed IPC file into an ordinary Table.
table = mmap_ipc("data.feather")
y = table.column("response")
X = table.drop_columns(["response", "id"])
fit = biglasso(X, y, family="gaussian")
prediction = fit.predict(X)
```

`mmap_ipc` names a single storage operation: map an IPC file, returning a standard
Table. There is no public X wrapper or file registration. Its result can be
projected with Arrow's `select`/`drop_columns`, sliced, or handed to another
library. Feature names come from `column_names` and must be unique. Projection,
response extraction and row alignment are the caller's responsibility.
Generic `__arrow_c_stream__`/`__arrow_c_array__` exporters, pandas and Polars
frames are not fitting inputs: explicitly produce a Table first, or use `ingest`
on a source file. Exporting or building a Table upstream may itself allocate.

All classifier/regressor estimators, including CV, accept these Arrow X types.
Arrow `Array` and `ChunkedArray` responses use the existing target rules;
sklearn's binary string class labels retain their normal meaning. The functional
`biglasso` response is numeric. Null targets are rejected.

## Chunks and arithmetic

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

## Loading, mutability and lifetime

`mmap_ipc` checks compression on **every footer-selected dictionary/record batch
before decoding any batch**. It rejects compressed files with guidance to
convert explicitly using ingest. Dictionary delta batches are also rejected
before decoding because the reader would concatenate their dictionary values.
The ordinary Table retains its mapped buffers;
no JSON sidecar is needed to reopen the file. Standard Arrow readers/writers
can also consume it. PyArrow's default Feather writer often compresses; request
uncompressed IPC explicitly when writing mapped storage. Multiple batches need
not be concatenated or rewritten to fit.

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
