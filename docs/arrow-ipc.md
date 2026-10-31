# Arrow IPC designs

`ArrowDesign(path)` can be passed as X to `biglasso`, the classifier and
regressor estimators (including CV), and path prediction/scoring. It owns a read-only
PyArrow memory mapping of an uncompressed IPC file (Feather v2), with one
nonempty record batch. PyArrow is an optional `arrow` extra; importing
pybiglasso does not import PyArrow until a design is opened.

```python
from pybiglasso import biglasso
from pybiglasso.arrow import ArrowDesign

X = ArrowDesign("features.feather")
fit = biglasso(X, y, family="gaussian")
prediction = fit.predict(X)
```

The file contains features only; supply the response separately. Feature names
must be unique. `n_threads="auto"` treats mapped Arrow files like NumPy memmaps
and uses one worker; an explicit positive worker count overrides this policy.

## Storage and lifetime

- Each column is bool, int8, uint8, int16, float32 or float64. Mixed designs
  retain each column's physical type; there is no common-dtype matrix.
- Bool stays bit-packed. Rust reads the indicated bit directly, returning the
  same 0.0 or 1.0 as a NumPy bool column. No unpacked row-length bool buffer.
- Primitive readers borrow read-only, aligned NumPy views of the mapped Arrow
  buffers. Creating a view changes metadata, not X's storage.
- The reader validates compression metadata before decoding a record batch, and verifies
  every values buffer lies within the mapping. Do not silently decode a
  compressed Feather file into allocated columns.
  The footer must select the batch message that passed preflight; files with
  an unindexed batch before the selected batch are rejected before decoding.
- It rejects nulls, unsupported types, empty designs and multiple record batches.
  In particular, nulls are rejected even outside a row subset. NaN/infinity
  retain the solver's existing selected-row policy for floating columns.
- The design keeps the mapping, batch and exported buffers alive. A fit holds
  its immutable buffer owners across GIL release and callbacks.

Do not overwrite, truncate or modify the underlying file while any design or
view refers to it. A read-only mapping protects against writes through this
API, not changes by another process. No temporary-file deletion or lifetime
management is implicit in opening a design.

Requiring one batch keeps every logical column contiguous without changing
chunk boundaries. Writing or combining batches is a separate ingest operation
(p4j.2), not something a fit does. Arrow's default compressed Feather output
is not suitable; write uncompressed with a batch size covering the whole file.

## Solver and prediction

The heterogeneous Rust `Design` has a per-column enum borrowing primitive
slices or a bool bitmap. Choose the representation once per column; reuse the
existing `Col` kernels and fixed logical row/chunk order, not Arrow's buffer
boundaries. No new arithmetic, unsafe pointer access or numerical policy.

Arrow designs always use checked indexing, including primitive blocks and
packed-bit reads. `PYBIGLASSO_BOUNDSCHECK=False` does not disable these checks
for Arrow inputs.

It reuses the Python binding's row selection, including the mask-safety fallback
to a gather. Both paths refer to the same mapped columns and never materialize
X or a row-subset matrix. Keep the existing ndarray path and direct-call
validation intact.

Prediction converts only bounded blocks of the columns used by the selected
fits, as on NumPy inputs; it does not expand the whole bool design. sklearn
validates design metadata without `check_array` on X, and validates the response
separately. Its prediction also checks floating values in bounded blocks.
Generic `np.asarray(X)` raises instead of silently materializing a matrix.
Prediction accepts integer row lists or arrays, resolves negative indices from
the end of the logical design and rejects out-of-range rows before accessing
primitive values or packed bits. Bitmap padding is never a logical row.

## Checks

1. Bit-identical complete paths against the same values in a Fortran .npy:
   every supported type, mixed columns, both families and penalties, supplied
   and automatic grids, mask/gather/repeated rows, chunk/tail boundaries and
   multiple thread counts. The comparison runs under both bounds settings;
   only the NumPy reference changes indexing mode, while Arrow stays checked.
2. Bool offset/tail probes and malformed buffer/type/size/alignment checks.
3. Null, compression and multiple-batch rejection, without a fallback copy.
4. A native-memory cap starting before the target design is opened, with
   explicit primitive-copy and bool-expansion negative controls. A no-copy
   claim must cover both loading and the first fit, not only warm fits.
5. Existing fast/native/live-R suites and hooks unchanged in strength. No new
   R fixtures and no altered expectations or fitted tolerances.
6. Independent review and CI's cross-platform bit check, including a mixed
   six-type Arrow design on Linux x86/ARM, macOS ARM and Windows x86.

References: [Arrow IPC](https://arrow.apache.org/docs/python/ipc.html),
[columnar layout](https://arrow.apache.org/docs/format/Columnar.html), and
[IPC message schema](https://github.com/apache/arrow/blob/main/format/Message.fbs).
