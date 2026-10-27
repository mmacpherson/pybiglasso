# Arrow IPC designs (draft, post-0.4.0)

This is the implementation plan for p4j.1, not a released API. The initial
interface is `ArrowDesign(path)`, passed as X to `biglasso`. It owns a read-only
PyArrow memory mapping of an uncompressed IPC file (Feather v2), with one
nonempty record batch. PyArrow is an optional `arrow` extra; importing
pybiglasso must not import it.

## Storage and lifetime

- Each column is bool, int8, uint8, int16, float32 or float64. Mixed designs
  retain each column's physical type; there is no common-dtype matrix.
- Bool stays bit-packed. Rust reads the indicated bit directly, returning the
  same 0.0 or 1.0 as a NumPy bool column. No unpacked row-length bool buffer.
- Primitive readers borrow read-only, aligned NumPy views of the mapped Arrow
  buffers. Creating a view changes metadata, not X's storage.
- Validate compression metadata before decoding a record batch, and verify
  every values buffer lies within the mapping. Do not silently decode a
  compressed Feather file into allocated columns.
- Reject nulls, unsupported types, empty designs and multiple record batches.
  In particular, nulls are rejected even outside a row subset. NaN/infinity
  retain the solver's existing selected-row policy for floating columns.
- The design keeps the mapping, batch and exported buffers alive. A fit holds
  its immutable buffer owners across GIL release and callbacks.

Requiring one batch keeps every logical column contiguous without changing
chunk boundaries. Writing or combining batches is a separate ingest operation
(p4j.2), not something a fit does. Arrow's default compressed Feather output
is not suitable; write uncompressed with a batch size covering the whole file.

## Solver boundary

Add a heterogeneous Rust `Design` whose per-column enum borrows primitive
slices or a bool bitmap. Choose the representation once per column; reuse the
existing `Col` kernels and fixed logical row/chunk order, not Arrow's buffer
boundaries. No new arithmetic, unsafe pointer access or numerical policy.

Reuse the Python binding's row selection, including the mask-safety fallback
to a gather. Both paths refer to the same mapped columns and never materialize
X or a row-subset matrix. Keep the existing ndarray path and direct-call
validation intact.

The initial public integration targets `biglasso` and bounded-block path
prediction. Do not let generic `np.asarray` or sklearn validation silently
materialize an Arrow design; unsupported consumers must fail explicitly until
they have a deliberate zero-copy path.

## Acceptance before leaving draft

1. Bit-identical complete paths against the same values in a Fortran .npy:
   every supported type, mixed columns, both families and penalties, supplied
   and automatic grids, bounds on/off, mask/gather/repeated rows, chunk/tail
   boundaries and multiple thread counts.
2. Bool offset/tail probes and malformed buffer/type/size/alignment checks.
3. Null, compression and multiple-batch rejection, without a fallback copy.
4. A native-memory cap starting before the target design is opened, with
   explicit primitive-copy and bool-expansion negative controls. A no-copy
   claim must cover both loading and the first fit, not only warm fits.
5. Existing fast/native/live-R suites and hooks unchanged in strength. No new
   R fixtures and no altered expectations or fitted tolerances.
6. Independent astra review; no merge until after the 0.4.0 release.

References: [Arrow IPC](https://arrow.apache.org/docs/python/ipc.html),
[columnar layout](https://arrow.apache.org/docs/format/Columnar.html), and
[IPC message schema](https://github.com/apache/arrow/blob/main/format/Message.fbs).
