# Packed bool byte lookup

On this in-memory Gaussian path, packed bool takes 0.328 seconds versus
0.336 seconds for NumPy uint8. The same bool values took 0.720 seconds before
this change. Four chunks take 0.329 seconds; 64 chunks take 0.377 seconds.

| Input | Chunks | Before (s) | After (s) |
|---|---:|---:|---:|
| NumPy uint8 | — | 0.358 | 0.336 |
| Arrow uint8 | 1 | 0.428 | 0.361 |
| Arrow uint8 | 4 | 0.434 | 0.360 |
| Arrow uint8 | 64 | 0.495 | 0.423 |
| Arrow float64 | 1 | 0.568 | 0.534 |
| Arrow float64 | 4 | 0.571 | 0.541 |
| Arrow float64 | 64 | 0.633 | 0.603 |
| Arrow bool | 1 | 0.720 | 0.328 |
| Arrow bool | 4 | 0.718 | 0.329 |
| Arrow bool | 64 | 0.772 | 0.377 |

The [raw paired measurements](results/bool-byte-lut.jsonl) retain all six
times per build, extension hashes, native source hashes and complete-path
hashes. Before is `47f2c6d` (#75); its extension matches the candidate hash
in [that change's measurements](results/arrow-chunks.jsonl). Every path hash
agrees across both builds, all types, all partitions and the NumPy control.
Hashes check invariance; they are not expected solver values.

The fixture and [script](arrow_chunks.py) are unchanged from the
[chunk-run benchmark](arrow-chunks.md): seed 0, 262,144 rows, 300
Bernoulli(0.3) features, response equal to the first five features' sum plus
standard normal noise, 20 lambdas, `lambda_min=0.1`, one thread. Both release
extensions run in one process on shared inputs, warmed once, with six
repetitions alternating AB/BA order. The table reports medians. No compilers
or other benchmarks ran during these measurements.

Environment: AMD Ryzen 5 7640U, one logical CPU pinned, performance governor,
OpenBLAS/OMP limited to one thread; Python 3.15.0, NumPy 2.5.4, PyArrow 26.0.0,
rustc 1.98.0. Both measured extensions retain symbols for profiling. The
committed stripping setting is unchanged. The tested stripped build has
identical `.text` and `.rodata` sections to the measured candidate.

## What changed

An aligned byte indexes a static 16 KiB table of eight exact positive
0.0/1.0 values. Unaligned blocks retain their existing decoder and the
same overflow-safe bounds guard. Scalar reads use the same table; run
views expose complete aligned bytes to the block scans without copying X.

Centering visits decoded lanes in row order. Unmasked Gaussian residual
updates do the same, carrying the running sum into each block and back out;
there are no block subtotals. Keeping this sum local also avoids a compiler
spill after each lane. The uint8 and float64 controls benefit from those
blocked scalar scans too. Masked reads retain their guarded scalar path.

Profiling the intermediate build identified the Gaussian visitor's repeated
sum stores and loads. Keeping the running sum local roughly halved that
visitor's cost. The dot loop already used direct byte-to-table SIMD loads.
No products are skipped, including multiplication by zero, and the SIMD
accumulators, reduction tree and chunk-crossing blocks keep their order.

These are in-memory results. They do not establish performance under a
memory cap or cold storage; that requires a separate measurement.
