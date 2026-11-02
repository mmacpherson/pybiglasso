# Packed bool columns out of core

A binary design stored as Arrow bool columns, one bit per value, fits in
roughly the same time as the same values in a uint8 `.npy` when both are in
memory. Under a memory cap that holds the bits but not the bytes, it is 3.4x
faster on one thread and 80x faster on six: the uint8 file doesn't fit, so the
fit re-reads it from disk repeatedly, while the bit file is read about once,
consistent with its staying cached.

The design is 50,000 rows × 240,000 binary columns (carrier status: 0/1),
written once as a Fortran-order uint8 `.npy` (12.0 GB) and as an uncompressed
Arrow file with bool columns (1.53 GB) in one record batch. Each fit is the
binomial lasso path with 20 lambdas down to `lambda_min=0.1`. Times are medians
of three fits, in seconds:

| Memory | Threads | uint8 `.npy` | Arrow bool |
|---|---|---|---|
| 4 GiB cap, cold cache | 1 | 171.4 | 50.2 |
| 4 GiB cap, cold cache | 6 | 905.9 | 11.3 |
| uncapped, warm | 1 | 48.4 | 49.9 |
| uncapped, warm | 6 | 10.2 | 11.2 |

Under the cap, each uint8 fit read 226–391 GB from disk, 19–33 times the
file, and its memory cgroup peaked at the cap (4.29 GB, which is 4 GiB); each
bool fit read 1.52 GB, about the file once, and its cgroup peaked at 2.07 GB.
The uint8 times vary between repetitions (168–218 s on one thread, 839–1450 s
on six) with how much the fit re-reads; the bool times vary by under 3%. On six
threads the uint8 fits read less than on one but took five times as long. A
likely cause, not measured here, is contention between concurrent page faults
and readahead through a cache too small for the file. Warm, the bool fits took
3% longer than uint8 on one thread and 9% longer on six; the uint8 process held
11.8 GB resident against 2.06 GB for bool.

Every fit's complete path (lambdas, intercepts, coefficients and iteration
counts, compared as bytes) is identical to the first fit's, across both
storages, both thread counts and both memory regimes.

## How it ran

`benchmarks/bool_storage.py generate --parquet` wrote the design (seed 0) and
`run --mem 4G --repeats 3 --threads 1 --threads 6` fitted it, each fit in a
fresh process:

- **Capped, cold:** each file is evicted from the page cache first, then the
  fit runs in a `systemd-run --user` scope with `MemoryMax=4G` and no swap.
- **Warm:** uncapped, after one discarded priming fit per storage and thread
  count.
- **Gate:** before every fit, the CPU had to be at most 10% busy for 180
  seconds in a row (waits ranged from 180 to 830 seconds).

The files sat on btrfs, whose mount enables zstd compression, in a directory
with compression turned off; each file's attributes confirm it was written
without filesystem compression, and its allocated size matches its length to
the block. (Parquet still compresses its own pages with zstd.) The
[provenance](results/bool-storage-zen3-5600x-nocompress/provenance.json)
records this, with an AMD Ryzen 5 5600X, the CPU frequency governor
`performance`, Python 3.14.0, NumPy 2.5.4, PyArrow 26.0.0, and the SHA256s of
the extension and the script. The solver source is that of `a9519aa` (#80);
the recorded checkout is #81 plus this script and its test. Fits ran from
00:58 to 03:50 UTC on 2026-11-02. Raw records:
[results.jsonl](results/bool-storage-zen3-5600x-nocompress/results.jsonl).

## One pass over the data

`bool_storage.py scan` times one cold pass summing every column, each file
evicted from the page cache first, as a single uncapped run
([scan.json](results/bool-storage-zen3-5600x-nocompress/scan.json)). Each
storage is summed by a different library's reduction, so this is neither the
solver's pass nor pure read bandwidth; it gives the scale of one cold pass:

| Storage | Seconds per pass |
|---|---|
| Parquet, zstd (0.77 GB), decoded | 16.2 |
| uint8 `.npy`, memory-mapped | 12.0 |
| Arrow bool, memory-mapped | 1.5 |
