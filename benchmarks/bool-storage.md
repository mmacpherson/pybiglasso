# Packed bool columns out of core

A binary design stored as Arrow bool columns, one bit per value, fits as fast
as the same values in a uint8 `.npy` when both are in memory. Under a memory
cap that holds the bits but not the bytes, it is 3.7x faster on one thread and
8.1x faster on six: the uint8 file doesn't fit, so the fit re-reads it from
disk several times over, while the bit file is read once and stays resident.

The design is 50,000 rows × 240,000 binary columns (carrier status: 0/1),
written once as a Fortran-order uint8 `.npy` (11.4 GB) and as an uncompressed
Arrow file with bool columns (1.46 GB) in one record batch. Each fit is the
binomial lasso path with 20 lambdas down to `lambda_min=0.1`. Times are medians
of three fits, in seconds:

| Memory | Threads | uint8 `.npy` | Arrow bool |
|---|---|---|---|
| 4 GB cap, cold cache | 1 | 185.2 | 50.4 |
| 4 GB cap, cold cache | 6 | 89.6 | 11.1 |
| uncapped, warm | 1 | 49.2 | 50.2 |
| uncapped, warm | 6 | 10.2 | 11.3 |

Under the cap, each uint8 fit read 30–45 GB from disk and ran at the cap's
4.29 GB peak; each bool fit read 0.78 GB and peaked at 2.07 GB. The uint8
times vary more between repetitions (157–191 s on one thread, 73–119 s on six)
because they depend on what the kernel evicts; the bool times vary by under
8%. Warm, the uint8 fits held 11.9 GB resident against 2.1 GB for bool.

The run also fitted a second bool file, written with `--batch-rows 65536`
(`arrow-bool-65536` in the records). At 50,000 rows that is still a single
batch, so it repeats the one-batch measurement (within 1%) and says nothing
about batch size; multi-batch reads were measured in memory in
[arrow-chunks.md](arrow-chunks.md).

Every fit's complete path (lambdas, intercepts, coefficients and iteration
counts) is bit-identical to the first fit's, across both storages, both thread
counts and both memory regimes.

## How it ran

`benchmarks/bool_storage.py generate` wrote the design (seed 0) and
`run --mem 4G --repeats 3 --threads 1 --threads 6 --batch-rows 0 --batch-rows
65536` fitted it, each fit in a fresh process:

- **Capped, cold:** each file is evicted from the page cache first, then the
  fit runs in a `systemd-run --user` scope with `MemoryMax=4G` and no swap.
- **Warm:** uncapped, after one discarded priming fit per storage.
- **Gate:** before every fit, the CPU had to be at most 10% busy for 180
  seconds in a row (waits ranged from 180 to 510 seconds).

The CPU frequency governor was `performance` for the run. The
[provenance](results/bool-storage-zen3-5600x/provenance.json) records an AMD
Ryzen 5 5600X, Python 3.14.0, NumPy 2.5.4, PyArrow 26.0.0, and the SHA256s of
the extension and the script. The measured solver is `a9519aa` (#80); the
recorded source commit added only this script on top of it, and the script's
hash matches the one committed here. Fits ran from 18:00 to 20:44 UTC on
2026-11-01. Raw records: [results.jsonl](results/bool-storage-zen3-5600x/results.jsonl).

## One pass over the data

For a solver that read Parquet directly, every pass over X would mean decoding
it. `bool_storage.py scan` times one cold pass over every column, each file
evicted from the page cache first, as a single uncapped run
([scan.json](results/bool-storage-zen3-5600x/scan.json)):

| Storage | Seconds per pass |
|---|---|
| Parquet, zstd (0.73 GB) | 19.1 |
| uint8 `.npy`, memory-mapped | 11.8 |
| Arrow bool, memory-mapped | 3.9 |

A path fit makes many passes, so converting once (with `ingest`) and
memory-mapping the result is the faster route than decoding Parquet each time.
