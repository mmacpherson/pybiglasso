# Getting your data in

The design matrix X can be any of the types below. The first two are
memory-mapped from a file: their data stays on disk and is read a column at a
time, so it can be larger than memory.

| X | Made with | Its data during the fit |
|---|---|---|
| NumPy memmap of a `.npy` file | `np.load(path, mmap_mode="r")` | on disk; one type for the whole matrix |
| PyArrow Table mapped from a Feather file | `pyarrow.feather.read_table(path, memory_map=True)` | on disk if the file is uncompressed; a type per column, bools at one bit |
| NumPy array | — | in memory |
| pandas or Polars DataFrame | — | copied into one NumPy array, in memory; feature names come from its columns |

y is a separate one-dimensional array with one value per row, or a column of
the same DataFrame or Table.

The rest of this page gets data from where it usually lives into one of those,
and ends each recipe with a fit:

- **Ready to fit:** [NumPy arrays](#numpy-arrays),
  [pandas or Polars](#pandas-or-polars-dataframes), [a `.npy` file](#a-npy-file),
  [a Feather file](#a-feather-file)
- **Files to convert:** [a `.npz` archive](#a-npz-archive),
  [CSV, Parquet, compressed Feather or Excel](#csv-parquet-compressed-feather-or-excel)
- **Databases:** [transforming with DuckDB](#transforming-with-duckdb),
  [SQLite](#sqlite), [PostgreSQL](#postgresql)
- **Genotypes:** [PLINK `.bed`](#plink-bed), [VCF or BCF](#vcf-or-bcf)

A few things hold for every route:

- **X and y must line up row for row.** Take them from the same file or the
  same query, with a stable row order (`ORDER BY` in SQL).
- **No missing values.** Impute or filter first; if it matters, impute inside
  each cross-validation fold rather than once up front.
- **Encode text and categories yourself**, with the same categories and column
  order for training and prediction. One-hot encoding can multiply the width.
- **Don't standardize.** pybiglasso centers and scales each column itself.
- **Use the narrowest type that holds your values**: bool or int8 rather than
  float64 means less to read from disk on every pass.

Every snippet on this page is run verbatim by the test suite:
`make test-recipes` runs most of them, `make test-recipes-postgres` the
PostgreSQL one (CI runs both), and `make test-recipes-tools` the PLINK 2 one,
on a machine that has `plink2`.

## Ready to fit

### NumPy arrays

<!-- recipe: numpy -->
```python
import numpy as np
import pybiglasso as bl

X = np.load("X.npy")
y = np.load("y.npy")
fit = bl.biglasso(X, y)
```

### pandas or Polars DataFrames

The DataFrame is copied into a NumPy array, so it and the copy must both fit
in memory:

<!-- recipe: pandas -->
```python
import pandas as pd
import pybiglasso as bl

frame = pd.read_csv("data.csv")
fit = bl.biglasso(frame.drop(columns=["id", "y"]), frame["y"])
```

<!-- recipe: polars -->
```python
import polars as pl
import pybiglasso as bl

frame = pl.read_csv("data.csv")
fit = bl.biglasso(frame.drop("id", "y"), frame["y"])
```

### A `.npy` file

Open it memory-mapped, so columns are read from disk as the fit needs them.
The solver reads a column at a time, so store X in Fortran (column-major)
order. If your `.npy` is in C order, convert it once, a block of columns at a
time, so the whole matrix is never in memory:

<!-- recipe: npy -->
```python
import numpy as np
import pybiglasso as bl

source = np.load("X_c_order.npy", mmap_mode="r")
X = np.lib.format.open_memmap(
    "X.npy", mode="w+", dtype=source.dtype, shape=source.shape, fortran_order=True
)
for start in range(0, source.shape[1], 1000):
    X[:, start : start + 1000] = source[:, start : start + 1000]
X.flush()

X = np.load("X.npy", mmap_mode="r")
y = np.load("y.npy")
fit = bl.biglasso(X, y)
```

### A Feather file

Open it memory-mapped and keep the feature columns. It stays on disk only if
it's uncompressed; a compressed file is decompressed into memory when it's
opened. [Feather files and Arrow Tables](arrow.md) shows how to rewrite one uncompressed, or
convert it with `ingest` as [below](#csv-parquet-compressed-feather-or-excel).

<!-- recipe: feather -->
```python
import pyarrow.feather as pf
import pybiglasso as bl

table = pf.read_table("data.feather", memory_map=True)
fit = bl.biglasso(table.drop_columns(["id", "y"]), table["y"])
```

## Files to convert

### A `.npz` archive

`np.load` loads `.npz` members into memory and ignores `mmap_mode`. Extract X
to its own `.npy` first; `zipfile` streams it to disk, decompressing as it
goes:

<!-- recipe: npz -->
```python
import zipfile

import numpy as np
import pybiglasso as bl

with zipfile.ZipFile("data.npz") as archive:
    archive.extract("X.npy")
    y = np.load(archive.open("y.npy"))
X = np.load("X.npy", mmap_mode="r")
fit = bl.biglasso(X, y)
```

If X was saved in C order, convert it as in the [`.npy` recipe](#a-npy-file).

### CSV, Parquet, compressed Feather or Excel

Convert once with `ingest`. It reads the file in blocks, picks the narrowest
exact type for each column, and writes the features to an uncompressed Feather
file, wherever you name it. It returns X, memory-mapped from that file, and the
response you name as y. Next time, open the file you kept as
[above](#a-feather-file) instead of converting again. See [ingest](ingest.md)
for choosing columns and types.

<!-- recipe: ingest-csv -->
```python
import pybiglasso as bl
from pybiglasso.ingest import ingest

data = ingest("data.csv", "data.feather", response="y", exclude=["id"])
fit = bl.biglasso(data.X, data.y)
```

Two formats can hold more than a block in memory while converting: a
compressed Feather file is decompressed a whole record batch at a time (one
batch can be the whole file), and an Excel workbook's shared-strings table is
read whole.

## Databases

### Transforming with DuckDB

Filter, join, derive and encode with a query engine, write the result, then
ingest it. With DuckDB, reading and writing Parquet stream, but some steps
don't: `ORDER BY` sorts the whole result (spilling to disk if it must), as do
joins and grouping on their inputs.

<!-- recipe: duckdb-transform -->
```python
import duckdb
import pybiglasso as bl
from pybiglasso.ingest import ingest

duckdb.sql("""
    COPY (
        SELECT * EXCLUDE (id, site, keep),
               (site = 'b')::TINYINT AS site_b,
               (site = 'c')::TINYINT AS site_c
        FROM 'raw.parquet'
        WHERE keep
        ORDER BY id
    ) TO 'model.parquet' (FORMAT parquet)
""")
data = ingest("model.parquet", "data.feather", response="y")
fit = bl.biglasso(data.X, data.y)
```

The same query reads a directory of Parquet files with
`FROM read_parquet('raw/**/*.parquet')`.

### SQLite

Stream the query to CSV with the standard library, then ingest it:

<!-- recipe: sqlite -->
```python
import csv
import sqlite3

import pybiglasso as bl
from pybiglasso.ingest import ingest

with sqlite3.connect("study.db") as db, open("data.csv", "w", newline="") as out:
    rows = db.execute("SELECT * FROM observations ORDER BY id")
    writer = csv.writer(out)
    writer.writerow(column[0] for column in rows.description)
    writer.writerows(rows)
data = ingest("data.csv", "data.feather", response="y", exclude=["id"])
fit = bl.biglasso(data.X, data.y)
```

### PostgreSQL

Have the server write the query result as CSV, streamed to a local file with
[psycopg](https://www.psycopg.org/), then ingest it:

<!-- recipe: postgres -->
```python
import os

import psycopg
import pybiglasso as bl
from pybiglasso.ingest import ingest

query = "SELECT * FROM observations ORDER BY id"
with psycopg.connect(os.environ["DATABASE_URL"]) as db, open("data.csv", "wb") as out:
    with db.cursor().copy(f"COPY ({query}) TO STDOUT WITH (FORMAT csv, HEADER)") as copy:
        for block in copy:
            out.write(block)
data = ingest("data.csv", "data.feather", response="y", exclude=["id"])
fit = bl.biglasso(data.X, data.y)
```

asyncpg works the same way, with `copy_from_query`. DuckDB can also read
PostgreSQL directly through its `postgres` extension.

## Genotypes

### PLINK `.bed`

Read blocks of variants with [bed-reader](https://pypi.org/project/bed-reader/)
into an int8 `.npy`, one byte per genotype. bed-reader marks a missing call
with -127; decide what to do with those before fitting:

<!-- recipe: plink-bed -->
```python
import numpy as np
import pybiglasso as bl
from bed_reader import open_bed

with open_bed("genotypes.bed") as bed:
    X = np.lib.format.open_memmap(
        "genotypes.npy", mode="w+", dtype=np.int8, shape=bed.shape, fortran_order=True
    )
    for start in range(0, bed.sid_count, 10_000):
        block = bed.read(np.s_[:, start : start + 10_000], dtype="int8")
        if (block == -127).any():
            raise ValueError("missing genotype calls: impute or filter them first")
        X[:, start : start + 10_000] = block
X.flush()

X = np.load("genotypes.npy", mmap_mode="r")
y = np.load("phenotype.npy")
fit = bl.biglasso(X, y)
```

Dosages (0, 1, 2) are integers, not bools: keep them in int8.

### VCF or BCF

Convert to `.bed` with [PLINK 2](https://www.cog-genomics.org/plink/2.0/),
then follow the [`.bed` recipe](#plink-bed):

<!-- recipe: vcf-plink2 -->
```sh
plink2 --vcf genotypes.vcf --make-bed --out genotypes
```

This is for biallelic hard calls (GT). The same works for BCF (`--bcf`) and
for PLINK 2's own `.pgen` files (`--pfile`). PLINK 2 keeps missing calls
missing, which the `.bed` recipe then reports. A few choices are yours to make
first, explicitly:

- **Multiallelic sites** make `--make-bed` fail. Split them into biallelic
  records first (`bcftools norm -m -any`), or drop them (`--max-alleles 2`).
- **Imputed dosages** (DS or GP fields) aren't hard calls. Choose a hard-call
  threshold in PLINK 2, or keep the dosages as float32 in a `.npy`; for BGEN
  files, `bgen-reader` reads them.
