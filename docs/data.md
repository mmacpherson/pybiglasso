# Getting your data in

pybiglasso fits a NumPy array, a memory-mapped `.npy`, or a pyarrow Table. Each
recipe below starts from somewhere data really lives and ends with a fit, using
the tools you'd use anyway. Every snippet on this page is run verbatim by the
test suite (`make test-recipes`).

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

## NumPy arrays in memory

<!-- recipe: numpy -->
```python
import numpy as np
import pybiglasso as bl

X = np.load("X.npy")
y = np.load("y.npy")
fit = bl.biglasso(X, y)
```

## A `.npy` file larger than memory

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

## A `.npz` archive

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

If X was saved in C order, convert it as in the `.npy` recipe.

## A pandas DataFrame

pandas keeps the data in memory, which is fine when it fits. Feature names come
from the columns:

<!-- recipe: pandas -->
```python
import pandas as pd
import pybiglasso as bl

frame = pd.read_csv("data.csv")
fit = bl.biglasso(frame.drop(columns=["id", "y"]), frame["y"])
```

## An Arrow (Feather) file

Open it memory-mapped and keep the feature columns. This stays on disk only if
the file is uncompressed; see [Arrow inputs](arrow.md) for compressed files.

<!-- recipe: arrow -->
```python
import pyarrow.feather as pf
import pybiglasso as bl

table = pf.read_table("data.arrow", memory_map=True)
fit = bl.biglasso(table.drop_columns(["id", "y"]), table["y"])
```

## CSV, Parquet, compressed Feather or Excel

Convert once with `ingest`. It streams the file in blocks, picks the narrowest
exact type for each column, and writes an uncompressed Arrow file you can reuse
(see [ingest](ingest.md)):

<!-- recipe: ingest-csv -->
```python
import pybiglasso as bl
from pybiglasso.ingest import ingest

data = ingest("data.csv", "design.arrow", response="y", exclude=["id"])
fit = bl.biglasso(data.design, data.response)
```

## Data that needs transforming first

Filter, join, derive and encode with a query engine, write the result, then
ingest it. With DuckDB, reading Parquet and writing Parquet both stream:

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
data = ingest("model.parquet", "design.arrow", response="y")
fit = bl.biglasso(data.design, data.response)
```

The same query reads a directory of Parquet files with
`FROM read_parquet('raw/**/*.parquet')`.

## A SQLite database

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
data = ingest("data.csv", "design.arrow", response="y", exclude=["id"])
fit = bl.biglasso(data.design, data.response)
```

## Genotypes in PLINK `.bed`

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
