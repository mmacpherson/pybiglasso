"""Run each recipe in docs/data.md verbatim, and check its fit.

Every snippet runs in a fresh process, in a directory holding the input files
it expects, and must leave a `fit`. Every array in that fit must be byte for
byte the one from fitting the same values from NumPy directly, and its feature
names the columns the recipe kept: an invariance across storage and loading
routes, not an expected value.

Values are multiples of 1/8, so every CSV writer and parser round-trips them.
"""

import ast
import csv
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import pybiglasso as bl

pytestmark = pytest.mark.recipes

DOCS = Path(__file__).parents[1] / "docs" / "data.md"
NROW, NCOL = 150, 10


def recipe(name, language="python"):
    text = DOCS.read_text()
    match = re.search(
        rf"<!-- recipe: {re.escape(name)} -->\n```{language}\n(.*?)```", text, re.S
    )
    assert match, f"no recipe {name!r} in {DOCS.name}"
    return match.group(1)


def dyadic(rng, shape, scale=8):
    return rng.integers(-40, 41, size=shape) / scale


def design(seed=0):
    rng = np.random.default_rng(seed)
    X = dyadic(rng, (NROW, NCOL))
    beta = np.zeros(NCOL)
    beta[:3] = [1.5, -1.0, 0.75]
    y = np.round((X @ beta + rng.normal(size=NROW)) * 64) / 64
    return X, y


def names(ncol=NCOL):
    return [f"x{j}" for j in range(ncol)]


def write_csv(path, columns):
    with open(path, "w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values(), strict=True))


def table_columns(X, y):
    columns = {"id": np.arange(NROW)}
    columns.update(zip(names(X.shape[1]), X.T, strict=True))
    columns["y"] = y
    return columns


def run_recipe(name, workdir):
    """Run the snippet as written, then save what its `fit` holds."""
    dump = """
import numpy as _np
_np.savez(
    "fit.npz",
    **{f: getattr(fit, f) for f in _FIELDS},
    **{"coef_" + f: getattr(fit.coef, f) for f in ("data", "indices", "indptr")},
    feature_names=_np.asarray(
        [] if fit.feature_names is None else list(fit.feature_names), dtype=str
    ),
)
""".replace("_FIELDS", repr(FIELDS))
    subprocess.run(
        [sys.executable, "-c", recipe(name) + dump],
        cwd=workdir,
        check=True,
        capture_output=True,
        text=True,
    )
    return np.load(workdir / "fit.npz")


FIELDS = (
    "lambdas", "intercept", "iterations", "n_rejected", "center", "scale", "deviance",
)  # fmt: skip


def assert_same_fit(result, X, y, names=None):
    """Byte-for-byte (so -0.0 differs from 0.0) against the NumPy fit."""
    reference = bl.biglasso(np.asfortranarray(X), y)
    for field in FIELDS:
        expected = getattr(reference, field)
        assert result[field].dtype == expected.dtype, field
        assert result[field].tobytes() == expected.tobytes(), field
    for field in ("data", "indices", "indptr"):
        expected = getattr(reference.coef, field)
        assert result["coef_" + field].tobytes() == expected.tobytes(), field
    if names is not None:
        assert result["feature_names"].tolist() == names


def test_numpy(tmp_path):
    X, y = design()
    np.save(tmp_path / "X.npy", X)
    np.save(tmp_path / "y.npy", y)
    assert_same_fit(run_recipe("numpy", tmp_path), X, y)


def test_npy_converts_c_order(tmp_path):
    X, y = design()
    np.save(tmp_path / "X_c_order.npy", np.ascontiguousarray(X))
    np.save(tmp_path / "y.npy", y)
    assert_same_fit(run_recipe("npy", tmp_path), X, y)
    assert np.load(tmp_path / "X.npy", mmap_mode="r").flags.f_contiguous


def test_npz(tmp_path):
    X, y = design()
    np.savez_compressed(tmp_path / "data.npz", X=np.asfortranarray(X), y=y)
    assert_same_fit(run_recipe("npz", tmp_path), X, y)


def test_pandas(tmp_path):
    X, y = design()
    write_csv(tmp_path / "data.csv", table_columns(X, y))
    assert_same_fit(run_recipe("pandas", tmp_path), X, y, names())


def test_arrow_file(tmp_path):
    import pyarrow as pa
    import pyarrow.feather as pf

    X, y = design()
    pf.write_feather(
        pa.table(table_columns(X, y)),
        tmp_path / "data.arrow",
        compression="uncompressed",
    )
    assert_same_fit(run_recipe("arrow", tmp_path), X, y, names())


def test_ingest_csv(tmp_path):
    X, y = design()
    write_csv(tmp_path / "data.csv", table_columns(X, y))
    assert_same_fit(run_recipe("ingest-csv", tmp_path), X, y, names())


def test_duckdb_transform(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(1)
    X, y = design()
    site = rng.choice(["a", "b", "c"], size=NROW)
    keep = rng.random(NROW) < 0.8
    order = rng.permutation(NROW)  # rows stored out of id order
    raw = table_columns(X, y) | {"site": site, "keep": keep}
    pq.write_table(
        pa.table({k: np.asarray(v)[order] for k, v in raw.items()}),
        tmp_path / "raw.parquet",
    )
    # the same transform, by hand: keep rows in id order, x columns, then
    # one-hot site with 'a' as the baseline
    expected = np.column_stack([X, site == "b", site == "c"])[keep].astype(np.float64)
    assert_same_fit(
        run_recipe("duckdb-transform", tmp_path),
        expected,
        y[keep],
        names() + ["site_b", "site_c"],
    )


def test_sqlite(tmp_path):
    X, y = design()
    columns = table_columns(X, y)
    with sqlite3.connect(tmp_path / "study.db") as db:
        db.execute(f"CREATE TABLE observations ({', '.join(columns)})")
        rows = list(
            zip(*(np.asarray(v).tolist() for v in columns.values()), strict=True)
        )
        # inserted out of order: the recipe's ORDER BY restores it
        db.executemany(
            f"INSERT INTO observations VALUES ({', '.join('?' * len(columns))})",
            rows[::-1],
        )
    assert_same_fit(run_recipe("sqlite", tmp_path), X, y, names())


def test_plink_bed(tmp_path):
    from bed_reader import to_bed

    rng = np.random.default_rng(2)
    G = rng.integers(0, 3, size=(NROW, 25)).astype(np.int8)
    _, y = design()
    to_bed(tmp_path / "genotypes.bed", G.astype(np.float32))
    np.save(tmp_path / "phenotype.npy", y)
    assert_same_fit(run_recipe("plink-bed", tmp_path), G, y)


def test_plink_bed_rejects_missing_calls(tmp_path):
    from bed_reader import to_bed

    G = np.random.default_rng(3).integers(0, 3, size=(NROW, 25)).astype(np.float32)
    G[5, 7] = np.nan
    to_bed(tmp_path / "genotypes.bed", G)
    np.save(tmp_path / "phenotype.npy", design()[1])
    with pytest.raises(subprocess.CalledProcessError) as error:
        run_recipe("plink-bed", tmp_path)
    assert "missing genotype calls" in error.value.stderr


@pytest.mark.postgres
def test_postgres(tmp_path):
    if "DATABASE_URL" not in os.environ:
        pytest.skip("needs DATABASE_URL; make test-recipes-postgres starts a database")
    import psycopg

    X, y = design()
    columns = table_columns(X, y)
    with psycopg.connect(os.environ["DATABASE_URL"]) as db:
        db.execute("DROP TABLE IF EXISTS observations")
        db.execute(
            "CREATE TABLE observations (id integer, "
            + ", ".join(f"{name} double precision" for name in list(columns)[1:])
            + ")"
        )
        rows = list(
            zip(*(np.asarray(v).tolist() for v in columns.values()), strict=True)
        )
        with db.cursor() as cursor:
            # inserted out of order: the recipe's ORDER BY restores it
            cursor.executemany(
                f"INSERT INTO observations VALUES ({', '.join(['%s'] * len(columns))})",
                rows[::-1],
            )
    assert_same_fit(run_recipe("postgres", tmp_path), X, y, names())


def write_vcf(path, G):
    """Genotypes (rows = samples) as a minimal VCF; dosage d is d ALT alleles."""
    calls = {0: "0/0", 1: "0/1", 2: "1/1"}
    with open(path, "w") as out:
        out.write("##fileformat=VCFv4.2\n##contig=<ID=1>\n")
        samples = "\t".join(f"s{i}" for i in range(G.shape[0]))
        out.write(f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{samples}\n")
        for j in range(G.shape[1]):
            genotypes = "\t".join(calls[d] for d in G[:, j])
            out.write(f"1\t{100 + j}\tsnp{j}\tA\tG\t.\tPASS\t.\tGT\t{genotypes}\n")


@pytest.mark.genotype_tools
@pytest.mark.skipif(shutil.which("plink2") is None, reason="needs plink2 on PATH")
def test_vcf_through_plink2(tmp_path):
    G = np.random.default_rng(4).integers(0, 3, size=(NROW, 25)).astype(np.int8)
    _, y = design()
    write_vcf(tmp_path / "genotypes.vcf", G)
    np.save(tmp_path / "phenotype.npy", y)
    subprocess.run(
        ["sh", "-c", recipe("vcf-plink2", language="sh")], cwd=tmp_path, check=True
    )
    assert_same_fit(run_recipe("plink-bed", tmp_path), G, y)


def test_every_recipe_is_tested():
    """Each recipe in the docs is run by some test here, and nothing else is."""
    tree = ast.parse(Path(__file__).read_text())
    run = {
        call.args[0].value
        for function in tree.body
        if isinstance(function, ast.FunctionDef) and function.name.startswith("test_")
        for call in ast.walk(function)
        if isinstance(call, ast.Call)
        and getattr(call.func, "id", None) in ("run_recipe", "recipe")
        and call.args
        and isinstance(call.args[0], ast.Constant)
    }
    documented = set(re.findall(r"<!-- recipe: ([\w-]+) -->", DOCS.read_text()))
    assert documented == run, json.dumps(sorted(documented ^ run))
