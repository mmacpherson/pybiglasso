# Tested dependency minimums

`make test-minimum` runs the full fast suite in a disposable Python 3.11
environment. It uses uv's `lowest-direct` resolution for the runtime extras,
minimum build backend and development tools that declare a floor. Unbounded
test tools and pandas/Polars use current compatible releases, constrained so
they cannot upgrade the resolved minimums. Before calling `make test`, the
target installs a wheel built outside the source tree, then checks that each
declared floor is actually installed; a transitive
requirement masking a floor is an error. The committed `uv.lock` is unchanged.

CI uses Python 3.11, the oldest supported Python, so missing wheels for a
newer interpreter do not silently raise the tested versions.

The minimum Maturin backend is tested as well. The explicit
`pyo3/extension-module` build feature keeps the extension from linking to one
interpreter's shared library. Minimum builds use a separate Cargo target and
install a wheel so they do not replace the editable extension used by the
normal development environment.

## Compatibility checks

The following probes used Python 3.11.14 on Linux x86-64. The Arrow bisect
held NumPy 2.0.0, SciPy 1.13.0, sklearn 1.9.0, pandas 3.0.6 and Polars 2.1.0
fixed while changing released PyArrow versions. Each passing candidate ran
`tests/test_arrow.py`, `test_arrow_memory.py`, `test_arch_bits.py`,
`test_ingest.py`, `test_ingest_excel.py` and `test_ingest_memory.py`:
519 passed and one expected upstream projection xfail.

| Dependency | Minimum | Evidence deciding the bound |
|---|---|---|
| NumPy | 2.0.0 | PyArrow 26.0.0 refuses NumPy 1.26.4 at import. NumPy 2.0.0 passes the Arrow/ingest suite and is the shared baseline for both minimum and current Arrow. Older NumPy can work with older Arrow; that mixed-version range is not the supported 0.5 baseline. |
| SciPy | 1.13.0 | SciPy 1.12.0 declares `numpy<1.29` and cannot resolve with NumPy 2.0.0. The 1.13.0 baseline passes the Arrow/ingest suite and the full minimum gate. |
| PyArrow | 16.0.0 | Bisect: 26.0.0 and 16.1.0 pass; 14.0.0 fails to import with NumPy 2; 15.0.0 and 15.0.2 require `numpy<2`; 16.0.0 passes. The adjacent lower release, 15.0.2, cannot resolve the fixed baseline. |
| scikit-learn | 1.6.0 | The existing floor is retained. Polars feature names in the estimators need scikit-learn 1.9 or newer: six cases lose `feature_names_in_` with 1.6.0, 1.7.0 and 1.8.0, and are strict xfails below 1.9. All sixteen pandas/Polars functional and estimator cases pass normally with 1.9.0. |
| seaborn | 0.13.0 | `test_plots` passes at the existing floor with pandas 3.0.6 and NumPy 2.0.0. |
| openpyxl | 3.0.6 | Excel bisect: 3.1.5, 3.0.10 and 3.0.6 pass all 21 Excel cases; 3.0.4 and the adjacent 3.0.5 fail on removed NumPy aliases during import. |

The Arrow cases cover mapped ownership, packed bools, slices and independent
chunk boundaries, full-path bit identity, callbacks that drop outer owners,
compression rejection before decoding, and native memory caps. The explicit
`RecordBatchReader.from_stream` conversion is tested too. A float16 rejection
fixture supplies NumPy float16 values because older Arrow does not construct a
half-float array from Python integers; its values and expected rejection are
unchanged.

Pandas and Polars are not extras. Install them directly when using
`BiglassoPath.to_pandas()` or `.to_polars()`. They are direct, unpinned development
dependencies so both output methods and array-like input compatibility are
exercised. The exporter ImportErrors name `pip install pandas` or
`pip install polars` and have missing-package regression tests.

No fitted expectations, R fixtures or numerical tolerances are generated or
changed by these checks.
