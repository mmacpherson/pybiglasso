# Browser playground

The Python package and this app share `rust/src/solver.rs`. The browser adapter
ingests selected columns once into a retained float64 dataset, then borrows it
for every fit. Parsing, fitting and result export run in one dedicated Web
Worker. No selected file is sent to a server. The UI does not run cross-validation;
the displayed deviance is a training statistic.

The first examples are colon (R biglasso's 62x2000 binary gene-expression
dataset), Wisconsin breast cancer and deterministic correlated features. Sources
and transformations are in `public/examples/README.txt`. A deterministic
continuous-response example exercises Gaussian regression. Uploads can use
either binomial or Gaussian responses; diabetes and riboflavin remain possible
additional examples.

## Develop

Install Rust, its `wasm32-unknown-unknown` target, `wasm-pack` 0.15.0,
and Node 24 or newer. From this directory:

```sh
npm ci
npm run build
npm run dev
```

The development server prints its local URL. A production build is in `dist/`.
Asset paths are relative so the app works under GitHub Pages' repository path.
Generated `pkg/`, `dist/` and `node_modules/` are ignored.

## Checks

```sh
npm test
npx playwright install chromium firefox webkit
npm run test:browser
```

Browser tests use the production build, exercise both import formats, compare
their outputs, cover early input errors, and check the phone layout. On Linux,
the browsers also need their system libraries (`playwright install --with-deps`
on a supported CI image).

From the repository root, `uv run python scripts/check_wasm.py` checks the Wasm
binding against the stored R comparison grid, including iteration and screening
counts. It generates input data only; expected results remain the stored R
outputs. Native kernel tests also pin the exact intercept summation, including
cancellation, subnormals, ties and exponent carries.

Use `--engine chromium`, `--engine firefox` or `--engine webkit` with
`scripts/check_wasm.py` to run that same grid in a browser engine. `--native-bits`
also checks the same-input identity contract against the running native solver;
correctness expectations remain R's output. `--container IMAGE` runs the JS side
in a Playwright image when the local OS lacks a browser's system dependencies.

`make test-browser` builds and runs these checks. The existing native hooks,
Python suites and live R gates still apply when changing the shared solver.

## Input contract

- CSV needs a header with unique nonempty column names. Parquet needs flat
  scalar columns. Snappy, gzip, zstd and the other hyparquet-compressors codecs
  are available in the worker.
- Pick numeric, finite predictors. For binomial fits choose a response with
  exactly two classes; the chosen positive label maps to 1. Gaussian fits
  accept finite numeric responses and report training RSS. Missing values and categorical
  predictors require preprocessing; nothing is silently imputed or encoded.
- CSV integers and Parquet bigint values that cannot round-trip through
  float64 are rejected. Browser data is stored as float64 even if the file's
  physical storage is narrower.
- The decoded feature budget is 128 MiB, with at most 1,000,000 rows and
  20,000 features. Decoder buffers, y and path results require extra memory.
  A compressed file's size is not a memory estimate.
- The initial implementation fits in memory. Native mmap-based larger-than-RAM
  execution and shared-memory Wasm threads are not implemented here.
- Cancelling terminates the worker and frees its model and data; reload the
  example or file to begin again.

Raw-scale coefficients use the same division by scale as Python. The intercept
correction rounds the sum of the rounded products exactly, using a fixed-width
integer accumulator rather than JS summation. This preserves cancellation that
ordinary dot products can lose. Full-path JSON includes feature names, class
mapping, normalization and fit settings; CSV exports the chosen raw-scale model.

## Publish

`.github/workflows/browser.yml` builds and tests pull requests. On main it also
publishes `dist/` to GitHub Pages. In repository Settings / Pages, select GitHub
Actions as the build source. The solver source stays GPL-3.0-or-later, with a
source/license link in the app; parser dependency licenses remain in the bundle.

To regenerate the example inputs from external sources (no fitting involved):

```sh
docker run --rm -v "$PWD:/work" -w /work pybiglasso-r:latest \
  Rscript scripts/export_browser_colon.R web/public/examples/colon.csv
uv run python scripts/export_browser_breast_cancer.py
```

Run these commands at the repository root. The colon exporter insists on the
pinned biglasso 1.7.2 package. Commit generated input changes only after checking
their source, response encoding and dimensions.
