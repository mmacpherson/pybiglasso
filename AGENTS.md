# pybiglasso — Contributor & Agent Guide

Guidance for anyone working in this repo, human or AI coding agent. This is the
canonical guide; `CLAUDE.md` is a symlink to this file so tool-specific loaders
pick up the same content. It follows the [AGENTS.md](https://agents.md) convention.

## Working in this repo

- **`main` is protected** — changes land through a pull request, not direct pushes.
- **Branch for each change**, and keep PRs focused.
- **Never force-push** `main` or any branch with an open PR.
- Never bypass pre-commit hooks with `--no-verify`; fix the underlying issue instead.
- Issues are tracked locally with beads (`bd`); `.beads/` is untracked and stays
  on this machine.

## Matching R biglasso is the compatibility contract

pybiglasso ports R biglasso's binomial and Gaussian solvers, matching them to
floating-point precision: coefficients, lambda grid, centering and scaling,
strong-rule rejections, and the iteration count at every lambda. A change that
moves any of these, on a case where R's behavior is intended, is a regression.
The R comparison exercises `r_compatible=True`; default mode is held to the
plain reference implementation and independent checks described below.

Where R's behavior looks unintended, pybiglasso may differ, but only like this:

1. Mark the code with a comment beginning `Differs from R biglasso`, saying what R
   does and why this differs. Keep the change as small and plain as possible.
2. Add an entry to the README's "Differences from R biglasso" section, in a
   courteous register: state what R does and what pybiglasso does, without
   verdicts.
3. Where feasible, keep R's behavior reachable with `r_compatible=True`.
4. Back the difference with a check that doesn't depend on R (optimality
   conditions, another solver, an invariance).

## Expected values come from outside pybiglasso

Never run pybiglasso to generate or adjust expected values. Passing tests are
not evidence that their expectations are correct, and tolerances are not tuned
to observed output.

- **R output**: `tests/fixtures/r_biglasso.json.gz` holds R biglasso's results for
  the comparison grid. Regenerate it only with `make fixtures`, which builds the
  pinned R image (`FIXTURES_R` in the Makefile) and refuses output from any other
  biglasso version (`FIXTURES_BIGLASSO` in `scripts/make_r_fixtures.py`). When
  adding a case to `tests/support/rcompare.py`, check that the old code fails it if it is
  meant to catch a regression.
- **glmnet output**: `tests/fixtures/r_glmnet.json.gz` holds R glmnet's fits of
  the default objective (`tests/support/rglmnet.py`), from the same image and
  `make fixtures`, pinned by `FIXTURES_GLMNET`. They are compared through
  certificates (both fits pass KKT, then a certified distance), not values.
- **Independent checks**: KKT conditions computed from first principles
  (`tests/test_default_enet.py`), scikit-learn's saga solver, and invariances
  such as shifting X.

## Dependency & toolchain versions

- **Release Rust**: wheel jobs pin `RUSTUP_TOOLCHAIN` in `.github/workflows/ci.yml`
  and pass it to maturin; the Linux matrix also pins both manylinux images by
  digest. To bump deliberately, update those pins, check `make wheel-provenance`
  logs (Rust, LLVM and maturin) and the wheel smoke checks on every platform,
  and run the usual gates. Other jobs and local development keep their selected
  toolchains.
- **Python (uv)**: loose `pyproject.toml` constraints for users, and a committed
  `uv.lock` that pins development and CI only (it never ships). CI installs from
  the lock as committed (`UV_LOCKED=1`), and a latest-deps job tests the newest
  versions `pyproject.toml` allows, which is what a new install gets. The
  minimum-deps job runs `make test-minimum` on Python 3.11, with an isolated
  lowest-direct resolution and exact installed-floor checks; it never edits
  the development lock.
- **R**: the R comparison runs in a container (`docker/r-biglasso.Dockerfile`),
  pinned by R version; packages come from that R release's P3M snapshot.
- Supported Pythons are those not yet at end of life.

## Development Commands

- `make test` — fast suite, against stored R output (no R needed)
- `make test-minimum` — the same suite with every declared dependency floor
- `make test-r` — the same grid live against the pinned R release that made the
  fixtures, in a container (`CONTAINER=podman` works too); use
  `R_VERSION=latest` for current CRAN
- `make test-recipes` — every snippet in `docs/data.md`, run verbatim;
  `make test-recipes-postgres` adds the PostgreSQL one (in a container unless
  `DATABASE_URL` is set), `make test-recipes-tools` the ones needing `plink2`
- `make soak` — larger memory-mapped problems against live R; for a big machine.
  Runs `make soak-hypothesis` first: the property tests on thousands of random
  examples each, where `make test` and CI run a fixed 20
- `make fixtures` — regenerate the stored R output (see above)
- `make run-hooks-all-files` — every pre-commit hook on every file
- `benchmarks/run-path.sh` — the README's in-memory path benchmarks

## Quality Gates

1. `make run-hooks-all-files`
2. `make test`
3. `make test-r`, for any change to the solver or the R comparison

What a pull request runs in CI follows from the files it changes. Packaging,
bindings, toolchain, dependency and CI changes run everything, with every
wheel, as main and releases do. Other changes under `rust/`, `src/` or `tests/`,
`scripts/test_rust.py` or the benchmark scripts add the macOS job, whose runners can queue for an hour, and build the Linux
x86 wheel only. Anything else skips macOS. To run everything on a branch by
hand, start the CI workflow from the Actions tab; that run is extra evidence
and does not replace the pull request's own required check.

## Python Style

- Use modern type hints (PEP 604 unions with `|`, generics without `typing` module)
- Prefer `pathlib.Path` over `os.path`
- Comment the why, not the what
- Use descriptive variable names over comments
- Keep functions focused and small
- Prefer explicit imports over star imports

## Architecture

- `src/pybiglasso/fit.py` — `biglasso()`, for both families: validation, then
  the Rust solver, and the fitted path's assembly
- `rust/src/solver.rs` — what both families share: parameters,
  standardization, scores, and `run`, the whole path in one call, which hands
  the fit to its family. Python bindings are in `rust/src/python.rs` and
  browser bindings in `rust/src/wasm.rs`. The README maps its steps to R
  biglasso's C++.
- `rust/src/columns.rs` — reading X a column at a time, whatever its layout
  or storage; `rust/src/kernels.rs` — the fixed-order sums and SIMD dot products
- `rust/src/binomial.rs` — logistic SSR coordinate descent: IRLS, the
  intercept, the fused coordinate sweep, and the lambda loop
- `rust/src/gaussian.rs` — Gaussian SSR coordinate descent, using the same
  column access, standardization and fixed-order score reductions.
- `web/` — the static browser playground and CSV/Parquet import worker;
  `make browser` builds it, `make test-browser` checks it against R and in browsers.
- `tests/support/reference.py` — the same algorithm in plain numpy, step by
  step; the tests hold the Rust solver to it
- `src/pybiglasso/path.py` — `BiglassoPath`, the fitted path: prediction, scoring,
  dataframe export
- `src/pybiglasso/arrow.py` — zero-copy column access for Arrow tables and
  chunked arrays; `src/pybiglasso/ingest.py` — bounded-memory conversion from
  tabular files to solver-ready NumPy or Feather data
- `src/pybiglasso/sklearn.py` — `BiglassoClassifier`, `BiglassoClassifierCV`
  and the corresponding `BiglassoRegressor`, `BiglassoRegressorCV`
- `tests/support/rcompare.py`, `tests/r/fit_biglasso.R` — the R comparison harness
- `benchmarks/` — in-memory and out-of-core benchmarks; results in
  `benchmarks/results/`

## Key Patterns

- **Out of core**: X is read a column at a time, through the memmap, and never
  loaded or copied; row subsets are index arrays into the same X.
- **Reproducibility**: every sum runs in an order fixed by the data's shape
  (CHUNK-row chunks added in order, LANES accumulators combined in a fixed
  tree, no fused multiply-add), so results are bit-identical at any thread
  count and SIMD width; don't
  replace them with `@` or `.sum()`, or let operations fuse.

## Architecture-specific code

Keep code portable, even if it is slower on one platform. An architecture-specific
path is allowed only when it is bit-identical by construction (the same arithmetic
in the same order), confined to one small function with a comment explaining why,
and exercised by CI's cross-platform bit check. It must have measured gains on at
least two chips of the winning architecture, with no measured loss on the other
architecture using its retained path.

## Public history

The repository is public. Keep internal and machine-identifying material out of
commits: no hostnames, home-directory paths, or employer-internal code or
templates. Benchmarks record the CPU and versions, not the host.

## Code Review Focus

- Agreement with R, and the iteration counts in particular
- Any new departure from R follows the rules above
- Memory: nothing that loads or copies X
- Thread-count reproducibility
