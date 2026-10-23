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

## Matching R biglasso is the contract

pybiglasso is a port of R biglasso's logistic solver, and matches it to
floating-point precision: coefficients, lambda grid, centering and scaling,
strong-rule rejections, and the iteration count at every lambda. A change that
moves any of these, on a case where R's behavior is intended, is a regression.

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
  adding a case to `tests/rcompare.py`, check that the old code fails it if it is
  meant to catch a regression.
- **Independent checks**: KKT conditions computed from first principles
  (`tests/test_default_enet.py`), scikit-learn's saga solver, and invariances
  such as shifting X.

## Dependency & toolchain versions

- **Python (uv)**: loose `pyproject.toml` constraints for users, and a committed
  `uv.lock` that pins development and CI only (it never ships). CI installs from
  the lock as committed (`UV_LOCKED=1`), and a latest-deps job tests the newest
  versions `pyproject.toml` allows, which is what a new install gets.
- **R**: the R comparison runs in a container (`docker/r-biglasso.Dockerfile`),
  pinned by R version; packages come from that R release's P3M snapshot.
- Supported Pythons are those not yet at end of life.

## Development Commands

- `make test` — fast suite, against stored R output (no R needed)
- `make test-r` — the same grid live against R, in a container
  (`CONTAINER=podman` works too; `R_VERSION=latest` for current CRAN)
- `make soak` — larger memory-mapped problems against live R; for a big machine
- `make fixtures` — regenerate the stored R output (see above)
- `make run-hooks-all-files` — every pre-commit hook on every file
- `make profile` — where a fit spends its time (Python 3.15's sampling profiler)
- `benchmarks/run-path.sh`, `benchmarks/run-extra.sh` — the README's benchmarks

## Quality Gates

1. `make run-hooks-all-files`
2. `make test`
3. `make test-r`, for any change to the solver or the R comparison

## Python Style

- Use modern type hints (PEP 604 unions with `|`, generics without `typing` module)
- Prefer `pathlib.Path` over `os.path`
- Comment the why, not the what
- Use descriptive variable names over comments
- Keep functions focused and small
- Prefer explicit imports over star imports

## Architecture

- `src/pybiglasso/binomial.py` — the solver: `biglasso()`, the driver
  (`binomial_hsr`, `_fit_lambda`), and one numba kernel per step. The README's
  "How the code is organized" maps each kernel to R biglasso's C++.
- `src/pybiglasso/path.py` — `BiglassoPath`, the fitted path: prediction, scoring,
  dataframe export
- `src/pybiglasso/sklearn.py` — `BiglassoClassifier`, `BiglassoClassifierCV`
- `tests/rcompare.py`, `tests/r/fit_biglasso.R` — the R comparison harness
- `benchmarks/` — in-memory and out-of-core benchmarks; results in
  `benchmarks/results/`

## Key Patterns

- **Out of core**: X is read a column at a time, through the memmap, and never
  loaded or copied; row subsets are index arrays into the same X.
- **Numba compiles each array type separately** (a read-only memmap is not a
  writable array), so warm-ups must use the same type as the fit.
- **Reproducibility**: kernels sum in a fixed order (`dot`, `seqsum`), so results
  are bit-identical at any thread count; don't replace them with `@` or `.sum()`.

## Public history

The repository is public. Keep internal and machine-identifying material out of
commits: no hostnames, home-directory paths, or employer-internal code or
templates. Benchmarks record the CPU and versions, not the host.

## Code Review Focus

- Agreement with R, and the iteration counts in particular
- Any new departure from R follows the rules above
- Memory: nothing that loads or copies X
- Thread-count reproducibility
