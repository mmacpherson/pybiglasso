# Gaussian endpoint contract

Explicit Gaussian fitting is implemented in the shared Rust solver, Python
and WASM bindings, and sklearn regression estimators. Version 0.3.0 makes
Gaussian the public default; logistic callers explicitly select binomial.

Version 0.3.0 adds Gaussian support and makes it the default family, matching
R biglasso. The deprecated numba backend remains available for binomial fits
through 0.3.x; its removal is deferred to 0.4.0. Gaussian minimizes

    RSS / (2 n) + lambda * sum_j pf_j (alpha |b_j| + (1-alpha) b_j² / 2)

with an unpenalized intercept and coefficients penalized on standardized X.
Only X is scaled; y stays in its original units. As for binomial fits, X is
read by column, row subsets index the original array, and results must be
identical across thread counts and Rust SIMD widths.

## Public API

`biglasso(X, y, family="gaussian", ...)` shares the existing path options.
The default family is `"gaussian"` in 0.3.0. Logistic calls explicitly pass
`family="binomial"`, as the classifier wrappers do. This breaks the 0.2.x
default and is called out in the README migration example.
Gaussian requires a finite numeric
response on the selected rows. The numba backend raises `NotImplementedError`
for Gaussian; the implementation is Rust only. Constant responses produce
the intercept-only optimum; supplied lambdas allow a single fit.

`BiglassoPath` records the family actually fitted, including `"gaussian"` for
default fits. Its dataclass field fallback stays `"binomial"` so paths pickled
before families existed retain their classification behavior. Gaussian `predict()`
and `decision_function()` return the linear prediction, with the existing
shape `(nrow, nlambda)`. `predict_proba()` raises for Gaussian. Coefficients,
intercepts, sparse exports and lambda selection keep their current formats.

Gaussian `deviance` is RSS (R biglasso's `loss`), whereas binomial deviance
remains twice NLL. `score(metric="deviance")` returns held-out RSS; Gaussian
also accepts `"mse"` and `"r2"`. Constant-response R² follows sklearn's finite
convention: 1 for an exact prediction and 0 otherwise. Classification metrics
are rejected for Gaussian, and regression metrics for binomial.

`BiglassoRegressor` and `BiglassoRegressorCV` expose the sklearn regression
contract: one-dimensional predictions at the selected lambda and R² from
`score()`. CV minimizes held-out MSE by default, selects the minimum-error or
one-standard-error lambda, and reports positive losses in `cv_scores_`, as
classifier CV currently reports deviance. The CV grid and final refit use the
full training data; fold fits use only their training rows. Response centering
must use those rows too. Classification estimators retain their current API.

## Numerical contract and implementation order

SSR coordinate descent is ported from
[R biglasso 1.7.2's Gaussian solver](https://github.com/pbreheny/biglasso/blob/v1.7.2/src/gaussian.cpp).
Stored R fixtures and a plain numpy reference establish coefficients, grids,
strong-rule rejections and iteration counts before adding the Rust kernel.
The Gaussian curvature is constant, so the elastic-net denominator is
`1 + lambda * pf * (1-alpha)` in both default and R-compatible modes.

Gaussian has no logistic saturation cutoff: fitting continues through the
requested path. `stop_on_saturation` therefore has no effect on this family.
`dfmax` and `max_iter` keep their existing meanings. Default mode reports loss
at the final iterate. Compatibility mode retains any R loss bookkeeping
differences; each difference needs the code comment, README entry and
independent check required by AGENTS.md.

R comparisons cover tall and wide designs, elastic net and lasso, monomorphic
and duplicate columns, penalty factors including zeros, row subsets, compact
integer and float32 storage, read-only memmaps, supplied/linear grids, dfmax,
and short iteration budgets. Existing binomial fixture entries must remain
unchanged. Independent checks use least-squares KKT conditions, exact
curvature, duality gaps and sklearn/glmnet, without deriving expected values
from pybiglasso.

Native Python and WASM bindings should share the same Rust solver core.
Gaussian belongs in that core, so both bindings can use the same algorithm,
fixed-order arithmetic and path representation.

## Exact residual-sum reuse

The coordinate sweep caches `seqsum(residual)`. After a nonzero coefficient
shift, it adds the updated residual values left to right in the existing
update loop, including masked zeros. Thus the cache has exactly the bits of
a separate `seqsum` pass. A zero shift leaves both residual and cache unchanged;
KKT scans can use the same cache. This is not an algebraic update such as
subtracting the shift times a column sum, which would round differently.
Only one scalar is added to the state; neither X nor a row buffer is copied.

The cache invariant has a Rust regression covering masked rows, chunk/tail
sizes and large column offsets. Existing R trajectories and independent
optimality checks remain unchanged. Additional before/after checks compare
complete path bytes, including coefficients, intercepts, centering/scaling,
grids, losses, iterations and rejection counts, rather than an optimum alone.

### Before/after measurements

[`benchmarks/gaussian-sum.py`](../benchmarks/gaussian-sum.py) alternates actual
baseline and candidate Rust extensions in one Python process, using identical
arrays. Each library/pool is warmed before timing. Timers cover the whole fit,
including validation and standardization, but exclude data generation, page
eviction and result hashing. Hashes check transformations of the same fit;
they are not stored expected solver values.

The committed [in-memory results](../benchmarks/results/gaussian-sum-memory.jsonl)
compare baseline numerical core `9ed067e` with candidate source `d06812f` on an
AMD Ryzen 5 7640U (6 cores, 12 threads), Python 3.15.0 and NumPy 2.5.4. These
are local desktop before/after measurements, not the dedicated Ryzen 5 5600X
R/glmnet comparisons in the README. No competing compiler was observed in the
retained runs; ordinary desktop activity and CPU-clock variation remain.
The table spans lasso/elastic net and 1, 6 and 12 threads, using medians of five
repetitions per version on a 30-lambda path. Ratios above 1 favor the candidate.

| Storage | 100,000 x 100: baseline/candidate time | 2,000 x 20,000: baseline/candidate time |
|---|---|---|
| float64 | 1.17–1.31x | 0.97–1.00x |
| float32 | 1.20–1.36x | 0.99–1.03x |
| int8 | 1.22–1.48x | 0.97–1.01x |

The gain is on tall data, not every shape. Wide changes are about ±3%.
The [initial out-of-core run](../benchmarks/results/gaussian-sum-out-of-core.jsonl)
uses a 1.6 GB read-only float64 memmap under a real 1 GiB cgroup cap, eight
lambdas, both penalties, 1/6 threads and three repetitions per version. Every
timed fit rereads multiple matrix sizes from disk, with refaults and scope
peak memory recorded. Pages are evicted before each timed fit.

Those six-thread medians were about 5% slower. A single
[balanced six-pair follow-up](../benchmarks/results/gaussian-sum-out-of-core-balanced.jsonl)
tested ordering/page-cache effects and was 1–2.3% slower. Both sets are retained;
there is no out-of-core speedup claim. All reported paths are bit-identical
between builds and thread counts. Earlier compiler-overlapped exploratory
timings are excluded.

Build the baseline extension in a separate worktree, then run from the candidate:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python benchmarks/gaussian-sum.py \
  --baseline-extension ../baseline/src/pybiglasso/_rust.abi3.so

systemd-run --user --scope -p MemoryMax=1G -p MemorySwapMax=0 \
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 UV_NO_SYNC=1 \
  uv run python benchmarks/gaussian-sum.py \
  --baseline-extension ../baseline/src/pybiglasso/_rust.abi3.so \
  --case out-of-core --threads 1 6 --repeats 3 --nlambda 8 \
  --scratch-dir benchmarks/results
```

Use `--dtype`, `--selection` and `--out` for storage/subset cases and raw JSONL.
Output appends, so choose a fresh destination. Out-of-core scratch must be on
real disk, not tmpfs; the runner rejects a missing or insufficient memory cap.
