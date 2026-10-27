# Gaussian endpoint contract (proposal)

Explicit Gaussian fitting is implemented in the shared Rust solver, Python
and WASM bindings, and sklearn regression estimators. The default-family
switch below is reserved for a separate final 0.3.0 release PR.

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
The default family becomes `"gaussian"`. Logistic calls must explicitly pass
`family="binomial"`, including calls inside the classifier wrappers. This
default change must be called out in the 0.3.0 release notes and examples.
Gaussian requires a finite numeric
response on the selected rows. The numba backend raises `NotImplementedError`
for Gaussian; the implementation is Rust only. Constant responses produce
the intercept-only optimum; supplied lambdas allow a single fit.

`BiglassoPath` gains a `family` field defaulting to `"gaussian"`; internal
constructors explicitly record the family actually fitted. Gaussian `predict()`
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
