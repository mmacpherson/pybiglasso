# bigstatsr and snpnet comparison protocol

The planned scope is native defaults and matched snpnet lasso paths. Local API,
correctness and row-subset diagnostics are unrecorded. Recorded measurements
follow the least-squares rerun, on the same benchmark CPU without overlap.
No historical timing set is reused.

There are two configurations, with separate rows for different delivered
artifacts. Native defaults are the primary comparison. Matched lasso paths
are an optimization comparison. There is no custom Python CMSA panel: CMSA
is a different model-selection and averaging procedure, not a solver
comparison. Native bigstatsr CMSA remains in the default-config panel.

## Inputs, environment and timing

Use the four existing path datasets, original seeds and complete 0/1/2
dosages. Freeze one stratified 20% outer test set per dataset; never use it
for calibration or selection. Publish input and split fingerprints. Test
metrics describe this one split, not a statistically established quality
ranking or a claim about missing/continuous dosages or ancestry structure.

Use the pinned release wheel and separate pinned R image. Rebuild cindex on
the benchmark CPU because it uses native CPU flags. Record package/source
versions, compiler, CPU affinity, RAYON/OMP/BLAS variables and effective
concurrency. Record storage types, logical/allocated bytes, conversion costs
and process-tree peak memory. Same values do not imply equal storage bandwidth.
File-backed inputs alone do not establish memory-capped performance.

Reset experiment RNGs for repetitions without replacing native fold
construction. Use at least three fresh-process repetitions and their median,
never their minimum. Require 180 consecutive seconds below 10% CPU load
before each repeat group and one-second load bookends at most 15%, outside
the fit clock. Retain all attempts; exclude whole load-rejected groups,
without choosing repetitions by speed. Serialize runs and restore the governor.

Measure **model-ready time**. For CMSA this includes the native fitting call,
`summary` selection/averaging, and obtaining usable averaged raw coefficients
and intercept. For native Python CV it includes selection and the full-data
refit; for snpnet it ends with the returned path ready to use. Include each
package's in-call standardization, grids and folds. Exclude serialization,
outer-test scoring and independent audits, and separately report imports,
storage attachment, genotype conversion and oracle-calibration/preparation.
Do not put required R summary/averaging outside the clock as coefficient export.
A wall safety timeout is not a fit-time lower bound. Only a cap applied to
these same native/model-ready intervals may supply such a bound; incomplete
paths and errors remain visible.

## Configuration 1: native defaults

Call `BiglassoClassifierCV()` and `big_spLogReg(X, y)` with no optional fitting
arguments. Supply the same training values and seed but leave each package's
folds, grids, convergence, feature limits, stopping and threading unchanged.
Do not force shared fold IDs or stratification inside a native default.
Do not silently constrain only one implementation's default concurrency.
Disclose the container/environment defaults as part of the configuration.

Record installed defaults and effective behavior, including:

- Fold count, construction and assignment; grid length and minimum-ratio rule.
- Tolerance, iteration and feature limits, saturation and early-stop reasons.
- Selection rule, tie handling, selected-index boundary hits and support size.
- Actual fold lengths, truncations, warnings and effective thread settings.

On the current pins, Python uses five stratified folds, a full-data 100-point
grid, `tol=1e-7`, `max_iter=1000`, `rule="min"`, no held-out-loss early stop
and automatic fold concurrency. Its automatic minimum ratio is 0.001 for
tall data and 0.05 otherwise; the estimator leaves the feature limit at the
solver default. bigstatsr 1.6.2 uses ten native random sets, 200-point
fold-local grids, `eps=1e-5`, `max.iter=1000`, `dfmax=50000`, `n.abort=10`,
`nlam.min=50`, and one fold worker; its minimum ratio is 0.0001 for tall data
and 0.001 otherwise. Assert these against installed code before recording.

Report held-out NLL and AUC, the outer-training null-model test baseline,
training NLL, support and completeness beside time. Native Python returns a
selected full-data refit; CMSA returns averaged fold-selected coefficients.
Do not assign a single penalized objective or KKT residual to that average.
Native R also includes a global null initialization before fold descent;
identify it when reporting selected-index boundary hits, without treating it
as a converged fold-training optimum.

For snpnet supply only genotype/phenotype locations, binomial family and
output placement. Keep its native grid, raw-dosage penalty, tolerance,
thread/round limits and filtering. Record the entire returned lambda and
accuracy curves and all warnings, failures or truncation. Its native path
is not CV: do not compare its time directly with a complete CV procedure,
choose a lambda on the outer test set, or silently apply the matched-path
workarounds. Defaults that cannot finish remain outcomes, not censored rows.

## Configuration 2: matched lasso paths, biglasso vs snpnet

Use the same full-training values, explicit grid, standardized lasso objective,
intercept and total thread budget. This is path-only, not CV. Use the existing
unpatched raw-dosage/std-penalty and zero-covariate workarounds, disclosed
explicitly. Independently check penalty-factor renormalization, zero-covariate
handling and raw/standardized conversion on the installed pin. Require complete,
finite paths and exact input-grid round trips. Do not extend the lasso-only
penalty mapping to elastic net.

Full returned paths permit calibration at **every optimized requested lambda**
against KKT acceptance ceilings of 1e-4 and 1e-6. Calibrate each solver
separately over the fixed tolerance sequence 1e-5 through 1e-14, retaining the
loosest setting that meets the chosen ceiling at every auditable lambda.
Disclose each solver's iteration, feature and round budgets; any limit exit
is an incomplete result, not a pass.
Freeze one setting per problem/accuracy target before repeated timing, retain
all calibration attempts and report their costs separately. These are
oracle-tuned same KKT acceptance ceilings, not equal convergence or a
certified objective gap. Identify any native initialization or saturation exception explicitly rather than calling
it converged. Recheck all timed outputs. Report per-lambda objective/KKT curves
and worst residual beside time, actual settings and calibration costs. If a
target cannot be reached or a path is truncated, retain that outcome.

## Independent accuracy calculations

Reconstruct raw logits from exported coefficients and intercept. With raw b,
training population scale s and mean c, use `beta=s*b`, `Z=(X-c)/s`,
`eta=X*b+b0`, `g=Z.T*(sigmoid(eta)-y)/n`. Include penalty factors q if present:

- Active residual: `abs(g_j + lambda*q_j*(alpha*sign(beta_j)+(1-alpha)*beta_j))`.
- Inactive residual: `max(abs(g_j)-lambda*q_j*alpha,0)`.
- Intercept residual: `abs(mean(sigmoid(eta)-y))`.

Compute stable NLL as `mean(logaddexp(0,eta)-y*eta)` and objective as NLL plus
`lambda*sum(q*(alpha*abs(beta)+(1-alpha)*beta**2/2))` at each auditable lambda.
Use actual logits, not a working residual, clipped probabilities or stale loss.
A small KKT residual does not certify coefficient distance on ill-conditioned
data. Any gap to the best observed objective is descriptive, not a certified
gap. Never generate expected values from pybiglasso output.

## Release conditions

Keep native-default and matched path-only rows separate, with quality,
accuracy coverage and completeness alongside time. No combined "faster than
bigstatsr/snpnet" headline. Publish seeds, folds/grids/input hashes, raw attempts,
settings, comparator source pins and audit calculations. Preparation must pass
current API/CV smokes, independent snpnet KKT and repository gates.
Review of the executable protocol precedes all recorded runs. Recorded ig3
follows the least-squares rerun and precedes the logistic README audit.

Sources: [bigstatsr 1.6.2 result export](https://github.com/cran/bigstatsr/blob/1.6.2/R/biglasso.R),
[null-initialization code](https://github.com/cran/bigstatsr/blob/1.6.2/inst/include/bigstatsr/biglasso/logistic.hpp),
[summary/averaging](https://github.com/cran/bigstatsr/blob/1.6.2/R/summary.R),
[pinned snpnet](https://github.com/rivas-lab/snpnet/tree/f7fca34569cd5c4cd83fd5c62ad8ab5b742591d9).
