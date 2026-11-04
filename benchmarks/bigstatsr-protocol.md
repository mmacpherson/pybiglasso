# bigstatsr and snpnet comparison protocol

Proposal for review before implementation or recorded runs. Local API,
correctness and row-subset diagnostics are unrecorded. Recorded measurements
follow the least-squares rerun, on the same benchmark CPU without overlap.
No historical timing set is reused.

There are two configurations, with separate rows for different delivered
artifacts. Native defaults are the primary comparison. Matched lasso paths
are an optimization comparison. The optional custom-CMSA study below has
**selected-model accuracy coverage only** and needs separate acceptance of
that limitation; it is not accuracy-matched model selection.

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

For snpnet supply only genotype/phenotype locations, binomial family and
output placement. Keep its native grid, raw-dosage penalty, tolerance,
thread/round limits and filtering. Record the entire returned lambda and
accuracy curves and all warnings, failures or truncation. Its native path
is not CV: do not compare its time directly with a complete CV procedure,
choose a lambda on the outer test set, or silently apply the matched-path
workarounds. Defaults that cannot finish remain outcomes, not censored rows.

## Configuration 2a: optional custom CMSA, lasso only

Label: **Custom Python CMSA using biglasso vs native bigstatsr CMSA — fixed R
fold grids; selected-fold KKT only, discarded fits unaudited.** This studies
public orchestration, not the built-in Python CV estimator or a pure kernel.
It proceeds only after review accepts the limited accuracy claim.

The public bigstatsr result contains each fold's selected coefficients and
intercept, grid and loss/iteration curves, **not coefficients at every lambda**.
`return.all` does not provide full-path export. Therefore no every-lambda KKT,
full-path accuracy matching or accuracy-matched selection claim is possible
from these public results. We will not add instrumentation or private solver
calls to overcome that limitation. Inaccurate discarded candidates can alter
selection even when the selected model passes KKT.

Use ten shared stratified folds, 100 requested candidates per fold, minimum ratio 0.05,
fold-training population scaling, an unpenalized intercept and lasso only.
Disable validation-loss stopping (`n.abort` above the path length on R),
set `dfmax=p+1` and `max.iter/max_iter=10000` on both before calibration.
Keep saturation exits visible; an incomplete requested path is not a completed
matched run. Use the same fold concurrency, one or six, and one solver/BLAS
thread per fold without nested parallelism.

bigstatsr constructs its fold grids internally; it cannot accept a supplied
common grid. Freeze its exact fold grids in untimed preparation, pass these
arrays to public Python row-subset fits, and assert every timed R run
reproduces them. Label time **conditional on supplied frozen grids**: R still
constructs summaries/grids in its measured call; Python receives those grids.
Report preparation/calibration costs separately. Do not charge a preparatory
R CV fit to Python as though it were a production grid-building requirement.
Python's public fits retain complete paths; R retains selected vectors. That
output/API cost is part of the orchestration, not hidden from the comparison.

### Null initialization and selection

Native R candidate index 0 is a global null initialization, with zero slopes,
intercept based on the entire outer-training prevalence and zero iterations.
Descent starts at index 1. It is not generally fold-training stationary:
its intercept residual is `p_global - mean(y_fold_train)`.

Keep this sentinel eligible for selection exactly as in native CMSA. In the
Python orchestration supply that same global-null candidate explicitly, then
fit indices 1 onward with public fold paths. Thus 100 candidates mean one
initialization and at most 99 optimized candidates, not 100 converged fits.
Public Python fits have their own initialization/trajectory; no bitwise or
iteration-identity claim is made. Publish the runnable orchestration, validate
native strict-improvement/first-minimum tie semantics on the installed pin,
and average selected **raw coefficients and intercepts**, not probabilities.
If public validation curves do not uniquely identify the selected lambda
(including rounding ties), report the ambiguity and do not certify that
fold against a selected-lambda KKT target. Do not invent an unexported index.

Retain a deliberately unequal-fold-prevalence check demonstrating the sentinel,
its nonzero intercept KKT, its eligibility and the case where it wins. If a
selected vector is this sentinel, label it initialization-selected and report
its actual KKT; do not count it as meeting a converged-fold target or remove
it from native results. If it happens to pass a ceiling, still identify its
initialization status. Do not silently replace it by a fold-null optimum.

### Observable accuracy coverage

Calibrate separately against selected-fold KKT ceilings 1e-4 and 1e-6. Use a fixed
tolerance sequence 1e-5 through 1e-14, choosing the loosest setting satisfying
the ceiling at every returned optimized selected-fold vector. Freeze one
setting per declared problem/configuration before repeated timing, not per
repeat or by speed. Retain all attempts and re-audit timed selected outputs.
Null selections, unreachable targets, max-iteration, feature-limit and
saturation exits remain explicit rather than changing ceilings to pass.

These are oracle-tuned **same KKT acceptance ceilings**, not equal convergence,
a certified objective gap, or evidence that discarded candidates were accurate.
Report selected indices/lambdas, disagreements between selectors, per-fold KKT
and objective, and held-out quality beside time. No single objective/KKT is
assigned to the averaged vector, and no matched full-path speed ratio is made.

Exclude alpha 0.5 from common-objective matched claims. Pinned bigstatsr's
logistic update divides by `v*(1+l2)`, whereas the standardized elastic-net
objective needs `v+l2`; tighter tolerance or a universal lambda conversion
cannot fix the feature/probability-dependent difference. No `r_compatible`
comparison is proposed here. Existing alpha 0.5 functional smokes are explicitly
unmatched-objective checks, not a fair-fight result.

## Configuration 2b: matched lasso paths, biglasso vs snpnet

Use the same full-training values, explicit grid, standardized lasso objective,
intercept and total thread budget. This is path-only, not CV. Use the existing
unpatched raw-dosage/std-penalty and zero-covariate workarounds, disclosed
explicitly. Independently check penalty-factor renormalization, zero-covariate
handling and raw/standardized conversion on the installed pin. Require complete,
finite paths and exact input-grid round trips. Do not extend the lasso-only
penalty mapping to elastic net.

Full returned paths permit calibration at **every optimized requested lambda**
against the same two KKT ceilings and fixed tolerance sequence. Identify any
native initialization or saturation exception explicitly rather than calling
it converged. Recheck all timed outputs. Report per-lambda objective/KKT curves
and worst residual beside time, actual settings and calibration costs. If a
target cannot be reached or a path is truncated, retain that outcome.

## Independent accuracy calculations

Reconstruct raw logits from exported coefficients and intercept. With raw b,
fold population scale s and mean c, use `beta=s*b`, `Z=(X-c)/s`,
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

Keep native-default, custom-CMSA and path-only rows separate, with quality,
accuracy coverage and completeness alongside time. No combined "faster than
bigstatsr/snpnet" headline. Publish seeds, folds/grids/input hashes, raw attempts,
settings, comparator source pins and audit calculations. Preparation must pass
current API/CV smokes, independent snpnet KKT, null-sentinel/selection checks
and repository gates. Protocol review precedes implementation of the optional
custom-CMSA study and all recorded runs. Recorded ig3 follows the least-squares
rerun and precedes the logistic README audit.

Sources: [bigstatsr 1.6.2 result export](https://github.com/cran/bigstatsr/blob/1.6.2/R/biglasso.R),
[selected-vector and null-initialization code](https://github.com/cran/bigstatsr/blob/1.6.2/inst/include/bigstatsr/biglasso/logistic.hpp),
[elastic-net update](https://github.com/cran/bigstatsr/blob/1.6.2/inst/include/bigstatsr/biglasso/utils.hpp),
[summary/averaging](https://github.com/cran/bigstatsr/blob/1.6.2/R/summary.R),
[pinned snpnet](https://github.com/rivas-lab/snpnet/tree/f7fca34569cd5c4cd83fd5c62ad8ab5b742591d9).
