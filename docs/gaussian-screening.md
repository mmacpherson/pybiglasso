# Gaussian score certificates: arithmetic derivation

A certificate omits an inactive-column read only when it proves the same
KKT and next strong-rule decisions as an exact score. A naive squared norm
can underflow (for example at 1e-200); padding after sqrt does not repair it.
Here "stock" means the exact-read solver under the selected arithmetic mode,
with certificates disabled, rather than a frozen previous release.

## Arithmetic assumptions and supported paths

All operands are binary64 floating point numbers. Arithmetic uses round to
nearest, ties to even, with gradual underflow, no FTZ/DAZ, no fast-math or
reassociation. `sqrt` is correctly rounded. The data and the floating point
environment stay fixed during a fit, as required for deterministic evaluation.
Rust's normal compilation is used; no reciprocal approximation is introduced.
The implementation tests rounding and denormal behavior using runtime `black_box`
operands. All Rayon workers are checked before enabling the cache; norm/error
helpers also check their executing thread. Unsupported arithmetic disables
certification and leaves exact reads in place. Changing the FP environment
from a callback during a fit is outside the fixed-environment assumptions.

### Detecting unsupported floating-point environments

The bound is **not** asserted under FTZ/DAZ or AArch64 FZ. The library does
not read or change control registers. Instead, `arithmetic_ok()` uses
runtime `black_box` operands and inspects result bits:

- `1 + 2^-53` and `1 + 3*2^-54` distinguish ties-to-even from directed modes.
- `2^-1074 * 1` must retain its subnormal input.
- `MIN_NORMAL * 0.5` must produce the exact subnormal with bits `1 << 51`.
- `MIN_NORMAL * 0.1` must produce bits `0x000199999999999a`, including
  rounding an inexact subnormal result. Using both output probes covers
  flush behavior that distinguishes exact and inexact products.

For the last probe, the binary64 value 0.1 has significand
`0x1999999999999a` and exponent -56. In units of `2^-1074`, its product
with `MIN_NORMAL = 2^-1022` is exactly `0x1999999999999a / 16`.
The discarded hexadecimal digit is a (greater than the half-way digit 8),
so round-to-nearest gives `0x199999999999a`.

At cache setup, all workers are probed; the fitting thread's residual norm
also probes its own arithmetic. Either failure makes the cache invalid.
An invalid cache requires exact column-score reads before even considering
the same-epoch shortcut. Norm/error helpers and scan decisions also probe
the executing thread. No safe-norm fallback or zero sentinel authorizes a
certificate when the arithmetic check fails: the safe norm requires the
same gradual-underflow assumptions. The original coordinate/score formulas
then execute under the caller's flags. Such a fit is compared with stock
under those same flags, not with an ordinary gradual-underflow fit.

The creator-mode regression sets flags before building a pool, then reads
each worker's actual control state before any worker-side mode setter.
Thread creation is platform-specific:

- Linux workers must preserve the requested flush and rounding bits, as
  described by [pthread_create(3)](https://man7.org/linux/man-pages/man3/pthread_create.3.html).
- macOS ARM64 workers must start with default FPCR=0 and supported arithmetic.
  Apple's thread-creation path calls
  [machine_thread_state_initialize](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/osfmk/arm64/status.c#L2066-L2084),
  which resets FPCR to
  [FPCR_DEFAULT=0](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/osfmk/arm64/proc_reg.h#L838).
  It does not copy the creator's FZ setting.
- Other platforms are checked against their observed uniform worker state,
  without assuming Linux inheritance.

Worker-probe results are recorded independently of cache metadata. Stock and
enabled fits must observe the same state and match full outputs and recorded
trajectory at 1/2/4/6 threads for both alphas, including bit-constructed
subnormal responses. Unsupported workers require zero skipped reads and
zero next-strong certificates. Supported workers require a valid cache and
positive-radius/next-threshold skips on the analytic fixture. Thus a flushed
creator with healthy workers on macOS is checked as well as Linux's inherited
flush mode. The product probes actual fitting/worker arithmetic; it does not
infer it from the platform or creator.

The separate explicit uniform-mode test sets every worker to the requested
unsupported state on all platforms. Fit-thread and single-other-worker cases
also remain covered. Test-only RAII controls use MXCSR on x86-64 and FPCR.FZ
on AArch64; the AArch64 register layout and access are specified in
[Arm DDI 0502G, section 2.3](https://documentation-service.arm.com/static/5e907926c8052b16087615a6).
Creating/fitting-thread control state is restored; worker controls are confined
to isolated child processes.

Local runtime tests execute on Linux x86-64 (FTZ, DAZ and both). AArch64
Linux/macOS cross-checks compile the native test code and FPCR controls;
cross-compilation is not runtime evidence. Native Linux ARM and macOS CI
separately validate the actual worker state and fit behavior.

Let `u=2^-53` and `eta=2^-1074`. For a finite result in this environment,
`fl(t)` satisfies `|fl(t)-t| <= u*|t| + eta`. This bound includes subnormal
rounding; it does not pretend that every operation has a relative error bound.

Every invalid/nonfinite enclosure returns None. A failed residual/prefix
maintenance check permanently disables new skips in that fit. A failed column
metadata/error check forces an exact read for that column, except that an
unchanged residual epoch may reuse its already exact deterministic predicate.
That shortcut does not turn an invalid norm into a valid radius. The invalid norm's
stored zero sentinel is not an admissible bound: it is unreachable while the
cache's valid flag is true. Any previously issued certificate remains tied to
its original score epoch and exact threshold bits.

## Outward primitives

For nonnegative finite inputs, `up(op)` evaluates the ordinary operation then
applies `next_up`; nonfinite outputs are rejected. One representable neighbor
above an RN result encloses the exact real result, including when RN produces
zero/subnormal. A rounded overflow is rejected, not saturated to a finite
bound. Lower prefix endpoints use `next_down`. All magnitudes below are
nonnegative enclosures; each written add/multiply/divide is outward rounded.
`eta` additions are at the unit of the operation that incurs the error.

## Norm dispatcher and guarded one-pass fast path

The fixed-input, round-to-nearest, gradual-underflow assumptions above still
apply. `norm` checks the executing thread's arithmetic first. A failed check
returns None; it does not enter either numerical norm path.

The fast path uses exact constants A=2^-240 and B=2^240. It deliberately
DOES NOT scale accepted values. Scaling by2^-256 was considered and removed:
this guard already gives ample overflow headroom, while scaling would move
small products unnecessarily closer to the normal floor and add operations.
Only the existing safe fallback uses dynamic scaling.

1. Form the checked integer m=2*n+64, checking integer overflow, exact binary64
   conversion (m<=2^53), and the existing `gamma_terms` validity guard. It
   returns an upper g>=gamma_m=m*u/(1-m*u), with its rounded upper m*u<1/4.
   In particular n<2^50 and g<1/2. Failure selects the safe path.
2. In ONE pass, for every v=abs(at(i)), accumulate the predicate
   `v==0 || A<=v<=B`. Nonfinite values fail this predicate. For passing values
   use w=v; for failing values use dummy w=0. Accumulate the ordinary
   round-to-nearest `T = fl(T + fl(w*w))`, starting at zero. This pass uses
   no per-element scaling, outward additions or divisions. The dummy branch
   prevents rejected tiny/huge/nonfinite values from generating products.
3. If ANY predicate failed, discard T entirely and run the unchanged safe
   norm below on the original values. No partial norm or zero sentinel from
   a failed guard can authorize a certificate. If all predicates passed and
   T=0, all values were zero, so return exact0.
4. Otherwise take d=next_down(fl(1-g)), form Qhi=up(T/d), then return
   `up(sqrt(Qhi))`. Each `up` uses the existing outward primitive; d is
   finite and positive. There is no rescaling step.

### Normal-range proof at the operation level

A nonzero passing input lies in[2^-240,2^240], so its exact square lies
in[2^-480,2^480]. Products and every nonzero partial sum are normal; sums of
zero terms are exact. For every prefix, positivity and n<2^50 bound the ideal
sum by2^530. At most2*n rounding factors bound each prefix by
`2^530/(1-m*u) < 2^531`. Induction proves that an overflowing partial sum
cannot be the first violation of this bound. No product or partial-sum
underflow occurs. Thus a relative-error-only argument is valid INSIDE this
guard, unlike the withdrawn arbitrary-magnitude unscaled recipe.

Let Q=sum_i v_i^2 in real arithmetic. At most2*n rounded product/addition
nodes affect the sum. Unfolding their multiplicative errors gives
`|T-Q|<=gamma_m*Q`; no additive underflow term is needed because all nonzero
products/sums are normal. An FMA contracts nodes and remains covered by m.
The operation-count validity guard implies gamma_m<1/3; its outward
implementation still gives g<1/2. Therefore

    Q <= T/(1-gamma_m) <= T/(1-g) <= T/d <= Qhi.

The lower denominator d accounts for rounding in constructing1-g. Outward
division and correctly rounded/outward sqrt enclose sqrt(Q)=||v||. The bounds
above leave Qhi<2^532 and the result<2^266; the smallest positive squared sum
is at least2^-480. Thus final division/sqrt also stay finite/normal, with a
large margin at both ends. The zero case is handled separately. An unexpected
enclosure failure still selects the safe path.

### Which quantity is guarded

The guard is INSIDE `norm_fast`, immediately on the value returned by its
`at(i)` closure and before that value is squared. It is not a precheck on raw
X reused for a later transform. Concretely:

- Raw score metadata's x norm guards x_i itself.
- Raw score metadata's coefficient norm guards0 for x_i==c, otherwise
  `next_up(abs(fl(x_i-c)))`, which encloses the exact difference.
- Centered-fallback metadata guards the actual rounded coefficient
  `fl(fl(x_i-c)/s)` used by the stock score.
- A residual norm guards r_i.
- A residual-change norm guards0 for a_i==b_i, otherwise
  `next_up(abs(fl(a_i-b_i)))`.

A raw value inside the guard can yield an outside centered difference; that
transformed norm must use the safe path. Conversely large raw values outside
the guard may yield small accepted centered coefficients. The branch is chosen
from the actual squared quantity in each case. The native transformed-input
test exercises BOTH directions. Rounded difference enclosures remain norm
inputs; unrounded real differences are never substituted into floating code.

### Exact differences and coverage

The fast path is applied to the same norm inputs as before (columns,
coefficient/difference vectors, residuals). In particular an EXACT equality
x_i==center now supplies0, matching the existing difference-norm specification.
Previously that closure supplied next_up(0)=eta, a valid but unnecessary
padding that forced fast-path rejection. Nonzero differences still supply
`next_up(abs(fl(x_i-center)))`; their norm therefore bounds the real vector
of exact differences. No rounded difference is substituted for an exact one.

`test_guarded_norm.py` evaluates fast, safe and dispatched results separately.
It uses exact Fraction sums of squares at both signs of A/B and each adjacent
float, including zeros, mixed ranges, boundary failures in different positions,
long repeated vectors, cancellation-generated difference inputs through the
score audit, and seeded nearby values. The old1e-200 counterexample still
selects the safe path. Native tests check endpoint routing, zero/empty norms,
the underflow fallback and integer-operation-count failure before any read.
The tests use mathematical expectations, not coefficients generated by the
solver. These tests support the proof and do not replace independent review.

## Scaled safe norm (unchanged fallback)

For finite vector v, let m=max_i |v_i| (an exact selection). If m=0 the exact
norm is zero. Otherwise the code computes:

1. `t_i = up(|v_i|/m)`.
2. `p_i = up(t_i*t_i)`.
3. `S_0=0`, `S_(i+1)=up(S_i+p_i)`.
4. `q=up(m * up(sqrt(S_n)))`.

Each ratio upper-bounds the exact normalized component. Each p_i bounds its
square. S bounds the sum of those squares by induction. Monotonicity of sqrt
and multiplication proves q >= ||v||. Importantly, a tiny squared term widens
to at least eta BEFORE sqrt; the final multiplication by a tiny m is also
rounded outward. If the upper norm overflows, no certificate is returned.
No gamma approximation or unscaled squaring is used in this norm.

For exact differences of two float vectors a,b, use zero if their components
are equal, otherwise `next_up(abs(fl(a_i-b_i)))` as the norm input. This
upper-bounds the real absolute difference; an overflowing subtraction causes
fallback. The same method bounds exact `|x_i-c|`. Column buffers are immutable
throughout the two norm passes; no cursor shared between workers is introduced.

## The exact linear model being bounded

The goal is to preserve STOCK FLOATING POINT PREDICATES, not substitute an
ideal mathematical gradient. Centers, scales and fitted-row counts are the
exact float constants the stock solver uses.

For stock's raw-moment score, define D as the actual float result of `s*n_fit`.
It must be finite and positive. Define the real linear functional

    Q(r) = [sum_i x_i*r_i - c*sum_i r_i] / D.

Its coefficient norm is ||x-c||/D, rounded upward. D is treated as that fixed
float constant, so no unaccounted denominator-construction error remains. This
also handles masks: unused residual entries are exactly zero, and the whole
column norm is a valid (possibly loose) bound. Repeated gathers retain all
repeated entries in their original lane positions.

For stock's centered fallback (`abs(c)>100*s` in the SAME float arithmetic),
let zhat_i be the actual rounded `(x_i-c)/s`, computed with the same scalar
operations as stock. The linear functional is `sum_i zhat_i*r_i / n_fit`.
Its coefficient norm is ||zhat||/n_fit. Coefficient construction is part of the
fixed coefficient definition, not silently replaced by ideal real z. Nonfinite
zhat makes the column unsupported and forces an exact read.

## Dot and residual-sum errors, with underflow

For a sum of products or a summation expression with at most m rounded nodes,
unfold each node's error `delta*t + epsilon`, with |delta|<=u and
|epsilon|<=eta. Each leaf coefficient accumulates at most m factors (1+delta).
For m*u<1, `(1+u)^m <= 1/(1-m*u)`. Thus the error is bounded by

    gamma_m * M + m*eta/(1-m*u),  gamma_m=m*u/(1-m*u),

where M bounds the sum of absolute exact input terms. Each additive epsilon
is propagated through no more than m later factors; there are no more than m
such nodes. The formula covers gradual underflow in products AND additions.
An FMA uses fewer rounded nodes and is covered by the same conservative count.

For stock's eight-lane dot and fixed fold/tail, m=2*n+64 exceeds the number of
products/additions on every reduction path and the total additive-error nodes.
For both left-to-right compatibility sums and the default fixed residual-sum
tree, m=n suffices. The default visits logical 2,048-row blocks, accumulates
complete groups in eight lanes, combines those lanes in a fixed binary tree,
adds the scalar tail, then adds block totals in order. Storage runs never
reset an accumulator or realign logical blocks.

To count rounded nodes, prune explicit zero initializers and empty lane/tail
sums: adding those floating-point zeros is exact. Retain computed zeros with
their contributing residual leaves. Every remaining node joins disjoint sets
of residual leaves. With n leaves,
there are at most n-1 such binary additions in total and on any path. The
zero initializers, empty lanes and tail addends do not increase that count;
signed zero does not change an absolute-error enclosure. Thus the existing
m=n gamma and additive-underflow bounds cover this order too. The argument
is about adding the actual rounded updated residuals, never an analytic
recurrence for their sum. The count uses the actual row workspace length,
including masked zeros and gathered repeats. Counts are checked for
integer overflow and exact conversion (<=2^53); m*u>=1/4 is rejected.
`gamma_m` uses an upper numerator and a LOWER `1-m*u` denominator. The
underflow term is formed with outward arithmetic, not allowed to vanish.

M_dot=||x||*||r|| and M_sum=sqrt(n)*||r|| follow from Cauchy and are evaluated
outward. The centered fallback substitutes ||zhat|| for ||x||. The routine
requires magnitudes below MAX/16 and checks every subsequent enclosure for
finiteness. Together with these absolute-term bounds this prevents overflow
of the stock dot/sum intermediates in every certified evaluation. A case
where cancellations make the stock score finite but the magnitude bounds
are too large is deliberately unsupported and evaluated exactly.

## Raw score: individual operations

Write Mx=M_dot and Mr=M_sum. Let Ed be the dot error above and Es the residual-sum
error. All arithmetic on the right sides below is outward rounded.

1. `vhat=fl(c*shat)` where shat is the selected deterministic residual sum:

       Ev = |c|*Es + u*|c|*(Mr+Es) + eta

   bounds |vhat-c*sum(r)|. The first term carries residual-sum error; the latter
   terms bound this multiplication's own rounding/underflow.

2. `that=fl(dhat-vhat)` where dhat is stock dot:

       Et = Ed+Ev + u*(Mx+Ed+|c|*Mr+Ev) + eta.

   The absolute subtraction result is bounded by the sum of its operand
   magnitudes. This includes its own rounding and underflow.

3. `qhat=fl(that/D)`:

       Eq = Et/D + u*(Mx+|c|*Mr+Et)/D + eta.

   The numerator error is divided by D BEFORE adding the division's own
   rounding term. Thus additive underflow errors are amplified by tiny D;
   they are not left as an incorrectly dimensioned unscaled padding.

If any intermediate magnitude/error expression overflows or is nonfinite,
return None. The same bound also covers contracting the multiply/subtract
into FMA: the separate-operation enclosure is more conservative.

For the centered fallback there is only a dot then division:

    Eq = Ed/n_fit + u*(M_dot+Ed)/n_fit + eta.

This is implemented separately; raw mean/scale cancellation is not used as
an approximation to the fallback.

## Score-change enclosure and path length

For last EXACT score qa at actual residual ra and current stock score qhat(r):

    |qhat(r)-qa| <= coefficient_norm*||r-ra|| + Eq(r)+Eq(ra).

Both score errors are necessary. A certified skip does NOT advance qa or its
residual-distance anchor. All error products/additions use outward primitives.

Instead of retaining a residual per feature, measure upper increments
`d_k >= ||r_k-r_(k-1)||` whenever the residual version changes before a scan.
Maintain lower/upper endpoints L_k,H_k for the SAME sum of these upper d's.
A feature stores L_a at its last exact score. The outward difference H_k-L_a
bounds the sum of increments since that anchor, hence ||r_k-ra|| by triangle
inequality. Subtracting two upper prefix endpoints would be invalid.
This uses one previous residual buffer and O(p) scalar metadata. Prefix
nonfiniteness invalidates the cache; it is never converted into zero distance.
The exact same residual epoch can reuse its exact scalar predicate directly.

## Predicate/epoch rules

With an inactive feature beta=0 and FINITE l1/l2, certify a KKT nonviolation
only if the outward upper absolute score <= stock l1. The strict violation
predicate is unchanged. Activation reads and all active-coordinate reads stay
exact. Infinite/NaN thresholds or products force exact scoring.

A candidate skip must also resolve the NEXT strong-rule comparison. Its exact
threshold expression remains `(2*next_lambda-current_lambda)*alpha*factor`,
in that order, just as stock. The certificate stores the decision, stock score
epoch, lambda index, lambda bits and threshold bits. Before using it at the next
lambda, all identities are asserted. A skipped score is NEVER approximated
or copied into `st.scores`; enabled mode leaves its previous exact value there.
The strong-rule loop uses the separate certificate when appropriate. If no
valid certificate exists, the score must still be exact at the represented
stock epoch; a stale exact scalar is not silently used for another epoch.

Prefit executes unchanged, and the cache initializes only afterward. `dfmax`
exits precede the next strong-rule use; `max_iter` follows the original control
flow and scans. Shadow mode forces all original reads, checking every candidate
KKT decision and next-strong decision against the actual stock score while
maintaining the candidate's own last-exact anchors. Enabled mode performs the
skips and is compared bitwise with stock, including activation/residual traces.

## Validation and scope

The native test executable embeds Python's arbitrary-precision Fractions to
check the actual compiled bound operations. Retained cases include nonzero
raw centers, ratios around the centered-score threshold, subnormals, large
exponents, cancellation, tiny scales/products, overflow, NaN/Inf, and adjacent
thresholds. The 1e-200 counterexample must safely fall back. These checks
support the derivation; they do not replace its arithmetic assumptions.

Separate isolated native processes force scalar, 128/256/512-bit emulated
widths and available CPU backends, with exact/shadow/enabled paths and fixed
thread counts. They compare numeric outputs, warnings and recorded trajectory,
and explicitly compare certificate metadata. Injected prefix failures must
fall back; stale epochs and threshold identities must fail before use.

The certificate is Gaussian only. It makes no claim for nonlinear updates,
mutating X or changing the floating-point environment during a fit.

## Residual-sum order and score epochs

Default Gaussian mode fuses each unchanged per-row update with the fixed
LANES/CHUNK tree above. Compatibility mode keeps the serial sum. The initial
score scan, scans after prefit, and cached sums after coordinate updates all
use their mode's same reduction. A score epoch therefore never names two
floating-point functions. The access-path optimization does not reconstruct
residuals, change their epoch increments, or write approximate scores.

The retained Fraction audits exercise the actual default tree; native tests
also compare shadow/enabled metadata and off/enabled numeric outputs, warnings
and recorded trajectory, across forced widths and threads. Unsupported FP
environments keep their existing exact-read fallback. No certificate formula,
range guard or threshold arithmetic changes with this summation order.

The native optimized Linux x86-64 tests also audit the executable's symbols
with `nm` (from binutils): SIMD arithmetic must not call standalone AVX math
helpers. Those helpers exposed an outlined mapped-column visitor that spilled
vectors through the stack. The direct block loop keeps arithmetic inside the
SIMD dispatch body. Functional width/storage tests run on every native CI
platform; the symbol audit is specific to optimized ELF x86-64 builds and
uses no wall-clock threshold.
