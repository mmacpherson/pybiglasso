# Logistic gap certificates

`duality_gap(Z, y, theta, lam, alpha, penalty_factor)` returns an upper bound
on objective error in **nats per observation**. Here `Z=[1,Xs]` and
`theta=[intercept,beta]` use standardized columns. The domain is finite small
test arrays, binary labels, positive lambda, `0 < alpha <= 1`, and strictly
positive penalty factors. A zero-penalty column would impose another exact dual
equality; it is rejected rather than silently treated as penalized.

Let `a_j = lambda*alpha*pf_j` and `c_j = lambda*(1-alpha)*pf_j`. The primal is

```
P(theta) = mean(log(1+exp(Z theta)) - y*(Z theta))
           + sum(a_j*abs(beta_j) + c_j*beta_j**2/2).
```

Writing `u=(q-y)/n`, the dual constraints are `0 <= q <= 1` and `sum(u)=0`
(the unpenalized intercept). With `h(q)=q log(q)+(1-q)log(1-q)`, the dual is

```
D(u) = -mean(h(y+n*u)) - sum((abs(Xs_j' u)-a_j)_+**2/(2*c_j)).
```

The scalar conjugate follows by maximizing `s*b-a*abs(b)-c*b**2/2` on either
half-line. When `c=0` it is zero for `abs(s)<=a`, infinite otherwise; lasso
therefore requires the hard correlation constraints. Any feasible `u` gives
`D(u) <= P* <= P(theta)`, including at non-converged or clipped iterates.

The constructor starts at logistic probabilities, balances the positive and
negative residual masses by contracting them, and, for lasso, contracts all
residuals together to satisfy the correlation constraints. These operations
use `Fraction`: balance, box constraints, scores and penalty constants are
exact for the floating-point inputs. No stationarity tolerance substitutes for
dual feasibility. The reference algorithm and solver kernels are not called.

Primal logits/penalties and dual conjugates are evaluated as exact rationals.
For the remaining logarithms/exponentials, 60-digit Decimal algebra is directed
upward for the primal and downward for the dual. Decimal's correctly rounded
`exp`, `ln` and `sqrt` are enclosed by the next representable number. Final
float conversion and subtraction are outward. Precision changes tightness,
not the direction of the bound; there is no chosen absolute gap floor. This
is intentionally more expensive than a production stopping check.

`gap_coefficient_distance(gap, lam, alpha, pf)` bounds coefficient distance to
the optimum by `sqrt(2*gap/min(c_j))`, when `alpha < 1`. Convexity at the optimum
and the ridge quadratic give `P(theta)-P* >= sum(c_j*(beta-beta*)_j**2)/2`.
This holds when `p>n`; it does **not** bound the unpenalized intercept. Sum two
radii to compare coefficients from two fits of the same objective. Pure lasso
has no such coefficient guarantee and may have multiple coefficient solutions.

`gap_safe_zeros(...)` certifies zeros at every optimum. Binary entropy has
curvature at least four, so the dual in `u` has concavity at least `4*n`:
`||u-u*|| <= sqrt(gap/(2*n))`. Cauchy–Schwarz then bounds each optimum score by
`abs(Xs_j'u) + ||Xs_j||*radius`. A strict bound below `a_j` forces a zero by
KKT, for lasso and elastic net. Roots, norms and the final comparison bounds
are rounded outward. This says nothing about literal zeros in an arbitrary
approximate iterate; tests asserting those separately cap its KKT residual and
check its own score margin.

Sources: [Ndiaye et al. (2017), Theorem 6 and Section 5.4](https://jmlr.org/papers/v18/16-577.html)
for logistic conjugacy and gap-safe spheres;
[celer, test_celer_path_logreg](https://github.com/mathurinm/celer/blob/main/celer/tests/test_logreg.py)
for a path-level gap accuracy test (its normalization/intercept differ);
[Python Decimal guarantees](https://docs.python.org/3/library/decimal.html)
for the directed evaluation.
