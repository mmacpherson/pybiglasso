# Test support

- `rcompare.py`: data/case construction, pinned external R results, live R
  plumbing and comparison policy. The R script and fixtures remain under
  `tests/r/` and `tests/fixtures/`.
- `reference.py`: a plain numpy implementation of R's algorithm, checked
  against R output. Keep its arithmetic independent of the certificate module
  and the production kernels.
- `certificates.py`: first-principles standardization, mean-logistic objective,
  subgradient residuals, segment curvature, distance and support bounds. All
  coefficients in a certificate are on standardized columns, with an
  unpenalized intercept. Callers retain their accuracy caps and domain checks;
  a small residual is not promised under clipping, saturation, non-convergence
  or R-compatible elastic-net updates. The eigensolver margin is a practical
  floating-point allowance, not interval arithmetic.
  The separate [logistic gap certificate](duality-gap.md) uses exact dual
  feasibility and directed objective bounds; it also works on wide problems.
- `paths.py`: `assert_bitwise` checks shape, dtype and bytes (including signed
  zero); prefixes are explicit through `length`. `assert_same_trajectory`
  checks exact iteration/rejection counts and the existing numerical policy
  for rounded values. Neither asserts that different optimization trajectories
  must agree. Compare independently capped solutions with a distance bound
  instead.

These helpers may copy small test designs. The production no-copy contract is
tested separately; no production solver imports this package. Analytic examples
and negative controls for the helpers live in `tests/test_support.py`; the
curvature counterexamples remain in `tests/test_invariances.py`.
