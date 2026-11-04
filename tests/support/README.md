# Test support

- `rcompare.py`: data/case construction, pinned external R results, live R
  plumbing and comparison policy. The R script and fixtures remain under
  `tests/r/` and `tests/fixtures/`.
- `reference.py`: a plain numpy implementation of R's algorithm, checked
  against R output. Keep its arithmetic independent of the certificate module
  and the production kernels.
- `reference_path.py`: the reference's fit in the attributes the certificates
  read (`lambdas`, `intercept`, `coef`, ...), so they compare the solver with
  the reference; it refuses a path cut short by saturation or `max_iter`.
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

## Kernel coverage after numba removal

The deleted `test_kernels.py` called the removed numba implementation, not the
production Rust solver. Its independent formulas now exercise Rust directly
under `make test-rust`; they are not replaced by agreement with another run
of pybiglasso.

| Former Python probe | Native coverage |
|---|---|
| Standardization, initial crossproducts, raw/used indices | `kernel_tests.rs`: `standardization_and_initial_scores_use_selected_rows_and_raw_column_map` |
| Moments, shifts, large-offset cancellation | `tests.rs`: `passes_*`; `kernel_tests.rs`: `centered_moments_and_score_avoid_large_offset_cancellation` |
| Coordinate updates, both enet forms, invalid/cancelled curvature | `kernel_tests.rs`: `coordinate_sweep_*`, `curvature_fallback_handles_positive_and_zero_cancelled_moments` |
| KKT scores, strict threshold neighbors, disabled re-entry scans | `kernel_tests.rs`: `kkt_scans_*`, `independent_optimality_check_detects_missing_reentry_scans` |
| Intercept normal equation, working state, strict clamp boundaries | `tests.rs`: `intercept_solves_weighted_normal_equation_with_shared_weight_sum`, `irls_matches_row_formulas_and_sequential_loss` |
| Stable final loss, row selection | `kernel_tests.rs`: `final_loss_is_stable_at_extreme_logits_and_skips_left_out_rows` |

Native checked-index negative controls accompany the normal row/layout
matrix. `test_rust_kernel_controls.py` separately checks that the Python
bounds setting reaches the extension. The old numba pool-restoration and
workqueue-fallback tests disappear with that pool and fallback; Rust thread
identity and concurrent CV tests remain.
