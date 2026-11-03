# Raw trace protocol (experiment only)

Each record is little-endian, concatenated without padding:

- Six u64: kind, lambda index, residual length n, varying-feature count p,
  original feature count q, residual version.
- Five f64: lambda, cached residual sum, upper cumulative distance prefix,
  lower prefix, residual norm upper bound.
- f64 arrays: residual[n], stock score storage[p], beta[p], center[q], scale[q].
- u64 col_map[p].
- u8 arrays: active[p], strong[p], candidate-skip mask[p].
- f64 arrays: last candidate radius[p], candidate's last exact anchor score[p].
- u64 arrays: anchor epoch[p], represented stock-score epoch[p], certified next
  lambda index[p], next lambda float bits[p], next cutoff float bits[p].
- u8 next-decision[p]: 0 absent, 1 false, 2 true.

Kinds: 0 cache initialized after prefit; 1 after strong-rule flags; 2/3 before/
after strong-set KKT scan; 4/5 before/after rest-set scan; 6 recorded lambda;
7 after original prefit/grid setup; 9 prefit sweep. Lambda index usize::MAX
indicates prefit. Off-mode metadata arrays are zeros. Shadow/enabled identity
checks compare residual bytes, beta bytes, flags, scan order, lambda and cached
residual-sum bits, excluding intentionally stale exact score storage in enabled
mode and its separate cache metadata. Source and original base SHA accompany
captures; `check_paths.py::trace_frames` is the reader.

Traces are disabled for intended timing runs. No timing claims use capture runs.
