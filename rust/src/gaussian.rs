//! R biglasso's Gaussian SSR path, sharing column access and fixed-order scores
//! with the logistic solver. Only row/state/path buffers are allocated; X is
//! always borrowed and read one column at a time.
//!
//! Each lambda chooses a strong set (`choose_strong`), then alternates
//! coordinate sweeps over the active set (`gaussian_sweep`) with KKT scans of
//! the columns outside it (`gaussian_scan`) until a scan finds no violation.
//! In default mode a score cache lets a scan skip reading an inactive column
//! whose KKT and next strong-rule decisions a certificate already proves; see
//! docs/gaussian-screening.md. Native-test instrumentation is in
//! gaussian_probe.rs.

use super::gaussian_bound as bound;
use super::*;

#[path = "gaussian_probe.rs"]
mod probe;
use probe::{Point, Probe, ScanMode, trace};

struct GaussianState {
    residual: Rows,
    // A fresh reduction of actual residuals, never an analytic running total.
    // Compatibility mode uses seqsum; default mode uses fixed LANES/CHUNK sums.
    residual_sum: f64,
    beta: Vec<f64>,
    active: Vec<bool>,
    strong: Vec<bool>,
    /// Each column's last exact score. With the cache, a skipped column keeps
    /// its earlier value here; it is never replaced by an approximation.
    scores: Vec<f64>,
    /// The residual's epoch: incremented whenever a coefficient changes
    version: u64,
    /// The lambda being fit; usize::MAX during the unpenalized prefit
    il: usize,
}

/// A skipped column's strong-rule decision for the next lambda, certified by
/// the same scan that skipped it. It is valid only at that lambda, with that
/// threshold, for the score at that epoch; the strong rule asserts all three.
#[derive(Clone, Copy)]
struct NextStrong {
    /// Whether the column is in the next lambda's strong set
    decision: bool,
    /// The residual epoch the certificate describes
    epoch: u64,
    il: usize,
    lambda_bits: u64,
    /// The strong rule's threshold, (2 * next - lambda) * alpha * factor
    cutoff_bits: u64,
}

/// A column's last exact score, and what is needed to bound how far its
/// score can have moved since.
#[derive(Clone, Copy)]
struct Anchor {
    score: f64,
    /// The epoch at which `score` was read
    epoch: u64,
    /// The latest epoch at which the column's KKT condition was settled: by
    /// reading its score (then equal to `epoch`) or by a certified skip (then
    /// later). A skip never moves `score`, `epoch`, `prefix_lo` or `rnorm`.
    represented_epoch: u64,
    /// ScoreCache::lo when `score` was read
    prefix_lo: f64,
    /// ScoreCache::rnorm when `score` was read
    rnorm: f64,
    /// Written only by a skip, and only for an inactive column; taken by the
    /// next lambda's strong rule. Its epoch is always `represented_epoch`.
    next: Option<NextStrong>,
}

/// The score cache (default mode only). It tracks a bound on how far the
/// residual has moved since each column's anchor, through a running sum of
/// upper bounds on the residual's step norms: `hi` and `lo` enclose that sum
/// from above and below, so `hi - anchor.prefix_lo` bounds the distance from
/// the anchor's residual to the current one.
struct ScoreCache {
    /// Each column's bound coefficients; None where its scores can't be bounded
    meta: Vec<Option<bound::Meta>>,
    anchors: Vec<Anchor>,
    /// The residual at `version`
    previous: Vec<f64>,
    /// The epoch `previous`, `hi`, `lo` and `rnorm` describe. Left behind once
    /// the cache is invalid.
    version: u64,
    hi: f64,
    lo: f64,
    /// The residual's norm at `version`
    rnorm: f64,
    /// False once the arithmetic is unsupported or a bound is not finite;
    /// every scan then reads every column it considers.
    valid: bool,
    probe: Probe,
}

/// What one scan does with column j, decided before any column is read.
#[derive(Clone, Copy, Default)]
struct Plan {
    /// The scan considers j: an inactive strong column in a strong-set
    /// scan, a column outside the strong set in a full scan
    selected: bool,
    /// A certificate proves j satisfies KKT here, so its score isn't read
    skip: bool,
    /// With `skip`, j's strong-rule decision at the next lambda (none at the
    /// last lambda)
    next: Option<NextStrong>,
    /// The bound on |score - anchor score| the certificate used; 0 when the
    /// residual hasn't changed since the anchor
    radius: f64,
}

impl ScoreCache {
    /// Advances the residual-distance prefix to the current epoch, from the
    /// change since `previous`; any nonfinite step invalidates the cache.
    fn catch_up(&mut self, residual: &[f64], version: u64) {
        if self.version == version || !self.valid {
            return;
        }
        let step = bound::difference_norm(residual, &self.previous);
        let advanced = step.and_then(|step| bound::advance(self.hi, self.lo, step));
        if let Some(((hi, lo), rnorm)) = advanced.zip(bound::norm(residual.len(), |i| residual[i])) {
            self.hi = hi;
            self.lo = lo;
            self.rnorm = rnorm;
            self.previous.copy_from_slice(residual);
        } else {
            self.valid = false;
        }
        self.version = version;
    }

    /// Inactive column j's strong-rule decision at lambda il, from the
    /// certificate its last skip left, or else from its exact score.
    ///
    /// The assertions cannot fire: a skip writes `next` together with
    /// `represented_epoch`, and a read clears `next` as it sets `epoch` and
    /// `represented_epoch` alike. A scan at lambda il - 1 writes `next` for il,
    /// and every strong rule takes the `next` of every inactive column, so
    /// any that remains is from lambda il - 1, whose threshold is this one's
    /// expression with the same operands. Without one, the last scan read
    /// the column or skipped it at the last lambda, after which no strong
    /// rule runs; so `scores[j]` is still exact.
    fn strong_decision(&mut self, j: usize, il: usize, lam: f64, cutoff: f64, exact_decision: bool) -> bool {
        self.probe.corrupt_certificate(&mut self.anchors[j]);
        let anchor = &mut self.anchors[j];
        let Some(next) = anchor.next.take() else {
            assert_eq!(anchor.epoch, anchor.represented_epoch, "stale exact score without certificate");
            return exact_decision;
        };
        assert_eq!(next.epoch, anchor.represented_epoch, "score epoch mismatch");
        assert_eq!(
            (next.il, next.lambda_bits, next.cutoff_bits),
            (il, lam.to_bits(), cutoff.to_bits()),
            "next threshold mismatch"
        );
        if self.probe.shadow() {
            assert_eq!(next.decision, exact_decision, "shadow SSR decision mismatch il={} j={}", il, j);
        }
        self.probe.certificate_used();
        if self.probe.shadow() { exact_decision } else { next.decision }
    }
}

// Differs from R biglasso: default mode reduces actual updated residuals in
// fixed LANES/CHUNK order so SIMD can share their additions. Compatibility
// mode keeps R's serial sum; the per-row update expressions are unchanged.
pub(super) fn blocked_sum(values: &[f64]) -> f64 {
    let mut total = 0.;
    for chunk in values.chunks(CHUNK) {
        let body = chunk.len() / LANES * LANES;
        let mut lanes = [0.; LANES];
        for i in 0..body {
            lanes[i % LANES] += chunk[i];
        }
        let tail = seqsum(&chunk[body..]);
        total += combine(&lanes) + tail;
    }
    total
}

/// Default mode's residual update for a coefficient change of `shift`, and
/// the updated residual's sum in blocked_sum's order, in one pass.
fn update_blocked<C: Col>(
    column: &C,
    residual: &mut [f64],
    mask: Option<&[f64]>,
    shift: f64,
    center: f64,
    scale: f64,
) -> f64 {
    struct Op<'a, C> {
        column: &'a C,
        residual: &'a mut [f64],
        mask: Option<&'a [f64]>,
        shift: f64,
        center: f64,
        scale: f64,
    }
    impl<C: Col> WithSimd for Op<'_, C> {
        type Output = f64;
        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> f64 {
            let Op { column, residual, mask, shift, center, scale } = self;
            let mut total = 0.;
            for (k, r) in residual.chunks_mut(CHUNK).enumerate() {
                let offset = k * CHUNK;
                let c = Span { col: column, i0: offset, n: r.len() };
                let body = r.len() / LANES * LANES;
                let mut acc = [V(simd, simd.splat_f64s(0.)); LANES];
                let mut scalar_lanes = [0.; LANES];
                if let Some(mask) = mask {
                    // Never read excluded X: selected-row nonfinite policy and
                    // masked residual zeros must remain exactly as in stock.
                    for i in 0..body {
                        if mask[offset + i] != 0. {
                            r[i] -= shift * ((c.at(i) - center) / scale);
                        }
                        scalar_lanes[i % LANES] += r[i];
                    }
                } else {
                    let mut start = 0;
                    while start < body {
                        let (run, end) = c.run(start, body);
                        // Keep SIMD arithmetic in this target-feature body.
                        // An outlined visitor makes the mapped-column build
                        // call individual vector intrinsics through the stack.
                        let bytes = run.packed_bytes();
                        let mut hint = 0;
                        for i in (0..end - start).step_by(LANES) {
                            let decoded;
                            let block = if let Some(bytes) = bytes {
                                &BOOL_BLOCKS[usize::from(bytes[i / LANES])]
                            } else {
                                decoded = load(&run, i, &mut hint);
                                &decoded
                            };
                            let x = regs_of::<S>(block);
                            let (rv, _) = S::as_mut_simd_f64s(&mut r[start + i..start + i + LANES]);
                            for m in 0..regs::<S>() {
                                let centered = simd.sub_f64s(x[m], simd.splat_f64s(center));
                                let standardized = simd.div_f64s(centered, simd.splat_f64s(scale));
                                let change = simd.mul_f64s(simd.splat_f64s(shift), standardized);
                                rv[m] = simd.sub_f64s(rv[m], change);
                                acc[m].1 = simd.add_f64s(acc[m].1, rv[m]);
                            }
                        }
                        start = end;
                    }
                }
                let mut tail = 0.;
                for i in body..r.len() {
                    if mask.is_none_or(|m| m[offset + i] != 0.) {
                        r[i] -= shift * ((c.at(i) - center) / scale);
                    }
                    tail += r[i];
                }
                total += if mask.is_some() { combine(&scalar_lanes) + tail } else { fold(acc) + tail };
            }
            total
        }
    }
    dispatch(Op { column, residual, mask, shift, center, scale })
}

impl<D: Design> Fit<'_, D> {
    /// One coordinate-descent pass over the active columns at lambda `lam`;
    /// returns the largest squared coefficient change.
    fn gaussian_sweep(&self, path: &Path, st: &mut GaussianState, lam: f64) -> f64 {
        let p = self.params;
        let mut largest: f64 = 0.0;
        for j in 0..st.beta.len() {
            if !st.active[j] {
                continue;
            }
            let jj = path.col_map[j];
            let column = self.x.col(jj);
            let (center, scale) = (path.center[jj], path.scale[jj]);
            let score = standardized_score(&column, &st.residual, st.residual_sum, center, scale, self.n_fit as f64)
                + st.beta[j];
            st.scores[j] = score;
            // Differs from R biglasso, which multiplies lambda by the penalty
            // factor first, as the binomial solver here does.
            let l1 = lam * p.alpha * p.penalty_factor[jj];
            let l2 = lam * (1.0 - p.alpha) * p.penalty_factor[jj];
            let value = if score.abs() <= l1 { 0.0 } else { score.signum() * (score.abs() - l1) / (1.0 + l2) };
            let shift = value - st.beta[j];
            if shift == 0.0 {
                continue;
            }
            st.version += 1;
            largest = largest.max(shift * shift);
            st.residual_sum = if p.r_compatible {
                self.update_in_row_order(&column, &mut st.residual, shift, center, scale)
            } else {
                update_blocked(&column, &mut st.residual, self.mask, shift, center, scale)
            };
            st.beta[j] = value;
        }
        if st.il == usize::MAX {
            trace(path, st, None, Point::PrefitSweep, lam);
        }
        largest
    }

    /// Compatibility mode's residual update for a coefficient change of
    /// `shift`, and the updated residual's sum in seqsum's left-to-right
    /// order, masked zeros included, in the same pass.
    #[inline(always)]
    fn update_in_row_order<C: Col>(
        &self,
        column: &C,
        residual: &mut [f64],
        shift: f64,
        center: f64,
        scale: f64,
    ) -> f64 {
        // Differs from R biglasso, which groups the update as
        // (shift * (x - c)) / scale. The two can differ in the low bits;
        // the R comparison checks agreement to rtol=1e-9.
        let mut sum = 0.0;
        let mut start = 0;
        let mut hint = 0;
        while start < residual.len() {
            let (run, end) = column.run(start, residual.len());
            // A byte lookup supplies eight values together; every
            // update and sum still happens in the original row order.
            // Masked fits retain their guarded scalar reads.
            let body = if self.mask.is_none() { (end - start) / LANES * LANES } else { 0 };
            for_each_block(&run, body, |i, block| {
                // Keep the serial accumulator local to this block.
                // Capturing it in the visitor otherwise makes LLVM
                // store/reload it after every lane.
                let mut block_sum = sum;
                for (residual, &value) in residual[start + i..start + i + LANES].iter_mut().zip(&block.0) {
                    *residual -= shift * ((value - center) / scale);
                    block_sum += *residual;
                }
                sum = block_sum;
            });
            for (i, residual) in residual[start + body..end].iter_mut().enumerate() {
                let i = body + i;
                if self.fits(start + i) {
                    *residual -= shift * ((run.at_seq(i, &mut hint) - center) / scale);
                }
                sum += *residual;
            }
            start = end;
        }
        sum
    }

    /// KKT scan: the columns a strong-set scan (`strong_only`) or full scan
    /// considers that violate KKT at `lam` join the active and strong sets.
    /// Returns how many did.
    fn gaussian_scan(
        &self,
        path: &Path,
        st: &mut GaussianState,
        mut cache: Option<&mut ScoreCache>,
        lam: f64,
        strong_only: bool,
    ) -> usize {
        let (before, after) = if strong_only {
            (Point::StrongScan, Point::StrongScanDone)
        } else {
            (Point::FullScan, Point::FullScanDone)
        };
        trace(path, st, cache.as_deref(), before, lam);
        let violations = match cache.as_deref_mut() {
            None => self.scan_reading_all(path, st, lam, strong_only),
            Some(cache) => {
                cache.probe.invalidate_prefix(&mut cache.hi);
                cache.catch_up(&st.residual, st.version);
                let plans = self.plan_scan(path, st, cache, lam, strong_only);
                let scores = self.read_scores(path, st, cache, &plans);
                self.apply_scan(path, st, cache, lam, &plans, &scores)
            }
        };
        trace(path, st, cache.as_deref(), after, lam);
        violations
    }

    /// The scan without a cache: every column considered is read.
    fn scan_reading_all(&self, path: &Path, st: &mut GaussianState, lam: f64, strong_only: bool) -> usize {
        let sum = st.residual_sum;
        let p = self.params;
        let residual = &st.residual;
        let beta = &st.beta;
        st.scores
            .par_iter_mut()
            .zip(&mut st.active)
            .zip(&mut st.strong)
            .enumerate()
            .map(|(j, ((score, active), strong))| {
                if if strong_only { *active || !*strong } else { *strong } {
                    return 0;
                }
                let jj = path.col_map[j];
                *score = standardized_score(
                    &self.x.col(jj),
                    residual,
                    sum,
                    path.center[jj],
                    path.scale[jj],
                    self.n_fit as f64,
                );
                // Differs from R biglasso in the order of the penalty product,
                // as in the sweep.
                let l1 = lam * p.alpha * p.penalty_factor[jj];
                let l2 = lam * (1.0 - p.alpha) * p.penalty_factor[jj];
                if (*score - beta[j] * l2).abs() > l1 {
                    *active = true;
                    *strong = true;
                    1
                } else {
                    0
                }
            })
            .sum()
    }

    /// Each column's plan for this scan. A column is skipped only if it is
    /// inactive and zero, its thresholds are finite, its anchor's score with
    /// the certified radius proves no KKT violation, and (before the last
    /// lambda) the same bound settles its next strong-rule decision.
    fn plan_scan(&self, path: &Path, st: &GaussianState, cache: &ScoreCache, lam: f64, strong_only: bool) -> Vec<Plan> {
        (0..st.beta.len())
            .into_par_iter()
            .map(|j| {
                if if strong_only { st.active[j] || !st.strong[j] } else { st.strong[j] } {
                    return Plan::default();
                }
                let mut plan = Plan { selected: true, ..Plan::default() };
                if st.active[j] || !cache.valid || !bound::arithmetic_ok() {
                    return plan;
                }
                let jj = path.col_map[j];
                let l1 = lam * self.params.alpha * self.params.penalty_factor[jj];
                let l2 = lam * (1. - self.params.alpha) * self.params.penalty_factor[jj];
                if !l1.is_finite() || !l2.is_finite() || st.beta[j] != 0. {
                    return plan;
                }
                let anchor = cache.anchors[j];
                let radius = if anchor.epoch == st.version {
                    Some(0.)
                } else {
                    cache.meta[j].and_then(|m| {
                        bound::distance(cache.hi, anchor.prefix_lo).and_then(|d| m.radius(d, anchor.rnorm, cache.rnorm))
                    })
                };
                let Some(radius) = radius else {
                    return plan;
                };
                plan.radius = radius;
                if bound::decide(anchor.score, radius, l1) != Some(false) {
                    return plan;
                }
                if st.il + 1 < path.lambda.len() {
                    let next = path.lambda[st.il + 1];
                    // EXACT stock order, including penalty-factor multiplication.
                    let cutoff = (2. * next - lam) * self.params.alpha * self.params.penalty_factor[jj];
                    let Some(decision) = bound::decide(anchor.score, radius, cutoff) else {
                        return plan;
                    };
                    plan.next = Some(NextStrong {
                        decision,
                        epoch: st.version,
                        il: st.il + 1,
                        lambda_bits: next.to_bits(),
                        cutoff_bits: cutoff.to_bits(),
                    });
                }
                plan.skip = true;
                plan
            })
            .collect()
    }

    /// The exact score of each column the plans select and don't skip (in
    /// shadow mode, of skipped ones too).
    fn read_scores(&self, path: &Path, st: &GaussianState, cache: &ScoreCache, plans: &[Plan]) -> Vec<Option<f64>> {
        plans
            .par_iter()
            .enumerate()
            .map(|(j, p)| {
                if !p.selected || (p.skip && !cache.probe.shadow()) {
                    return None;
                }
                let jj = path.col_map[j];
                Some(standardized_score(
                    &self.x.col(jj),
                    &st.residual,
                    st.residual_sum,
                    path.center[jj],
                    path.scale[jj],
                    self.n_fit as f64,
                ))
            })
            .collect()
    }

    /// Records each plan's outcome, in column order: a skip moves only the
    /// anchor's represented epoch and next certificate; a read re-anchors the
    /// column, and a violation activates it. Returns the violations.
    fn apply_scan(
        &self,
        path: &Path,
        st: &mut GaussianState,
        cache: &mut ScoreCache,
        lam: f64,
        plans: &[Plan],
        scores: &[Option<f64>],
    ) -> usize {
        let mut violations = 0;
        for (j, plan) in plans.iter().enumerate() {
            cache.probe.planned(j, plan);
            if !plan.selected {
                continue;
            }
            let jj = path.col_map[j];
            let l1 = lam * self.params.alpha * self.params.penalty_factor[jj];
            let l2 = lam * (1. - self.params.alpha) * self.params.penalty_factor[jj];
            if plan.skip {
                // Shadow mode: check the certificate against the exact score.
                if let Some(q) = scores[j] {
                    assert!(
                        q.is_finite() && (q - st.beta[j] * l2).abs() <= l1,
                        "shadow KKT mismatch il={} j={}",
                        st.il,
                        j
                    );
                    if let Some(next) = plan.next {
                        let cutoff = f64::from_bits(next.cutoff_bits);
                        assert_eq!(q.abs() > cutoff, next.decision, "shadow next SSR mismatch il={} j={}", st.il, j);
                    }
                    st.scores[j] = q;
                    cache.probe.exact_read();
                }
                cache.anchors[j].represented_epoch = st.version;
                cache.anchors[j].next = plan.next;
                // An enabled skip NEVER writes an approximate or anchor score.
                continue;
            }
            let q = scores[j].expect("mandatory exact score");
            cache.probe.exact_read();
            st.scores[j] = q;
            cache.anchors[j] = Anchor {
                score: q,
                epoch: st.version,
                represented_epoch: st.version,
                prefix_lo: cache.lo,
                rnorm: cache.rnorm,
                next: None,
            };
            if (q - st.beta[j] * l2).abs() > l1 {
                st.active[j] = true;
                st.strong[j] = true;
                violations += 1;
            }
        }
        violations
    }

    fn gaussian_loss(&self, st: &GaussianState) -> f64 {
        let mut rss = 0.0;
        for i in 0..st.residual.len() {
            if self.fits(i) {
                rss += st.residual[i] * st.residual[i];
            }
        }
        rss
    }

    /// Lambda il's strong set: R's sequential strong rule, from each column's
    /// score at the previous lambda, or from a skipped column's certificate.
    fn choose_strong(
        &self,
        st: &mut GaussianState,
        mut cache: Option<&mut ScoreCache>,
        il: usize,
        lam: f64,
        previous: f64,
        mult: &[f64],
    ) {
        let p = self.params;
        for j in 0..mult.len() {
            let cutoff = (2.0 * lam - previous) * p.alpha * mult[j];
            let exact_decision = st.scores[j].abs() > cutoff;
            st.strong[j] = match cache.as_deref_mut() {
                Some(cache) if !st.active[j] => cache.strong_decision(j, il, lam, cutoff, exact_decision),
                _ => exact_decision,
            };
        }
    }

    /// Fits lambda il: sweeps the active set to convergence, then scans the
    /// strong set and then every other column for KKT violations, repeating
    /// after any violation, all within max_iter sweeps.
    fn gaussian_lambda(
        &self,
        path: &mut Path,
        st: &mut GaussianState,
        mut cache: Option<&mut ScoreCache>,
        il: usize,
        lam: f64,
        tolerance: f64,
    ) {
        let p = self.params;
        while path.iters[il] < p.max_iter {
            while path.iters[il] < p.max_iter {
                while path.iters[il] < p.max_iter {
                    path.iters[il] += 1;
                    let change = self.gaussian_sweep(path, st, lam);
                    // Differs from R biglasso: its strict comparison against
                    // zero makes constant-response fits exhaust max_iter.
                    if change < tolerance || (!p.r_compatible && change == 0.0) {
                        break;
                    }
                }
                if self.gaussian_scan(path, st, cache.as_deref_mut(), lam, true) == 0 {
                    break;
                }
            }
            if self.gaussian_scan(path, st, cache.as_deref_mut(), lam, false) == 0 {
                path.deviance[il] = self.gaussian_loss(st);
                break;
            }
        }
        if path.iters[il] == p.max_iter {
            path.events.push(Event::MaxIter(il));
        }
        // Differs from R biglasso: R can leave loss at zero when the budget
        // ends with outstanding KKT violations; default mode reports RSS.
        if !p.r_compatible {
            path.deviance[il] = self.gaussian_loss(st);
        }
    }

    fn gaussian_record(&self, path: &mut Path, st: &GaussianState, il: usize) {
        let rows: Vec<usize> = (0..st.beta.len()).filter(|&j| st.beta[j] != 0.0).collect();
        path.coef_vals[il] = Some(rows.iter().map(|&j| st.beta[j]).collect());
        path.coef_rows[il] = Some(rows);
        path.intercept[il] = self.params.ybar;
    }

    /// Fits the unpenalized columns alone, from zero, before the automatic
    /// grid is set from the residual they leave.
    fn gaussian_prefit(&self, path: &Path, st: &mut GaussianState, mult: &[f64], tolerance: f64) -> Result<(), String> {
        for j in 0..mult.len() {
            st.active[j] = mult[j] == 0.0;
        }
        for _ in 0..self.params.max_iter {
            let change = self.gaussian_sweep(path, st, 0.0);
            if change < tolerance || change == 0.0 {
                return Ok(());
            }
        }
        Err("Unpenalized prefit failed to converge within max_iter; increase max_iter or supply lambdas explicitly."
            .into())
    }

    /// The score cache, from the scores and residual at the start of the
    /// lambda loop; None when native tests turn it off.
    fn gaussian_cache(&self, path: &Path, st: &GaussianState) -> Option<ScoreCache> {
        let mode = probe::scan_mode();
        if mode == ScanMode::Off {
            return None;
        }
        let pool_ok = if rayon::current_num_threads() == 1 {
            bound::arithmetic_ok()
        } else {
            rayon::broadcast(|_| bound::arithmetic_ok()).into_iter().all(|v| v)
        };
        let meta: Vec<_> = path
            .col_map
            .par_iter()
            .map(|&jj| {
                bound::Meta::make(&self.x.col(jj), path.center[jj], path.scale[jj], self.n_fit, |i| self.fits(i))
            })
            .collect();
        let rnorm = bound::norm(st.residual.len(), |i| st.residual[i]);
        let rn = rnorm.unwrap_or(0.);
        let anchor = |score| Anchor {
            score,
            epoch: st.version,
            represented_epoch: st.version,
            prefix_lo: 0.,
            rnorm: rn,
            next: None,
        };
        Some(ScoreCache {
            meta,
            anchors: st.scores.iter().map(|&score| anchor(score)).collect(),
            previous: st.residual.to_vec(),
            version: st.version,
            hi: 0.,
            lo: 0.,
            rnorm: rn,
            valid: pool_ok && rnorm.is_some(),
            probe: Probe::new(mode, st.beta.len()),
        })
    }

    pub(super) fn run_gaussian(&self, center: Vec<f64>, scale: Vec<f64>) -> Result<Path, String> {
        let p = self.params;
        let col_map: Vec<usize> = (0..scale.len()).filter(|&j| scale[j] > SCALE_MIN).collect();
        if col_map.is_empty() {
            return Err("X has no varying features after standardization.".into());
        }
        let mult: Vec<f64> = col_map.iter().map(|&j| p.penalty_factor[j]).collect();
        if p.lambdas.is_none() && !mult.iter().any(|&m| m > 0.0) {
            return Err(
                "An automatic lambda grid requires at least one varying penalized feature; supply lambdas explicitly."
                    .into(),
            );
        }
        let nl = p.nlambda;
        let mut path = Path {
            center,
            scale,
            col_map,
            lambda: vec![0.0; nl],
            intercept: vec![p.ybar; nl],
            coef_rows: vec![None; nl],
            coef_vals: vec![None; nl],
            iters: vec![0; nl],
            deviance: vec![0.0; nl],
            n_reject: vec![0; nl],
            events: vec![],
        };
        let mut residual = Rows::new(self.x.nrow(), 0.0);
        for i in 0..residual.len() {
            if self.fits(i) {
                residual[i] = self.y[i] - p.ybar;
            }
        }
        let nc = mult.len();
        let mut st = GaussianState {
            version: 0,
            il: usize::MAX,
            scores: self.compute_z(&path, &residual),
            residual_sum: self.score_sum(&residual),
            residual,
            beta: vec![0.0; nc],
            active: vec![false; nc],
            strong: vec![false; nc],
        };
        let tolerance = p.eps * p.null_deviance / self.n_fit as f64;
        let maximum = |scores: &[f64]| {
            let mut maximum: f64 = 0.0;
            for j in 0..nc {
                if mult[j] > 0.0 {
                    maximum = maximum.max(scores[j].abs() / mult[j]);
                }
            }
            maximum / p.alpha
        };
        let mut lambda_max = maximum(&st.scores);
        let start;
        if let Some(lambdas) = p.lambdas {
            path.lambda.copy_from_slice(lambdas);
            start = 0;
        } else {
            if mult.contains(&0.0) {
                // Differs from R biglasso: determine the grid from the
                // residual after fitting unpenalized columns, as for binomial.
                self.gaussian_prefit(&path, &mut st, &mult, tolerance)?;
                st.scores = self.compute_z(&path, &st.residual);
                lambda_max = maximum(&st.scores);
            }
            // Differs from R biglasso: reject unrepresentable automatic grids
            // rather than propagating nonfinite penalties into the fit.
            if !lambda_max.is_finite() {
                return Err("The automatic lambda grid must be finite; increase alpha or supply a finite grid.".into());
            }
            if lambda_max == 0.0 {
                // Differs from R biglasso: its logarithmic grid can be
                // nonfinite at zero; retain the finite unpenalized problem.
                path.lambda.fill(0.0);
            } else if p.lambda_log_scale {
                let values = linspace(log(p.lambda_min * lambda_max), log(lambda_max), nl);
                for k in 0..nl {
                    path.lambda[k] = exp(values[nl - 1 - k]);
                }
            } else {
                path.lambda = linspace(lambda_max, p.lambda_min * lambda_max, nl);
            }
            if path.lambda.iter().any(|value| !value.is_finite()) {
                return Err("The automatic lambda grid must be finite; increase alpha or supply a finite grid.".into());
            }
            path.deviance[0] = self.gaussian_loss(&st);
            path.n_reject[0] = nc as i64;
            self.gaussian_record(&mut path, &st, 0);
            if self.stopped(&mut path, 0)? {
                return Ok(path);
            }
            start = 1;
        }

        trace(&path, &st, None, Point::GridReady, 0.);
        let mut cache = self.gaussian_cache(&path, &st);
        trace(&path, &st, cache.as_ref(), Point::CacheReady, 0.);
        for il in start..nl {
            st.il = il;

            if il > 0 && st.beta.iter().filter(|&&v| v != 0.0).count() > p.dfmax {
                path.events.push(Event::Dfmax(il));
                path.iters[il..].fill(-1);
                break;
            }
            let previous = if il == 0 { lambda_max } else { path.lambda[il - 1] };
            let lam = path.lambda[il];
            self.choose_strong(&mut st, cache.as_mut(), il, lam, previous, &mult);
            trace(&path, &st, cache.as_ref(), Point::StrongSet, lam);
            path.n_reject[il] = st.strong.iter().filter(|&&v| !v).count() as i64;
            self.gaussian_lambda(&mut path, &mut st, cache.as_mut(), il, lam, tolerance);
            self.gaussian_record(&mut path, &st, il);
            trace(&path, &st, cache.as_ref(), Point::Recorded, lam);
            if self.stopped(&mut path, il)? {
                break;
            }
        }
        Ok(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, ShapeBuilder};

    #[test]
    fn default_cached_sum_matches_documented_lane_tree() {
        // Default mode deliberately changed from row-order summation to the
        // documented SIMD-independent tree; do not adjust numeric tolerances.
        assert_cached_sum_order(false);
    }

    #[test]
    fn compatible_cached_sum_matches_updated_residual_in_row_order() {
        assert_cached_sum_order(true);
    }

    fn assert_cached_sum_order(r_compatible: bool) {
        for n in [9, CHUNK + 1] {
            for masked in [false, true] {
                for offset in [0.0, 1e9] {
                    let x = Array2::from_shape_fn((n, 3).f(), |(i, j)| {
                        offset + ((i * 17 + j * 7) % 31) as f64 * 0.123456789
                    });
                    let design = ContigX::<_, true> { x: x.view() };
                    let mask: Vec<f64> = (0..n).map(|i| if masked && i % 3 == 1 { 0.0 } else { 1.0 }).collect();
                    let rows: Vec<usize> = (0..n).filter(|&i| mask[i] != 0.0).collect();
                    let y = vec![0.0; n];
                    let params = Params {
                        family: Family::Gaussian,
                        alpha: 0.5,
                        lambdas: None,
                        nlambda: 1,
                        eps: 1e-7,
                        lambda_min: 0.01,
                        lambda_log_scale: true,
                        penalty_factor: &[0.0, 1.0, 2.0],
                        dfmax: 3,
                        max_iter: 1000,
                        stop_on_saturation: false,
                        r_compatible,
                        ybar: 0.0,
                        null_deviance: 1.0,
                        on_lambda: None,
                    };
                    let fit = Fit {
                        x: &design,
                        y: &y,
                        params: &params,
                        mask: Some(&mask),
                        rows: Some(&rows),
                        n_fit: rows.len(),
                    };
                    let (center, scale, _, _) = fit.center_scale().unwrap();
                    let path = Path {
                        center,
                        scale,
                        col_map: vec![0, 1, 2],
                        lambda: vec![1.0],
                        intercept: vec![0.0],
                        coef_rows: vec![None],
                        coef_vals: vec![None],
                        iters: vec![0],
                        deviance: vec![0.0],
                        n_reject: vec![0],
                        events: vec![],
                    };
                    let mut residual = Rows::new(n, 0.0);
                    for i in 0..n {
                        if fit.fits(i) {
                            residual[i] = ((i * 13) % 19) as f64 * 0.314159265 - 2.0;
                        }
                    }
                    let mut state = GaussianState {
                        version: 0,
                        il: 0,
                        residual_sum: fit.score_sum(&residual),
                        residual,
                        beta: vec![0.0; 3],
                        active: vec![true; 3],
                        strong: vec![true; 3],
                        scores: vec![0.0; 3],
                    };
                    for lambda in [1.0, 0.1, 0.0, 1e20, 1e20] {
                        fit.gaussian_sweep(&path, &mut state, lambda);
                        let expected = if r_compatible {
                            state.residual.iter().fold(0.0, |sum, &value| sum + value)
                        } else {
                            super::sum_tests::reference_sum(&state.residual)
                        };
                        assert_eq!(state.residual_sum.to_bits(), expected.to_bits());
                        for i in (0..n).filter(|&i| !fit.fits(i)) {
                            assert_eq!(state.residual[i].to_bits(), 0.0f64.to_bits());
                        }
                    }
                    // No selected coordinates must retain the valid cache.
                    state.active.fill(false);
                    let before = state.residual_sum.to_bits();
                    assert_eq!(fit.gaussian_sweep(&path, &mut state, 0.0), 0.0);
                    assert_eq!(state.residual_sum.to_bits(), before);
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "gaussian_skip_tests.rs"]
mod skip_tests;

#[cfg(test)]
#[path = "gaussian_sum_tests.rs"]
mod sum_tests;
