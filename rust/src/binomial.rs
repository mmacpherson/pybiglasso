//! R biglasso's logistic SSR path: IRLS, the intercept update, the binomial
//! coordinate sweep and its fused row passes, and the lambda loop. Column
//! access, standardization and scores, shared with the Gaussian solver, are in
//! solver.rs, columns.rs and kernels.rs.

use super::*;

const ETA_MAX: f64 = 10.0;
const W_MIN: f64 = 0.0001;

fn irls_limits(r_compatible: bool) -> (f64, f64) {
    if r_compatible {
        (ETA_MAX, W_MIN)
    } else {
        // Differs from R biglasso: use glmnet's pmin boundary instead of
        // |eta| > 10 to reduce clipping's score bias. Retain positive boundary
        // curvature: our working residual s/w needs it, unlike glmnet's
        // undivided residual, which allows zero weights.
        let p_min = 1e-9;
        (log(1.0 / p_min - 1.0), p_min * (1.0 - p_min))
    }
}

#[inline(always)]
fn irls_probability_weight(eta: f64, eta_max: f64, w_min: f64) -> (f64, f64) {
    if eta > eta_max {
        (1.0, w_min)
    } else if eta < -eta_max {
        (0.0, w_min)
    } else {
        let e = exp(eta);
        let probability = e / (1.0 + e);
        (probability, probability * (1.0 - probability))
    }
}

/// Null-model statistics in the selected row order, without NumPy's
/// platform-dependent logarithms or reduction order in the stopping threshold.
/// The caller has range-checked rows; unused responses need not be valid.
pub(crate) fn null_model(y: &[f64], rows: Option<&[usize]>) -> Result<(f64, f64), &'static str> {
    let n = rows.map_or(y.len(), |idx| idx.len());
    let row = |i| rows.map_or(i, |idx| idx[i]);
    let mut ones = 0;
    for i in 0..n {
        let value = y[row(i)];
        if value == 1.0 {
            ones += 1;
        } else if value != 0.0 {
            return Err("Selected responses must be 0 or 1.");
        }
    }
    if ones == 0 || ones == n {
        return Err("Selected responses must contain both 0s and 1s.");
    }
    let ybar = ones as f64 / n as f64;
    let (positive, negative) = (log(ybar), log(1.0 - ybar));
    let mut loss = 0.0;
    for i in 0..n {
        loss -= if y[row(i)] == 1.0 { positive } else { negative };
    }
    Ok((ybar, loss))
}

/// Stable score and curvature from centered values in left-to-right row order.
fn centered_moments<C: Col>(c: &C, r: &[f64], w: &[f64], center: f64, scale: f64, n_fit: f64) -> (f64, f64) {
    let n = c.n();
    let (mut xwr, mut wxx) = (0.0, 0.0);
    for i in 0..n {
        let z = (c.at(i) - center) / scale;
        xwr += z * w[i] * r[i];
        wxx += w[i] * z * z;
    }
    (xwr, wxx / n_fit)
}

// The coordinate sweep's pass over the rows. Each sums a column CHUNK rows
// at a time and then adds the chunk sums in order, so threads can share out
// a tall column's rows without changing any sum.

/// Rows per chunk: a multiple of LANES, and part of the results, since
/// changing it changes their last bits.
pub(super) const CHUNK: usize = 2048;
/// Columns at least this tall share their chunks across threads; shorter
/// ones take longer to hand out than to sum.
const PAR_ROWS: usize = 16_384;
/// IRLS has portable exp/log work per row; keep its dispatch threshold
/// separate from the coordinate sweep's arithmetic-only passes.
const PAR_IRLS_ROWS: usize = 16_384;

/// A pass's sums; those it doesn't compute stay zero.
#[derive(Clone, Copy)]
struct Sums<T> {
    /// sum(w r), after the shift
    wr: T,
    /// sum(x w r), sum(w x^2), sum(w x) for the next column
    xwr: T,
    wxx: T,
    wx: T,
}
impl<T: Num> Sums<T> {
    fn zero(z: T) -> Self {
        Sums { wr: z, xwr: z, wxx: z, wx: z }
    }
}
impl Sums<f64> {
    /// Two chunks' sums, the earlier first.
    fn add(self, b: Self) -> Self {
        Sums { wr: self.wr + b.wr, xwr: self.xwr + b.xwr, wxx: self.wxx + b.wxx, wx: self.wx + b.wx }
    }
}

/// One pass of the coordinate sweep over the rows: with SHIFT, move r and eta
/// by the last coordinate's change, delta * (x - c) on column `prev`, and sum
/// w r; then with MOMENTS, the next column's moments on the shifted r. Fused,
/// each row sees the same arithmetic, in the same order, as the two passes.
struct Pass<'a, C, const SHIFT: bool, const MOMENTS: bool> {
    prev: &'a C,
    delta: f64,
    c: f64,
    next: &'a C,
}

impl<'a, C> Pass<'a, C, true, false> {
    fn shift(prev: &'a C, delta: f64, c: f64) -> Self {
        Pass { prev, delta, c, next: prev }
    }
}
impl<'a, C> Pass<'a, C, false, true> {
    fn moments(next: &'a C) -> Self {
        Pass { prev: next, delta: 0.0, c: 0.0, next }
    }
}
impl<'a, C> Pass<'a, C, true, true> {
    fn shift_then_moments(prev: &'a C, delta: f64, c: f64, next: &'a C) -> Self {
        Pass { prev, delta, c, next }
    }
}

impl<C: Col, const SHIFT: bool, const MOMENTS: bool> Pass<'_, C, SHIFT, MOMENTS> {
    /// One row, or one register of rows.
    #[inline(always)]
    fn row<T: Num>((delta, c): (T, T), (xp, xn): (T, T), r: &mut T, eta: &mut T, w: T, acc: &mut Sums<T>) {
        if SHIFT {
            let d = delta * (xp - c);
            *r -= d;
            *eta += d;
            acc.wr += w * *r;
        }
        if MOMENTS {
            // Differs from R biglasso, which forms (x * r) * w and w * (x * x)
            // and sums serially: w * x is shared by all three sums, and the
            // sums run in LANES accumulators per CHUNK, in a fixed order.
            let wx = w * xn;
            acc.xwr += wx * *r;
            acc.wxx += wx * xn;
            acc.wx += wx;
        }
    }

    /// Rows i0 .. i0 + r.len(), given their r, eta and w; the body in
    /// registers, LANES accumulators each, then the tail one row at a time.
    #[inline(always)]
    fn chunk<S: Simd>(&self, simd: S, i0: usize, r: &mut [f64], eta: &mut [f64], w: &[f64]) -> Sums<f64> {
        let n = r.len();
        let (prev, next) = (&Span { col: self.prev, i0, n }, &Span { col: self.next, i0, n });
        let body = n / LANES * LANES;
        let (rb, rt) = r.split_at_mut(body);
        let (eb, et) = eta.split_at_mut(body);
        let (rv, _) = S::as_mut_simd_f64s(rb);
        let (ev, _) = S::as_mut_simd_f64s(eb);
        let (wv, _) = S::as_simd_f64s(&w[..body]);
        let v = |x| V(simd, x);
        let k = (v(simd.splat_f64s(self.delta)), v(simd.splat_f64s(self.c)));
        let zero = Block([0.0; LANES]);
        let mut acc = [Sums::zero(v(simd.splat_f64s(0.0))); LANES];
        let mut start = 0;
        while start < body {
            let (prev, prev_end) = prev.run(start, body);
            let (next, next_end) = next.run(start, body);
            let end = prev_end.min(next_end);
            let range = start / LANES * regs::<S>()..end / LANES * regs::<S>();
            let blocks = rv[range.clone()]
                .chunks_exact_mut(regs::<S>())
                .zip(ev[range.clone()].chunks_exact_mut(regs::<S>()))
                .zip(wv[range].chunks_exact(regs::<S>()));
            let (mut hint_prev, mut hint_next) = (0, 0);
            for (b, ((r, eta), w)) in blocks.enumerate() {
                let xp = if SHIFT { load(&prev, b * LANES, &mut hint_prev) } else { zero };
                let xn = if MOMENTS { load(&next, b * LANES, &mut hint_next) } else { zero };
                let (xp, xn) = (regs_of::<S>(&xp), regs_of::<S>(&xn));
                for m in 0..regs::<S>() {
                    let (mut rm, mut em) = (v(r[m]), v(eta[m]));
                    Self::row(k, (v(xp[m]), v(xn[m])), &mut rm, &mut em, v(w[m]), &mut acc[m]);
                    if SHIFT {
                        (r[m], eta[m]) = (rm.1, em.1);
                    }
                }
            }
            start = end;
        }
        let mut tail = Sums::zero(0.0);
        for (i, (r, eta)) in rt.iter_mut().zip(et.iter_mut()).enumerate() {
            let i = body + i;
            let xp = if SHIFT { prev.at(i) } else { 0.0 };
            let xn = if MOMENTS { next.at(i) } else { 0.0 };
            Self::row((self.delta, self.c), (xp, xn), r, eta, w[i], &mut tail);
        }
        Sums {
            wr: fold(acc.map(|a| a.wr)) + tail.wr,
            xwr: fold(acc.map(|a| a.xwr)) + tail.xwr,
            wxx: fold(acc.map(|a| a.wxx)) + tail.wxx,
            wx: fold(acc.map(|a| a.wx)) + tail.wx,
        }
    }

    /// The pass over the first n rows: the chunks shared out across threads
    /// for a tall column, otherwise run in turn under one dispatch.
    fn sweep(&self, n: usize, r: &mut [f64], eta: &mut [f64], w: &[f64]) -> Sums<f64> {
        let (r, eta, w) = (&mut r[..n], &mut eta[..n], &w[..n]);
        if n >= PAR_ROWS && rayon::current_num_threads() > 1 {
            let chunks = r.par_chunks_mut(CHUNK).zip(eta.par_chunks_mut(CHUNK)).zip(w.par_chunks(CHUNK));
            let parts: Vec<Sums<f64>> = chunks
                .enumerate()
                .map(|(k, ((r, eta), w))| dispatch(Chunks { pass: self, i0: k * CHUNK, r, eta, w }))
                .collect();
            parts.into_iter().reduce(Sums::add).expect("a column has rows")
        } else {
            dispatch(Chunks { pass: self, i0: 0, r, eta, w })
        }
    }
}

/// A pass over rows i0 .. i0 + r.len(), a chunk at a time, sums added in order.
struct Chunks<'a, P> {
    pass: &'a P,
    i0: usize,
    r: &'a mut [f64],
    eta: &'a mut [f64],
    w: &'a [f64],
}
impl<C: Col, const SHIFT: bool, const MOMENTS: bool> WithSimd for Chunks<'_, Pass<'_, C, SHIFT, MOMENTS>> {
    type Output = Sums<f64>;
    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Sums<f64> {
        let Chunks { pass, i0, r, eta, w } = self;
        // A loop, not map and fold: LLVM didn't inline that closure for Arrow
        // columns, which compiled it without the SIMD target features and
        // made every vector operation a call.
        let mut chunks = r.chunks_mut(CHUNK).zip(eta.chunks_mut(CHUNK)).zip(w.chunks(CHUNK)).enumerate();
        let (_, ((r0, eta0), w0)) = chunks.next().expect("a column has rows");
        let mut sums = pass.chunk(simd, i0, r0, eta0, w0);
        for (k, ((r, eta), w)) in chunks {
            sums = Sums::add(sums, pass.chunk(simd, i0 + k * CHUNK, r, eta, w));
        }
        sums
    }
}

#[derive(PartialEq)]
enum Status {
    Converged,
    Saturated,
    MaxIter,
}

struct State {
    eta: Rows,
    // Scratch used only inside IRLS: probabilities on the serial path;
    // log-likelihood terms on the parallel path. Neither escapes the step.
    irls_work: Rows,
    w: Rows,
    s: Rows,
    r: Rows,
    beta: Vec<f64>,
    active: Vec<u8>,
    strong: Vec<u8>,
    z: Vec<f64>,
    intercept: f64,
    sum_w_residuals: f64,
}

impl<D: Design> Fit<'_, D> {
    fn y_at(&self, i: usize) -> f64 {
        self.y[i]
    }

    /// w, s, r from eta; returns -log-likelihood with probabilities clipped
    /// at the mode's boundary. Rows are independent; the loss sum stays serial.
    fn update_irls(&self, st: &mut State) -> f64 {
        if st.eta.len() >= PAR_IRLS_ROWS && rayon::current_num_threads() > 1 {
            return self.update_irls_parallel(st);
        }
        let (eta_max, w_min) = irls_limits(self.params.r_compatible);
        for i in 0..st.eta.len() {
            if !self.fits(i) {
                (st.irls_work[i], st.w[i], st.s[i], st.r[i]) = (0.0, 0.0, 0.0, 0.0);
                continue;
            }
            (st.irls_work[i], st.w[i]) = irls_probability_weight(st.eta[i], eta_max, w_min);
            st.s[i] = self.y_at(i) - st.irls_work[i];
            st.r[i] = st.s[i] / st.w[i];
        }
        let mut acc = 0.0;
        for i in (0..st.eta.len()).filter(|&i| self.fits(i)) {
            acc += if self.y_at(i) == 1.0 { log(st.irls_work[i]) } else { log(1.0 - st.irls_work[i]) };
        }
        -acc
    }

    // Keep the Rayon plumbing out of the original serial loop's optimizer
    // scope; both the fused serial loop and a combined function regressed
    // single-thread fits in the before/after measurements.
    #[inline(never)]
    fn update_irls_parallel(&self, st: &mut State) -> f64 {
        let (eta_max, w_min) = irls_limits(self.params.r_compatible);
        let update = |i: usize, (((w, s), r), loss): (((&mut f64, &mut f64), &mut f64), &mut f64)| {
            if !self.fits(i) {
                (*w, *s, *r, *loss) = (0.0, 0.0, 0.0, 0.0);
                return;
            }
            let (probability, weight) = irls_probability_weight(st.eta[i], eta_max, w_min);
            *w = weight;
            *s = self.y_at(i) - probability;
            *r = *s / *w;
            *loss = if self.y_at(i) == 1.0 { log(probability) } else { log(1.0 - probability) };
        };
        st.w.par_iter_mut()
            .zip(st.s.par_iter_mut())
            .zip(st.r.par_iter_mut())
            .zip(st.irls_work.par_iter_mut())
            .enumerate()
            .for_each(|(i, row)| update(i, row));
        // Only the independent row work is parallel. Keep the original
        // sequential loss sum, including its masked-row omissions: chunk
        // subtotals would change its rounding and could tip saturation.
        let mut acc = 0.0;
        for i in (0..st.eta.len()).filter(|&i| self.fits(i)) {
            acc += st.irls_work[i];
        }
        -acc
    }

    fn update_intercept(st: &mut State, sum_w: f64) -> f64 {
        let xwr = dot(&st.w, &st.r);
        let shift = xwr / sum_w;
        let intercept = shift + st.intercept;
        if shift != 0.0 {
            st.intercept = intercept;
            for v in st.r.iter_mut() {
                *v -= shift;
            }
            for v in st.eta.iter_mut() {
                *v += shift;
            }
            st.sum_w_residuals = dot(&st.w, &st.r);
        } else {
            st.sum_w_residuals = xwr;
        }
        intercept
    }

    fn coordinate_sweep(&self, path: &Path, st: &mut State, lam: f64, sum_w: f64) -> Result<f64, String> {
        let p = self.params;
        let n = self.x.nrow();
        let nrow = self.n_fit as f64;
        let mut max_update: f64 = 0.0;
        let mut swr = st.sum_w_residuals;
        // A coordinate's shift waits for the next active column, so the two
        // share a pass over the rows.
        let mut pending: Option<(usize, f64, f64)> = None;
        for j in 0..path.col_map.len() {
            if st.active[j] == 0 {
                continue;
            }
            let jj = path.col_map[j];
            let col = self.x.col(jj);
            let (c, sc) = (path.center[jj], path.scale[jj]);
            let (xwr, v);
            // Differs from R biglasso, which always uses raw moments: use
            // centered values when cancellation would make them unreliable.
            if c.abs() > 100.0 * sc {
                if let Some((pj, delta, pc)) = pending.take() {
                    swr = Pass::shift(&self.x.col(pj), delta, pc).sweep(n, &mut st.r, &mut st.eta, &st.w).wr;
                }
                (xwr, v) = centered_moments(&col, &st.r, &st.w, c, sc, nrow);
            } else {
                let sums = match pending.take() {
                    Some((pj, delta, pc)) => {
                        let sums = Pass::shift_then_moments(&self.x.col(pj), delta, pc, &col).sweep(
                            n,
                            &mut st.r,
                            &mut st.eta,
                            &st.w,
                        );
                        swr = sums.wr;
                        sums
                    }
                    None => Pass::moments(&col).sweep(n, &mut st.r, &mut st.eta, &st.w),
                };
                let (sum_xwr, sum_wxx, sum_wx) = (sums.xwr, sums.wxx, sums.wx);
                let numerator = sum_wxx - 2.0 * c * sum_wx + c * c * sum_w;
                let magnitude = sum_wxx.abs() + (2.0 * c * sum_wx).abs() + (c * c * sum_w).abs();
                let v0 = numerator / (sc * sc) / nrow;
                // recompute if subtraction has lost more than about eight
                // digits, including a positive but unreliable curvature
                if !v0.is_finite() || numerator <= 1e-8 * magnitude {
                    (xwr, v) = centered_moments(&col, &st.r, &st.w, c, sc, nrow);
                } else {
                    (xwr, v) = ((sum_xwr - c * swr) / sc, v0);
                }
            }
            if !v.is_finite() || v <= 0.0 || !xwr.is_finite() {
                return Err("Non-positive or non-finite coordinate curvature; cannot fit safely.".into());
            }

            let u = xwr / nrow + v * st.beta[j];
            let l1 = lam * p.penalty_factor[jj] * p.alpha;
            let l2 = lam * p.penalty_factor[jj] * (1.0 - p.alpha);
            let new_beta = if u.abs() < l1 {
                0.0
            } else {
                let sign = if u > 0.0 { 1.0 } else { -1.0 };
                // Differs from R biglasso: R divides by v * (1 + l2), while
                // the documented elastic-net objective requires v + l2.
                // r_compatible keeps R's formula, dividing by v and then by
                // 1 + l2 where R divides by v * (1 + l2); l2 = 0 agrees in both.
                if p.r_compatible { sign * (u.abs() - l1) / v / (1.0 + l2) } else { sign * (u.abs() - l1) / (v + l2) }
            };
            let shift = new_beta - st.beta[j];
            if shift != 0.0 {
                max_update = max_update.max(v * (shift * shift));
                // Differs from R biglasso, which updates by (shift * (x - c)) / scale:
                // one division per coordinate, not per row, so the row pass vectorizes.
                pending = Some((jj, shift / sc, c));
            }
            st.beta[j] = new_beta;
        }
        if let Some((pj, delta, pc)) = pending {
            swr = Pass::shift(&self.x.col(pj), delta, pc).sweep(n, &mut st.r, &mut st.eta, &st.w).wr;
        }
        st.sum_w_residuals = swr;
        Ok(max_update)
    }

    /// Score the columns that `eligible` picks; flag violators active (and strong,
    /// for the rest of the predictors). Returns the number of violations.
    fn check(&self, path: &Path, st: &mut State, lam: f64, strong_only: bool) -> usize {
        let p = self.params;
        let s_sum = seqsum(&st.s);
        let (s, beta) = (&st.s, &st.beta);
        let nf = self.n_fit as f64;
        st.z.par_iter_mut()
            .zip(st.active.par_iter_mut())
            .zip(st.strong.par_iter_mut())
            .enumerate()
            .with_min_len(16)
            .map(|(j, ((z, active), strong))| {
                let eligible = if strong_only { *active == 0 && *strong == 1 } else { *strong == 0 };
                if !eligible {
                    return 0;
                }
                let jj = path.col_map[j];
                *z = standardized_score(&self.x.col(jj), s, s_sum, path.center[jj], path.scale[jj], nf);
                let l1 = lam * p.penalty_factor[jj] * p.alpha;
                let l2 = lam * p.penalty_factor[jj] * (1.0 - p.alpha);
                if (*z - beta[j] * l2).abs() > l1 {
                    *active = 1;
                    *strong = 1;
                    1
                } else {
                    0
                }
            })
            .sum()
    }

    fn fit_lambda(
        &self,
        path: &mut Path,
        st: &mut State,
        il: usize,
        lam: f64,
        thresh: f64,
        check_kkt: bool,
    ) -> Result<Status, String> {
        let p = self.params;
        // the scans the last iteration didn't run: (strong set, rest)
        let mut unscanned = (true, true);
        while path.iters[il] < p.max_iter {
            path.iters[il] += 1;
            unscanned = (true, true);
            path.deviance[il] = self.update_irls(st);
            // Differs from R biglasso, which always stops here
            if p.stop_on_saturation && path.deviance[il] / p.null_deviance < 0.01 {
                return Ok(Status::Saturated);
            }
            // W stays fixed through the intercept and coordinate sweep. Both
            // used to repeat this same sequential sum; reuse its exact bits.
            let sum_w = seqsum(&st.w);
            path.intercept[il] = Self::update_intercept(st, sum_w);
            let max_update = self.coordinate_sweep(path, st, lam, sum_w)?;
            if max_update >= thresh {
                continue;
            }
            if !check_kkt {
                return Ok(Status::Converged);
            }
            if self.check(path, st, lam, true) > 0 {
                unscanned = (false, true);
                continue;
            }
            if self.check(path, st, lam, false) == 0 {
                return Ok(Status::Converged);
            }
            unscanned = (false, false);
        }
        // as R biglasso does, finish the scans when max_iter cuts a fit short:
        // they set the z and the strong set the next lambda starts from
        if check_kkt && unscanned.0 {
            self.check(path, st, lam, true);
        }
        if check_kkt && unscanned.1 {
            self.check(path, st, lam, false);
        }
        Ok(Status::MaxIter)
    }

    // Differs from R biglasso: report stable loss at the returned predictor.
    // R uses clipped probabilities from before the final coordinate sweep;
    // r_compatible keeps that earlier loss instead of calling this function.
    fn final_loss(&self, st: &State) -> f64 {
        let mut loss = 0.0;
        for i in (0..st.eta.len()).filter(|&i| self.fits(i)) {
            let signed = if self.y_at(i) == 1.0 { -st.eta[i] } else { st.eta[i] };
            loss += signed.max(0.0) + log1p(exp(-signed.abs()));
        }
        loss
    }

    fn record(path: &mut Path, st: &State, il: usize) {
        let rows: Vec<usize> = (0..st.beta.len()).filter(|&j| st.beta[j] != 0.0).collect();
        path.coef_vals[il] = Some(rows.iter().map(|&j| st.beta[j]).collect());
        path.coef_rows[il] = Some(rows);
    }

    pub(super) fn run_binomial(&self, center: Vec<f64>, scale: Vec<f64>) -> Result<Path, String> {
        let p = self.params;
        let nrow = self.x.nrow();
        let col_map: Vec<usize> = (0..scale.len()).filter(|&j| scale[j] > SCALE_MIN).collect();
        let ncol_used = col_map.len();
        if ncol_used == 0 {
            return Err("X has no varying features after standardization.".into());
        }
        let mult: Vec<f64> = col_map.iter().map(|&jj| p.penalty_factor[jj]).collect();
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
            intercept: vec![0.0; nl],
            coef_rows: vec![None; nl],
            coef_vals: vec![None; nl],
            iters: vec![0; nl],
            deviance: vec![0.0; nl],
            n_reject: vec![0; nl],
            events: vec![],
        };
        let z = self.compute_z(&path, self.y);
        let null_intercept = log(p.ybar / (1.0 - p.ybar));
        let thresh = p.eps * p.null_deviance / self.n_fit as f64;
        let lambda_max_of = |z: &[f64]| {
            let mut m: f64 = 0.0;
            for j in 0..z.len() {
                if mult[j] > 0.0 {
                    m = m.max(z[j].abs() / mult[j]);
                }
            }
            m / p.alpha
        };
        let lambda_max_r = lambda_max_of(&z);

        let mut st = State {
            eta: Rows::new(nrow, null_intercept),
            irls_work: Rows::new(nrow, 0.0),
            w: Rows::new(nrow, 0.0),
            s: Rows::new(nrow, 0.0),
            r: Rows::new(nrow, 0.0),
            beta: vec![0.0; ncol_used],
            active: vec![0; ncol_used],
            strong: vec![0; ncol_used],
            z,
            intercept: null_intercept,
            sum_w_residuals: 0.0,
        };

        let (lambda_max, lstart);
        if let Some(lambdas) = p.lambdas {
            path.lambda.copy_from_slice(lambdas);
            lambda_max = lambda_max_r;
            lstart = 0;
        } else {
            if mult.contains(&0.0) {
                // Differs from R biglasso, whose default grid is all zero
                // with unpenalized columns. Fit those columns first: their
                // residuals determine lambda_max for the penalized columns.
                for j in 0..ncol_used {
                    if mult[j] == 0.0 {
                        st.active[j] = 1;
                        st.strong[j] = 1;
                    }
                }
                match self.fit_lambda(&mut path, &mut st, 0, 0.0, thresh, false)? {
                    Status::Saturated => {
                        path.events.push(Event::PrefitSaturated);
                        path.iters.fill(-1);
                        return Ok(path);
                    }
                    Status::MaxIter => {
                        return Err("Unpenalized prefit failed to converge within max_iter; increase max_iter or supply lambdas explicitly.".into());
                    }
                    Status::Converged => {}
                }
                self.update_irls(&mut st);
                st.z = self.compute_z(&path, &st.s);
            } else {
                path.deviance[0] = p.null_deviance;
                path.intercept[0] = null_intercept;
            }
            lambda_max = lambda_max_of(&st.z);
            if lambda_max == 0.0 {
                path.lambda.fill(0.0);
            } else if p.lambda_log_scale {
                let ys = linspace(log(p.lambda_min * lambda_max), log(lambda_max), nl);
                for k in 0..nl {
                    path.lambda[k] = exp(ys[nl - 1 - k]);
                }
            } else {
                path.lambda.copy_from_slice(&linspace(lambda_max, p.lambda_min * lambda_max, nl));
            }
            // Differs from R biglasso, which fits a grid that overflowed (at a
            // tiny alpha, say) and returns its nonfinite penalties
            if path.lambda.iter().any(|value| !value.is_finite()) {
                return Err("The automatic lambda grid must be finite; increase alpha or supply a finite grid.".into());
            }
            Self::record(&mut path, &st, 0);
            if !p.r_compatible {
                path.deviance[0] = self.final_loss(&st);
            }
            path.n_reject[0] = ncol_used as i64;
            lstart = 1;
            if self.stopped(&mut path, 0)? {
                return Ok(path);
            }
        }

        for il in lstart..nl {
            let lam = path.lambda[il];
            let cutoff = if il > 0 {
                if st.beta.iter().filter(|&&v| v != 0.0).count() > p.dfmax {
                    path.events.push(Event::Dfmax(il));
                    path.iters[il..].fill(-1);
                    return Ok(path);
                }
                2.0 * lam - path.lambda[il - 1]
            } else {
                2.0 * lam - lambda_max
            };
            let mut kept = 0;
            for j in 0..ncol_used {
                st.strong[j] = (st.z[j].abs() > cutoff * p.alpha * mult[j]) as u8;
                kept += st.strong[j] as i64;
            }
            path.n_reject[il] = ncol_used as i64 - kept;

            match self.fit_lambda(&mut path, &mut st, il, lam, thresh, true)? {
                Status::Saturated => {
                    path.events.push(Event::Saturated(il));
                    path.iters[il..].fill(-1);
                    return Ok(path);
                }
                Status::MaxIter => path.events.push(Event::MaxIter(il)),
                Status::Converged => {}
            }
            Self::record(&mut path, &st, il);
            if !p.r_compatible {
                path.deviance[il] = self.final_loss(&st);
            }
            if self.stopped(&mut path, il)? {
                return Ok(path);
            }
        }
        Ok(path)
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

#[cfg(test)]
#[path = "kernel_tests.rs"]
mod kernel_tests;
