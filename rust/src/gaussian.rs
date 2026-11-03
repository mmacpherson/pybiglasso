//! R biglasso's Gaussian SSR path, sharing column access and fixed-order scores
//! with the logistic solver. Only row/state/path buffers are allocated; X is
//! always borrowed and read one column at a time.

use super::gaussian_bound as bound;
use super::*;

struct GaussianState {
    residual: Rows,
    // Exactly seqsum(residual), not an algebraically updated running total.
    residual_sum: f64,
    beta: Vec<f64>,
    active: Vec<bool>,
    strong: Vec<bool>,
    scores: Vec<f64>,
    version: u64,
    il: usize,
    skip: Option<ScoreCache>,
}

#[derive(Clone, Copy)]
struct NextStrong {
    decision: bool,
    epoch: u64,
    il: usize,
    lambda_bits: u64,
    cutoff_bits: u64,
}
#[derive(Clone, Copy)]
struct Anchor {
    score: f64,
    epoch: u64,
    represented_epoch: u64,
    prefix_lo: f64,
    rnorm: f64,
    next: Option<NextStrong>,
}
struct ScoreCache {
    #[cfg(test)]
    shadow: bool,
    meta: Vec<Option<bound::Meta>>,
    anchors: Vec<Anchor>,
    previous: Vec<f64>,
    version: u64,
    hi: f64,
    lo: f64,
    rnorm: f64,
    valid: bool,
    #[cfg(test)]
    baseline_reads: usize,
    #[cfg(test)]
    actual_reads: usize,
    #[cfg(test)]
    candidate_skips: usize,
    #[cfg(test)]
    next_certificates_used: usize,
    #[cfg(test)]
    last_skips: Vec<u8>,
    #[cfg(test)]
    last_radius: Vec<f64>,
    #[cfg(test)]
    positive_skips: usize,
    #[cfg(test)]
    fault_applied: bool,
}
#[derive(Clone, Copy, Default)]
struct Plan {
    selected: bool,
    skip: bool,
    next: Option<NextStrong>,
    #[cfg(test)]
    radius: f64,
}

// Native-test capture: exact trajectory plus the separate certificate metadata.
// No instrumentation or controls are compiled into the library.
#[cfg(test)]
fn observe(path: &Path, st: &GaussianState, kind: u64, lam: f64) {
    use std::io::Write;
    let Some(name) = std::env::var_os("PYBIGLASSO_TEST_TRACE") else {
        return;
    };
    let mut f = std::io::BufWriter::new(std::fs::OpenOptions::new().create(true).append(true).open(name).unwrap());
    let h = [kind, st.il as u64, st.residual.len() as u64, st.beta.len() as u64, path.center.len() as u64, st.version];
    f.write_all(bytemuck::cast_slice(&h)).unwrap();
    let (hi, lo, rn) = st.skip.as_ref().map_or((0., 0., 0.), |s| (s.hi, s.lo, s.rnorm));
    f.write_all(bytemuck::cast_slice(&[lam, st.residual_sum, hi, lo, rn])).unwrap();
    for a in [&st.residual[..], &st.scores[..], &st.beta[..], &path.center[..], &path.scale[..]] {
        f.write_all(bytemuck::cast_slice(a)).unwrap();
    }
    f.write_all(bytemuck::cast_slice(&path.col_map.iter().map(|&j| j as u64).collect::<Vec<_>>())).unwrap();
    for a in [&st.active, &st.strong] {
        f.write_all(&a.iter().map(|&v| u8::from(v)).collect::<Vec<_>>()).unwrap();
    }
    let zeros = vec![0u8; st.beta.len()];
    let fzeros = vec![0.; st.beta.len()];
    f.write_all(st.skip.as_ref().map_or(&zeros[..], |s| &s.last_skips[..])).unwrap();
    f.write_all(bytemuck::cast_slice(st.skip.as_ref().map_or(&fzeros[..], |s| &s.last_radius[..]))).unwrap();
    if let Some(s) = &st.skip {
        let values: Vec<f64> = s.anchors.iter().map(|a| a.score).collect();
        f.write_all(bytemuck::cast_slice(&values)).unwrap();
        for values in [
            s.anchors.iter().map(|a| a.epoch).collect::<Vec<_>>(),
            s.anchors.iter().map(|a| a.represented_epoch).collect(),
            s.anchors.iter().map(|a| a.next.map_or(u64::MAX, |v| v.il as u64)).collect(),
            s.anchors.iter().map(|a| a.next.map_or(0, |v| v.lambda_bits)).collect(),
            s.anchors.iter().map(|a| a.next.map_or(0, |v| v.cutoff_bits)).collect(),
        ] {
            f.write_all(bytemuck::cast_slice(&values)).unwrap();
        }
        f.write_all(
            &s.anchors.iter().map(|a| a.next.map_or(0, |v| if v.decision { 2 } else { 1 })).collect::<Vec<_>>(),
        )
        .unwrap();
    } else {
        f.write_all(bytemuck::cast_slice(&fzeros)).unwrap();
        for _ in 0..5 {
            f.write_all(bytemuck::cast_slice(&vec![0u64; st.beta.len()])).unwrap();
        }
        f.write_all(&zeros).unwrap();
    }
    if let Some(s) = &st.skip {
        f.write_all(bytemuck::cast_slice(&[s.version, u64::from(s.valid)])).unwrap();
        f.write_all(bytemuck::cast_slice(&s.previous)).unwrap();
        for values in
            [s.anchors.iter().map(|a| a.prefix_lo).collect::<Vec<_>>(), s.anchors.iter().map(|a| a.rnorm).collect()]
        {
            f.write_all(bytemuck::cast_slice(&values)).unwrap();
        }
        let epochs: Vec<u64> = s.anchors.iter().map(|a| a.next.map_or(0, |v| v.epoch)).collect();
        f.write_all(bytemuck::cast_slice(&epochs)).unwrap();
        for m in &s.meta {
            let bits = m.map_or([0; 7], |m| m.test_metadata());
            f.write_all(bytemuck::cast_slice(&bits)).unwrap();
        }
    } else {
        f.write_all(&vec![0; 16 + 8 * st.residual.len() + 80 * st.beta.len()]).unwrap();
    }
}

#[cfg(not(test))]
#[inline(always)]
fn observe(_path: &Path, _state: &GaussianState, _kind: u64, _lambda: f64) {}

impl ScoreCache {
    #[inline(always)]
    fn shadow(&self) -> bool {
        #[cfg(test)]
        {
            self.shadow
        }
        #[cfg(not(test))]
        {
            false
        }
    }
}

fn cache_mode() -> (bool, bool) {
    #[cfg(test)]
    {
        match std::env::var("PYBIGLASSO_TEST_SCAN_MODE").as_deref() {
            Ok("off") => (false, false),
            Ok("shadow") => (true, true),
            _ => (true, false),
        }
    }
    #[cfg(not(test))]
    {
        (true, false)
    }
}

impl<D: Design> Fit<'_, D> {
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
            let l1 = lam * p.alpha * p.penalty_factor[jj];
            let l2 = lam * (1.0 - p.alpha) * p.penalty_factor[jj];
            let value = if score.abs() <= l1 { 0.0 } else { score.signum() * (score.abs() - l1) / (1.0 + l2) };
            let shift = value - st.beta[j];
            if shift != 0.0 {
                st.version += 1;
                largest = largest.max(shift * shift);
                // Differs from R biglasso, which groups the update as
                // (shift * (x - c)) / scale; the results agree to the last bits.
                // Keep seqsum's exact left-to-right order over the updated
                // values, including masked zeros, without a second row pass.
                let mut sum = 0.0;
                let mut start = 0;
                let mut hint = 0;
                while start < st.residual.len() {
                    let (run, end) = column.run(start, st.residual.len());
                    // A byte lookup supplies eight values together; every
                    // update and sum still happens in the original row order.
                    // Masked fits retain their guarded scalar reads.
                    let body = if self.mask.is_none() { (end - start) / LANES * LANES } else { 0 };
                    for_each_block(&run, body, |i, block| {
                        // Keep the serial accumulator local to this block.
                        // Capturing it in the visitor otherwise makes LLVM
                        // store/reload it after every lane.
                        let mut block_sum = sum;
                        for (residual, &value) in st.residual[start + i..start + i + LANES].iter_mut().zip(&block.0) {
                            *residual -= shift * ((value - center) / scale);
                            block_sum += *residual;
                        }
                        sum = block_sum;
                    });
                    for (i, residual) in st.residual[start + body..end].iter_mut().enumerate() {
                        let i = body + i;
                        if self.fits(start + i) {
                            *residual -= shift * ((run.at_seq(i, &mut hint) - center) / scale);
                        }
                        sum += *residual;
                    }
                    start = end;
                }
                st.residual_sum = sum;
                st.beta[j] = value;
            }
        }
        #[cfg(test)]
        if st.il == usize::MAX {
            observe(path, st, 9, lam);
        }
        largest
    }

    fn gaussian_scan_original(&self, path: &Path, st: &mut GaussianState, lam: f64, strong_only: bool) -> usize {
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

    fn gaussian_scan(&self, path: &Path, st: &mut GaussianState, lam: f64, strong_only: bool) -> usize {
        observe(path, st, if strong_only { 2 } else { 4 }, lam);
        let Some(cache) = st.skip.as_mut() else {
            let answer = self.gaussian_scan_original(path, st, lam, strong_only);
            observe(path, st, if strong_only { 3 } else { 5 }, lam);
            return answer;
        };
        #[cfg(test)]
        if std::env::var("PYBIGLASSO_TEST_FAULT").as_deref() == Ok("invalidate")
            && cache.positive_skips > 0
            && !cache.fault_applied
        {
            cache.hi = f64::INFINITY;
            cache.fault_applied = true;
        }
        if cache.version != st.version && cache.valid {
            let prepared = bound::difference_norm(&st.residual, &cache.previous)
                .and_then(|step| bound::advance(cache.hi, cache.lo, step))
                .zip(bound::norm(st.residual.len(), |i| st.residual[i]));
            if let Some(((hi, lo), rnorm)) = prepared {
                cache.hi = hi;
                cache.lo = lo;
                cache.rnorm = rnorm;
                cache.previous.copy_from_slice(&st.residual);
            } else {
                cache.valid = false;
            }
            cache.version = st.version;
        }
        let plans: Vec<Plan> = (0..st.beta.len())
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
                #[cfg(test)]
                {
                    plan.radius = radius;
                }
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
            .collect();
        let oracle: Vec<Option<f64>> = plans
            .par_iter()
            .enumerate()
            .map(|(j, p)| {
                if !p.selected || (p.skip && !cache.shadow()) {
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
            .collect();
        let mut violations = 0;
        for (j, plan) in plans.iter().enumerate() {
            #[cfg(test)]
            {
                cache.last_skips[j] = u8::from(plan.skip);
            }
            #[cfg(test)]
            {
                cache.last_radius[j] = plan.radius;
            }
            if !plan.selected {
                continue;
            }
            #[cfg(test)]
            {
                cache.baseline_reads += 1;
            }
            let jj = path.col_map[j];
            let l1 = lam * self.params.alpha * self.params.penalty_factor[jj];
            let l2 = lam * (1. - self.params.alpha) * self.params.penalty_factor[jj];
            if plan.skip {
                #[cfg(test)]
                {
                    cache.candidate_skips += 1;
                    cache.positive_skips += usize::from(plan.radius > 0.);
                }
                if let Some(q) = oracle[j] {
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
                    #[cfg(test)]
                    {
                        cache.actual_reads += 1;
                    }
                }
                cache.anchors[j].represented_epoch = st.version;
                cache.anchors[j].next = plan.next;
                // An enabled skip NEVER writes an approximate or anchor score.
            } else {
                let q = oracle[j].expect("mandatory exact score");
                #[cfg(test)]
                {
                    cache.actual_reads += 1;
                }
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
        }
        observe(path, st, if strong_only { 3 } else { 5 }, lam);
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

    fn gaussian_lambda(&self, path: &mut Path, st: &mut GaussianState, il: usize, lam: f64, tolerance: f64) {
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
                if self.gaussian_scan(path, st, lam, true) == 0 {
                    break;
                }
            }
            if self.gaussian_scan(path, st, lam, false) == 0 {
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
            skip: None,
            scores: self.compute_z(&path, &residual),
            residual_sum: seqsum(&residual),
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
                for j in 0..nc {
                    st.active[j] = mult[j] == 0.0;
                }
                let mut converged = false;
                for _ in 0..p.max_iter {
                    let change = self.gaussian_sweep(&path, &mut st, 0.0);
                    if change < tolerance || change == 0.0 {
                        converged = true;
                        break;
                    }
                }
                if !converged {
                    return Err("Unpenalized prefit failed to converge within max_iter; increase max_iter or supply lambdas explicitly.".into());
                }
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

        observe(&path, &st, 7, 0.);
        let (enabled, _shadow) = cache_mode();
        if enabled {
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
            st.skip = Some(ScoreCache {
                #[cfg(test)]
                shadow: _shadow,
                meta,
                anchors: st
                    .scores
                    .iter()
                    .map(|&score| Anchor {
                        score,
                        epoch: st.version,
                        represented_epoch: st.version,
                        prefix_lo: 0.,
                        rnorm: rn,
                        next: None,
                    })
                    .collect(),
                previous: st.residual.to_vec(),
                version: st.version,
                hi: 0.,
                lo: 0.,
                rnorm: rn,
                valid: pool_ok && rnorm.is_some(),
                #[cfg(test)]
                baseline_reads: 0,
                #[cfg(test)]
                actual_reads: 0,
                #[cfg(test)]
                candidate_skips: 0,
                #[cfg(test)]
                next_certificates_used: 0,
                #[cfg(test)]
                last_skips: vec![0; nc],
                #[cfg(test)]
                last_radius: vec![0.; nc],
                #[cfg(test)]
                positive_skips: 0,
                #[cfg(test)]
                fault_applied: false,
            });
        }
        observe(&path, &st, 0, 0.);
        for il in start..nl {
            st.il = il;

            if il > 0 && st.beta.iter().filter(|&&v| v != 0.0).count() > p.dfmax {
                path.events.push(Event::Dfmax(il));
                path.iters[il..].fill(-1);
                break;
            }
            let previous = if il == 0 { lambda_max } else { path.lambda[il - 1] };
            let lam = path.lambda[il];
            for j in 0..nc {
                let cutoff = (2.0 * lam - previous) * p.alpha * mult[j];
                let exact_decision = st.scores[j].abs() > cutoff;
                if let Some(cache) = st.skip.as_mut() {
                    #[cfg(test)]
                    if !cache.fault_applied
                        && cache.positive_skips > 0
                        && !st.active[j]
                        && cache.anchors[j].next.is_some()
                        && let Ok(fault) = std::env::var("PYBIGLASSO_TEST_FAULT")
                    {
                        let a = &mut cache.anchors[j];
                        let next = a.next.as_mut().unwrap();
                        match fault.as_str() {
                            "epoch" => next.epoch ^= 1,
                            "represented" => a.represented_epoch ^= 1,
                            "index" => next.il += 1,
                            "lambda" => next.lambda_bits ^= 1,
                            "cutoff" => next.cutoff_bits ^= 1,
                            _ => {}
                        }
                        cache.fault_applied = fault != "invalidate";
                    }
                    let anchor = &mut cache.anchors[j];
                    if !st.active[j] {
                        if let Some(next) = anchor.next.take() {
                            assert_eq!(next.epoch, anchor.represented_epoch, "score epoch mismatch");
                            assert_eq!(
                                (next.il, next.lambda_bits, next.cutoff_bits),
                                (il, lam.to_bits(), cutoff.to_bits()),
                                "next threshold mismatch"
                            );
                            if cache.shadow() {
                                assert_eq!(
                                    next.decision, exact_decision,
                                    "shadow SSR decision mismatch il={} j={}",
                                    il, j
                                );
                            }
                            st.strong[j] = if cache.shadow() { exact_decision } else { next.decision };
                            #[cfg(test)]
                            {
                                cache.next_certificates_used += 1;
                            }
                        } else {
                            assert_eq!(anchor.epoch, anchor.represented_epoch, "stale exact score without certificate");
                            st.strong[j] = exact_decision;
                        }
                    } else {
                        st.strong[j] = exact_decision;
                    }
                } else {
                    st.strong[j] = exact_decision;
                }
            }
            observe(&path, &st, 1, lam);
            path.n_reject[il] = st.strong.iter().filter(|&&v| !v).count() as i64;
            self.gaussian_lambda(&mut path, &mut st, il, lam, tolerance);
            self.gaussian_record(&mut path, &st, il);
            observe(&path, &st, 6, lam);
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
    fn cached_sum_matches_updated_residual_in_row_order() {
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
                        r_compatible: false,
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
                    let (center, scale, _) = fit.center_scale().unwrap();
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
                        skip: None,
                        residual_sum: seqsum(&residual),
                        residual,
                        beta: vec![0.0; 3],
                        active: vec![true; 3],
                        strong: vec![true; 3],
                        scores: vec![0.0; 3],
                    };
                    for lambda in [1.0, 0.1, 0.0, 1e20, 1e20] {
                        fit.gaussian_sweep(&path, &mut state, lambda);
                        let expected = state.residual.iter().fold(0.0, |sum, &value| sum + value);
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
impl Drop for ScoreCache {
    fn drop(&mut self) {
        use std::io::Write;
        if let Some(file) = std::env::var_os("PYBIGLASSO_TEST_STATS") {
            let mut f = std::fs::OpenOptions::new().create(true).append(true).open(file).unwrap();
            writeln!(
                f,
                "{} {} {} {} {} {} {}",
                self.baseline_reads,
                self.actual_reads,
                self.candidate_skips,
                self.positive_skips,
                self.next_certificates_used,
                self.valid as u8,
                self.fault_applied as u8
            )
            .unwrap();
        }
    }
}
