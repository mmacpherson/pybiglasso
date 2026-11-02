//! R biglasso's Gaussian SSR path, sharing column access and fixed-order scores
//! with the logistic solver. Only row/state/path buffers are allocated; X is
//! always borrowed and read one column at a time.

use super::*;

struct GaussianState {
    residual: Rows,
    // Exactly seqsum(residual), not an algebraically updated running total.
    residual_sum: f64,
    beta: Vec<f64>,
    active: Vec<bool>,
    strong: Vec<bool>,
    scores: Vec<f64>,
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
                largest = largest.max(shift * shift);
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
        largest
    }

    fn gaussian_scan(&self, path: &Path, st: &mut GaussianState, lam: f64, strong_only: bool) -> usize {
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
        for il in start..nl {
            if il > 0 && st.beta.iter().filter(|&&v| v != 0.0).count() > p.dfmax {
                path.events.push(Event::Dfmax(il));
                path.iters[il..].fill(-1);
                break;
            }
            let previous = if il == 0 { lambda_max } else { path.lambda[il - 1] };
            let lam = path.lambda[il];
            for j in 0..nc {
                st.strong[j] = st.scores[j].abs() > (2.0 * lam - previous) * p.alpha * mult[j];
            }
            path.n_reject[il] = st.strong.iter().filter(|&&v| !v).count() as i64;
            self.gaussian_lambda(&mut path, &mut st, il, lam, tolerance);
            self.gaussian_record(&mut path, &st, il);
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
