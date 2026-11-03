//! R biglasso's Gaussian SSR path, sharing column access and fixed-order scores
//! with the logistic solver. X is always borrowed and read one column at a
//! time; optional covariance updates cache Gram columns within a fixed budget.

use super::*;

struct GaussianState {
    residual: Rows,
    // Exactly seqsum(residual), not an algebraically updated running total.
    residual_sum: f64,
    beta: Vec<f64>,
    active: Vec<bool>,
    strong: Vec<bool>,
    scores: Vec<f64>,
    gram: Option<GramState>,
}

// A fixed byte budget, independent of hardware, threads and available RAM.
const GRAM_BUDGET: usize = 1 << 30;
const GRAM_LIMIT: &str = "update_from='gram' exceeds the 1 GiB Gram budget; use update_from='residuals' instead.";

struct GramState {
    // Always z_j' r / n, unlike scores, which also holds partial scores.
    gradients: Vec<f64>,
    columns: Vec<Option<Vec<f64>>>,
    cached: usize,
}

fn gram_allocation(ncol: usize, cached: usize, budget: usize) -> Result<(), String> {
    match ncol.checked_mul(cached).and_then(|entries| entries.checked_mul(size_of::<f64>())) {
        Some(bytes) if bytes <= budget => Ok(()),
        _ => Err(GRAM_LIMIT.into()),
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

    fn gaussian_gram_column(&self, path: &Path, j: usize) -> Result<Vec<f64>, String> {
        let jj = path.col_map[j];
        let column = self.x.col(jj);
        let mut standardized = Rows::new(self.x.nrow(), 0.0);
        // One column of row scratch, never a standardized copy of X. Masked
        // zeros retain exactly the same lanes as a residual score reduction.
        let mut hint = 0;
        for i in 0..standardized.len() {
            if self.fits(i) {
                standardized[i] = (column.at_seq(i, &mut hint) - path.center[jj]) / path.scale[jj];
            }
        }
        let sum = seqsum(&standardized);
        let nf = self.n_fit as f64;
        let mut gram = Vec::new();
        gram.try_reserve_exact(path.col_map.len()).map_err(|_| GRAM_LIMIT.to_string())?;
        gram.resize(path.col_map.len(), 0.0);
        gram.par_iter_mut().enumerate().for_each(|(k, value)| {
            let kk = path.col_map[k];
            *value = standardized_score(&self.x.col(kk), &standardized, sum, path.center[kk], path.scale[kk], nf);
        });
        // Differs from R biglasso: residual updates assume a unit diagonal.
        // Covariance updates use the actual standardized curvature, including
        // its rounding, so this is the coordinate minimizer of the Gram model.
        gram[j] = dot(&standardized, &standardized) / nf;
        Ok(gram)
    }

    // Differs from R biglasso: optional covariance updates maintain all
    // gradients instead of updating row residuals after each coordinate.
    // The default continues to use R's residual sweep without this cache.
    fn gaussian_step(&self, path: &Path, st: &mut GaussianState, lam: f64) -> Result<f64, String> {
        if st.gram.is_none() {
            return Ok(self.gaussian_sweep(path, st, lam));
        }
        let gram = st.gram.as_mut().unwrap();
        let mut largest: f64 = 0.0;
        for j in 0..st.beta.len() {
            if !st.active[j] {
                continue;
            }
            let jj = path.col_map[j];
            let l1 = lam * self.params.alpha * self.params.penalty_factor[jj];
            let l2 = lam * (1.0 - self.params.alpha) * self.params.penalty_factor[jj];
            if st.beta[j] == 0.0 && gram.gradients[j].abs() <= l1 {
                continue;
            }
            if gram.columns[j].is_none() {
                // dfmax counts current nonzeros, not ever-active columns. It
                // cannot replace this check on every lazy expansion.
                gram_allocation(st.beta.len(), gram.cached + 1, GRAM_BUDGET)?;
                gram.columns[j] = Some(self.gaussian_gram_column(path, j)?);
                gram.cached += 1;
            }
            let column = gram.columns[j].as_ref().unwrap();
            let diagonal = column[j];
            let score = gram.gradients[j] + diagonal * st.beta[j];
            let value = if score.abs() <= l1 { 0.0 } else { score.signum() * (score.abs() - l1) / (diagonal + l2) };
            let shift = value - st.beta[j];
            if shift != 0.0 {
                // Same convergence quantity and threshold as residual mode.
                largest = largest.max(shift * shift);
                for (gradient, covariance) in gram.gradients.iter_mut().zip(column) {
                    *gradient -= shift * covariance;
                }
                st.beta[j] = value;
            }
        }
        Ok(largest)
    }

    fn gaussian_refresh(&self, path: &Path, st: &mut GaussianState) {
        for i in 0..st.residual.len() {
            st.residual[i] = if self.fits(i) { self.y[i] - self.params.ybar } else { 0.0 };
        }
        // Reconstruct from the coefficients, rather than propagate residual
        // rounding or cancellation along the path. Feature/row order is fixed.
        for j in 0..st.beta.len() {
            let beta = st.beta[j];
            if beta == 0.0 {
                continue;
            }
            let jj = path.col_map[j];
            let column = self.x.col(jj);
            let mut i = 0;
            for_each_value(&column, |value| {
                if self.fits(i) {
                    st.residual[i] -= beta * ((value - path.center[jj]) / path.scale[jj]);
                }
                i += 1;
            });
        }
        st.residual_sum = seqsum(&st.residual);
        st.scores = self.compute_z(path, &st.residual);
        st.gram.as_mut().unwrap().gradients.clone_from(&st.scores);
    }

    fn gaussian_gram_correction(&self, path: &Path, st: &GaussianState, lam: f64) -> f64 {
        let gram = st.gram.as_ref().unwrap();
        let mut largest: f64 = 0.0;
        for j in 0..st.beta.len() {
            if !st.active[j] {
                continue;
            }
            let jj = path.col_map[j];
            let diagonal = gram.columns[j].as_ref().map_or(1.0, |c| c[j]);
            let score = gram.gradients[j] + diagonal * st.beta[j];
            let l1 = lam * self.params.alpha * self.params.penalty_factor[jj];
            let l2 = lam * (1.0 - self.params.alpha) * self.params.penalty_factor[jj];
            let value = if score.abs() <= l1 { 0.0 } else { score.signum() * (score.abs() - l1) / (diagonal + l2) };
            largest = largest.max((value - st.beta[j]).powi(2));
        }
        largest
    }

    fn gaussian_scan(&self, path: &Path, st: &mut GaussianState, lam: f64, strong_only: bool) -> usize {
        let sum = st.residual_sum;
        let p = self.params;
        let residual = &st.residual;
        let beta = &st.beta;
        let gradients = st.gram.as_ref().map(|g| &g.gradients);
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
                *score = if let Some(gradients) = gradients {
                    gradients[j]
                } else {
                    standardized_score(
                        &self.x.col(jj),
                        residual,
                        sum,
                        path.center[jj],
                        path.scale[jj],
                        self.n_fit as f64,
                    )
                };
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

    fn gaussian_lambda(
        &self,
        path: &mut Path,
        st: &mut GaussianState,
        il: usize,
        lam: f64,
        tolerance: f64,
    ) -> Result<(), String> {
        let p = self.params;
        while path.iters[il] < p.max_iter {
            while path.iters[il] < p.max_iter {
                while path.iters[il] < p.max_iter {
                    path.iters[il] += 1;
                    let change = self.gaussian_step(path, st, lam)?;
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
                if st.gram.is_some() {
                    self.gaussian_refresh(path, st);
                    let added = self.gaussian_scan(path, st, lam, true) + self.gaussian_scan(path, st, lam, false);
                    let correction = self.gaussian_gram_correction(path, st, lam);
                    if added != 0 || !(correction < tolerance || correction == 0.0) {
                        continue;
                    }
                }
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
            if st.gram.is_some() && path.iters[il] == p.max_iter {
                self.gaussian_refresh(path, st);
            }
            path.deviance[il] = self.gaussian_loss(st);
        }
        Ok(())
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
        let scores = self.compute_z(&path, &residual);
        let gram = if p.update_from_gram {
            gram_allocation(nc, 1, GRAM_BUDGET)?;
            Some(GramState { gradients: scores.clone(), columns: (0..nc).map(|_| None).collect(), cached: 0 })
        } else {
            None
        };
        let mut st = GaussianState {
            scores,
            gram,
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
                    let change = self.gaussian_step(&path, &mut st, 0.0)?;
                    if change < tolerance || change == 0.0 {
                        if st.gram.is_some() {
                            self.gaussian_refresh(&path, &mut st);
                            let correction = self.gaussian_gram_correction(&path, &st, 0.0);
                            if !(correction < tolerance || correction == 0.0) {
                                continue;
                            }
                        }
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
            self.gaussian_lambda(&mut path, &mut st, il, lam, tolerance)?;
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
    fn gram_budget_checks_each_expansion_and_integer_overflow() {
        assert!(gram_allocation(4, 1, 32).is_ok());
        assert!(gram_allocation(4, 2, 32).unwrap_err().contains("update_from='residuals'"));
        assert!(gram_allocation(usize::MAX, 2, GRAM_BUDGET).is_err());
        assert!(gram_allocation(1, usize::MAX, GRAM_BUDGET).is_err());
        assert!(gram_allocation(11_585, 11_585, GRAM_BUDGET).is_ok());
        assert!(gram_allocation(11_586, 11_586, GRAM_BUDGET).is_err());
    }

    #[test]
    fn refresh_rechecks_active_stationarity_and_caches_only_nonzero_columns() {
        let x = Array2::from_shape_vec((4, 2).f(), vec![-1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0]).unwrap();
        let design = ContigX::<_, true> { x: x.view() };
        let y = [-1.0, -1.0, 1.0, 1.0];
        let penalties = [1.0, 1.0];
        let lambda = [0.25];
        let params = Params {
            family: Family::Gaussian,
            update_from_gram: true,
            alpha: 1.0,
            lambdas: Some(&lambda),
            nlambda: 1,
            eps: 1e-14,
            lambda_min: 0.1,
            lambda_log_scale: true,
            penalty_factor: &penalties,
            dfmax: 2,
            max_iter: 100,
            stop_on_saturation: false,
            r_compatible: false,
            ybar: 0.0,
            null_deviance: 4.0,
            on_lambda: None,
        };
        let fit = Fit { x: &design, y: &y, params: &params, mask: None, rows: None, n_fit: 4 };
        let mut path = Path {
            center: vec![0.0; 2],
            scale: vec![1.0; 2],
            col_map: vec![0, 1],
            lambda: lambda.to_vec(),
            intercept: vec![0.0],
            coef_rows: vec![None],
            coef_vals: vec![None],
            iters: vec![0],
            deviance: vec![0.0],
            n_reject: vec![0],
            events: vec![],
        };
        let mut state = GaussianState {
            residual: Rows::new(4, 0.0),
            residual_sum: 0.0,
            beta: vec![0.0; 2],
            active: vec![true; 2],
            strong: vec![true; 2],
            scores: vec![0.0; 2],
            // Simulate drift that falsely says both active coordinates are
            // stationary. Inactive-only KKT checks cannot catch this.
            gram: Some(GramState { gradients: vec![0.0; 2], columns: vec![None, None], cached: 0 }),
        };
        fit.gaussian_lambda(&mut path, &mut state, 0, 0.25, 1e-14).unwrap();
        assert_eq!(state.beta, [0.75, 0.0]); // orthogonal soft threshold
        assert_eq!(path.deviance, [0.25]); // four residuals of magnitude 1/4
        assert!(path.iters[0] > 1);
        let cache = state.gram.as_ref().unwrap();
        assert_eq!(cache.cached, 1);
        assert!(cache.columns[0].is_some() && cache.columns[1].is_none());
        assert_eq!(cache.gradients, [0.25, 0.0]);
    }

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
                        update_from_gram: false,
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
                        gram: None,
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
