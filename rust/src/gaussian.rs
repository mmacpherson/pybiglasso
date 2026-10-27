//! R biglasso's Gaussian SSR path, sharing column access and fixed-order scores
//! with the logistic solver. Only row/state/path buffers are allocated; X is
//! always borrowed and read one column at a time.

use super::*;

struct GaussianState {
    residual: Rows,
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
            let score =
                standardized_score(&column, &st.residual, seqsum(&st.residual), center, scale, self.n_fit as f64)
                    + st.beta[j];
            st.scores[j] = score;
            let l1 = lam * p.alpha * p.penalty_factor[jj];
            let l2 = lam * (1.0 - p.alpha) * p.penalty_factor[jj];
            let value = if score.abs() <= l1 { 0.0 } else { score.signum() * (score.abs() - l1) / (1.0 + l2) };
            let shift = value - st.beta[j];
            if shift != 0.0 {
                largest = largest.max(shift * shift);
                for i in 0..st.residual.len() {
                    if self.fits(i) {
                        st.residual[i] -= shift * ((column.at(i) - center) / scale);
                    }
                }
                st.beta[j] = value;
            }
        }
        largest
    }

    fn gaussian_scan(&self, path: &Path, st: &mut GaussianState, lam: f64, strong_only: bool) -> usize {
        let sum = seqsum(&st.residual);
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
