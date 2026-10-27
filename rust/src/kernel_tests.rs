//! Independent formula and boundary probes formerly available only via numba.
//! The fused row passes, intercept, IRLS and SIMD bits have separate tests.rs checks.

use super::*;
use ndarray::{Array2, ShapeBuilder};

fn params(penalty_factor: &[f64]) -> Params<'_> {
    Params {
        family: Family::Binomial,
        alpha: 1.0,
        lambdas: None,
        nlambda: 1,
        eps: 1e-14,
        lambda_min: 0.05,
        lambda_log_scale: true,
        penalty_factor,
        dfmax: penalty_factor.len() + 1,
        max_iter: 10_000,
        stop_on_saturation: false,
        r_compatible: false,
        ybar: 0.5,
        null_deviance: 1.0,
        on_lambda: None,
    }
}

fn path(center: Vec<f64>, scale: Vec<f64>, col_map: Vec<usize>) -> Path {
    Path {
        center,
        scale,
        col_map,
        lambda: vec![0.0],
        intercept: vec![0.0],
        coef_rows: vec![None],
        coef_vals: vec![None],
        iters: vec![0],
        deviance: vec![0.0],
        n_reject: vec![0],
        events: vec![],
    }
}

fn state(n: usize, p: usize) -> State {
    State {
        eta: Rows::new(n, 0.0),
        irls_work: Rows::new(n, 0.0),
        w: Rows::new(n, 0.0),
        s: Rows::new(n, 0.0),
        r: Rows::new(n, 0.0),
        beta: vec![0.0; p],
        active: vec![0; p],
        strong: vec![0; p],
        z: vec![0.0; p],
        intercept: 0.0,
        sum_w_residuals: 0.0,
    }
}

#[test]
fn standardization_and_initial_scores_use_selected_rows_and_raw_column_map() {
    // Both full data and the deliberately permuted/repeated subset balance
    // signs: columns have known centers/scales and exact dyadic crossproducts.
    let a = Array2::<f32>::from_shape_fn((16, 4).f(), |(i, j)| {
        let a = if i % 2 == 0 { -1.0 } else { 1.0 };
        let b = if (i / 2) % 2 == 0 { -1.0 } else { 1.0 };
        [7.0, 3.0 + 2.0 * a, -4.0 + b, 2.0 - 4.0 * a][j]
    });
    let selected = [15, 2, 5, 0, 15, 2, 5, 0];
    let pf = [1.0; 4];
    let p = params(&pf);
    let all = ContigX::<f32, true> { x: a.view() };
    let gather = GatherX { x: ContigX::<f32, true> { x: a.view() }, idx: &selected };
    fn check<D: Design>(x: &D, p: &Params, source_rows: &[usize]) {
        let y: Vec<f64> = source_rows.iter().map(|&i| (i % 2) as f64).collect();
        let fit = Fit { x, y: &y, params: p, mask: None, rows: None, n_fit: source_rows.len() };
        let (center, scale, safe) = fit.center_scale().unwrap();
        assert!(safe);
        assert_eq!(center, [7.0, 3.0, -4.0, 2.0]);
        assert_eq!(scale, [0.0, 2.0, 1.0, 4.0]);
        let path = path(center, scale, vec![3, 1, 2]);
        assert_eq!(fit.compute_z(&path, &y), [-0.5, 0.5, 0.0]);
    }
    check(&all, &p, &(0..16).collect::<Vec<_>>());
    check(&gather, &p, &selected);
}

#[test]
fn centered_moments_and_score_avoid_large_offset_cancellation() {
    let center = 2f64.powi(30);
    let x: Vec<f64> = (0..16).map(|i| center + if i % 2 == 0 { -2.0 } else { 2.0 }).collect();
    let r: Vec<f64> = (0..16).map(|i| if i % 2 == 0 { -0.25 } else { 0.25 }).collect();
    let w = [0.125; 16];
    let col = Contig::<_, true>(&x);
    // z=+/-1, r=z/4: sum(z*w*r)=1/2, mean(w*z^2)=1/8.
    assert_eq!(centered_moments(&col, &r, &w, center, 2.0, 16.0), (0.5, 0.125));
    assert_eq!(standardized_score(&col, &r, 0.0, center, 2.0, 16.0), 0.25);
    // With a nonzero residual mean, the raw dot and center*sum(r) are both
    // about 2^34. Their difference loses this exact small correlation.
    let small = 2f64.powi(-25);
    let r: Vec<f64> = (0..16).map(|i| 1.0 + if i % 2 == 0 { -small } else { small }).collect();
    assert_eq!(standardized_score(&col, &r, 16.0, center, 2.0, 16.0), small);
}

fn close(actual: f64, expected: f64, operations: usize) {
    // Standard-model gamma_m, using twice unit roundoff conservatively.
    // Callers supply an operation budget for these small normalized fixtures;
    // this is not a universal bound for arbitrarily ill-conditioned inputs.
    let m_u = operations as f64 * f64::EPSILON;
    let allowance = m_u / (1.0 - m_u) * (1.0 + expected.abs());
    assert!((actual - expected).abs() <= allowance, "actual={actual}, expected={expected}, allowance={allowance}");
}

fn check_sweep<D: Design>(x: &D, source_rows: &[usize], mask: Option<&[f64]>, offset: f64, compat: bool) {
    let n = source_rows.len();
    let n_fit = mask.map_or(n, |m| m.iter().filter(|&&v| v != 0.0).count());
    // The original columns include an unused constant, an inactive slope,
    // zero and unequal penalties, and nonunit raw scales. All standardized
    // columns are orthogonal +/-1 on both the full rows and mask subset.
    let pf = [32.0, 1.0, 0.0, 9.0, 0.5];
    let mut p = params(&pf);
    p.alpha = 0.25;
    p.r_compatible = compat;
    let path = path(vec![7.0, offset, offset, offset, offset], vec![0.0, 2.0, 4.0, 1.0, 0.5], vec![1, 2, 3, 4]);
    let y = vec![0.0; n];
    let fit = Fit { x, y: &y, params: &p, mask, rows: None, n_fit };
    let mut st = state(n, 4);
    st.beta = vec![0.25, -0.125, 0.0, 0.5];
    st.active = vec![1, 1, 0, 1];
    for i in 0..n {
        st.r[i] = (source_rows[i] % 7) as f64 / 4.0 - 0.75;
        st.eta[i] = (source_rows[i] % 3) as f64 / 8.0;
        st.w[i] = if mask.is_some_and(|m| m[i] == 0.0) { 0.0 } else { 0.125 };
    }
    let mut beta = st.beta.clone();
    let (mut r, mut eta) = (st.r.to_vec(), st.eta.to_vec());
    let mut max_update: f64 = 0.0;
    let lam = 0.125;
    for j in [0, 1, 3] {
        let z: Vec<f64> = source_rows.iter().map(|&i| if (i >> j) % 2 == 0 { -1.0 } else { 1.0 }).collect();
        let v = z.iter().zip(&*st.w).map(|(&z, &w)| w * z * z).sum::<f64>() / n_fit as f64;
        let score = z.iter().zip(r.iter().zip(&*st.w)).map(|(&z, (&r, &w))| z * w * r).sum::<f64>() / n_fit as f64;
        let u = score + v * beta[j];
        let l1 = lam * p.alpha * pf[j + 1];
        let l2 = lam * (1.0 - p.alpha) * pf[j + 1];
        let denominator = if compat { v * (1.0 + l2) } else { v + l2 };
        let value = u.signum() * (u.abs() - l1).max(0.0) / denominator;
        let shift = value - beta[j];
        for i in 0..n {
            r[i] -= shift * z[i];
            eta[i] += shift * z[i];
        }
        beta[j] = value;
        max_update = max_update.max(v * shift * shift);
    }
    let sum_w = st.w.iter().sum();
    st.sum_w_residuals = st.w.iter().zip(&*st.r).map(|(&w, &r)| w * r).sum();
    let got_max = fit.coordinate_sweep(&path, &mut st, lam, sum_w).unwrap();
    // Each of three coordinates uses O(n) products/additions. Here v=1/8,
    // the standardized Gram is I, and there is no conditioning amplification
    // across coordinates. 64n covers the two reduction orders, the division
    // by v, and reconstruction of r, eta and the update measure.
    let budget = 64 * n;
    for j in 0..4 {
        close(st.beta[j], beta[j], budget);
    }
    for i in 0..n {
        close(st.r[i], r[i], budget);
        close(st.eta[i], eta[i], budget);
    }
    close(st.sum_w_residuals, st.w.iter().zip(&r).map(|(&w, &r)| w * r).sum(), budget);
    close(got_max, max_update, budget);
}

#[test]
fn coordinate_sweep_matches_standardized_formulas_in_both_modes_and_row_paths() {
    for offset in [0.0, 2f64.powi(30)] {
        let a = Array2::from_shape_fn((32, 5).f(), |(i, j)| {
            if j == 0 {
                7.0
            } else {
                offset + [2.0, 4.0, 1.0, 0.5][j - 1] * if (i >> (j - 1)) % 2 == 0 { -1.0 } else { 1.0 }
            }
        });
        let all = ContigX::<f64, true> { x: a.view() };
        let rows: Vec<usize> = (0..48).map(|i| (i * 5 + 3) % 16).collect();
        let gather = GatherX { x: ContigX::<f64, true> { x: a.view() }, idx: &rows };
        let mask: Vec<f64> = (0..32).map(|i| if i < 16 { 1.0 } else { 0.0 }).collect();
        for compat in [false, true] {
            check_sweep(&all, &(0..32).collect::<Vec<_>>(), None, offset, compat);
            check_sweep(&all, &(0..32).collect::<Vec<_>>(), Some(&mask), offset, compat);
            check_sweep(&gather, &rows, None, offset, compat);
        }
    }
}

#[test]
fn coordinate_sweep_rejects_invalid_curvature() {
    let a = Array2::from_shape_vec((4, 1).f(), vec![-1.0, 1.0, -1.0, 1.0]).unwrap();
    let x = ContigX::<f64, true> { x: a.view() };
    let pf = [1.0];
    let p = params(&pf);
    let y = [0.0; 4];
    let fit = Fit { x: &x, y: &y, params: &p, mask: None, rows: None, n_fit: 4 };
    let path = path(vec![0.0], vec![1.0], vec![0]);
    for weight in [0.0, f64::NAN] {
        let mut st = state(4, 1);
        st.active[0] = 1;
        st.w.fill(weight);
        assert!(fit.coordinate_sweep(&path, &mut st, 0.1, weight * 4.0).unwrap_err().contains("curvature"));
    }
}

#[test]
fn curvature_fallback_handles_positive_and_zero_cancelled_moments() {
    let a = Array2::from_shape_vec((4, 1).f(), vec![0.0, 1.0, 1.0, 2.0]).unwrap();
    let x = ContigX::<f64, true> { x: a.view() };
    let pf = [1.0];
    let p = params(&pf);
    let y = [0.0; 4];
    let fit = Fit { x: &x, y: &y, params: &p, mask: None, rows: None, n_fit: 4 };
    let scale = 0.5f64.sqrt();
    let path = path(vec![1.0], vec![scale], vec![0]);
    for tiny in [1e-9, 1e-16] {
        let mut st = state(4, 1);
        st.active[0] = 1;
        st.w.copy_from_slice(&[tiny, 1.0, 1.0, tiny]);
        st.r.copy_from_slice(&[-1.0, 0.0, 0.0, 1.0]);
        let sum_w = st.w.iter().sum();
        fit.coordinate_sweep(&path, &mut st, 0.0, sum_w).unwrap();
        // x-center=(-1,0,0,1): the score/curvature ratio is exactly scale.
        close(st.beta[0], scale, 32);
    }
}

#[test]
fn kkt_scans_are_strict_at_adjacent_threshold_values() {
    let a = Array2::from_shape_fn((8, 2).f(), |(i, j)| {
        if j == 0 {
            7.0
        } else if i % 2 == 0 {
            -1.0
        } else {
            1.0
        }
    });
    let x = ContigX::<f64, true> { x: a.view() };
    let pf = [32.0, 1.0];
    let p = params(&pf);
    let y = [0.0; 8];
    let fit = Fit { x: &x, y: &y, params: &p, mask: None, rows: None, n_fit: 8 };
    let path = path(vec![7.0, 0.0], vec![0.0, 1.0], vec![1]);
    for power in -5..=-1 {
        let threshold = 2f64.powi(power);
        for (score, violates) in [(threshold.next_down(), false), (threshold, false), (threshold.next_up(), true)] {
            for sign in [-1.0, 1.0] {
                for strong_only in [false, true] {
                    let mut st = state(8, 1);
                    st.strong[0] = strong_only as u8;
                    for i in 0..8 {
                        st.s[i] = sign * score * if i % 2 == 0 { -1.0 } else { 1.0 };
                    }
                    let count = fit.check(&path, &mut st, threshold, strong_only);
                    assert_eq!(count, violates as usize);
                    assert_eq!(st.z[0], sign * score);
                    assert_eq!(st.active[0], violates as u8);
                    assert_eq!(st.strong[0], (strong_only || violates) as u8);
                }
            }
        }
    }
}

#[test]
fn kkt_scans_preserve_ineligible_scores_and_apply_elastic_net_penalties() {
    // All used columns have the same exact score, so only eligibility and
    // raw-indexed penalties/beta distinguish their decisions.
    let a = Array2::from_shape_fn((8, 5).f(), |(i, j)| {
        if j == 0 {
            7.0
        } else if i % 2 == 0 {
            -1.0
        } else {
            1.0
        }
    });
    let x = ContigX::<f64, true> { x: a.view() };
    let pf = [32.0, 1.0, 0.125, 4.0, 0.125];
    let mut p = params(&pf);
    p.alpha = 0.5;
    let y = [0.0; 8];
    let fit = Fit { x: &x, y: &y, params: &p, mask: None, rows: None, n_fit: 8 };
    let path = path(vec![7.0, 0.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 1.0, 1.0, 1.0], vec![1, 2, 3, 4]);
    for strong_only in [false, true] {
        let mut st = state(8, 4);
        st.active = vec![1, 0, 0, 0];
        st.strong = vec![1, 1, 1, 0];
        st.beta = vec![0.0, 7.0, 0.0, 0.0];
        st.z.fill(123.0);
        for i in 0..8 {
            st.s[i] = if i % 2 == 0 { -0.25 } else { 0.25 };
        }
        assert_eq!(fit.check(&path, &mut st, 0.5, strong_only), if strong_only { 0 } else { 1 });
        assert_eq!(st.z, if strong_only { vec![123.0, 0.25, 0.25, 123.0] } else { vec![123.0, 123.0, 123.0, 0.25] });
        assert_eq!(st.active, if strong_only { vec![1, 0, 0, 0] } else { vec![1, 0, 0, 1] });
        assert_eq!(st.strong, if strong_only { vec![1, 1, 1, 0] } else { vec![1, 1, 1, 1] });
    }
}

fn reentry_residual(st: &State, x: &Array2<f64>, y: &[f64], lam: f64) -> f64 {
    let mut score = [0.0; 3];
    for i in 0..y.len() {
        let z = [x[[i, 0]], x[[i, 1]] / 5f64.sqrt()];
        let eta = st.intercept + z[0] * st.beta[0] + z[1] * st.beta[1];
        let probability = 1.0 / (1.0 + libm::exp(-eta));
        let error = y[i] - probability;
        score[0] += error;
        for j in 0..2 {
            score[j + 1] += z[j] * error;
        }
    }
    let mut maximum = (score[0] / y.len() as f64).abs();
    for j in 0..2 {
        let s = score[j + 1] / y.len() as f64;
        let l1 = lam * [1.0, 1.0 / 16.0][j];
        let error = if st.beta[j] == 0.0 { (s.abs() - l1).max(0.0) } else { (s - l1 * st.beta[j].signum()).abs() };
        maximum = maximum.max(error);
    }
    maximum
}

#[test]
fn independent_optimality_check_detects_missing_reentry_scans() {
    // Orthogonal x1,q, with proportions .5+x1/4-q/8, give initial scores
    // (1/4,0). At lambda=3/16 and pf=(1,1/16), only x1 is strong. Fitting
    // x1 makes the rejected x2=x1+2q score -1/(16*sqrt(5)), beyond 3/256.
    let mut a = Array2::<f64>::zeros((32, 2).f());
    let mut y = vec![0.0; 32];
    for i in 0..32 {
        let cell = i / 8;
        let x1 = if cell % 2 == 0 { 1.0 } else { -1.0 };
        let q = if cell < 2 { 1.0 } else { -1.0 };
        a[[i, 0]] = x1;
        a[[i, 1]] = x1 + 2.0 * q;
        y[i] = ((i % 8) < (4.0 + 2.0 * x1 - q) as usize) as u8 as f64;
    }
    let x = ContigX::<f64, true> { x: a.view() };
    let pf = [1.0, 1.0 / 16.0];
    let mut p = params(&pf);
    p.null_deviance = 32.0 * std::f64::consts::LN_2;
    let fit = Fit { x: &x, y: &y, params: &p, mask: None, rows: None, n_fit: 32 };
    let mut path = path(vec![0.0, 0.0], vec![1.0, 5f64.sqrt()], vec![0, 1]);
    let lam = 3.0 / 16.0;
    let threshold = p.eps * p.null_deviance / 32.0;
    for check_kkt in [false, true] {
        for signal_active in [false, true] {
            let mut st = state(32, 2);
            st.strong[0] = 1;
            st.active[0] = signal_active as u8;
            path.iters[0] = 0;
            assert!(fit.fit_lambda(&mut path, &mut st, 0, lam, threshold, check_kkt).unwrap() == Status::Converged);
            let error = reentry_residual(&st, &a, &y, lam);
            if check_kkt {
                // Same independent standardized KKT ceiling as the Python
                // certificate suite, not a coefficient expectation.
                assert!(error <= 1e-6, "KKT residual {error}");
                assert!(st.beta[0] > 0.0 && st.beta[1] < 0.0);
            } else {
                // Missing strong scan: the signal never enters. Missing
                // rest scan after fitting it: the suppressor never enters.
                // Both analytic violations exceed 1/64 at this fixture.
                assert!(error > 1.0 / 64.0, "KKT residual {error}");
                assert_eq!(st.beta[1], 0.0);
            }
        }
    }
}

#[test]
fn final_loss_is_stable_at_extreme_logits_and_skips_left_out_rows() {
    let a = Array2::<f64>::zeros((5, 1).f());
    let x = ContigX::<f64, true> { x: a.view() };
    let pf = [1.0];
    let p = params(&pf);
    let y = [0.0, 1.0, 1.0, 0.0, f64::NAN];
    let mask = [1.0, 1.0, 1.0, 1.0, 0.0];
    let fit = Fit { x: &x, y: &y, params: &p, mask: Some(&mask), rows: None, n_fit: 4 };
    let mut st = state(5, 1);
    st.eta.copy_from_slice(&[-1000.0, 1000.0, -1000.0, 1000.0, f64::NAN]);
    assert_eq!(fit.final_loss(&st), 2000.0);
}

#[test]
#[should_panic(expected = "out of range")]
fn checked_contiguous_block_rejects_a_short_column() {
    Contig::<_, true>(&[0.0; LANES - 1]).block(0);
}

#[test]
#[should_panic(expected = "out of bounds")]
fn checked_gather_rejects_an_out_of_range_source_row() {
    let col = Contig::<_, true>(&[1.0, 2.0]);
    Gather { col, idx: &[0, 2] }.at(1);
}

#[test]
#[should_panic(expected = "out of bounds")]
fn checked_strided_block_rejects_an_out_of_range_source_row() {
    let a = Array2::<f64>::zeros((LANES - 1, 3));
    Strided::<_, true>(a.column(1)).block(0);
}
