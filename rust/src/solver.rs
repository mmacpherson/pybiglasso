//! What both families' solvers share: the path's parameters and result,
//! standardization, the scores the strong rule and KKT checks use, and `run`,
//! which hands the fit to its family (binomial.rs or gaussian.rs). Column
//! access is in columns.rs, and the fixed-order sums in kernels.rs.

use ndarray::{ArrayView1, ArrayView2};
// Portable exp/log/log1p keep the Rust path's bits independent of the
// platform math library, whose last-bit rounding varies by OS.
use libm::{exp, log, log1p};
use pulp::{Simd, WithSimd};
use rayon::prelude::*;

#[path = "binomial.rs"]
mod binomial;
#[path = "columns.rs"]
mod columns;
#[path = "gaussian.rs"]
mod gaussian;
#[path = "gaussian_bound.rs"]
mod gaussian_bound;
#[path = "kernels.rs"]
mod kernels;

use binomial::*;
pub(crate) use columns::*;
use kernels::*;

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Family {
    Binomial,
    Gaussian,
}

impl Family {
    // Used by the Python/JS bindings, which are exercised outside native tests.
    #[cfg_attr(test, allow(dead_code))]
    pub(crate) fn parse(name: &str) -> Result<Self, &'static str> {
        match name {
            "binomial" => Ok(Self::Binomial),
            "gaussian" => Ok(Self::Gaussian),
            _ => Err("family must be 'binomial' or 'gaussian'."),
        }
    }

    pub(crate) fn null_model(self, y: &[f64], rows: Option<&[usize]>) -> Result<(f64, f64), &'static str> {
        if self == Self::Binomial {
            return null_model(y, rows);
        }
        let n = rows.map_or(y.len(), <[usize]>::len);
        if n == 0 {
            return Err("y must contain at least one selected response.");
        }
        let at = |i: usize| y[rows.map_or(i, |r| r[i])];
        let mut total = 0.0;
        for i in 0..n {
            let value = at(i);
            if !value.is_finite() {
                return Err("Gaussian y must be finite in the rows used for fitting.");
            }
            total += value;
        }
        let mean = total / n as f64;
        let mut rss = 0.0;
        for i in 0..n {
            rss += (at(i) - mean).powi(2);
        }
        if !mean.is_finite() || !rss.is_finite() {
            return Err("Gaussian y has values too large to fit safely.");
        }
        Ok((mean, rss))
    }
}

const SCALE_MIN: f64 = 1e-6;
/// row_indices naming at least this share of X's rows, each once, are fit as
/// a 0/1 weight on every row, so the kernels read whole columns with vector
/// loads; fewer rows, or repeated ones, are gathered. Part of the results:
/// the two group sums differently, so changing it changes their last bits.
/// Half is about where they break even (AMD Ryzen 5 7640U, one thread: there
/// the mask is 9% faster on wide data and 3% slower on tall, and either way
/// the gap widens quickly with the share).
// The Python driver chooses mask versus gather; pytest covers that boundary.
#[cfg_attr(test, allow(dead_code))]
pub(crate) const MASK_SHARE: f64 = 0.5;
/// The mask sweeps the rows it leaves out with weight 0, where each enters
/// only as a product with a 0 (w, s, and so the residual's terms), and those
/// are exactly 0 while x, z = (x - center) / scale and the residual stay
/// finite. Left-out values within this many scales of the center keep z^2
/// finite, and the residual's drift (each step's standardized change times z)
/// unless a step exceeds 1e150; otherwise the fit gathers instead.
const LEFT_OUT_Z_MAX: f64 = 3e150;

// The solver

pub(crate) struct Params<'a> {
    pub(crate) family: Family,
    pub(crate) alpha: f64,
    pub(crate) lambdas: Option<&'a [f64]>,
    pub(crate) nlambda: usize,
    pub(crate) eps: f64,
    pub(crate) lambda_min: f64,
    pub(crate) lambda_log_scale: bool,
    pub(crate) penalty_factor: &'a [f64],
    pub(crate) dfmax: usize,
    pub(crate) max_iter: i64,
    pub(crate) stop_on_saturation: bool,
    pub(crate) r_compatible: bool,
    pub(crate) ybar: f64,
    pub(crate) null_deviance: f64,
    /// Called after each lambda is fit; true ends the path there
    pub(crate) on_lambda: Option<&'a OnLambda<'a>>,
}

pub(crate) type OnLambda<'a> = dyn Fn(&Path, usize) -> Result<bool, String> + Sync + 'a;

/// What the Python side needs to warn about.
pub(crate) enum Event {
    PrefitSaturated,
    Dfmax(usize),
    Saturated(usize),
    MaxIter(usize),
}

pub(crate) struct Path {
    pub(crate) center: Vec<f64>,
    pub(crate) scale: Vec<f64>,
    pub(crate) col_map: Vec<usize>,
    pub(crate) lambda: Vec<f64>,
    pub(crate) intercept: Vec<f64>,
    pub(crate) coef_rows: Vec<Option<Vec<usize>>>,
    pub(crate) coef_vals: Vec<Option<Vec<f64>>>,
    pub(crate) iters: Vec<i64>,
    pub(crate) deviance: Vec<f64>,
    pub(crate) n_reject: Vec<i64>,
    pub(crate) events: Vec<Event>,
}

pub(crate) struct Fit<'a, D> {
    pub(crate) x: &'a D,
    pub(crate) y: &'a [f64],
    pub(crate) params: &'a Params<'a>,
    /// With row_indices fit as a 0/1 weight on every row of X: 1 for rows
    /// being fit. The kernels then read whole columns, and y is 0 elsewhere.
    pub(crate) mask: Option<&'a [f64]>,
    /// With a mask, the rows being fit in row_indices' order, which center
    /// and scale sum in, as gathering does: for a column far from zero, the
    /// center's rounding moves every centered value
    pub(crate) rows: Option<&'a [usize]>,
    /// The number of rows being fit
    pub(crate) n_fit: usize,
}

impl<D: Design> Fit<'_, D> {
    /// Each column's center and scale, from the rows being fit; and, with a
    /// mask, whether the rows it leaves out are safe to sweep (see
    /// LEFT_OUT_Z_MAX). X may hold anything in rows that aren't fit.
    /// Differs from R biglasso: R derives scales from raw sums, which can lose
    /// precision for columns far from zero; this uses two passes. R drops
    /// columns holding NaN or infinity, whereas this rejects nonfinite fit rows.
    pub(crate) fn center_scale(&self) -> Result<(Vec<f64>, Vec<f64>, bool), String> {
        let stats: Vec<(f64, f64, bool, bool)> = (0..self.x.ncol())
            .into_par_iter()
            .map(|j| {
                let c = self.x.col(j);
                let n = c.n();
                let nf = self.n_fit as f64;
                let (mut acc, mut invalid) = (0.0, false);
                let mut hint = 0;
                match self.rows {
                    Some(rows) => rows.iter().for_each(|&i| {
                        let v = c.at_seq(i, &mut hint);
                        invalid |= !v.is_finite();
                        acc += v;
                    }),
                    None => for_each_value(&c, |v| {
                        invalid |= !v.is_finite();
                        acc += v;
                    }),
                }
                let mean = acc / nf;
                let mut acc = 0.0;
                let mut hint = 0;
                match self.rows {
                    Some(rows) => rows.iter().for_each(|&i| acc += (c.at_seq(i, &mut hint) - mean).powi(2)),
                    None => for_each_value(&c, |v| acc += (v - mean).powi(2)),
                }
                let (mut out_lo, mut out_hi, mut out_finite) = (f64::INFINITY, f64::NEG_INFINITY, true);
                let mut hint = 0;
                for i in (0..n).filter(|&i| !self.fits(i)) {
                    let v = c.at_seq(i, &mut hint);
                    out_finite &= v.is_finite();
                    (out_lo, out_hi) = (out_lo.min(v), out_hi.max(v));
                }
                let scale = (acc / nf).sqrt();
                // columns too constant to use are never swept
                let out_deviation =
                    if out_lo > out_hi { 0.0 } else { (out_lo - mean).abs().max((out_hi - mean).abs()) };
                let out_safe = scale <= SCALE_MIN || (out_finite && out_deviation <= LEFT_OUT_Z_MAX * scale);
                (mean, scale, invalid, out_safe)
            })
            .collect();
        if stats.iter().any(|s| s.2) {
            return Err("X must contain only finite values (no NaN or infinity) in the rows used for fitting.".into());
        }
        if stats.iter().any(|s| !s.0.is_finite() || !s.1.is_finite()) {
            return Err("X has values too large to standardize safely.".into());
        }
        let (center, scale) = stats.iter().map(|s| (s.0, s.1)).unzip();
        Ok((center, scale, stats.iter().all(|s| s.3)))
    }

    fn score_sum(&self, values: &[f64]) -> f64 {
        if self.params.family == Family::Gaussian && !self.params.r_compatible {
            return gaussian::blocked_sum(values);
        }
        seqsum(values)
    }

    fn compute_z(&self, path: &Path, s: &[f64]) -> Vec<f64> {
        let s_sum = self.score_sum(s);
        let nf = self.n_fit as f64;
        path.col_map
            .par_iter()
            .with_min_len(16)
            .map(|&jj| standardized_score(&self.x.col(jj), s, s_sum, path.center[jj], path.scale[jj], nf))
            .collect()
    }

    #[inline(always)]
    fn fits(&self, i: usize) -> bool {
        self.mask.is_none_or(|m| m[i] != 0.0)
    }

    /// Whether params.on_lambda ends the path after lambda il; if so, the
    /// rest of the path is marked unfit.
    fn stopped(&self, path: &mut Path, il: usize) -> Result<bool, String> {
        let Some(on_lambda) = self.params.on_lambda else { return Ok(false) };
        if !on_lambda(path, il)? {
            return Ok(false);
        }
        path.iters[il + 1..].fill(-1);
        Ok(true)
    }

    pub(crate) fn run(&self, center: Vec<f64>, scale: Vec<f64>) -> Result<Path, String> {
        if self.params.family == Family::Gaussian {
            return self.run_gaussian(center, scale);
        }
        self.run_binomial(center, scale)
    }
}

/// numpy.linspace(start, stop, num), element for element.
fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num == 1 {
        return vec![start];
    }
    let div = (num - 1) as f64;
    let delta = stop - start;
    let step = delta / div;
    let mut y: Vec<f64> =
        (0..num).map(|i| if step == 0.0 { i as f64 / div * delta } else { i as f64 * step } + start).collect();
    y[num - 1] = stop;
    y
}
