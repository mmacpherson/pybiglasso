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
/// Largest error tolerated in a standardized column's mean, which should be
/// exactly 0. Above it, the computed center is inaccurate (for values near 2^54
/// that vary by a few units, say, through the sum's rounding or because no
/// binary64 value is close enough), which can make the fit, its intercept
/// especially, inaccurate; the caller warns. A practical warning threshold,
/// not an accuracy certificate. Also the material-error eligibility threshold
/// for the default mode's one residual-mean correction.
pub(crate) const CENTER_ERROR_MAX: f64 = 1e-6;

fn corrected_center(mean: f64, scale: f64, residual_mean: f64, compatible: bool) -> Option<f64> {
    if compatible
        || !mean.is_finite()
        || !scale.is_finite()
        || scale <= SCALE_MIN
        || (residual_mean / scale).abs() <= CENTER_ERROR_MAX
    {
        return None;
    }
    let corrected = mean + residual_mean;
    (corrected.is_finite() && corrected != mean).then_some(corrected)
}
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
    /// A column of X (by its index) that couldn't be centered precisely
    ImpreciseCenter(usize),
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

/// What center_scale finds.
pub(crate) struct Standardization {
    pub(crate) center: Vec<f64>,
    pub(crate) scale: Vec<f64>,
    /// With a mask, whether the rows it leaves out are safe to sweep (see
    /// LEFT_OUT_Z_MAX); always true otherwise
    pub(crate) mask_safe: bool,
    /// The columns that couldn't be centered precisely
    pub(crate) imprecise: Vec<usize>,
}

/// One column's part of a Standardization.
struct ColumnStandardization {
    center: f64,
    scale: f64,
    /// The rows being fit hold NaN or infinity
    nonfinite: bool,
    /// The rows a mask leaves out are safe to sweep
    left_out_safe: bool,
    /// The mean of the centered values, which should be exactly 0
    center_error: f64,
}

/// A column's sums about a center, over the rows being fit.
struct CenteredSums {
    squares: f64,
    deviations: f64,
}

impl Path {
    /// Adds an event for each column center_scale couldn't center precisely.
    pub(crate) fn note_imprecise(&mut self, columns: Vec<usize>) {
        self.events.splice(0..0, columns.into_iter().map(Event::ImpreciseCenter));
    }
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
    /// precision for columns far from zero; this uses centered moments. R drops
    /// columns holding NaN or infinity, whereas this rejects nonfinite fit rows.
    /// Also returns the columns whose standardized mean, measured in the
    /// final moment pass, is off by more than CENTER_ERROR_MAX.
    pub(crate) fn center_scale(&self) -> Result<Standardization, String> {
        let columns: Vec<ColumnStandardization> =
            (0..self.x.ncol()).into_par_iter().map(|j| self.standardize_column(&self.x.col(j))).collect();
        if columns.iter().any(|c| c.nonfinite) {
            return Err("X must contain only finite values (no NaN or infinity) in the rows used for fitting.".into());
        }
        if columns.iter().any(|c| !c.center.is_finite() || !c.scale.is_finite()) {
            return Err("X has values too large to standardize safely.".into());
        }
        let imprecise = (columns.iter().enumerate())
            .filter(|(_, c)| c.scale > SCALE_MIN && (c.center_error / c.scale).abs() > CENTER_ERROR_MAX)
            .map(|(j, _)| j)
            .collect();
        Ok(Standardization {
            center: columns.iter().map(|c| c.center).collect(),
            scale: columns.iter().map(|c| c.scale).collect(),
            mask_safe: columns.iter().all(|c| c.left_out_safe),
            imprecise,
        })
    }

    fn standardize_column<C: Col>(&self, c: &C) -> ColumnStandardization {
        let nf = self.n_fit as f64;
        let (mut total, mut nonfinite) = (0.0, false);
        self.for_each_fitted(c, |v| {
            nonfinite |= !v.is_finite();
            total += v;
        });
        let mut center = total / nf;
        // deviations: what the centered values leave over, which binary64
        // rounding of the center (or of its sum) leaves nonzero
        let mut sums = self.centered_sums(c, center);
        let mut scale = (sums.squares / nf).sqrt();
        if !nonfinite
            && let Some(corrected) = corrected_center(center, scale, sums.deviations / nf, self.params.r_compatible)
        {
            // Differs from R biglasso: R retains the raw-sum mean.
            // Default mode partially repairs a material residual mean;
            // ordinary columns and compatibility keep their old bits.
            // Re-read in fitted-row order: updating the old moments
            // algebraically can cancel, and may not describe the actual
            // rounded differences about this new, one-float center.
            center = corrected;
            sums = self.centered_sums(c, center);
            scale = (sums.squares / nf).sqrt();
        }
        let (mut out_lo, mut out_hi, mut out_finite) = (f64::INFINITY, f64::NEG_INFINITY, true);
        let mut hint = 0;
        for i in (0..c.n()).filter(|&i| !self.fits(i)) {
            let v = c.at_seq(i, &mut hint);
            out_finite &= v.is_finite();
            (out_lo, out_hi) = (out_lo.min(v), out_hi.max(v));
        }
        // columns too constant to use are never swept
        let out_deviation = if out_lo > out_hi { 0.0 } else { (out_lo - center).abs().max((out_hi - center).abs()) };
        ColumnStandardization {
            center,
            scale,
            nonfinite,
            left_out_safe: scale <= SCALE_MIN || (out_finite && out_deviation <= LEFT_OUT_Z_MAX * scale),
            center_error: sums.deviations / nf,
        }
    }

    /// The sums of squares and of deviations about `center`, in the order
    /// for_each_fitted visits the rows.
    fn centered_sums<C: Col>(&self, c: &C, center: f64) -> CenteredSums {
        let mut sums = CenteredSums { squares: 0.0, deviations: 0.0 };
        self.for_each_fitted(c, |v| {
            let d = v - center;
            sums.squares += d.powi(2);
            sums.deviations += d;
        });
        sums
    }

    /// Visits a column's values in the rows being fit: in row_indices' order
    /// with a mask, as gathering does, otherwise in row order.
    #[inline(always)]
    fn for_each_fitted<C: Col>(&self, c: &C, mut visit: impl FnMut(f64)) {
        match self.rows {
            Some(rows) => {
                let mut hint = 0;
                rows.iter().for_each(|&i| visit(c.at_seq(i, &mut hint)));
            }
            None => for_each_value(c, visit),
        }
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
