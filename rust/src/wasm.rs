//! Browser binding. A dataset is ingested once; every path borrows its X.

use super::model_scale::rounded_sum;
use super::solver::*;
use ndarray::{ArrayView2, ShapeBuilder};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

// This is an application budget, not an advertised browser allocation limit.
const MAX_VALUES: usize = 16 * 1024 * 1024;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = globalThis, js_name = pybiglassoProgress)]
    fn progress(done: usize, total: usize);
}

#[derive(Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Options {
    family: String,
    alpha: f64,
    nlambda: usize,
    eps: f64,
    lambda_min: Option<f64>,
    lambdas: Option<Vec<f64>>,
    penalty_factor: Option<Vec<f64>>,
    max_iter: i64,
    dfmax: Option<usize>,
    r_compatible: bool,
    /// Fit only these rows, as a cross-validation fold does
    train_rows: Option<Vec<usize>>,
    /// Score every fitted lambda on these rows
    test_rows: Option<Vec<usize>>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            family: "binomial".into(),
            alpha: 1.0,
            nlambda: 100,
            eps: 1e-7,
            lambda_min: None,
            lambdas: None,
            penalty_factor: None,
            max_iter: 10000,
            dfmax: None,
            r_compatible: false,
            train_rows: None,
            test_rows: None,
        }
    }
}

#[derive(Serialize)]
struct ModelPath {
    family: String,
    lambdas: Vec<f64>,
    intercept: Vec<f64>,
    indices: Vec<Vec<usize>>,
    coefficients: Vec<Vec<f64>>,
    standardized_coefficients: Vec<Vec<f64>>,
    center: Vec<f64>,
    scale: Vec<f64>,
    iterations: Vec<i64>,
    n_rejected: Vec<i64>,
    deviance: Vec<f64>,
    /// With test_rows, each fit's held-out loss per row: deviance for
    /// binomial, squared error for gaussian
    holdout: Option<Vec<f64>>,
    warnings: Vec<String>,
}

#[wasm_bindgen]
pub struct Dataset {
    nrow: usize,
    ncol: usize,
    x: Vec<f64>,
    y: Vec<f64>,
}

#[wasm_bindgen]
impl Dataset {
    #[wasm_bindgen(constructor)]
    pub fn new(nrow: usize, ncol: usize) -> Result<Dataset, JsValue> {
        let count = nrow.checked_mul(ncol).filter(|&n| n <= MAX_VALUES);
        if !(2..=1_000_000).contains(&nrow) || ncol == 0 || ncol > 20_000 || count.is_none() {
            return Err(JsValue::from_str(
                "Use at least two rows and one feature, with at most 128 MiB of decoded float64 features.",
            ));
        }
        Ok(Self { nrow, ncol, x: vec![f64::NAN; count.unwrap()], y: vec![f64::NAN; nrow] })
    }

    pub fn write_column(&mut self, column: usize, start: usize, values: &[f64]) -> Result<(), JsValue> {
        if column >= self.ncol
            || start.checked_add(values.len()).is_none_or(|end| end > self.nrow)
            || values.iter().any(|x| !x.is_finite())
        {
            return Err(JsValue::from_str(
                "A feature is missing, nonnumeric, nonfinite, or outside the dataset's dimensions.",
            ));
        }
        self.x[column * self.nrow + start..column * self.nrow + start + values.len()].copy_from_slice(values);
        Ok(())
    }

    pub fn write_response(&mut self, start: usize, values: &[f64]) -> Result<(), JsValue> {
        if start.checked_add(values.len()).is_none_or(|end| end > self.nrow) || values.iter().any(|y| !y.is_finite()) {
            return Err(JsValue::from_str("The response must be finite, with one value per row."));
        }
        self.y[start..start + values.len()].copy_from_slice(values);
        Ok(())
    }

    pub fn fit(&self, options: JsValue) -> Result<JsValue, JsValue> {
        self.fit_inner(options).map_err(|e| JsValue::from_str(&e))
    }
}

impl Dataset {
    /// A path on X stored column-major with nrow rows, over every row or,
    /// with a mask, over the rows it weights; None when the mask is unsafe.
    fn run(
        &self,
        x: &[f64],
        y: &[f64],
        nrow: usize,
        params: &Params,
        mask: Option<(&[f64], &[usize])>,
    ) -> Result<Option<Path>, String> {
        let x =
            ContigX::<f64, true> { x: ArrayView2::from_shape((nrow, self.ncol).f(), x).map_err(|e| e.to_string())? };
        let fit = Fit {
            x: &x,
            y,
            params,
            mask: mask.map(|(m, _)| m),
            rows: mask.map(|(_, rows)| rows),
            n_fit: mask.map_or(nrow, |(_, rows)| rows.len()),
        };
        let (center, scale, mask_safe) = fit.center_scale()?;
        if !mask_safe {
            return Ok(None);
        }
        fit.run(center, scale).map(Some)
    }

    fn fit_inner(&self, options: JsValue) -> Result<JsValue, String> {
        let options: Options = serde_wasm_bindgen::from_value(options).map_err(|e| e.to_string())?;
        if !options.alpha.is_finite() || options.alpha <= 0.0 || options.alpha > 1.0 {
            return Err("alpha must be in (0, 1]; alpha=0 (ridge) is not implemented.".into());
        }
        if !options.eps.is_finite() || options.eps <= 0.0 || options.max_iter < 1 || options.max_iter > 100000 {
            return Err("eps must be finite and positive; max_iter must be between 1 and 100000.".into());
        }
        let nlambda = options.lambdas.as_ref().map_or(options.nlambda, Vec::len);
        if nlambda == 0 || nlambda > 200 || (options.lambdas.is_none() && nlambda < 2) {
            return Err("Use 2–200 automatic lambdas, or 1–200 supplied lambdas.".into());
        }
        if let Some(lambdas) = &options.lambdas
            && (lambdas.iter().any(|l| !l.is_finite() || *l < 0.0) || lambdas.windows(2).any(|w| w[1] > w[0]))
        {
            return Err("Supplied lambdas must be finite, nonnegative, and nonincreasing.".into());
        }
        let lambda_min = options.lambda_min.unwrap_or(if self.nrow > self.ncol { 0.001 } else { 0.05 });
        if !lambda_min.is_finite() || lambda_min <= 0.0 || lambda_min > 1.0 {
            return Err("lambda_min must be in (0, 1].".into());
        }
        let pf = options.penalty_factor.unwrap_or_else(|| vec![1.0; self.ncol]);
        if pf.len() != self.ncol || pf.iter().any(|m| !m.is_finite() || *m < 0.0) {
            return Err("Penalty factors must be finite, nonnegative, and have one entry per feature.".into());
        }
        if self.x.iter().any(|v| !v.is_finite()) {
            return Err("Every selected feature must be completely loaded with finite numeric values.".into());
        }
        for rows in [&options.train_rows, &options.test_rows].into_iter().flatten() {
            let mut seen = vec![false; self.nrow];
            if rows.is_empty() || rows.iter().any(|&i| i >= self.nrow || std::mem::replace(&mut seen[i], true)) {
                return Err("Fold rows must be nonempty, distinct, and inside the dataset.".into());
            }
        }
        let family = Family::parse(&options.family).map_err(str::to_owned)?;
        let train = options.train_rows.as_deref();
        let (ybar, null_deviance) = family.null_model(&self.y, train).map_err(str::to_owned)?;
        let params = Params {
            family,
            alpha: options.alpha,
            lambdas: options.lambdas.as_deref(),
            nlambda,
            eps: options.eps,
            lambda_min: lambda_min.max(1e-6),
            lambda_log_scale: true,
            penalty_factor: &pf,
            dfmax: options.dfmax.unwrap_or(self.ncol + 1),
            max_iter: options.max_iter,
            stop_on_saturation: true,
            r_compatible: options.r_compatible,
            ybar,
            null_deviance,
            on_lambda: None,
        };
        let on_lambda = |_path: &Path, index| {
            progress(index + 1, nlambda);
            Ok(false)
        };
        let params = Params { on_lambda: Some(&on_lambda), ..params };
        // Do not build a native thread pool: Rayon's global fallback executes
        // these iterators serially inside the browser's dedicated Worker.
        let path = match train {
            None => self.run(&self.x, &self.y, self.nrow, &params, None)?,
            Some(rows) => {
                // a fold is a 0/1 weight on every row, as in the Python
                // binding, so X is never copied; y is 0 outside the fold
                let mut mask = vec![0.0; self.nrow];
                rows.iter().for_each(|&i| mask[i] = 1.0);
                let y: Vec<f64> = self.y.iter().zip(&mask).map(|(&y, &m)| if m != 0.0 { y } else { 0.0 }).collect();
                match self.run(&self.x, &y, self.nrow, &params, Some((&mask, rows)))? {
                    Some(path) => Some(path),
                    // a left-out row the mask can't sweep safely: gather
                    None => {
                        let x: Vec<f64> =
                            (0..self.ncol).flat_map(|j| rows.iter().map(move |&i| self.x[j * self.nrow + i])).collect();
                        let y: Vec<f64> = rows.iter().map(|&i| self.y[i]).collect();
                        self.run(&x, &y, rows.len(), &params, None)?
                    }
                }
            }
        }
        .expect("an unmasked fit always runs");
        if path.lambda.iter().any(|l| !l.is_finite()) {
            return Err("The automatic lambda grid is not finite; increase alpha or supply a finite grid.".into());
        }
        let mut indices = Vec::new();
        let mut coefficients = Vec::new();
        let mut standardized_coefficients = Vec::new();
        let mut intercept = Vec::new();
        // The core leaves -1 sentinels after saturation or dfmax, just as the
        // Python adapter expects; they are not fitted null models.
        let fitted: Vec<usize> = path.iters.iter().enumerate().filter_map(|(k, &n)| (n >= 0).then_some(k)).collect();
        for &k in &fitted {
            let cols: Vec<usize> =
                path.coef_rows[k].as_deref().unwrap_or_default().iter().map(|&j| path.col_map[j]).collect();
            let values = path.coef_vals[k].as_deref().unwrap_or_default();
            let beta: Vec<f64> = cols.iter().zip(values).map(|(&j, &v)| v / path.scale[j]).collect();
            let correction: Vec<f64> = cols.iter().zip(&beta).map(|(&j, &v)| path.center[j] * v).collect();
            intercept.push(path.intercept[k] - rounded_sum(&correction).map_err(str::to_owned)?);
            indices.push(cols);
            coefficients.push(beta);
            standardized_coefficients.push(values.to_vec());
        }
        let warnings = path
            .events
            .iter()
            .map(|e| match e {
                Event::PrefitSaturated => "The unpenalized columns saturated the model before the path.".into(),
                Event::Dfmax(i) => format!("Feature limit reached at lambda {}; the path stopped early.", i + 1),
                Event::Saturated(i) => format!("Model saturated at lambda {}; the path stopped early.", i + 1),
                Event::MaxIter(i) => {
                    format!("Convergence limit reached at lambda {}; consider a larger max_iter.", i + 1)
                }
            })
            .collect();
        let holdout = options.test_rows.as_ref().map(|test| {
            (0..fitted.len())
                .map(|f| {
                    let loss: f64 = test
                        .iter()
                        .map(|&i| {
                            let eta = intercept[f]
                                + indices[f]
                                    .iter()
                                    .zip(&coefficients[f])
                                    .map(|(&j, &b)| self.x[j * self.nrow + i] * b)
                                    .sum::<f64>();
                            match family {
                                // log(1 + exp(eta)) - y eta, computed stably
                                Family::Binomial => {
                                    2.0 * (eta.max(0.0) + libm::log1p(libm::exp(-eta.abs())) - self.y[i] * eta)
                                }
                                Family::Gaussian => (self.y[i] - eta).powi(2),
                            }
                        })
                        .sum();
                    loss / test.len() as f64
                })
                .collect()
        });
        let result = ModelPath {
            family: options.family,
            lambdas: fitted.iter().map(|&k| path.lambda[k]).collect(),
            intercept,
            indices,
            coefficients,
            standardized_coefficients,
            center: path.center,
            scale: path.scale,
            iterations: fitted.iter().map(|&k| path.iters[k]).collect(),
            n_rejected: fitted.iter().map(|&k| path.n_reject[k]).collect(),
            deviance: fitted
                .iter()
                .map(|&k| (if family == Family::Binomial { 2.0 } else { 1.0 }) * path.deviance[k])
                .collect(),
            holdout,
            warnings,
        };
        serde_wasm_bindgen::to_value(&result).map_err(|e| e.to_string())
    }
}
