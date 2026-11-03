use super::solver::*;
use numpy::{
    Element, IntoPyArray, PyArrayDescrMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArray,
    PyUntypedArrayMethods, ToPyArray,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::sync::Mutex;

// The Python interface

fn to_dict<'py>(py: Python<'py>, path: Path) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let opt_list = |rows: Vec<Option<Vec<usize>>>| -> PyResult<Bound<'py, PyList>> {
        let l = PyList::empty(py);
        for r in rows {
            match r {
                Some(v) => l.append(v.into_iter().map(|x| x as i64).collect::<Vec<_>>().into_pyarray(py))?,
                None => l.append(py.None())?,
            }
        }
        Ok(l)
    };
    d.set_item("coef_rows", opt_list(path.coef_rows)?)?;
    let vals = PyList::empty(py);
    for v in path.coef_vals {
        match v {
            Some(v) => vals.append(v.into_pyarray(py))?,
            None => vals.append(py.None())?,
        }
    }
    d.set_item("coef_vals", vals)?;
    d.set_item("center", path.center.into_pyarray(py))?;
    d.set_item("scale", path.scale.into_pyarray(py))?;
    d.set_item("col_map", path.col_map.into_iter().map(|x| x as i64).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("lambda_", path.lambda.into_pyarray(py))?;
    d.set_item("intercept", path.intercept.into_pyarray(py))?;
    d.set_item("iters", path.iters.into_pyarray(py))?;
    d.set_item("deviance", path.deviance.into_pyarray(py))?;
    d.set_item("n_reject", path.n_reject.into_pyarray(py))?;
    let events = PyList::empty(py);
    for e in path.events {
        let (kind, il): (&str, i64) = match e {
            Event::PrefitSaturated => ("prefit_saturated", -1),
            Event::Dfmax(il) => ("dfmax", il as i64),
            Event::Saturated(il) => ("saturated", il as i64),
            Event::MaxIter(il) => ("max_iter", il as i64),
        };
        events.append((kind, il))?;
    }
    d.set_item("events", events)?;
    Ok(d)
}

/// The rows being fit: all of X's, or row_indices', gathered or as a 0/1
/// mask over every row (see MASK_SHARE).
#[derive(Clone, Copy)]
enum Selection<'a> {
    All,
    Gather(&'a [usize]),
    Mask { idx: &'a [usize], mask: &'a [f64] },
}

impl<'a> Selection<'a> {
    fn masked_rows(&self) -> Option<&'a [usize]> {
        match *self {
            Selection::Mask { idx, .. } => Some(idx),
            _ => None,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Selection::All => "all",
            Selection::Gather(_) => "gather",
            Selection::Mask { .. } => "mask",
        }
    }
}

fn run_typed<T: Elem + Element>(
    py: Python<'_>,
    x: &Bound<'_, PyUntypedArray>,
    y_all: &[f64],
    rows: Selection<'_>,
    params: &Params,
    n_threads: usize,
    boundscheck: bool,
) -> PyResult<(Path, &'static str)> {
    let x: PyReadonlyArray2<T> = x.extract()?;
    let view = x.as_array();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let fortran = view.column(0).to_slice().is_some();
    // y as the rows being fit, so kernels index it like the residuals; with a
    // mask, every row of y, 0 where the mask is (not y * 0: y need only be
    // finite in the rows being fit)
    let (y_gathered, y_masked): (Vec<f64>, Vec<f64>) = match rows {
        Selection::All => (vec![], vec![]),
        Selection::Gather(idx) => (idx.iter().map(|&i| y_all[i]).collect(), vec![]),
        Selection::Mask { idx, mask } => (
            idx.iter().map(|&i| y_all[i]).collect(),
            y_all.iter().zip(mask).map(|(&y, &m)| if m != 0.0 { y } else { 0.0 }).collect(),
        ),
    };

    let result = py.detach(|| {
        pool.install(|| {
            macro_rules! go {
                ($design:expr, $y:expr, $mask:expr, $n_fit:expr, $rows:expr) => {{
                    let d = $design;
                    let fit = Fit { x: &d, y: $y, params, mask: $mask, rows: $rows.masked_rows(), n_fit: $n_fit };
                    let (center, scale, mask_safe) = fit.center_scale().map_err(Fail)?;
                    if !mask_safe {
                        // a row left out holds something the mask can't sweep
                        // safely: gather the rows instead
                        return Err(Fallback);
                    }
                    fit.run(center, scale).map(|path| (path, $rows.name())).map_err(Fail)
                }};
            }
            macro_rules! by_layout {
                ($checked:literal, $rows:expr) => {
                    match ($rows, fortran) {
                        (Selection::All, true) => {
                            go!(ContigX::<T, $checked> { x: view }, y_all, None, view.nrows(), $rows)
                        }
                        (Selection::All, false) => {
                            go!(StridedX::<T, $checked> { x: view }, y_all, None, view.nrows(), $rows)
                        }
                        (Selection::Mask { idx, mask }, true) => {
                            go!(ContigX::<T, $checked> { x: view }, &y_masked, Some(mask), idx.len(), $rows)
                        }
                        (Selection::Mask { idx, mask }, false) => {
                            go!(StridedX::<T, $checked> { x: view }, &y_masked, Some(mask), idx.len(), $rows)
                        }
                        (Selection::Gather(idx), true) => {
                            go!(
                                GatherX { x: ContigX::<T, $checked> { x: view }, idx },
                                &y_gathered,
                                None,
                                idx.len(),
                                $rows
                            )
                        }
                        (Selection::Gather(idx), false) => {
                            go!(
                                GatherX { x: StridedX::<T, $checked> { x: view }, idx },
                                &y_gathered,
                                None,
                                idx.len(),
                                $rows
                            )
                        }
                    }
                };
            }
            let attempt = |rows: Selection<'_>| -> Result<(Path, &'static str), Stop> {
                if boundscheck { by_layout!(true, rows) } else { by_layout!(false, rows) }
            };
            match attempt(rows) {
                Err(Fallback) => match rows {
                    Selection::Mask { idx, .. } => attempt(Selection::Gather(idx)),
                    _ => unreachable!("only a mask leaves rows out"),
                },
                other => other,
            }
        })
    });
    result.map_err(|e| match e {
        Fail(msg) => PyValueError::new_err(msg),
        Fallback => unreachable!("a gather never falls back"),
    })
}

/// Why a fit stopped short: an error, or a mask that must gather instead.
enum Stop {
    Fail(String),
    Fallback,
}
use Stop::{Fail, Fallback};

// Read guards retain mapped buffer owners across GIL release and callbacks.
// The descriptor carries arrays, never unchecked addresses supplied by Python.
enum ColumnOwner<'py> {
    F64(PyReadonlyArray1<'py, f64>),
    F32(PyReadonlyArray1<'py, f32>),
    I8(PyReadonlyArray1<'py, i8>),
    U8(PyReadonlyArray1<'py, u8>),
    I16(PyReadonlyArray1<'py, i16>),
    U64(PyReadonlyArray1<'py, u64>),
    I64(PyReadonlyArray1<'py, i64>),
    U32(PyReadonlyArray1<'py, u32>),
    I32(PyReadonlyArray1<'py, i32>),
    U16(PyReadonlyArray1<'py, u16>),
    Bits(PyReadonlyArray1<'py, u8>, usize),
}

type ChunkDescriptor<'py> = (Bound<'py, PyUntypedArray>, Option<usize>, usize);
type MappedDescriptor<'py> = (usize, Vec<Vec<ChunkDescriptor<'py>>>);
type ColumnChunks<'py> = Vec<(ColumnOwner<'py>, usize)>;

impl<'py> ColumnOwner<'py> {
    fn extract(array: &Bound<'py, PyUntypedArray>, offset: Option<usize>) -> PyResult<Self> {
        if !array.getattr("flags")?.getattr("aligned")?.extract::<bool>()?
            || array.getattr("flags")?.getattr("writeable")?.extract::<bool>()?
        {
            return Err(PyValueError::new_err("Mapped columns must be read-only and aligned."));
        }
        if let Some(offset) = offset {
            return Ok(Self::Bits(array.extract()?, offset));
        }
        let dtype = array.dtype();
        if dtype.is_equiv_to(&numpy::dtype::<f64>(array.py())) {
            return Ok(Self::F64(array.extract()?));
        }
        if dtype.is_equiv_to(&numpy::dtype::<f32>(array.py())) {
            return Ok(Self::F32(array.extract()?));
        }
        if dtype.is_equiv_to(&numpy::dtype::<i8>(array.py())) {
            return Ok(Self::I8(array.extract()?));
        }
        if dtype.is_equiv_to(&numpy::dtype::<u8>(array.py())) {
            return Ok(Self::U8(array.extract()?));
        }
        if dtype.is_equiv_to(&numpy::dtype::<i16>(array.py())) {
            return Ok(Self::I16(array.extract()?));
        }
        if dtype.is_equiv_to(&numpy::dtype::<u16>(array.py())) {
            return Ok(Self::U16(array.extract()?));
        }
        if dtype.is_equiv_to(&numpy::dtype::<i32>(array.py())) {
            return Ok(Self::I32(array.extract()?));
        }
        if dtype.is_equiv_to(&numpy::dtype::<u32>(array.py())) {
            return Ok(Self::U32(array.extract()?));
        }
        if dtype.is_equiv_to(&numpy::dtype::<i64>(array.py())) {
            return Ok(Self::I64(array.extract()?));
        }
        if dtype.is_equiv_to(&numpy::dtype::<u64>(array.py())) {
            return Ok(Self::U64(array.extract()?));
        }
        Err(PyValueError::new_err("Unsupported mapped column dtype."))
    }

    fn column(&self, nrow: usize) -> PyResult<MappedCol<'_>> {
        let col = match self {
            Self::F64(a) => MappedCol::F64(a.as_slice()?),
            Self::F32(a) => MappedCol::F32(a.as_slice()?),
            Self::I8(a) => MappedCol::I8(a.as_slice()?),
            Self::U8(a) => MappedCol::U8(a.as_slice()?),
            Self::I16(a) => MappedCol::I16(a.as_slice()?),
            Self::U64(a) => MappedCol::U64(a.as_slice()?),
            Self::I64(a) => MappedCol::I64(a.as_slice()?),
            Self::U32(a) => MappedCol::U32(a.as_slice()?),
            Self::I32(a) => MappedCol::I32(a.as_slice()?),
            Self::U16(a) => MappedCol::U16(a.as_slice()?),
            Self::Bits(a, offset) => {
                let bytes = a.as_slice()?;
                let end = offset.checked_add(nrow).ok_or_else(|| PyValueError::new_err("Bitmap offset overflows."))?;
                if end.div_ceil(8) > bytes.len() {
                    return Err(PyValueError::new_err("Bitmap is too short for its offset and row count."));
                }
                MappedCol::Bits { bytes, offset: *offset, len: nrow }
            }
        };
        if col.n() != nrow {
            return Err(PyValueError::new_err("Mapped columns must have one value per row."));
        }
        Ok(col)
    }
}

#[allow(clippy::too_many_arguments)]
fn run_mapped(
    py: Python<'_>,
    owners: &[ColumnChunks<'_>],
    nrow: usize,
    y_all: &[f64],
    rows: Selection<'_>,
    params: &Params,
    n_threads: usize,
) -> PyResult<(Path, &'static str)> {
    let columns = owners
        .iter()
        .enumerate()
        .map(|(j, chunks)| {
            let chunks = chunks
                .iter()
                .enumerate()
                .map(|(k, (owner, len))| {
                    owner.column(*len).map_err(|e| PyValueError::new_err(format!("Column {j}, chunk {k}: {e}")))
                })
                .collect::<PyResult<Vec<_>>>()?;
            ChunkedColumn::new(chunks, nrow).map_err(|e| PyValueError::new_err(format!("Column {j}: {e}")))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    py.detach(|| {
        pool.install(|| {
            let attempt = |rows: Selection<'_>| -> Result<(Path, &'static str), Stop> {
                let x = MappedX { columns: columns.clone() };
                let y_selected: Vec<f64>;
                macro_rules! go {
                    ($design:expr, $y:expr, $mask:expr, $n:expr) => {{
                        let d = $design;
                        let fit = Fit { x: &d, y: $y, params, mask: $mask, rows: rows.masked_rows(), n_fit: $n };
                        let (center, scale, safe) = fit.center_scale().map_err(Fail)?;
                        if !safe {
                            return Err(Fallback);
                        }
                        fit.run(center, scale).map(|p| (p, rows.name())).map_err(Fail)
                    }};
                }
                match rows {
                    Selection::All => go!(x, y_all, None, nrow),
                    Selection::Gather(idx) => {
                        y_selected = idx.iter().map(|&i| y_all[i]).collect();
                        go!(GatherX { x, idx }, &y_selected, None, idx.len())
                    }
                    Selection::Mask { idx, mask } => {
                        y_selected = y_all.iter().zip(mask).map(|(&y, &m)| if m != 0.0 { y } else { 0.0 }).collect();
                        go!(x, &y_selected, Some(mask), idx.len())
                    }
                }
            };
            match attempt(rows) {
                Err(Fallback) => match rows {
                    Selection::Mask { idx, .. } => attempt(Selection::Gather(idx)),
                    _ => unreachable!("only masks fall back"),
                },
                result => result,
            }
        })
    })
    .map_err(|e| match e {
        Fail(msg) => PyValueError::new_err(msg),
        Fallback => unreachable!("gather cannot fall back"),
    })
}

/// Fit a binomial path; arguments and result as the numba backend's
/// fit_path. Null-model statistics use the same portable math as the fit.
#[pyfunction]
#[pyo3(signature = (x, y, row_indices, alpha, lambdas, nlambda, eps, lambda_min, lambda_log_scale, penalty_factor, dfmax, max_iter, stop_on_saturation, r_compatible, n_threads, boundscheck, row_path = "auto", on_lambda = None, family = "binomial"))]
#[allow(clippy::too_many_arguments)]
fn fit_path<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    y: PyReadonlyArray1<'py, f64>,
    row_indices: Option<PyReadonlyArray1<'py, i64>>,
    alpha: f64,
    lambdas: Option<PyReadonlyArray1<'py, f64>>,
    nlambda: usize,
    eps: f64,
    lambda_min: f64,
    lambda_log_scale: bool,
    penalty_factor: PyReadonlyArray1<'py, f64>,
    dfmax: usize,
    max_iter: i64,
    stop_on_saturation: bool,
    r_compatible: bool,
    n_threads: usize,
    boundscheck: bool,
    row_path: &str,
    on_lambda: Option<Bound<'py, PyAny>>,
    family: &str,
) -> PyResult<Bound<'py, PyDict>> {
    if !["auto", "gather", "mask"].contains(&row_path) {
        return Err(PyValueError::new_err("row_path must be 'auto', 'gather' or 'mask'."));
    }
    // biglasso() checks all of this; checked again here so a direct call
    // gets an error rather than a panic
    let ndarray = x.cast::<PyUntypedArray>().ok();
    let (shape, owners) = if let Some(a) = ndarray {
        (a.shape().to_vec(), vec![])
    } else {
        let (nrow, columns): MappedDescriptor<'py> = x.extract()?;
        let shape = vec![nrow, columns.len()];
        let owners = columns
            .iter()
            .enumerate()
            .map(|(j, chunks)| {
                chunks
                    .iter()
                    .enumerate()
                    .map(|(k, (a, bit, len))| {
                        ColumnOwner::extract(a, *bit)
                            .map(|owner| (owner, *len))
                            .map_err(|e| PyValueError::new_err(format!("Column {j}, chunk {k}: {e}")))
                    })
                    .collect::<PyResult<Vec<_>>>()
            })
            .collect::<PyResult<Vec<_>>>()?;
        (shape, owners)
    };
    for (name, a) in [("y", y.as_any()), ("penalty factors", penalty_factor.as_any())]
        .into_iter()
        .chain(ndarray.map(|a| ("X", a.as_any())))
        .chain(row_indices.as_ref().map(|a| ("row_indices", a.as_any())))
        .chain(lambdas.as_ref().map(|a| ("lambdas", a.as_any())))
    {
        // numpy can build arrays whose elements aren't aligned for their type
        // (offset buffers), and the views below read them through typed
        // pointers, which would be undefined behavior
        if !a.getattr("flags")?.getattr("aligned")?.extract::<bool>()? {
            return Err(PyValueError::new_err(format!("{name} must be aligned to its dtype.")));
        }
    }
    if shape.len() != 2 || shape[0] == 0 || shape[1] == 0 {
        return Err(PyValueError::new_err("X must be a non-empty 2-D array."));
    }
    let y_all = y.as_slice()?;
    if y_all.len() != shape[0] {
        return Err(PyValueError::new_err("y must have one entry per row of X."));
    }
    if penalty_factor.len() != shape[1] {
        return Err(PyValueError::new_err("penalty factors must have one entry per column of X."));
    }
    let idx: Option<Vec<usize>> = match &row_indices {
        Some(ri) => {
            let ri = ri.as_slice()?;
            let nrow = shape[0];
            if ri.iter().any(|&i| i < 0 || i as usize >= nrow) {
                return Err(PyValueError::new_err("row_indices has indices outside valid range."));
            }
            Some(ri.iter().map(|&i| i as usize).collect())
        }
        None => None,
    };
    let family = Family::parse(family).map_err(PyValueError::new_err)?;
    let (ybar, null_deviance) = family.null_model(y_all, idx.as_deref()).map_err(PyValueError::new_err)?;
    let nrow_x = shape[0];
    // row_indices with no repeats, naming at least MASK_SHARE of X's rows, are
    // fit through a mask (row_path forces either path, for tests)
    let mask: Option<Vec<f64>> = idx.as_ref().and_then(|idx| {
        let wanted = match row_path {
            "auto" => idx.len() as f64 >= MASK_SHARE * nrow_x as f64,
            "mask" => true,
            _ => false,
        };
        if !wanted {
            // a few rows of a tall X: no row-length buffer
            return None;
        }
        let mut m = vec![0.0; nrow_x];
        for &i in idx {
            if m[i] != 0.0 {
                return None;
            }
            m[i] = 1.0;
        }
        Some(m)
    });
    if row_path == "mask" && idx.is_some() && mask.is_none() {
        return Err(PyValueError::new_err("row_path='mask' needs row_indices without repeats."));
    }
    let lambdas_vec = lambdas.as_ref().map(|l| l.as_slice().map(|s| s.to_vec())).transpose()?;
    // on_lambda(il, lambdas, cols, vals, intercept, center, scale), as the
    // numba backend calls it; its exception is kept, and raised once the fit
    // has stopped
    let on_lambda = on_lambda.map(Bound::unbind);
    let raised: Mutex<Option<PyErr>> = Mutex::new(None);
    let call = |path: &Path, il: usize| -> Result<bool, String> {
        let on_lambda = on_lambda.as_ref().expect("only set with a callable");
        Python::attach(|py| {
            let rows = path.coef_rows[il].as_deref().unwrap_or_default();
            let cols: Vec<usize> = rows.iter().map(|&j| path.col_map[j]).collect();
            let args = (
                il,
                path.lambda.to_pyarray(py),
                cols.iter().map(|&c| c as i64).collect::<Vec<_>>().into_pyarray(py),
                path.coef_vals[il].as_deref().unwrap_or_default().to_pyarray(py),
                path.intercept[il],
                cols.iter().map(|&c| path.center[c]).collect::<Vec<_>>().into_pyarray(py),
                cols.iter().map(|&c| path.scale[c]).collect::<Vec<_>>().into_pyarray(py),
            );
            on_lambda.call1(py, args)?.extract::<bool>(py)
        })
        .map_err(|e| {
            *raised.lock().unwrap() = Some(e);
            "on_lambda raised".into()
        })
    };
    let params = Params {
        family,
        alpha,
        lambdas: lambdas_vec.as_deref(),
        nlambda,
        eps,
        lambda_min,
        lambda_log_scale,
        penalty_factor: penalty_factor.as_slice()?,
        dfmax,
        max_iter,
        stop_on_saturation,
        r_compatible,
        ybar,
        null_deviance,
        on_lambda: on_lambda.as_ref().map(|_| &call as &OnLambda),
    };
    let rows = match (&idx, &mask) {
        (Some(idx), Some(mask)) => Selection::Mask { idx, mask },
        (Some(idx), None) => Selection::Gather(idx),
        (None, _) => Selection::All,
    };
    let finish = |result: PyResult<(Path, &'static str)>| {
        if let Some(e) = raised.lock().unwrap().take() {
            return Err(e);
        }
        let (path, ran) = result?;
        let d = to_dict(py, path)?;
        d.set_item("row_path", ran)?;
        Ok(d)
    };
    let Some(x) = ndarray else {
        return finish(run_mapped(py, &owners, shape[0], y_all, rows, &params, n_threads));
    };
    let dtype = x.dtype();
    macro_rules! dispatch {
        ($($t:ty),*) => {
            $(if dtype.is_equiv_to(&numpy::dtype::<$t>(py)) {
                let result = run_typed::<$t>(py, x, y_all, rows, &params, n_threads, boundscheck);
                return finish(result);
            })*
        };
    }
    dispatch!(f64, f32, i8, i16, i32, i64, u8, u16, u32, u64, bool);
    Err(PyValueError::new_err("X must have a native real numeric dtype (float32/64 or integer)."))
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit_path, m)?)?;
    Ok(())
}
