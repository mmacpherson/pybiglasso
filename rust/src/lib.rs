//! Rust backend for pybiglasso's binomial solver: the path loop of
//! `fit_path` and every kernel it calls, in one call from Python.
//!
//! It follows the numba backend step for step, so that it matches R biglasso
//! the same way: same coordinate order, same screening, same stopping rules.
//! Two things are under our control here and are part of the contract:
//!
//! - Summation order. Every column reduction runs in a fixed order that does
//!   not depend on the thread count: sequential where the numba backend is
//!   strict, and LANES independent accumulators, combined in a fixed tree,
//!   where it uses fastmath. Threads split work across columns, and a tall
//!   column's rows into CHUNK-row chunks, whose sums are added in order, so
//!   the chunks and their order are the same at every thread count.
//! - X is read in place, a column at a time, through whatever strides numpy
//!   gives; it is never copied or converted.

#![deny(unsafe_code)]
// index loops, as in the numba backend, so the two read side by side
#![allow(clippy::needless_range_loop)]

use ndarray::{ArrayView1, ArrayView2};
// Differs from R biglasso: R uses the platform math library. Portable
// exp/log/log1p keep the Rust path's bits independent of that library;
// agreement with R remains to floating-point precision.
use libm::{exp, log, log1p};
use numpy::{
    Element, IntoPyArray, PyArrayDescrMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArray,
    PyUntypedArrayMethods, ToPyArray,
};
use pulp::{Simd, WithSimd};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;
use std::sync::Mutex;

const LANES: usize = 8;
const SCALE_MIN: f64 = 1e-6;
const ETA_MAX: f64 = 10.0;
const W_MIN: f64 = 0.0001;

// Element types and column access

trait Elem: Element + Copy + Send + Sync {
    fn f(self) -> f64;
}
macro_rules! elem {
    ($($t:ty),*) => { $(impl Elem for $t { #[inline(always)] fn f(self) -> f64 { self as f64 } })* };
}
elem!(f64, f32, i8, i16, i32, i64, u8, u16, u32, u64);
impl Elem for bool {
    #[inline(always)]
    fn f(self) -> f64 {
        if self { 1.0 } else { 0.0 }
    }
}

/// One column of X, as the rows being fit (all of them, or row_indices).
///
/// The two reads of X itself, `Contig::at` and `Strided::at`, check their
/// bounds unless CHECKED is false, the crate's only unsafe code: checking
/// every gathered or strided row costs 11-15% of a fit. Every caller indexes
/// below n(), and Gather's row_indices are range-checked against X's rows
/// before the fit, so the check guards against a bug here, not bad input; it
/// is on by default (the boundscheck setting), and changes no result.
trait Col: Sync {
    fn n(&self) -> usize;
    /// Row i of the fit; i < n().
    fn at(&self, i: usize) -> f64;
    /// Rows i0 .. i0 + LANES; i0 + LANES <= n().
    #[inline(always)]
    fn block(&self, i0: usize) -> [f64; LANES] {
        std::array::from_fn(|k| self.at(i0 + k))
    }
}

/// A column contiguous in memory (X in Fortran order).
struct Contig<'a, T, const CHECKED: bool>(&'a [T]);
impl<T: Elem, const CHECKED: bool> Col for Contig<'_, T, CHECKED> {
    #[inline(always)]
    fn n(&self) -> usize {
        self.0.len()
    }
    #[inline(always)]
    #[allow(unsafe_code)]
    fn at(&self, i: usize) -> f64 {
        if CHECKED {
            return self.0[i].f();
        }
        debug_assert!(i < self.0.len());
        // SAFETY: i < n(); see Col
        unsafe { self.0.get_unchecked(i) }.f()
    }
    #[inline(always)]
    fn block(&self, i0: usize) -> [f64; LANES] {
        let rows: &[T; LANES] = self.0[i0..i0 + LANES].try_into().expect("LANES rows");
        rows.map(Elem::f)
    }
}

/// A column with an arbitrary element stride (X in C order, or a view).
struct Strided<'a, T, const CHECKED: bool>(ArrayView1<'a, T>);
impl<T: Elem, const CHECKED: bool> Col for Strided<'_, T, CHECKED> {
    #[inline(always)]
    fn n(&self) -> usize {
        self.0.len()
    }
    #[inline(always)]
    #[allow(unsafe_code)]
    fn at(&self, i: usize) -> f64 {
        if CHECKED {
            return self.0[i].f();
        }
        debug_assert!(i < self.0.len());
        // SAFETY: i < n(); see Col
        unsafe { self.0.uget(i) }.f()
    }
}

/// The rows of a column picked out by row_indices, in their given order.
struct Gather<'a, C> {
    col: C,
    idx: &'a [usize],
}
impl<C: Col> Col for Gather<'_, C> {
    #[inline(always)]
    fn n(&self) -> usize {
        self.idx.len()
    }
    #[inline(always)]
    fn at(&self, i: usize) -> f64 {
        self.col.at(self.idx[i])
    }
    #[inline(always)]
    fn block(&self, i0: usize) -> [f64; LANES] {
        let rows: &[usize; LANES] = self.idx[i0..i0 + LANES].try_into().expect("LANES rows");
        rows.map(|i| self.col.at(i))
    }
}

/// Rows i0 .. i0 + n of a column.
struct Span<'a, C> {
    col: &'a C,
    i0: usize,
    n: usize,
}
impl<C: Col> Col for Span<'_, C> {
    #[inline(always)]
    fn n(&self) -> usize {
        self.n
    }
    #[inline(always)]
    fn at(&self, i: usize) -> f64 {
        self.col.at(self.i0 + i)
    }
    #[inline(always)]
    fn block(&self, i0: usize) -> [f64; LANES] {
        self.col.block(self.i0 + i0)
    }
}

trait Design: Sync {
    type C<'b>: Col
    where
        Self: 'b;
    fn col(&self, j: usize) -> Self::C<'_>;
    fn ncol(&self) -> usize;
    fn nrow(&self) -> usize;
}

struct ContigX<'a, T, const CHECKED: bool> {
    x: ArrayView2<'a, T>,
}
impl<T: Elem, const CHECKED: bool> Design for ContigX<'_, T, CHECKED> {
    type C<'b>
        = Contig<'b, T, CHECKED>
    where
        Self: 'b;
    fn col(&self, j: usize) -> Contig<'_, T, CHECKED> {
        Contig(self.x.column(j).to_slice().expect("Fortran-contiguous column"))
    }
    fn ncol(&self) -> usize {
        self.x.ncols()
    }
    fn nrow(&self) -> usize {
        self.x.nrows()
    }
}

struct StridedX<'a, T, const CHECKED: bool> {
    x: ArrayView2<'a, T>,
}
impl<T: Elem, const CHECKED: bool> Design for StridedX<'_, T, CHECKED> {
    type C<'b>
        = Strided<'b, T, CHECKED>
    where
        Self: 'b;
    fn col(&self, j: usize) -> Strided<'_, T, CHECKED> {
        Strided(self.x.column(j))
    }
    fn ncol(&self) -> usize {
        self.x.ncols()
    }
    fn nrow(&self) -> usize {
        self.x.nrows()
    }
}

struct GatherX<'a, D> {
    x: D,
    idx: &'a [usize],
}
impl<D: Design> Design for GatherX<'_, D> {
    type C<'b>
        = Gather<'b, D::C<'b>>
    where
        Self: 'b;
    fn col(&self, j: usize) -> Gather<'_, D::C<'_>> {
        Gather { col: self.x.col(j), idx: self.idx }
    }
    fn ncol(&self) -> usize {
        self.x.ncol()
    }
    fn nrow(&self) -> usize {
        self.idx.len()
    }
}

// Reductions

#[inline(always)]
fn combine(acc: &[f64; LANES]) -> f64 {
    ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]))
}

/// Sequential sum: the order the numba backend's seqsum fixes.
fn seqsum(a: &[f64]) -> f64 {
    let mut out = 0.0;
    for &v in a {
        out += v;
    }
    out
}

/// Null-model statistics in the selected row order, without NumPy's
/// platform-dependent logarithms or reduction order in the stopping threshold.
/// The caller has range-checked rows; unused responses need not be valid.
fn null_model(y: &[f64], rows: Option<&[usize]>) -> Result<(f64, f64), &'static str> {
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

// The fastmath kernels below keep LANES accumulators, lane k summing rows
// i with i % LANES == k, combined in a fixed tree. pulp picks the widest
// vectors the CPU has at run time (AVX-512, AVX2, NEON or scalar) and a block
// of LANES rows spans LANES / width registers; the sums are the same at every
// width, and with no fused multiply-add, the same on every machine (under
// the default floating-point environment: round to nearest, subnormals kept).
// The tests check this against plain scalar sums, at every width.

/// The CPU's widest vector instructions, detected once.
static ARCH: std::sync::LazyLock<pulp::Arch> = std::sync::LazyLock::new(pulp::Arch::new);

/// A block of LANES rows, aligned so it reads as whole vectors at any width.
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(64))]
struct Block([f64; LANES]);

/// A vector of rows that starts on a cache line: a register's worth of rows
/// read from it never straddles two lines, which with AVX-512 can double a
/// sweep's time.
struct Rows {
    blocks: Vec<Block>,
    len: usize,
}
impl Rows {
    fn new(len: usize, value: f64) -> Self {
        Rows { blocks: vec![Block([value; LANES]); len.div_ceil(LANES)], len }
    }
}
impl std::ops::Deref for Rows {
    type Target = [f64];
    fn deref(&self) -> &[f64] {
        &bytemuck::cast_slice::<_, f64>(&self.blocks)[..self.len]
    }
}
impl std::ops::DerefMut for Rows {
    fn deref_mut(&mut self) -> &mut [f64] {
        &mut bytemuck::cast_slice_mut::<_, f64>(&mut self.blocks)[..self.len]
    }
}

/// A register of rows, with the arithmetic operators: each is one IEEE
/// operation per lane, never fused, so it rounds as it would on one f64.
#[derive(Clone, Copy)]
struct V<S: Simd>(S, S::f64s);
macro_rules! v_op {
    ($($op:ident $f:ident $assign:ident $fa:ident $simd:ident),*) => {$(
        impl<S: Simd> std::ops::$op for V<S> {
            type Output = Self;
            #[inline(always)]
            fn $f(self, o: Self) -> Self {
                V(self.0, self.0.$simd(self.1, o.1))
            }
        }
        impl<S: Simd> std::ops::$assign for V<S> {
            #[inline(always)]
            fn $fa(&mut self, o: Self) {
                *self = std::ops::$op::$f(*self, o);
            }
        }
    )*};
}
v_op!(Add add AddAssign add_assign add_f64s, Sub sub SubAssign sub_assign sub_f64s, Mul mul MulAssign mul_assign mul_f64s);

/// One row (f64) or a register of rows (V): the kernels' arithmetic on a
/// row is written once, for both.
trait Num:
    Copy
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::AddAssign
    + std::ops::SubAssign
{
}
impl Num for f64 {}
impl<S: Simd> Num for V<S> {}

/// Registers per block of LANES rows.
#[inline(always)]
fn regs<S: Simd>() -> usize {
    LANES / S::F64_LANES
}

/// LANES accumulators, as the first `regs` registers, combined.
#[inline(always)]
fn fold<S: Simd>(acc: [V<S>; LANES]) -> f64 {
    let acc = acc.map(|v| v.1);
    let lanes: &[f64] = bytemuck::cast_slice(&acc[..regs::<S>()]);
    combine(lanes.try_into().expect("LANES lanes"))
}

/// Column c's rows i0 .. i0 + LANES.
#[inline(always)]
fn load<C: Col>(c: &C, i0: usize) -> Block {
    Block(c.block(i0))
}

/// A block's rows as registers.
#[inline(always)]
fn regs_of<S: Simd>(b: &Block) -> &[S::f64s] {
    bytemuck::cast_slice(&b.0)
}

/// sum(x * s), in LANES accumulators.
#[inline(always)]
fn dot_col<C: Col>(c: &C, s: &[f64]) -> f64 {
    struct Op<'a, C>(&'a C, &'a [f64]);
    impl<C: Col> WithSimd for Op<'_, C> {
        type Output = f64;
        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> f64 {
            let Op(c, s) = self;
            let v = |x| V(simd, x);
            let body = c.n() / LANES * LANES;
            let (sv, _) = S::as_simd_f64s(&s[..body]);
            let mut acc = [v(simd.splat_f64s(0.0)); LANES];
            for (b, s) in sv.chunks_exact(regs::<S>()).enumerate() {
                let x = load(c, b * LANES);
                let x = regs_of::<S>(&x);
                for m in 0..regs::<S>() {
                    acc[m] += v(x[m]) * v(s[m]);
                }
            }
            let mut tail = 0.0;
            for i in body..c.n() {
                tail += c.at(i) * s[i];
            }
            fold(acc) + tail
        }
    }
    ARCH.dispatch(Op(c, s))
}

/// sum(a * b) over two vectors, in LANES accumulators.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    dot_col(&Contig::<_, true>(a), b)
}

/// Stable score and curvature from centered values; strict order, as numba.
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

/// Standardized score x_j'(s) / n, with the centered fallback for columns far
/// from zero (see the numba backend's standardized_score).
#[inline(always)]
fn standardized_score<C: Col>(c: &C, s: &[f64], s_sum: f64, center: f64, scale: f64, n_fit: f64) -> f64 {
    let n = c.n();
    if center.abs() > 100.0 * scale {
        let mut score = 0.0;
        for i in 0..n {
            score += ((c.at(i) - center) / scale) * s[i];
        }
        return score / n_fit;
    }
    (dot_col(c, s) - center * s_sum) / (scale * n_fit)
}

// The coordinate sweep's pass over the rows. Each sums a column CHUNK rows
// at a time and then adds the chunk sums in order, so threads can share out
// a tall column's rows without changing any sum.

/// Rows per chunk: a multiple of LANES, and part of the results, since
/// changing it changes their last bits.
const CHUNK: usize = 2048;
/// Columns at least this tall share their chunks across threads; shorter
/// ones take longer to hand out than to sum.
const PAR_ROWS: usize = 16_384;
/// row_indices naming at least this share of X's rows, each once, are fit as
/// a 0/1 weight on every row, so the kernels read whole columns with vector
/// loads; fewer rows, or repeated ones, are gathered. Part of the results:
/// the two group sums differently, so changing it changes their last bits.
/// Half is about where they break even (AMD Ryzen 5 7640U, one thread: there
/// the mask is 9% faster on wide data and 3% slower on tall, and either way
/// the gap widens quickly with the share).
const MASK_SHARE: f64 = 0.5;
/// The mask sweeps the rows it leaves out with weight 0, where each enters
/// only as a product with a 0 (w, s, and so the residual's terms), and those
/// are exactly 0 while x, z = (x - center) / scale and the residual stay
/// finite. Left-out values within this many scales of the center keep z^2
/// finite, and the residual's drift (each step's standardized change times z)
/// unless a step exceeds 1e150; otherwise the fit gathers instead.
const LEFT_OUT_Z_MAX: f64 = 3e150;

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
        let blocks =
            rv.chunks_exact_mut(regs::<S>()).zip(ev.chunks_exact_mut(regs::<S>())).zip(wv.chunks_exact(regs::<S>()));
        for (b, ((r, eta), w)) in blocks.enumerate() {
            let xp = if SHIFT { load(prev, b * LANES) } else { zero };
            let xn = if MOMENTS { load(next, b * LANES) } else { zero };
            let (xp, xn) = (regs_of::<S>(&xp), regs_of::<S>(&xn));
            for m in 0..regs::<S>() {
                let (mut rm, mut em) = (v(r[m]), v(eta[m]));
                Self::row(k, (v(xp[m]), v(xn[m])), &mut rm, &mut em, v(w[m]), &mut acc[m]);
                if SHIFT {
                    (r[m], eta[m]) = (rm.1, em.1);
                }
            }
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
                .map(|(k, ((r, eta), w))| ARCH.dispatch(Chunks { pass: self, i0: k * CHUNK, r, eta, w }))
                .collect();
            parts.into_iter().reduce(Sums::add).expect("a column has rows")
        } else {
            ARCH.dispatch(Chunks { pass: self, i0: 0, r, eta, w })
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
        let chunks = r.chunks_mut(CHUNK).zip(eta.chunks_mut(CHUNK)).zip(w.chunks(CHUNK));
        let mut parts = chunks.enumerate().map(|(k, ((r, eta), w))| pass.chunk(simd, i0 + k * CHUNK, r, eta, w));
        let first = parts.next().expect("a column has rows");
        parts.fold(first, Sums::add)
    }
}

// The solver

struct Params<'a> {
    alpha: f64,
    lambdas: Option<&'a [f64]>,
    nlambda: usize,
    eps: f64,
    lambda_min: f64,
    lambda_log_scale: bool,
    penalty_factor: &'a [f64],
    dfmax: usize,
    max_iter: i64,
    stop_on_saturation: bool,
    r_compatible: bool,
    ybar: f64,
    null_deviance: f64,
    /// Called after each lambda is fit; true ends the path there
    on_lambda: Option<&'a OnLambda<'a>>,
}

type OnLambda<'a> = dyn Fn(&Path, usize) -> Result<bool, String> + Sync + 'a;

#[derive(PartialEq)]
enum Status {
    Converged,
    Saturated,
    MaxIter,
}

/// What the Python side needs to warn about.
enum Event {
    PrefitSaturated,
    Dfmax(usize),
    Saturated(usize),
    MaxIter(usize),
}

struct Path {
    center: Vec<f64>,
    scale: Vec<f64>,
    col_map: Vec<usize>,
    lambda: Vec<f64>,
    intercept: Vec<f64>,
    coef_rows: Vec<Option<Vec<usize>>>,
    coef_vals: Vec<Option<Vec<f64>>>,
    iters: Vec<i64>,
    deviance: Vec<f64>,
    n_reject: Vec<i64>,
    events: Vec<Event>,
}

struct State {
    eta: Rows,
    p: Rows,
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

struct Fit<'a, D> {
    x: &'a D,
    y: &'a [f64],
    params: &'a Params<'a>,
    /// With row_indices fit as a 0/1 weight on every row of X: 1 for rows
    /// being fit. The kernels then read whole columns, and y is 0 elsewhere.
    mask: Option<&'a [f64]>,
    /// With a mask, the rows being fit in row_indices' order, which center
    /// and scale sum in, as gathering does: for a column far from zero, the
    /// center's rounding moves every centered value
    rows: Option<&'a [usize]>,
    /// The number of rows being fit
    n_fit: usize,
}

impl<D: Design> Fit<'_, D> {
    /// Each column's center and scale, from the rows being fit; and, with a
    /// mask, whether the rows it leaves out are safe to sweep (see
    /// LEFT_OUT_Z_MAX). X may hold anything in rows that aren't fit.
    fn center_scale(&self) -> Result<(Vec<f64>, Vec<f64>, bool), String> {
        let stats: Vec<(f64, f64, bool, bool)> = (0..self.x.ncol())
            .into_par_iter()
            .map(|j| {
                let c = self.x.col(j);
                let n = c.n();
                let nf = self.n_fit as f64;
                let (mut acc, mut invalid) = (0.0, false);
                match self.rows {
                    Some(rows) => rows.iter().for_each(|&i| {
                        let v = c.at(i);
                        invalid |= !v.is_finite();
                        acc += v;
                    }),
                    None => (0..n).for_each(|i| {
                        let v = c.at(i);
                        invalid |= !v.is_finite();
                        acc += v;
                    }),
                }
                let mean = acc / nf;
                let mut acc = 0.0;
                match self.rows {
                    Some(rows) => rows.iter().for_each(|&i| acc += (c.at(i) - mean).powi(2)),
                    None => (0..n).for_each(|i| acc += (c.at(i) - mean).powi(2)),
                }
                let (mut out_lo, mut out_hi, mut out_finite) = (f64::INFINITY, f64::NEG_INFINITY, true);
                for i in (0..n).filter(|&i| !self.fits(i)) {
                    let v = c.at(i);
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

    fn compute_z(&self, path: &Path, s: &[f64]) -> Vec<f64> {
        let s_sum = seqsum(s);
        let nf = self.n_fit as f64;
        path.col_map
            .par_iter()
            .with_min_len(16)
            .map(|&jj| standardized_score(&self.x.col(jj), s, s_sum, path.center[jj], path.scale[jj], nf))
            .collect()
    }

    fn y_at(&self, i: usize) -> f64 {
        self.y[i]
    }

    #[inline(always)]
    fn fits(&self, i: usize) -> bool {
        self.mask.is_none_or(|m| m[i] != 0.0)
    }

    /// p, w, s, r from eta; returns -log-likelihood with probabilities clipped
    /// as R does.
    fn update_irls(&self, st: &mut State) -> f64 {
        for i in 0..st.eta.len() {
            if !self.fits(i) {
                (st.p[i], st.w[i], st.s[i], st.r[i]) = (0.0, 0.0, 0.0, 0.0);
                continue;
            }
            let e = st.eta[i];
            if e > ETA_MAX {
                st.p[i] = 1.0;
                st.w[i] = W_MIN;
            } else if e < -ETA_MAX {
                st.p[i] = 0.0;
                st.w[i] = W_MIN;
            } else {
                let ee = exp(e);
                st.p[i] = ee / (1.0 + ee);
                st.w[i] = st.p[i] * (1.0 - st.p[i]);
            }
            st.s[i] = self.y_at(i) - st.p[i];
            st.r[i] = st.s[i] / st.w[i];
        }
        let mut acc = 0.0;
        for i in (0..st.eta.len()).filter(|&i| self.fits(i)) {
            acc += if self.y_at(i) == 1.0 { log(st.p[i]) } else { log(1.0 - st.p[i]) };
        }
        -acc
    }

    fn update_intercept(st: &mut State) -> f64 {
        let xwr = dot(&st.w, &st.r);
        let shift = xwr / seqsum(&st.w);
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

    fn coordinate_sweep(&self, path: &Path, st: &mut State, lam: f64) -> Result<f64, String> {
        let p = self.params;
        let n = self.x.nrow();
        let nrow = self.n_fit as f64;
        let sum_w = seqsum(&st.w);
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
                // Differs from R biglasso: see the numba backend's coordinate_sweep
                if p.r_compatible { sign * (u.abs() - l1) / v / (1.0 + l2) } else { sign * (u.abs() - l1) / (v + l2) }
            };
            let shift = new_beta - st.beta[j];
            if shift != 0.0 {
                max_update = max_update.max(v * (shift * shift));
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
            path.intercept[il] = Self::update_intercept(st);
            let max_update = self.coordinate_sweep(path, st, lam)?;
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

    fn run(&self, center: Vec<f64>, scale: Vec<f64>) -> Result<Path, String> {
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
            p: Rows::new(nrow, 0.0),
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
                // fit the unpenalized columns alone; see the numba backend
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

fn run_typed<T: Elem>(
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

/// Fit a binomial path; arguments and result as the numba backend's
/// fit_path. Null-model statistics use the same portable math as the fit.
#[pyfunction]
#[pyo3(signature = (x, y, row_indices, alpha, lambdas, nlambda, eps, lambda_min, lambda_log_scale, penalty_factor, dfmax, max_iter, stop_on_saturation, r_compatible, n_threads, boundscheck, row_path = "auto", on_lambda = None))]
#[allow(clippy::too_many_arguments)]
fn fit_path<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyUntypedArray>,
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
) -> PyResult<Bound<'py, PyDict>> {
    if !["auto", "gather", "mask"].contains(&row_path) {
        return Err(PyValueError::new_err("row_path must be 'auto', 'gather' or 'mask'."));
    }
    // biglasso() checks all of this; checked again here so a direct call
    // gets an error rather than a panic
    for (name, a) in [("X", x.as_any()), ("y", y.as_any()), ("penalty factors", penalty_factor.as_any())]
        .into_iter()
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
    let shape = x.shape();
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
            let nrow = x.shape()[0];
            if ri.iter().any(|&i| i < 0 || i as usize >= nrow) {
                return Err(PyValueError::new_err("row_indices has indices outside valid range."));
            }
            Some(ri.iter().map(|&i| i as usize).collect())
        }
        None => None,
    };
    let (ybar, null_deviance) = null_model(y_all, idx.as_deref()).map_err(PyValueError::new_err)?;
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
    let dtype = x.dtype();
    macro_rules! dispatch {
        ($($t:ty),*) => {
            $(if dtype.is_equiv_to(&numpy::dtype::<$t>(py)) {
                let result = run_typed::<$t>(py, x, y_all, rows, &params, n_threads, boundscheck);
                if let Some(e) = raised.lock().unwrap().take() {
                    return Err(e);
                }
                let (path, ran) = result?;
                let d = to_dict(py, path)?;
                d.set_item("row_path", ran)?;
                return Ok(d);
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

#[cfg(test)]
mod tests;
