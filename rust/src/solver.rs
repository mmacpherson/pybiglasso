use ndarray::{ArrayView1, ArrayView2};
// Portable exp/log/log1p keep the Rust path's bits independent of the
// platform math library, whose last-bit rounding varies by OS.
use libm::{exp, log, log1p};
use pulp::{Simd, WithSimd};
use rayon::prelude::*;

#[path = "gaussian.rs"]
mod gaussian;

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Family {
    Binomial,
    Gaussian,
}

impl Family {
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

const LANES: usize = 8;
const SCALE_MIN: f64 = 1e-6;
const ETA_MAX: f64 = 10.0;
const W_MIN: f64 = 0.0001;

fn irls_limits(r_compatible: bool) -> (f64, f64) {
    if r_compatible {
        (ETA_MAX, W_MIN)
    } else {
        // Differs from R biglasso: use glmnet's pmin boundary instead of
        // |eta| > 10 to reduce clipping's score bias. Retain positive boundary
        // curvature: our working residual s/w needs it, unlike glmnet's
        // undivided residual, which allows zero weights.
        let p_min = 1e-9;
        (log(1.0 / p_min - 1.0), p_min * (1.0 - p_min))
    }
}

#[inline(always)]
fn irls_probability_weight(eta: f64, eta_max: f64, w_min: f64) -> (f64, f64) {
    if eta > eta_max {
        (1.0, w_min)
    } else if eta < -eta_max {
        (0.0, w_min)
    } else {
        let e = exp(eta);
        let probability = e / (1.0 + e);
        (probability, probability * (1.0 - probability))
    }
}

// Element types and column access

pub(crate) trait Elem: Copy + Send + Sync {
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
pub(crate) trait Col: Sync {
    fn n(&self) -> usize;
    /// Row i of the fit; i < n().
    fn at(&self, i: usize) -> f64;
    /// Rows i0 .. i0 + LANES; i0 + LANES <= n().
    #[inline(always)]
    fn block(&self, i0: usize) -> [f64; LANES] {
        std::array::from_fn(|k| self.at(i0 + k))
    }
    /// block(i0), read in one ascending pass over a column. `hint` is the
    /// pass's own position (one per kernel call, never shared), so a chunked
    /// column can step to the next chunk instead of searching for it.
    #[inline(always)]
    fn block_seq(&self, i0: usize, _hint: &mut usize) -> [f64; LANES] {
        self.block(i0)
    }
    /// at(i), read in one ascending pass, as for block_seq.
    #[inline(always)]
    fn at_seq(&self, i: usize, _hint: &mut usize) -> f64 {
        self.at(i)
    }
}

/// A column contiguous in memory (X in Fortran order).
pub(crate) struct Contig<'a, T, const CHECKED: bool>(&'a [T]);
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
        // For int8 on NEON, map gives one 64-bit load, not eight byte loads.
        // On x86, indexing exposes packed sign-extension/conversion instead.
        // Same conversions and lane order; measurements: benchmarks/int8.md.
        #[cfg(target_arch = "aarch64")]
        {
            rows.map(Elem::f)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            std::array::from_fn(|k| rows[k].f())
        }
    }
}

/// A column with an arbitrary element stride (X in C order, or a view).
pub(crate) struct Strided<'a, T, const CHECKED: bool>(ArrayView1<'a, T>);
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
pub(crate) struct Gather<'a, C> {
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
    #[inline(always)]
    fn block_seq(&self, i0: usize, hint: &mut usize) -> [f64; LANES] {
        self.col.block_seq(self.i0 + i0, hint)
    }
    #[inline(always)]
    fn at_seq(&self, i: usize, hint: &mut usize) -> f64 {
        self.col.at_seq(self.i0 + i, hint)
    }
}

pub(crate) trait Design: Sync {
    type C<'b>: Col
    where
        Self: 'b;
    fn col(&self, j: usize) -> Self::C<'_>;
    fn ncol(&self) -> usize;
    fn nrow(&self) -> usize;
}

/// Heterogeneous mapped columns: dispatch storage, never the reduction order.
#[derive(Clone, Copy)]
pub(crate) enum MappedCol<'a> {
    F64(&'a [f64]),
    F32(&'a [f32]),
    I8(&'a [i8]),
    U8(&'a [u8]),
    I16(&'a [i16]),
    U64(&'a [u64]),
    I64(&'a [i64]),
    U32(&'a [u32]),
    I32(&'a [i32]),
    U16(&'a [u16]),
    Bits { bytes: &'a [u8], offset: usize, len: usize },
}

impl Col for MappedCol<'_> {
    fn n(&self) -> usize {
        match self {
            Self::F64(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::I8(v) => v.len(),
            Self::U8(v) => v.len(),
            Self::I16(v) => v.len(),
            Self::U64(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::U16(v) => v.len(),
            Self::Bits { len, .. } => *len,
        }
    }
    #[inline(always)]
    fn at(&self, i: usize) -> f64 {
        match self {
            Self::F64(v) => v[i],
            Self::F32(v) => v[i].f(),
            Self::I8(v) => v[i].f(),
            Self::U8(v) => v[i].f(),
            Self::I16(v) => v[i].f(),
            Self::U64(v) => v[i].f(),
            Self::I64(v) => v[i].f(),
            Self::U32(v) => v[i].f(),
            Self::I32(v) => v[i].f(),
            Self::U16(v) => v[i].f(),
            Self::Bits { bytes, offset, len } => {
                assert!(i < *len);
                let bit = offset + i;
                ((bytes[bit / 8] >> (bit % 8)) & 1).f()
            }
        }
    }
    #[inline(always)]
    fn block(&self, i0: usize) -> [f64; LANES] {
        match self {
            Self::F64(v) => Contig::<_, true>(v).block(i0),
            Self::F32(v) => Contig::<_, true>(v).block(i0),
            Self::I8(v) => Contig::<_, true>(v).block(i0),
            Self::U8(v) => Contig::<_, true>(v).block(i0),
            Self::I16(v) => Contig::<_, true>(v).block(i0),
            Self::U64(v) => Contig::<_, true>(v).block(i0),
            Self::I64(v) => Contig::<_, true>(v).block(i0),
            Self::U32(v) => Contig::<_, true>(v).block(i0),
            Self::I32(v) => Contig::<_, true>(v).block(i0),
            Self::U16(v) => Contig::<_, true>(v).block(i0),
            Self::Bits { bytes, offset, len } => {
                assert!(*len >= LANES && i0 <= *len - LANES);
                // The block's LANES bits span at most two bytes: read them
                // once rather than locating each bit.
                let bit = offset + i0;
                let lo = u16::from(bytes[bit / 8]);
                let hi = if bit.is_multiple_of(8) { 0 } else { u16::from(bytes[bit / 8 + 1]) };
                let bits = (lo | hi << 8) >> (bit % 8);
                std::array::from_fn(|k| ((bits >> k) & 1).f())
            }
        }
    }
}

// Chunk boundaries affect addressing only. Logical kernel blocks and reduction
// accumulators are unchanged, including when a block crosses several chunks.
#[derive(Clone)]
pub(crate) struct ChunkedColumn<'a> {
    pub(crate) chunks: Vec<MappedCol<'a>>,
    pub(crate) ends: Vec<usize>,
}
impl<'a> ChunkedColumn<'a> {
    pub(crate) fn new(chunks: Vec<MappedCol<'a>>, nrow: usize) -> Result<Self, String> {
        let mut end = 0usize;
        let mut ends = Vec::with_capacity(chunks.len());
        for chunk in &chunks {
            end = end.checked_add(chunk.n()).ok_or("Chunk lengths overflow.")?;
            ends.push(end);
        }
        if end != nrow || chunks.is_empty() {
            return Err("Chunk lengths must match X rows.".into());
        }
        Ok(Self { chunks, ends })
    }
}

#[derive(Clone, Copy)]
pub(crate) struct ChunkedCol<'a> {
    chunks: &'a [MappedCol<'a>],
    ends: &'a [usize],
}
impl ChunkedCol<'_> {
    #[inline(always)]
    fn locate(&self, i: usize) -> (usize, usize) {
        assert!(i < self.n());
        let k = if self.chunks.len() == 1 { 0 } else { self.ends.partition_point(|&end| end <= i) };
        (k, i - if k == 0 { 0 } else { self.ends[k - 1] })
    }
}
// A one-chunk column (the common case) skips locating rows entirely; the
// chunk's own indexing checks bounds, since its length is the column's.
impl Col for ChunkedCol<'_> {
    #[inline(always)]
    fn n(&self) -> usize {
        *self.ends.last().expect("nonempty chunks")
    }
    #[inline(always)]
    fn at(&self, i: usize) -> f64 {
        if let [only] = self.chunks {
            return only.at(i);
        }
        let (k, local) = self.locate(i);
        self.chunks[k].at(local)
    }
    #[inline(always)]
    fn block(&self, i0: usize) -> [f64; LANES] {
        if let [only] = self.chunks {
            return only.block(i0);
        }
        let (k, local) = self.locate(i0);
        if self.chunks[k].n() - local >= LANES {
            self.chunks[k].block(local)
        } else {
            std::array::from_fn(|lane| self.at(i0 + lane))
        }
    }
    #[inline(always)]
    fn block_seq(&self, i0: usize, hint: &mut usize) -> [f64; LANES] {
        if let [only] = self.chunks {
            return only.block(i0);
        }
        // Step past chunks ending at or before i0 (empty chunks included);
        // anything but an ascending pass falls back to searching.
        let mut k = *hint;
        while k < self.ends.len() && self.ends[k] <= i0 {
            k += 1;
        }
        let start = if k == 0 { 0 } else { self.ends[k - 1] };
        if k < self.ends.len() && start <= i0 && self.ends[k] - i0 >= LANES {
            *hint = k;
            return self.chunks[k].block(i0 - start);
        }
        self.block(i0)
    }
    #[inline(always)]
    fn at_seq(&self, i: usize, hint: &mut usize) -> f64 {
        if let [only] = self.chunks {
            return only.at(i);
        }
        let mut k = *hint;
        while k < self.ends.len() && self.ends[k] <= i {
            k += 1;
        }
        let start = if k == 0 { 0 } else { self.ends[k - 1] };
        if k < self.ends.len() && start <= i {
            *hint = k;
            return self.chunks[k].at(i - start);
        }
        self.at(i)
    }
}

pub(crate) struct MappedX<'a> {
    pub(crate) columns: Vec<ChunkedColumn<'a>>,
}
impl Design for MappedX<'_> {
    type C<'b>
        = ChunkedCol<'b>
    where
        Self: 'b;
    fn col(&self, j: usize) -> ChunkedCol<'_> {
        ChunkedCol { chunks: &self.columns[j].chunks, ends: &self.columns[j].ends }
    }
    fn ncol(&self) -> usize {
        self.columns.len()
    }
    fn nrow(&self) -> usize {
        *self.columns[0].ends.last().expect("nonempty chunks")
    }
}

pub(crate) struct ContigX<'a, T, const CHECKED: bool> {
    pub(crate) x: ArrayView2<'a, T>,
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

pub(crate) struct StridedX<'a, T, const CHECKED: bool> {
    pub(crate) x: ArrayView2<'a, T>,
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

pub(crate) struct GatherX<'a, D> {
    pub(crate) x: D,
    pub(crate) idx: &'a [usize],
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

/// Sequential sum in left-to-right row order.
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
pub(crate) fn null_model(y: &[f64], rows: Option<&[usize]>) -> Result<(f64, f64), &'static str> {
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
fn load<C: Col>(c: &C, i0: usize, hint: &mut usize) -> Block {
    Block(c.block_seq(i0, hint))
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
            let mut hint = 0;
            for (b, s) in sv.chunks_exact(regs::<S>()).enumerate() {
                let x = load(c, b * LANES, &mut hint);
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

/// Stable score and curvature from centered values in left-to-right row order.
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

/// Standardized score x_j'(s) / n, with a centered fallback.
/// Differs from R biglasso, which always uses raw moments: cancellation can
/// lose precision when a column's mean is large relative to its spread.
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
/// IRLS has portable exp/log work per row; keep its dispatch threshold
/// separate from the coordinate sweep's arithmetic-only passes.
const PAR_IRLS_ROWS: usize = 16_384;
/// row_indices naming at least this share of X's rows, each once, are fit as
/// a 0/1 weight on every row, so the kernels read whole columns with vector
/// loads; fewer rows, or repeated ones, are gathered. Part of the results:
/// the two group sums differently, so changing it changes their last bits.
/// Half is about where they break even (AMD Ryzen 5 7640U, one thread: there
/// the mask is 9% faster on wide data and 3% slower on tall, and either way
/// the gap widens quickly with the share).
pub(crate) const MASK_SHARE: f64 = 0.5;
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
        let (mut hint_prev, mut hint_next) = (0, 0);
        for (b, ((r, eta), w)) in blocks.enumerate() {
            let xp = if SHIFT { load(prev, b * LANES, &mut hint_prev) } else { zero };
            let xn = if MOMENTS { load(next, b * LANES, &mut hint_next) } else { zero };
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

#[derive(PartialEq)]
enum Status {
    Converged,
    Saturated,
    MaxIter,
}

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

struct State {
    eta: Rows,
    // Scratch used only inside IRLS: probabilities on the serial path;
    // log-likelihood terms on the parallel path. Neither escapes the step.
    irls_work: Rows,
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
                    None => (0..n).for_each(|i| {
                        let v = c.at_seq(i, &mut hint);
                        invalid |= !v.is_finite();
                        acc += v;
                    }),
                }
                let mean = acc / nf;
                let mut acc = 0.0;
                let mut hint = 0;
                match self.rows {
                    Some(rows) => rows.iter().for_each(|&i| acc += (c.at_seq(i, &mut hint) - mean).powi(2)),
                    None => (0..n).for_each(|i| acc += (c.at_seq(i, &mut hint) - mean).powi(2)),
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

    /// w, s, r from eta; returns -log-likelihood with probabilities clipped
    /// at the mode's boundary. Rows are independent; the loss sum stays serial.
    fn update_irls(&self, st: &mut State) -> f64 {
        if st.eta.len() >= PAR_IRLS_ROWS && rayon::current_num_threads() > 1 {
            return self.update_irls_parallel(st);
        }
        let (eta_max, w_min) = irls_limits(self.params.r_compatible);
        for i in 0..st.eta.len() {
            if !self.fits(i) {
                (st.irls_work[i], st.w[i], st.s[i], st.r[i]) = (0.0, 0.0, 0.0, 0.0);
                continue;
            }
            (st.irls_work[i], st.w[i]) = irls_probability_weight(st.eta[i], eta_max, w_min);
            st.s[i] = self.y_at(i) - st.irls_work[i];
            st.r[i] = st.s[i] / st.w[i];
        }
        let mut acc = 0.0;
        for i in (0..st.eta.len()).filter(|&i| self.fits(i)) {
            acc += if self.y_at(i) == 1.0 { log(st.irls_work[i]) } else { log(1.0 - st.irls_work[i]) };
        }
        -acc
    }

    // Keep the Rayon plumbing out of the original serial loop's optimizer
    // scope; both the fused serial loop and a combined function regressed
    // single-thread fits in the before/after measurements.
    #[inline(never)]
    fn update_irls_parallel(&self, st: &mut State) -> f64 {
        let (eta_max, w_min) = irls_limits(self.params.r_compatible);
        let update = |i: usize, (((w, s), r), loss): (((&mut f64, &mut f64), &mut f64), &mut f64)| {
            if !self.fits(i) {
                (*w, *s, *r, *loss) = (0.0, 0.0, 0.0, 0.0);
                return;
            }
            let (probability, weight) = irls_probability_weight(st.eta[i], eta_max, w_min);
            *w = weight;
            *s = self.y_at(i) - probability;
            *r = *s / *w;
            *loss = if self.y_at(i) == 1.0 { log(probability) } else { log(1.0 - probability) };
        };
        st.w.par_iter_mut()
            .zip(st.s.par_iter_mut())
            .zip(st.r.par_iter_mut())
            .zip(st.irls_work.par_iter_mut())
            .enumerate()
            .for_each(|(i, row)| update(i, row));
        // Only the independent row work is parallel. Keep the original
        // sequential loss sum, including its masked-row omissions: chunk
        // subtotals would change its rounding and could tip saturation.
        let mut acc = 0.0;
        for i in (0..st.eta.len()).filter(|&i| self.fits(i)) {
            acc += st.irls_work[i];
        }
        -acc
    }

    fn update_intercept(st: &mut State, sum_w: f64) -> f64 {
        let xwr = dot(&st.w, &st.r);
        let shift = xwr / sum_w;
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

    fn coordinate_sweep(&self, path: &Path, st: &mut State, lam: f64, sum_w: f64) -> Result<f64, String> {
        let p = self.params;
        let n = self.x.nrow();
        let nrow = self.n_fit as f64;
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
            // Differs from R biglasso, which always uses raw moments: use
            // centered values when cancellation would make them unreliable.
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
                // Differs from R biglasso: R divides by v * (1 + l2), while
                // the documented elastic-net objective requires v + l2.
                // r_compatible retains R's arithmetic; l2 = 0 agrees in both.
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
            // W stays fixed through the intercept and coordinate sweep. Both
            // used to repeat this same sequential sum; reuse its exact bits.
            let sum_w = seqsum(&st.w);
            path.intercept[il] = Self::update_intercept(st, sum_w);
            let max_update = self.coordinate_sweep(path, st, lam, sum_w)?;
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

    // Differs from R biglasso: report stable loss at the returned predictor.
    // R uses clipped probabilities from before the final coordinate sweep;
    // r_compatible keeps that earlier loss instead of calling this function.
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

    pub(crate) fn run(&self, center: Vec<f64>, scale: Vec<f64>) -> Result<Path, String> {
        if self.params.family == Family::Gaussian {
            return self.run_gaussian(center, scale);
        }
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
            irls_work: Rows::new(nrow, 0.0),
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
                // Differs from R biglasso, whose default grid is all zero
                // with unpenalized columns. Fit those columns first: their
                // residuals determine lambda_max for the penalized columns.
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
            // Differs from R biglasso, which fits a grid that overflowed (at a
            // tiny alpha, say) and returns its nonfinite penalties
            if path.lambda.iter().any(|value| !value.is_finite()) {
                return Err("The automatic lambda grid must be finite; increase alpha or supply a finite grid.".into());
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

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

#[cfg(test)]
#[path = "kernel_tests.rs"]
mod kernel_tests;
