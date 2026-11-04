//! The fixed-order reductions both families share: sequential sums, and
//! SIMD dot products in LANES accumulators combined in a fixed tree.

use super::*;

// Reductions

#[inline(always)]
pub(super) fn combine(acc: &[f64; LANES]) -> f64 {
    ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]))
}

/// Sequential sum in left-to-right row order.
pub(super) fn seqsum(a: &[f64]) -> f64 {
    let mut out = 0.0;
    for &v in a {
        out += v;
    }
    out
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

#[inline(always)]
pub(super) fn dispatch<O: WithSimd>(op: O) -> O::Output {
    // These controls exist in the native test executable only, never the wheel.
    #[cfg(test)]
    match std::env::var("PYBIGLASSO_TEST_SIMD").as_deref() {
        Ok("scalar") => return Simd::vectorize(pulp::Scalar, op),
        Ok("128") => return Simd::vectorize(pulp::Scalar128b, op),
        Ok("256") => return Simd::vectorize(pulp::Scalar256b, op),
        Ok("512") => return Simd::vectorize(pulp::Scalar512b, op),
        #[cfg(target_arch = "x86_64")]
        Ok("avx2") => return Simd::vectorize(pulp::x86::V3::try_new().expect("AVX2 available"), op),
        #[cfg(target_arch = "x86_64")]
        Ok("avx512") => return Simd::vectorize(pulp::x86::V4::try_new().expect("AVX512 available"), op),
        _ => {}
    }
    ARCH.dispatch(op)
}

/// A block of LANES rows, aligned so it reads as whole vectors at any width.
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(64))]
pub(super) struct Block(pub(super) [f64; LANES]);

/// A vector of rows that starts on a cache line: a register's worth of rows
/// read from it never straddles two lines, which with AVX-512 can double a
/// sweep's time.
pub(super) struct Rows {
    blocks: Vec<Block>,
    len: usize,
}
impl Rows {
    pub(super) fn new(len: usize, value: f64) -> Self {
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
pub(super) struct V<S: Simd>(pub(super) S, pub(super) S::f64s);
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
pub(super) trait Num:
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
pub(super) fn regs<S: Simd>() -> usize {
    LANES / S::F64_LANES
}

/// LANES accumulators, as the first `regs` registers, combined.
#[inline(always)]
pub(super) fn fold<S: Simd>(acc: [V<S>; LANES]) -> f64 {
    let acc = acc.map(|v| v.1);
    let lanes: &[f64] = bytemuck::cast_slice(&acc[..regs::<S>()]);
    combine(lanes.try_into().expect("LANES lanes"))
}

/// Column c's rows i0 .. i0 + LANES.
#[inline(always)]
pub(super) fn load<C: Col>(c: &C, i0: usize, hint: &mut usize) -> Block {
    Block(c.block_seq(i0, hint))
}

/// A block's rows as registers.
#[inline(always)]
pub(super) fn regs_of<S: Simd>(b: &Block) -> &[S::f64s] {
    bytemuck::cast_slice(&b.0)
}

/// Decode aligned runs by byte, with type/alignment dispatch outside the
/// loop. Generic and crossing readers retain their checked block loads.
#[inline(always)]
pub(super) fn for_each_block<C: Col>(c: &C, n: usize, mut visit: impl FnMut(usize, &Block)) {
    if let Some(bytes) = c.packed_bytes() {
        for (b, &byte) in bytes[..n / LANES].iter().enumerate() {
            visit(b * LANES, &BOOL_BLOCKS[usize::from(byte)]);
        }
    } else {
        let mut hint = 0;
        for i in (0..n / LANES * LANES).step_by(LANES) {
            visit(i, &load(c, i, &mut hint));
        }
    }
}

/// A serial scan, with storage dispatch fixed over each run. In particular,
/// centering still adds individual rows left to right, never chunk subtotals.
#[inline(always)]
pub(super) fn for_each_value<C: Col>(c: &C, mut visit: impl FnMut(f64)) {
    let mut start = 0;
    let mut hint = 0;
    while start < c.n() {
        let (run, end) = c.run(start, c.n());
        let n = end - start;
        let body = n / LANES * LANES;
        for_each_block(&run, body, |_, block| {
            // Conversion may use a whole byte/vector, but visiting its lanes
            // in order preserves the scalar sum across blocks and runs.
            for &value in &block.0 {
                visit(value);
            }
        });
        for i in body..n {
            visit(run.at_seq(i, &mut hint));
        }
        start = end;
    }
}

/// sum(x * s), in LANES accumulators.
/// Differs from R biglasso, which sums serially: the fixed accumulator order
/// lets SIMD and threads share the work without changing the result, so
/// scores agree with R's to floating-point precision rather than bit for bit.
#[inline(always)]
pub(super) fn dot_col<C: Col>(c: &C, s: &[f64]) -> f64 {
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
            let mut start = 0;
            while start < body {
                let (run, end) = c.run(start, body);
                let mut hint = 0;
                let range = start / LANES * regs::<S>()..end / LANES * regs::<S>();
                let blocks = sv[range].chunks_exact(regs::<S>());
                let mut add = |x: &Block, s: &[S::f64s]| {
                    let x = regs_of::<S>(x);
                    for m in 0..regs::<S>() {
                        acc[m] += v(x[m]) * v(s[m]);
                    }
                };
                if let Some(bytes) = run.packed_bytes() {
                    for (&byte, s) in bytes[..(end - start) / LANES].iter().zip(blocks) {
                        add(&BOOL_BLOCKS[usize::from(byte)], s);
                    }
                } else {
                    for (b, s) in blocks.enumerate() {
                        add(&load(&run, b * LANES, &mut hint), s);
                    }
                }
                start = end;
            }
            let mut tail = 0.0;
            for i in body..c.n() {
                tail += c.at(i) * s[i];
            }
            fold(acc) + tail
        }
    }
    dispatch(Op(c, s))
}

/// sum(a * b) over two vectors, in LANES accumulators.
pub(super) fn dot(a: &[f64], b: &[f64]) -> f64 {
    dot_col(&Contig::<_, true>(a), b)
}

/// Standardized score x_j'(s) / n, with a centered fallback.
/// Differs from R biglasso, which always uses raw moments: cancellation can
/// lose precision when a column's mean is large relative to its spread. R's
/// KKT scan also divides by scale * n, but its Gaussian coordinate update
/// divides by scale and then by n.
#[inline(always)]
pub(super) fn standardized_score<C: Col>(c: &C, s: &[f64], s_sum: f64, center: f64, scale: f64, n_fit: f64) -> f64 {
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
