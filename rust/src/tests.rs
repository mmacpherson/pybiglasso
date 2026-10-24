//! The kernels against plain scalar sums, bit for bit: at every SIMD width
//! this CPU runs, through every kind of column, across chunk and tail sizes,
//! serial and threaded.

use super::*;
use ndarray::{Array2, ShapeBuilder};

/// Values of mixed sign and magnitude, so that summing in a different order
/// changes the last bits; deterministic, so a failure reproduces.
fn values(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
            u * 10f64.powi((s % 9) as i32 - 4)
        })
        .collect()
}

/// The order the kernels promise, written plainly: within each CHUNK of
/// rows, row i of a whole block of LANES goes to accumulator i % LANES, the
/// accumulators are combined, and the rest of the rows follow in order; the
/// chunks' sums are added in order.
fn ordered_sum(terms: &[f64]) -> f64 {
    let mut chunks = terms.chunks(CHUNK).map(|chunk| {
        let body = chunk.len() / LANES * LANES;
        let mut lanes = [0.0; LANES];
        for (i, t) in chunk[..body].iter().enumerate() {
            lanes[i % LANES] += t;
        }
        let mut tail = 0.0;
        for t in &chunk[body..] {
            tail += t;
        }
        combine(&lanes) + tail
    });
    let first = chunks.next().expect("rows");
    chunks.fold(first, |a, b| a + b)
}

/// A pass's sums, r and eta, from the formulas one row at a time.
fn reference(
    shift: Option<(&[f64], f64, f64)>,
    next: Option<&[f64]>,
    r: &[f64],
    eta: &[f64],
    w: &[f64],
) -> ([f64; 4], Vec<f64>, Vec<f64>) {
    let (mut r, mut eta) = (r.to_vec(), eta.to_vec());
    let n = r.len();
    let mut terms = [vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]];
    for i in 0..n {
        if let Some((xp, delta, c)) = shift {
            let d = delta * (xp[i] - c);
            r[i] -= d;
            eta[i] += d;
            terms[0][i] = w[i] * r[i];
        }
        if let Some(xn) = next {
            let wx = w[i] * xn[i];
            terms[1][i] = wx * r[i];
            terms[2][i] = wx * xn[i];
            terms[3][i] = wx;
        }
    }
    let sum =
        |q: usize| if (q == 0 && shift.is_some()) || (q > 0 && next.is_some()) { ordered_sum(&terms[q]) } else { 0.0 };
    ([sum(0), sum(1), sum(2), sum(3)], r, eta)
}

fn bits(s: Sums<f64>) -> [u64; 4] {
    [s.wr, s.xwr, s.wxx, s.wx].map(f64::to_bits)
}

fn bits_of(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// A run's name, sums, r and eta.
type Run = (String, Sums<f64>, Vec<f64>, Vec<f64>);

/// A pass at every width this CPU runs, then as the solver runs it (the
/// widest width) on one thread and on four.
fn every_way<C: Col, const SHIFT: bool, const MOMENTS: bool>(
    pass: &Pass<C, SHIFT, MOMENTS>,
    r: &[f64],
    eta: &[f64],
    w: &[f64],
) -> Vec<Run> {
    let mut out = vec![];
    macro_rules! at {
        ($name:expr, $simd:expr) => {{
            let (mut r, mut eta) = (r.to_vec(), eta.to_vec());
            let sums = Simd::vectorize($simd, Chunks { pass, i0: 0, r: &mut r, eta: &mut eta, w });
            out.push(($name.to_string(), sums, r, eta));
        }};
    }
    at!("scalar", pulp::Scalar);
    at!("scalar128b", pulp::Scalar128b);
    at!("scalar256b", pulp::Scalar256b);
    at!("scalar512b", pulp::Scalar512b);
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(simd) = pulp::x86::V3::try_new() {
            at!("avx2", simd);
        }
        if let Some(simd) = pulp::x86::V4::try_new() {
            at!("avx512", simd);
        }
    }
    for threads in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().expect("pool");
        let (mut r, mut eta) = (r.to_vec(), eta.to_vec());
        let sums = pool.install(|| pass.sweep(r.len(), &mut r, &mut eta, w));
        out.push((format!("sweep, {threads} threads"), sums, r, eta));
    }
    out
}

fn check<C: Col, const SHIFT: bool, const MOMENTS: bool>(
    what: &str,
    pass: &Pass<C, SHIFT, MOMENTS>,
    want: ([f64; 4], Vec<f64>, Vec<f64>),
    r: &[f64],
    eta: &[f64],
    w: &[f64],
) {
    let (sums, r_want, eta_want) = want;
    for (how, got, r_got, eta_got) in every_way(pass, r, eta, w) {
        assert_eq!(bits(got), sums.map(f64::to_bits), "{what}, {how}: sums");
        assert_eq!(bits_of(&r_got), bits_of(&r_want), "{what}, {how}: r");
        assert_eq!(bits_of(&eta_got), bits_of(&eta_want), "{what}, {how}: eta");
    }
}

/// Row counts: all tail, one block, a block and a tail, one chunk, chunks
/// with a tail, and tall enough to share chunks across threads.
const SIZES: [usize; 7] = [5, 8, 13, CHUNK, 3 * CHUNK + 11, PAR_ROWS, PAR_ROWS + 3 * CHUNK + 5];

/// Every pass, on one column's fit rows `x` and `y`, read through `col_x`
/// and `col_y`.
fn check_passes<C: Col>(what: &str, col_x: &C, col_y: &C, x: &[f64], y: &[f64]) {
    let n = x.len();
    let (r, eta) = (values(n, 3), values(n, 4));
    let w: Vec<f64> = values(n, 5).iter().map(|v| v.abs()).collect();
    let (delta, c) = (0.37, 1.25);
    check(
        &format!("{what}, n={n}, shift"),
        &Pass::shift(col_x, delta, c),
        reference(Some((x, delta, c)), None, &r, &eta, &w),
        &r,
        &eta,
        &w,
    );
    check(
        &format!("{what}, n={n}, moments"),
        &Pass::moments(col_y),
        reference(None, Some(y), &r, &eta, &w),
        &r,
        &eta,
        &w,
    );
    check(
        &format!("{what}, n={n}, shift then moments"),
        &Pass::shift_then_moments(col_x, delta, c, col_y),
        reference(Some((x, delta, c)), Some(y), &r, &eta, &w),
        &r,
        &eta,
        &w,
    );
}

#[test]
fn passes_contiguous() {
    for n in SIZES {
        let (x, y) = (values(n, 1), values(n, 2));
        check_passes("contiguous", &Contig(&x[..]), &Contig(&y[..]), &x, &y);
        let x32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        let y32: Vec<f32> = y.iter().map(|&v| v as f32).collect();
        let (xf, yf): (Vec<f64>, Vec<f64>) =
            (x32.iter().map(|&v| v as f64).collect(), y32.iter().map(|&v| v as f64).collect());
        check_passes("contiguous f32", &Contig(&x32[..]), &Contig(&y32[..]), &xf, &yf);
    }
}

#[test]
fn passes_strided() {
    for n in SIZES {
        // C order: each column's elements three apart
        let x = Array2::from_shape_vec((n, 3), values(3 * n, 6)).expect("shape");
        let (cx, cy) = (Strided(x.column(0)), Strided(x.column(2)));
        check_passes("strided", &cx, &cy, &x.column(0).to_vec(), &x.column(2).to_vec());
    }
}

#[test]
fn passes_gathered() {
    for n in SIZES {
        // a row subset in shuffled order, with repeats, from a taller X
        let nrow = n + 41;
        let idx: Vec<usize> = values(n, 7).iter().map(|v| (v.abs() * 1e9) as usize % nrow).collect();
        let x = Array2::from_shape_vec((nrow, 2).f(), values(2 * nrow, 8)).expect("shape");
        let (x0, x1) = (x.column(0).to_slice().expect("Fortran"), x.column(1).to_slice().expect("Fortran"));
        let (cx, cy) = (Gather { col: Contig(x0), idx: &idx }, Gather { col: Contig(x1), idx: &idx });
        let (gx, gy): (Vec<f64>, Vec<f64>) =
            (idx.iter().map(|&i| x0[i]).collect(), idx.iter().map(|&i| x1[i]).collect());
        check_passes("gathered", &cx, &cy, &gx, &gy);
    }
}

#[test]
fn dot_matches_lanes() {
    // unchunked: the whole column's blocks into LANES accumulators
    for n in SIZES {
        let (x, s) = (values(n, 9), values(n, 10));
        let body = n / LANES * LANES;
        let mut lanes = [0.0; LANES];
        for i in 0..body {
            lanes[i % LANES] += x[i] * s[i];
        }
        let mut tail = 0.0;
        for i in body..n {
            tail += x[i] * s[i];
        }
        let want = combine(&lanes) + tail;
        assert_eq!(dot_col(&Contig(&x[..]), &s).to_bits(), want.to_bits(), "n={n}");
    }
}

#[test]
fn ordered_sum_is_order_sensitive() {
    // the check above can fail: these values sum differently in plain order
    let t = values(3 * CHUNK + 11, 11);
    assert_ne!(ordered_sum(&t).to_bits(), seqsum(&t).to_bits());
}
