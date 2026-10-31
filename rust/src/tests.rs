//! The kernels against plain scalar sums, bit for bit: at every SIMD width
//! this CPU runs, through every kind of column, bounds-checked or not, across
//! chunk and tail sizes, serial and threaded.

use super::*;
use ndarray::{Array2, ShapeBuilder};

#[test]
fn intercept_solves_weighted_normal_equation_with_shared_weight_sum() {
    let n = 20;
    let mut st = State {
        eta: Rows::new(n, 2.0),
        irls_work: Rows::new(n, 0.0),
        w: Rows::new(n, 0.0),
        s: Rows::new(n, 0.0),
        r: Rows::new(n, 0.0),
        beta: vec![],
        active: vec![],
        strong: vec![],
        z: vec![],
        intercept: 0.5,
        sum_w_residuals: 0.0,
    };
    for i in 0..n {
        st.w[i] = [0.125, 0.25, 0.375, 0.5][i % 4];
        st.r[i] = [-1.0, 0.0, 1.0, 2.0][i % 4];
    }
    // Each block has sum(w) = sum(w*r) = 1.25, hence shift = 1.
    // All terms are dyadic: these expectations need no rounding allowance.
    let sum_w = seqsum(&st.w);
    assert_eq!(sum_w, 6.25);
    assert_eq!(Fit::<ContigX<f64, true>>::update_intercept(&mut st, sum_w), 1.5);
    assert_eq!(st.intercept, 1.5);
    assert_eq!(st.sum_w_residuals, 0.0);
    assert_eq!(seqsum(&st.w), sum_w);
    for i in 0..n {
        assert_eq!(st.r[i], [-2.0, -1.0, 0.0, 1.0][i % 4]);
        assert_eq!(st.eta[i], 3.0);
    }
}

#[test]
fn irls_matches_row_formulas_and_sequential_loss() {
    for compatible in [false, true] {
        for wrong_label in [false, true] {
            check_irls_row_formulas(compatible, wrong_label);
        }
    }
}

fn check_irls_row_formulas(compatible: bool, wrong_label: bool) {
    // Independent specifications: R's boundary and glmnet's pmin=1e-9.
    // Neighbors pin the strict comparison, including the new default cutoff.
    let cutoff = if compatible { 10.0 } else { libm::log(1.0 / 1e-9 - 1.0) };
    let floor = if compatible { 0.0001 } else { 1e-9 * (1.0 - 1e-9) };
    let probes = [
        -1000.0,
        -cutoff.next_up(),
        -cutoff,
        -cutoff.next_down(),
        -11.0,
        -10.0,
        -0.7,
        0.0,
        0.7,
        10.0,
        11.0,
        cutoff.next_down(),
        cutoff,
        cutoff.next_up(),
        1000.0,
    ];
    for n in [7, CHUNK + 1, PAR_IRLS_ROWS - 1, PAR_IRLS_ROWS, PAR_IRLS_ROWS + 7, 40_003] {
        let eta: Vec<f64> = (0..n).map(|i| probes[i % probes.len()]).collect();
        let y: Vec<f64> = eta.iter().map(|&e| if (e > 0.0) != wrong_label { 1.0 } else { 0.0 }).collect();
        let mask: Vec<f64> = (0..n).map(|i| if i % 3 == 0 { 0.0 } else { 1.0 }).collect();
        let array = Array2::<f64>::zeros((n, 1).f());
        let x = ContigX::<f64, true> { x: array.view() };
        let pf = [1.0];
        let params = Params {
            family: Family::Binomial,
            alpha: 1.0,
            lambdas: None,
            nlambda: 2,
            eps: 1e-7,
            lambda_min: 0.05,
            lambda_log_scale: true,
            penalty_factor: &pf,
            dfmax: 2,
            max_iter: 100,
            stop_on_saturation: true,
            r_compatible: compatible,
            ybar: 0.5,
            null_deviance: 1.0,
            on_lambda: None,
        };
        for masked in [false, true] {
            let fit = Fit { x: &x, y: &y, params: &params, mask: masked.then_some(&mask), rows: None, n_fit: n };
            for threads in [1, 2, 6] {
                let mut st = State {
                    eta: Rows::new(n, 0.0),
                    irls_work: Rows::new(n, f64::NAN),
                    w: Rows::new(n, f64::NAN),
                    s: Rows::new(n, f64::NAN),
                    r: Rows::new(n, f64::NAN),
                    beta: vec![],
                    active: vec![],
                    strong: vec![],
                    z: vec![],
                    intercept: 0.0,
                    sum_w_residuals: 0.0,
                };
                st.eta.copy_from_slice(&eta);
                if masked {
                    for i in (0..n).filter(|&i| mask[i] == 0.0) {
                        st.eta[i] = f64::NAN;
                    }
                }
                let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
                let loss = pool.install(|| fit.update_irls(&mut st));
                let mut expected_loss = 0.0;
                for i in 0..n {
                    let (w, s, r) = if masked && mask[i] == 0.0 {
                        (0.0, 0.0, 0.0)
                    } else {
                        // The clipped Bernoulli row equations, independently
                        // of the driver and its serial/parallel dispatch.
                        let p = if eta[i] > cutoff {
                            1.0
                        } else if eta[i] < -cutoff {
                            0.0
                        } else {
                            let e = libm::exp(eta[i]);
                            e / (1.0 + e)
                        };
                        let w = if eta[i].abs() > cutoff { floor } else { p * (1.0 - p) };
                        let s = y[i] - p;
                        let term = if y[i] == 1.0 { libm::log(p) } else { libm::log(1.0 - p) };
                        expected_loss += term;
                        (w, s, s / w)
                    };
                    assert!(w.is_finite() && r.is_finite());
                    assert!(w > 0.0 || masked && mask[i] == 0.0);
                    for (actual, expected) in [(st.w[i], w), (st.s[i], s), (st.r[i], r)] {
                        assert_eq!(
                            actual.to_bits(),
                            expected.to_bits(),
                            "n={n}, row={i}, threads={threads}, masked={masked}"
                        );
                    }
                }
                assert_eq!(loss.to_bits(), (-expected_loss).to_bits());
            }
        }
    }
}

#[test]
fn null_model_counts_only_selected_rows() {
    let y = [0.0, 1.0, f64::NAN, 0.0];
    let rows = [1, 3, 1, 0];
    let (mean, loss) = null_model(&y, Some(&rows)).unwrap();
    assert_eq!(mean, 0.5);
    // Four balanced Bernoulli observations: negative log likelihood = 4 ln(2).
    assert_eq!(loss, 4.0 * std::f64::consts::LN_2);
}

#[test]
fn null_model_rejects_invalid_selected_responses() {
    assert!(null_model(&[], None).is_err());
    assert!(null_model(&[0.0, 0.0], None).is_err());
    assert!(null_model(&[1.0, f64::NAN], None).is_err());
}

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

/// Each test below runs bounds-checked reads of X and unchecked ones.
macro_rules! checked_and_not {
    ($body:ident) => {
        $body::<true>("checked");
        $body::<false>("unchecked");
    };
}

#[test]
fn passes_contiguous() {
    fn run<const CHECKED: bool>(how: &str) {
        for n in SIZES {
            let (x, y) = (values(n, 1), values(n, 2));
            check_passes(&format!("contiguous, {how}"), &Contig::<_, CHECKED>(&x[..]), &Contig(&y[..]), &x, &y);
            let x32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
            let y32: Vec<f32> = y.iter().map(|&v| v as f32).collect();
            let (xf, yf): (Vec<f64>, Vec<f64>) =
                (x32.iter().map(|&v| v as f64).collect(), y32.iter().map(|&v| v as f64).collect());
            check_passes(
                &format!("contiguous f32, {how}"),
                &Contig::<_, CHECKED>(&x32[..]),
                &Contig(&y32[..]),
                &xf,
                &yf,
            );
        }
    }
    checked_and_not!(run);
}

#[test]
fn packed_bits_preserve_pass_order_at_every_offset_and_tail() {
    for n in SIZES {
        for offset in 0..8 {
            let x: Vec<f64> = (0..n).map(|i| ((i * 17 + 3) % 7 < 3) as u8 as f64).collect();
            let mut bytes = vec![0u8; (n + offset).div_ceil(8)];
            for i in 0..n {
                bytes[(offset + i) / 8] |= (x[i] as u8) << ((offset + i) % 8);
            }
            let col = MappedCol::Bits { bytes: &bytes, offset, len: n };
            check_passes("packed bits", &col, &col, &x, &x);
        }
    }
}

#[test]
fn independently_chunked_columns_preserve_kernel_blocks() {
    for n in SIZES {
        let x: Vec<f64> = (0..n).map(|i| ((i * 17 + 3) % 7 < 3) as u8 as f64).collect();
        for size in [1, 7, 8, 9, 2047, 2048, 2049] {
            let mut primitive = vec![MappedCol::F64(&[])];
            primitive.extend(x.chunks(size).map(MappedCol::F64));
            primitive.push(MappedCol::F64(&[]));
            let storage = ChunkedColumn::new(primitive, n).unwrap();
            let col = MappedX { columns: vec![storage] };
            check_passes("chunked primitive", &col.col(0), &col.col(0), &x, &x);

            let bytes: Vec<Vec<u8>> = x
                .chunks(size)
                .enumerate()
                .map(|(k, chunk)| {
                    let offset = k % 8;
                    let mut bitmap = vec![0u8; (offset + chunk.len()).div_ceil(8)];
                    for (i, &v) in chunk.iter().enumerate() {
                        bitmap[(offset + i) / 8] |= (v as u8) << ((offset + i) % 8);
                    }
                    bitmap
                })
                .collect();
            let mut packed = vec![MappedCol::Bits { bytes: &[], offset: 0, len: 0 }];
            packed.extend(bytes.iter().zip(x.chunks(size)).enumerate().map(|(k, (bytes, chunk))| MappedCol::Bits {
                bytes,
                offset: k % 8,
                len: chunk.len(),
            }));
            let storage = ChunkedColumn::new(packed, n).unwrap();
            let col = MappedX { columns: vec![storage] };
            check_passes("chunked packed", &col.col(0), &col.col(0), &x, &x);
        }
    }
}

#[test]
fn passes_strided() {
    fn run<const CHECKED: bool>(how: &str) {
        for n in SIZES {
            // C order: each column's elements three apart
            let x = Array2::from_shape_vec((n, 3), values(3 * n, 6)).expect("shape");
            let (cx, cy) = (Strided::<_, CHECKED>(x.column(0)), Strided(x.column(2)));
            check_passes(&format!("strided, {how}"), &cx, &cy, &x.column(0).to_vec(), &x.column(2).to_vec());
        }
    }
    checked_and_not!(run);
}

#[test]
fn passes_contiguous_int8_with_exact_signed_conversion() {
    fn run<const CHECKED: bool>(how: &str) {
        for n in SIZES {
            // Every int8 value, including -128 and 127; conversion to f64 is
            // exact. The reference sums the independently expanded scalars
            // in the specified LANES/CHUNK order, with no fitted oracle.
            let x: Vec<i8> = (0..n).map(|i| (i % 256) as i16 - 128).map(|v| v as i8).collect();
            let y: Vec<i8> = (0..n).map(|i| ((i * 71 + 53) % 256) as i16 - 128).map(|v| v as i8).collect();
            let xf: Vec<f64> = x.iter().map(|&v| v as f64).collect();
            let yf: Vec<f64> = y.iter().map(|&v| v as f64).collect();
            check_passes(
                &format!("contiguous int8, {how}"),
                &Contig::<_, CHECKED>(&x),
                &Contig::<_, CHECKED>(&y),
                &xf,
                &yf,
            );
        }
    }
    checked_and_not!(run);
}

#[test]
fn passes_gathered() {
    fn run<const CHECKED: bool>(how: &str) {
        for n in SIZES {
            // a row subset in shuffled order, with repeats, from a taller X
            let nrow = n + 41;
            let idx: Vec<usize> = values(n, 7).iter().map(|v| (v.abs() * 1e9) as usize % nrow).collect();
            let x = Array2::from_shape_vec((nrow, 2).f(), values(2 * nrow, 8)).expect("shape");
            let (x0, x1) = (x.column(0).to_slice().expect("Fortran"), x.column(1).to_slice().expect("Fortran"));
            let (cx, cy) = (
                Gather { col: Contig::<_, CHECKED>(x0), idx: &idx },
                Gather { col: Contig::<_, CHECKED>(x1), idx: &idx },
            );
            let (gx, gy): (Vec<f64>, Vec<f64>) =
                (idx.iter().map(|&i| x0[i]).collect(), idx.iter().map(|&i| x1[i]).collect());
            check_passes(&format!("gathered, {how}"), &cx, &cy, &gx, &gy);
        }
    }
    checked_and_not!(run);
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
        assert_eq!(dot_col(&Contig::<_, true>(&x[..]), &s).to_bits(), want.to_bits(), "n={n}");
        assert_eq!(dot_col(&Contig::<_, false>(&x[..]), &s).to_bits(), want.to_bits(), "n={n}, unchecked");
    }
}

#[test]
fn ordered_sum_is_order_sensitive() {
    // the check above can fail: these values sum differently in plain order
    let t = values(3 * CHUNK + 11, 11);
    assert_ne!(ordered_sum(&t).to_bits(), seqsum(&t).to_bits());
}

#[test]
#[should_panic(expected = "out of bounds")]
fn checked_contiguous_read_checks() {
    let x = values(5, 12);
    Contig::<_, true>(&x[..]).at(5);
}

#[test]
#[should_panic(expected = "out of bounds")]
fn checked_strided_read_checks() {
    let x = Array2::from_shape_vec((5, 3), values(15, 13)).expect("shape");
    Strided::<_, true>(x.column(1)).at(5);
}
