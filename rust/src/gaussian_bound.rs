//! Score-change enclosures for inactive Gaussian columns.
//! See docs/gaussian-screening.md for the arithmetic assumptions and proof.
use super::*;
const U: f64 = 1.1102230246251565e-16;
const ETA: f64 = f64::from_bits(1);
fn checked(v: f64) -> Option<f64> {
    (v.is_finite() && v >= 0.0).then_some(v)
}
fn up(v: f64) -> Option<f64> {
    checked(v.next_up())
}
fn add(a: f64, b: f64) -> Option<f64> {
    up(a + b)
}
fn mul(a: f64, b: f64) -> Option<f64> {
    up(a * b)
}
fn div(a: f64, b: f64) -> Option<f64> {
    if b <= 0. || !b.is_finite() { None } else { up(a / b) }
}

pub(crate) fn arithmetic_ok() -> bool {
    let x = std::hint::black_box(1.0);
    let half = std::hint::black_box(U);
    let three_quarters = std::hint::black_box(1.5 * U);
    let tiny = std::hint::black_box(ETA);
    let min = std::hint::black_box(f64::MIN_POSITIVE);
    // Distinguish round-to-nearest/even from all three directed modes;
    // check both denormal inputs (DAZ) and outputs (FTZ) without unsafe code.
    (x + half).to_bits() == 1.0f64.to_bits()
        && (x + three_quarters).to_bits() == 1.0f64.next_up().to_bits()
        && (tiny * x).to_bits() == 1
        && (min * std::hint::black_box(0.5)).to_bits() == (1u64 << 51)
        // Include an inexact subnormal output, not only an exactly representable
        // product: flush policies can distinguish the two.
        && (min * std::hint::black_box(0.1)).to_bits() == 0x0001_9999_9999_999a
}

/// A norm's input, read in ascending row order, once per pass the norm makes.
trait Pass {
    fn len(&self) -> usize;
    fn each(&self, visit: impl FnMut(f64));
}
/// Values computed from an index: a slice, or a difference of two.
struct Indexed<F>(usize, F);
impl<F: Fn(usize) -> f64> Pass for Indexed<F> {
    fn len(&self) -> usize {
        self.0
    }
    fn each(&self, mut visit: impl FnMut(f64)) {
        (0..self.0).for_each(|i| visit((self.1)(i)));
    }
}
/// A transform of each value in a column of X. The column is read in runs,
/// as the kernels read it: the same values in the same order as at(i), but
/// a chunked column isn't searched for every row.
struct Mapped<'a, C, F>(&'a C, F);
impl<C: Col, F: Fn(f64) -> f64> Pass for Mapped<'_, C, F> {
    fn len(&self) -> usize {
        self.0.n()
    }
    fn each(&self, mut visit: impl FnMut(f64)) {
        for_each_value(self.0, |v| visit((self.1)(v)));
    }
}

// Guard the actual norm input, after any centering/differencing transform.
// Its unscaled square is normal; the count bounds every partial sum.
const FAST_MIN: f64 = f64::from_bits((1023 - 240) << 52);
const FAST_MAX: f64 = f64::from_bits((1023 + 240) << 52);

fn norm_fast(values: &impl Pass) -> Option<f64> {
    let (gamma, _) = gamma_terms(values.len().checked_mul(2)?.checked_add(64)?)?;
    let mut valid = true;
    let mut squared = 0.;
    values.each(|value| {
        let value = value.abs();
        let in_range = value == 0. || (FAST_MIN..=FAST_MAX).contains(&value);
        valid &= in_range;
        // Invalid inputs contribute a dummy zero; no overflowing or tiny
        // candidate products are evaluated. The entire candidate is discarded.
        let guarded = if in_range { value } else { 0. };
        squared += guarded * guarded;
    });
    if !valid {
        return None;
    }
    if squared == 0. {
        return Some(0.);
    }
    // |T-Q| <= gamma*Q, so Q <= T/(1-gamma). Lower the
    // denominator, then round division and sqrt outward.
    let denominator = (1. - gamma).next_down();
    up(div(squared, denominator)?.sqrt())
}

pub(crate) fn norm(n: usize, at: impl Fn(usize) -> f64) -> Option<f64> {
    norm_of(&Indexed(n, at))
}

fn norm_of(values: &impl Pass) -> Option<f64> {
    if !arithmetic_ok() {
        return None;
    }
    norm_fast(values).or_else(|| norm_safe(values))
}

fn norm_safe(values: &impl Pass) -> Option<f64> {
    if !arithmetic_ok() {
        return None;
    }
    let (mut finite, mut maximum) = (true, 0f64);
    values.each(|v| {
        let v = v.abs();
        finite &= v.is_finite();
        maximum = maximum.max(v);
    });
    if !finite {
        return None;
    }
    if maximum == 0. {
        return Some(0.);
    }
    // None once any step fails, as returning from a loop would be.
    let mut squared = Some(0.);
    values.each(|v| {
        squared = squared.and_then(|s| {
            let ratio = div(v.abs(), maximum)?;
            add(s, mul(ratio, ratio)?)
        });
    });
    let squared = squared?;
    // The additive enclosure is in the squared, scaled domain, before sqrt.
    // next_up of a flushed-to-zero operation is never used: arithmetic_ok
    // requires gradual underflow. Genuine rounded zero widens to ETA.
    mul(maximum, up(squared.sqrt())?)
}

fn gamma_terms(operations: usize) -> Option<(f64, f64)> {
    if operations as u128 > (1u128 << 53) {
        return None;
    }
    let k = operations as f64;
    let ku = mul(k, U)?;
    if ku >= 0.25 {
        return None;
    }
    let denominator = (1. - ku).next_down();
    Some((div(ku, denominator)?, div(mul(k, ETA)?, denominator)?))
}
fn summation_error(magnitude: f64, operations: usize) -> Option<f64> {
    let (g, a) = gamma_terms(operations)?;
    add(mul(g, magnitude)?, a)
}
fn safe(v: f64) -> Option<f64> {
    (v.is_finite() && (0. ..f64::MAX / 16.).contains(&v)).then_some(v)
}

#[derive(Clone, Copy)]
pub(crate) struct Meta {
    xnorm: f64,
    coefficient_norm: f64,
    center: f64,
    denominator: f64,
    n: usize,
    centered: bool,
}
impl Meta {
    pub(crate) fn make<C: Col>(
        column: &C,
        center: f64,
        scale: f64,
        n_fit: usize,
        _fits: impl Fn(usize) -> bool,
    ) -> Option<Self> {
        if !center.is_finite() || !scale.is_finite() || scale <= 0. || n_fit == 0 || n_fit as u128 > (1u128 << 53) {
            return None;
        }
        let n = column.n();
        let centered = center.abs() > 100. * scale;
        let denominator = if centered { n_fit as f64 } else { scale * n_fit as f64 };
        if !denominator.is_finite() || denominator <= 0. {
            return None;
        }
        let (xnorm, coefficient_norm) = if centered {
            // These rounded coefficients are exactly those used by stock's
            // centered fallback. Their construction is outside the linear dot.
            let zn = norm_of(&Mapped(column, |value| (value - center) / scale))?;
            (zn, div(zn, denominator)?)
        } else {
            let xn = norm_of(&Mapped(column, |value| value))?;
            let difference = |value: f64| if value == center { 0. } else { (value - center).abs().next_up() };
            (xn, div(norm_of(&Mapped(column, difference))?, denominator)?)
        };
        Some(Self { xnorm, coefficient_norm, center: center.abs(), denominator, n, centered })
    }
    #[cfg(test)]
    pub(super) fn test_metadata(self) -> [u64; 7] {
        [
            1,
            self.xnorm.to_bits(),
            self.coefficient_norm.to_bits(),
            self.center.to_bits(),
            self.denominator.to_bits(),
            self.n as u64,
            u64::from(self.centered),
        ]
    }
    pub(crate) fn error(&self, rnorm: f64) -> Option<f64> {
        if !arithmetic_ok() {
            return None;
        }
        let mx = safe(mul(self.xnorm, rnorm)?)?;
        let ed = summation_error(mx, self.n.checked_mul(2)?.checked_add(64)?)?;
        let (magnitude, et) = if self.centered {
            (mx, ed)
        } else {
            let mr = safe(mul(up((self.n as f64).sqrt())?, rnorm)?)?;
            let es = summation_error(mr, self.n)?;
            let csum = safe(mul(self.center, mr)?)?;
            let csum_round = safe(mul(self.center, add(mr, es)?)?)?;
            let ev = add(mul(self.center, es)?, add(mul(U, csum_round)?, ETA)?)?;
            let magnitude = safe(add(mx, csum)?)?;
            let subtract_magnitude = safe(add(add(mx, ed)?, add(csum, ev)?)?)?;
            let et = add(add(ed, ev)?, add(mul(U, subtract_magnitude)?, ETA)?)?;
            (magnitude, et)
        };
        let quotient_magnitude = safe(div(add(magnitude, et)?, self.denominator)?)?;
        add(div(et, self.denominator)?, add(mul(U, quotient_magnitude)?, ETA)?)
    }
    pub(crate) fn radius(&self, distance: f64, anchor_norm: f64, current_norm: f64) -> Option<f64> {
        add(mul(self.coefficient_norm, distance)?, add(self.error(anchor_norm)?, self.error(current_norm)?)?)
    }
}

pub(crate) fn difference_norm(a: &[f64], b: &[f64]) -> Option<f64> {
    assert_eq!(a.len(), b.len());
    norm(a.len(), |i| if a[i] == b[i] { 0. } else { (a[i] - b[i]).abs().next_up() })
}
pub(crate) fn decide(score: f64, radius: f64, cutoff: f64) -> Option<bool> {
    if !score.is_finite() || !radius.is_finite() || radius < 0. || !cutoff.is_finite() {
        return None;
    }
    if radius == 0. {
        return Some(score.abs() > cutoff);
    }
    let upper = add(score.abs(), radius)?;
    let lower = (score.abs() - radius).next_down().max(0.);
    if upper <= cutoff {
        Some(false)
    } else if lower > cutoff {
        Some(true)
    } else {
        None
    }
}
// Outward cumulative prefix endpoints. The lower endpoint encloses the SUM
// OF UPPER INCREMENTS, not the sum of the exact residual distances.
pub(crate) fn advance(hi: f64, lo: f64, step: f64) -> Option<(f64, f64)> {
    checked(hi)?;
    checked(lo)?;
    checked(step)?;
    if lo > hi {
        return None;
    }
    if step == 0. { Some((hi, lo)) } else { Some((add(hi, step)?, (lo + step).next_down().max(0.))) }
}
pub(crate) fn distance(hi: f64, anchor_lo: f64) -> Option<f64> {
    checked(hi)?;
    checked(anchor_lo)?;
    if hi < anchor_lo {
        return None;
    }
    up(hi - anchor_lo)
}

#[cfg(test)]
pub(crate) fn test_norm(values: &[f64]) -> Option<f64> {
    norm(values.len(), |i| values[i])
}
#[cfg(test)]
pub(crate) fn test_norm_paths(values: &[f64]) -> (Option<f64>, Option<f64>, Option<f64>) {
    if !arithmetic_ok() {
        return (None, None, None);
    }
    (
        norm_fast(&Indexed(values.len(), |i| values[i])),
        norm_safe(&Indexed(values.len(), |i| values[i])),
        test_norm(values),
    )
}
/// The scores use the residual sum the solver keeps: `seqsum` order with
/// `r_compatible`, the default mode's blocked order otherwise.
#[cfg(test)]
pub(crate) fn test_bound(
    x: &[f64],
    a: &[f64],
    b: &[f64],
    center: f64,
    scale: f64,
    n_fit: usize,
    r_compatible: bool,
) -> Option<Vec<f64>> {
    if x.len() != a.len() || x.len() != b.len() {
        return None;
    }
    let column = Contig::<_, true>(x);
    let m = Meta::make(&column, center, scale, n_fit, |_| true)?;
    let ra = norm(a.len(), |i| a[i])?;
    let rb = norm(b.len(), |i| b[i])?;
    let d = difference_norm(a, b)?;
    let sum = if r_compatible { seqsum } else { super::gaussian::blocked_sum };
    Some(vec![
        standardized_score(&column, a, sum(a), center, scale, n_fit as f64),
        standardized_score(&column, b, sum(b), center, scale, n_fit as f64),
        m.radius(d, ra, rb)?,
        ra,
        rb,
        d,
        m.coefficient_norm,
        m.error(ra)?,
        m.error(rb)?,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_guard_includes_endpoints_and_excludes_neighbors() {
        for value in [FAST_MIN, FAST_MIN.next_up(), FAST_MAX.next_down(), FAST_MAX] {
            for sign in [-1., 1.] {
                let values = [0., sign * value, -0.];
                assert!(norm_fast(&Indexed(values.len(), |i| values[i])).unwrap() >= value);
            }
        }
        for value in [FAST_MIN.next_down(), FAST_MAX.next_up(), f64::NAN, f64::INFINITY] {
            assert!(norm_fast(&Indexed(3, |i| if i == 1 { value } else { 1. })).is_none());
        }
    }

    #[test]
    fn guarded_norm_keeps_safe_underflow_fallback() {
        assert!(norm_fast(&Indexed(1, |_| 1e-200)).is_none());
        assert!(norm(1, |_| 1e-200).unwrap() >= 1e-200);
        assert_eq!(norm(0, |_| unreachable!()), Some(0.));
        assert_eq!(norm(3, |_| -0.), Some(0.));
    }

    #[test]
    fn operation_count_rejects_before_reading() {
        assert!(norm_fast(&Indexed(usize::MAX, |_| unreachable!())).is_none());
    }

    #[test]
    fn guard_checks_the_transformed_quantity() {
        let raw = [FAST_MAX, -FAST_MAX];
        assert!(norm_fast(&Indexed(raw.len(), |i| raw[i])).is_some());
        assert!(
            norm_fast(&Indexed(raw.len(), |i: usize| {
                if raw[i] == FAST_MAX { 0. } else { (raw[i] - FAST_MAX).abs().next_up() }
            }))
            .is_none()
        );

        let center = 2. * FAST_MAX;
        let raw = [center, center.next_up()];
        let scale = raw[1] - center;
        assert!(norm_fast(&Indexed(raw.len(), |i| raw[i])).is_none());
        assert!(norm_fast(&Indexed(raw.len(), |i| (raw[i] - center) / scale)).unwrap() >= 1.);
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn gamma_guard_near_its_largest_accepted_count() {
        let last = (1usize << 51) - 2;
        let (upper, _) = gamma_terms(last).unwrap();
        assert!(upper < 0.5);
        assert!(gamma_terms(1usize << 51).is_none());
        assert!(norm_fast(&Indexed((1usize << 50) - 32, |_| unreachable!())).is_none());
    }
    /// Meta's bits from each storage, against those from the same values
    /// in one contiguous f64 slice, which reads them one by one in order.
    /// Both go through the run reader, so its visit order is checked first,
    /// row by row against at(i), the scalar access it replaced.
    fn assert_meta_matches_values<C: Col>(what: &str, column: &C, center: f64, scale: f64) {
        let values: Vec<f64> = (0..column.n()).map(|i| column.at(i)).collect();
        let mut row = 0;
        Mapped(column, |value| value).each(|value| {
            assert_eq!(value.to_bits(), values[row].to_bits(), "{what}: row {row}");
            row += 1;
        });
        assert_eq!(row, values.len(), "{what}: rows visited");
        let expected = Meta::make(&Contig::<_, true>(&values), center, scale, values.len(), |_| true);
        let actual = Meta::make(column, center, scale, values.len(), |_| true);
        assert_eq!(actual.map(Meta::test_metadata), expected.map(Meta::test_metadata), "{what}");
    }

    #[test]
    fn meta_reads_every_storage_as_its_values() {
        let n = 61;
        let small: Vec<i16> = (0..n).map(|i| ((i * 7919) % 101) as i16 - 50).collect();
        let as_f64: Vec<f64> = small.iter().map(|&v| f64::from(v)).collect();
        // A value below the fast guard sends the norm to its safe fallback;
        // values far from zero relative to their spread take the centered branch.
        let mut tiny = as_f64.clone();
        tiny[30] = 1e-300;
        let offset: Vec<f64> = as_f64.iter().map(|v| 1e6 + v / 64.).collect();
        let mut nonfinite = as_f64.clone();
        nonfinite[40] = f64::NAN;
        let bools: Vec<u8> = small.iter().map(|&v| u8::from(v > 0)).collect();
        let cases: [(&str, &[f64], f64, f64); 5] = [
            ("ordinary", &as_f64, 1.5, 29.),
            ("exact center", &as_f64, 3., 29.),
            ("safe fallback", &tiny, 1.5, 29.),
            ("centered", &offset, 1e6, 0.45),
            ("nonfinite", &nonfinite, 1.5, 29.),
        ];
        for (name, x, center, scale) in cases {
            assert_meta_matches_values(name, &Contig::<_, true>(x), center, scale);
            let wide = ndarray::Array2::from_shape_fn((x.len(), 3), |(i, j)| if j == 1 { x[i] } else { 0. });
            assert_meta_matches_values(name, &Strided::<_, true>(wide.column(1)), center, scale);
            let rows: Vec<usize> = (0..x.len()).rev().step_by(2).collect();
            assert_meta_matches_values(name, &Gather { col: Contig::<_, true>(x), idx: &rows }, center, scale);
            for size in [1, 7, 8, 9, 20] {
                let mut chunks = vec![MappedCol::F64(&[])];
                chunks.extend(x.chunks(size).map(MappedCol::F64));
                chunks.push(MappedCol::F64(&[]));
                let mapped = MappedX { columns: vec![ChunkedColumn::new(chunks, x.len()).unwrap()] };
                assert_meta_matches_values(name, &mapped.col(0), center, scale);
            }
        }
        let narrow: Vec<i8> = small.iter().map(|&v| v as i8).collect();
        let single: Vec<f32> = as_f64.iter().map(|&v| v as f32).collect();
        let (center, scale) = (1.5, 29.);
        assert_meta_matches_values("i16", &MappedCol::I16(&small), center, scale);
        assert_meta_matches_values("i8", &MappedCol::I8(&narrow), center, scale);
        assert_meta_matches_values("f32", &MappedCol::F32(&single), center, scale);
        for size in [3, 8, 13] {
            let chunks: Vec<_> = narrow.chunks(size).map(MappedCol::I8).collect();
            let mapped = MappedX { columns: vec![ChunkedColumn::new(chunks, n).unwrap()] };
            assert_meta_matches_values("i8 chunks", &mapped.col(0), center, scale);
        }
        // 64-bit integers at and around 2^53, where conversion to f64 rounds.
        let edge = 1i64 << 53;
        let wide: Vec<i64> = (0..n as i64).map(|i| (edge + i % 5 - 2) * if i % 3 == 0 { -1 } else { 1 }).collect();
        let unsigned: Vec<u64> = (0..n as u64).map(|i| (edge as u64) + i % 5 - 2).collect();
        for size in [1, 7, 8, 13] {
            let chunks: Vec<_> = wide.chunks(size).map(MappedCol::I64).collect();
            let mapped = MappedX { columns: vec![ChunkedColumn::new(chunks, n).unwrap()] };
            assert_meta_matches_values("i64 chunks", &mapped.col(0), 2f64.powi(52), 2f64.powi(53));
            let chunks: Vec<_> = unsigned.chunks(size).map(MappedCol::U64).collect();
            let mapped = MappedX { columns: vec![ChunkedColumn::new(chunks, n).unwrap()] };
            assert_meta_matches_values("u64 chunks", &mapped.col(0), 2f64.powi(53), 2.);
        }
        for bit_offset in [0, 3] {
            let mut bytes = vec![0u8; (bit_offset + n).div_ceil(8)];
            for (i, &b) in bools.iter().enumerate() {
                bytes[(bit_offset + i) / 8] |= b << ((bit_offset + i) % 8);
            }
            let bits = MappedCol::Bits { bytes: &bytes, offset: bit_offset, len: n };
            assert_meta_matches_values("bits", &bits, 0.5, 0.5);
        }
    }
}

#[cfg(all(test, feature = "python"))]
mod rational_tests {
    use super::{test_bound, test_norm, test_norm_paths};
    use pyo3::{prelude::*, types::PyModule};
    #[pyfunction]
    fn norm(values: Vec<f64>) -> Option<f64> {
        test_norm(&values)
    }
    #[pyfunction]
    fn norm_paths(values: Vec<f64>) -> (Option<f64>, Option<f64>, Option<f64>) {
        test_norm_paths(&values)
    }
    #[pyfunction]
    fn bound(x: Vec<f64>, a: Vec<f64>, b: Vec<f64>, center: f64, scale: f64, n_fit: usize) -> Option<Vec<f64>> {
        test_bound(&x, &a, &b, center, scale, n_fit, false)
    }
    #[pyfunction(name = "bound")]
    fn bound_compatible(
        x: Vec<f64>,
        a: Vec<f64>,
        b: Vec<f64>,
        center: f64,
        scale: f64,
        n_fit: usize,
    ) -> Option<Vec<f64>> {
        test_bound(&x, &a, &b, center, scale, n_fit, true)
    }
    #[pyfunction]
    fn decide(score: f64, radius: f64, cutoff: f64) -> Option<bool> {
        super::decide(score, radius, cutoff)
    }

    #[test]
    fn retained_fraction_audits_of_actual_native_operations() {
        // Embedded Python supplies arbitrary precision Fraction expectations;
        // these wrappers exist only in this unit-test executable, not the wheel.
        Python::initialize();
        Python::attach(|py| {
            // Each audit runs once per residual-sum order the solver keeps.
            let native = PyModule::new(py, "bound_audit").unwrap();
            let compatible = PyModule::new(py, "bound_audit_compatible").unwrap();
            for module in [&native, &compatible] {
                module.add_function(wrap_pyfunction!(norm, module).unwrap()).unwrap();
                module.add_function(wrap_pyfunction!(norm_paths, module).unwrap()).unwrap();
                module.add_function(wrap_pyfunction!(decide, module).unwrap()).unwrap();
            }
            native.add_function(wrap_pyfunction!(bound, &native).unwrap()).unwrap();
            compatible.add_function(wrap_pyfunction!(bound_compatible, &compatible).unwrap()).unwrap();
            for (name, code) in [
                ("guarded_norm", include_str!("../../tests/support/gaussian_screening/guarded_norm.py")),
                ("bound", include_str!("../../tests/support/gaussian_screening/bound.py")),
                ("raw_centers", include_str!("../../tests/support/gaussian_screening/raw_centers.py")),
                ("long_bound", include_str!("../../tests/support/gaussian_screening/long_bound.py")),
            ] {
                let code = std::ffi::CString::new(code).unwrap();
                let name = std::ffi::CString::new(name).unwrap();
                let module = PyModule::from_code(py, &code, &name, &name).unwrap();
                for target in [&native, &compatible] {
                    module.getattr("audit").unwrap().call1((target,)).unwrap();
                }
            }
        });
    }

    #[test]
    fn audit_modules_score_with_different_sum_orders() {
        // The blocked order cancels ±1e16 within one lane before adding the
        // ones; the serial order absorbs seven of the ones into 1e16 first.
        let n = 17;
        let mut a = vec![1.; n];
        a[0] = 1e16;
        a[8] = -1e16;
        assert_ne!(super::super::gaussian::blocked_sum(&a), super::seqsum(&a));
        let x: Vec<f64> = (0..n).map(|i| 37. + (i % 5) as f64 / 8.).collect();
        let default = bound(x.clone(), a.clone(), a.clone(), 37., 0.5, n).unwrap();
        let compatible = bound_compatible(x, a.clone(), a, 37., 0.5, n).unwrap();
        assert_ne!(default[0].to_bits(), compatible[0].to_bits());
    }
}
