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

// Guard the actual norm input, after any centering/differencing transform.
// Its unscaled square is normal; the count bounds every partial sum.
const FAST_MIN: f64 = f64::from_bits((1023 - 240) << 52);
const FAST_MAX: f64 = f64::from_bits((1023 + 240) << 52);

fn norm_fast(n: usize, at: impl Fn(usize) -> f64) -> Option<f64> {
    let (gamma, _) = gamma_terms(n.checked_mul(2)?.checked_add(64)?)?;
    let mut valid = true;
    let mut squared = 0.;
    for i in 0..n {
        let value = at(i).abs();
        let in_range = value == 0. || (FAST_MIN..=FAST_MAX).contains(&value);
        valid &= in_range;
        // Invalid inputs contribute a dummy zero; no overflowing or tiny
        // candidate products are evaluated. The entire candidate is discarded.
        let guarded = if in_range { value } else { 0. };
        squared += guarded * guarded;
    }
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
    if !arithmetic_ok() {
        return None;
    }
    norm_fast(n, &at).or_else(|| norm_safe(n, at))
}

fn norm_safe(n: usize, at: impl Fn(usize) -> f64) -> Option<f64> {
    if !arithmetic_ok() {
        return None;
    }
    let mut maximum: f64 = 0.;
    for i in 0..n {
        let v = at(i).abs();
        if !v.is_finite() {
            return None;
        }
        maximum = maximum.max(v);
    }
    if maximum == 0. {
        return Some(0.);
    }
    let mut squared = 0.;
    for i in 0..n {
        let ratio = div(at(i).abs(), maximum)?;
        squared = add(squared, mul(ratio, ratio)?)?;
    }
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
        let at = |i| column.at(i);
        let (xnorm, coefficient_norm) = if centered {
            // These rounded coefficients are exactly those used by stock's
            // centered fallback. Their construction is outside the linear dot.
            let z = |i| (column.at(i) - center) / scale;
            let zn = norm(n, z)?;
            (zn, div(zn, denominator)?)
        } else {
            let xn = norm(n, at)?;
            let difference = |i| {
                let value = column.at(i);
                if value == center { 0. } else { (value - center).abs().next_up() }
            };
            (xn, div(norm(n, difference)?, denominator)?)
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
    (norm_fast(values.len(), |i| values[i]), norm_safe(values.len(), |i| values[i]), test_norm(values))
}
#[cfg(test)]
pub(crate) fn test_bound(x: &[f64], a: &[f64], b: &[f64], center: f64, scale: f64, n_fit: usize) -> Option<Vec<f64>> {
    if x.len() != a.len() || x.len() != b.len() {
        return None;
    }
    let column = Contig::<_, true>(x);
    let m = Meta::make(&column, center, scale, n_fit, |_| true)?;
    let ra = norm(a.len(), |i| a[i])?;
    let rb = norm(b.len(), |i| b[i])?;
    let d = difference_norm(a, b)?;
    Some(vec![
        standardized_score(&column, a, seqsum(a), center, scale, n_fit as f64),
        standardized_score(&column, b, seqsum(b), center, scale, n_fit as f64),
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
                assert!(norm_fast(values.len(), |i| values[i]).unwrap() >= value);
            }
        }
        for value in [FAST_MIN.next_down(), FAST_MAX.next_up(), f64::NAN, f64::INFINITY] {
            assert!(norm_fast(3, |i| if i == 1 { value } else { 1. }).is_none());
        }
    }

    #[test]
    fn guarded_norm_keeps_safe_underflow_fallback() {
        assert!(norm_fast(1, |_| 1e-200).is_none());
        assert!(norm(1, |_| 1e-200).unwrap() >= 1e-200);
        assert_eq!(norm(0, |_| unreachable!()), Some(0.));
        assert_eq!(norm(3, |_| -0.), Some(0.));
    }

    #[test]
    fn operation_count_rejects_before_reading() {
        assert!(norm_fast(usize::MAX, |_| unreachable!()).is_none());
    }

    #[test]
    fn guard_checks_the_transformed_quantity() {
        let raw = [FAST_MAX, -FAST_MAX];
        assert!(norm_fast(raw.len(), |i| raw[i]).is_some());
        assert!(
            norm_fast(raw.len(), |i| { if raw[i] == FAST_MAX { 0. } else { (raw[i] - FAST_MAX).abs().next_up() } })
                .is_none()
        );

        let center = 2. * FAST_MAX;
        let raw = [center, center.next_up()];
        let scale = raw[1] - center;
        assert!(norm_fast(raw.len(), |i| raw[i]).is_none());
        assert!(norm_fast(raw.len(), |i| (raw[i] - center) / scale).unwrap() >= 1.);
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn gamma_guard_near_its_largest_accepted_count() {
        let last = (1usize << 51) - 2;
        let (upper, _) = gamma_terms(last).unwrap();
        assert!(upper < 0.5);
        assert!(gamma_terms(1usize << 51).is_none());
        assert!(norm_fast((1usize << 50) - 32, |_| unreachable!()).is_none());
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
        test_bound(&x, &a, &b, center, scale, n_fit)
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
            let native = PyModule::new(py, "bound_audit").unwrap();
            native.add_function(wrap_pyfunction!(norm, &native).unwrap()).unwrap();
            native.add_function(wrap_pyfunction!(norm_paths, &native).unwrap()).unwrap();
            native.add_function(wrap_pyfunction!(bound, &native).unwrap()).unwrap();
            native.add_function(wrap_pyfunction!(decide, &native).unwrap()).unwrap();
            for (name, code) in [
                ("guarded_norm", include_str!("../../tests/support/gaussian_screening/guarded_norm.py")),
                ("bound", include_str!("../../tests/support/gaussian_screening/bound.py")),
                ("raw_centers", include_str!("../../tests/support/gaussian_screening/raw_centers.py")),
                ("long_bound", include_str!("../../tests/support/gaussian_screening/long_bound.py")),
            ] {
                let code = std::ffi::CString::new(code).unwrap();
                let name = std::ffi::CString::new(name).unwrap();
                let module = PyModule::from_code(py, &code, &name, &name).unwrap();
                module.getattr("audit").unwrap().call1((&native,)).unwrap();
            }
        });
    }
}
