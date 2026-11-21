//! The raw-scale intercept that pairs with the exported slopes, for both the
//! Python and browser bindings, so the two agree on every input.
//!
//! Integer accumulation in units of the smallest subnormal gives the exactly
//! rounded sum of the already-rounded products on any build, without relying
//! on JS sums, the platform libm, or Python's math.fsum, whose documentation
//! allows double rounding on some builds.

const WORDS: usize = 34;
const FRACTION: u64 = (1 << 52) - 1;

/// `centered_intercept - S`, where S is the exactly rounded sum of
/// `products`, the rounded products center * coef (and offset * coef for
/// anchored columns). None when that isn't a finite number: a nonfinite
/// product, a sum that overflows, or a final subtraction that overflows. The
/// raw pair then can't represent the model, though centered predictions can.
pub(crate) fn exported_intercept(centered_intercept: f64, products: &[f64]) -> Option<f64> {
    if !centered_intercept.is_finite() {
        return None;
    }
    let sum = rounded_sum(products).ok()?;
    let intercept = centered_intercept - sum;
    intercept.is_finite().then_some(intercept)
}

/// The sum of `values` rounded once to nearest, ties to even; an error for a
/// nonfinite value or a sum outside the finite range. Exact cancellation of
/// large terms, as in [MAX, MAX, -MAX], is handled exactly.
pub(crate) fn rounded_sum(values: &[f64]) -> Result<f64, &'static str> {
    let mut positive = [0u64; WORDS];
    let mut negative = [0u64; WORDS];
    for &value in values {
        if !value.is_finite() {
            return Err("The model's intercept correction is not finite.");
        }
        let bits = value.to_bits();
        let exponent = ((bits >> 52) & 0x7ff) as usize;
        let mantissa = (bits & FRACTION) | if exponent == 0 { 0 } else { 1 << 52 };
        let shift = exponent.saturating_sub(1);
        let words = if bits >> 63 == 0 { &mut positive } else { &mut negative };
        add(words, shift / 64, mantissa << (shift % 64));
        if !shift.is_multiple_of(64) {
            add(words, shift / 64 + 1, mantissa >> (64 - shift % 64));
        }
    }
    let sign = positive.iter().rev().cmp(negative.iter().rev()).is_lt();
    let (mut magnitude, small) = if sign { (negative, positive) } else { (positive, negative) };
    let mut borrow = false;
    for (a, b) in magnitude.iter_mut().zip(small) {
        let (x, first) = a.overflowing_sub(b);
        let (x, second) = x.overflowing_sub(u64::from(borrow));
        *a = x;
        borrow = first || second;
    }
    let Some(last) = magnitude.iter().rposition(|&word| word != 0) else {
        return Ok(0.0);
    };
    let mut highest = last * 64 + 63 - magnitude[last].leading_zeros() as usize;
    let sign = u64::from(sign) << 63;
    if highest < 52 {
        return Ok(f64::from_bits(sign | magnitude[0]));
    }
    let shift = highest - 52;
    let word = shift / 64;
    let offset = shift % 64;
    let mut leading = magnitude[word] >> offset;
    if offset != 0 {
        leading |= magnitude.get(word + 1).copied().unwrap_or(0) << (64 - offset);
    }
    if shift != 0 {
        let guard = shift - 1;
        let half = magnitude[guard / 64] & (1 << (guard % 64)) != 0;
        let sticky = magnitude[..guard / 64].iter().any(|&w| w != 0)
            || magnitude[guard / 64] & ((1u64 << (guard % 64)) - 1) != 0;
        if half && (sticky || leading & 1 != 0) {
            leading += 1;
            if leading == 1 << 53 {
                leading >>= 1;
                highest += 1;
            }
        }
    }
    let exponent = highest - 51;
    if exponent >= 0x7ff {
        return Err("The model's intercept correction overflows float64.");
    }
    Ok(f64::from_bits(sign | (exponent as u64) << 52 | (leading & FRACTION)))
}

fn add(words: &mut [u64; WORDS], mut index: usize, mut value: u64) {
    // 34 words cover every f64 exponent plus more carry bits than a slice's
    // addressable length can require, even on a 64-bit native build.
    while value != 0 {
        let (sum, carry) = words[index].overflowing_add(value);
        words[index] = sum;
        value = u64::from(carry);
        index += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancellation_and_sign() {
        let mut values = vec![1e16];
        values.extend([1.0; 1000]);
        values.push(-1e16);
        assert_eq!(rounded_sum(&values), Ok(1000.0));
        assert_eq!(rounded_sum(&values.iter().map(|v| -v).collect::<Vec<_>>()), Ok(-1000.0));
        assert_eq!(rounded_sum(&[1.0, -1.0]), Ok(0.0));
    }

    #[test]
    fn ties_round_to_even_and_subnormals_are_exact() {
        let half = 2.0f64.powi(-53);
        assert_eq!(rounded_sum(&[1.0, half]), Ok(1.0));
        assert_eq!(
            rounded_sum(&[f64::from_bits(1.0f64.to_bits() + 1), half]),
            Ok(f64::from_bits(1.0f64.to_bits() + 2))
        );
        assert_eq!(rounded_sum(&[1.0, half, f64::from_bits(1)]), Ok(f64::from_bits(1.0f64.to_bits() + 1)));
        assert_eq!(rounded_sum(&[f64::from_bits(1), f64::from_bits(1)]), Ok(f64::from_bits(2)));
        assert_eq!(rounded_sum(&[f64::MIN_POSITIVE, -f64::from_bits(1)]), Ok(f64::from_bits((1 << 52) - 1)));
        assert_eq!(rounded_sum(&[-0.0]).unwrap().to_bits(), 0);
    }

    #[test]
    fn nonfinite_and_overflow_are_errors() {
        assert!(rounded_sum(&[f64::INFINITY]).is_err());
        assert!(rounded_sum(&[f64::NAN]).is_err());
        assert!(rounded_sum(&[f64::MAX, f64::MAX]).is_err());
    }

    #[test]
    fn intermediate_overflow_is_exact() {
        assert_eq!(rounded_sum(&[f64::MAX, f64::MAX, -f64::MAX]), Ok(f64::MAX));
    }

    #[test]
    fn exported_intercept_is_none_when_not_representable() {
        assert_eq!(exported_intercept(1.0, &[0.25, 0.5]), Some(0.25));
        assert_eq!(exported_intercept(1.0, &[]), Some(1.0));
        // a nonfinite product, including mixed infinities that would cancel
        assert_eq!(exported_intercept(1.0, &[f64::INFINITY, f64::NEG_INFINITY]), None);
        assert_eq!(exported_intercept(1.0, &[f64::NAN]), None);
        // the sum overflows
        assert_eq!(exported_intercept(0.0, &[f64::MAX, f64::MAX]), None);
        // every term finite, but the final subtraction overflows
        assert_eq!(exported_intercept(f64::MAX, &[-f64::MAX]), None);
        assert_eq!(exported_intercept(f64::NAN, &[]), None);
    }

    #[test]
    fn cancellation_crosses_words_and_rounding_carries_into_the_exponent() {
        for exponent in (-1000..1000).step_by(37) {
            let large = 2.0f64.powi(exponent);
            let small = 2.0f64.powi(exponent - 54);
            for values in [[large, small, -large], [-large, small, large], [small, large, -large]] {
                assert_eq!(rounded_sum(&values), Ok(small));
                assert_eq!(rounded_sum(&values.map(|x| -x)), Ok(-small));
            }
        }
        let below_two = f64::from_bits(2.0f64.to_bits() - 1);
        assert_eq!(rounded_sum(&[below_two, 2.0f64.powi(-53)]), Ok(2.0));
    }
}
