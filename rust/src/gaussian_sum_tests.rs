//! Per-row updates, storage boundaries and the declared default sum order.
use super::*;

#[test]
fn per_row_arithmetic_and_excluded_rows_are_unchanged() {
    for n in [1, 7, 8, 9, CHUNK - 1, CHUNK, CHUNK + 1, 3 * CHUNK + 9] {
        for masked in [false, true] {
            for (center, scale, shift) in [(0., 0.5, 0.123456789), (1e9, 0.25, -0.15), (0., 1e150, 1e-200)] {
                let x: Vec<_> = (0..n).map(|i| center + ((i * 17) % 31) as f64 * scale).collect();
                let mask: Vec<_> = (0..n).map(|i| if masked && i % 3 == 1 { 0. } else { 1. }).collect();
                let initial: Vec<_> =
                    (0..n).map(|i| if mask[i] == 0. { 0. } else { (i % 7) as f64 * 0.314159265 - 1. }).collect();
                let mut expected = initial.clone();
                for i in 0..n {
                    if mask[i] != 0. {
                        expected[i] -= shift * ((x[i] - center) / scale);
                    }
                }
                let mut actual = initial;
                let sum =
                    update_blocked(&Contig::<_, true>(&x), &mut actual, masked.then_some(&mask), shift, center, scale);
                assert_eq!(
                    actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
                assert_eq!(sum.to_bits(), reference_sum(&expected).to_bits());
            }
        }
    }
}

#[test]
fn current_update_does_not_regroup_tiny_shift_over_large_scale() {
    let mut residual = [0.; 8];
    update_blocked(&Contig::<_, true>(&[1e150; 8]), &mut residual, None, 1e-200, 0., 1e150);
    assert_eq!(residual.map(f64::to_bits), [-1e-200_f64; 8].map(f64::to_bits));
}

#[test]
fn mapped_block_updates_keep_logical_lanes_across_storage_runs() {
    for n in [7, 8, 9, CHUNK - 1, CHUNK, CHUNK + 1, 3 * CHUNK + 9] {
        let x: Vec<f64> = (0..n).map(|i| f64::from(i % 3 == 1)).collect();
        let bytes: Vec<u8> = x.iter().map(|&v| v as u8).collect();
        let mut packed = vec![0_u8; (n + 3).div_ceil(8)];
        for (i, &v) in bytes.iter().enumerate() {
            packed[(i + 3) / 8] |= v << ((i + 3) % 8);
        }
        let initial: Vec<f64> = (0..n).map(|i| (i % 7) as f64 * 0.314159265 - 1.).collect();
        let expected: Vec<f64> = initial.iter().zip(&x).map(|(&r, &v)| r - 0.123456789 * ((v - 0.3) / 1.3)).collect();
        for batch in [n, 7, CHUNK + 1] {
            for kind in 0..3 {
                let chunks = (0..n)
                    .step_by(batch)
                    .map(|start| {
                        let len = batch.min(n - start);
                        match kind {
                            0 => MappedCol::F64(&x[start..start + len]),
                            1 => MappedCol::U8(&bytes[start..start + len]),
                            _ => {
                                let bit = start + 3;
                                let offset = bit % 8;
                                let count = (offset + len).div_ceil(8);
                                MappedCol::Bits { bytes: &packed[bit / 8..bit / 8 + count], offset, len }
                            }
                        }
                    })
                    .collect();
                let design = MappedX { columns: vec![ChunkedColumn::new(chunks, n).unwrap()] };
                let mut actual = initial.clone();
                let sum = update_blocked(&design.col(0), &mut actual, None, 0.123456789, 0.3, 1.3);
                assert_eq!(
                    actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
                assert_eq!(sum.to_bits(), reference_sum(&expected).to_bits());
            }
        }
    }
}

// Evaluate the declared tree lane-by-lane rather than the fused row traversal.
// Expectations come from its arithmetic definition, not a fitted path.
pub(super) fn reference_sum(values: &[f64]) -> f64 {
    values.chunks(CHUNK).fold(0., |total, chunk| {
        let body = chunk.len() / LANES * LANES;
        let mut tree: Vec<f64> =
            (0..LANES).map(|lane| chunk[..body].iter().skip(lane).step_by(LANES).fold(0., |s, &v| s + v)).collect();
        while tree.len() > 1 {
            tree = tree.as_chunks::<2>().0.iter().map(|pair| pair[0] + pair[1]).collect();
        }
        total + (tree[0] + chunk[body..].iter().fold(0., |s, &v| s + v))
    })
}

// Release-only: debug builds intentionally outline intrinsics. ELF x86-64
// names expose the exact regression measured in the mapped-column visitor.
// Other platforms retain the functional width/storage tests, without a flaky
// wall-clock threshold or an architecture-specific production kernel.
#[cfg(all(target_os = "linux", target_arch = "x86_64", not(debug_assertions)))]
#[test]
fn release_simd_arithmetic_has_no_out_of_line_avx_helpers() {
    let output = std::process::Command::new("nm")
        .arg("-C")
        .arg(std::env::current_exe().unwrap())
        .output()
        .expect("native code-generation audit requires binutils nm");
    assert!(output.status.success(), "nm failed: {}", String::from_utf8_lossy(&output.stderr));
    let symbols = String::from_utf8(output.stdout).unwrap();
    for width in [256, 512] {
        for operation in ["add", "sub", "mul", "div", "set1"] {
            let name = format!("::_mm{width}_{operation}_pd");
            assert!(
                !symbols.lines().any(|line| line.ends_with(&name)),
                "outlined SIMD arithmetic {name}: keep vector operations inside the target-feature dispatch body"
            );
        }
    }
}

#[test]
fn default_and_compatible_reductions_pin_distinct_documented_orders() {
    // At magnitude 1e16 the binary64 spacing is 2. Adding 1 rounds to the
    // even endpoint. The default pairs cancel; the serial fourth add is 1.
    let residual = [1e16, 1., -1e16, 1., 0., 0., 0., 0.];
    assert_eq!(reference_sum(&residual).to_bits(), 0_f64.to_bits());
    assert_eq!(blocked_sum(&residual).to_bits(), 0_f64.to_bits());
    assert_eq!(seqsum(&residual).to_bits(), 1_f64.to_bits());
}
