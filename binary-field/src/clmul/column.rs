// CREDIT: leanEthereum/leanVM, MIT OR Apache-2.0 (the eight-column carryless accumulation kernels).
//! Weighted sums of polynomial-field matrix columns, with one reduction per output.

#[cfg(all(
    target_arch = "aarch64",
    target_endian = "little",
    target_feature = "aes"
))]
use core::arch::aarch64::*;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "vpclmulqdq",
    target_feature = "avx512f"
))]
use core::arch::x86_64::*;

use crate::Poly64;

/// Column sums of rows weighted by a table: `sums[w] = Σ_j table[j] · rows[j][w]`.
///
/// The rows are `sums.len()` words each, one per table entry.
pub(crate) fn dot_columns(table: &[Poly64], rows: &[Poly64], sums: &mut [Poly64]) {
    let width = sums.len();
    assert_eq!(Some(rows.len()), table.len().checked_mul(width));
    // Empty sums are zero; no pointer offset into an empty matrix is valid.
    if table.is_empty() {
        sums.fill(Poly64::new(0));
        return;
    }
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "vpclmulqdq",
        target_feature = "avx512f"
    ))]
    let done = {
        let vectors = width / 8;
        for c in 0..vectors {
            // SAFETY: the target features are enabled at compile time, and words `8c..8c + 8` lie inside every row.
            let column = unsafe { dot_column_avx512(table, rows.as_ptr().add(8 * c), width) };
            for (out, product) in sums[8 * c..][..8].iter_mut().zip(column) {
                *out = Poly64::new(super::reduce_64(product));
            }
        }
        8 * vectors
    };
    #[cfg(all(
        target_arch = "aarch64",
        target_endian = "little",
        target_feature = "aes"
    ))]
    let done = {
        let vectors = width / 8;
        for c in 0..vectors {
            // SAFETY: the target feature is enabled at compile time, and words `8c..8c + 8` lie inside every row.
            let column = unsafe { dot_column_neon(table, rows.as_ptr().add(8 * c), width) };
            for (out, product) in sums[8 * c..][..8].iter_mut().zip(column) {
                *out = Poly64::new(super::reduce_64(product));
            }
        }
        8 * vectors
    };
    #[cfg(not(any(
        all(
            target_arch = "x86_64",
            target_feature = "vpclmulqdq",
            target_feature = "avx512f"
        ),
        all(
            target_arch = "aarch64",
            target_endian = "little",
            target_feature = "aes"
        )
    )))]
    let done = 0;
    for (w, sum) in sums.iter_mut().enumerate().skip(done) {
        *sum = Poly64::new(super::poly_dot_64(
            table
                .iter()
                .zip(rows[w..].iter().step_by(width))
                .map(|(t, x)| (t.to_bits(), x.to_bits())),
        ));
    }
}

/// [`dot_columns`] over eight words of every row, the first at `column`, rows `width` words apart.
///
/// # Safety
///
/// - Requires VPCLMULQDQ and AVX-512F.
/// - `column` must address eight readable words in each of `table.len()` rows.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "vpclmulqdq",
    target_feature = "avx512f"
))]
#[inline]
#[target_feature(enable = "vpclmulqdq", enable = "avx512f")]
unsafe fn dot_column_avx512(table: &[Poly64], column: *const Poly64, width: usize) -> [u128; 8] {
    // SAFETY:
    // - The caller supplies eight readable words in every row.
    // - This function's target features cover every intrinsic below.
    unsafe {
        // Products of the even words, then of the odd words, one 128-bit sum per 128-bit lane.
        let (mut even, mut odd) = (_mm512_setzero_si512(), _mm512_setzero_si512());
        for (j, t) in table.iter().enumerate() {
            let x = _mm512_loadu_si512(column.add(j * width).cast());
            let t = _mm512_set1_epi64(t.to_bits() as i64);
            even = _mm512_xor_si512(even, _mm512_clmulepi64_epi128::<0x00>(t, x));
            odd = _mm512_xor_si512(odd, _mm512_clmulepi64_epi128::<0x10>(t, x));
        }
        let (mut e, mut o) = ([0u128; 4], [0u128; 4]);
        _mm512_storeu_si512(e.as_mut_ptr().cast(), even);
        _mm512_storeu_si512(o.as_mut_ptr().cast(), odd);
        core::array::from_fn(|w| if w % 2 == 0 { e[w / 2] } else { o[w / 2] })
    }
}

/// Unreduced column sums over eight words of every row, the first at `column`, rows `width` words apart.
///
/// - Each 128-bit load holds two words.
/// - `PMULL` multiplies the low one by the row's table entry, `PMULL2` the high one.
/// - Two rows go per step, so one three-way XOR folds both of a word's products into its sum.
///
/// # Safety
///
/// - Requires the `aes` target feature.
/// - `column` must address eight readable words in each of `table.len()` rows.
#[cfg(all(
    target_arch = "aarch64",
    target_endian = "little",
    target_feature = "aes"
))]
#[inline]
#[target_feature(enable = "aes")]
unsafe fn dot_column_neon(table: &[Poly64], column: *const Poly64, width: usize) -> [u128; 8] {
    use super::wide::Lanes64;
    // SAFETY:
    // - The caller supplies eight readable words in every row.
    // - This function's target feature covers every intrinsic below.
    unsafe {
        // Row j's eight words as four vectors, each word times the row's table entry t_j.
        //
        //     x[v] = [w_2v, w_2v+1]   ->   (t_j * w_2v, t_j * w_2v+1)
        let products = |j: usize| {
            let row = column.add(j * width).cast::<u64>();
            let t = vdupq_n_u64(table[j].to_bits());
            core::array::from_fn::<_, 4, _>(|v| {
                let x = vld1q_u64(row.add(2 * v));
                let lo = vmull_p64(vgetq_lane_u64::<0>(t), vgetq_lane_u64::<0>(x));
                let hi = vmull_high_p64(vreinterpretq_p64_u64(t), vreinterpretq_p64_u64(x));
                (vreinterpretq_u64_p128(lo), vreinterpretq_u64_p128(hi))
            })
        };
        // One 128-bit sum per word of the column.
        let mut sums = [vdupq_n_u64(0); 8];
        // Rows two at a time: each sum takes both rows' products in one three-way XOR.
        let pairs = table.len() / 2;
        for j in 0..pairs {
            let (a, b) = (products(2 * j), products(2 * j + 1));
            for v in 0..4 {
                sums[2 * v] = sums[2 * v].xor3(a[v].0, b[v].0);
                sums[2 * v + 1] = sums[2 * v + 1].xor3(a[v].1, b[v].1);
            }
        }
        // An odd last row goes alone.
        if table.len() % 2 == 1 {
            let a = products(table.len() - 1);
            for v in 0..4 {
                sums[2 * v] = veorq_u64(sums[2 * v], a[v].0);
                sums[2 * v + 1] = veorq_u64(sums[2 * v + 1], a[v].1);
            }
        }
        sums.map(|s| vreinterpretq_p128_u64(s))
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_field::PrimeCharacteristicRing;
    use proptest::prelude::*;

    use super::*;

    proptest! {
        #[test]
        fn weighted_columns_match_scalar_products(
            height in 0usize..33, width in 0usize..25,
            values in prop::collection::vec(any::<u64>(), 825),
            weights in prop::collection::vec(any::<u64>(), 33),
        ) {
            // Rectangles cover empty inputs, full eight-column groups, and every tail.
            let rows: Vec<_> = values[..height * width].iter().copied().map(Poly64::new).collect();
            let weights: Vec<_> = weights[..height].iter().copied().map(Poly64::new).collect();
            let mut out = alloc::vec![Poly64::ONE; width];
            dot_columns(&weights, &rows, &mut out);
            for (column, actual) in out.into_iter().enumerate() {
                let expected: Poly64 = weights.iter().enumerate()
                    .map(|(row, weight)| *weight * rows[row * width + column]).sum();
                prop_assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    #[should_panic]
    fn malformed_matrix_is_rejected() {
        // Two weights and three output columns require exactly six matrix elements.
        dot_columns(&[Poly64::ONE; 2], &[Poly64::ONE; 5], &mut [Poly64::ZERO; 3]);
    }
}
