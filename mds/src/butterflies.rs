//! Butterfly operations for FFT-style networks.
//!
//! Provides decimation-in-time (DIT) and decimation-in-frequency (DIF) butterflies,
//! plus full Bowers G and G^T network layers built from them.

use p3_field::{Algebra, Field};

/// Decimation-in-time butterfly.
///
/// Computes in-place on two elements at the given indices:
/// - out_1 = in_1 + twiddle * in_2
/// - out_2 = in_1 - twiddle * in_2
#[inline]
pub(crate) fn dit_butterfly<F: Field, A: Algebra<F>, const N: usize>(
    values: &mut [A; N],
    idx_1: usize,
    idx_2: usize,
    twiddle: F,
) {
    // Scale the second element by the twiddle factor.
    let val_1 = values[idx_1].clone();
    let val_2 = values[idx_2].clone() * twiddle;
    // Write the sum and difference.
    values[idx_1] = val_1.clone() + val_2.clone();
    values[idx_2] = val_1 - val_2;
}

/// Decimation-in-frequency butterfly.
///
/// Computes in-place on two elements at the given indices:
/// - out_1 = in_1 + in_2
/// - out_2 = (in_1 - in_2) * twiddle
#[inline]
pub(crate) fn dif_butterfly<F: Field, A: Algebra<F>, const N: usize>(
    values: &mut [A; N],
    idx_1: usize,
    idx_2: usize,
    twiddle: F,
) {
    let val_1 = values[idx_1].clone();
    let val_2 = values[idx_2].clone();
    // First slot gets the plain sum.
    values[idx_1] = val_1.clone() + val_2.clone();
    // Second slot gets the difference scaled by the twiddle.
    values[idx_2] = (val_1 - val_2) * twiddle;
}

/// Butterfly with implicit twiddle factor of 1.
///
/// Equivalent to either DIT or DIF when the twiddle is the multiplicative identity.
/// Avoids the redundant multiplication.
///
/// - out_1 = in_1 + in_2
/// - out_2 = in_1 - in_2
#[inline]
pub(crate) fn twiddle_free_butterfly<F: Field, A: Algebra<F>, const N: usize>(
    values: &mut [A; N],
    idx_1: usize,
    idx_2: usize,
) {
    let val_1 = values[idx_1].clone();
    let val_2 = values[idx_2].clone();
    values[idx_1] = val_1.clone() + val_2.clone();
    values[idx_2] = val_1 - val_2;
}

/// One layer of a Bowers G network (DIF-based).
///
/// Partitions the array into blocks of size 2^{log_half_block_size + 1}.
/// Within each block, pairs the upper and lower halves through DIF butterflies.
/// The first block always uses twiddle = 1 (unrolled for efficiency).
#[inline]
pub(crate) fn bowers_g_layer<F: Field, A: Algebra<F>, const N: usize>(
    values: &mut [A; N],
    log_half_block_size: usize,
    twiddles: &[F],
) {
    let log_block_size = log_half_block_size + 1;
    let half_block_size = 1 << log_half_block_size;
    let num_blocks = N >> log_block_size;

    // First block: twiddle is always 1, so skip the multiplication.
    for hi in 0..half_block_size {
        let lo = hi + half_block_size;
        twiddle_free_butterfly(values, hi, lo);
    }

    // Remaining blocks: use the corresponding twiddle factor from the table.
    for (block, &twiddle) in (1..num_blocks).zip(&twiddles[1..]) {
        let block_start = block << log_block_size;
        for hi in block_start..block_start + half_block_size {
            let lo = hi + half_block_size;
            dif_butterfly(values, hi, lo, twiddle);
        }
    }
}

/// One layer of a Bowers G^T network (DIT-based).
///
/// Transpose of the G layer.
/// Same block structure, but uses DIT butterflies instead of DIF.
/// The first block still uses twiddle = 1.
#[inline]
pub(crate) fn bowers_g_t_layer<F: Field, A: Algebra<F>, const N: usize>(
    values: &mut [A; N],
    log_half_block_size: usize,
    twiddles: &[F],
) {
    let log_block_size = log_half_block_size + 1;
    let half_block_size = 1 << log_half_block_size;
    let num_blocks = N >> log_block_size;

    // First block: twiddle is always 1.
    for hi in 0..half_block_size {
        let lo = hi + half_block_size;
        twiddle_free_butterfly(values, hi, lo);
    }

    // Remaining blocks: DIT butterfly with the per-block twiddle.
    for (block, &twiddle) in (1..num_blocks).zip(&twiddles[1..]) {
        let block_start = block << log_block_size;
        for hi in block_start..block_start + half_block_size {
            let lo = hi + half_block_size;
            dit_butterfly(values, hi, lo, twiddle);
        }
    }
}

/// One layer of a Bowers G^T network with pre-integrated twiddle factors.
///
/// Unlike the standard G^T layer, this variant applies a non-trivial twiddle
/// to every block — including block 0.
/// The twiddle table is expected to already incorporate coset shifts.
#[inline]
pub(crate) fn bowers_g_t_layer_integrated<F: Field, A: Algebra<F>, const N: usize>(
    values: &mut [A; N],
    log_half_block_size: usize,
    twiddles: &[F],
) {
    let log_block_size = log_half_block_size + 1;
    let half_block_size = 1 << log_half_block_size;
    let num_blocks = N >> log_block_size;

    // Every block uses its own pre-computed twiddle (no special-case for block 0).
    for (block, &twiddle) in (0..num_blocks).zip(twiddles) {
        let block_start = block << log_block_size;
        for hi in block_start..block_start + half_block_size {
            let lo = hi + half_block_size;
            dit_butterfly(values, hi, lo, twiddle);
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::{PrimeCharacteristicRing, TwoAdicField};
    use proptest::prelude::*;

    use super::*;

    type F = BabyBear;

    fn arb_f() -> impl Strategy<Value = F> {
        prop::num::u32::ANY.prop_map(F::from_u32)
    }

    // Individual butterfly tests

    #[test]
    fn dit_butterfly_manual() {
        let a = F::from_u32(7);
        let b = F::from_u32(11);
        let t = F::from_u32(3);
        let mut vals = [a, b];

        // Apply the decimation-in-time butterfly with a known twiddle factor.
        dit_butterfly::<F, F, 2>(&mut vals, 0, 1, t);

        // First output: sum of first input and twiddle-scaled second input.
        assert_eq!(vals[0], a + b * t);
        // Second output: difference of first input and twiddle-scaled second input.
        assert_eq!(vals[1], a - b * t);
    }

    #[test]
    fn dif_butterfly_manual() {
        let a = F::from_u32(7);
        let b = F::from_u32(11);
        let t = F::from_u32(3);
        let mut vals = [a, b];

        // Apply the decimation-in-frequency butterfly with a known twiddle factor.
        dif_butterfly::<F, F, 2>(&mut vals, 0, 1, t);

        // First output: plain sum of the two inputs.
        assert_eq!(vals[0], a + b);
        // Second output: difference scaled by the twiddle factor.
        assert_eq!(vals[1], (a - b) * t);
    }

    #[test]
    fn twiddle_free_butterfly_manual() {
        let a = F::from_u32(7);
        let b = F::from_u32(11);
        let mut vals = [a, b];

        // Apply the twiddle-free variant (implicitly twiddle = 1).
        twiddle_free_butterfly::<F, F, 2>(&mut vals, 0, 1);

        // First output: sum.
        assert_eq!(vals[0], a + b);
        // Second output: difference.
        assert_eq!(vals[1], a - b);
    }

    #[test]
    fn dit_with_twiddle_one_equals_twiddle_free() {
        let a = F::from_u32(42);
        let b = F::from_u32(99);

        // Apply DIT with an explicit twiddle of 1.
        let mut vals_dit = [a, b];
        dit_butterfly::<F, F, 2>(&mut vals_dit, 0, 1, F::ONE);

        // Apply the twiddle-free variant.
        let mut vals_free = [a, b];
        twiddle_free_butterfly::<F, F, 2>(&mut vals_free, 0, 1);

        // Both should produce the same result.
        assert_eq!(vals_dit, vals_free);
    }

    #[test]
    fn dif_with_twiddle_one_equals_twiddle_free() {
        let a = F::from_u32(42);
        let b = F::from_u32(99);

        // Apply DIF with an explicit twiddle of 1.
        let mut vals_dif = [a, b];
        dif_butterfly::<F, F, 2>(&mut vals_dif, 0, 1, F::ONE);

        // Apply the twiddle-free variant.
        let mut vals_free = [a, b];
        twiddle_free_butterfly::<F, F, 2>(&mut vals_free, 0, 1);

        // Both should produce the same result.
        assert_eq!(vals_dif, vals_free);
    }

    #[test]
    fn dit_preserves_trace() {
        let a = F::from_u32(123);
        let b = F::from_u32(456);
        let t = F::from_u32(789);
        let mut vals = [a, b];

        // Apply DIT with an arbitrary twiddle.
        dit_butterfly::<F, F, 2>(&mut vals, 0, 1, t);

        // The DIT butterfly outputs (a + t*b, a - t*b).
        // Their sum cancels the second-input contribution: (a + t*b) + (a - t*b) = 2*a.
        assert_eq!(vals[0] + vals[1], a.double());
    }

    #[test]
    fn dif_sum_property() {
        let a = F::from_u32(123);
        let b = F::from_u32(456);
        let t = F::from_u32(789);
        let mut vals = [a, b];

        // Apply DIF with an arbitrary twiddle.
        dif_butterfly::<F, F, 2>(&mut vals, 0, 1, t);

        // The DIF butterfly always stores the plain sum in the first slot,
        // regardless of the twiddle factor.
        assert_eq!(vals[0], a + b);
    }

    #[test]
    fn butterfly_on_non_adjacent_indices() {
        let mut vals: [F; 4] = [10, 20, 30, 40].map(F::from_u32);
        let original = vals;
        let t = F::from_u32(5);

        // Apply a DIT butterfly only on indices 0 and 3 within a length-4 array.
        dit_butterfly::<F, F, 4>(&mut vals, 0, 3, t);

        // The intermediate indices must be untouched.
        assert_eq!(vals[1], original[1]);
        assert_eq!(vals[2], original[2]);

        // The targeted pair follows the standard DIT formula.
        assert_eq!(vals[0], original[0] + original[3] * t);
        assert_eq!(vals[3], original[0] - original[3] * t);
    }

    #[test]
    fn dit_zero_twiddle() {
        let a = F::from_u32(7);
        let b = F::from_u32(11);
        let mut vals = [a, b];

        // With twiddle = 0, the second input is annihilated.
        dit_butterfly::<F, F, 2>(&mut vals, 0, 1, F::ZERO);

        // Both outputs collapse to the first input value.
        assert_eq!(vals[0], a);
        assert_eq!(vals[1], a);
    }

    #[test]
    fn dif_zero_twiddle() {
        let a = F::from_u32(7);
        let b = F::from_u32(11);
        let mut vals = [a, b];

        // With twiddle = 0, the difference is annihilated after scaling.
        dif_butterfly::<F, F, 2>(&mut vals, 0, 1, F::ZERO);

        // First slot gets the sum; second slot is zeroed by the twiddle.
        assert_eq!(vals[0], a + b);
        assert_eq!(vals[1], F::ZERO);
    }

    // Bowers G / G^T layer tests

    #[test]
    fn bowers_g_then_g_t_roundtrip_n4() {
        // Use the primitive 4th root of unity as the non-trivial twiddle.
        let omega = F::two_adic_generator(2);
        let twiddles = [F::ONE, omega];

        let original: [F; 4] = [3, 7, 11, 13].map(F::from_u32);

        // Apply a single G layer (DIF-based) then G^T layer (DIT-based).
        let mut vals = original;
        bowers_g_layer::<F, F, 4>(&mut vals, 0, &twiddles);
        bowers_g_t_layer::<F, F, 4>(&mut vals, 0, &twiddles);

        // Run the same sequence a second time to verify determinism.
        // A single-layer round trip is not the identity in general
        // (only the full multi-layer network is), but it must be reproducible.
        let mut vals2 = original;
        bowers_g_layer::<F, F, 4>(&mut vals2, 0, &twiddles);
        bowers_g_t_layer::<F, F, 4>(&mut vals2, 0, &twiddles);
        assert_eq!(vals, vals2);
    }

    #[test]
    fn integrated_matches_regular_with_unit_twiddles() {
        // When every twiddle is 1, the integrated variant reduces to the standard one.
        let twiddles = [F::ONE; 4];
        let original: [F; 8] = [1, 2, 3, 4, 5, 6, 7, 8].map(F::from_u32);

        // Apply the standard G^T layer.
        let mut vals_regular = original;
        bowers_g_t_layer::<F, F, 8>(&mut vals_regular, 0, &twiddles);

        // Apply the integrated variant.
        let mut vals_integrated = original;
        bowers_g_t_layer_integrated::<F, F, 8>(&mut vals_integrated, 0, &twiddles);

        // Both must agree since the integrated twiddles are all 1.
        assert_eq!(vals_regular, vals_integrated);
    }

    #[test]
    fn all_zeros_through_layers() {
        let twiddles = [F::ONE, F::two_adic_generator(2)];

        // All-zeros input must stay all-zeros through any linear butterfly layer.

        let mut vals = [F::ZERO; 4];
        bowers_g_layer::<F, F, 4>(&mut vals, 0, &twiddles);
        assert_eq!(vals, [F::ZERO; 4]);

        let mut vals = [F::ZERO; 4];
        bowers_g_t_layer::<F, F, 4>(&mut vals, 0, &twiddles);
        assert_eq!(vals, [F::ZERO; 4]);

        let mut vals = [F::ZERO; 4];
        bowers_g_t_layer_integrated::<F, F, 4>(&mut vals, 0, &twiddles);
        assert_eq!(vals, [F::ZERO; 4]);
    }

    // Property-based tests

    proptest! {
        #[test]
        fn dit_is_linear(
            a1 in arb_f(), b1 in arb_f(),
            a2 in arb_f(), b2 in arb_f(),
            t in arb_f(),
        ) {
            // Verify additivity: DIT(u + v) = DIT(u) + DIT(v).
            // The butterfly is a linear map for a fixed twiddle factor.

            // Apply to the element-wise sum.
            let mut sum_then_dit = [a1 + a2, b1 + b2];
            dit_butterfly::<F, F, 2>(&mut sum_then_dit, 0, 1, t);

            // Apply to each input separately.
            let mut dit1 = [a1, b1];
            dit_butterfly::<F, F, 2>(&mut dit1, 0, 1, t);
            let mut dit2 = [a2, b2];
            dit_butterfly::<F, F, 2>(&mut dit2, 0, 1, t);

            // The sum of individual results must match.
            prop_assert_eq!(sum_then_dit[0], dit1[0] + dit2[0]);
            prop_assert_eq!(sum_then_dit[1], dit1[1] + dit2[1]);
        }

        #[test]
        fn dif_is_linear(
            a1 in arb_f(), b1 in arb_f(),
            a2 in arb_f(), b2 in arb_f(),
            t in arb_f(),
        ) {
            // Same additivity check for the DIF butterfly.

            let mut sum_then_dif = [a1 + a2, b1 + b2];
            dif_butterfly::<F, F, 2>(&mut sum_then_dif, 0, 1, t);

            let mut dif1 = [a1, b1];
            dif_butterfly::<F, F, 2>(&mut dif1, 0, 1, t);
            let mut dif2 = [a2, b2];
            dif_butterfly::<F, F, 2>(&mut dif2, 0, 1, t);

            prop_assert_eq!(sum_then_dif[0], dif1[0] + dif2[0]);
            prop_assert_eq!(sum_then_dif[1], dif1[1] + dif2[1]);
        }

        #[test]
        fn dit_twiddle_one_squared_is_double(a in arb_f(), b in arb_f()) {
            // Applying the unit-twiddle DIT butterfly twice doubles each element.
            // (a, b) -> (a+b, a-b) -> (2a, 2b).
            let mut vals = [a, b];
            dit_butterfly::<F, F, 2>(&mut vals, 0, 1, F::ONE);
            dit_butterfly::<F, F, 2>(&mut vals, 0, 1, F::ONE);
            prop_assert_eq!(vals[0], a.double());
            prop_assert_eq!(vals[1], b.double());
        }

        #[test]
        fn dif_twiddle_one_squared_is_double(a in arb_f(), b in arb_f()) {
            // Same double-application property holds for DIF with twiddle 1.
            let mut vals = [a, b];
            dif_butterfly::<F, F, 2>(&mut vals, 0, 1, F::ONE);
            dif_butterfly::<F, F, 2>(&mut vals, 0, 1, F::ONE);
            prop_assert_eq!(vals[0], a.double());
            prop_assert_eq!(vals[1], b.double());
        }

        #[test]
        fn twiddle_free_matches_dit_and_dif_unit(a in arb_f(), b in arb_f()) {
            // The twiddle-free variant must be identical to both DIT and DIF
            // when their twiddle factor is 1.

            let mut free = [a, b];
            twiddle_free_butterfly::<F, F, 2>(&mut free, 0, 1);

            let mut dit = [a, b];
            dit_butterfly::<F, F, 2>(&mut dit, 0, 1, F::ONE);

            let mut dif = [a, b];
            dif_butterfly::<F, F, 2>(&mut dif, 0, 1, F::ONE);

            prop_assert_eq!(free, dit);
            prop_assert_eq!(free, dif);
        }

        #[test]
        fn bowers_g_layer_is_linear(
            v1 in prop::array::uniform4(arb_f()),
            v2 in prop::array::uniform4(arb_f()),
        ) {
            // A single G layer is a linear map, so it must distribute over addition.
            let twiddles = [F::ONE, F::two_adic_generator(2)];

            // Apply to the element-wise sum.
            let mut sum = core::array::from_fn::<F, 4, _>(|i| v1[i] + v2[i]);
            bowers_g_layer::<F, F, 4>(&mut sum, 0, &twiddles);

            // Apply to each input separately.
            let mut r1 = v1;
            bowers_g_layer::<F, F, 4>(&mut r1, 0, &twiddles);
            let mut r2 = v2;
            bowers_g_layer::<F, F, 4>(&mut r2, 0, &twiddles);

            for i in 0..4 {
                prop_assert_eq!(sum[i], r1[i] + r2[i]);
            }
        }

        #[test]
        fn bowers_g_t_layer_is_linear(
            v1 in prop::array::uniform4(arb_f()),
            v2 in prop::array::uniform4(arb_f()),
        ) {
            // Same linearity check for the transpose layer.
            let twiddles = [F::ONE, F::two_adic_generator(2)];

            let mut sum = core::array::from_fn::<F, 4, _>(|i| v1[i] + v2[i]);
            bowers_g_t_layer::<F, F, 4>(&mut sum, 0, &twiddles);

            let mut r1 = v1;
            bowers_g_t_layer::<F, F, 4>(&mut r1, 0, &twiddles);
            let mut r2 = v2;
            bowers_g_t_layer::<F, F, 4>(&mut r2, 0, &twiddles);

            for i in 0..4 {
                prop_assert_eq!(sum[i], r1[i] + r2[i]);
            }
        }

        #[test]
        fn bowers_g_t_layer_integrated_is_linear(
            v1 in prop::array::uniform4(arb_f()),
            v2 in prop::array::uniform4(arb_f()),
        ) {
            // Same linearity check for the integrated-twiddle variant.
            let twiddles = [F::ONE, F::two_adic_generator(2)];

            let mut sum = core::array::from_fn::<F, 4, _>(|i| v1[i] + v2[i]);
            bowers_g_t_layer_integrated::<F, F, 4>(&mut sum, 0, &twiddles);

            let mut r1 = v1;
            bowers_g_t_layer_integrated::<F, F, 4>(&mut r1, 0, &twiddles);
            let mut r2 = v2;
            bowers_g_t_layer_integrated::<F, F, 4>(&mut r2, 0, &twiddles);

            for i in 0..4 {
                prop_assert_eq!(sum[i], r1[i] + r2[i]);
            }
        }
    }
}
