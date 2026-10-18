//! Zero-tail coefficient transforms must retain the full DFT's roots and row ordering.

use p3_baby_bear::BabyBear;
use p3_dft::{Radix2Bowers, Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::extension::BinomialExtensionField;
use p3_field::{PrimeCharacteristicRing, TwoAdicField};
use p3_goldilocks::Goldilocks;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use proptest::prelude::*;
use rand::SeedableRng;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;

fn check_padded<F, D>(dft: &D)
where
    F: TwoAdicField,
    D: TwoAdicSubgroupDft<F>,
    StandardUniform: Distribution<F>,
{
    let mut rng = SmallRng::seed_from_u64(0xabcdef);
    for log_height in 0..=8 {
        for log_inv_rate in 0..=log_height {
            for width in [1, 3, 4, 17] {
                let mut mat =
                    RowMajorMatrix::<F>::rand(&mut rng, 1 << (log_height - log_inv_rate), width);
                mat.values.resize((1 << log_height) * width, F::ZERO);
                let expected = dft.dft_batch(mat.clone()).to_row_major_matrix();
                let actual = dft
                    .dft_batch_padded(mat, log_inv_rate)
                    .to_row_major_matrix();
                assert_eq!(
                    actual, expected,
                    "height={log_height}, rate={log_inv_rate}, width={width}"
                );
            }
        }
    }
}

#[test]
fn padded_dit_matches_full_dft_across_fields_and_layer_boundaries() {
    check_padded(&Radix2DitParallel::<BabyBear>::default());
    check_padded(&Radix2DitParallel::<Goldilocks>::default());
    check_padded(&Radix2DitParallel::<BinomialExtensionField<BabyBear, 4>>::default());
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    #[test]
    fn padded_dit_matches_full_dft_for_larger_low_rate_batches(
        log_height in 9usize..=11,
        log_inv_rate in 2usize..=3,
        width in 1usize..=17,
        seed in any::<u64>(),
    ) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut mat = RowMajorMatrix::<BabyBear>::rand(
            &mut rng,
            1 << (log_height - log_inv_rate),
            width,
        );
        mat.values.resize((1 << log_height) * width, BabyBear::ZERO);
        let dft = Radix2DitParallel::default();
        let expected = dft.dft_batch(mat.clone()).to_row_major_matrix();
        let actual = dft.dft_batch_padded(mat, log_inv_rate).to_row_major_matrix();
        prop_assert_eq!(actual, expected);
    }
}

#[test]
fn padded_default_preserves_backend_layout() {
    check_padded::<BabyBear, _>(&Radix2Bowers);
}

// A downstream backend may use the inverse primitive root as its forward root.
// The default hook must call that backend, rather than impose another convention.
#[derive(Clone, Default)]
struct InverseRootDft;

impl TwoAdicSubgroupDft<BabyBear> for InverseRootDft {
    type Evaluations = RowMajorMatrix<BabyBear>;

    fn dft_batch(&self, mat: RowMajorMatrix<BabyBear>) -> Self::Evaluations {
        let dft = Radix2DitParallel::default();
        let mut result = dft.dft_batch(mat).to_row_major_matrix();
        let height = result.height();
        for row in 1..height.div_ceil(2) {
            p3_matrix::util::swap_rows(&mut result, row, height - row);
        }
        result
    }
}

#[test]
fn padded_default_preserves_custom_root_convention() {
    check_padded::<BabyBear, _>(&InverseRootDft);
}

#[test]
#[should_panic(expected = "padding exceeds matrix height")]
fn padded_dit_rejects_excess_padding() {
    let mat = RowMajorMatrix::new_col(vec![BabyBear::ONE; 4]);
    let _ = Radix2DitParallel::default().dft_batch_padded(mat, 3);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "padded tail must be zero")]
fn padded_dit_rejects_nonzero_tail() {
    let mut mat = RowMajorMatrix::new(vec![BabyBear::ZERO; 16 * 3], 3);
    mat.values[4 * 3 + 1] = BabyBear::ONE;
    let _ = Radix2DitParallel::default().dft_batch_padded(mat, 2);
}

#[test]
#[should_panic(expected = "padding exceeds matrix height")]
fn padded_default_rejects_shift_overflow() {
    let mat = RowMajorMatrix::new_col(vec![BabyBear::ONE; 4]);
    let _ = InverseRootDft.dft_batch_padded(mat, usize::MAX);
}

#[test]
#[should_panic]
fn padded_dit_rejects_non_power_of_two_height() {
    let mat = RowMajorMatrix::new_col(vec![BabyBear::ONE; 3]);
    let _ = Radix2DitParallel::default().dft_batch_padded(mat, 1);
}

#[test]
fn padded_dit_evaluates_sparse_coefficients_at_declared_roots() {
    let dft = Radix2DitParallel::<BabyBear>::default();
    let mut values = vec![BabyBear::ZERO; 128 * 2];
    for (row, col, coefficient) in [(0, 0, 2), (2, 0, 5), (1, 1, 7), (5, 1, 3)] {
        values[row * 2 + col] = BabyBear::from_u8(coefficient);
    }
    let evaluations = dft.dft_batch_padded(RowMajorMatrix::new(values, 2), 4);
    for (row, x) in BabyBear::two_adic_generator(7)
        .powers()
        .take(128)
        .enumerate()
    {
        assert_eq!(
            evaluations.get(row, 0).unwrap(),
            BabyBear::from_u8(2) + BabyBear::from_u8(5) * x.square()
        );
        assert_eq!(
            evaluations.get(row, 1).unwrap(),
            BabyBear::from_u8(7) * x + BabyBear::from_u8(3) * x.exp_u64(5)
        );
    }

    let zero = RowMajorMatrix::new(vec![BabyBear::ZERO; 64 * 3], 3);
    assert_eq!(
        dft.dft_batch_padded(zero.clone(), 3).to_row_major_matrix(),
        zero
    );
}

// A field with only two subgroup points keeps invalid-domain tests small.
mod small_domain {
    use p3_monty_31::{
        FieldParameters, MontyField31, MontyParameters, PackedMontyParameters, TwoAdicData,
    };

    use super::*;

    #[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
    struct MersenneParameters;

    type F = MontyField31<MersenneParameters>;

    impl MontyParameters for MersenneParameters {
        const PRIME: u32 = (1 << 31) - 1;
        const MONTY_BITS: u32 = 32;
        const MONTY_MU: u32 = Self::PRIME;
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    impl p3_monty_31::MontyParametersNeon for MersenneParameters {
        // SAFETY: These arrays are the corresponding broadcast vector representations.
        const PACKED_P: core::arch::aarch64::uint32x4_t =
            unsafe { core::mem::transmute([Self::PRIME; 4]) };
        const PACKED_MU: core::arch::aarch64::int32x4_t =
            unsafe { core::mem::transmute([Self::MONTY_MU as i32; 4]) };
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        not(target_feature = "avx512f")
    ))]
    impl p3_monty_31::MontyParametersAVX2 for MersenneParameters {
        // SAFETY: These arrays are the corresponding broadcast vector representations.
        const PACKED_P: core::arch::x86_64::__m256i =
            unsafe { core::mem::transmute([Self::PRIME; 8]) };
        const PACKED_MU: core::arch::x86_64::__m256i =
            unsafe { core::mem::transmute([Self::MONTY_MU; 8]) };
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    impl p3_monty_31::MontyParametersAVX512 for MersenneParameters {
        // SAFETY: These arrays are the corresponding broadcast vector representations.
        const PACKED_P: core::arch::x86_64::__m512i =
            unsafe { core::mem::transmute([Self::PRIME; 16]) };
        const PACKED_MU: core::arch::x86_64::__m512i =
            unsafe { core::mem::transmute([Self::MONTY_MU; 16]) };
    }

    impl PackedMontyParameters for MersenneParameters {}

    impl FieldParameters for MersenneParameters {
        const MONTY_GEN: F = F::new(7);
    }

    impl TwoAdicData for MersenneParameters {
        const TWO_ADICITY: usize = 1;
        type ArrayLike = &'static [F];
        const TWO_ADIC_GENERATORS: Self::ArrayLike = &[F::ONE, F::NEG_ONE];
        // This field has no eighth or sixteenth roots of unity.
        const ROOTS_8: Self::ArrayLike = &[];
        const INV_ROOTS_8: Self::ArrayLike = &[];
        const ROOTS_16: Self::ArrayLike = &[];
        const INV_ROOTS_16: Self::ArrayLike = &[];
    }

    #[test]
    #[should_panic(expected = "transform height exceeds field two-adicity")]
    fn padded_dit_rejects_rate_two_height_above_field_two_adicity() {
        let mat = RowMajorMatrix::new_col(vec![F::ONE, F::ONE, F::ZERO, F::ZERO]);
        let _ = Radix2DitParallel::default().dft_batch_padded(mat, 1);
    }
}
