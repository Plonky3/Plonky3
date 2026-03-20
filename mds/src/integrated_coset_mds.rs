use alloc::vec::Vec;

use p3_field::{Algebra, Field, Powers, TwoAdicField};
use p3_symmetric::Permutation;
use p3_util::{log2_strict_usize, reverse_slice_index_bits};

use crate::MdsPermutation;
use crate::butterflies::{bowers_g_layer, bowers_g_t_layer_integrated};

/// Optimized Reed-Solomon MDS permutation with integrated coset shifts.
///
/// Compared to the standard coset-based approach:
/// - Uses DIF + DIT (both bit-reversed) instead of DIT + DIF.
/// - Skips bit-reversals of both input and output.
/// - Omits the 1/N rescaling (does not affect the MDS property).
/// - Folds the coset shift powers into the forward DFT twiddle factors,
///   eliminating the separate weighting step.
#[derive(Clone, Debug)]
pub struct IntegratedCosetMds<F, const N: usize> {
    /// Twiddle factors for the inverse DFT, bit-reversed.
    ifft_twiddles: Vec<F>,
    /// Per-layer twiddle factors for the forward DFT.
    /// Each inner vector combines the standard root-of-unity powers
    /// with the corresponding coset shift power for that layer.
    fft_twiddles: Vec<Vec<F>>,
}

impl<F: TwoAdicField, const N: usize> Default for IntegratedCosetMds<F, N> {
    fn default() -> Self {
        let log_n = log2_strict_usize(N);

        // Primitive N-th root of unity and its inverse.
        let root = F::two_adic_generator(log_n);
        let root_inv = root.inverse();
        let coset_shift = F::GENERATOR;

        // Inverse-DFT twiddles: powers of root^{-1}, bit-reversed.
        let mut ifft_twiddles = root_inv.powers().collect_n(N / 2);
        reverse_slice_index_bits(&mut ifft_twiddles);

        // Forward-DFT twiddles: for each layer, combine the root power
        // with the coset shift raised to the same power-of-2 exponent.
        // This folds the separate weighting step into the DFT itself.
        let fft_twiddles = (0..log_n)
            .map(|layer| {
                let powers = Powers {
                    base: root.exp_power_of_2(layer),
                    current: coset_shift.exp_power_of_2(layer),
                };
                let mut twiddles = powers.collect_n(N >> (layer + 1));
                reverse_slice_index_bits(&mut twiddles);
                twiddles
            })
            .collect();

        Self {
            ifft_twiddles,
            fft_twiddles,
        }
    }
}

impl<F: Field, A: Algebra<F>, const N: usize> Permutation<[A; N]> for IntegratedCosetMds<F, N> {
    fn permute_mut(&self, values: &mut [A; N]) {
        let log_n = log2_strict_usize(N);

        // Step 1: bit-reversed DIF (Bowers G) — acts as an inverse DFT.
        for layer in 0..log_n {
            bowers_g_layer(values, layer, &self.ifft_twiddles);
        }

        // Step 2: bit-reversed DIT (Bowers G^T) with integrated coset shifts.
        //
        // Each layer uses its own twiddle table that already includes the shift.
        for layer in (0..log_n).rev() {
            bowers_g_t_layer_integrated(values, layer, &self.fft_twiddles[layer]);
        }
    }
}

impl<F: Field, A: Algebra<F>, const N: usize> MdsPermutation<A, N> for IntegratedCosetMds<F, N> {}

#[cfg(test)]
mod tests {
    use core::array;

    use p3_baby_bear::BabyBear;
    use p3_dft::{NaiveDft, TwoAdicSubgroupDft};
    use p3_field::{Field, PrimeCharacteristicRing, TwoAdicField};
    use p3_goldilocks::Goldilocks;
    use p3_symmetric::Permutation;
    use p3_util::reverse_slice_index_bits;
    use proptest::prelude::*;
    use rand::distr::{Distribution, StandardUniform};
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use crate::integrated_coset_mds::IntegratedCosetMds;

    fn matches_naive_for<F, const N: usize>()
    where
        F: TwoAdicField,
        StandardUniform: Distribution<F>,
    {
        // Generate a random input with a fixed seed.
        let mut rng = SmallRng::seed_from_u64(1);
        let mut arr: [F; N] = array::from_fn(|_| rng.random());

        // The integrated variant works on bit-reversed data,
        // so bit-reverse the input before feeding it to the naive reference.
        let mut arr_rev = arr.to_vec();
        reverse_slice_index_bits(&mut arr_rev);

        // Compute the reference via naive coset LDE, then bit-reverse the output
        // to match the integrated variant's convention.
        let shift = F::GENERATOR;
        let mut coset_lde_naive = NaiveDft.coset_lde(arr_rev, 0, shift);
        reverse_slice_index_bits(&mut coset_lde_naive);

        // Compensate for the omitted 1/N rescaling.
        let scale = F::from_usize(N);
        coset_lde_naive.iter_mut().for_each(|x| *x *= scale);

        // Apply the permutation under test and compare.
        IntegratedCosetMds::<F, N>::default().permute_mut(&mut arr);
        assert_eq!(coset_lde_naive, arr);
    }

    macro_rules! matches_naive_test {
        ($name:ident, $field:ty, $n:expr) => {
            #[test]
            fn $name() {
                matches_naive_for::<$field, $n>();
            }
        };
    }

    matches_naive_test!(matches_naive_baby_bear_1, BabyBear, 1);
    matches_naive_test!(matches_naive_baby_bear_2, BabyBear, 2);
    matches_naive_test!(matches_naive_baby_bear_4, BabyBear, 4);
    matches_naive_test!(matches_naive_baby_bear_8, BabyBear, 8);
    matches_naive_test!(matches_naive_baby_bear_16, BabyBear, 16);
    matches_naive_test!(matches_naive_baby_bear_32, BabyBear, 32);

    matches_naive_test!(matches_naive_goldilocks_1, Goldilocks, 1);
    matches_naive_test!(matches_naive_goldilocks_2, Goldilocks, 2);
    matches_naive_test!(matches_naive_goldilocks_4, Goldilocks, 4);
    matches_naive_test!(matches_naive_goldilocks_8, Goldilocks, 8);
    matches_naive_test!(matches_naive_goldilocks_16, Goldilocks, 16);
    matches_naive_test!(matches_naive_goldilocks_32, Goldilocks, 32);

    #[test]
    fn all_zeros_baby_bear() {
        // All-zeros must map to all-zeros (the permutation is linear).
        let mds = IntegratedCosetMds::<BabyBear, 8>::default();
        let mut zeros = [BabyBear::ZERO; 8];
        mds.permute_mut(&mut zeros);
        assert_eq!(zeros, [BabyBear::ZERO; 8]);
    }

    #[test]
    fn all_zeros_goldilocks() {
        // Same zero-preservation check on a different field.
        let mds = IntegratedCosetMds::<Goldilocks, 8>::default();
        let mut zeros = [Goldilocks::ZERO; 8];
        mds.permute_mut(&mut zeros);
        assert_eq!(zeros, [Goldilocks::ZERO; 8]);
    }

    fn check_linearity<F, const N: usize>(a: [F; N], b: [F; N])
    where
        F: TwoAdicField,
    {
        let mds = IntegratedCosetMds::<F, N>::default();

        // Apply the permutation to the element-wise sum.
        let mut sum: [F; N] = core::array::from_fn(|i| a[i] + b[i]);
        mds.permute_mut(&mut sum);

        // Apply to each vector individually.
        let mut ra = a;
        mds.permute_mut(&mut ra);
        let mut rb = b;
        mds.permute_mut(&mut rb);

        // Linearity: MDS(a + b) must equal MDS(a) + MDS(b).
        let expected: [F; N] = core::array::from_fn(|i| ra[i] + rb[i]);
        assert_eq!(sum, expected);
    }

    fn arb_babybear() -> impl Strategy<Value = BabyBear> {
        prop::num::u32::ANY.prop_map(BabyBear::from_u32)
    }

    proptest! {
        #[test]
        fn integrated_coset_mds_linear_bb8(
            a in prop::array::uniform8(arb_babybear()),
            b in prop::array::uniform8(arb_babybear()),
        ) {
            check_linearity::<BabyBear, 8>(a, b);
        }

        #[test]
        fn integrated_coset_mds_linear_bb16(
            a in prop::array::uniform16(arb_babybear()),
            b in prop::array::uniform16(arb_babybear()),
        ) {
            check_linearity::<BabyBear, 16>(a, b);
        }

        #[test]
        fn integrated_coset_mds_matches_naive_random_bb8(
            input in prop::array::uniform8(arb_babybear()),
        ) {
            // Bit-reverse the input and compute the naive coset LDE as reference.
            let mut arr_rev = input.to_vec();
            reverse_slice_index_bits(&mut arr_rev);

            let shift = BabyBear::GENERATOR;
            let mut coset_lde_naive = NaiveDft.coset_lde(arr_rev, 0, shift);
            reverse_slice_index_bits(&mut coset_lde_naive);

            // Compensate for the omitted 1/N rescaling.
            let scale = BabyBear::from_usize(8);
            coset_lde_naive.iter_mut().for_each(|x| *x *= scale);

            // Apply the permutation and compare.
            let mut result = input;
            IntegratedCosetMds::<BabyBear, 8>::default().permute_mut(&mut result);
            prop_assert_eq!(coset_lde_naive, result);
        }
    }
}
