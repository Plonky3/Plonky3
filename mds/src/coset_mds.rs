use alloc::vec::Vec;

use p3_field::{Algebra, Field, TwoAdicField};
use p3_symmetric::Permutation;
use p3_util::{log2_strict_usize, reverse_slice_index_bits};

use crate::MdsPermutation;
use crate::butterflies::{bowers_g_layer, bowers_g_t_layer};

/// Reed-Solomon based MDS permutation.
///
/// Interprets the input as evaluations of a polynomial over a
/// power-of-two subgroup, then computes evaluations over a coset
/// of that subgroup.
/// This is equivalent to returning the parity elements of a
/// systematic Reed-Solomon code.
/// Since Reed-Solomon codes are MDS, the resulting map is MDS.
///
/// # Algorithm
///
/// 1. Inverse DFT via Bowers G^T (skip bit-reversal and 1/N rescaling).
/// 2. Multiply by powers of the coset shift.
/// 3. Forward DFT via Bowers G (assumes bit-reversed input).
#[derive(Clone, Debug)]
pub struct CosetMds<F, const N: usize> {
    /// Twiddle factors for the forward DFT, bit-reversed.
    fft_twiddles: Vec<F>,
    /// Twiddle factors for the inverse DFT, bit-reversed.
    ifft_twiddles: Vec<F>,
    /// Powers of the coset shift generator, bit-reversed.
    weights: [F; N],
}

impl<F, const N: usize> Default for CosetMds<F, N>
where
    F: TwoAdicField,
{
    fn default() -> Self {
        let log_n = log2_strict_usize(N);

        // Primitive N-th root of unity and its inverse.
        let root = F::two_adic_generator(log_n);
        let root_inv = root.inverse();

        // Collect N/2 powers and bit-reverse for the Bowers network layout.
        let mut fft_twiddles: Vec<F> = root.powers().collect_n(N / 2);
        let mut ifft_twiddles: Vec<F> = root_inv.powers().collect_n(N / 2);
        reverse_slice_index_bits(&mut fft_twiddles);
        reverse_slice_index_bits(&mut ifft_twiddles);

        // Coset shift weights: generator^0, generator^1, ..., generator^{N-1}, bit-reversed.
        let shift = F::GENERATOR;
        let mut weights: [F; N] = shift.powers().collect_n(N).try_into().unwrap();
        reverse_slice_index_bits(&mut weights);

        Self {
            fft_twiddles,
            ifft_twiddles,
            weights,
        }
    }
}

impl<F: Field, A: Algebra<F>, const N: usize> Permutation<[A; N]> for CosetMds<F, N> {
    fn permute_mut(&self, values: &mut [A; N]) {
        // Step 1: inverse DFT (skip bit-reversal and 1/N rescaling).
        bowers_g_t(values, &self.ifft_twiddles);

        // Step 2: multiply each coefficient by the corresponding coset shift power.
        for (value, weight) in values.iter_mut().zip(self.weights) {
            *value = value.clone() * weight;
        }

        // Step 3: forward DFT on the now bit-reversed, shifted coefficients.
        bowers_g(values, &self.fft_twiddles);
    }
}

impl<F: Field, A: Algebra<F>, const N: usize> MdsPermutation<A, N> for CosetMds<F, N> {}

/// Full Bowers G network (forward DFT on bit-reversed input).
///
/// Applies layers from smallest to largest block size.
#[inline]
fn bowers_g<F: Field, A: Algebra<F>, const N: usize>(values: &mut [A; N], twiddles: &[F]) {
    let log_n = log2_strict_usize(N);
    // Sweep from fine blocks (size 2) to the full array.
    for log_half_block_size in 0..log_n {
        bowers_g_layer(values, log_half_block_size, twiddles);
    }
}

/// Full Bowers G^T network (inverse DFT without 1/N rescaling; output is bit-reversed).
///
/// Applies layers from largest to smallest block size.
#[inline]
fn bowers_g_t<F: Field, A: Algebra<F>, const N: usize>(values: &mut [A; N], twiddles: &[F]) {
    let log_n = log2_strict_usize(N);
    // Sweep from the full array down to fine blocks (size 2).
    for log_half_block_size in (0..log_n).rev() {
        bowers_g_t_layer(values, log_half_block_size, twiddles);
    }
}

#[cfg(test)]
mod tests {
    use core::array;

    use p3_baby_bear::BabyBear;
    use p3_dft::{NaiveDft, TwoAdicSubgroupDft};
    use p3_field::{Field, PrimeCharacteristicRing, TwoAdicField};
    use p3_goldilocks::Goldilocks;
    use p3_symmetric::Permutation;
    use proptest::prelude::*;
    use rand::distr::{Distribution, StandardUniform};
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use crate::coset_mds::CosetMds;

    fn matches_naive_for<F, const N: usize>()
    where
        F: TwoAdicField,
        StandardUniform: Distribution<F>,
    {
        // Generate a random input array with a fixed seed for reproducibility.
        let mut rng = SmallRng::seed_from_u64(1);
        let mut arr: [F; N] = array::from_fn(|_| rng.random());

        // Compute the reference via a naive coset LDE.
        let shift = F::GENERATOR;
        let mut coset_lde_naive = NaiveDft.coset_lde(arr.to_vec(), 0, shift);

        // The Bowers-based implementation skips the 1/N rescaling,
        // so compensate by multiplying the naive result by N.
        let scale = F::from_usize(N);
        coset_lde_naive.iter_mut().for_each(|x| *x *= scale);

        // Apply the permutation under test and compare.
        CosetMds::<F, N>::default().permute_mut(&mut arr);
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
        let mds = CosetMds::<BabyBear, 8>::default();
        let mut zeros = [BabyBear::ZERO; 8];
        mds.permute_mut(&mut zeros);
        assert_eq!(zeros, [BabyBear::ZERO; 8]);
    }

    #[test]
    fn all_zeros_goldilocks() {
        // Same zero-preservation check on a different field.
        let mds = CosetMds::<Goldilocks, 8>::default();
        let mut zeros = [Goldilocks::ZERO; 8];
        mds.permute_mut(&mut zeros);
        assert_eq!(zeros, [Goldilocks::ZERO; 8]);
    }

    fn check_linearity<F, const N: usize>(a: [F; N], b: [F; N])
    where
        F: TwoAdicField,
    {
        let mds = CosetMds::<F, N>::default();

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
        fn coset_mds_linear_bb8(
            a in prop::array::uniform8(arb_babybear()),
            b in prop::array::uniform8(arb_babybear()),
        ) {
            check_linearity::<BabyBear, 8>(a, b);
        }

        #[test]
        fn coset_mds_linear_bb16(
            a in prop::array::uniform16(arb_babybear()),
            b in prop::array::uniform16(arb_babybear()),
        ) {
            check_linearity::<BabyBear, 16>(a, b);
        }

        #[test]
        fn coset_mds_matches_naive_random_bb8(input in prop::array::uniform8(arb_babybear())) {
            // Compute the naive reference, scaled by N to match the un-rescaled Bowers output.
            let shift = BabyBear::GENERATOR;
            let mut coset_lde_naive = NaiveDft.coset_lde(input.to_vec(), 0, shift);
            let scale = BabyBear::from_usize(8);
            coset_lde_naive.iter_mut().for_each(|x| *x *= scale);

            // Apply the permutation under test and compare.
            let mut result = input;
            CosetMds::<BabyBear, 8>::default().permute_mut(&mut result);
            prop_assert_eq!(coset_lde_naive, result);
        }
    }
}
