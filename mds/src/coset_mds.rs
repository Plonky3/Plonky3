use alloc::vec::Vec;

use p3_field::{Algebra, Field, TwoAdicField};
use p3_symmetric::Permutation;
use p3_util::{log2_strict_usize, reverse_slice_index_bits};

use crate::MdsPermutation;
use crate::butterflies::{bowers_g_layer, bowers_g_t_layer};

/// A Reed-Solomon based MDS permutation.
///
/// An MDS permutation which works by interpreting the input as evaluations of a polynomial over a
/// power-of-two subgroup, and computing evaluations over a coset of that subgroup. This can be
/// viewed as returning the parity elements of a systematic Reed-Solomon code. Since Reed-Solomon
/// codes are MDS, this is an MDS permutation.
#[derive(Clone, Debug)]
pub struct CosetMds<F, const N: usize> {
    fft_twiddles: Vec<F>,
    ifft_twiddles: Vec<F>,
    weights: [F; N],
}

impl<F, const N: usize> Default for CosetMds<F, N>
where
    F: TwoAdicField,
{
    fn default() -> Self {
        let log_n = log2_strict_usize(N);

        let root = F::two_adic_generator(log_n);
        let root_inv = root.inverse();
        let mut fft_twiddles: Vec<F> = root.powers().collect_n(N / 2);
        let mut ifft_twiddles: Vec<F> = root_inv.powers().collect_n(N / 2);
        reverse_slice_index_bits(&mut fft_twiddles);
        reverse_slice_index_bits(&mut ifft_twiddles);

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

impl<F: TwoAdicField, A: Algebra<F>, const N: usize> Permutation<[A; N]> for CosetMds<F, N> {
    fn permute_mut(&self, values: &mut [A; N]) {
        // Inverse DFT, except we skip bit reversal and rescaling by 1/N.
        bowers_g_t(values, &self.ifft_twiddles);

        // Multiply by powers of the coset shift (see default coset LDE impl for an explanation)
        for (value, weight) in values.iter_mut().zip(self.weights) {
            *value = value.clone() * weight;
        }

        // DFT, assuming bit-reversed input.
        bowers_g(values, &self.fft_twiddles);
    }
}

impl<F: TwoAdicField, A: Algebra<F>, const N: usize> MdsPermutation<A, N> for CosetMds<F, N> {}

/// Executes the Bowers G network. This is like a DFT, except it assumes the input is in
/// bit-reversed order.
#[inline]
fn bowers_g<F: Field, A: Algebra<F>, const N: usize>(values: &mut [A; N], twiddles: &[F]) {
    let log_n = log2_strict_usize(N);
    for log_half_block_size in 0..log_n {
        bowers_g_layer(values, log_half_block_size, twiddles);
    }
}

/// Executes the Bowers G^T network. This is like an inverse DFT, except we skip rescaling by
/// `1/N`, and the output is bit-reversed.
#[inline]
fn bowers_g_t<F: Field, A: Algebra<F>, const N: usize>(values: &mut [A; N], twiddles: &[F]) {
    let log_n = log2_strict_usize(N);
    for log_half_block_size in (0..log_n).rev() {
        bowers_g_t_layer(values, log_half_block_size, twiddles);
    }
}

#[cfg(test)]
mod tests {
    use core::array;

    use p3_baby_bear::BabyBear;
    use p3_dft::{NaiveDft, TwoAdicSubgroupDft};
    use p3_field::TwoAdicField;
    use p3_goldilocks::Goldilocks;
    use p3_symmetric::Permutation;
    use rand::distr::{Distribution, StandardUniform};
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use crate::coset_mds::CosetMds;

    fn matches_naive_for<F, const N: usize>()
    where
        F: TwoAdicField,
        StandardUniform: Distribution<F>,
    {
        let mut rng = SmallRng::seed_from_u64(1);
        let mut arr: [F; N] = array::from_fn(|_| rng.random());

        let shift = F::GENERATOR;
        let mut coset_lde_naive = NaiveDft.coset_lde(arr.to_vec(), 0, shift);

        let scale = F::from_usize(N);
        coset_lde_naive.iter_mut().for_each(|x| *x *= scale);

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
}
