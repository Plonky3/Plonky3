use alloc::vec::Vec;

use p3_field::{Algebra, Field, Powers, TwoAdicField};
use p3_symmetric::Permutation;
use p3_util::{log2_strict_usize, reverse_slice_index_bits};

use crate::MdsPermutation;
use crate::butterflies::{bowers_g_layer, bowers_g_t_layer_integrated};

/// Like `CosetMds`, with a few differences:
/// - (Bit reversed, a la Bowers) DIF + DIT rather than DIT + DIF
/// - We skip bit reversals of the inputs and outputs
/// - We don't weight by `1/N`, since this doesn't affect the MDS property
/// - We integrate the coset shifts into the DIF's twiddle factors
#[derive(Clone, Debug)]
pub struct IntegratedCosetMds<F, const N: usize> {
    ifft_twiddles: Vec<F>,
    fft_twiddles: Vec<Vec<F>>,
}

impl<F: TwoAdicField, const N: usize> Default for IntegratedCosetMds<F, N> {
    fn default() -> Self {
        let log_n = log2_strict_usize(N);
        let root = F::two_adic_generator(log_n);
        let root_inv = root.inverse();
        let coset_shift = F::GENERATOR;

        let mut ifft_twiddles = root_inv.powers().collect_n(N / 2);
        reverse_slice_index_bits(&mut ifft_twiddles);

        let fft_twiddles: Vec<Vec<F>> = (0..log_n)
            .map(|layer| {
                let shift_power = coset_shift.exp_power_of_2(layer);
                let powers = Powers {
                    base: root.exp_power_of_2(layer),
                    current: shift_power,
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
    fn permute(&self, mut input: [A; N]) -> [A; N] {
        self.permute_mut(&mut input);
        input
    }

    fn permute_mut(&self, values: &mut [A; N]) {
        let log_n = log2_strict_usize(N);

        // Bit-reversed DIF, aka Bowers G
        for layer in 0..log_n {
            bowers_g_layer(values, layer, &self.ifft_twiddles);
        }

        // Bit-reversed DIT, aka Bowers G^T
        for layer in (0..log_n).rev() {
            bowers_g_t_layer_integrated(values, layer, &self.fft_twiddles[layer]);
        }
    }
}

impl<F: Field, A: Algebra<F>, const N: usize> MdsPermutation<A, N> for IntegratedCosetMds<F, N> {}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_dft::{NaiveDft, TwoAdicSubgroupDft};
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_symmetric::Permutation;
    use p3_util::reverse_slice_index_bits;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use crate::integrated_coset_mds::IntegratedCosetMds;

    type F = BabyBear;
    const N: usize = 16;

    #[test]
    fn matches_naive() {
        let mut rng = SmallRng::seed_from_u64(1);
        let mut arr: [F; N] = rng.random();

        let mut arr_rev = arr.to_vec();
        reverse_slice_index_bits(&mut arr_rev);

        let shift = F::GENERATOR;
        let mut coset_lde_naive = NaiveDft.coset_lde(arr_rev, 0, shift);
        reverse_slice_index_bits(&mut coset_lde_naive);
        coset_lde_naive
            .iter_mut()
            .for_each(|x| *x *= F::from_u8(N as u8));
        IntegratedCosetMds::default().permute_mut(&mut arr);
        assert_eq!(coset_lde_naive, arr);
    }
}
