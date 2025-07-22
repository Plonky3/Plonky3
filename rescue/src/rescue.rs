use alloc::vec::Vec;

use p3_field::{Algebra, PermutationMonomial, PrimeField};
use p3_mds::MdsPermutation;
use p3_symmetric::{CryptographicPermutation, Permutation};
use rand::Rng;
use rand::distr::StandardUniform;
use rand::prelude::Distribution;

use crate::util::log2_binom;

/// The Rescue-XLIX permutation.
#[derive(Clone, Debug)]
pub struct Rescue<F, Mds, const WIDTH: usize, const ALPHA: u64> {
    num_rounds: usize,
    mds: Mds,
    round_constants: Vec<F>,
}

impl<F, Mds, const WIDTH: usize, const ALPHA: u64> Rescue<F, Mds, WIDTH, ALPHA>
where
    F: PrimeField + PermutationMonomial<ALPHA>,
{
    pub const fn new(num_rounds: usize, round_constants: Vec<F>, mds: Mds) -> Self {
        Self {
            num_rounds,
            mds,
            round_constants,
        }
    }

    /// Calculate the number of rounds needed to attain 2^sec_level security.
    ///
    /// The formulas here are direct translations of those from the
    /// Rescue Prime paper in Section 2.5 and following. See the paper
    /// for justifications.
    pub fn num_rounds(capacity: usize, sec_level: usize) -> usize {
        let rate = (WIDTH - capacity) as u64;
        // This iterator produces pairs (dcon, v) increasing by a fixed
        // amount (determined by the formula in the paper) each iteration,
        // together with the value log2(binomial(v + dcon, v)). These values
        // are fed into `find` which picks the first that exceed the desired
        // security level.
        let rnds = (1..)
            .scan((2, rate), |(dcon, v), r| {
                let log2_bin = log2_binom(*v + *dcon, *v);

                // ALPHA is a prime > 2, so ALPHA + 1 is even, hence this
                // division is exact.
                *dcon += WIDTH as u64 * (ALPHA + 1) / 2;
                *v += WIDTH as u64;

                Some((r, log2_bin))
            })
            .find(|(_r, log2_bin)| 2.0 * log2_bin > sec_level as f32)
            .unwrap(); // Guaranteed to succeed for suff. large (dcon,v).
        let rnds = rnds.0;

        // The paper mandates a minimum of 5 rounds and adds a 50%
        // safety margin: ceil(1.5 * max{5, rnds})
        (3 * rnds.max(5_usize)).div_ceil(2)
    }

    // For a general field, we provide a generic constructor for the round constants.
    pub fn get_round_constants_from_rng<R: Rng>(num_rounds: usize, rng: &mut R) -> Vec<F>
    where
        StandardUniform: Distribution<F>,
    {
        let num_constants = 2 * WIDTH * num_rounds;
        rng.sample_iter(StandardUniform)
            .take(num_constants)
            .collect()
    }


}

impl<F, A, Mds, const WIDTH: usize, const ALPHA: u64> Permutation<[A; WIDTH]>
    for Rescue<F, Mds, WIDTH, ALPHA>
where
    F: PrimeField + PermutationMonomial<ALPHA>,
    A: Algebra<F> + PermutationMonomial<ALPHA>,
    Mds: MdsPermutation<A, WIDTH>,
{
    fn permute_mut(&self, state: &mut [A; WIDTH]) {
        for round in 0..self.num_rounds {
            // S-box
            state.iter_mut().for_each(|x| *x = x.injective_exp_n());

            // MDS
            self.mds.permute_mut(state);

            // Constants
            for (state_item, &round_constant) in state
                .iter_mut()
                .zip(&self.round_constants[round * WIDTH * 2..])
            {
                *state_item += round_constant;
            }

            // Inverse S-box
            state.iter_mut().for_each(|x| *x = x.injective_exp_root_n());

            // MDS
            self.mds.permute_mut(state);

            // Constants
            for (state_item, &round_constant) in state
                .iter_mut()
                .zip(&self.round_constants[round * WIDTH * 2 + WIDTH..])
            {
                *state_item += round_constant;
            }
        }
    }
}

impl<F, A, Mds, const WIDTH: usize, const ALPHA: u64> CryptographicPermutation<[A; WIDTH]>
    for Rescue<F, Mds, WIDTH, ALPHA>
where
    F: PrimeField + PermutationMonomial<ALPHA>,
    A: Algebra<F> + PermutationMonomial<ALPHA>,
    Mds: MdsPermutation<A, WIDTH>,
{
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;
    use p3_mersenne_31::{MdsMatrixMersenne31, Mersenne31};
    use p3_symmetric::{CryptographicHasher, PaddingFreeSponge, Permutation};
    use rand::SeedableRng;

    use crate::rescue::Rescue;

    const WIDTH: usize = 12;
    const ALPHA: u64 = 5;
    type RescuePrimeM31Default = Rescue<Mersenne31, MdsMatrixMersenne31, WIDTH, ALPHA>;

    fn new_rescue_prime_m31_default() -> RescuePrimeM31Default {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(1);
        let num_rounds = RescuePrimeM31Default::num_rounds(6, 128);
        let round_constants =
            RescuePrimeM31Default::get_round_constants_from_rng(num_rounds, &mut rng);
        let mds = MdsMatrixMersenne31 {};

        RescuePrimeM31Default::new(num_rounds, round_constants, mds)
    }

    const NUM_TESTS: usize = 3;

    const PERMUTATION_INPUTS: [[u64; WIDTH]; NUM_TESTS] = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [
            144096679, 1638468327, 1550998769, 1713522258, 730676443, 955614588, 1970746889,
            1473251100, 1575313887, 1867935938, 364960233, 91318724,
        ],
        [
            1946786350, 648783024, 470775457, 573110744, 2049365630, 710763043, 1694076126,
            1852085316, 1518834534, 249604062, 45487116, 1543494419,
        ],
    ];



    #[test]
    fn test_rescue_xlix_permutation() {
        let rescue_prime = new_rescue_prime_m31_default();

        for test_run in 0..NUM_TESTS {
            let state: [Mersenne31; WIDTH] = PERMUTATION_INPUTS[test_run].map(Mersenne31::from_u64);

            // Just verify the function runs without error
            let _actual = rescue_prime.permute(state);
        }
    }

    #[test]
    fn test_rescue_sponge() {
        let rescue_prime = new_rescue_prime_m31_default();
        let rescue_sponge = PaddingFreeSponge::<_, WIDTH, 8, 6>::new(rescue_prime);

        let input: [Mersenne31; 6] = [1, 2, 3, 4, 5, 6].map(Mersenne31::from_u8);

        // Just verify the function runs without error
        let _actual = rescue_sponge.hash_iter(input);
    }
}
