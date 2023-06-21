use crate::inverse_sbox::{BasicInverseSboxLayer, InverseSboxLayer};
use crate::mds_matrix_naive::{rescue_prime_m31_width_12_mds_matrix, MDSMatrixNaive};
use crate::util::{binomial, shake256_hash};

use ethereum_types::U256;
use itertools::Itertools;
use p3_field::{PrimeField, PrimeField64};
use p3_mersenne_31::Mersenne31;
use p3_symmetric::permutation::{CryptographicPermutation, MDSPermutation};
use p3_util::ceil_div_usize;
use rand::distributions::Standard;
use rand::prelude::Distribution;
use rand::Rng;
use std::marker::PhantomData;

#[derive(Clone)]
pub struct Rescue<
    F,
    MDS,
    ISL,
    const WIDTH: usize,
    const CAPACITY: usize,
    const ALPHA: u64,
    const SEC_LEVEL: usize,
> where
    F: PrimeField,
    MDS: MDSPermutation<F, WIDTH>,
    ISL: InverseSboxLayer<F, WIDTH, ALPHA>,
{
    num_rounds: usize,
    mds: MDS,
    isl: ISL,
    round_constants: Vec<F>,

    _phantom_f: PhantomData<F>,
}

impl<
        F,
        MDS,
        ISL,
        const WIDTH: usize,
        const CAPACITY: usize,
        const ALPHA: u64,
        const SEC_LEVEL: usize,
    > Rescue<F, MDS, ISL, WIDTH, CAPACITY, ALPHA, SEC_LEVEL>
where
    F: PrimeField,
    MDS: MDSPermutation<F, WIDTH>,
    ISL: InverseSboxLayer<F, WIDTH, ALPHA>,
{
    pub fn new(num_rounds: usize, round_constants: Vec<F>, mds: MDS, isl: ISL) -> Self {
        Self {
            num_rounds,
            mds,
            isl,
            round_constants,
            _phantom_f: PhantomData,
        }
    }

    fn num_rounds() -> usize {
        let rate = WIDTH - CAPACITY;
        let dcon = |n: usize| {
            (0.5 * ((ALPHA - 1) * WIDTH as u64 * (n as u64 - 1)) as f64 + 2.0).floor() as usize
        };
        let v = |n: usize| WIDTH * (n - 1) + rate;
        let target = U256::one() << SEC_LEVEL;

        let is_sufficient = |l1: &usize| {
            let bin = binomial(v(*l1) + dcon(*l1), v(*l1));
            bin * bin > target
        };
        (1..25).find(is_sufficient).unwrap()
    }

    fn sbox_layer(state: &mut [F; WIDTH]) {
        for x in state.iter_mut() {
            *x = x.exp_u64(ALPHA);
        }
    }

    // For a general field, we provide a generic constructor for the round constants.
    pub(crate) fn get_round_constants_from_rng<R: Rng>(num_rounds: usize, rng: &mut R) -> Vec<F>
    where
        Standard: Distribution<F>,
    {
        let num_constants = 2 * WIDTH * num_rounds;
        rng.sample_iter(Standard).take(num_constants).collect()
    }
}

impl<
        F,
        MDS,
        ISL,
        const WIDTH: usize,
        const CAPACITY: usize,
        const ALPHA: u64,
        const SEC_LEVEL: usize,
    > Rescue<F, MDS, ISL, WIDTH, CAPACITY, ALPHA, SEC_LEVEL>
where
    F: PrimeField64,
    MDS: MDSPermutation<F, WIDTH>,
    ISL: InverseSboxLayer<F, WIDTH, ALPHA>,
{
    fn get_round_constants_rescue_prime(num_rounds: usize) -> Vec<F> {
        let num_constants = 2 * WIDTH * num_rounds;
        let bytes_per_constant = ceil_div_usize(F::bits(), 8) + 1;
        let num_bytes = bytes_per_constant * num_constants;

        let seed_string = format!(
            "Rescue-XLIX({},{},{},{}",
            F::ORDER_U64,
            WIDTH,
            CAPACITY,
            SEC_LEVEL,
        );
        let byte_string = shake256_hash(seed_string.as_bytes(), num_bytes);

        byte_string
            .iter()
            .chunks(bytes_per_constant)
            .into_iter()
            .map(|chunk| {
                let integer = chunk
                    .collect_vec()
                    .iter()
                    .rev()
                    .fold(0, |acc, &byte| (acc << 8) + *byte as u64);
                F::from_canonical_u64(integer % F::ORDER_U64)
            })
            .collect()
    }
}

impl<
        F,
        MDS,
        ISL,
        const WIDTH: usize,
        const CAPACITY: usize,
        const ALPHA: u64,
        const SEC_LEVEL: usize,
    > CryptographicPermutation<[F; WIDTH]>
    for Rescue<F, MDS, ISL, WIDTH, CAPACITY, ALPHA, SEC_LEVEL>
where
    F: PrimeField,
    MDS: MDSPermutation<F, WIDTH>,
    ISL: InverseSboxLayer<F, WIDTH, ALPHA>,
{
    fn permute(&self, state: [F; WIDTH]) -> [F; WIDTH] {
        // Rescue-XLIX permutation

        let mut state = state;

        for round in 0..self.num_rounds {
            println!("round {}", round);

            dbg!(state.clone());

            // S-box
            Self::sbox_layer(&mut state);

            println!("AFTER S-BOX");
            dbg!(state.clone());

            // MDS
            self.mds.permute_mut(&mut state);

            println!("AFTER MDS");
            dbg!(state.clone());

            // Constants
            for j in 0..WIDTH {
                state[j] += self.round_constants[round * WIDTH * 2 + j];
            }

            println!("AFTER CONSTANTS");
            dbg!(state.clone());

            // Inverse S-box
            self.isl.inverse_sbox_layer(&mut state);

            println!("AFTER INVERSE S-BOX");
            dbg!(state.clone());

            // MDS
            self.mds.permute_mut(&mut state);

            println!("AFTER MDS");
            dbg!(state.clone());

            // Constants
            for j in 0..WIDTH {
                state[j] += self.round_constants[round * WIDTH * 2 + WIDTH + j];
            }

            println!("AFTER CONSTANTS");
            dbg!(state.clone());
        }

        state
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use p3_field::PrimeField;
    use p3_mersenne_31::Mersenne31;
    use p3_symmetric::permutation::CryptographicPermutation;

    use crate::inverse_sbox::BasicInverseSboxLayer;
    use crate::mds_matrix_naive::{rescue_prime_m31_width_12_mds_matrix, MDSMatrixNaive};
    use crate::rescue::Rescue;

    type RescuePrimeM31Default =
        Rescue<Mersenne31, MDSMatrixNaive<Mersenne31, 12>, BasicInverseSboxLayer, 12, 6, 5, 128>;

    fn new_rescue_prime_m31_default() -> RescuePrimeM31Default {
        let num_rounds = RescuePrimeM31Default::num_rounds();
        let round_constants = RescuePrimeM31Default::get_round_constants_rescue_prime(num_rounds);
        let mds = rescue_prime_m31_width_12_mds_matrix();
        let isl = BasicInverseSboxLayer {};

        RescuePrimeM31Default::new(num_rounds, round_constants, mds, isl)
    }

    #[test]
    fn test_rescue_prime_m31_default() {
        let rescue_prime = new_rescue_prime_m31_default();

        let state: [Mersenne31; 12] = (0..12)
            .map(|i| Mersenne31::from_canonical_u8(i))
            .collect_vec()
            .try_into()
            .unwrap();

        let expected: [Mersenne31; 12] = [
            1174355075, 506638036, 1293741855, 669671042, 881673047, 1403310363, 1489659750,
            106483224, 1578796769, 289825640, 498340024, 564347160,
        ]
        .iter()
        .map(|x| Mersenne31::from_canonical_u64(*x))
        .collect_vec()
        .try_into()
        .unwrap();

        let actual = rescue_prime.permute(state);
        assert_eq!(actual, expected);
    }
}
