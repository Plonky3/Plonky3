use alloc::format;
use alloc::vec::Vec;

use itertools::Itertools;
use num::{BigUint, One};
use num_integer::binomial;
use p3_field::{AbstractField, PrimeField, PrimeField64};
use p3_mds::MdsPermutation;
use p3_symmetric::{CryptographicPermutation, Permutation};
use rand::distributions::Standard;
use rand::prelude::Distribution;
use rand::Rng;

use crate::sbox::SboxLayers;
use crate::util::shake256_hash;

/// The Rescue-XLIX permutation.
#[derive(Clone, Debug)]
pub struct Rescue<F, Mds, Sbox, const WIDTH: usize> {
    num_rounds: usize,
    mds: Mds,
    sbox: Sbox,
    round_constants: Vec<F>,
}

impl<F, Mds, Sbox, const WIDTH: usize> Rescue<F, Mds, Sbox, WIDTH>
where
    F: PrimeField,
{
    pub fn new(num_rounds: usize, round_constants: Vec<F>, mds: Mds, sbox: Sbox) -> Self {
        Self {
            num_rounds,
            mds,
            sbox,
            round_constants,
        }
    }

    fn num_rounds(capacity: usize, sec_level: usize, alpha: u64) -> usize {
        let rate = WIDTH - capacity;
        let dcon = |n: usize| {
            (0.5 * ((alpha - 1) * WIDTH as u64 * (n as u64 - 1)) as f64 + 2.0).floor() as usize
        };
        let v = |n: usize| WIDTH * (n - 1) + rate;
        let target = BigUint::one() << sec_level;

        let is_sufficient = |l1: &usize| {
            let n = BigUint::from(v(*l1) + dcon(*l1));
            let k = BigUint::from(v(*l1));
            let bin = binomial(n, k);
            &bin * &bin > target
        };
        let l1 = (1..25).find(is_sufficient).unwrap();
        (l1.max(5) as f32 * 1.5).ceil() as usize
    }

    // For a general field, we provide a generic constructor for the round constants.
    pub fn get_round_constants_from_rng<R: Rng>(num_rounds: usize, rng: &mut R) -> Vec<F>
    where
        Standard: Distribution<F>,
    {
        let num_constants = 2 * WIDTH * num_rounds;
        rng.sample_iter(Standard).take(num_constants).collect()
    }

    fn get_round_constants_rescue_prime(
        num_rounds: usize,
        capacity: usize,
        sec_level: usize,
    ) -> Vec<F>
    where
        F: PrimeField64,
    {
        let num_constants = 2 * WIDTH * num_rounds;
        let bytes_per_constant = F::bits().div_ceil(8) + 1;
        let num_bytes = bytes_per_constant * num_constants;

        let seed_string = format!(
            "Rescue-XLIX({},{},{},{})",
            F::ORDER_U64,
            WIDTH,
            capacity,
            sec_level,
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

impl<AF, Mds, Sbox, const WIDTH: usize> Permutation<[AF; WIDTH]> for Rescue<AF::F, Mds, Sbox, WIDTH>
where
    AF: AbstractField,
    AF::F: PrimeField,
    Mds: MdsPermutation<AF, WIDTH>,
    Sbox: SboxLayers<AF, WIDTH>,
{
    fn permute_mut(&self, state: &mut [AF; WIDTH]) {
        for round in 0..self.num_rounds {
            // S-box
            self.sbox.sbox_layer(state);

            // MDS
            self.mds.permute_mut(state);

            // Constants
            for (state_item, &round_constant) in state
                .iter_mut()
                .zip(&self.round_constants[round * WIDTH * 2..])
            {
                *state_item += AF::from_f(round_constant);
            }

            // Inverse S-box
            self.sbox.inverse_sbox_layer(state);

            // MDS
            self.mds.permute_mut(state);

            // Constants
            for (state_item, &round_constant) in state
                .iter_mut()
                .zip(&self.round_constants[round * WIDTH * 2 + WIDTH..])
            {
                *state_item += AF::from_f(round_constant);
            }
        }
    }
}

impl<AF, Mds, Sbox, const WIDTH: usize> CryptographicPermutation<[AF; WIDTH]>
    for Rescue<AF::F, Mds, Sbox, WIDTH>
where
    AF: AbstractField,
    AF::F: PrimeField,
    Mds: MdsPermutation<AF, WIDTH>,
    Sbox: SboxLayers<AF, WIDTH>,
{
}

#[cfg(test)]
mod tests {
    use p3_field::AbstractField;
    use p3_mersenne_31::{MdsMatrixMersenne31, Mersenne31};
    use p3_symmetric::{CryptographicHasher, PaddingFreeSponge, Permutation};

    use crate::rescue::Rescue;
    use crate::sbox::BasicSboxLayer;

    const WIDTH: usize = 12;
    const ALPHA: u64 = 5;
    type RescuePrimeM31Default =
        Rescue<Mersenne31, MdsMatrixMersenne31, BasicSboxLayer<Mersenne31>, WIDTH>;

    fn new_rescue_prime_m31_default() -> RescuePrimeM31Default {
        let num_rounds = RescuePrimeM31Default::num_rounds(6, 128, ALPHA);
        let round_constants =
            RescuePrimeM31Default::get_round_constants_rescue_prime(num_rounds, 6, 128);
        let mds = MdsMatrixMersenne31 {};
        let sbox = BasicSboxLayer::for_alpha(ALPHA);

        RescuePrimeM31Default::new(num_rounds, round_constants, mds, sbox)
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

    // Generated using the rescue_XLIX_permutation function of
    // https://github.com/KULeuven-COSIC/Marvellous/blob/master/rescue_prime.sage
    const PERMUTATION_OUTPUTS: [[u64; WIDTH]; NUM_TESTS] = [
        [
            1415867641, 1662872101, 1070605392, 450708029, 1752877321, 144003686, 623713963,
            13124252, 1719755748, 1164265443, 1031746503, 656034061,
        ],
        [
            745601819, 399135364, 1705560828, 1125372012, 2039222953, 1144119753, 1606567447,
            1152559313, 1762793605, 424623198, 651056006, 1227670410,
        ],
        [
            277798368, 1055656487, 366843969, 917136738, 1286790161, 1840518903, 161567750,
            974017246, 1102241644, 633393178, 896102012, 1791619348,
        ],
    ];

    #[test]
    fn test_rescue_xlix_permutation() {
        let rescue_prime = new_rescue_prime_m31_default();

        for test_run in 0..NUM_TESTS {
            let state: [Mersenne31; WIDTH] =
                PERMUTATION_INPUTS[test_run].map(Mersenne31::from_canonical_u64);

            let expected: [Mersenne31; WIDTH] =
                PERMUTATION_OUTPUTS[test_run].map(Mersenne31::from_canonical_u64);

            let actual = rescue_prime.permute(state);
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_rescue_sponge() {
        let rescue_prime = new_rescue_prime_m31_default();
        let rescue_sponge = PaddingFreeSponge::<_, WIDTH, 8, 6>::new(rescue_prime);

        let input: [Mersenne31; 6] = [1, 2, 3, 4, 5, 6].map(Mersenne31::from_canonical_u64);

        let expected: [Mersenne31; 6] = [
            2055426095, 968531194, 1592692524, 136824376, 175318858, 1160805485,
        ]
        .map(Mersenne31::from_canonical_u64);

        let actual = rescue_sponge.hash_iter(input);
        assert_eq!(actual, expected);
    }
}
