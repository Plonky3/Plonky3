use alloc::format;
use alloc::vec::Vec;

use itertools::Itertools;
use num::{BigUint, One};
use num_integer::binomial;
use p3_field::{AbstractField, PrimeField, PrimeField64};
use p3_mds::MdsPermutation;
use p3_symmetric::{CryptographicPermutation, Permutation};
use p3_util::ceil_div_usize;
use rand::distributions::Standard;
use rand::prelude::Distribution;
use rand::Rng;

use crate::sbox::SboxLayers;
use crate::util::shake256_hash;

/// The Rescue-XLIX permutation.
#[derive(Clone)]
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
        let bytes_per_constant = ceil_div_usize(F::bits(), 8) + 1;
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
    use p3_mds::mersenne31::MdsMatrixMersenne31;
    use p3_mersenne_31::Mersenne31;
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
            144_096_679, 1_638_468_327, 1_550_998_769, 1_713_522_258, 730_676_443, 955_614_588, 1_970_746_889,
            1_473_251_100, 1_575_313_887, 1_867_935_938, 364_960_233, 91_318_724,
        ],
        [
            1_946_786_350, 648_783_024, 470_775_457, 573_110_744, 2_049_365_630, 710_763_043, 1_694_076_126,
            1_852_085_316, 1_518_834_534, 249_604_062, 45_487_116, 1_543_494_419,
        ],
    ];

    // Generated using the rescue_XLIX_permutation function of
    // https://github.com/KULeuven-COSIC/Marvellous/blob/master/rescue_prime.sage
    const PERMUTATION_OUTPUTS: [[u64; WIDTH]; NUM_TESTS] = [
        [
            983_158_113, 88_736_227, 182_376_113, 380_581_876, 1_054_929_865, 873_254_619, 1_742_172_525,
            1_018_880_997, 1_922_857_524, 2_128_461_101, 1_878_468_735, 736_900_567,
        ],
        [
            504_747_180, 1_708_979_401, 1_023_327_691, 414_948_293, 1_811_202_621, 427_591_394, 666_516_466,
            1_900_855_073, 1_511_950_466, 346_735_768, 708_718_627, 2_070_146_754,
        ],
        [
            2_043_076_197, 1_832_583_290, 59_074_227, 991_951_621, 1_166_633_601, 629_305_333, 1_869_192_382,
            1_355_209_324, 1_919_016_607, 175_801_753, 279_984_593, 2_086_613_859,
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
            337_439_389, 568_168_673, 983_336_666, 1_144_682_541, 1_342_961_449, 386_074_361,
        ]
        .map(Mersenne31::from_canonical_u64);

        let actual = rescue_sponge.hash_iter(input);
        assert_eq!(actual, expected);
    }
}
