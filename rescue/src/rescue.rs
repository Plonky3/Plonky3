use crate::inverse_sbox::InverseSboxLayer;
use crate::util::shake256_hash;

use itertools::Itertools;
use num::{BigUint, One};
use num_integer::binomial;
use p3_field::{PrimeField, PrimeField64};
use p3_symmetric::permutation::{ArrayPermutation, CryptographicPermutation, MDSPermutation};
use p3_util::ceil_div_usize;
use rand::distributions::Standard;
use rand::prelude::Distribution;
use rand::Rng;

#[derive(Clone)]
pub struct Rescue<F, MDS, ISL, const WIDTH: usize, const ALPHA: u64>
where
    F: PrimeField,
    MDS: MDSPermutation<F, WIDTH>,
    ISL: InverseSboxLayer<F, WIDTH, ALPHA>,
{
    num_rounds: usize,
    mds: MDS,
    isl: ISL,
    round_constants: Vec<F>,
}

impl<F, MDS, ISL, const WIDTH: usize, const ALPHA: u64> Rescue<F, MDS, ISL, WIDTH, ALPHA>
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
        }
    }

    fn num_rounds(capacity: usize, sec_level: usize) -> usize {
        let rate = WIDTH - capacity;
        let dcon = |n: usize| {
            (0.5 * ((ALPHA - 1) * WIDTH as u64 * (n as u64 - 1)) as f64 + 2.0).floor() as usize
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

    fn sbox_layer(state: &mut [F; WIDTH]) {
        for x in state.iter_mut() {
            *x = x.exp_u64(ALPHA);
        }
    }

    // For a general field, we provide a generic constructor for the round constants.
    pub fn get_round_constants_from_rng<R: Rng>(num_rounds: usize, rng: &mut R) -> Vec<F>
    where
        Standard: Distribution<F>,
    {
        let num_constants = 2 * WIDTH * num_rounds;
        rng.sample_iter(Standard).take(num_constants).collect()
    }
}

impl<F, MDS, ISL, const WIDTH: usize, const ALPHA: u64> Rescue<F, MDS, ISL, WIDTH, ALPHA>
where
    F: PrimeField64,
    MDS: MDSPermutation<F, WIDTH>,
    ISL: InverseSboxLayer<F, WIDTH, ALPHA>,
{
    fn get_round_constants_rescue_prime(
        num_rounds: usize,
        capacity: usize,
        sec_level: usize,
    ) -> Vec<F> {
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

impl<F, MDS, ISL, const WIDTH: usize, const ALPHA: u64> CryptographicPermutation<[F; WIDTH]>
    for Rescue<F, MDS, ISL, WIDTH, ALPHA>
where
    F: PrimeField,
    MDS: MDSPermutation<F, WIDTH>,
    ISL: InverseSboxLayer<F, WIDTH, ALPHA>,
{
    fn permute(&self, state: [F; WIDTH]) -> [F; WIDTH] {
        // Rescue-XLIX permutation

        let mut state = state;

        for round in 0..self.num_rounds {
            // S-box
            Self::sbox_layer(&mut state);

            // MDS
            self.mds.permute_mut(&mut state);

            // Constants
            for j in 0..WIDTH {
                state[j] += self.round_constants[round * WIDTH * 2 + j];
            }

            // Inverse S-box
            self.isl.inverse_sbox_layer(&mut state);

            // MDS
            self.mds.permute_mut(&mut state);

            // Constants
            for j in 0..WIDTH {
                state[j] += self.round_constants[round * WIDTH * 2 + WIDTH + j];
            }
        }

        state
    }
}

impl<F, MDS, ISL, const WIDTH: usize, const ALPHA: u64> ArrayPermutation<F, WIDTH>
    for Rescue<F, MDS, ISL, WIDTH, ALPHA>
where
    F: PrimeField,
    MDS: MDSPermutation<F, WIDTH>,
    ISL: InverseSboxLayer<F, WIDTH, ALPHA>,
{
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use p3_field::AbstractField;
    use p3_mersenne_31::Mersenne31;
    use p3_symmetric::hasher::CryptographicHasher;
    use p3_symmetric::permutation::CryptographicPermutation;
    use p3_symmetric::sponge::PaddingFreeSponge;

    use crate::inverse_sbox::BasicInverseSboxLayer;
    use crate::mds_matrix_naive::{rescue_prime_m31_width_12_mds_matrix, MDSMatrixNaive};
    use crate::rescue::Rescue;

    const WIDTH: usize = 12;
    const ALPHA: u64 = 5;
    type RescuePrimeM31Default =
        Rescue<Mersenne31, MDSMatrixNaive<Mersenne31, WIDTH>, BasicInverseSboxLayer, WIDTH, ALPHA>;

    fn new_rescue_prime_m31_default() -> RescuePrimeM31Default {
        let num_rounds = RescuePrimeM31Default::num_rounds(6, 128);
        let round_constants =
            RescuePrimeM31Default::get_round_constants_rescue_prime(num_rounds, 6, 128);
        let mds = rescue_prime_m31_width_12_mds_matrix();
        let isl = BasicInverseSboxLayer {};

        RescuePrimeM31Default::new(num_rounds, round_constants, mds, isl)
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
            1174355075, 506638036, 1293741855, 669671042, 881673047, 1403310363, 1489659750,
            106483224, 1578796769, 289825640, 498340024, 564347160,
        ],
        [
            1341954293, 1462092714, 1382783160, 288894489, 1768710137, 1938423223, 288009985,
            684142220, 1708749517, 773110691, 916511285, 553593472,
        ],
        [
            868623386, 984305610, 478195671, 1835744746, 2122442506, 495239130, 1519185684,
            1631691838, 1813476755, 1147911813, 2000740064, 986040905,
        ],
    ];

    #[test]
    fn test_rescue_xlix_permutation() {
        let rescue_prime = new_rescue_prime_m31_default();

        for test_run in 0..NUM_TESTS {
            let state: [Mersenne31; WIDTH] = PERMUTATION_INPUTS[test_run]
                .iter()
                .map(|x| Mersenne31::from_canonical_u64(*x))
                .collect_vec()
                .try_into()
                .unwrap();

            let expected: [Mersenne31; WIDTH] = PERMUTATION_OUTPUTS[test_run]
                .iter()
                .map(|x| Mersenne31::from_canonical_u64(*x))
                .collect_vec()
                .try_into()
                .unwrap();

            let actual = rescue_prime.permute(state);
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_rescue_sponge() {
        let rescue_prime = new_rescue_prime_m31_default();
        let rescue_sponge = PaddingFreeSponge::new(rescue_prime);

        let input: [Mersenne31; 6] = [1, 2, 3, 4, 5, 6]
            .iter()
            .map(|x| Mersenne31::from_canonical_u64(*x))
            .collect_vec()
            .try_into()
            .unwrap();

        let expected: [Mersenne31; 6] = [
            599387515, 345626813, 50230127, 538251572, 279746862, 2080222279,
        ]
        .iter()
        .map(|x| Mersenne31::from_canonical_u64(*x))
        .collect_vec()
        .try_into()
        .unwrap();

        let actual = rescue_sponge.hash_iter(input);
        assert_eq!(actual, expected);
    }
}
