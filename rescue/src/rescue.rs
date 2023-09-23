use itertools::Itertools;
use num::{BigUint, One};
use num_integer::binomial;
use p3_field::{PrimeField, PrimeField64};
use p3_mds::MdsPermutation;
use p3_symmetric::permutation::{ArrayPermutation, CryptographicPermutation};
use p3_util::ceil_div_usize;
use rand::distributions::Standard;
use rand::prelude::Distribution;
use rand::Rng;

use crate::inverse_sbox::InverseSboxLayer;
use crate::util::{get_inverse, shake256_hash};

#[derive(Clone)]
pub struct Rescue<F, Mds, Isl, const WIDTH: usize, const ALPHA: u64>
where
    F: PrimeField,
    Mds: MdsPermutation<F, WIDTH>,
    Isl: InverseSboxLayer<F, WIDTH, ALPHA>,
{
    num_rounds: usize,
    mds: Mds,
    isl: Isl,
    round_constants: Vec<F>,
}

impl<F, Mds, Isl, const WIDTH: usize, const ALPHA: u64> Rescue<F, Mds, Isl, WIDTH, ALPHA>
where
    F: PrimeField,
    Mds: MdsPermutation<F, WIDTH>,
    Isl: InverseSboxLayer<F, WIDTH, ALPHA>,
{
    pub fn new(num_rounds: usize, round_constants: Vec<F>, mds: Mds, isl: Isl) -> Self {
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

impl<F, Mds, Isl, const WIDTH: usize, const ALPHA: u64> Rescue<F, Mds, Isl, WIDTH, ALPHA>
where
    F: PrimeField64,
    Mds: MdsPermutation<F, WIDTH>,
    Isl: InverseSboxLayer<F, WIDTH, ALPHA>,
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

impl<F, Mds, Isl, const WIDTH: usize, const ALPHA: u64> CryptographicPermutation<[F; WIDTH]>
    for Rescue<F, Mds, Isl, WIDTH, ALPHA>
where
    F: PrimeField64,
    Mds: MdsPermutation<F, WIDTH>,
    Isl: InverseSboxLayer<F, WIDTH, ALPHA>,
{
    fn permute(&self, state: [F; WIDTH]) -> [F; WIDTH] {
        // Rescue-XLIX permutation

        let mut state = state;

        // It might be worth adding this to the inputs as opposed to computing it every time.
        let alpha_inv = get_inverse::<F>(ALPHA);

        for round in 0..self.num_rounds {
            // S-box
            Self::sbox_layer(&mut state);

            // MDS
            self.mds.permute_mut(&mut state);

            // Constants
            for (state_item, &round_constant) in itertools::izip!(
                state.iter_mut(),
                self.round_constants[round * WIDTH * 2..].iter()
            ) {
                *state_item += round_constant;
            }

            // Inverse S-box
            self.isl.inverse_sbox_layer(&mut state, alpha_inv);

            // MDS
            self.mds.permute_mut(&mut state);

            // Constants
            for (state_item, &round_constant) in itertools::izip!(
                state.iter_mut(),
                self.round_constants[round * WIDTH * 2 + WIDTH..].iter()
            ) {
                *state_item += round_constant;
            }
        }

        state
    }
}

impl<F, Mds, Isl, const WIDTH: usize, const ALPHA: u64> ArrayPermutation<F, WIDTH>
    for Rescue<F, Mds, Isl, WIDTH, ALPHA>
where
    F: PrimeField64,
    Mds: MdsPermutation<F, WIDTH>,
    Isl: InverseSboxLayer<F, WIDTH, ALPHA>,
{
}

#[cfg(test)]
mod tests {
    use p3_field::AbstractField;
    use p3_mds::mersenne31::MdsMatrixMersenne31;
    use p3_mersenne_31::Mersenne31;
    use p3_symmetric::hasher::CryptographicHasher;
    use p3_symmetric::permutation::CryptographicPermutation;
    use p3_symmetric::sponge::PaddingFreeSponge;

    use crate::inverse_sbox::BasicInverseSboxLayer;
    use crate::rescue::Rescue;

    const WIDTH: usize = 12;
    const ALPHA: u64 = 5;
    type RescuePrimeM31Default =
        Rescue<Mersenne31, MdsMatrixMersenne31, BasicInverseSboxLayer, WIDTH, ALPHA>;

    fn new_rescue_prime_m31_default() -> RescuePrimeM31Default {
        let num_rounds = RescuePrimeM31Default::num_rounds(6, 128);
        let round_constants =
            RescuePrimeM31Default::get_round_constants_rescue_prime(num_rounds, 6, 128);
        let mds = MdsMatrixMersenne31 {};
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
            983158113, 88736227, 182376113, 380581876, 1054929865, 873254619, 1742172525,
            1018880997, 1922857524, 2128461101, 1878468735, 736900567,
        ],
        [
            504747180, 1708979401, 1023327691, 414948293, 1811202621, 427591394, 666516466,
            1900855073, 1511950466, 346735768, 708718627, 2070146754,
        ],
        [
            2043076197, 1832583290, 59074227, 991951621, 1166633601, 629305333, 1869192382,
            1355209324, 1919016607, 175801753, 279984593, 2086613859,
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
        let rescue_sponge = PaddingFreeSponge::<_, _, WIDTH, 8, 6>::new(rescue_prime);

        let input: [Mersenne31; 6] = [1, 2, 3, 4, 5, 6].map(Mersenne31::from_canonical_u64);

        let expected: [Mersenne31; 6] = [
            337439389, 568168673, 983336666, 1144682541, 1342961449, 386074361,
        ]
        .map(Mersenne31::from_canonical_u64);

        let actual = rescue_sponge.hash_iter(input);
        assert_eq!(actual, expected);
    }
}
