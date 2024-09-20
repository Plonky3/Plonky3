/*!
 * Implementation of Poseidon2, see: https://eprint.iacr.org/2023/323
 *
 * For the diffusion matrix, 1 + Diag(V), we perform a search to find an optimized
 * vector V composed of elements with efficient multiplication algorithms in AVX2/AVX512/NEON.
 *
 * This leads to using small values (e.g. 1, 2, 3, 4) where multiplication is implemented using addition
 * and, inverse powers of 2 where it is possible to avoid monty reduction can be avoided.
 * Additionally, for technical reasons, having the first entry be -2 is useful.
 *
 * Optimized Diagonal for KoalaBear16:
 * [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, -1/2^8, 1/8, -1/8, -1/16, 1/2^24, -1/2^24]
 * Optimized Diagonal for KoalaBear24:
 * [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, -1/2^8, 1/4, 1/8, -1/8, 1/16, -1/16, 1/32, -1/32, 1/64, -1/64, -1/2^7, -1/2^9, 1/2^24, -1/2^24]
 * See poseidon2\src\diffusion.rs for information on how to double check these matrices in Sage.
*/
use p3_field::{AbstractField, Field};
use p3_monty_31::{
    construct_2_exp_neg_n, InternalLayerBaseParameters, InternalLayerParameters, MontyField31,
    Poseidon2ExternalLayerMonty31, Poseidon2InternalLayerMonty31,
};
use p3_poseidon2::Poseidon2;

use crate::{KoalaBear, KoalaBearParameters};

pub type Poseidon2InternalLayerKoalaBear<const WIDTH: usize> =
    Poseidon2InternalLayerMonty31<KoalaBearParameters, WIDTH, KoalaBearInternalLayerParameters>;

pub type Poseidon2ExternalLayerKoalaBear<const WIDTH: usize> =
    Poseidon2ExternalLayerMonty31<KoalaBearParameters, WIDTH>;

/// Degree of the chosen permutation polynomial for KoalaBear, used as the Poseidon2 S-Box.
///
/// As p - 1 = 127 * 2^{24} we have a a lot of choice in degree D satisfying gcd(p - 1, D) = 1.
/// Experimentation suggests that the optimal choice is the smallest available one, namely 3.
const KOALABEAR_S_BOX_DEGREE: u64 = 3;

/// Poseidon2KoalaBear contains the implementations of Poseidon2
/// specialised to run on the current architecture. It acts on
/// arrays of the form either [KoalaBear::Packing; WIDTH] or [KoalaBear; WIDTH]
pub type Poseidon2KoalaBear<const WIDTH: usize> = Poseidon2<
    <KoalaBear as Field>::Packing,
    Poseidon2ExternalLayerKoalaBear<WIDTH>,
    Poseidon2InternalLayerKoalaBear<WIDTH>,
    WIDTH,
    KOALABEAR_S_BOX_DEGREE,
>;

#[derive(Debug, Clone, Default)]
pub struct KoalaBearInternalLayerParameters;

impl InternalLayerBaseParameters<KoalaBearParameters, 16> for KoalaBearInternalLayerParameters {
    type ArrayLike = [MontyField31<KoalaBearParameters>; 15];

    /// Perform the internal matrix multiplication: s -> (1 + Diag(V))s.
    /// We ignore `state[0]` as it is handled separately.
    fn internal_layer_mat_mul(
        state: &mut [MontyField31<KoalaBearParameters>; 16],
        sum: MontyField31<KoalaBearParameters>,
    ) {
        // The diagonal matrix is defined by the vector:
        // V = [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, -1/2^8, 1/8, -1/8, -1/16, 1/2^24, -1/2^24]
        state[1] += sum;
        state[2] = state[2].double() + sum;
        state[3] = state[3].halve() + sum;
        state[4] = sum + state[4].double() + state[4];
        state[5] = sum + state[5].double().double();
        state[6] = sum - state[6].halve();
        state[7] = sum - (state[7].double() + state[7]);
        state[8] = sum - state[8].double().double();
        state[9] *= construct_2_exp_neg_n(8);
        state[9] += sum;
        state[10] *= construct_2_exp_neg_n(8);
        state[10] = sum - state[10];
        state[11] *= construct_2_exp_neg_n(3);
        state[11] += sum;
        state[12] *= construct_2_exp_neg_n(3);
        state[12] = sum - state[12];
        state[13] *= construct_2_exp_neg_n(4);
        state[13] = sum - state[13];
        state[14] *= construct_2_exp_neg_n(24);
        state[14] += sum;
        state[15] *= construct_2_exp_neg_n(24);
        state[15] = sum - state[15];
    }
}

impl InternalLayerBaseParameters<KoalaBearParameters, 24> for KoalaBearInternalLayerParameters {
    type ArrayLike = [MontyField31<KoalaBearParameters>; 23];

    /// Perform the internal matrix multiplication: s -> (1 + Diag(V))s.
    /// We ignore `state[0]` as it is handled separately.
    fn internal_layer_mat_mul(
        state: &mut [MontyField31<KoalaBearParameters>; 24],
        sum: MontyField31<KoalaBearParameters>,
    ) {
        // The diagonal matrix is defined by the vector:
        // V = [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, -1/2^8, 1/4, 1/8, -1/8, 1/16, -1/16, 1/32, -1/32, 1/64, -1/64, -1/2^7, -1/2^9, 1/2^24, -1/2^24]
        state[1] += sum;
        state[2] = state[2].double() + sum;
        state[3] = state[3].halve() + sum;
        state[4] = sum + state[4].double() + state[4];
        state[5] = sum + state[5].double().double();
        state[6] = sum - state[6].halve();
        state[7] = sum - (state[7].double() + state[7]);
        state[8] = sum - state[8].double().double();
        state[9] *= construct_2_exp_neg_n(8);
        state[9] += sum;
        state[10] *= construct_2_exp_neg_n(8);
        state[10] = sum - state[10];
        state[11] *= construct_2_exp_neg_n(2);
        state[11] += sum;
        state[12] *= construct_2_exp_neg_n(3);
        state[12] += sum;
        state[13] *= construct_2_exp_neg_n(3);
        state[13] = sum - state[13];
        state[14] *= construct_2_exp_neg_n(4);
        state[14] += sum;
        state[15] *= construct_2_exp_neg_n(4);
        state[15] = sum - state[15];
        state[16] *= construct_2_exp_neg_n(5);
        state[16] += sum;
        state[17] *= construct_2_exp_neg_n(5);
        state[17] = sum - state[17];
        state[18] *= construct_2_exp_neg_n(6);
        state[18] += sum;
        state[19] *= construct_2_exp_neg_n(6);
        state[19] = sum - state[19];
        state[20] *= construct_2_exp_neg_n(7);
        state[20] = sum - state[20];
        state[21] *= construct_2_exp_neg_n(9);
        state[21] = sum - state[21];
        state[22] *= construct_2_exp_neg_n(24);
        state[22] += sum;
        state[23] *= construct_2_exp_neg_n(24);
        state[23] = sum - state[23];
    }
}

impl InternalLayerParameters<KoalaBearParameters, 16> for KoalaBearInternalLayerParameters {}
impl InternalLayerParameters<KoalaBearParameters, 24> for KoalaBearInternalLayerParameters {}

#[derive(Debug, Clone, Default)]
pub struct KoalaBearExternalLayerParameters;

#[cfg(test)]
mod tests {
    use p3_field::AbstractField;
    use p3_symmetric::Permutation;
    use rand::SeedableRng;
    use rand_xoshiro::Xoroshiro128Plus;

    use super::*;

    type F = KoalaBear;

    // We need to make some round constants. We use Xoroshiro128Plus for this as we can easily match this PRNG in sage.
    // See: https://github.com/0xPolygonZero/hash-constants for the sage code used to create all these tests.

    /// Test on a roughly random input.
    /// This random input is generated by the following sage code:
    /// set_random_seed(16)
    /// vector([KB.random_element() for t in range(16)]).
    #[test]
    fn test_poseidon2_width_16_random() {
        let mut input: [F; 16] = [
            894848333, 1437655012, 1200606629, 1690012884, 71131202, 1749206695, 1717947831,
            120589055, 19776022, 42382981, 1831865506, 724844064, 171220207, 1299207443, 227047920,
            1783754913,
        ]
        .map(F::from_canonical_u32);

        let expected: [F; 16] = [
            1472124395, 1149134692, 2066945197, 497546554, 1210038209, 133688735, 1494484535,
            1505600411, 1511438408, 1374012105, 820507391, 2019428848, 686883592, 619968952,
            1959306394, 1373405731,
        ]
        .map(F::from_canonical_u32);

        let mut rng = Xoroshiro128Plus::seed_from_u64(1);
        let perm = Poseidon2KoalaBear::new_from_rng_128(&mut rng);

        perm.permute_mut(&mut input);
        assert_eq!(input, expected);
    }

    /// Test on a roughly random input.
    /// This random input is generated by the following sage code:
    /// set_random_seed(24)
    /// vector([KB.random_element() for t in range(24)]).
    #[test]
    fn test_poseidon2_width_24_random() {
        let mut input: [F; 24] = [
            886409618, 1327899896, 1902407911, 591953491, 648428576, 1844789031, 1198336108,
            355597330, 1799586834, 59617783, 790334801, 1968791836, 559272107, 31054313,
            1042221543, 474748436, 135686258, 263665994, 1962340735, 1741539604, 2026927696,
            449439011, 1131357108, 50869465,
        ]
        .map(F::from_canonical_u32);

        let expected: [F; 24] = [
            383159477, 1853122842, 141680496, 292525701, 1259330520, 412236438, 2060624596,
            1222507449, 958106053, 1235449514, 956433966, 1740904776, 1248898185, 1255690239,
            678044138, 158528918, 59290002, 698848812, 1527585185, 801440866, 1870481147, 1837554,
            176075260, 502918143,
        ]
        .map(F::from_canonical_u32);

        let mut rng = Xoroshiro128Plus::seed_from_u64(1);
        let perm = Poseidon2KoalaBear::new_from_rng_128(&mut rng);

        perm.permute_mut(&mut input);
        assert_eq!(input, expected);
    }
}
