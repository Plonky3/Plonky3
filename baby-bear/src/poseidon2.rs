//* Implementation of Poseidon2, see: https://eprint.iacr.org/2023/323
//*
//* For the diffusion matrix, 1 + Diag(V), we perform a search to find an optimized
//* vector V composed of elements with efficient multiplication algorithms in AVX2/AVX512/NEON.
//*
//* This leads to using small values (e.g. 1, 2, 3, 4) where multiplication is implemented using addition
//* and, inverse powers of 2 where it is possible to avoid monty reduction can be avoided.
//* Additionally, for technical reasons, having the first entry be -2 is useful.
//*
//* Optimized Diagonal for BabyBear16:
//* [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, -1/2^8, 1/4, 1/8, -1/16, 1/2^27, -1/2^27]
//* Optimized Diagonal for BabyBear24:
//* [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, -1/2^8, 1/4, -1/4, 1/8, -1/8, 1/16, -1/16, -1/32, -1/64, 1/2^7, -1/2^7, 1/2^9, 1/2^27, -1/2^27]
//* See poseidon2\src\diffusion.rs for information on how to double check these matrices in Sage.

use p3_field::{AbstractField, Field};
use p3_monty_31::{
    mul_2_exp_neg_n, InternalLayerBaseParameters, InternalLayerParameters, MontyField31,
    Poseidon2ExternalLayerMonty31, Poseidon2InternalLayerMonty31,
};
use p3_poseidon2::Poseidon2;

use crate::{BabyBear, BabyBearParameters};

pub type Poseidon2InternalLayerBabyBear<const WIDTH: usize> =
    Poseidon2InternalLayerMonty31<BabyBearParameters, WIDTH, BabyBearInternalLayerParameters>;

pub type Poseidon2ExternalLayerBabyBear<const WIDTH: usize> =
    Poseidon2ExternalLayerMonty31<BabyBearParameters, WIDTH>;

/// Degree of the chosen permutation polynomial for BabyBear, used as the Poseidon2 S-Box.
///
/// As p - 1 = 15 * 2^{27} the neither 3 nor 5 satisfy gcd(p - 1, D) = 1.
/// Instead we use the next smallest available value, namely 7.
const BABYBEAR_S_BOX_DEGREE: u64 = 7;

/// Poseidon2BabyBear contains the implementations of Poseidon2
/// specialised to run on the current architecture. It acts on
/// arrays of the form either [BabyBear::Packing; WIDTH] or [BabyBear; WIDTH].
pub type Poseidon2BabyBear<const WIDTH: usize> = Poseidon2<
    <BabyBear as Field>::Packing,
    Poseidon2ExternalLayerBabyBear<WIDTH>,
    Poseidon2InternalLayerBabyBear<WIDTH>,
    WIDTH,
    BABYBEAR_S_BOX_DEGREE,
>;

#[derive(Debug, Clone, Default)]
pub struct BabyBearInternalLayerParameters;

impl InternalLayerBaseParameters<BabyBearParameters, 16> for BabyBearInternalLayerParameters {
    type ArrayLike = [MontyField31<BabyBearParameters>; 15];

    /// Perform the internal matrix multiplication: s -> (1 + Diag(V))s.
    /// We ignore `state[0]` as it is handled separately.
    fn internal_layer_mat_mul(
        state: &mut [MontyField31<BabyBearParameters>; 16],
        sum: MontyField31<BabyBearParameters>,
    ) {
        // The diagonal matrix is defined by the vector:
        // V = [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, -1/2^8, 1/4, 1/8, -1/16, 1/2^27, -1/2^27]
        state[1] += sum;
        state[2] = state[2].double() + sum;
        state[3] = state[3].halve() + sum;
        state[4] = sum + state[4].double() + state[4];
        state[5] = sum + state[5].double().double();
        state[6] = sum - state[6].halve();
        state[7] = sum - (state[7].double() + state[7]);
        state[8] = sum - state[8].double().double();
        state[9] = mul_2_exp_neg_n::<BabyBearParameters>(state[9], 8);
        state[9] += sum;
        state[10] = mul_2_exp_neg_n::<BabyBearParameters>(state[10], 8);
        state[10] = sum - state[10];
        state[11] = mul_2_exp_neg_n::<BabyBearParameters>(state[11], 2);
        state[11] += sum;
        state[12] = mul_2_exp_neg_n::<BabyBearParameters>(state[12], 3);
        state[12] += sum;
        state[13] = mul_2_exp_neg_n::<BabyBearParameters>(state[13], 4);
        state[13] = sum - state[13];
        state[14] = mul_2_exp_neg_n::<BabyBearParameters>(state[14], 27);
        state[14] += sum;
        state[15] = mul_2_exp_neg_n::<BabyBearParameters>(state[15], 27);
        state[15] = sum - state[15];
    }
}

impl InternalLayerBaseParameters<BabyBearParameters, 24> for BabyBearInternalLayerParameters {
    type ArrayLike = [MontyField31<BabyBearParameters>; 23];

    /// Perform the internal matrix multiplication: s -> (1 + Diag(V))s.
    /// We ignore `state[0]` as it is handled separately.
    fn internal_layer_mat_mul(
        state: &mut [MontyField31<BabyBearParameters>; 24],
        sum: MontyField31<BabyBearParameters>,
    ) {
        // The diagonal matrix is defined by the vector:
        // V = [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, -1/2^8, 1/4, -1/4, 1/8, -1/8, 1/16, -1/16, -1/32, -1/64, 1/2^7, -1/2^7, 1/2^9, 1/2^27, -1/2^27]
        state[1] += sum;
        state[2] = state[2].double() + sum;
        state[3] = state[3].halve() + sum;
        state[4] = sum + state[4].double() + state[4];
        state[5] = sum + state[5].double().double();
        state[6] = sum - state[6].halve();
        state[7] = sum - (state[7].double() + state[7]);
        state[8] = sum - state[8].double().double();
        state[9] = mul_2_exp_neg_n::<BabyBearParameters>(state[9], 8);
        state[9] += sum;
        state[10] = mul_2_exp_neg_n::<BabyBearParameters>(state[10], 8);
        state[10] = sum - state[10];
        state[11] = mul_2_exp_neg_n::<BabyBearParameters>(state[11], 2);
        state[11] += sum;
        state[12] = mul_2_exp_neg_n::<BabyBearParameters>(state[12], 2);
        state[12] = sum - state[12];
        state[13] = mul_2_exp_neg_n::<BabyBearParameters>(state[13], 3);
        state[13] += sum;
        state[14] = mul_2_exp_neg_n::<BabyBearParameters>(state[14], 3);
        state[14] = sum - state[14];
        state[15] = mul_2_exp_neg_n::<BabyBearParameters>(state[15], 4);
        state[15] += sum;
        state[16] = mul_2_exp_neg_n::<BabyBearParameters>(state[16], 4);
        state[16] = sum - state[16];
        state[17] = mul_2_exp_neg_n::<BabyBearParameters>(state[17], 5);
        state[17] = sum - state[17];
        state[18] = mul_2_exp_neg_n::<BabyBearParameters>(state[18], 6);
        state[18] = sum - state[18];
        state[19] = mul_2_exp_neg_n::<BabyBearParameters>(state[19], 7);
        state[19] += sum;
        state[20] = mul_2_exp_neg_n::<BabyBearParameters>(state[20], 7);
        state[20] = sum - state[20];
        state[21] = mul_2_exp_neg_n::<BabyBearParameters>(state[21], 9);
        state[21] += sum;
        state[22] = mul_2_exp_neg_n::<BabyBearParameters>(state[22], 27);
        state[22] += sum;
        state[23] = mul_2_exp_neg_n::<BabyBearParameters>(state[23], 27);
        state[23] = sum - state[23];
    }
}

impl InternalLayerParameters<BabyBearParameters, 16> for BabyBearInternalLayerParameters {}
impl InternalLayerParameters<BabyBearParameters, 24> for BabyBearInternalLayerParameters {}

#[derive(Debug, Clone, Default)]
pub struct BabyBearExternalLayerParameters;

#[cfg(test)]
mod tests {
    use p3_field::AbstractField;
    use p3_symmetric::Permutation;
    use rand::SeedableRng;
    use rand_xoshiro::Xoroshiro128Plus;

    use super::*;

    type F = BabyBear;

    // We need to make some round constants. We use Xoroshiro128Plus for this as we can easily match this PRNG in sage.
    // See: https://github.com/0xPolygonZero/hash-constants for the sage code used to create all these tests.

    /// Test on a roughly random input.
    /// This random input is generated by the following sage code:
    /// set_random_seed(16)
    /// vector([BB.random_element() for t in range(16)]).
    #[test]
    fn test_poseidon2_width_16_random() {
        let mut input: [F; 16] = [
            894848333, 1437655012, 1200606629, 1690012884, 71131202, 1749206695, 1717947831,
            120589055, 19776022, 42382981, 1831865506, 724844064, 171220207, 1299207443, 227047920,
            1783754913,
        ]
        .map(F::from_canonical_u32);

        let expected: [F; 16] = [
            923104148, 723833968, 73911496, 859420332, 1117510264, 1681542795, 283153958,
            301704038, 708126461, 43189957, 881325743, 877238538, 177615896, 148062838, 1616599690,
            1795131333,
        ]
        .map(F::from_canonical_u32);

        let mut rng = Xoroshiro128Plus::seed_from_u64(1);
        let perm = Poseidon2BabyBear::new_from_rng_128(&mut rng);

        perm.permute_mut(&mut input);
        assert_eq!(input, expected);
    }

    /// Test on a roughly random input.
    /// This random input is generated by the following sage code:
    /// set_random_seed(24)
    /// vector([BB.random_element() for t in range(24)]).
    #[test]
    fn test_poseidon2_width_24_random() {
        let mut input: [F; 24] = [
            886409618, 1327899896, 1902407911, 591953491, 648428576, 1844789031, 1198336108,
            355597330, 1799586834, 59617783, 790334801, 1968791836, 559272107, 31054313,
            1042221543, 474748436, 135686258, 263665994, 1962340735, 1741539604, 449439011,
            1131357108, 50869465, 1589724894,
        ]
        .map(F::from_canonical_u32);

        let expected: [F; 24] = [
            219531206, 715648852, 1715811273, 1236113408, 1091221184, 1900745022, 266009652,
            1283445203, 1356464870, 622765279, 917026370, 1793372862, 613903282, 779986614,
            186887694, 1046427991, 961524701, 496059678, 1694185952, 1073981289, 352658227,
            355182047, 480818189, 702516780,
        ]
        .map(F::from_canonical_u32);

        let mut rng = Xoroshiro128Plus::seed_from_u64(1);
        let perm = Poseidon2BabyBear::new_from_rng_128(&mut rng);

        perm.permute_mut(&mut input);

        assert_eq!(input, expected);
    }
}
