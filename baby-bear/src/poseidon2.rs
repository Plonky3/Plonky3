//! Implementation of Poseidon2, see: https://eprint.iacr.org/2023/323
//!
//! For the diffusion matrix, 1 + Diag(V), we perform a search to find an optimized
//! vector V composed of elements with efficient multiplication algorithms in AVX2/AVX512/NEON.
//!
//! This leads to using small values (e.g. 1, 2, 3, 4) where multiplication is implemented using addition
//! and inverse powers of 2 where it is possible to avoid monty reductions.
//! Additionally, for technical reasons, having the first entry be -2 is useful.
//!
//! Optimized Diagonal for BabyBear16:
//! [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/4, 1/8, 1/2^27, -1/2^8, -1/16, -1/2^27].
//! Optimized Diagonal for BabyBear24:
//! [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/4, 1/8, 1/16, 1/2^7, 1/2^9, 1/2^27, -1/2^8, -1/4, -1/8, -1/16, -1/32, -1/64, -1/2^7, -1/2^27]
//! See poseidon2\src\diffusion.rs for information on how to double check these matrices in Sage.

use p3_field::{Algebra, Field, PrimeCharacteristicRing, PrimeField32};
use p3_monty_31::{
    GenericPoseidon2LinearLayersMonty31, InternalLayerBaseParameters, InternalLayerParameters,
    MontyField31, Poseidon2ExternalLayerMonty31, Poseidon2InternalLayerMonty31,
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

/// An implementation of the Poseidon2 hash function specialised to run on the current architecture.
///
/// It acts on arrays of the form either `[BabyBear::Packing; WIDTH]` or `[BabyBear; WIDTH]`. For speed purposes,
/// wherever possible, input arrays should of the form `[BabyBear::Packing; WIDTH]`.
pub type Poseidon2BabyBear<const WIDTH: usize> = Poseidon2<
    BabyBear,
    Poseidon2ExternalLayerBabyBear<WIDTH>,
    Poseidon2InternalLayerBabyBear<WIDTH>,
    WIDTH,
    BABYBEAR_S_BOX_DEGREE,
>;

/// An implementation of the matrix multiplications in the internal and external layers of Poseidon2.
///
/// This can act on `[A; WIDTH]` for any ring implementing `Algebra<BabyBear>`.
/// If you have either `[BabyBear::Packing; WIDTH]` or `[BabyBear; WIDTH]` it will be much faster
/// to use `Poseidon2BabyBear<WIDTH>` instead of building a Poseidon2 permutation using this.
pub type GenericPoseidon2LinearLayersBabyBear =
    GenericPoseidon2LinearLayersMonty31<BabyBearParameters, BabyBearInternalLayerParameters>;

// In order to use BabyBear::new_array we need to convert our vector to a vector of u32's.
// To do this we make use of the fact that BabyBear::ORDER_U32 - 1 = 15 * 2^27 so for 0 <= n <= 27:
// -1/2^n = (BabyBear::ORDER_U32 - 1) >> n
// 1/2^n = -(-1/2^n) = BabyBear::ORDER_U32 - ((BabyBear::ORDER_U32 - 1) >> n)

/// The vector `[-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/4, 1/8, 1/2^27, -1/2^8, -1/16, -1/2^27]`
/// saved as an array of BabyBear elements.
const INTERNAL_DIAG_MONTY_16: [BabyBear; 16] = BabyBear::new_array([
    BabyBear::ORDER_U32 - 2,
    1,
    2,
    (BabyBear::ORDER_U32 + 1) >> 1,
    3,
    4,
    (BabyBear::ORDER_U32 - 1) >> 1,
    BabyBear::ORDER_U32 - 3,
    BabyBear::ORDER_U32 - 4,
    BabyBear::ORDER_U32 - ((BabyBear::ORDER_U32 - 1) >> 8),
    BabyBear::ORDER_U32 - ((BabyBear::ORDER_U32 - 1) >> 2),
    BabyBear::ORDER_U32 - ((BabyBear::ORDER_U32 - 1) >> 3),
    BabyBear::ORDER_U32 - 15,
    (BabyBear::ORDER_U32 - 1) >> 8,
    (BabyBear::ORDER_U32 - 1) >> 4,
    15,
]);

/// The vector [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/4, 1/8, 1/16, 1/2^7, 1/2^9, 1/2^27, -1/2^8, -1/4, -1/8, -1/16, -1/32, -1/64, -1/2^7, -1/2^27]
/// saved as an array of BabyBear elements.
const INTERNAL_DIAG_MONTY_24: [BabyBear; 24] = BabyBear::new_array([
    BabyBear::ORDER_U32 - 2,
    1,
    2,
    (BabyBear::ORDER_U32 + 1) >> 1,
    3,
    4,
    (BabyBear::ORDER_U32 - 1) >> 1,
    BabyBear::ORDER_U32 - 3,
    BabyBear::ORDER_U32 - 4,
    BabyBear::ORDER_U32 - ((BabyBear::ORDER_U32 - 1) >> 8),
    BabyBear::ORDER_U32 - ((BabyBear::ORDER_U32 - 1) >> 2),
    BabyBear::ORDER_U32 - ((BabyBear::ORDER_U32 - 1) >> 3),
    BabyBear::ORDER_U32 - ((BabyBear::ORDER_U32 - 1) >> 4),
    BabyBear::ORDER_U32 - ((BabyBear::ORDER_U32 - 1) >> 7),
    BabyBear::ORDER_U32 - ((BabyBear::ORDER_U32 - 1) >> 9),
    BabyBear::ORDER_U32 - 15,
    (BabyBear::ORDER_U32 - 1) >> 8,
    (BabyBear::ORDER_U32 - 1) >> 2,
    (BabyBear::ORDER_U32 - 1) >> 3,
    (BabyBear::ORDER_U32 - 1) >> 4,
    (BabyBear::ORDER_U32 - 1) >> 5,
    (BabyBear::ORDER_U32 - 1) >> 6,
    (BabyBear::ORDER_U32 - 1) >> 7,
    15,
]);

/// Contains data needed to define the internal layers of the Poseidon2 permutation.
#[derive(Debug, Clone, Default)]
pub struct BabyBearInternalLayerParameters;

impl InternalLayerBaseParameters<BabyBearParameters, 16> for BabyBearInternalLayerParameters {
    type ArrayLike = [MontyField31<BabyBearParameters>; 15];

    const INTERNAL_DIAG_MONTY: [BabyBear; 16] = INTERNAL_DIAG_MONTY_16;

    /// Perform the internal matrix multiplication: s -> (1 + Diag(V))s.
    /// We ignore `state[0]` as it is handled separately.
    fn internal_layer_mat_mul(
        state: &mut [MontyField31<BabyBearParameters>; 16],
        sum: MontyField31<BabyBearParameters>,
    ) {
        // The diagonal matrix is defined by the vector:
        // V = [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/4, 1/8, 1/2^27, -1/2^8, -1/16, -1/2^27]
        state[1] += sum;
        state[2] = state[2].double() + sum;
        state[3] = state[3].halve() + sum;
        state[4] = sum + state[4].double() + state[4];
        state[5] = sum + state[5].double().double();
        state[6] = sum - state[6].halve();
        state[7] = sum - (state[7].double() + state[7]);
        state[8] = sum - state[8].double().double();
        state[9] = state[9].mul_2exp_neg_n(8);
        state[9] += sum;
        state[10] = state[10].mul_2exp_neg_n(2);
        state[10] += sum;
        state[11] = state[11].mul_2exp_neg_n(3);
        state[11] += sum;
        state[12] = state[12].mul_2exp_neg_n(27);
        state[12] += sum;
        state[13] = state[13].mul_2exp_neg_n(8);
        state[13] = sum - state[13];
        state[14] = state[14].mul_2exp_neg_n(4);
        state[14] = sum - state[14];
        state[15] = state[15].mul_2exp_neg_n(27);
        state[15] = sum - state[15];
    }

    fn generic_internal_linear_layer<A: Algebra<BabyBear>>(state: &mut [A; 16]) {
        let part_sum: A = state[1..].iter().cloned().sum();
        let full_sum = part_sum.clone() + state[0].clone();

        // The first three diagonal elements are -2, 1, 2 so we do something custom.
        state[0] = part_sum - state[0].clone();
        state[1] = full_sum.clone() + state[1].clone();
        state[2] = full_sum.clone() + state[2].double();

        // For the remaining elements we use multiplication.
        // This could probably be improved slightly by making use of the
        // mul_2exp_u64 and div_2exp_u64 but this would involve porting div_2exp_u64 to PrimeCharacteristicRing.
        state
            .iter_mut()
            .zip(INTERNAL_DIAG_MONTY_16)
            .skip(3)
            .for_each(|(val, diag_elem)| {
                *val = full_sum.clone() + val.clone() * diag_elem;
            });
    }
}

impl InternalLayerBaseParameters<BabyBearParameters, 24> for BabyBearInternalLayerParameters {
    type ArrayLike = [MontyField31<BabyBearParameters>; 23];

    const INTERNAL_DIAG_MONTY: [BabyBear; 24] = INTERNAL_DIAG_MONTY_24;

    /// Perform the internal matrix multiplication: s -> (1 + Diag(V))s.
    /// We ignore `state[0]` as it is handled separately.
    fn internal_layer_mat_mul(
        state: &mut [MontyField31<BabyBearParameters>; 24],
        sum: MontyField31<BabyBearParameters>,
    ) {
        // The diagonal matrix is defined by the vector:
        // V = [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/4, 1/8, 1/16, 1/2^7, 1/2^9, 1/2^27, -1/2^8, -1/4, -1/8, -1/16, -1/32, -1/64, -1/2^7, -1/2^27]
        state[1] += sum;
        state[2] = state[2].double() + sum;
        state[3] = state[3].halve() + sum;
        state[4] = sum + state[4].double() + state[4];
        state[5] = sum + state[5].double().double();
        state[6] = sum - state[6].halve();
        state[7] = sum - (state[7].double() + state[7]);
        state[8] = sum - state[8].double().double();
        state[9] = state[9].mul_2exp_neg_n(8);
        state[9] += sum;
        state[10] = state[10].mul_2exp_neg_n(2);
        state[10] += sum;
        state[11] = state[11].mul_2exp_neg_n(3);
        state[11] += sum;
        state[12] = state[12].mul_2exp_neg_n(4);
        state[12] += sum;
        state[13] = state[13].mul_2exp_neg_n(7);
        state[13] += sum;
        state[14] = state[14].mul_2exp_neg_n(9);
        state[14] += sum;
        state[15] = state[15].mul_2exp_neg_n(27);
        state[15] += sum;
        state[16] = state[16].mul_2exp_neg_n(8);
        state[16] = sum - state[16];
        state[17] = state[17].mul_2exp_neg_n(2);
        state[17] = sum - state[17];
        state[18] = state[18].mul_2exp_neg_n(3);
        state[18] = sum - state[18];
        state[19] = state[19].mul_2exp_neg_n(4);
        state[19] = sum - state[19];
        state[20] = state[20].mul_2exp_neg_n(5);
        state[20] = sum - state[20];
        state[21] = state[21].mul_2exp_neg_n(6);
        state[21] = sum - state[21];
        state[22] = state[22].mul_2exp_neg_n(7);
        state[22] = sum - state[22];
        state[23] = state[23].mul_2exp_neg_n(27);
        state[23] = sum - state[23];
    }

    fn generic_internal_linear_layer<A: Algebra<BabyBear>>(state: &mut [A; 24]) {
        let part_sum: A = state[1..].iter().cloned().sum();
        let full_sum = part_sum.clone() + state[0].clone();

        // The first three diagonal elements are -2, 1, 2 so we do something custom.
        state[0] = part_sum - state[0].clone();
        state[1] = full_sum.clone() + state[1].clone();
        state[2] = full_sum.clone() + state[2].double();

        // For the remaining elements we use multiplication.
        // This could probably be improved slightly by making use of the
        // mul_2exp_u64 and div_2exp_u64 but this would involve porting div_2exp_u64 to PrimeCharacteristicRing.
        state
            .iter_mut()
            .zip(INTERNAL_DIAG_MONTY_24)
            .skip(3)
            .for_each(|(val, diag_elem)| {
                *val = full_sum.clone() + val.clone() * diag_elem;
            });
    }
}

impl InternalLayerParameters<BabyBearParameters, 16> for BabyBearInternalLayerParameters {}
impl InternalLayerParameters<BabyBearParameters, 24> for BabyBearInternalLayerParameters {}

#[cfg(test)]
mod tests {
    use p3_symmetric::Permutation;
    use rand::{Rng, SeedableRng};
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
        let mut input: [F; 16] = BabyBear::new_array([
            894848333, 1437655012, 1200606629, 1690012884, 71131202, 1749206695, 1717947831,
            120589055, 19776022, 42382981, 1831865506, 724844064, 171220207, 1299207443, 227047920,
            1783754913,
        ]);

        let expected: [F; 16] = BabyBear::new_array([
            1255099308, 941729227, 93609187, 112406640, 492658670, 1824768948, 812517469,
            1055381989, 670973674, 1407235524, 891397172, 1003245378, 1381303998, 1564172645,
            1399931635, 1005462965,
        ]);

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
        let mut input: [F; 24] = BabyBear::new_array([
            886409618, 1327899896, 1902407911, 591953491, 648428576, 1844789031, 1198336108,
            355597330, 1799586834, 59617783, 790334801, 1968791836, 559272107, 31054313,
            1042221543, 474748436, 135686258, 263665994, 1962340735, 1741539604, 449439011,
            1131357108, 50869465, 1589724894,
        ]);

        let expected: [F; 24] = BabyBear::new_array([
            249424342, 562262148, 757431114, 354243402, 57767055, 976981973, 1393169022,
            1774550827, 1527742125, 1019514605, 1776327602, 266236737, 1412355182, 1070239213,
            426390978, 1775539440, 1527732214, 1101406020, 1417710778, 1699632661, 413672313,
            820348291, 1067197851, 1669055675,
        ]);

        let mut rng = Xoroshiro128Plus::seed_from_u64(1);
        let perm = Poseidon2BabyBear::new_from_rng_128(&mut rng);

        perm.permute_mut(&mut input);

        assert_eq!(input, expected);
    }

    /// Test the generic internal layer against the optimized internal layer
    /// for a random input of width 16.
    #[test]
    fn test_generic_internal_linear_layer_16() {
        let mut rng = rand::rng();
        let mut input1: [F; 16] = rng.random();
        let mut input2 = input1;

        let part_sum: F = input1[1..].iter().copied().sum();
        let full_sum = part_sum + input1[0];

        input1[0] = part_sum - input1[0];

        BabyBearInternalLayerParameters::internal_layer_mat_mul(&mut input1, full_sum);
        BabyBearInternalLayerParameters::generic_internal_linear_layer(&mut input2);

        assert_eq!(input1, input2);
    }

    /// Test the generic internal layer against the optimized internal layer
    /// for a random input of width 24.
    #[test]
    fn test_generic_internal_linear_layer_24() {
        let mut rng = rand::rng();
        let mut input1: [F; 24] = rng.random();
        let mut input2 = input1;

        let part_sum: F = input1[1..].iter().copied().sum();
        let full_sum = part_sum + input1[0];

        input1[0] = part_sum - input1[0];

        BabyBearInternalLayerParameters::internal_layer_mat_mul(&mut input1, full_sum);
        BabyBearInternalLayerParameters::generic_internal_linear_layer(&mut input2);

        assert_eq!(input1, input2);
    }
}
