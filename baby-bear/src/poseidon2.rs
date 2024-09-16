//! Implementation of Poseidon2, see: https://eprint.iacr.org/2023/323
use p3_field::{AbstractField, Field};

use p3_monty_31::{
    construct_2_exp_neg_n, InternalLayerBaseParameters, InternalLayerParameters, MontyField31,
    Poseidon2ExternalLayerMonty31, Poseidon2InternalLayerMonty31,
};

use p3_poseidon2::Poseidon2;

use crate::{BabyBear, BabyBearParameters};

// See poseidon2\src\diffusion.rs for information on how to double check these matrices in Sage.
// Optimized Diffusion matrices for Babybear16.
// Small entries: [-2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17]
// Power of 2 entries: [-2, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 32768]
//                 = 2^[ ?, 0, 1, 2, 3,  4,  5,  6,   7,   8,   9,   10,   11,   12,   13,    15]
//
// Optimized Diffusion matrices for Babybear24.
// Small entries: [-2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 24, 25]
// Power of 2 entries: [-2, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 8388608]
//                 = 2^[ ?, 0, 1, 2, 3,  4,  5,  6,   7,   8,   9,   10,   11,   12,   13,    14,    15,    16,     17,     18,     19,      20,      21,      23]
//
// In order to use these to their fullest potential we need to slightly reimagine what the matrix looks like.
// Note that if (1 + Diag(vec)) is a valid matrix then so is r(1 + Diag(vec)) for any constant scalar r. Hence we should operate
// such that (1 + Diag(vec)) is the monty form of the matrix. This allows for delayed reduction tricks.

// Long term, INTERNAL_DIAG_MONTY will be removed.
// Currently we need them for each Packed field implementation so they are given here to prevent code duplication.

pub type Poseidon2InternalLayerBabyBear<const WIDTH: usize> =
    Poseidon2InternalLayerMonty31<BabyBearParameters, WIDTH, BabyBearInternalLayerParameters>;

pub type Poseidon2ExternalLayerBabyBear<const WIDTH: usize> =
    Poseidon2ExternalLayerMonty31<BabyBearParameters, WIDTH>;

pub type Poseidon2BabyBear<const WIDTH: usize, const D: u64> = Poseidon2<
    <BabyBear as Field>::Packing,
    Poseidon2ExternalLayerBabyBear<WIDTH>,
    Poseidon2InternalLayerBabyBear<WIDTH>,
    WIDTH,
    D,
>;

#[derive(Debug, Clone, Default)]
pub struct BabyBearInternalLayerParameters;

impl InternalLayerBaseParameters<BabyBearParameters, 16> for BabyBearInternalLayerParameters {
    type ArrayLike = [MontyField31<BabyBearParameters>; 15];

    fn internal_diag_mul(
        state: &mut [MontyField31<BabyBearParameters>; 16],
        sum: MontyField31<BabyBearParameters>,
    ) {
        // We ignore state[0] as it has already been handled.
        // [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/(2**8), -1/(2**8), 1/4, 1/8, -1/16, 1/2**27, -1/2**27]
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
        state[13] *= construct_2_exp_neg_n(4);
        state[13] = sum - state[13];
        state[14] *= construct_2_exp_neg_n(27);
        state[14] += sum;
        state[15] *= construct_2_exp_neg_n(27);
        state[15] = sum - state[15];
    }
}

impl InternalLayerBaseParameters<BabyBearParameters, 24> for BabyBearInternalLayerParameters {
    type ArrayLike = [MontyField31<BabyBearParameters>; 23];

    fn internal_diag_mul(
        state: &mut [MontyField31<BabyBearParameters>; 24],
        sum: MontyField31<BabyBearParameters>,
    ) {
        // We ignore state[0] as it has already been handled.
        // [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/(2**8), -1/(2**8), 1/2**2, -1/2**2, 1/(2**3), -1/(2**3), 1/(2**4), -1/(2**4), -1/(2**5), -1/(2**6), 1/(2**7), -1/(2**7), 1/(2**9), 1/2**24, -1/2**24]
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
        state[12] *= construct_2_exp_neg_n(2);
        state[12] = sum - state[12];
        state[13] *= construct_2_exp_neg_n(3);
        state[13] += sum;
        state[14] *= construct_2_exp_neg_n(3);
        state[14] = sum - state[14];
        state[15] *= construct_2_exp_neg_n(4);
        state[15] += sum;
        state[16] *= construct_2_exp_neg_n(4);
        state[16] = sum - state[16];
        state[17] *= construct_2_exp_neg_n(5);
        state[17] = sum - state[17];
        state[18] *= construct_2_exp_neg_n(6);
        state[18] = sum - state[18];
        state[19] *= construct_2_exp_neg_n(7);
        state[19] += sum;
        state[20] *= construct_2_exp_neg_n(7);
        state[20] = sum - state[20];
        state[21] *= construct_2_exp_neg_n(9);
        state[21] += sum;
        state[22] *= construct_2_exp_neg_n(27);
        state[22] += sum;
        state[23] *= construct_2_exp_neg_n(27);
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
    use p3_poseidon2::{ExternalLayer, InternalLayer, Poseidon2};
    use p3_symmetric::Permutation;
    use rand::SeedableRng;
    use rand_xoshiro::Xoroshiro128Plus;

    use super::*;

    use crate::BabyBear;

    type F = BabyBear;

    // We need to make some round constants. We use Xoroshiro128Plus for this as we can easily match this PRNG in sage.
    // See: https://github.com/0xPolygonZero/hash-constants for the sage code used to create all these tests.

    // Our Poseidon2 Implementation for BabyBear
    fn poseidon2_babybear<const WIDTH: usize, const WIDTH_MIN_1: usize, const D: u64>(
        input: &mut [F; WIDTH],
    ) where
        BabyBearInternalLayerParameters: InternalLayerParameters<BabyBearParameters, WIDTH>,
        Poseidon2ExternalLayerBabyBear<WIDTH>: ExternalLayer<BabyBear, WIDTH, D>,
        Poseidon2InternalLayerBabyBear<WIDTH>: InternalLayer<
            BabyBear,
            WIDTH,
            D,
            InternalState = <Poseidon2ExternalLayerBabyBear<WIDTH> as ExternalLayer<
                BabyBear,
                WIDTH,
                D,
            >>::InternalState,
        >,
    {
        let mut rng = Xoroshiro128Plus::seed_from_u64(1);

        // Our Poseidon2 implementation.
        let poseidon2: Poseidon2BabyBear<WIDTH, D> = Poseidon2::new_from_rng_128(&mut rng);

        poseidon2.permute_mut(input);
    }

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
            575479740, 1350824762, 2117880331, 1034350182, 1722317281, 988412135, 1272198010,
            2022533539, 1465703323, 648698653, 439658904, 878238659, 1163940027, 287402877,
            685135400, 1397893936,
        ]
        .map(F::from_canonical_u32);

        poseidon2_babybear::<16, 15, 7>(&mut input);
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
            960059210, 1580868478, 1801196597, 904704071, 855821469, 1913275695, 1509383446,
            1044214192, 627721401, 905385372, 1577681198, 1162796264, 2082498994, 488108023,
            909588461, 1160073886, 1386956787, 10169827, 1492928499, 843558832, 580466197,
            1008002900, 1086108283, 697296755,
        ]
        .map(F::from_canonical_u32);

        poseidon2_babybear::<24, 23, 7>(&mut input);
        assert_eq!(input, expected);
    }
}
