//! Implementation of Poseidon2, see: https://eprint.iacr.org/2023/323

use p3_field::PrimeField32;
use p3_poseidon2::DiffusionPermutation;
use p3_symmetric::Permutation;
use serde::{Deserialize, Serialize};

use crate::{monty_reduce, to_babybear_array, BabyBear};

// Optimised diffusion matrices for Babybear16:
// Small entries: [-2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16]
// Power of 2 entries: [-2,   1,   2,   4,   8,  16,  32,  64, 128, 256, 512, 1024, 2048, 4096, 8192, 32768]
//                   = [ ?, 2^0, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6, 2^7, 2^8, 2^9, 2^10, 2^11, 2^12, 2^13, 2^15]

// Optimised diffusion matrices for Babybear24:
// Small entries: [-2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24]
// Power of 2 entries: [-2,   1,   2,   4,   8,  16,  32,  64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 262144, 524288, 1048576, 2097152, 4194304, 8388608]
//                   = [ ?, 2^0, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6, 2^7, 2^8, 2^9, 2^10, 2^11, 2^12, 2^13,  2^14,  2^15,  2^16,   2^18,   2^19,    2^20,    2^21,    2^22,    2^23]

// In order to use these to their fullest potential we need to slightly reimage what the matrix looks like.
// Note that if (1 + D(v)) is a valid matrix then so is r(1 + D(v)) for any constant scalar r. Hence we should operate
// such that (1 + D(v)) is the monty form of the matrix. This should allow for some delayed reduction tricks.

// Long term, MONTY_INVERSE, POSEIDON2_INTERNAL_MATRIX_DIAG_16_BABYBEAR_MONTY, POSEIDON2_INTERNAL_MATRIX_DIAG_24_BABYBEAR_MONTY can all be removed.
// Currently we need them for each Packed field implementation so they are given here to prevent code duplication.
// They need to be pub and not pub(crate) as otherwise clippy gets annoyed if no vector intrinsics are available.
pub const MONTY_INVERSE: BabyBear = BabyBear { value: 1 };

pub const POSEIDON2_INTERNAL_MATRIX_DIAG_16_BABYBEAR_MONTY: [BabyBear; 16] = to_babybear_array([
    BabyBear::ORDER_U32 - 2,
    1,
    1 << 1,
    1 << 2,
    1 << 3,
    1 << 4,
    1 << 5,
    1 << 6,
    1 << 7,
    1 << 8,
    1 << 9,
    1 << 10,
    1 << 11,
    1 << 12,
    1 << 13,
    1 << 15,
]);

const POSEIDON2_INTERNAL_MATRIX_DIAG_16_MONTY_SHIFTS: [u8; 15] =
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15];

pub const POSEIDON2_INTERNAL_MATRIX_DIAG_24_BABYBEAR_MONTY: [BabyBear; 24] = to_babybear_array([
    BabyBear::ORDER_U32 - 2,
    1,
    1 << 1,
    1 << 2,
    1 << 3,
    1 << 4,
    1 << 5,
    1 << 6,
    1 << 7,
    1 << 8,
    1 << 9,
    1 << 10,
    1 << 11,
    1 << 12,
    1 << 13,
    1 << 14,
    1 << 15,
    1 << 16,
    1 << 18,
    1 << 19,
    1 << 20,
    1 << 21,
    1 << 22,
    1 << 23,
]);

const POSEIDON2_INTERNAL_MATRIX_DIAG_24_MONTY_SHIFTS: [u8; 23] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23,
];

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiffusionMatrixBabyBear;

impl Permutation<[BabyBear; 16]> for DiffusionMatrixBabyBear {
    #[inline]
    fn permute_mut(&self, state: &mut [BabyBear; 16]) {
        let part_sum: u64 = state.iter().skip(1).map(|x| x.value as u64).sum();
        let full_sum = part_sum + (state[0].value as u64);
        let s0 = part_sum + (-state[0]).value as u64;
        state[0] = BabyBear {
            value: monty_reduce(s0),
        };
        for i in 1..16 {
            let si = full_sum
                + ((state[i].value as u64)
                    << POSEIDON2_INTERNAL_MATRIX_DIAG_16_MONTY_SHIFTS[i - 1]);
            state[i] = BabyBear {
                value: monty_reduce(si),
            };
        }
    }
}

impl DiffusionPermutation<BabyBear, 16> for DiffusionMatrixBabyBear {}

impl Permutation<[BabyBear; 24]> for DiffusionMatrixBabyBear {
    #[inline]
    fn permute_mut(&self, state: &mut [BabyBear; 24]) {
        let part_sum: u64 = state.iter().skip(1).map(|x| x.value as u64).sum();
        let full_sum = part_sum + (state[0].value as u64);
        let s0 = part_sum + (-state[0]).value as u64;
        state[0] = BabyBear {
            value: monty_reduce(s0),
        };
        for i in 1..24 {
            let si = full_sum
                + ((state[i].value as u64)
                    << POSEIDON2_INTERNAL_MATRIX_DIAG_24_MONTY_SHIFTS[i - 1]);
            state[i] = BabyBear {
                value: monty_reduce(si),
            };
        }
    }
}

impl DiffusionPermutation<BabyBear, 24> for DiffusionMatrixBabyBear {}

#[cfg(test)]
mod tests {
    use p3_field::AbstractField;
    use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
    use rand::SeedableRng;
    use rand_xoshiro::Xoroshiro128Plus;

    use super::*;

    type F = BabyBear;

    // We need to make some round constants. We use Xoroshiro128Plus for this as we can easily match this PRNG in sage.
    // See: https://github.com/0xPolygonZero/hash-constants for the sage code used to create all these tests.

    // Our Poseidon2 Implementation for BabyBear
    fn poseidon2_babybear<const WIDTH: usize, const D: u64, DiffusionMatrix>(
        input: &mut [F; WIDTH],
        diffusion_matrix: DiffusionMatrix,
    ) where
        DiffusionMatrix: DiffusionPermutation<F, WIDTH>,
    {
        let mut rng = Xoroshiro128Plus::seed_from_u64(1);

        // Our Poseidon2 implementation.
        let poseidon2: Poseidon2<F, Poseidon2ExternalMatrixGeneral, DiffusionMatrix, WIDTH, D> =
            Poseidon2::new_from_rng_128(Poseidon2ExternalMatrixGeneral, diffusion_matrix, &mut rng);

        poseidon2.permute_mut(input);
    }

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
            512585766, 975869435, 1921378527, 1238606951, 899635794, 132650430, 1426417547,
            1734425242, 57415409, 67173027, 1535042492, 1318033394, 1070659233, 17258943,
            856719028, 1500534995,
        ]
        .map(F::from_canonical_u32);

        poseidon2_babybear::<16, 7, _>(&mut input, DiffusionMatrixBabyBear);
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
            162275163, 462059149, 1096991565, 924509284, 300323988, 608502870, 427093935,
            733126108, 1676785000, 669115065, 441326760, 60861458, 124006210, 687842154, 270552480,
            1279931581, 1030167257, 126690434, 1291783486, 669126431, 1320670824, 1121967237,
            458234203, 142219603,
        ]
        .map(F::from_canonical_u32);

        poseidon2_babybear::<24, 7, _>(&mut input, DiffusionMatrixBabyBear);
        assert_eq!(input, expected);
    }
}
