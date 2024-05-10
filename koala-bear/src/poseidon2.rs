//! Implementation of Poseidon2, see: https://eprint.iacr.org/2023/323

use p3_field::PrimeField32;
use p3_poseidon2::DiffusionPermutation;
use p3_symmetric::Permutation;

use crate::{monty_reduce, to_koalabear_array, KoalaBear};

// See poseidon2\src\diffusion.rs for information on how to double check these matrices in Sage.
// Optimised Diffusion matrices for Koalabear16.
// Small entries: [-2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17]
// Power of 2 entries: [-2, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 32768]
//                 = 2^[ ?, 0, 1, 2, 3,  4,  5,  6,   7,   8,   9,   10,   11,   12,   13,    15]
//
// Optimised Diffusion matrices for Koalabear24.
// Small entries: [-2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 24, 25]
// Power of 2 entries: [-2, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 8388608]
//                 = 2^[ ?, 0, 1, 2, 3,  4,  5,  6,   7,   8,   9,   10,   11,   12,   13,    14,    15,    16,     17,     18,     19,      20,      21,      23]
//
// In order to use these to their fullest potential we need to slightly reimage what the matrix looks like.
// Note that if (1 + D(v)) is a valid matrix then so is r(1 + D(v)) for any constant scalar r. Hence we should operate
// such that (1 + D(v)) is the monty form of the matrix. This allows for delayed reduction tricks.

// Long term, MONTY_INVERSE, POSEIDON2_INTERNAL_MATRIX_DIAG_16_KOALABEAR_MONTY, POSEIDON2_INTERNAL_MATRIX_DIAG_24_KOALABEAR_MONTY can all be removed.
// Currently we need them for each Packed field implementation so they are given here to prevent code duplication.
// They need to be pub and not pub(crate) as otherwise clippy gets annoyed if no vector intrinsics are available.
pub const MONTY_INVERSE: KoalaBear = KoalaBear { value: 1 };

pub const POSEIDON2_INTERNAL_MATRIX_DIAG_16_KOALABEAR_MONTY: [KoalaBear; 16] =
    to_koalabear_array([
        KoalaBear::ORDER_U32 - 2,
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

pub const POSEIDON2_INTERNAL_MATRIX_DIAG_24_KOALABEAR_MONTY: [KoalaBear; 24] =
    to_koalabear_array([
        KoalaBear::ORDER_U32 - 2,
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
        1 << 17,
        1 << 18,
        1 << 19,
        1 << 20,
        1 << 21,
        1 << 23,
    ]);

const POSEIDON2_INTERNAL_MATRIX_DIAG_24_MONTY_SHIFTS: [u8; 23] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23,
];

#[derive(Debug, Clone, Default)]
pub struct DiffusionMatrixKoalaBear;

impl Permutation<[KoalaBear; 16]> for DiffusionMatrixKoalaBear {
    #[inline]
    fn permute_mut(&self, state: &mut [KoalaBear; 16]) {
        let part_sum: u64 = state.iter().skip(1).map(|x| x.value as u64).sum();
        let full_sum = part_sum + (state[0].value as u64);
        let s0 = part_sum + (-state[0]).value as u64;
        state[0] = KoalaBear {
            value: monty_reduce(s0),
        };
        for i in 1..16 {
            let si = full_sum
                + ((state[i].value as u64)
                    << POSEIDON2_INTERNAL_MATRIX_DIAG_16_MONTY_SHIFTS[i - 1]);
            state[i] = KoalaBear {
                value: monty_reduce(si),
            };
        }
    }
}

impl DiffusionPermutation<KoalaBear, 16> for DiffusionMatrixKoalaBear {}

impl Permutation<[KoalaBear; 24]> for DiffusionMatrixKoalaBear {
    #[inline]
    fn permute_mut(&self, state: &mut [KoalaBear; 24]) {
        let part_sum: u64 = state.iter().skip(1).map(|x| x.value as u64).sum();
        let full_sum = part_sum + (state[0].value as u64);
        let s0 = part_sum + (-state[0]).value as u64;
        state[0] = KoalaBear {
            value: monty_reduce(s0),
        };
        for i in 1..24 {
            let si = full_sum
                + ((state[i].value as u64)
                    << POSEIDON2_INTERNAL_MATRIX_DIAG_24_MONTY_SHIFTS[i - 1]);
            state[i] = KoalaBear {
                value: monty_reduce(si),
            };
        }
    }
}

impl DiffusionPermutation<KoalaBear, 24> for DiffusionMatrixKoalaBear {}

#[cfg(test)]
mod tests {
    use p3_field::AbstractField;
    use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
    use rand::SeedableRng;
    use rand_xoshiro::Xoroshiro128Plus;

    use super::*;

    type F = KoalaBear;

    // We need to make some round constants. We use Xoroshiro128Plus for this as we can easily match this PRNG in sage.
    // See: https://github.com/0xPolygonZero/hash-constants for the sage code used to create all these tests.

    // Our Poseidon2 Implementation for KoalaBear
    fn poseidon2_koalabear<const WIDTH: usize, const D: u64, DiffusionMatrix>(
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

        poseidon2_koalabear::<16, 3, _>(&mut input, DiffusionMatrixKoalaBear);
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

        poseidon2_koalabear::<24, 3, _>(&mut input, DiffusionMatrixKoalaBear);
        assert_eq!(input, expected);
    }
}
