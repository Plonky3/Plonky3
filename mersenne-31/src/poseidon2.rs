use p3_field::PrimeField32;
use p3_poseidon2::DiffusionPermutation;
use p3_symmetric::Permutation;

use crate::{from_u62, to_mersenne31_array, Mersenne31};

// Optimised diffusion matrices for Mersenne31/16:
// Small entries: [-2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17]
// Power of 2 entries: [-2,  1,   2,   4,   8,  16,  32,  64, 128, 256, 1024, 4096, 8192, 16384, 32768, 65536]
//                   = [?, 2^0, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6, 2^7, 2^8, 2^10, 2^12, 2^13,  2^14,  2^15, 2^16]

// Optimised diffusion matrices for Mersenne31/24:
// Small entries: [-2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24]
// Power of 2 entries: [-2,  1,   2,   4,   8,  16,  32,  64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304]
//                   = [?, 2^0, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6, 2^7, 2^8, 2^9, 2^10, 2^11, 2^12, 2^13,  2^14,  2^15,  2^16,   2^17,   2^18,   2^19,    2^20,    2^21,    2^22]

// Long term, POSEIDON2_INTERNAL_MATRIX_DIAG_16, POSEIDON2_INTERNAL_MATRIX_DIAG_24 can be removed.
// Currently they are needed for packed field implementations.
// They need to be pub and not pub(crate) as otherwise clippy gets annoyed if no vector intrinsics are avaliable.
pub const POSEIDON2_INTERNAL_MATRIX_DIAG_16: [Mersenne31; 16] = to_mersenne31_array([
    Mersenne31::ORDER_U32 - 2,
    1,
    1 << 1,
    1 << 2,
    1 << 3,
    1 << 4,
    1 << 5,
    1 << 6,
    1 << 7,
    1 << 8,
    1 << 10,
    1 << 12,
    1 << 13,
    1 << 14,
    1 << 15,
    1 << 16,
]);

const POSEIDON2_INTERNAL_MATRIX_DIAG_16_SHIFTS: [u8; 15] =
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16];

pub const POSEIDON2_INTERNAL_MATRIX_DIAG_24: [Mersenne31; 24] = to_mersenne31_array([
    Mersenne31::ORDER_U32 - 2,
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
    1 << 22,
]);

const POSEIDON2_INTERNAL_MATRIX_DIAG_24_SHIFTS: [u8; 23] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
];

#[derive(Debug, Clone, Default)]
pub struct DiffusionMatrixMersenne31;

impl Permutation<[Mersenne31; 16]> for DiffusionMatrixMersenne31 {
    #[inline]
    fn permute_mut(&self, state: &mut [Mersenne31; 16]) {
        let part_sum: u64 = state.iter().skip(1).map(|x| x.value as u64).sum();
        let full_sum = part_sum + (state[0].value as u64);
        let s0 = part_sum + (-state[0]).value as u64;
        state[0] = from_u62(s0);
        for i in 1..16 {
            let si = full_sum
                + ((state[i].value as u64) << POSEIDON2_INTERNAL_MATRIX_DIAG_16_SHIFTS[i - 1]);
            state[i] = from_u62(si);
        }
    }
}

impl DiffusionPermutation<Mersenne31, 16> for DiffusionMatrixMersenne31 {}

impl Permutation<[Mersenne31; 24]> for DiffusionMatrixMersenne31 {
    #[inline]
    fn permute_mut(&self, state: &mut [Mersenne31; 24]) {
        let part_sum: u64 = state.iter().skip(1).map(|x| x.value as u64).sum();
        let full_sum = part_sum + (state[0].value as u64);
        let s0 = part_sum + (-state[0]).value as u64;
        state[0] = from_u62(s0);
        for i in 1..24 {
            let si = full_sum
                + ((state[i].value as u64) << POSEIDON2_INTERNAL_MATRIX_DIAG_24_SHIFTS[i - 1]);
            state[i] = from_u62(si);
        }
    }
}

impl DiffusionPermutation<Mersenne31, 24> for DiffusionMatrixMersenne31 {}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use core::array;

    use p3_field::AbstractField;
    use p3_poseidon2::{poseidon2_round_numbers_128, Poseidon2, Poseidon2ExternalMatrixGeneral};

    use super::*;

    type F = Mersenne31;

    // We need to make some round constants. We do this is a pseudo random way using sage code.
    // See: https://github.com/0xPolygonZero/hash-constants for the sage code used to create all these tests.
    // We start with setting: M31 = GF(2^31 - 1)

    // SIMPLE_SEED comes from:
    // set_random_seed(11111)
    // M31.random_element()
    const SIMPLE_SEED: F = F { value: 389183646 };

    // We also fix a large prime which is relatively prime to 2^31 - 2.
    // P = Primes()
    // P.next(11111)
    const POWER: u64 = 11113;

    fn constants_from_seed<const WIDTH: usize>(
        seed: F,
        power: u64,
        rounds_f: usize,
        rounds_p: usize,
    ) -> (Vec<[F; WIDTH]>, Vec<F>) {
        let external_round_constants: Vec<[F; WIDTH]> = (0..rounds_f)
            .map(|j| {
                array::from_fn(|i| {
                    (F::from_wrapped_u32((i + WIDTH * j) as u32) + seed).exp_u64(power)
                })
            })
            .collect();
        let seed_update = seed.exp_u64(power).exp_u64(power);
        let internal_round_constants: Vec<F> = (0..rounds_p)
            .map(|i| (F::from_wrapped_u32(i as u32) + seed_update).exp_u64(power))
            .collect();

        (external_round_constants, internal_round_constants)
    }

    // Our Poseidon2 Implementation for Mersenne31
    fn poseidon2_mersenne31<const WIDTH: usize, const D: u64, DiffusionMatrix>(
        input: &mut [F; WIDTH],
        diffusion_matrix: DiffusionMatrix,
    ) where
        DiffusionMatrix: DiffusionPermutation<F, WIDTH>,
    {
        let (rounds_f, rounds_p) = poseidon2_round_numbers_128::<F>(WIDTH, D);

        let (external_round_constants, internal_round_constants) =
            constants_from_seed::<WIDTH>(SIMPLE_SEED, POWER, rounds_f, rounds_p);

        // Our Poseidon2 implementation.
        let poseidon2: Poseidon2<F, Poseidon2ExternalMatrixGeneral, DiffusionMatrix, WIDTH, D> =
            Poseidon2::new(
                rounds_f,
                external_round_constants.to_vec(),
                Poseidon2ExternalMatrixGeneral,
                rounds_p,
                internal_round_constants.to_vec(),
                diffusion_matrix,
            );

        poseidon2.permute_mut(input);
    }

    /// Test on a roughly random input.
    /// This random input is generated by the following sage code:
    /// set_random_seed(16)
    /// vector([M31.random_element() for t in range(16)]).
    #[test]
    fn test_poseidon2_width_16_random() {
        let mut input: [F; 16] = [
            894848333, 1437655012, 1200606629, 1690012884, 71131202, 1749206695, 1717947831,
            120589055, 19776022, 42382981, 1831865506, 724844064, 171220207, 1299207443, 227047920,
            1783754913,
        ]
        .map(F::from_canonical_u32);

        let expected: [F; 16] = [
            687671392, 187739990, 474872297, 1025723782, 1958464721, 1004876398, 972043176,
            1231017992, 1815473754, 997812498, 1891950360, 94240126, 1834774779, 2146393033,
            1194588914, 1694651572,
        ]
        .map(F::from_canonical_u32);

        poseidon2_mersenne31::<16, 5, _>(&mut input, DiffusionMatrixMersenne31);
        assert_eq!(input, expected);
    }

    /// Test on a roughly random input.
    /// This random input is generated by the following sage code:
    /// set_random_seed(24)
    /// vector([M31.random_element() for t in range(24)]).
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
            1647982233, 532201789, 2015259638, 680414033, 1176515591, 1163262902, 2052469886,
            679834118, 2079721199, 1936663286, 66010933, 593633651, 69437652, 889870443,
            1128148983, 1865068789, 649133836, 1434401472, 648402879, 1081496263, 388204549,
            380976594, 1146523540, 841407217,
        ]
        .map(F::from_canonical_u32);

        poseidon2_mersenne31::<24, 5, _>(&mut input, DiffusionMatrixMersenne31);
        assert_eq!(input, expected);
    }
}
