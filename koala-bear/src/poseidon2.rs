//! Implementation of Poseidon2, see: https://eprint.iacr.org/2023/323

use p3_field::PrimeField32;
use p3_poseidon2::DiffusionPermutation;
use p3_symmetric::Permutation;

use crate::{monty_reduce, to_koalabear_array, KoalaBear};

// Optimised Diffusion matrices for Koalabear16.
// Small entries: [-2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17]
// Power of 2 entries: [-2, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 32768]
//                 = 2^[ ?, 0, 1, 2, 3,  4,  5,  6,   7,   8,   9,   10,   11,   12,   13,    15]

// Optimised Diffusion matrices for Koalabear24.
// Small entries: [-2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 24, 25]
// Power of 2 entries: [-2, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 8388608]
//                 = 2^[ ?, 0, 1, 2, 3,  4,  5,  6,   7,   8,   9,   10,   11,   12,   13,    14,    15,    16,     17,     18,     19,      20,      21,      23]
// Thuse can be verified by the following sage code (Changing vector/length as desired):
//
// field = GF(2^31 - 2^24 + 1);
// length = 16;
// vector = [-2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17];
// const_mat = matrix(field, length, lambda i, j: 1);
// diag_mat  = diagonal_matrix(field, vector);
// assert (const_mat + diag_mat).characteristic_polynomial().is_irreducible()
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
    use core::array;

    use p3_field::AbstractField;
    use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};

    use super::*;

    type F = KoalaBear;

    fn poseidon2_koalabear_width_16(input: &mut [F; 16]) {
        const WIDTH: usize = 16;
        const D: u64 = 7;
        const ROUNDS_F: usize = 8;
        const ROUNDS_P: usize = 13;

        // Need to make some round constants. We use sage to get a random element which we use as a seed.
        // set_random_seed(11111);
        // ZZ.random_element(2**31);
        const SIMPLE_SEED: u32 = 899088431;
        let external_round_constants: [[F; 16]; 8] = array::from_fn(|j| {
            array::from_fn(|i| F::from_wrapped_u32((i + 16 * j) as u32 + SIMPLE_SEED).exp_u64(11))
        });
        let internal_round_constants: [F; 13] =
            array::from_fn(|i| F::from_wrapped_u32(i as u32 + SIMPLE_SEED).exp_u64(7));

        // Our Poseidon2 implementation.
        let poseidon2: Poseidon2<
            F,
            Poseidon2ExternalMatrixGeneral,
            DiffusionMatrixKoalaBear,
            WIDTH,
            D,
        > = Poseidon2::new(
            ROUNDS_F,
            external_round_constants.to_vec(),
            Poseidon2ExternalMatrixGeneral,
            ROUNDS_P,
            internal_round_constants.to_vec(),
            DiffusionMatrixKoalaBear,
        );
        poseidon2.permute_mut(input);
    }

    fn poseidon2_koalabear_width_24(input: &mut [F; 24]) {
        const WIDTH: usize = 24;
        const D: u64 = 7;
        const ROUNDS_F: usize = 8;
        const ROUNDS_P: usize = 21;

        // Need to make some round constants. We use sage to get a random element which we use as a seed.
        // set_random_seed(11111);
        // ZZ.random_element(2**31);
        const SIMPLE_SEED: u32 = 899088431;
        let external_round_constants: [[F; 24]; 8] = array::from_fn(|j| {
            array::from_fn(|i| F::from_wrapped_u32((i + 24 * j) as u32 + SIMPLE_SEED).exp_u64(11))
        });
        let internal_round_constants: [F; 21] =
            array::from_fn(|i| F::from_wrapped_u32(i as u32 + SIMPLE_SEED).exp_u64(7));

        // Our Poseidon2 implementation.
        let poseidon2: Poseidon2<
            F,
            Poseidon2ExternalMatrixGeneral,
            DiffusionMatrixKoalaBear,
            WIDTH,
            D,
        > = Poseidon2::new(
            ROUNDS_F,
            external_round_constants.to_vec(),
            Poseidon2ExternalMatrixGeneral,
            ROUNDS_P,
            internal_round_constants.to_vec(),
            DiffusionMatrixKoalaBear,
        );
        poseidon2.permute_mut(input);
    }

    /// Test on a roughly random input.
    /// This random input is generated by the following sage code:
    /// set_random_seed(2468)
    /// vector([ZZ.random_element(2**31) for t in range(16)])
    #[test]
    fn test_poseidon2_width_16_random() {
        let mut input: [F; 16] = [
            1179785652, 1291567559, 2079538220, 471640172, 653876821, 478855335, 871063984,
            540251327, 1506944720, 1403776782, 770420443, 126472305, 1535928603, 1017977016,
            818646757, 359411429,
        ]
        .map(F::from_wrapped_u32);

        let expected: [F; 16] = [
            153573417, 912646680, 312278853, 1934353212, 792707553, 386983851, 470582679,
            1088774782, 1984802780, 693802370, 695614875, 2129012796, 898643682, 1996915498,
            1258236015, 2124822813,
        ]
        .map(F::from_canonical_u32);

        poseidon2_koalabear_width_16(&mut input);
        assert_eq!(input, expected);
    }

    /// Test on a roughly random input.
    /// This random input is generated by the following sage code:
    /// set_random_seed(2468)
    /// vector([ZZ.random_element(2**31) for t in range(24)])
    #[test]
    fn test_poseidon2_width_24_random() {
        let mut input: [F; 24] = [
            1179785652, 1291567559, 2079538220, 471640172, 653876821, 478855335, 871063984,
            540251327, 1506944720, 1403776782, 770420443, 126472305, 1535928603, 1017977016,
            818646757, 359411429, 860757874, 286641299, 1346664023, 1674494652, 1209824408,
            1264153249, 679420963, 520737796,
        ]
        .map(F::from_wrapped_u32);

        let expected: [F; 24] = [
            1320249636, 526729738, 1489177406, 1415455092, 1835137531, 2010060075, 2126714408,
            1038840462, 1167417876, 1277596940, 1881969597, 538762592, 1100510666, 1020253537,
            1145279199, 1590121738, 1002408295, 973134356, 664832342, 1695574270, 401141288,
            783976957, 12011863, 1879084694,
        ]
        .map(F::from_canonical_u32);

        poseidon2_koalabear_width_24(&mut input);
        assert_eq!(input, expected);
    }
}
