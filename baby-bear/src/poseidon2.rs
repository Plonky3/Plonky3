//! Implementation of Poseidon2, see: https://eprint.iacr.org/2023/323

use p3_field::PrimeField32;
use p3_poseidon2::{matmul_internal, DiffusionPermutation};
use p3_symmetric::Permutation;

use crate::{monty_reduce, to_babybear_array, BabyBear};

// The matrix used in the poseidon2 implementation from Horizen Labs: https://github.com/HorizenLabs/poseidon2
pub(crate) const MATRIX_DIAG_16_BABYBEAR_MONTY_HL: [BabyBear; 16] = to_babybear_array([
    0x0a632d94, 0x6db657b7, 0x56fbdc9e, 0x052b3d8a, 0x33745201, 0x5c03108c, 0x0beba37b, 0x258c2e8b,
    0x12029f39, 0x694909ce, 0x6d231724, 0x21c3b222, 0x3c0904a5, 0x01d6acda, 0x27705c83, 0x5231c802,
]);

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
// They need to be pub and not pub(crate) as otherwise clippy gets annoyed if no vector intrinsics are avaliable.
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

#[derive(Debug, Clone, Default)]
pub struct DiffusionMatrixBabybear;

impl Permutation<[BabyBear; 16]> for DiffusionMatrixBabybear {
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

impl DiffusionPermutation<BabyBear, 16> for DiffusionMatrixBabybear {}

impl Permutation<[BabyBear; 24]> for DiffusionMatrixBabybear {
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

impl DiffusionPermutation<BabyBear, 24> for DiffusionMatrixBabybear {}

#[derive(Debug, Clone, Default)]
pub struct DiffusionMatrixBabybearHL;

impl Permutation<[BabyBear; 16]> for DiffusionMatrixBabybearHL {
    fn permute_mut(&self, state: &mut [BabyBear; 16]) {
        matmul_internal::<BabyBear, BabyBear, 16>(state, MATRIX_DIAG_16_BABYBEAR_MONTY_HL);
    }
}

impl DiffusionPermutation<BabyBear, 16> for DiffusionMatrixBabybearHL {}

pub const HL_BABYBEAR_16_EXTERNAL_ROUND_CONSTANTS: [[BabyBear; 16]; 8] = [
    to_babybear_array([
        0x69cbb6af, 0x46ad93f9, 0x60a00f4e, 0x6b1297cd, 0x23189afe, 0x732e7bef, 0x72c246de,
        0x2c941900, 0x0557eede, 0x1580496f, 0x3a3ea77b, 0x54f3f271, 0x0f49b029, 0x47872fe1,
        0x221e2e36, 0x1ab7202e,
    ]),
    to_babybear_array([
        0x487779a6, 0x3851c9d8, 0x38dc17c0, 0x209f8849, 0x268dcee8, 0x350c48da, 0x5b9ad32e,
        0x0523272b, 0x3f89055b, 0x01e894b2, 0x13ddedde, 0x1b2ef334, 0x7507d8b4, 0x6ceeb94e,
        0x52eb6ba2, 0x50642905,
    ]),
    to_babybear_array([
        0x05453f3f, 0x06349efc, 0x6922787c, 0x04bfff9c, 0x768c714a, 0x3e9ff21a, 0x15737c9c,
        0x2229c807, 0x0d47f88c, 0x097e0ecc, 0x27eadba0, 0x2d7d29e4, 0x3502aaa0, 0x0f475fd7,
        0x29fbda49, 0x018afffd,
    ]),
    to_babybear_array([
        0x0315b618, 0x6d4497d1, 0x1b171d9e, 0x52861abd, 0x2e5d0501, 0x3ec8646c, 0x6e5f250a,
        0x148ae8e6, 0x17f5fa4a, 0x3e66d284, 0x0051aa3b, 0x483f7913, 0x2cfe5f15, 0x023427ca,
        0x2cc78315, 0x1e36ea47,
    ]),
    to_babybear_array([
        0x7290a80d, 0x6f7e5329, 0x598ec8a8, 0x76a859a0, 0x6559e868, 0x657b83af, 0x13271d3f,
        0x1f876063, 0x0aeeae37, 0x706e9ca6, 0x46400cee, 0x72a05c26, 0x2c589c9e, 0x20bd37a7,
        0x6a2d3d10, 0x20523767,
    ]),
    to_babybear_array([
        0x5b8fe9c4, 0x2aa501d6, 0x1e01ac3e, 0x1448bc54, 0x5ce5ad1c, 0x4918a14d, 0x2c46a83f,
        0x4fcf6876, 0x61d8d5c8, 0x6ddf4ff9, 0x11fda4d3, 0x02933a8f, 0x170eaf81, 0x5a9c314f,
        0x49a12590, 0x35ec52a1,
    ]),
    to_babybear_array([
        0x58eb1611, 0x5e481e65, 0x367125c9, 0x0eba33ba, 0x1fc28ded, 0x066399ad, 0x0cbec0ea,
        0x75fd1af0, 0x50f5bf4e, 0x643d5f41, 0x6f4fe718, 0x5b3cbbde, 0x1e3afb3e, 0x296fb027,
        0x45e1547b, 0x4a8db2ab,
    ]),
    to_babybear_array([
        0x59986d19, 0x30bcdfa3, 0x1db63932, 0x1d7c2824, 0x53b33681, 0x0673b747, 0x038a98a3,
        0x2c5bce60, 0x351979cd, 0x5008fb73, 0x547bca78, 0x711af481, 0x3f93bf64, 0x644d987b,
        0x3c8bcd87, 0x608758b8,
    ]),
];

pub const HL_BABYBEAR_16_INTERNAL_ROUND_CONSTANTS: [BabyBear; 13] = to_babybear_array([
    0x5a8053c0, 0x693be639, 0x3858867d, 0x19334f6b, 0x128f0fd8, 0x4e2b1ccb, 0x61210ce0, 0x3c318939,
    0x0b5b2f22, 0x2edb11d5, 0x213effdf, 0x0cac4606, 0x241af16d,
]);

#[cfg(test)]
mod tests {
    use core::array;

    use p3_field::AbstractField;
    use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral, Poseidon2ExternalMatrixHL};

    use super::*;

    type F = BabyBear;

    // A function which recreates the poseidon2 implementation in
    // https://github.com/HorizenLabs/poseidon2
    fn hl_poseidon2_babybear_width_16(input: &mut [F; 16]) {
        const WIDTH: usize = 16;
        const D: u64 = 7;
        const ROUNDS_F: usize = 8;
        const ROUNDS_P: usize = 13;

        // Our Poseidon2 implementation.
        let poseidon2: Poseidon2<
            BabyBear,
            Poseidon2ExternalMatrixHL,
            DiffusionMatrixBabybearHL,
            WIDTH,
            D,
        > = Poseidon2::new(
            ROUNDS_F,
            HL_BABYBEAR_16_EXTERNAL_ROUND_CONSTANTS.to_vec(),
            Poseidon2ExternalMatrixHL,
            ROUNDS_P,
            HL_BABYBEAR_16_INTERNAL_ROUND_CONSTANTS.to_vec(),
            DiffusionMatrixBabybearHL,
        );
        poseidon2.permute_mut(input);
    }

    fn poseidon2_babybear_width_16(input: &mut [F; 16]) {
        const WIDTH: usize = 16;
        const D: u64 = 7;
        const ROUNDS_F: usize = 8;
        const ROUNDS_P: usize = 13;

        // Need to make some round constants. We use sage to get a random element which we use as a seed.
        // set_random_seed(11111);
        // ZZ.random_element(2**31);
        const SIMPLE_SEED: u32 = 899088431;
        let external_round_constants: [[BabyBear; 16]; 8] = array::from_fn(|j| {
            array::from_fn(|i| F::from_wrapped_u32((i + 16 * j) as u32 + SIMPLE_SEED).exp_u64(11))
        });
        let internal_round_constants: [BabyBear; 13] =
            array::from_fn(|i| F::from_wrapped_u32(i as u32 + SIMPLE_SEED).exp_u64(7));

        // Our Poseidon2 implementation.
        let poseidon2: Poseidon2<
            BabyBear,
            Poseidon2ExternalMatrixGeneral,
            DiffusionMatrixBabybear,
            WIDTH,
            D,
        > = Poseidon2::new(
            ROUNDS_F,
            external_round_constants.to_vec(),
            Poseidon2ExternalMatrixGeneral,
            ROUNDS_P,
            internal_round_constants.to_vec(),
            DiffusionMatrixBabybear,
        );
        poseidon2.permute_mut(input);
    }

    fn poseidon2_babybear_width_24(input: &mut [F; 24]) {
        const WIDTH: usize = 24;
        const D: u64 = 7;
        const ROUNDS_F: usize = 8;
        const ROUNDS_P: usize = 21;

        // Need to make some round constants. We use sage to get a random element which we use as a seed.
        // set_random_seed(11111);
        // ZZ.random_element(2**31);
        const SIMPLE_SEED: u32 = 899088431;
        let external_round_constants: [[BabyBear; 24]; 8] = array::from_fn(|j| {
            array::from_fn(|i| F::from_wrapped_u32((i + 24 * j) as u32 + SIMPLE_SEED).exp_u64(11))
        });
        let internal_round_constants: [BabyBear; 21] =
            array::from_fn(|i| F::from_wrapped_u32(i as u32 + SIMPLE_SEED).exp_u64(7));

        // Our Poseidon2 implementation.
        let poseidon2: Poseidon2<
            BabyBear,
            Poseidon2ExternalMatrixGeneral,
            DiffusionMatrixBabybear,
            WIDTH,
            D,
        > = Poseidon2::new(
            ROUNDS_F,
            external_round_constants.to_vec(),
            Poseidon2ExternalMatrixGeneral,
            ROUNDS_P,
            internal_round_constants.to_vec(),
            DiffusionMatrixBabybear,
        );
        poseidon2.permute_mut(input);
    }

    /// Test on the constant 0 input.
    #[test]
    fn test_poseidon2_hl_width_16_zeroes() {
        let mut input: [F; 16] = [0_u32; 16].map(F::from_wrapped_u32);

        let expected: [F; 16] = [
            1337856655, 1843094405, 328115114, 964209316, 1365212758, 1431554563, 210126733,
            1214932203, 1929553766, 1647595522, 1496863878, 324695999, 1569728319, 1634598391,
            597968641, 679989771,
        ]
        .map(F::from_canonical_u32);
        hl_poseidon2_babybear_width_16(&mut input);
        assert_eq!(input, expected);
    }

    /// Test on the input 0..16.
    #[test]
    fn test_poseidon2_hl_width_16_range() {
        let mut input: [F; 16] = array::from_fn(|i| F::from_wrapped_u32(i as u32));

        let expected: [F; 16] = [
            896560466, 771677727, 128113032, 1378976435, 160019712, 1452738514, 682850273,
            223500421, 501450187, 1804685789, 1671399593, 1788755219, 1736880027, 1352180784,
            1928489698, 1128802977,
        ]
        .map(F::from_canonical_u32);
        hl_poseidon2_babybear_width_16(&mut input);
        assert_eq!(input, expected);
    }

    /// Test on a roughly random input.
    /// This random input is generated by the following sage code:
    /// set_random_seed(2468)
    /// vector([ZZ.random_element(2**31) for t in range(16)])
    #[test]
    fn test_poseidon2_hl_width_16_random() {
        let mut input: [F; 16] = [
            1179785652, 1291567559, 66272299, 471640172, 653876821, 478855335, 871063984,
            540251327, 1506944720, 1403776782, 770420443, 126472305, 1535928603, 1017977016,
            818646757, 359411429,
        ]
        .map(F::from_wrapped_u32);

        let expected: [F; 16] = [
            1736862924, 1950079822, 952072292, 1965704005, 236226362, 1113998185, 1624488077,
            391891139, 1194078311, 1040746778, 1898067001, 774167026, 193702242, 859952892,
            732204701, 1744970965,
        ]
        .map(F::from_canonical_u32);

        hl_poseidon2_babybear_width_16(&mut input);
        assert_eq!(input, expected);
    }

    /// Test on a roughly random input.
    /// This random input is generated by the following sage code:
    /// set_random_seed(2468)
    /// vector([ZZ.random_element(2**31) for t in range(16)])
    #[test]
    fn test_poseidon2_width_16_random() {
        let mut input: [F; 16] = [
            1179785652, 1291567559, 66272299, 471640172, 653876821, 478855335, 871063984,
            540251327, 1506944720, 1403776782, 770420443, 126472305, 1535928603, 1017977016,
            818646757, 359411429,
        ]
        .map(F::from_wrapped_u32);

        let expected: [F; 16] = [
            1248638253, 979391849, 357005207, 914460411, 1344873150, 1266301567, 462106912,
            1416072532, 1666037773, 210481833, 1286244123, 680556324, 1204134065, 167595403,
            634347617, 917336235,
        ]
        .map(F::from_canonical_u32);

        poseidon2_babybear_width_16(&mut input);
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
            1960609465, 35511609, 1437571456, 1200188766, 1073548088, 1433200340, 35789002,
            1138157699, 202307864, 1222136958, 138511458, 1630086227, 1633931354, 489046210,
            287720696, 781743383, 246718259, 402408558, 517880637, 57645822, 86443773, 636182948,
            562190295, 567338916,
        ]
        .map(F::from_canonical_u32);

        poseidon2_babybear_width_24(&mut input);
        assert_eq!(input, expected);
    }
}
