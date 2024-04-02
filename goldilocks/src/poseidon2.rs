//! Diffusion matrices for Goldilocks8, Goldilocks12, Goldilocks16, and Goldilocks20.
//!
//! Reference: https://github.com/HorizenLabs/poseidon2/blob/main/plain_implementations/src/poseidon2/poseidon2_instance_goldilocks.rs

use p3_field::AbstractField;
use p3_poseidon2::{matmul_internal, DiffusionPermutation};
use p3_symmetric::Permutation;

use crate::{to_goldilocks_array, Goldilocks};

pub const MATRIX_DIAG_8_GOLDILOCKS_U64: [u64; 8] = [
    0xa98811a1fed4e3a5,
    0x1cc48b54f377e2a0,
    0xe40cd4f6c5609a26,
    0x11de79ebca97a4a3,
    0x9177c73d8b7e929c,
    0x2a6fe8085797e791,
    0x3de6e93329f8d5ad,
    0x3f7af9125da962fe,
];

pub const MATRIX_DIAG_12_GOLDILOCKS_U64: [u64; 12] = [
    0xc3b6c08e23ba9300,
    0xd84b5de94a324fb6,
    0x0d0c371c5b35b84f,
    0x7964f570e7188037,
    0x5daf18bbd996604b,
    0x6743bc47b9595257,
    0x5528b9362c59bb70,
    0xac45e25b7127b68b,
    0xa2077d7dfbb606b5,
    0xf3faac6faee378ae,
    0x0c6388b51545e883,
    0xd27dbb6944917b60,
];

pub const MATRIX_DIAG_16_GOLDILOCKS_U64: [u64; 16] = [
    0xde9b91a467d6afc0,
    0xc5f16b9c76a9be17,
    0x0ab0fef2d540ac55,
    0x3001d27009d05773,
    0xed23b1f906d3d9eb,
    0x5ce73743cba97054,
    0x1c3bab944af4ba24,
    0x2faa105854dbafae,
    0x53ffb3ae6d421a10,
    0xbcda9df8884ba396,
    0xfc1273e4a31807bb,
    0xc77952573d5142c0,
    0x56683339a819b85e,
    0x328fcbd8f0ddc8eb,
    0xb5101e303fce9cb7,
    0x774487b8c40089bb,
];

pub const MATRIX_DIAG_20_GOLDILOCKS_U64: [u64; 20] = [
    0x95c381fda3b1fa57,
    0xf36fe9eb1288f42c,
    0x89f5dcdfef277944,
    0x106f22eadeb3e2d2,
    0x684e31a2530e5111,
    0x27435c5d89fd148e,
    0x3ebed31c414dbf17,
    0xfd45b0b2d294e3cc,
    0x48c904473a7f6dbf,
    0xe0d1b67809295b4d,
    0xddd1941e9d199dcb,
    0x8cfe534eeb742219,
    0xa6e5261d9e3b8524,
    0x6897ee5ed0f82c1b,
    0x0e7dcd0739ee5f78,
    0x493253f3d0d32363,
    0xbb2737f5845f05c0,
    0xa187e810b06ad903,
    0xb635b995936c4918,
    0x0b3694a940bd2394,
];

// Convert the above arrays of u64's into arrays of Goldilocks field elements.
const MATRIX_DIAG_8_GOLDILOCKS: [Goldilocks; 8] = to_goldilocks_array(MATRIX_DIAG_8_GOLDILOCKS_U64);
const MATRIX_DIAG_12_GOLDILOCKS: [Goldilocks; 12] =
    to_goldilocks_array(MATRIX_DIAG_12_GOLDILOCKS_U64);
const MATRIX_DIAG_16_GOLDILOCKS: [Goldilocks; 16] =
    to_goldilocks_array(MATRIX_DIAG_16_GOLDILOCKS_U64);
const MATRIX_DIAG_20_GOLDILOCKS: [Goldilocks; 20] =
    to_goldilocks_array(MATRIX_DIAG_20_GOLDILOCKS_U64);

#[derive(Debug, Clone, Default)]
pub struct DiffusionMatrixGoldilocks;

impl<AF: AbstractField<F = Goldilocks>> Permutation<[AF; 8]> for DiffusionMatrixGoldilocks {
    fn permute_mut(&self, state: &mut [AF; 8]) {
        matmul_internal::<Goldilocks, AF, 8>(state, MATRIX_DIAG_8_GOLDILOCKS);
    }
}

impl<AF: AbstractField<F = Goldilocks>> DiffusionPermutation<AF, 8> for DiffusionMatrixGoldilocks {}

impl<AF: AbstractField<F = Goldilocks>> Permutation<[AF; 12]> for DiffusionMatrixGoldilocks {
    fn permute_mut(&self, state: &mut [AF; 12]) {
        matmul_internal::<Goldilocks, AF, 12>(state, MATRIX_DIAG_12_GOLDILOCKS);
    }
}

impl<AF: AbstractField<F = Goldilocks>> DiffusionPermutation<AF, 12> for DiffusionMatrixGoldilocks {}

impl<AF: AbstractField<F = Goldilocks>> Permutation<[AF; 16]> for DiffusionMatrixGoldilocks {
    fn permute_mut(&self, state: &mut [AF; 16]) {
        matmul_internal::<Goldilocks, AF, 16>(state, MATRIX_DIAG_16_GOLDILOCKS);
    }
}

impl<AF: AbstractField<F = Goldilocks>> DiffusionPermutation<AF, 16> for DiffusionMatrixGoldilocks {}

impl<AF: AbstractField<F = Goldilocks>> Permutation<[AF; 20]> for DiffusionMatrixGoldilocks {
    fn permute_mut(&self, state: &mut [AF; 20]) {
        matmul_internal::<Goldilocks, AF, 20>(state, MATRIX_DIAG_20_GOLDILOCKS);
    }
}

impl<AF: AbstractField<F = Goldilocks>> DiffusionPermutation<AF, 20> for DiffusionMatrixGoldilocks {}

#[cfg(test)]
mod tests {
    use core::array;

    use p3_poseidon2::{
        Poseidon2, HL_GOLDILOCKS_8_EXTERNAL_ROUND_CONSTANTS,
        HL_GOLDILOCKS_8_INTERNAL_ROUND_CONSTANTS,
    };

    use super::*;

    type F = Goldilocks;

    #[test]
    fn test_poseidon2_constants() {
        let monty_constant = MATRIX_DIAG_8_GOLDILOCKS_U64.map(Goldilocks::from_canonical_u64);
        assert_eq!(monty_constant, MATRIX_DIAG_8_GOLDILOCKS);

        let monty_constant = MATRIX_DIAG_12_GOLDILOCKS_U64.map(Goldilocks::from_canonical_u64);
        assert_eq!(monty_constant, MATRIX_DIAG_12_GOLDILOCKS);

        let monty_constant = MATRIX_DIAG_16_GOLDILOCKS_U64.map(Goldilocks::from_canonical_u64);
        assert_eq!(monty_constant, MATRIX_DIAG_16_GOLDILOCKS);

        let monty_constant = MATRIX_DIAG_20_GOLDILOCKS_U64.map(Goldilocks::from_canonical_u64);
        assert_eq!(monty_constant, MATRIX_DIAG_20_GOLDILOCKS);
    }

    // A function which recreates the poseidon2 implementation in
    // https://github.com/HorizenLabs/poseidon2
    fn hl_poseidon2_goldilocks_width_8(input: &mut [F; 8]) {
        const WIDTH: usize = 8;
        const D: u64 = 7;
        const ROUNDS_F: usize = 8;
        const ROUNDS_P: usize = 22;

        // Our Poseidon2 implementation.
        let poseidon2: Poseidon2<Goldilocks, DiffusionMatrixGoldilocks, WIDTH, D> = Poseidon2::new(
            ROUNDS_F,
            HL_GOLDILOCKS_8_EXTERNAL_ROUND_CONSTANTS
                .map(to_goldilocks_array)
                .to_vec(),
            ROUNDS_P,
            to_goldilocks_array(HL_GOLDILOCKS_8_INTERNAL_ROUND_CONSTANTS).to_vec(),
            DiffusionMatrixGoldilocks,
        );

        poseidon2.permute_mut(input);
    }

    /// Test on the constant 0 input.
    #[test]
    fn test_poseidon2_width_8_zeroes() {
        let mut input: [F; 8] = [0_u64; 8].map(F::from_wrapped_u64);

        let expected: [F; 8] = [
            4214787979728720400,
            12324939279576102560,
            10353596058419792404,
            15456793487362310586,
            10065219879212154722,
            16227496357546636742,
            2959271128466640042,
            14285409611125725709,
        ]
        .map(F::from_canonical_u64);
        hl_poseidon2_goldilocks_width_8(&mut input);
        assert_eq!(input, expected);
    }

    /// Test on the input 0..16.
    #[test]
    fn test_poseidon2_width_8_range() {
        let mut input: [F; 8] = array::from_fn(|i| F::from_wrapped_u64(i as u64));

        let expected: [F; 8] = [
            14266028122062624699,
            5353147180106052723,
            15203350112844181434,
            17630919042639565165,
            16601551015858213987,
            10184091939013874068,
            16774100645754596496,
            12047415603622314780,
        ]
        .map(F::from_canonical_u64);
        hl_poseidon2_goldilocks_width_8(&mut input);
        assert_eq!(input, expected);
    }

    /// Test on a roughly random input.
    /// This random input is generated by the following sage code:
    /// set_random_seed(2468)
    /// vector([ZZ.random_element(2**31) for t in range(16)])
    #[test]
    fn test_poseidon2_width_8_random() {
        let mut input: [F; 8] = [
            5116996373749832116,
            8931548647907683339,
            17132360229780760684,
            11280040044015983889,
            11957737519043010992,
            15695650327991256125,
            17604752143022812942,
            543194415197607509,
        ]
        .map(F::from_wrapped_u64);

        let expected: [F; 8] = [
            1831346684315917658,
            13497752062035433374,
            12149460647271516589,
            15656333994315312197,
            4671534937670455565,
            3140092508031220630,
            4251208148861706881,
            6973971209430822232,
        ]
        .map(F::from_canonical_u64);

        hl_poseidon2_goldilocks_width_8(&mut input);
        assert_eq!(input, expected);
    }
}
