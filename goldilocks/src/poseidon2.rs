//! Implementation of Poseidon2, see: https://eprint.iacr.org/2023/323

//! For now we recreate the implementation given in:
//! https://github.com/HorizenLabs/poseidon2/blob/main/plain_implementations/src/poseidon2/poseidon2_instance_goldilocks.rs
//! This uses the constants below along with using the 4x4 matrix:
//! [[5, 7, 1, 3], [4, 6, 1, 1], [1, 3, 5, 7], [1, 1, 4, 6]]
//! to build the 4t x 4t matrix used for the external (full) rounds).

//! Long term we will use more optimised internal and external linear layers.

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

pub const HL_GOLDILOCKS_8_EXTERNAL_ROUND_CONSTANTS: [[u64; 8]; 8] = [
    [
        0xdd5743e7f2a5a5d9,
        0xcb3a864e58ada44b,
        0xffa2449ed32f8cdc,
        0x42025f65d6bd13ee,
        0x7889175e25506323,
        0x34b98bb03d24b737,
        0xbdcc535ecc4faa2a,
        0x5b20ad869fc0d033,
    ],
    [
        0xf1dda5b9259dfcb4,
        0x27515210be112d59,
        0x4227d1718c766c3f,
        0x26d333161a5bd794,
        0x49b938957bf4b026,
        0x4a56b5938b213669,
        0x1120426b48c8353d,
        0x6b323c3f10a56cad,
    ],
    [
        0xce57d6245ddca6b2,
        0xb1fc8d402bba1eb1,
        0xb5c5096ca959bd04,
        0x6db55cd306d31f7f,
        0xc49d293a81cb9641,
        0x1ce55a4fe979719f,
        0xa92e60a9d178a4d1,
        0x002cc64973bcfd8c,
    ],
    [
        0xcea721cce82fb11b,
        0xe5b55eb8098ece81,
        0x4e30525c6f1ddd66,
        0x43c6702827070987,
        0xaca68430a7b5762a,
        0x3674238634df9c93,
        0x88cee1c825e33433,
        0xde99ae8d74b57176,
    ],
    [
        0x014ef1197d341346,
        0x9725e20825d07394,
        0xfdb25aef2c5bae3b,
        0xbe5402dc598c971e,
        0x93a5711f04cdca3d,
        0xc45a9a5b2f8fb97b,
        0xfe8946a924933545,
        0x2af997a27369091c,
    ],
    [
        0xaa62c88e0b294011,
        0x058eb9d810ce9f74,
        0xb3cb23eced349ae4,
        0xa3648177a77b4a84,
        0x43153d905992d95d,
        0xf4e2a97cda44aa4b,
        0x5baa2702b908682f,
        0x082923bdf4f750d1,
    ],
    [
        0x98ae09a325893803,
        0xf8a6475077968838,
        0xceb0735bf00b2c5f,
        0x0a1a5d953888e072,
        0x2fcb190489f94475,
        0xb5be06270dec69fc,
        0x739cb934b09acf8b,
        0x537750b75ec7f25b,
    ],
    [
        0xe9dd318bae1f3961,
        0xf7462137299efe1a,
        0xb1f6b8eee9adb940,
        0xbdebcc8a809dfe6b,
        0x40fc1f791b178113,
        0x3ac1c3362d014864,
        0x9a016184bdb8aeba,
        0x95f2394459fbc25e,
    ],
];
pub const HL_GOLDILOCKS_8_INTERNAL_ROUND_CONSTANTS: [u64; 22] = [
    0x488897d85ff51f56,
    0x1140737ccb162218,
    0xa7eeb9215866ed35,
    0x9bd2976fee49fcc9,
    0xc0c8f0de580a3fcc,
    0x4fb2dae6ee8fc793,
    0x343a89f35f37395b,
    0x223b525a77ca72c8,
    0x56ccb62574aaa918,
    0xc4d507d8027af9ed,
    0xa080673cf0b7e95c,
    0xf0184884eb70dcf8,
    0x044f10b0cb3d5c69,
    0xe9e3f7993938f186,
    0x1b761c80e772f459,
    0x606cec607a1b5fac,
    0x14a0c2e1d45f03cd,
    0x4eace8855398574f,
    0xf905ca7103eff3e6,
    0xf8c8f8d20862c059,
    0xb524fe8bdd678e5a,
    0xfbb7865901a1ec41,
];

#[cfg(test)]
mod tests {
    use core::array;

    use p3_poseidon2::Poseidon2;

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
