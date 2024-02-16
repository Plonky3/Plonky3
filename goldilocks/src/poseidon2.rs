//! Diffusion matrices for Goldilocks8, Goldilocks12, Goldilocks16, and Goldilocks20.
//!
//! Reference: https://github.com/HorizenLabs/poseidon2/blob/main/plain_implementations/src/poseidon2/poseidon2_instance_goldilocks.rs


use p3_poseidon2::{DiffusionPermutation, matmul_internal};
use p3_symmetric::Permutation;

use crate::{Goldilocks};

pub const MATRIX_DIAG_8_GOLDILOCKS: [u64; 8] = [
    0xa98811a1fed4e3a5,
    0x1cc48b54f377e2a0,
    0xe40cd4f6c5609a26,
    0x11de79ebca97a4a3,
    0x9177c73d8b7e929c,
    0x2a6fe8085797e791,
    0x3de6e93329f8d5ad,
    0x3f7af9125da962fe,
];

pub const MATRIX_DIAG_12_GOLDILOCKS: [u64; 12] = [
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

pub const MATRIX_DIAG_16_GOLDILOCKS: [u64; 16] = [
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

pub const MATRIX_DIAG_20_GOLDILOCKS: [u64; 20] = [
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

#[derive(Debug, Clone, Default)]
pub struct DiffusionMatrixGoldilocks;

impl Permutation<[Goldilocks; 8]> for DiffusionMatrixGoldilocks {
    fn permute_mut(&self, state: &mut [Goldilocks; 8]) {
        matmul_internal::<Goldilocks, 8>(state, MATRIX_DIAG_8_GOLDILOCKS);
    }
}

impl DiffusionPermutation<Goldilocks, 8> for DiffusionMatrixGoldilocks {}

impl Permutation<[Goldilocks; 12]> for DiffusionMatrixGoldilocks {
    fn permute_mut(&self, state: &mut [Goldilocks; 12]) {
        matmul_internal::<Goldilocks, 12>(state, MATRIX_DIAG_12_GOLDILOCKS);
    }
}

impl DiffusionPermutation<Goldilocks, 12> for DiffusionMatrixGoldilocks {}

impl Permutation<[Goldilocks; 16]> for DiffusionMatrixGoldilocks {
    fn permute_mut(&self, state: &mut [Goldilocks; 16]) {
        matmul_internal::<Goldilocks, 16>(state, MATRIX_DIAG_16_GOLDILOCKS);
    }
}

impl DiffusionPermutation<Goldilocks, 16> for DiffusionMatrixGoldilocks {}

impl Permutation<[Goldilocks; 20]> for DiffusionMatrixGoldilocks {
    fn permute_mut(&self, state: &mut [Goldilocks; 20]) {
        matmul_internal::<Goldilocks, 20>(state, MATRIX_DIAG_20_GOLDILOCKS);
    }
}

impl DiffusionPermutation<Goldilocks, 20> for DiffusionMatrixGoldilocks {}


#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    
    use ark_ff::{BigInteger, PrimeField};
    use p3_field::AbstractField;
    use crate::Goldilocks;
    use p3_symmetric::Permutation;
    use rand::Rng;
    use zkhash::fields::goldilocks::FpGoldiLocks;
    use zkhash::poseidon2::poseidon2::Poseidon2 as Poseidon2Ref;  
    use zkhash::poseidon2::poseidon2_instance_goldilocks::{
        POSEIDON2_GOLDILOCKS_12_PARAMS, POSEIDON2_GOLDILOCKS_8_PARAMS, RC12, RC8,
    };

    use crate::DiffusionMatrixGoldilocks;
    use p3_poseidon2::{Poseidon2};

    fn goldilocks_from_ark_ff(input: FpGoldiLocks) -> Goldilocks {
        let as_bigint = input.into_bigint();
        let mut as_bytes = as_bigint.to_bytes_le();
        as_bytes.resize(8, 0);
        let as_u64 = u64::from_le_bytes(as_bytes[0..8].try_into().unwrap());
        Goldilocks::from_wrapped_u64(as_u64)
    }

    #[test]
    fn test_poseidon2_goldilocks_width_8() {
        const WIDTH: usize = 8;
        const D: u64 = 7;
        const ROUNDS_F: usize = 8;
        const ROUNDS_P: usize = 22;

        type F = Goldilocks;

        let mut rng = rand::thread_rng();

        // Poiseidon2 reference implementation from zkhash repo.
        let poseidon2_ref = Poseidon2Ref::new(&POSEIDON2_GOLDILOCKS_8_PARAMS);

        // Copy over round constants from zkhash.
        let round_constants: Vec<[F; WIDTH]> = RC8
            .iter()
            .map(|vec| {
                vec.iter()
                    .cloned()
                    .map(goldilocks_from_ark_ff)
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            })
            .collect();

        // Our Poseidon2 implementation.
        let poseidon2: Poseidon2<Goldilocks, DiffusionMatrixGoldilocks, WIDTH, D> = Poseidon2::new(
            ROUNDS_F,
            ROUNDS_P,
            round_constants,
            DiffusionMatrixGoldilocks,
        );

        // Generate random input and convert to both Goldilocks field formats.
        let input_u64 = rng.gen::<[u64; WIDTH]>();
        let input_ref = input_u64
            .iter()
            .cloned()
            .map(FpGoldiLocks::from)
            .collect::<Vec<_>>();
        let input = input_u64.map(F::from_wrapped_u64);

        // Check that the conversion is correct.
        assert!(input_ref
            .iter()
            .zip(input.iter())
            .all(|(a, b)| goldilocks_from_ark_ff(*a) == *b));

        // Run reference implementation.
        let output_ref = poseidon2_ref.permutation(&input_ref);
        let expected: [F; WIDTH] = output_ref
            .iter()
            .cloned()
            .map(goldilocks_from_ark_ff)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        // Run our implementation.
        let mut output = input;
        poseidon2.permute_mut(&mut output);

        assert_eq!(output, expected);
    }

    #[test]
    fn test_poseidon2_goldilocks_width_12() {
        const WIDTH: usize = 12;
        const D: u64 = 7;
        const ROUNDS_F: usize = 8;
        const ROUNDS_P: usize = 22;

        type F = Goldilocks;

        let mut rng = rand::thread_rng();

        // Poiseidon2 reference implementation from zkhash repo.
        let poseidon2_ref = Poseidon2Ref::new(&POSEIDON2_GOLDILOCKS_12_PARAMS);

        // Copy over round constants from zkhash.
        let round_constants: Vec<[F; WIDTH]> = RC12
            .iter()
            .map(|vec| {
                vec.iter()
                    .cloned()
                    .map(goldilocks_from_ark_ff)
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            })
            .collect();

        // Our Poseidon2 implementation.
        let poseidon2: Poseidon2<Goldilocks, DiffusionMatrixGoldilocks, WIDTH, D> = Poseidon2::new(
            ROUNDS_F,
            ROUNDS_P,
            round_constants,
            DiffusionMatrixGoldilocks,
        );

        // Generate random input and convert to both Goldilocks field formats.
        let input_u64 = rng.gen::<[u64; WIDTH]>();
        let input_ref = input_u64
            .iter()
            .cloned()
            .map(FpGoldiLocks::from)
            .collect::<Vec<_>>();
        let input = input_u64.map(F::from_wrapped_u64);

        // Check that the conversion is correct.
        assert!(input_ref
            .iter()
            .zip(input.iter())
            .all(|(a, b)| goldilocks_from_ark_ff(*a) == *b));

        // Run reference implementation.
        let output_ref = poseidon2_ref.permutation(&input_ref);
        let expected: [F; WIDTH] = output_ref
            .iter()
            .cloned()
            .map(goldilocks_from_ark_ff)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        // Run our implementation.
        let mut output = input;
        poseidon2.permute_mut(&mut output);

        assert_eq!(output, expected);
    }
}