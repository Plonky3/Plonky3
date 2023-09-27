//! Diffusion matrices for Goldilocks8, Goldilocks12, Goldilocks16, and Goldilocks20.
//!
//! Reference: https://github.com/HorizenLabs/poseidon2/blob/main/plain_implementations/src/poseidon2/poseidon2_instance_goldilocks.rs

use p3_goldilocks::Goldilocks;
use p3_symmetric::permutation::Permutation;

use crate::diffusion::matmul_internal;
use crate::DiffusionPermutation;

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
