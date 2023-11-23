//! Diffusion matrices for Goldilocks8, Goldilocks12, Goldilocks16, and Goldilocks20.
//!
//! Reference: https://github.com/HorizenLabs/poseidon2/blob/main/plain_implementations/src/poseidon2/poseidon2_instance_goldilocks.rs

use p3_field::AbstractField;
use p3_goldilocks::Goldilocks;
use p3_symmetric::Permutation;

use crate::diffusion::matmul_internal;
use crate::DiffusionPermutation;

pub const MATRIX_DIAG_8_GOLDILOCKS: [u64; 8] = [
    0xa988_11a1_fed4_e3a5,
    0x1cc4_8b54_f377_e2a0,
    0xe40c_d4f6_c560_9a26,
    0x11de_79eb_ca97_a4a3,
    0x9177_c73d_8b7e_929c,
    0x2a6f_e808_5797_e791,
    0x3de6_e933_29f8_d5ad,
    0x3f7a_f912_5da9_62fe,
];

pub const MATRIX_DIAG_12_GOLDILOCKS: [u64; 12] = [
    0xc3b6_c08e_23ba_9300,
    0xd84b_5de9_4a32_4fb6,
    0x0d0c_371c_5b35_b84f,
    0x7964_f570_e718_8037,
    0x5daf_18bb_d996_604b,
    0x6743_bc47_b959_5257,
    0x5528_b936_2c59_bb70,
    0xac45_e25b_7127_b68b,
    0xa207_7d7d_fbb6_06b5,
    0xf3fa_ac6f_aee3_78ae,
    0x0c63_88b5_1545_e883,
    0xd27d_bb69_4491_7b60,
];

pub const MATRIX_DIAG_16_GOLDILOCKS: [u64; 16] = [
    0xde9b_91a4_67d6_afc0,
    0xc5f1_6b9c_76a9_be17,
    0x0ab0_fef2_d540_ac55,
    0x3001_d270_09d0_5773,
    0xed23_b1f9_06d3_d9eb,
    0x5ce7_3743_cba9_7054,
    0x1c3b_ab94_4af4_ba24,
    0x2faa_1058_54db_afae,
    0x53ff_b3ae_6d42_1a10,
    0xbcda_9df8_884b_a396,
    0xfc12_73e4_a318_07bb,
    0xc779_5257_3d51_42c0,
    0x5668_3339_a819_b85e,
    0x328f_cbd8_f0dd_c8eb,
    0xb510_1e30_3fce_9cb7,
    0x7744_87b8_c400_89bb,
];

pub const MATRIX_DIAG_20_GOLDILOCKS: [u64; 20] = [
    0x95c3_81fd_a3b1_fa57,
    0xf36f_e9eb_1288_f42c,
    0x89f5_dcdf_ef27_7944,
    0x106f_22ea_deb3_e2d2,
    0x684e_31a2_530e_5111,
    0x2743_5c5d_89fd_148e,
    0x3ebe_d31c_414d_bf17,
    0xfd45_b0b2_d294_e3cc,
    0x48c9_0447_3a7f_6dbf,
    0xe0d1_b678_0929_5b4d,
    0xddd1_941e_9d19_9dcb,
    0x8cfe_534e_eb74_2219,
    0xa6e5_261d_9e3b_8524,
    0x6897_ee5e_d0f8_2c1b,
    0x0e7d_cd07_39ee_5f78,
    0x4932_53f3_d0d3_2363,
    0xbb27_37f5_845f_05c0,
    0xa187_e810_b06a_d903,
    0xb635_b995_936c_4918,
    0x0b36_94a9_40bd_2394,
];

#[derive(Debug, Clone, Default)]
pub struct DiffusionMatrixGoldilocks;

impl<AF: AbstractField<F = Goldilocks>> Permutation<[AF; 8]> for DiffusionMatrixGoldilocks {
    fn permute_mut(&self, state: &mut [AF; 8]) {
        matmul_internal::<AF, 8>(state, MATRIX_DIAG_8_GOLDILOCKS);
    }
}

impl<AF: AbstractField<F = Goldilocks>> DiffusionPermutation<AF, 8> for DiffusionMatrixGoldilocks {}

impl<AF: AbstractField<F = Goldilocks>> Permutation<[AF; 12]> for DiffusionMatrixGoldilocks {
    fn permute_mut(&self, state: &mut [AF; 12]) {
        matmul_internal::<AF, 12>(state, MATRIX_DIAG_12_GOLDILOCKS);
    }
}

impl<AF: AbstractField<F = Goldilocks>> DiffusionPermutation<AF, 12> for DiffusionMatrixGoldilocks {}

impl<AF: AbstractField<F = Goldilocks>> Permutation<[AF; 16]> for DiffusionMatrixGoldilocks {
    fn permute_mut(&self, state: &mut [AF; 16]) {
        matmul_internal::<AF, 16>(state, MATRIX_DIAG_16_GOLDILOCKS);
    }
}

impl<AF: AbstractField<F = Goldilocks>> DiffusionPermutation<AF, 16> for DiffusionMatrixGoldilocks {}

impl<AF: AbstractField<F = Goldilocks>> Permutation<[AF; 20]> for DiffusionMatrixGoldilocks {
    fn permute_mut(&self, state: &mut [AF; 20]) {
        matmul_internal::<AF, 20>(state, MATRIX_DIAG_20_GOLDILOCKS);
    }
}

impl<AF: AbstractField<F = Goldilocks>> DiffusionPermutation<AF, 20> for DiffusionMatrixGoldilocks {}
