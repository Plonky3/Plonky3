//! Diffusion matrices for Babybear16 and Babybear24.
//!
//! Reference: https://github.com/HorizenLabs/poseidon2/blob/main/plain_implementations/src/poseidon2/poseidon2_instance_babybear.rs

use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_symmetric::Permutation;

use crate::diffusion::matmul_internal;
use crate::DiffusionPermutation;

pub const MATRIX_DIAG_16_BABYBEAR: [u64; 16] = [
    0x0a63_2d94, 0x6db6_57b7, 0x56fb_dc9e, 0x052b_3d8a, 0x3374_5201, 0x5c03_108c, 0x0beb_a37b, 0x258c_2e8b,
    0x1202_9f39, 0x6949_09ce, 0x6d23_1724, 0x21c3_b222, 0x3c09_04a5, 0x01d6_acda, 0x2770_5c83, 0x5231_c802,
];

pub const MATRIX_DIAG_24_BABYBEAR: [u64; 24] = [
    0x4091_33f0, 0x1667_a8a1, 0x06a6_c7b6, 0x6f53_160e, 0x273b_11d1, 0x0317_6c5d, 0x72f9_bbf9, 0x73ce_ba91,
    0x5cde_f81d, 0x0139_3285, 0x46da_ee06, 0x065d_7ba6, 0x52d7_2d6f, 0x05dd_05e0, 0x3bab_4b63, 0x6ada_3842,
    0x2fc5_fbec, 0x770d_61b0, 0x5715_aae9, 0x03ef_0e90, 0x75b6_c770, 0x242a_df5f, 0x00d0_ca4c, 0x36c0_e388,
];

#[derive(Debug, Clone, Default)]
pub struct DiffusionMatrixBabybear;

impl<AF: AbstractField<F = BabyBear>> Permutation<[AF; 16]> for DiffusionMatrixBabybear {
    fn permute_mut(&self, state: &mut [AF; 16]) {
        matmul_internal::<AF, 16>(state, MATRIX_DIAG_16_BABYBEAR);
    }
}

impl<AF: AbstractField<F = BabyBear>> DiffusionPermutation<AF, 16> for DiffusionMatrixBabybear {}

impl<AF: AbstractField<F = BabyBear>> Permutation<[AF; 24]> for DiffusionMatrixBabybear {
    fn permute_mut(&self, state: &mut [AF; 24]) {
        matmul_internal::<AF, 24>(state, MATRIX_DIAG_24_BABYBEAR);
    }
}

impl DiffusionPermutation<BabyBear, 24> for DiffusionMatrixBabybear {}
