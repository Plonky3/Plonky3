use p3_poseidon2::{matmul_internal, DiffusionPermutation};
use p3_symmetric::Permutation;

use crate::{
    BabyBear, DiffusionMatrixBabybear, PackedBabyBearAVX512, MATRIX_DIAG_16_BABYBEAR_MONTY,
    MATRIX_DIAG_24_BABYBEAR_MONTY,
};

impl Permutation<[PackedBabyBearAVX512; 16]> for DiffusionMatrixBabybear {
    fn permute_mut(&self, state: &mut [PackedBabyBearAVX512; 16]) {
        matmul_internal::<BabyBear, PackedBabyBearAVX512, 16>(state, MATRIX_DIAG_16_BABYBEAR_MONTY);
    }
}

impl DiffusionPermutation<PackedBabyBearAVX512, 16> for DiffusionMatrixBabybear {}

impl Permutation<[PackedBabyBearAVX512; 24]> for DiffusionMatrixBabybear {
    fn permute_mut(&self, state: &mut [PackedBabyBearAVX512; 24]) {
        matmul_internal::<BabyBear, PackedBabyBearAVX512, 24>(state, MATRIX_DIAG_24_BABYBEAR_MONTY);
    }
}

impl DiffusionPermutation<PackedBabyBearAVX512, 24> for DiffusionMatrixBabybear {}
