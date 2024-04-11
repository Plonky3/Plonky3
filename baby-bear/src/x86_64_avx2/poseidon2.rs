use p3_poseidon2::{matmul_internal, DiffusionPermutation};
use p3_symmetric::Permutation;

use crate::{
    BabyBear, DiffusionMatrixBabybear, PackedBabyBearAVX2, MATRIX_DIAG_16_BABYBEAR_MONTY,
    MATRIX_DIAG_24_BABYBEAR_MONTY,
};

impl Permutation<[PackedBabyBearAVX2; 16]> for DiffusionMatrixBabybear {
    fn permute_mut(&self, state: &mut [PackedBabyBearAVX2; 16]) {
        matmul_internal::<BabyBear, PackedBabyBearAVX2, 16>(state, MATRIX_DIAG_16_BABYBEAR_MONTY);
    }
}

impl DiffusionPermutation<PackedBabyBearAVX2, 16> for DiffusionMatrixBabybear {}

impl Permutation<[PackedBabyBearAVX2; 24]> for DiffusionMatrixBabybear {
    fn permute_mut(&self, state: &mut [PackedBabyBearAVX2; 24]) {
        matmul_internal::<BabyBear, PackedBabyBearAVX2, 24>(state, MATRIX_DIAG_24_BABYBEAR_MONTY);
    }
}

impl DiffusionPermutation<PackedBabyBearAVX2, 24> for DiffusionMatrixBabybear {}
