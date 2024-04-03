use p3_poseidon2::{matmul_internal, DiffusionPermutation};
use p3_symmetric::Permutation;

use crate::{
    BabyBear, DiffusionMatrixBabybear, PackedBabyBearNeon, MATRIX_DIAG_16_BABYBEAR_MONTY,
    MATRIX_DIAG_24_BABYBEAR_MONTY,
};

impl Permutation<[PackedBabyBearNeon; 16]> for DiffusionMatrixBabybear {
    fn permute_mut(&self, state: &mut [PackedBabyBearNeon; 16]) {
        matmul_internal::<BabyBear, PackedBabyBearNeon, 16>(state, MATRIX_DIAG_16_BABYBEAR_MONTY);
    }
}

impl DiffusionPermutation<PackedBabyBearNeon, 16> for DiffusionMatrixBabybear {}

impl Permutation<[PackedBabyBearNeon; 24]> for DiffusionMatrixBabybear {
    fn permute_mut(&self, state: &mut [PackedBabyBearNeon; 24]) {
        matmul_internal::<BabyBear, PackedBabyBearNeon, 24>(state, MATRIX_DIAG_24_BABYBEAR_MONTY);
    }
}

impl DiffusionPermutation<PackedBabyBearNeon, 24> for DiffusionMatrixBabybear {}
