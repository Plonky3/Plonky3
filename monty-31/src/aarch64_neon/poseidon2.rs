use p3_poseidon2::{matmul_internal, DiffusionPermutation};
use p3_symmetric::Permutation;

use crate::{
    DiffusionMatrixMontyField31, DiffusionMatrixParameters, FieldParameters, MontyField31,
    PackedFieldPoseidon2Helpers, PackedMontyField31Neon,
};

// We need to change from the standard implementation as we are interpreting the matrix (1 + D(v)) as the monty form of the matrix not the raw form.
// matmul_internal internal performs a standard matrix multiplication so we need to additional rescale by the inverse monty constant.
// These will be removed once we have architecture specific implementations.

impl<FP, const WIDTH: usize, MD> Permutation<[PackedMontyField31Neon<FP>; WIDTH]>
    for DiffusionMatrixMontyField31<FP, MD>
where
    FP: FieldParameters,
    MD: DiffusionMatrixParameters<FP, WIDTH> + PackedFieldPoseidon2Helpers<FP>,
{
    fn permute_mut(&self, state: &mut [PackedMontyField31Neon<FP>; WIDTH]) {
        matmul_internal::<MontyField31<FP>, PackedMontyField31Neon<FP>, WIDTH>(
            state,
            MD::INTERNAL_DIAG_MONTY,
        );
        state.iter_mut().for_each(|i| *i *= MD::MONTY_INVERSE);
    }
}

impl<FP, const WIDTH: usize, MD> DiffusionPermutation<PackedMontyField31Neon<FP>, WIDTH>
    for DiffusionMatrixMontyField31<FP, MD>
where
    FP: FieldParameters,
    MD: DiffusionMatrixParameters<FP, WIDTH> + PackedFieldPoseidon2Helpers<FP>,
{
}
