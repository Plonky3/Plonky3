use p3_poseidon2::{matmul_internal, DiffusionPermutation};
use p3_symmetric::Permutation;

use crate::{
    DiffusionMatrixMontyField31, FieldParameters, MontyField31, MultipleDiffusionMatrixParameters,
    PackedMontyField31AVX2,
};

// We need to change from the standard implementation as we are interpreting the matrix (1 + D(v)) as the monty form of the matrix not the raw form.
// matmul_internal internal performs a standard matrix multiplication so we need to additional rescale by the inverse monty constant.
// These will be removed once we have architecture specific implementations.

impl<FP: FieldParameters, MD: MultipleDiffusionMatrixParameters<FP>>
    Permutation<[PackedMontyField31AVX2<FP>; 16]> for DiffusionMatrixMontyField31<FP, MD>
{
    fn permute_mut(&self, state: &mut [PackedMontyField31AVX2<FP>; 16]) {
        matmul_internal::<MontyField31<FP>, PackedMontyField31AVX2<FP>, 16>(
            state,
            MD::INTERNAL_DIAG_MONTY,
        );
        state.iter_mut().for_each(|i| *i *= MD::MONTY_INVERSE);
    }
}

impl<FP: FieldParameters, MD: MultipleDiffusionMatrixParameters<FP>>
    DiffusionPermutation<PackedMontyField31AVX2<FP>, 16> for DiffusionMatrixMontyField31<FP, MD>
{
}

impl<FP: FieldParameters, MD: MultipleDiffusionMatrixParameters<FP>>
    Permutation<[PackedMontyField31AVX2<FP>; 24]> for DiffusionMatrixMontyField31<FP, MD>
{
    fn permute_mut(&self, state: &mut [PackedMontyField31AVX2<FP>; 24]) {
        matmul_internal::<MontyField31<FP>, PackedMontyField31AVX2<FP>, 24>(
            state,
            MD::INTERNAL_DIAG_MONTY,
        );
        state.iter_mut().for_each(|i| *i *= MD::MONTY_INVERSE);
    }
}

impl<FP: FieldParameters, MD: MultipleDiffusionMatrixParameters<FP>>
    DiffusionPermutation<PackedMontyField31AVX2<FP>, 24> for DiffusionMatrixMontyField31<FP, MD>
{
}
