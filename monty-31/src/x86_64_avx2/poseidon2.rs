use p3_poseidon2::{matmul_internal, DiffusionPermutation};
use p3_symmetric::Permutation;

use crate::{
    DiffusionMatrixMontyField31, FieldParameters, MontyField31, PackedMontyField31AVX2,
    Poseidon2Monty31,
};

// We need to change from the standard implementation as we are interpreting the matrix (1 + D(v)) as the monty form of the matrix not the raw form.
// matmul_internal internal performs a standard matrix multiplication so we need to additional rescale by the inverse monty constant.
// These will be removed once we have architecture specific implementations.

impl<FP: FieldParameters, PU: Poseidon2Monty31<FP>> Permutation<[PackedMontyField31AVX2<FP>; 16]>
    for DiffusionMatrixMontyField31<FP, PU>
{
    fn permute_mut(&self, state: &mut [PackedMontyField31AVX2<FP>; 16]) {
        matmul_internal::<MontyField31<FP>, PackedMontyField31AVX2<FP>, 16>(
            state,
            PU::INTERNAL_DIAG_MONTY,
        );
        state.iter_mut().for_each(|i| *i *= PU::MONTY_INVERSE);
    }
}

impl<FP: FieldParameters, PU: Poseidon2Monty31<FP>>
    DiffusionPermutation<PackedMontyField31AVX2<FP>, 16> for DiffusionMatrixMontyField31<FP, PU>
{
}

impl<FP: FieldParameters, PU: Poseidon2Monty31<FP>> Permutation<[PackedMontyField31AVX2<FP>; 24]>
    for DiffusionMatrixMontyField31<FP, PU>
{
    fn permute_mut(&self, state: &mut [PackedMontyField31AVX2<FP>; 24]) {
        matmul_internal::<MontyField31<FP>, PackedMontyField31AVX2<FP>, 24>(
            state,
            PU::INTERNAL_DIAG_MONTY,
        );
        state.iter_mut().for_each(|i| *i *= PU::MONTY_INVERSE);
    }
}

impl<FP: FieldParameters, PU: Poseidon2Monty31<FP>>
    DiffusionPermutation<PackedMontyField31AVX2<FP>, 24> for DiffusionMatrixMontyField31<FP, PU>
{
}
