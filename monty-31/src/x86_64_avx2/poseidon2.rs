use core::arch::x86_64::{self, __m256i};
use p3_poseidon2::{
    matmul_internal, Poseidon2ExternalPackedConstants, Poseidon2InternalPackedConstants,
};
use p3_symmetric::Permutation;

use crate::{
    FieldParameters, MontyField31, MontyParameters, PackedFieldPoseidon2Helpers,
    PackedMontyField31AVX2, Poseidon2ExternalLayerMonty31, Poseidon2InternalLayerMonty31,
    Poseidon2Parameters,
};

/// A trait containing the specific information needed to
/// implement the Poseidon2 Permutation for Monty31 Fields.
pub trait PackedPoseidon2Parameters<FP: FieldParameters, const WIDTH: usize>: Clone + Sync {
    /// Implements multiplication by the diffusion matrix 1 + Diag(vec) using a delayed reduction strategy.
    fn internal_shifts(state: &mut [MontyField31<FP>; WIDTH]);

    fn s_box_plus_rc(input: MontyField31<FP>) -> MontyField31<FP>;

    // Possibly an extra couple of things? s_box internal/external might be different.
}

/// Convert elements from the standard form {0, ..., P} to {-P, ..., 0} and copy into a vector
fn convert_to_vec_neg_form<MP: MontyParameters>(input: i32) -> __m256i {
    let input_sub_p = input - (MP::PRIME as i32);
    unsafe {
        // Safety: If this code got compiled then AVX2 intrinsics are available.
        x86_64::_mm256_set1_epi32(input_sub_p)
    }
}

/// We save the round constants in the {-P, ..., 0} representation instead of the standard
/// {0, ..., P} one. This saves several instructions later.
impl<P2P, MP> Poseidon2InternalPackedConstants<MontyField31<MP>>
    for Poseidon2InternalLayerMonty31<P2P>
where
    P2P: Clone + Sync,
    MP: MontyParameters,
{
    type ConstantsType = __m256i;

    fn convert_from_field(internal_constant: &MontyField31<MP>) -> Self::ConstantsType {
        convert_to_vec_neg_form::<MP>(internal_constant.value as i32)
    }
}

/// We save the round constants in the {-P, ..., 0} representation instead of the standard
/// {0, ..., P} one. This saves several instructions later.
impl<P2P, MP, const WIDTH: usize> Poseidon2ExternalPackedConstants<MontyField31<MP>, WIDTH>
    for Poseidon2ExternalLayerMonty31<P2P, WIDTH>
where
    P2P: Clone + Sync,
    MP: MontyParameters,
{
    type ConstantsType = [__m256i; WIDTH];

    /// Convert elements from the standard form {0, ..., P} to {-P, ..., 0}.
    fn convert_from_field_array(
        external_constants: &[MontyField31<MP>; WIDTH],
    ) -> [__m256i; WIDTH] {
        external_constants
            .map(|external_constant| convert_to_vec_neg_form::<MP>(external_constant.value as i32))
    }
}
