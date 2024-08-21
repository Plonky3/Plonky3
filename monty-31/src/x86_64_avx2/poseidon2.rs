use core::{
    arch::x86_64::{self, __m256i},
    mem::transmute,
};
use p3_poseidon2::{
    mds_light_permutation, ExternalLayer, InternalLayer, MDSMat4, Poseidon2ExternalPackedConstants,
    Poseidon2InternalPackedConstants,
};

use crate::{
    ExternalLayerParameters, FieldParameters, InternalLayerParameters, MontyField31,
    MontyParameters, PackedMontyField31AVX2, PackedMontyParameters, Poseidon2ExternalLayerMonty31,
    Poseidon2InternalLayerMonty31,
};

/// A trait containing the specific information needed to
/// implement the Poseidon2 Permutation for Monty31 Fields.
pub trait ExternalLayerParametersAVX2: Clone + Sync {
    fn add_rc_and_sbox(input: __m256i, rc: __m256i) -> __m256i;

    // Possibly an extra couple of things? s_box internal/external might be different.
}

/// A trait containing the specific information needed to
/// implement the Poseidon2 Permutation for Monty31 Fields.
pub trait InternalLayerParametersAVX2<const WIDTH: usize>: Clone + Sync {
    fn add_rc_and_sbox(input: __m256i, rc: __m256i) -> __m256i;

    // fn reduce(input: __m256i) -> __m256i;

    fn diagonal_mul(input: [__m256i; WIDTH]) -> [__m256i; WIDTH];
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
impl<P2P, MP, const WIDTH: usize> Poseidon2InternalPackedConstants<MontyField31<MP>>
    for Poseidon2InternalLayerMonty31<P2P, WIDTH>
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
impl<MP, ELP, const WIDTH: usize> Poseidon2ExternalPackedConstants<MontyField31<MP>, WIDTH>
    for Poseidon2ExternalLayerMonty31<ELP, WIDTH>
where
    ELP: Sync + Clone,
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

#[inline(always)]
fn sum_16<PMP: PackedMontyParameters>(
    state: &[PackedMontyField31AVX2<PMP>; 16],
) -> PackedMontyField31AVX2<PMP> {
    let sum23 = state[2] + state[3];
    let sum45 = state[4] + state[5];
    let sum67 = state[6] + state[7];
    let sum89 = state[8] + state[9];
    let sum1011 = state[10] + state[11];
    let sum1213 = state[12] + state[13];
    let sum1415 = state[14] + state[15];

    let sum123 = state[1] + sum23;
    let sum4567 = sum45 + sum67;
    let sum891011 = sum89 + sum1011;
    let sum12131415 = sum1213 + sum1415;

    let sum1234567 = sum123 + sum4567;
    let sum_top_half = sum891011 + sum12131415;

    let sum_all_but_0 = sum1234567 + sum_top_half;

    sum_all_but_0 + state[0]
}

impl<FP, ILP, const D: u64> InternalLayer<PackedMontyField31AVX2<FP>, 16, D>
    for Poseidon2InternalLayerMonty31<ILP, 16>
where
    FP: FieldParameters,
    ILP: InternalLayerParametersAVX2<16>,
{
    type InternalState = [PackedMontyField31AVX2<FP>; 16];

    fn permute_state(
        &self,
        state: &mut Self::InternalState,
        _internal_constants: &[<PackedMontyField31AVX2<FP> as p3_field::AbstractField>::F],
        packed_internal_constants: &[Self::ConstantsType],
    ) {
        unsafe {
            packed_internal_constants.iter().for_each(|&rc| {
                state[0] = transmute(ILP::add_rc_and_sbox(transmute(state[0]), rc));
                let sum = sum_16::<FP>(state);
                *state = transmute(ILP::diagonal_mul(transmute(*state)));
                state.iter_mut().for_each(|elem| *elem += sum)
            })
        }
    }
}

/// Compute a collection of Poseidon2 external layers.
/// One layer for every constant supplied.
#[inline]
fn external_rounds<ELP, FP, const WIDTH: usize>(
    state: &mut [PackedMontyField31AVX2<FP>; WIDTH],
    packed_external_constants: &[[__m256i; WIDTH]],
) where
    FP: FieldParameters,
    ELP: ExternalLayerParametersAVX2,
{
    packed_external_constants.iter().for_each(|round_consts| {
        state
            .iter_mut()
            .zip(round_consts.iter())
            .for_each(|(val, &rc)| unsafe {
                let vec_val = transmute(*val);
                let vec_val_post_sbox = ELP::add_rc_and_sbox(vec_val, rc);
                *val = transmute(vec_val_post_sbox);
            });
        mds_light_permutation(state, &MDSMat4);
    });
}

impl<FP, ELP, const WIDTH: usize, const D: u64> ExternalLayer<PackedMontyField31AVX2<FP>, WIDTH, D>
    for Poseidon2ExternalLayerMonty31<ELP, WIDTH>
where
    FP: FieldParameters,
    ELP: ExternalLayerParametersAVX2,
{
    type InternalState = [PackedMontyField31AVX2<FP>; WIDTH];

    /// Compute the first half of the Poseidon2 external layers.
    fn permute_state_initial(
        &self,
        mut state: [PackedMontyField31AVX2<FP>; WIDTH],
        _initial_external_constants: &[[MontyField31<FP>; WIDTH]],
        packed_initial_external_constants: &[[__m256i; WIDTH]],
    ) -> [PackedMontyField31AVX2<FP>; WIDTH] {
        mds_light_permutation(&mut state, &MDSMat4);
        external_rounds::<ELP, FP, WIDTH>(&mut state, packed_initial_external_constants);
        state
    }

    /// Compute the second half of the Poseidon2 external layers.
    fn permute_state_final(
        &self,
        mut state: [PackedMontyField31AVX2<FP>; WIDTH],
        _final_external_constants: &[[MontyField31<FP>; WIDTH]],
        packed_final_external_constants: &[[__m256i; WIDTH]],
    ) -> [PackedMontyField31AVX2<FP>; WIDTH] {
        external_rounds::<ELP, FP, WIDTH>(&mut state, packed_final_external_constants);
        state
    }
}
