use core::{
    arch::x86_64::{self, __m256i},
    mem::transmute,
};
use p3_poseidon2::{
    mds_light_permutation, ExternalLayer, InternalLayer, MDSMat4, Poseidon2ExternalPackedConstants,
    Poseidon2InternalPackedConstants,
};

use crate::{
    apply_func_to_even_odd, movehdup_epi32, packed_exp_3, packed_exp_5, packed_exp_7,
    ExternalLayerParameters, FieldParameters, InternalLayerParameters, MontyField31,
    MontyParameters, PackedMontyField31AVX2, PackedMontyParameters, Poseidon2ExternalLayerMonty31,
    Poseidon2InternalLayerMonty31,
};

const ZEROS: __m256i = unsafe { transmute([0_i64; 4]) };

fn add_rc_and_sbox<PMP: PackedMontyParameters, const D: u64>(
    val: PackedMontyField31AVX2<PMP>,
    rc: __m256i,
) -> PackedMontyField31AVX2<PMP> {
    unsafe {
        let vec_val = transmute(val);
        let val_plus_rc = x86_64::_mm256_add_epi32(vec_val, rc);
        let func = match D {
            3 => packed_exp_3::<PMP>,
            5 => packed_exp_5::<PMP>,
            7 => packed_exp_7::<PMP>,
            _ => panic!("No exp function for given D"),
        };
        let output = apply_func_to_even_odd::<PMP>(val_plus_rc, func);
        transmute(output)
    }
}

fn apply_external_linear_layer<const WIDTH: usize>(_state: &[__m256i; WIDTH]) -> [__m256i; WIDTH] {
    todo!()
}

/// A trait containing the specific information needed to
/// implement the Poseidon2 Permutation for Monty31 Fields.
pub trait ExternalLayerParametersAVX2<const D: u64>: Clone + Sync {
    // fn add_rc_and_sbox(input: __m256i, rc: __m256i) -> __m256i;

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

impl<FP, ILP, const D: u64> InternalLayer<PackedMontyField31AVX2<FP>, 16, D>
    for Poseidon2InternalLayerMonty31<ILP, 16>
where
    FP: FieldParameters,
    ILP: InternalLayerParametersAVX2<16>,
{
    type InternalState = [[__m256i; 16]; 2];

    fn permute_state(
        &self,
        state: &mut Self::InternalState,
        _internal_constants: &[<PackedMontyField31AVX2<FP> as p3_field::AbstractField>::F],
        packed_internal_constants: &[Self::ConstantsType],
    ) {
        state.iter_mut().for_each(|sub_state| unsafe {
            packed_internal_constants.iter().for_each(|&rc| {
                sub_state[0] = transmute(ILP::add_rc_and_sbox(transmute(sub_state[0]), rc));

                let sum = sub_state
                    .iter()
                    .fold(ZEROS, |acc, &val| x86_64::_mm256_add_epi64(acc, val));

                *sub_state = transmute(ILP::diagonal_mul(transmute(*sub_state)));
                sub_state
                    .iter_mut()
                    .for_each(|elem| *elem = x86_64::_mm256_add_epi64(*elem, sum))
            })
        })
    }
}

/// Compute a collection of Poseidon2 external layers.
/// One layer for every constant supplied.
#[inline]
fn external_rounds<ELP, FP, const WIDTH: usize, const D: u64>(
    state: &mut [PackedMontyField31AVX2<FP>; WIDTH],
    packed_external_constants: &[[__m256i; WIDTH]],
) where
    FP: FieldParameters,
    ELP: ExternalLayerParametersAVX2<D>,
{
    packed_external_constants.iter().for_each(|round_consts| {
        state
            .iter_mut()
            .zip(round_consts.iter())
            .for_each(|(val, &rc)| {
                *val = add_rc_and_sbox::<FP, D>(*val, rc);
            });
        mds_light_permutation(state, &MDSMat4);
    });
}

#[inline]
fn final_initial_external_round<ELP, FP, const WIDTH: usize, const D: u64>(
    state: [PackedMontyField31AVX2<FP>; WIDTH],
    round_consts: &[__m256i; WIDTH],
) -> [[__m256i; WIDTH]; 2]
where
    FP: FieldParameters,
    ELP: ExternalLayerParametersAVX2<D>,
{
    unsafe {
        let mut state_even: [__m256i; WIDTH] = state.map(|x| transmute(x));
        state_even
            .iter_mut()
            .zip(round_consts.iter())
            .for_each(|(val, &rc)| *val = x86_64::_mm256_add_epi32(*val, rc));

        let state_odd = state_even.map(movehdup_epi32);

        let post_sbox = match D {
            3 => [
                state_even.map(packed_exp_3::<FP>),
                state_odd.map(packed_exp_3::<FP>),
            ],
            5 => [
                state_even.map(packed_exp_5::<FP>),
                state_odd.map(packed_exp_5::<FP>),
            ],
            7 => [
                state_even.map(packed_exp_7::<FP>),
                state_odd.map(packed_exp_7::<FP>),
            ],
            _ => panic!("No exp function for given D"),
        };

        [
            apply_external_linear_layer(&post_sbox[0]),
            apply_external_linear_layer(&post_sbox[1]),
        ]
    }
}

impl<FP, ELP, const WIDTH: usize, const D: u64> ExternalLayer<PackedMontyField31AVX2<FP>, WIDTH, D>
    for Poseidon2ExternalLayerMonty31<ELP, WIDTH>
where
    FP: FieldParameters,
    ELP: ExternalLayerParametersAVX2<D>,
{
    type InternalState = [[__m256i; WIDTH]; 2];

    /// Compute the first half of the Poseidon2 external layers.
    fn permute_state_initial(
        &self,
        mut state: [PackedMontyField31AVX2<FP>; WIDTH],
        _initial_external_constants: &[[MontyField31<FP>; WIDTH]],
        packed_initial_external_constants: &[[__m256i; WIDTH]],
    ) -> [[__m256i; WIDTH]; 2] {
        mds_light_permutation(&mut state, &MDSMat4);
        // We need to do something special for the last round as we want our output in a different form.
        let num_constants = packed_initial_external_constants.len();
        external_rounds::<ELP, FP, WIDTH, D>(
            &mut state,
            &(packed_initial_external_constants[..(num_constants - 1)]),
        );
        final_initial_external_round::<ELP, FP, WIDTH, D>(
            state,
            &packed_initial_external_constants[num_constants - 1],
        )
    }

    /// Compute the second half of the Poseidon2 external layers.
    fn permute_state_final(
        &self,
        mut state: [[__m256i; WIDTH]; 2],
        _final_external_constants: &[[MontyField31<FP>; WIDTH]],
        packed_final_external_constants: &[[__m256i; WIDTH]],
    ) -> [PackedMontyField31AVX2<FP>; WIDTH] {
        external_rounds::<ELP, FP, WIDTH, D>(&mut state, packed_final_external_constants);
        state
    }
}
