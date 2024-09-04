use core::{
    arch::x86_64::{self, __m256i},
    mem::transmute,
};
use p3_poseidon2::{
    mds_light_permutation, ExternalLayer, InternalLayer, MDSMat4, Poseidon2ExternalPackedConstants,
    Poseidon2InternalPackedConstants,
};

use alloc::vec::Vec;

use crate::{
    apply_func_to_even_odd, movehdup_epi32, packed_exp_3, packed_exp_5, packed_exp_7,
    FieldParameters, MontyField31, MontyParameters, PackedMontyField31AVX2, PackedMontyParameters,
    Poseidon2ExternalLayerMonty31, Poseidon2InternalLayerMonty31,
};

#[repr(C)]
pub struct InternalLayer16<PMP: PackedMontyParameters> {
    s0: PackedMontyField31AVX2<PMP>,
    s_hi: [__m256i; 15],
}

#[repr(C)]
pub struct InternalLayer24<PMP: PackedMontyParameters> {
    s0: PackedMontyField31AVX2<PMP>,
    s_hi: [__m256i; 23],
}

#[inline(always)]
#[must_use]
fn exp_small<PMP: PackedMontyParameters, const D: u64>(val: __m256i) -> __m256i {
    match D {
        3 => packed_exp_3::<PMP>(val),
        5 => packed_exp_5::<PMP>(val),
        7 => packed_exp_7::<PMP>(val),
        _ => panic!("No exp function for given D"),
    }
}

#[inline(always)]
#[must_use]
/// Compute val -> (val + rc)^D. Each entry of val should be represented by an
/// element in [0, P]. Each entry of rc should be represented by an element
/// in [-P, 0]. Each entry of the output will be represented by an element in [0, P].
/// If the inputs do not conform to this representation, the result is undefined.
fn add_rc_and_sbox_external<PMP: PackedMontyParameters, const D: u64>(
    val: PackedMontyField31AVX2<PMP>,
    rc: __m256i,
) -> PackedMontyField31AVX2<PMP> {
    unsafe {
        // As our exponential functions simply assume that
        // the input lies in [-P, P] we do not need to perform a reduction provided
        // rc is represented by an element in [-P, 0]
        let vec_val = val.to_vector();
        let val_plus_rc = x86_64::_mm256_add_epi32(vec_val, rc);
        let output = apply_func_to_even_odd::<PMP>(val_plus_rc, exp_small::<PMP, D>);

        PackedMontyField31AVX2::<PMP>::from_vector(output)
    }
}

#[inline(always)]
#[must_use]
/// A variant of x -> (x + rc)^D where the input still needs a monty reduction.
/// Both x and rc must be positive with x + rc < 2^32P.
fn add_rc_and_sbox_internal<PMP: PackedMontyParameters, const D: u64>(
    val: PackedMontyField31AVX2<PMP>,
    rc: __m256i,
) -> PackedMontyField31AVX2<PMP> {
    unsafe {
        let val_vec = val.to_vector();
        // Add in rc. Note that rc should be saved with an extra factor of the monty constant.
        let val_plus_rc = x86_64::_mm256_add_epi32(val_vec, rc);

        // Copy the indices into the even indices.
        let input_odd = movehdup_epi32(val_plus_rc);

        // Perform the S-Box. The output lies in (-P, P) stored in the odd indices
        let output_even = exp_small::<PMP, D>(val_plus_rc);
        let output_odd = exp_small::<PMP, D>(input_odd);

        let d_evn_hi = movehdup_epi32(output_even);
        let t = x86_64::_mm256_blend_epi32::<0b10101010>(d_evn_hi, output_odd);

        let u = x86_64::_mm256_add_epi32(t, PMP::PACKED_P);
        let output = x86_64::_mm256_min_epu32(t, u);

        PackedMontyField31AVX2::<PMP>::from_vector(output)
    }
}

/// The compiler doesn't realize that add is associative
/// so we help it out and minimize the dependency chains by hand.
/// Note that state[0] is involved in a large s-box immediately before this
/// so we keep it separate for as long as possible.
#[must_use]
#[inline(always)]
fn sum_16<PMP: PackedMontyParameters>(
    state: &[PackedMontyField31AVX2<PMP>; 15],
) -> PackedMontyField31AVX2<PMP> {
    let sum01 = state[0] + state[1];
    let sum23 = state[2] + state[3];
    let sum45 = state[4] + state[5];
    let sum67 = state[6] + state[7];
    let sum89 = state[8] + state[9];
    let sum1011 = state[10] + state[11];
    let sum1213 = state[12] + state[13];

    let sum0123 = sum01 + sum23;
    let sum4567 = sum45 + sum67;
    let sum891011 = sum89 + sum1011;
    let sum121314 = sum1213 + state[14];

    let sum01234567 = sum0123 + sum4567;
    let sum_top_half = sum891011 + sum121314;

    sum01234567 + sum_top_half
}

#[must_use]
#[inline(always)]
fn sum_24<PMP: PackedMontyParameters>(
    state: &[PackedMontyField31AVX2<PMP>; 23],
) -> PackedMontyField31AVX2<PMP> {
    let sum01 = state[0] + state[1];
    let sum23 = state[2] + state[3];
    let sum45 = state[4] + state[5];
    let sum67 = state[6] + state[7];
    let sum89 = state[8] + state[9];
    let sum1011 = state[10] + state[11];
    let sum1213 = state[12] + state[13];
    let sum1415 = state[14] + state[15];
    let sum1617 = state[16] + state[17];
    let sum1819 = state[18] + state[19];
    let sum2021 = state[20] + state[21];

    let sum0123 = sum01 + sum23;
    let sum4567 = sum45 + sum67;
    let sum891011 = sum89 + sum1011;
    let sum12131415 = sum1213 + sum1415;
    let sum16171819 = sum1617 + sum1819;
    let sum202122 = sum2021 + state[22];

    let sum_bot_third = sum0123 + sum4567;
    let sum_mid_third = sum891011 + sum12131415;
    let sum_top_third = sum16171819 + sum202122;

    sum_bot_third + sum_mid_third + sum_top_third
}

/// A trait containing the specific information needed to
/// implement the Poseidon2 Permutation for Monty31 Fields.
pub trait InternalLayerParametersAVX2<const WIDTH: usize>: Clone + Sync {
    type ArrayLike;

    fn diagonal_mul(input: &mut Self::ArrayLike);

    fn add_sum(input: &mut Self::ArrayLike, sum: __m256i);
}

#[inline(always)]
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
        let rc = (internal_constant.value as u64) << 32;
        let rc_mod_p = rc % (MP::PRIME as u64);
        unsafe { x86_64::_mm256_set1_epi64x(rc_mod_p as i64) }
    }

    fn convert_from_field_list(
        internal_constants: &[MontyField31<MP>],
    ) -> Vec<Self::ConstantsType> {
        internal_constants
            .iter()
            .map(|constant| convert_to_vec_neg_form::<MP>(constant.value as i32))
            .collect()
    }
}

/// We save the round constants in the {-P, ..., 0} representation instead of the standard
/// {0, ..., P} one. This saves several instructions later.
impl<MP, const WIDTH: usize> Poseidon2ExternalPackedConstants<MontyField31<MP>, WIDTH>
    for Poseidon2ExternalLayerMonty31<WIDTH>
where
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

    fn convert_from_field_array_list(
        external_constants_list: [&[[MontyField31<MP>; WIDTH]]; 2],
    ) -> [Vec<Self::ConstantsType>; 2] {
        let constants_0 = external_constants_list[0]
            .iter()
            .map(Self::convert_from_field_array)
            .collect();
        let constants_1 = external_constants_list[1]
            .iter()
            .map(Self::convert_from_field_array)
            .collect();

        [constants_0, constants_1]
    }
}

// First constant of internal round should be the standard thing.
// remaining constants should be saved as R^2 rc instead of rc.
// All should be positive.
impl<FP, ILP, const D: u64> InternalLayer<PackedMontyField31AVX2<FP>, 16, D>
    for Poseidon2InternalLayerMonty31<ILP, 16>
where
    FP: FieldParameters,
    ILP: InternalLayerParametersAVX2<16, ArrayLike = [__m256i; 15]>,
{
    type InternalState = InternalLayer16<FP>;

    /// Need to keep things positive as we don't have signed shifts in AVX2.
    /// Might be able to do things slightly cheaper in AVX512.
    /// Need initial round to be different too.
    fn permute_state(
        &self,
        state: &mut Self::InternalState,
        _internal_constants: &[<PackedMontyField31AVX2<FP> as p3_field::AbstractField>::F],
        packed_internal_constants: &[Self::ConstantsType],
    ) {
        unsafe {
            packed_internal_constants.iter().for_each(|&rc| {
                state.s0 = add_rc_and_sbox_internal::<FP, D>(state.s0, rc);
                let sum_non_0 = sum_16(&transmute(state.s_hi));
                ILP::diagonal_mul(&mut state.s_hi);
                let sum = sum_non_0 + state.s0;
                state.s0 = sum_non_0 - state.s0;
                ILP::add_sum(&mut state.s_hi, transmute(sum));
            })
        }
    }
}

// First constant of internal round should be the standard thing.
// remaining constants should be saved as R^2 rc instead of rc.
// All should be positive.
impl<FP, ILP, const D: u64> InternalLayer<PackedMontyField31AVX2<FP>, 24, D>
    for Poseidon2InternalLayerMonty31<ILP, 24>
where
    FP: FieldParameters,
    ILP: InternalLayerParametersAVX2<24, ArrayLike = [__m256i; 23]>,
{
    type InternalState = InternalLayer24<FP>;

    /// Need to keep things positive as we don't have signed shifts in AVX2.
    /// Might be able to do things slightly cheaper in AVX512.
    /// Need initial round to be different too.
    fn permute_state(
        &self,
        state: &mut Self::InternalState,
        _internal_constants: &[<PackedMontyField31AVX2<FP> as p3_field::AbstractField>::F],
        packed_internal_constants: &[Self::ConstantsType],
    ) {
        unsafe {
            packed_internal_constants.iter().for_each(|&rc| {
                state.s0 = add_rc_and_sbox_internal::<FP, D>(state.s0, rc);
                let sum_non_0 = sum_24(&transmute(state.s_hi));
                ILP::diagonal_mul(&mut state.s_hi);
                let sum = sum_non_0 + state.s0;
                state.s0 = sum_non_0 - state.s0;
                ILP::add_sum(&mut state.s_hi, transmute(sum));
            })
        }
    }
}

/// Compute a collection of Poseidon2 external layers.
/// One layer for every constant supplied.
#[inline]
fn external_rounds<FP, const WIDTH: usize, const D: u64>(
    state: &mut [PackedMontyField31AVX2<FP>; WIDTH],
    packed_external_constants: &[[__m256i; WIDTH]],
) where
    FP: FieldParameters,
{
    packed_external_constants.iter().for_each(|round_consts| {
        state
            .iter_mut()
            .zip(round_consts.iter())
            .for_each(|(val, &rc)| {
                *val = add_rc_and_sbox_external::<FP, D>(*val, rc);
            });
        mds_light_permutation(state, &MDSMat4);
    });
}

impl<FP, const D: u64> ExternalLayer<PackedMontyField31AVX2<FP>, 16, D>
    for Poseidon2ExternalLayerMonty31<16>
where
    FP: FieldParameters,
{
    type InternalState = InternalLayer16<FP>;

    /// Compute the first half of the Poseidon2 external layers.
    fn permute_state_initial(
        &self,
        mut state: [PackedMontyField31AVX2<FP>; 16],
        _initial_external_constants: &[[MontyField31<FP>; 16]],
        packed_initial_external_constants: &[[__m256i; 16]],
    ) -> Self::InternalState {
        mds_light_permutation(&mut state, &MDSMat4);

        external_rounds::<FP, 16, D>(&mut state, packed_initial_external_constants);

        unsafe { transmute(state) }
    }

    /// Compute the second half of the Poseidon2 external layers.
    fn permute_state_final(
        &self,
        state: Self::InternalState,
        _final_external_constants: &[[MontyField31<FP>; 16]],
        packed_final_external_constants: &[[__m256i; 16]],
    ) -> [PackedMontyField31AVX2<FP>; 16] {
        // Need to do something slightly special for the first round.
        let mut output_state = unsafe { transmute(state) };
        external_rounds::<FP, 16, D>(&mut output_state, packed_final_external_constants);
        output_state
    }
}

impl<FP, const D: u64> ExternalLayer<PackedMontyField31AVX2<FP>, 24, D>
    for Poseidon2ExternalLayerMonty31<24>
where
    FP: FieldParameters,
{
    type InternalState = InternalLayer24<FP>;

    /// Compute the first half of the Poseidon2 external layers.
    fn permute_state_initial(
        &self,
        mut state: [PackedMontyField31AVX2<FP>; 24],
        _initial_external_constants: &[[MontyField31<FP>; 24]],
        packed_initial_external_constants: &[[__m256i; 24]],
    ) -> Self::InternalState {
        mds_light_permutation(&mut state, &MDSMat4);

        external_rounds::<FP, 24, D>(&mut state, packed_initial_external_constants);

        unsafe { transmute(state) }
    }

    /// Compute the second half of the Poseidon2 external layers.
    fn permute_state_final(
        &self,
        state: Self::InternalState,
        _final_external_constants: &[[MontyField31<FP>; 24]],
        packed_final_external_constants: &[[__m256i; 24]],
    ) -> [PackedMontyField31AVX2<FP>; 24] {
        // Need to do something slightly special for the first round.
        let mut output_state = unsafe { transmute(state) };
        external_rounds::<FP, 24, D>(&mut output_state, packed_final_external_constants);
        output_state
    }
}
