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
    apply_func_to_even_odd, monty_red_unsigned, monty_red_unsigned_pos, movehdup_epi32,
    packed_exp_3, packed_exp_5, packed_exp_7, FieldParameters, MontyField31, MontyParameters,
    PackedMontyField31AVX2, PackedMontyParameters, Poseidon2ExternalLayerMonty31,
    Poseidon2InternalLayerMonty31,
};

const ZEROS: __m256i = unsafe { transmute([0_i64; 4]) };

fn exp_small<PMP: PackedMontyParameters, const D: u64>(val: __m256i) -> __m256i {
    match D {
        3 => packed_exp_3::<PMP>(val),
        5 => packed_exp_5::<PMP>(val),
        7 => packed_exp_7::<PMP>(val),
        _ => panic!("No exp function for given D"),
    }
}

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

/// A variant of x -> (x + rc)^D where the input still needs a monty reduction.
/// Both x and rc must be positive with x + rc < 2^32P.
fn add_rc_and_sbox_internal<PMP: PackedMontyParameters, const D: u64>(
    val: __m256i,
    rc: __m256i,
) -> __m256i {
    unsafe {
        // Add in rc. Note that rc should be saved with an extra factor of the monty constant.
        let val_plus_rc = x86_64::_mm256_add_epi64(val, rc);

        // Perform the monty reduction producing something in (-P, P) in the odd indices
        let val_plus_rc_red = monty_red_unsigned::<PMP>(val_plus_rc);

        // Copy the indices into the even indices.
        let sbox_input = movehdup_epi32(val_plus_rc_red);

        // Perform the S-Box. The output lies in (-P, P) stored in the odd indices
        let output_shifted_signed = exp_small::<PMP, D>(sbox_input);

        // Shift the output down into the even indices.
        let output_signed = x86_64::_mm256_srli_epi64::<32>(output_shifted_signed);

        // Add P. This means each i64 contains something in (0, 2P)
        x86_64::_mm256_add_epi32(output_signed, PMP::PACKED_P_U64)
    }
}

/// A variant of x -> (x + rc)^D where the input still needs a monty reduction.
/// rc should be positive and x must be < 127P.
fn add_rc_and_sbox_first_internal<
    PMP: PackedMontyParameters,
    ILP: InternalLayerParametersAVX2<WIDTH>,
    const WIDTH: usize,
    const D: u64,
>(
    val: __m256i,
    rc: __m256i,
) -> __m256i {
    unsafe {
        // Add in rc.
        let val_plus_rc = x86_64::_mm256_add_epi64(val, rc);

        // Perform a small reduction returning something in (0, 2P).
        let val_plus_rc_red = ILP::reduce(val_plus_rc);

        // Reduce to something in (-P, P).
        let sbox_input = x86_64::_mm256_sub_epi32(val_plus_rc_red, PMP::PACKED_P_U64);

        // Perform the S-Box. The output lies in (-P, P) stored in the odd indices
        let output_shifted_signed = exp_small::<PMP, D>(sbox_input);

        // Shift the output down into the even indices.
        let output_signed = x86_64::_mm256_srli_epi64::<32>(output_shifted_signed);

        // Add P. This means each i64 contains something in (0, 2P)
        x86_64::_mm256_add_epi32(output_signed, PMP::PACKED_P_U64)
    }
}

/// Multiply a 4-element vector x by:
/// [ 2 3 1 1 ]
/// [ 1 2 3 1 ]
/// [ 1 1 2 3 ]
/// [ 3 1 1 2 ].
/// If inputs are <= L, outputs are <= 7L.
#[inline(always)]
fn apply_packed_mat4(x: &mut [__m256i; 4]) {
    unsafe {
        let t01 = x86_64::_mm256_add_epi64(x[0], x[1]);
        let t23 = x86_64::_mm256_add_epi64(x[2], x[3]);
        let t0123 = x86_64::_mm256_add_epi64(t01, t23);
        let t01123 = x86_64::_mm256_add_epi64(t0123, x[1]);
        let t01233 = x86_64::_mm256_add_epi64(t0123, x[3]);

        let t00 = x86_64::_mm256_add_epi64(x[0], x[0]);
        let t22 = x86_64::_mm256_add_epi64(x[2], x[2]);

        x[0] = x86_64::_mm256_add_epi64(t01123, t01); // 2*x[0] + 3*x[1] + x[2] + x[3]
        x[1] = x86_64::_mm256_add_epi64(t01123, t22); // x[0] + 2*x[1] + 3*x[2] + x[3]
        x[2] = x86_64::_mm256_add_epi64(t01233, t23); // x[0] + x[1] + 2*x[2] + 3*x[3]
        x[3] = x86_64::_mm256_add_epi64(t01233, t00); // 3*x[0] + x[1] + x[2] + 2*x[3]
    }
}

/// If inputs are <= L, outputs are <= 7 * (WIDTH/4 + 1)L.
/// In particular for with 16/24 these are <= 28/35L respectively.
fn apply_external_linear_layer<const WIDTH: usize>(state: &mut [__m256i; WIDTH]) {
    match WIDTH {
        16 | 24 => {
            // First, we apply M_4 to each consecutive four elements of the state.
            // In Appendix B's terminology, this replaces each x_i with x_i'.
            for i in (0..WIDTH).step_by(4) {
                let mut state_4 = [state[i], state[i + 1], state[i + 2], state[i + 3]];
                apply_packed_mat4(&mut state_4);
                state[i..i + 4].clone_from_slice(&state_4);
            }
            // Now, we apply the outer circulant matrix (to compute the y_i values).

            unsafe {
                // We first precompute the four sums of every four elements.
                let sums: [__m256i; 4] = core::array::from_fn(|k| {
                    (0..WIDTH)
                        .step_by(4)
                        .map(|j| state[j + k])
                        .fold(ZEROS, |acc, val| x86_64::_mm256_add_epi64(acc, val))
                });

                // The formula for each y_i involves 2x_i' term and x_j' terms for each j that equals i mod 4.
                // In other words, we can add a single copy of x_i' to the appropriate one of our precomputed sums
                for i in 0..WIDTH {
                    state[i] = x86_64::_mm256_add_epi64(state[i], sums[i % 4]);
                }
            }
        }

        _ => {
            panic!("Unsupported width");
        }
    }
}

/// A trait containing the specific information needed to
/// implement the Poseidon2 Permutation for Monty31 Fields.
pub trait InternalLayerParametersAVX2<const WIDTH: usize>: Clone + Sync {
    fn reduce(input: __m256i) -> __m256i;

    fn diagonal_mul_standard_input(input: &mut [__m256i; WIDTH]);

    fn diagonal_mul_shifted_input(input: &mut [__m256i; WIDTH]);
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
        let rc = (internal_constant.value as u64) << 32;
        let rc_mod_p = rc % (MP::PRIME as u64);
        unsafe { x86_64::_mm256_set1_epi64x(rc_mod_p as i64) }
    }

    fn convert_from_field_list(
        internal_constants: &[MontyField31<MP>],
    ) -> Vec<Self::ConstantsType> {
        internal_constants
            .iter()
            .enumerate()
            .map(|(i, constant)| {
                if i == 0 {
                    unsafe { x86_64::_mm256_set1_epi64x(constant.value as i64) }
                } else {
                    Self::convert_from_field(constant)
                }
            })
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
            .enumerate()
            .map(|(i, const_array)| {
                if i == 0 {
                    const_array.map(|val| {
                        let rc = (val.value as u64) << 32;
                        let rc_mod_p = rc % (MP::PRIME as u64);
                        unsafe { x86_64::_mm256_set1_epi64x(rc_mod_p as i64) }
                    })
                } else {
                    Self::convert_from_field_array(const_array)
                }
            })
            .collect();

        [constants_0, constants_1]
    }
}

fn first_internal_round<FP, ILP, const WIDTH: usize, const D: u64>(
    state: &mut [__m256i; WIDTH],
    rc: __m256i,
) where
    FP: FieldParameters,
    ILP: InternalLayerParametersAVX2<WIDTH>,
{
    // The first internal round is a little special as the inputs are smaller.
    // Hence we do not need to do any monty reductions except for the first element which involves the s-box.
    unsafe {
        state[0] = add_rc_and_sbox_first_internal::<FP, ILP, WIDTH, D>(state[0], rc);
        let sum = state
            .iter()
            .fold(ZEROS, |acc, &val| x86_64::_mm256_add_epi64(acc, val));
        ILP::diagonal_mul_standard_input(state);
        state
            .iter_mut()
            .for_each(|elem| *elem = x86_64::_mm256_add_epi64(*elem, sum));
    }
}

// First constant of internal round should be the standard thing.
// remaining constants should be saved as R^2 rc instead of rc.
// All should be positive.
impl<FP, ILP, const WIDTH: usize, const D: u64> InternalLayer<PackedMontyField31AVX2<FP>, WIDTH, D>
    for Poseidon2InternalLayerMonty31<ILP, WIDTH>
where
    FP: FieldParameters,
    ILP: InternalLayerParametersAVX2<WIDTH>,
{
    type InternalState = [[__m256i; WIDTH]; 2];

    /// Need to keep things positive as we don't have signed shifts in AVX2.
    /// Might be able to do things slightly cheaper in AVX512.
    /// Need initial round to be different too.
    fn permute_state(
        &self,
        state: &mut Self::InternalState,
        _internal_constants: &[<PackedMontyField31AVX2<FP> as p3_field::AbstractField>::F],
        packed_internal_constants: &[Self::ConstantsType],
    ) {
        state.iter_mut().for_each(|sub_state| unsafe {
            first_internal_round::<FP, ILP, WIDTH, D>(sub_state, packed_internal_constants[0]);
            packed_internal_constants.iter().skip(1).for_each(|&rc| {
                let sum_non_0 = sub_state[1..]
                    .iter()
                    .fold(ZEROS, |acc, &val| x86_64::_mm256_add_epi64(acc, val));

                sub_state[0] = add_rc_and_sbox_internal::<FP, D>(sub_state[0], rc);

                sub_state[1..]
                    .iter_mut()
                    .for_each(|elem| *elem = monty_red_unsigned_pos::<FP>(*elem));

                // Now we have to reduce the sum of the elements.
                let reduced_sum_non_0_shifted = monty_red_unsigned_pos::<FP>(sum_non_0);
                let reduced_sum_non_0 = x86_64::_mm256_srli_epi64::<32>(reduced_sum_non_0_shifted);

                let final_sum = x86_64::_mm256_add_epi64(reduced_sum_non_0, sub_state[0]);

                ILP::diagonal_mul_shifted_input(sub_state);
                sub_state
                    .iter_mut()
                    .for_each(|elem| *elem = x86_64::_mm256_add_epi64(*elem, final_sum))
            })
        })
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

/// The final external round of the initial set of external rounds needs to be done slightly differently.
/// As the internal rounds will use delayed reduction, we don't need to recombine the vectors after the
/// sbox. Instead we keep them separate.
#[inline]
fn final_initial_external_round<FP, const WIDTH: usize, const D: u64>(
    state: [PackedMontyField31AVX2<FP>; WIDTH],
    round_consts: &[__m256i; WIDTH],
) -> [[__m256i; WIDTH]; 2]
where
    FP: FieldParameters,
{
    unsafe {
        let mut state_even: [__m256i; WIDTH] = state.map(|x| transmute(x));
        state_even
            .iter_mut()
            .zip(round_consts.iter())
            .for_each(|(val, &rc)| *val = x86_64::_mm256_add_epi32(*val, rc));

        let state_odd = state_even.map(movehdup_epi32);

        let mut output_even = state_even.map(exp_small::<FP, D>);
        let mut output_odd = state_odd.map(exp_small::<FP, D>);

        // 2 Issues with state_even/state_odd. Currently signed and stored in the upper 32 bits.
        // Need to shift down and ensure the result is positive in {0..P}.
        output_even.iter_mut().for_each(|elem| {
            *elem = x86_64::_mm256_srli_epi64::<32>(*elem);
            let elem_plus_p = x86_64::_mm256_add_epi32(*elem, FP::PACKED_P_U64);
            *elem = x86_64::_mm256_min_epu32(*elem, elem_plus_p);
        });

        output_odd.iter_mut().for_each(|elem| {
            *elem = x86_64::_mm256_srli_epi64::<32>(*elem);
            let elem_plus_p = x86_64::_mm256_add_epi32(*elem, FP::PACKED_P_U64);
            *elem = x86_64::_mm256_min_epu32(*elem, elem_plus_p);
        });

        // Now elements of output are <= P (Might be enough to do 2P here.)

        apply_external_linear_layer(&mut output_even);
        apply_external_linear_layer(&mut output_odd);
        // Now elements of output are <= 28/35 P (56/70 if we started with 2P)

        [output_even, output_odd]
    }
}

// First constant of external round needs to be saved differently. (R^2rc and positive instead of Rrc)
fn first_final_external_round<FP, const WIDTH: usize, const D: u64>(
    mut state: [[__m256i; WIDTH]; 2],
    round_consts: &[__m256i; WIDTH],
) -> [PackedMontyField31AVX2<FP>; WIDTH]
where
    FP: FieldParameters,
{
    unsafe {
        state.iter_mut().for_each(|substate| {
            substate
                .iter_mut()
                .zip(round_consts.iter())
                .for_each(|(val, &rc)| {
                    *val = x86_64::_mm256_add_epi64(*val, rc);
                    *val = monty_red_unsigned::<FP>(*val);
                    *val = x86_64::_mm256_srli_epi64::<32>(*val);
                    *val = exp_small::<FP, D>(*val);
                })
        });
        let [mut state_evn, state_odd] = state;
        state_evn.iter_mut().zip(state_odd).for_each(|(evn, odd)| {
            let evn_hi = movehdup_epi32(*evn);
            let t = x86_64::_mm256_blend_epi32::<0b10101010>(evn_hi, odd);
            let u = x86_64::_mm256_add_epi32(t, FP::PACKED_P);
            *evn = x86_64::_mm256_min_epu32(t, u)
        });

        let mut output_state = state_evn.map(|val| PackedMontyField31AVX2::<FP>::from_vector(val));

        mds_light_permutation(&mut output_state, &MDSMat4);
        output_state
    }
}

impl<FP, const WIDTH: usize, const D: u64> ExternalLayer<PackedMontyField31AVX2<FP>, WIDTH, D>
    for Poseidon2ExternalLayerMonty31<WIDTH>
where
    FP: FieldParameters,
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

        let num_constants = packed_initial_external_constants.len();
        external_rounds::<FP, WIDTH, D>(
            &mut state,
            &(packed_initial_external_constants[..(num_constants - 1)]),
        );

        // We need to do something special for the last round as we want our output in a different form.
        final_initial_external_round::<FP, WIDTH, D>(
            state,
            &packed_initial_external_constants[num_constants - 1],
        )
    }

    /// Compute the second half of the Poseidon2 external layers.
    fn permute_state_final(
        &self,
        state: [[__m256i; WIDTH]; 2],
        _final_external_constants: &[[MontyField31<FP>; WIDTH]],
        packed_final_external_constants: &[[__m256i; WIDTH]],
    ) -> [PackedMontyField31AVX2<FP>; WIDTH] {
        // Need to do something slightly special for the first round.
        let mut output_state =
            first_final_external_round::<FP, WIDTH, D>(state, &packed_final_external_constants[0]);
        external_rounds::<FP, WIDTH, D>(&mut output_state, &packed_final_external_constants[1..]);
        output_state
    }
}
