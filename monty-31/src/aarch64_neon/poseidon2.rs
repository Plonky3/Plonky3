//! Vectorized Neon implementation of Poseidon2 for MontyField31

use alloc::vec::Vec;
use core::arch::aarch64::{self, int32x4_t, uint32x4_t};
use core::marker::PhantomData;
use core::mem::transmute;

use p3_field::{PrimeCharacteristicRing, uint32x4_mod_add, uint32x4_mod_sub};
use p3_poseidon2::{
    ExternalLayer, ExternalLayerConstants, ExternalLayerConstructor, InternalLayer,
    InternalLayerConstructor, MDSMat4, external_initial_permute_state,
    external_terminal_permute_state,
};

use super::exp_small;
use super::utils::halve_neon;
use crate::{
    FieldParameters, MontyField31, MontyParameters, PackedMontyField31Neon, PackedMontyParameters,
    RelativelyPrimePower,
};

/// A specialized representation of the Poseidon2 state for a width of 16.
///
/// The primary purpose of this struct is to optimize the internal rounds by separating the state
/// into two distinct parts:
///
/// 1. `s0`: The first element, which is the only one to undergo the S-box operation.
/// 2. `s_hi`: The remaining 15 elements, which only undergo the linear layer transformation.
///
/// By splitting the state in this way, we provide a strong hint to the compiler that the S-box
/// on `s0` and the linear operations on `s_hi` are independent data paths. This allows the CPU's
/// out-of-order execution engine to run these operations in parallel, effectively hiding the S-box latency
/// and significantly improving performance.
#[derive(Clone, Copy)]
#[repr(C)] // This is needed to make `transmute`s safe.
pub struct InternalLayer16<PMP: PackedMontyParameters> {
    /// The first element of the state, which undergoes the S-box transformation.
    s0: PackedMontyField31Neon<PMP>,
    /// The remaining 15 elements of the state, which undergo the linear layer transformation.
    s_hi: [uint32x4_t; 15],
}

impl<PMP: PackedMontyParameters> InternalLayer16<PMP> {
    /// Converts the specialized `InternalLayer16` representation into a standard array `[PackedMontyField31Neon<PMP>; 16]`.
    ///
    /// This is a zero-cost conversion that leverages the `#[repr(C)]` layout of the struct.
    ///
    /// # Safety
    /// The caller *must* ensure that every raw `uint32x4_t` vector within `self.s_hi` contains
    /// valid `MontyField31` elements in canonical form `[0, P)`.
    #[inline]
    unsafe fn to_packed_field_array(self) -> [PackedMontyField31Neon<PMP>; 16] {
        unsafe {
            // This `transmute` is safe because `InternalLayer16` is `#[repr(C)]` and is guaranteed
            // to have the exact same memory layout as the target array `[PackedMontyField31Neon<PMP>; 16]`.
            transmute(self)
        }
    }

    /// Converts a standard array `[PackedMontyField31Neon<PMP>; 16]` into the specialized `InternalLayer16` representation.
    ///
    /// This is a zero-cost conversion that leverages the `#[repr(C)]` layout of the struct.
    #[inline]
    #[must_use]
    fn from_packed_field_array(vector: [PackedMontyField31Neon<PMP>; 16]) -> Self {
        unsafe {
            // This `transmute` is safe because `InternalLayer16` is `#[repr(C)]` and so is guaranteed
            // to have the exact same memory layout as `[PackedMontyField31Neon<PMP>; 16]`.
            transmute(vector)
        }
    }
}

/// A specialized representation of the Poseidon2 state for a width of 24.
///
/// The primary purpose of this struct is to optimize the internal rounds by separating the state
/// into two distinct parts:
///
/// 1. `s0`: The first element, which is the only one to undergo the S-box operation.
/// 2. `s_hi`: The remaining 23 elements, which only undergo the linear layer transformation.
///
/// By splitting the state in this way, we provide a strong hint to the compiler that the S-box
/// on `s0` and the linear operations on `s_hi` are independent data paths. This allows the CPU's
/// out-of-order execution engine to run these operations in parallel, effectively hiding the S-box latency
/// and significantly improving performance.
#[derive(Clone, Copy)]
#[repr(C)] // This is needed to make `transmute`s safe.
pub struct InternalLayer24<PMP: PackedMontyParameters> {
    /// The first element of the state, which undergoes the S-box transformation.
    s0: PackedMontyField31Neon<PMP>,
    /// The remaining 23 elements of the state, which undergo the linear layer transformation.
    s_hi: [uint32x4_t; 23],
}

impl<PMP: PackedMontyParameters> InternalLayer24<PMP> {
    /// Converts the specialized `InternalLayer24` representation into a standard array `[PackedMontyField31Neon<PMP>; 24]`.
    ///
    /// This is a zero-cost conversion that leverages the `#[repr(C)]` layout of the struct.
    ///
    /// # Safety
    /// The caller *must* ensure that every raw `uint32x4_t` vector within `self.s_hi` contains
    /// valid `MontyField31` elements in canonical form `[0, P)`.
    #[inline]
    unsafe fn to_packed_field_array(self) -> [PackedMontyField31Neon<PMP>; 24] {
        unsafe {
            // This `transmute` is safe because `InternalLayer24` is `#[repr(C)]` and is guaranteed
            // to have the exact same memory layout as the target array `[PackedMontyField31Neon<PMP>; 24]`.
            transmute(self)
        }
    }

    /// Converts a standard array `[PackedMontyField31Neon<PMP>; 24]` into the specialized `InternalLayer24` representation.
    ///
    /// This is a zero-cost conversion that leverages the `#[repr(C)]` layout of the struct.
    #[inline]
    #[must_use]
    fn from_packed_field_array(vector: [PackedMontyField31Neon<PMP>; 24]) -> Self {
        unsafe {
            // This `transmute` is safe because `InternalLayer24` is `#[repr(C)]` and so is guaranteed
            // to have the exact same memory layout as `[PackedMontyField31Neon<PMP>; 24]`.
            transmute(vector)
        }
    }
}

/// Represents the internal layers of the Poseidon2 permutation for NEON-accelerated operations.
#[derive(Debug, Clone)]
pub struct Poseidon2InternalLayerMonty31<
    PMP: PackedMontyParameters,
    const WIDTH: usize,
    ILP: InternalLayerParametersNeon<PMP, WIDTH>,
> {
    /// The original, scalar round constants for each internal round.
    pub(crate) internal_constants: Vec<MontyField31<PMP>>,
    /// The pre-processed round constants, packed into NEON vectors in negative form (`c - P`).
    ///
    /// This format is optimized for the vectorized permutation loop.
    packed_internal_constants: Vec<int32x4_t>,
    _phantom: PhantomData<ILP>,
}

impl<FP: FieldParameters, const WIDTH: usize, ILP: InternalLayerParametersNeon<FP, WIDTH>>
    InternalLayerConstructor<MontyField31<FP>> for Poseidon2InternalLayerMonty31<FP, WIDTH, ILP>
{
    fn new_from_constants(internal_constants: Vec<MontyField31<FP>>) -> Self {
        let packed_internal_constants = internal_constants
            .iter()
            .map(|c| convert_to_vec_neg_form_neon::<FP>(c.value as i32))
            .collect();
        Self {
            internal_constants,
            packed_internal_constants,
            _phantom: PhantomData,
        }
    }
}

impl<FP, ILP, const D: u64> InternalLayer<PackedMontyField31Neon<FP>, 16, D>
    for Poseidon2InternalLayerMonty31<FP, 16, ILP>
where
    FP: FieldParameters + RelativelyPrimePower<D>,
    ILP: InternalLayerParametersNeon<FP, 16, ArrayLike = [uint32x4_t; 15]>,
{
    fn permute_state(&self, state: &mut [PackedMontyField31Neon<FP>; 16]) {
        unsafe {
            // Safety: This returns values in canonical form when given values in canonical form.
            /*
                Fix a vector v and let Diag(v) denote the diagonal matrix with diagonal given by v.
                Additionally, let 1 denote the matrix with all elements equal to 1.
                The internal layer consists of an sbox operation then a matrix multiplication by 1 + Diag(v).
                Explicitly the internal layer consists of the following 2 operations:

                s0 -> (s0 + rc)^d
                s -> (1 + Diag(v))s

                Note that this matrix multiplication can be implemented as:
                sum = sum_i s_i
                s_i -> sum + s_iv_i

                which is essentially how we implement it.
            */

            // Convert the state array into the specialized `InternalLayer16` representation
            // for optimized processing.
            let mut internal_state = InternalLayer16::from_packed_field_array(*state);

            self.packed_internal_constants.iter().for_each(|&rc| {
                // Apply AddRoundConstant and the S-Box to the first state element (`s0`).
                add_rc_and_sbox::<FP, D>(&mut internal_state.s0, rc);

                // Compute the sum of all other state elements (`s_hi`).
                //
                // This can execute in parallel with the S-box operation on `s0`.
                let s_hi_transmuted: &[PackedMontyField31Neon<FP>; 15] =
                    transmute(&internal_state.s_hi);
                let sum_tail = PackedMontyField31Neon::<FP>::sum_array::<15>(s_hi_transmuted);

                // Perform the diagonal multiplication on `s_hi`.
                //
                // This can also execute in parallel with the S-box.
                ILP::diagonal_mul(&mut internal_state.s_hi);

                // Compute the total sum of the entire state (`s0` + `sum_tail`)
                // after the S-box on `s0` is complete.
                let sum = sum_tail + internal_state.s0;

                // Update `s0`. The transformation is `s0_new = sum + v_0*s0`.
                //
                // Since the diagonal element `v_0` is always -2, this becomes `sum - 2*s0`.
                //
                // For efficiency, this is calculated as `(sum - s0) - s0`, which is `sum_tail - s0`.
                internal_state.s0 = sum_tail - internal_state.s0;

                // Update the rest of the state. For each `s_i` (where `i > 0`), the
                // transformation is `s_i_new = (v_i * s_i) + sum`. The `v_i * s_i` part
                // was already computed in `diagonal_mul`.
                ILP::add_sum(
                    &mut internal_state.s_hi,
                    transmute::<PackedMontyField31Neon<FP>, uint32x4_t>(sum),
                );
            });

            // Convert the specialized state representation back to a standard array before returning.
            *state = InternalLayer16::to_packed_field_array(internal_state);
        }
    }
}

impl<FP, ILP, const D: u64> InternalLayer<PackedMontyField31Neon<FP>, 24, D>
    for Poseidon2InternalLayerMonty31<FP, 24, ILP>
where
    FP: FieldParameters + RelativelyPrimePower<D>,
    ILP: InternalLayerParametersNeon<FP, 24, ArrayLike = [uint32x4_t; 23]>,
{
    fn permute_state(&self, state: &mut [PackedMontyField31Neon<FP>; 24]) {
        unsafe {
            // Safety: This returns values in canonical form when given values in canonical form.

            /*
                Fix a vector v and let Diag(v) denote the diagonal matrix with diagonal given by v.
                Additionally, let 1 denote the matrix with all elements equal to 1.
                The internal layer consists of an sbox operation then a matrix multiplication by 1 + Diag(v).
                Explicitly the internal layer consists of the following 2 operations:

                s0 -> (s0 + rc)^d
                s -> (1 + Diag(v))s

                Note that this matrix multiplication is implemented as:
                sum = sum_i s_i
                s_i -> sum + s_iv_i.
            */

            // Convert the state array into the specialized `InternalLayer24` representation
            // for optimized processing.
            let mut internal_state = InternalLayer24::from_packed_field_array(*state);

            self.packed_internal_constants.iter().for_each(|&rc| {
                // Apply AddRoundConstant and the S-Box to the first state element (`s0`).
                add_rc_and_sbox::<FP, D>(&mut internal_state.s0, rc);

                // Compute the sum of all other state elements (`s_hi`).
                //
                // This can execute in parallel with the S-box operation on `s0`.
                let s_hi_transmuted: &[PackedMontyField31Neon<FP>; 23] =
                    transmute(&internal_state.s_hi);
                let sum_tail = PackedMontyField31Neon::<FP>::sum_array::<23>(s_hi_transmuted);

                // Perform the diagonal multiplication on `s_hi`.
                //
                // This can also execute in parallel with the S-box.
                ILP::diagonal_mul(&mut internal_state.s_hi);

                // Compute the total sum of the entire state (`s0` + `sum_tail`)
                // after the S-box on `s0` is complete.
                let sum = sum_tail + internal_state.s0;

                // Update `s0`. The transformation is `s0_new = sum + v_0*s0`.
                //
                // Since the diagonal element `v_0` is always -2, this becomes `sum - 2*s0`.
                //
                // For efficiency, this is calculated as `(sum - s0) - s0`, which is `sum_tail - s0`.
                internal_state.s0 = sum_tail - internal_state.s0;

                // Update the rest of the state. For each `s_i` (where `i > 0`), the
                // transformation is `s_i_new = (v_i * s_i) + sum`. The `v_i * s_i` part
                // was already computed in `diagonal_mul`.
                ILP::add_sum(
                    &mut internal_state.s_hi,
                    transmute::<PackedMontyField31Neon<FP>, uint32x4_t>(sum),
                );
            });

            // Convert the specialized state representation back to a standard array before returning.
            *state = InternalLayer24::to_packed_field_array(internal_state);
        }
    }
}

/// Represents the external layers (initial and terminal) of the Poseidon2 permutation for NEON-accelerated operations.
#[derive(Debug, Clone)]
pub struct Poseidon2ExternalLayerMonty31<MP: MontyParameters, const WIDTH: usize> {
    /// The original, scalar round constants for both initial and terminal external rounds.
    pub(crate) external_constants: ExternalLayerConstants<MontyField31<MP>, WIDTH>,
    /// Pre-processed constants for the initial external rounds, packed into NEON vectors in negative form (`c - P`).
    packed_initial_external_constants: Vec<[int32x4_t; WIDTH]>,
    /// Pre-processed constants for the terminal external rounds, packed into NEON vectors in negative form (`c - P`).
    packed_terminal_external_constants: Vec<[int32x4_t; WIDTH]>,
}

impl<FP: FieldParameters, const WIDTH: usize> ExternalLayerConstructor<MontyField31<FP>, WIDTH>
    for Poseidon2ExternalLayerMonty31<FP, WIDTH>
{
    fn new_from_constants(
        external_constants: ExternalLayerConstants<MontyField31<FP>, WIDTH>,
    ) -> Self {
        let packed_initial_external_constants = external_constants
            .get_initial_constants()
            .iter()
            .map(|arr| arr.map(|c| convert_to_vec_neg_form_neon::<FP>(c.value as i32)))
            .collect();
        let packed_terminal_external_constants = external_constants
            .get_terminal_constants()
            .iter()
            .map(|arr| arr.map(|c| convert_to_vec_neg_form_neon::<FP>(c.value as i32)))
            .collect();
        Self {
            external_constants,
            packed_initial_external_constants,
            packed_terminal_external_constants,
        }
    }
}

impl<FP, const D: u64, const WIDTH: usize> ExternalLayer<PackedMontyField31Neon<FP>, WIDTH, D>
    for Poseidon2ExternalLayerMonty31<FP, WIDTH>
where
    FP: FieldParameters + RelativelyPrimePower<D>,
{
    /// Perform the initial external layers of the Poseidon2 permutation on the given state.
    fn permute_state_initial(&self, state: &mut [PackedMontyField31Neon<FP>; WIDTH]) {
        external_initial_permute_state(
            state,
            &self.packed_initial_external_constants,
            add_rc_and_sbox::<FP, D>,
            &MDSMat4,
        );
    }

    /// Perform the terminal external layers of the Poseidon2 permutation on the given state.
    fn permute_state_terminal(&self, state: &mut [PackedMontyField31Neon<FP>; WIDTH]) {
        external_terminal_permute_state(
            state,
            &self.packed_terminal_external_constants,
            add_rc_and_sbox::<FP, D>,
            &MDSMat4,
        );
    }
}

/// Converts a scalar constant into a packed NEON vector in "negative form".
///
/// Instead of storing a constant `c`, we pre-compute `c' = c - P` and store it as a packed vector.
#[inline(always)]
fn convert_to_vec_neg_form_neon<MP: MontyParameters>(input: i32) -> int32x4_t {
    unsafe {
        let input_sub_p = input - (MP::PRIME as i32);

        // Broadcast (duplicate) the scalar result into all four lanes of a 128-bit NEON vector.
        aarch64::vdupq_n_s32(input_sub_p)
    }
}

/// Performs the fused AddRoundConstant and S-Box operation `x -> (x + c)^D`.
///
/// # Safety
/// - `val` must contain elements in canonical form `[0, P)`.
/// - `rc` must contain round constants saved in negative form, i.e. as elements in `[-P, 0)`.
fn add_rc_and_sbox<PMP, const D: u64>(val: &mut PackedMontyField31Neon<PMP>, rc: int32x4_t)
where
    PMP: PackedMontyParameters + FieldParameters,
{
    unsafe {
        // Convert the field element vector to its raw signed i32 representation.
        let vec_val_s = val.to_signed_vector();

        // Add the round constant. As it is saved in negative form the result is
        // guaranteed to be in the range `[-P, P)`.
        let val_plus_rc = aarch64::vaddq_s32(vec_val_s, rc);

        // Apply the power S-box `x -> x^D`.
        //
        // The `exp_small` function:
        // - accepts inputs in `[-P, P]`,
        // - returns a canonical result in `[0, P)`.
        let output = exp_small::<PMP, D>(val_plus_rc);

        *val = PackedMontyField31Neon::<PMP>::from_vector(output);
    }
}

/// Trait for NEON-specific parameters and operations for the Poseidon2 internal layer.
pub trait InternalLayerParametersNeon<PMP: PackedMontyParameters, const WIDTH: usize>:
    Clone + Sync
{
    /// The state array type used in this implementation, which holds NEON vectors.
    type ArrayLike: AsMut<[uint32x4_t]>;

    // The internal layer linear transformation is `M = I + Diag(V)`, where `I` is the all-ones matrix.
    // The operation `x -> M * x` can be split into two parts:
    // 1. `diagonal_mul`: A cheap, element-wise multiplication `x_i -> v_i * x_i`.
    // 2. `add_sum`: An expensive sum of all elements, added back to each element.
    //
    // This split is a key performance optimization. The S-box `(s0 + rc)^D` has high latency.
    // By splitting the linear layer, we can execute the cheap `diagonal_mul` operation on the rest
    // of the state `s_hi` in parallel with the S-box operation on `s0`, effectively hiding its latency.
    // Only after both are complete do we compute the sum and perform the final additions.
    //
    // For these reasons, `diagonal_mul` and `add_sum` are marked `unsafe` as they are only
    // guaranteed to be correct when used together in the permutation's internal loop.

    /// Performs the element-wise multiplication part of the internal layer's linear transformation.
    ///
    /// The diagonal matrix used in Poseidon2 has a common structure for the first 9 elements:
    /// `V = [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, ...]`.
    /// This function is composed of two sub-routines that handle different parts of this multiplication.
    ///
    /// # Safety
    /// This function is only safe to call as part of the internal layer permutation.
    /// Its output is not guaranteed to be in canonical form and must be passed directly to `add_sum`.
    #[inline(always)]
    unsafe fn diagonal_mul(input: &mut Self::ArrayLike) {
        // The `unsafe` block is for clarity, as both sub-functions are unsafe.
        unsafe {
            Self::diagonal_mul_first_eight(input);
            Self::diagonal_mul_remainder(input);
        }
    }

    /// Multiplies the first 8 elements of `input` (from index 1 to 8 of the state) by the diagonal vector.
    ///
    /// This default implementation handles the common prefix of the diagonal matrix `[1, 2, 1/2, 3, 4, -1/2, -3, -4]`.
    ///
    /// Note that for negative coefficients (e.g., -1/2), this function computes the positive equivalent (1/2).
    /// This irregularity is cheaper to compute and is corrected later in the `add_sum` function, which
    /// will perform a subtraction instead of an addition for these elements.
    ///
    /// # Safety
    /// This function's output must be passed directly to `add_sum` for the result to be correct.
    #[inline(always)]
    unsafe fn diagonal_mul_first_eight(input: &mut Self::ArrayLike) {
        let input = input.as_mut();

        // The first element `input[0]` (state `s_1`) is multiplied by 1, which is a no-op.

        // Multiply `input[1]` (state `s_2`) by 2.
        input[1] = uint32x4_mod_add(input[1], input[1], PMP::PACKED_P);
        // Multiply `input[2]` (state `s_3`) by 1/2.
        input[2] = halve_neon::<PMP>(input[2]);

        // Multiply `input[3]` (state `s_4`) by 3.
        let acc3 = uint32x4_mod_add(input[3], input[3], PMP::PACKED_P);
        input[3] = uint32x4_mod_add(acc3, input[3], PMP::PACKED_P);

        // Multiply `input[4]` (state `s_5`) by 4.
        let acc4 = uint32x4_mod_add(input[4], input[4], PMP::PACKED_P);
        input[4] = uint32x4_mod_add(acc4, acc4, PMP::PACKED_P);

        // For the next 3 elements, we multiply by the positive value (e.g., 1/2 instead of -1/2).
        // `add_sum` will correct this by subtracting the result instead of adding.

        // Multiply `input[5]` (state `s_6`) by 1/2.
        input[5] = halve_neon::<PMP>(input[5]);
        // Multiply `input[6]` (state `s_7`) by 3.
        let acc6 = uint32x4_mod_add(input[6], input[6], PMP::PACKED_P);
        input[6] = uint32x4_mod_add(acc6, input[6], PMP::PACKED_P);
        // Multiply `input[7]` (state `s_8`) by 4.
        let acc7 = uint32x4_mod_add(input[7], input[7], PMP::PACKED_P);
        input[7] = uint32x4_mod_add(acc7, acc7, PMP::PACKED_P);
    }

    /// Multiplies the remaining elements of the state by the rest of the diagonal matrix.
    ///
    /// This function is implemented by each concrete field.
    ///
    /// # Safety
    /// This function must not touch the first 8 elements of `input`.
    unsafe fn diagonal_mul_remainder(input: &mut Self::ArrayLike);

    /// The number of *positive* inverse powers of two in the diagonal matrix after the first 8 elements.
    ///
    /// This constant is used by `add_sum` to know how many elements require correction (subtraction).
    const NUM_POS: usize;

    /// Adds the state sum to each element, completing the linear layer transformation.
    ///
    /// Morally, this function computes `x_i -> x_i + sum`. However, it also corrects for the
    /// irregularities introduced in `diagonal_mul` for efficiency. Where `diagonal_mul` computed `k*x`
    /// instead of `-k*x`, this function computes `sum - (k*x)` to get the correct result.
    ///
    /// # Safety
    /// `sum` must be in canonical form, and `input` must be the immediate output of `diagonal_mul`.
    #[inline(always)]
    unsafe fn add_sum(input: &mut Self::ArrayLike, sum: uint32x4_t) {
        // For the first 5 elements (s_1 to s_5), the diagonal coefficients are positive, so we add the sum.
        input.as_mut()[..5]
            .iter_mut()
            .for_each(|x| *x = uint32x4_mod_add(sum, *x, PMP::PACKED_P));

        // For the next 3 elements (s_6 to s_8), the diagonal coefficients are negative, so we
        // subtract the result from the sum.
        input.as_mut()[5..8]
            .iter_mut()
            .for_each(|x| *x = uint32x4_mod_sub(sum, *x, PMP::PACKED_P));

        // For the next block of elements, the diagonal coefficients are positive, so we add the sum.
        input.as_mut()[8..(8 + Self::NUM_POS)]
            .iter_mut()
            .for_each(|x| *x = uint32x4_mod_add(sum, *x, PMP::PACKED_P));

        // For the final block of elements, the diagonal coefficients are negative, so we
        // subtract the result from the sum.
        input.as_mut()[8 + Self::NUM_POS..]
            .iter_mut()
            .for_each(|x| *x = uint32x4_mod_sub(sum, *x, PMP::PACKED_P));
    }
}
