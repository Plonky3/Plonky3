//! Vectorized AVX2 implementation of Poseidon2 for MontyField31

use alloc::vec::Vec;
use core::arch::x86_64::{self, __m256i};
use core::marker::PhantomData;
use core::mem::transmute;

use p3_poseidon2::{
    mds_light_permutation, sum_15, sum_23, ExternalLayer, ExternalLayerConstants,
    ExternalLayerConstructor, InternalLayer, InternalLayerConstructor, MDSMat4,
};

use crate::{
    apply_func_to_even_odd, packed_exp_3, packed_exp_5, packed_exp_7, FieldParameters,
    MontyField31, MontyParameters, PackedMontyField31AVX2, PackedMontyParameters,
};

// In the internal layers, it is valuable to treat the first entry of the state differently
// as it is the only entry to which we apply s-box.
// It seems to help the compiler if we introduce a different data structure for these layers.
// Note that we use this structure instead of a tuple so we can force the memory layout to align for transmutes.
#[derive(Clone, Copy)]
#[repr(C)] // This is needed to make `transmute`s safe.
pub struct InternalLayer16<PMP: PackedMontyParameters> {
    s0: PackedMontyField31AVX2<PMP>,
    s_hi: [__m256i; 15],
}

impl<PMP: PackedMontyParameters> InternalLayer16<PMP> {
    #[inline]
    #[must_use]
    /// Convert from `InternalLayer16<PMP>` to `[PackedMontyField31AVX2<PMP>; 16]`
    ///
    /// SAFETY: The caller must ensure that each element of `s_hi` represents a valid `MontyField31<PMP>`.
    /// In particular, each element of each vector must be in `[0, P)` (canonical form).
    unsafe fn to_packed_field_array(self) -> [PackedMontyField31AVX2<PMP>; 16] {
        // Safety: It is up to the user to ensure that elements of `s_hi` represent valid
        // `MontyField31<PMP>` values. We must only reason about memory representations.
        // As described in packing.rs, PackedMontyField31AVX2<PMP> can be transmuted to and from `__m256i`.

        // `InternalLayer16` is `repr(C)` so its memory layout looks like:
        // `[PackedMontyField31AVX2<PMP>, __m256i, ..., __m256i]`
        // Thus as `__m256i` can be can be transmuted to `PackedMontyField31AVX2<FP>`,
        // `InternalLayer16` can be transmuted to `[PackedMontyField31AVX2<FP>; 16]`.
        transmute(self)
    }

    #[inline]
    #[must_use]
    /// Convert from `[PackedMontyField31AVX2<PMP>; 16]` to `InternalLayer16<PMP>`
    fn from_packed_field_array(vector: [PackedMontyField31AVX2<PMP>; 16]) -> Self {
        unsafe {
            // Safety: As described in packing.rs, PackedMontyField31AVX2<PMP> can be transmuted to and from `__m256i`.

            // `InternalLayer16` is `repr(C)` so its memory layout looks like:
            // `[PackedMontyField31AVX2<PMP>, __m256i, ..., __m256i]`
            // Thus as `PackedMontyField31AVX2<FP>` can be can be transmuted to `__m256i`,
            // `[PackedMontyField31AVX2<FP>; 16]` can be transmuted to `InternalLayer16`.
            transmute(vector)
        }
    }
}

#[derive(Clone, Copy)]
#[repr(C)] // This is needed to make `transmute`s safe.
pub struct InternalLayer24<PMP: PackedMontyParameters> {
    s0: PackedMontyField31AVX2<PMP>,
    s_hi: [__m256i; 23],
}

impl<PMP: PackedMontyParameters> InternalLayer24<PMP> {
    #[inline]
    #[must_use]
    /// Convert from `InternalLayer24<PMP>` to `[PackedMontyField31AVX2<PMP>; 24]`
    ///
    /// SAFETY: The caller must ensure that each element of `s_hi` represents a valid `MontyField31<PMP>`.
    /// In particular, each element of each vector must be in `[0, P)` (canonical form).
    unsafe fn to_packed_field_array(self) -> [PackedMontyField31AVX2<PMP>; 24] {
        // Safety: As described in packing.rs, PackedMontyField31AVX2<PMP> can be transmuted to and from `__m256i`.

        // `InternalLayer24` is `repr(C)` so its memory layout looks like:
        // `[PackedMontyField31AVX2<PMP>, __m256i, ..., __m256i]`
        // Thus as `__m256i` can be can be transmuted to `PackedMontyField31AVX2<FP>`,
        // `InternalLayer24` can be transmuted to `[PackedMontyField31AVX2<FP>; 24]`.
        transmute(self)
    }

    #[inline]
    #[must_use]
    /// Convert from `[PackedMontyField31AVX2<PMP>; 24]` to `InternalLayer24<PMP>`
    fn from_packed_field_array(vector: [PackedMontyField31AVX2<PMP>; 24]) -> Self {
        unsafe {
            // Safety: As described in packing.rs, PackedMontyField31AVX2<PMP> can be transmuted to and from `__m256i`.

            // `InternalLayer24` is `repr(C)` so its memory layout looks like:
            // `[PackedMontyField31AVX2<PMP>, __m256i, ..., __m256i]`
            // Thus as `PackedMontyField31AVX2<FP>` can be can be transmuted to `__m256i`,
            // `[PackedMontyField31AVX2<FP>; 24]` can be transmuted to `InternalLayer24`.
            transmute(vector)
        }
    }
}

/// A struct containing the constants and methods needed to perform the internal layers of the Poseidon2 permutation.
#[derive(Debug, Clone)]
pub struct Poseidon2InternalLayerMonty31<
    PMP: PackedMontyParameters,
    const WIDTH: usize,
    ILP: InternalLayerParametersAVX2<WIDTH>,
> {
    pub(crate) internal_constants: Vec<MontyField31<PMP>>,
    packed_internal_constants: Vec<__m256i>,
    _phantom: PhantomData<ILP>,
}

impl<FP: FieldParameters, const WIDTH: usize, ILP: InternalLayerParametersAVX2<WIDTH>>
    InternalLayerConstructor<PackedMontyField31AVX2<FP>>
    for Poseidon2InternalLayerMonty31<FP, WIDTH, ILP>
{
    /// Construct an instance of Poseidon2InternalLayerMersenne31AVX2 from a vector containing
    /// the constants for each round. Internally, the constants are transformed into the
    /// {-P, ..., 0} representation instead of the standard {0, ..., P} one.
    fn new_from_constants(internal_constants: Vec<MontyField31<FP>>) -> Self {
        let packed_internal_constants = internal_constants
            .iter()
            .map(|constant| convert_to_vec_neg_form::<FP>(constant.value as i32))
            .collect();
        Self {
            internal_constants,
            packed_internal_constants,
            _phantom: PhantomData,
        }
    }
}

/// A struct containing the constants and methods needed to perform the external layers of the Poseidon2 permutation.
#[derive(Debug, Clone)]
pub struct Poseidon2ExternalLayerMonty31<PMP: PackedMontyParameters, const WIDTH: usize> {
    pub(crate) external_constants: ExternalLayerConstants<MontyField31<PMP>, WIDTH>,
    packed_initial_external_constants: Vec<[__m256i; WIDTH]>,
    packed_terminal_external_constants: Vec<[__m256i; WIDTH]>,
}

impl<FP: FieldParameters, const WIDTH: usize>
    ExternalLayerConstructor<PackedMontyField31AVX2<FP>, WIDTH>
    for Poseidon2ExternalLayerMonty31<FP, WIDTH>
{
    /// Construct an instance of Poseidon2ExternalLayerMersenne31AVX2 from a array of
    /// vectors containing the constants for each round. Internally, the constants
    ///  are transformed into the {-P, ..., 0} representation instead of the standard {0, ..., P} one.
    fn new_from_constants(
        external_constants: ExternalLayerConstants<MontyField31<FP>, WIDTH>,
    ) -> Self {
        let packed_initial_external_constants = external_constants.get_initial_constants()
            .iter()
            .map(|array| array.map(|constant| convert_to_vec_neg_form::<FP>(constant.value as i32)))
            .collect();
        let packed_terminal_external_constants = external_constants.get_terminal_constants()
            .iter()
            .map(|array| array.map(|constant| convert_to_vec_neg_form::<FP>(constant.value as i32)))
            .collect();
        Self {
            external_constants,
            packed_initial_external_constants,
            packed_terminal_external_constants,
        }
    }
}

/// Use hard coded methods to compute x -> x^d for the even index entries and small d.
/// Inputs should be signed 32-bit integers in [-P, ..., P].
/// Outputs will also be signed integers in (-P, ..., P) stored in the odd indices.
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

/// Compute val -> (val + rc)^D. Each entry of val should be represented in canonical form.
/// Each entry of rc should be represented by an element in in [-P, 0].
/// Each entry of the output will be represented by an element in canonical form.
/// If the inputs do not conform to this representation, the result is undefined.
#[inline(always)]
#[must_use]
fn add_rc_and_sbox<PMP: PackedMontyParameters, const D: u64>(
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

/// A trait containing the specific information needed to
/// implement the Poseidon2 Permutation for Monty31 Fields.
pub trait InternalLayerParametersAVX2<const WIDTH: usize>: Clone + Sync {
    type ArrayLike;

    // diagonal_mul and add_sum morally should be one function but are split because diagonal_mul can happen simultaneously to
    // the sbox being applied to the first element of the state which is advantageous as this s-box has very high latency.
    // However these functions should only ever be used together and we only make safety guarantees about the output
    // of the combined function add_sum(diagonal_mul(state), sum) which will output field elements in canonical form provided inputs are in canonical form.

    // Diagonal_mul will not output field elements in canonical form and indeed may even output incorrect values in places where
    // it is efficient to pipe computation to add_sum. E.g. it might output 3*x instead of -3*x and then add_sum does sum - x.
    // Similarly add_sum assumes its input has been piped directly from diagonal_mul so might assume that some inputs
    // are the negative of the correct value or in some form other than canonical.

    // For these reason we mark both functions as unsafe.

    /// # Safety
    ///
    /// This function assumes its output is piped directly into add_sum.
    unsafe fn diagonal_mul(input: &mut Self::ArrayLike);

    /// # Safety
    ///
    /// This function assumes its input is taken directly from diagonal_mul.
    unsafe fn add_sum(input: &mut Self::ArrayLike, sum: __m256i);
}

/// Convert elements from canonical form [0, P) to a negative form in [-P, ..., 0) and copy into a vector.
#[inline(always)]
fn convert_to_vec_neg_form<MP: MontyParameters>(input: i32) -> __m256i {
    let input_sub_p = input - (MP::PRIME as i32);
    unsafe {
        // Safety: If this code got compiled then AVX2 intrinsics are available.
        x86_64::_mm256_set1_epi32(input_sub_p)
    }
}

impl<FP, ILP, const D: u64> InternalLayer<PackedMontyField31AVX2<FP>, 16, D>
    for Poseidon2InternalLayerMonty31<FP, 16, ILP>
where
    FP: FieldParameters,
    ILP: InternalLayerParametersAVX2<16, ArrayLike = [__m256i; 15]>,
{
    type InternalState = InternalLayer16<FP>;

    /// Compute a collection of Poseidon2 internal layers.
    /// One layer for every constant supplied.
    fn permute_state(&self, state: &mut Self::InternalState) {
        unsafe {
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
            self.packed_internal_constants.iter().for_each(|&rc| {
                state.s0 = add_rc_and_sbox::<FP, D>(state.s0, rc); // s0 -> (s0 + rc)^D
                let sum_non_0 = sum_15(
                    &transmute::<[__m256i; 15], [PackedMontyField31AVX2<FP>; 15]>(state.s_hi),
                ); // Get the sum of all elements other than s0.
                ILP::diagonal_mul(&mut state.s_hi); // si -> vi * si for all i > 0.
                let sum = sum_non_0 + state.s0; // Get the full sum.
                state.s0 = sum_non_0 - state.s0; // s0 -> sum - 2*s0 = sum_non_0 - s0.
                ILP::add_sum(
                    &mut state.s_hi,
                    transmute::<PackedMontyField31AVX2<FP>, __m256i>(sum),
                ); // si -> si + sum for all i > 0.
            })
        }
    }
}

impl<FP, ILP, const D: u64> InternalLayer<PackedMontyField31AVX2<FP>, 24, D>
    for Poseidon2InternalLayerMonty31<FP, 24, ILP>
where
    FP: FieldParameters,
    ILP: InternalLayerParametersAVX2<24, ArrayLike = [__m256i; 23]>,
{
    type InternalState = InternalLayer24<FP>;

    /// Compute a collection of Poseidon2 internal layers.
    /// One layer for every constant supplied.
    fn permute_state(&self, state: &mut Self::InternalState) {
        unsafe {
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

            self.packed_internal_constants.iter().for_each(|&rc| {
                state.s0 = add_rc_and_sbox::<FP, D>(state.s0, rc); // s0 -> (s0 + rc)^D
                let sum_non_0 = sum_23(
                    &transmute::<[__m256i; 23], [PackedMontyField31AVX2<FP>; 23]>(state.s_hi),
                ); // Get the sum of all elements other than s0.
                ILP::diagonal_mul(&mut state.s_hi); // si -> vi * si for all i > 0.
                let sum = sum_non_0 + state.s0; // Get the full sum.
                state.s0 = sum_non_0 - state.s0; // s0 -> sum - 2*s0 = sum_non_0 - s0.
                ILP::add_sum(
                    &mut state.s_hi,
                    transmute::<PackedMontyField31AVX2<FP>, __m256i>(sum),
                ); // si -> si + sum for all i > 0.
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
    /*
        The external layer consists of the following 2 operations:

        s -> s + rc
        s -> s^d
        s -> Ms

        Where by s^d we mean to apply this power function element wise.

        Multiplication by M is implemented efficiently in p3_poseidon2/matrix.
    */
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

impl<FP, const D: u64> ExternalLayer<PackedMontyField31AVX2<FP>, 16, D>
    for Poseidon2ExternalLayerMonty31<FP, 16>
where
    FP: FieldParameters,
{
    type InternalState = InternalLayer16<FP>;

    /// Compute the first half of the Poseidon2 external layers.
    fn permute_state_initial(
        &self,
        mut state: [PackedMontyField31AVX2<FP>; 16],
    ) -> Self::InternalState {
        mds_light_permutation(&mut state, &MDSMat4);

        external_rounds::<FP, 16, D>(&mut state, &self.packed_initial_external_constants);

        InternalLayer16::from_packed_field_array(state)
    }

    /// Compute the second half of the Poseidon2 external layers.
    fn permute_state_terminal(
        &self,
        state: Self::InternalState,
    ) -> [PackedMontyField31AVX2<FP>; 16] {
        let mut output_state = unsafe { 
            // SAFETY: The internal layer outputs elements in canonical form when given elements in canonical form.
            // Thus to_packed_field_array is safe to use.
            state.to_packed_field_array() 
        };

        external_rounds::<FP, 16, D>(&mut output_state, &self.packed_terminal_external_constants);
        output_state
    }
}

impl<FP, const D: u64> ExternalLayer<PackedMontyField31AVX2<FP>, 24, D>
    for Poseidon2ExternalLayerMonty31<FP, 24>
where
    FP: FieldParameters,
{
    type InternalState = InternalLayer24<FP>;

    /// Compute the first half of the Poseidon2 external layers.
    fn permute_state_initial(
        &self,
        mut state: [PackedMontyField31AVX2<FP>; 24],
    ) -> Self::InternalState {
        mds_light_permutation(&mut state, &MDSMat4);

        external_rounds::<FP, 24, D>(&mut state, &self.packed_initial_external_constants);

        InternalLayer24::from_packed_field_array(state)
    }

    /// Compute the second half of the Poseidon2 external layers.
    fn permute_state_terminal(
        &self,
        state: Self::InternalState,
    ) -> [PackedMontyField31AVX2<FP>; 24] {
        let mut output_state = unsafe { 
            // SAFETY: The internal layer outputs elements in canonical form when given elements in canonical form.
            // Thus to_packed_field_array is safe to use.
            state.to_packed_field_array() 
        };

        external_rounds::<FP, 24, D>(&mut output_state, &self.packed_terminal_external_constants);
        output_state
    }
}
