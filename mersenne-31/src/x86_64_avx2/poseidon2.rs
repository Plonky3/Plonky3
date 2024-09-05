use core::arch::x86_64::{self, __m256i};
use p3_poseidon2::{
    mds_light_permutation, sum_15, sum_23, ExternalLayer, InternalLayer, MDSMat4,
    Poseidon2ExternalPackedConstants, Poseidon2InternalPackedConstants,
};

use crate::{
    exp5, Mersenne31, PackedMersenne31AVX2, Poseidon2ExternalLayerMersenne31,
    Poseidon2InternalLayerMersenne31, P, P_AVX2,
};

/// Convert elements from the standard form {0, ..., P} to {-P, ..., 0} and copy into a vector
fn convert_to_vec_neg_form(input: i32) -> __m256i {
    let input_sub_p = input - (P as i32);
    unsafe {
        // Safety: If this code got compiled then AVX2 intrinsics are available.
        x86_64::_mm256_set1_epi32(input_sub_p)
    }
}

/// We save the round constants in the {-P, ..., 0} representation instead of the standard
/// {0, ..., P} one. This saves several instructions later.
impl Poseidon2InternalPackedConstants<Mersenne31> for Poseidon2InternalLayerMersenne31 {
    type ConstantsType = __m256i;

    fn convert_from_field(internal_constant: &Mersenne31) -> Self::ConstantsType {
        convert_to_vec_neg_form(internal_constant.value as i32)
    }
}

/// We save the round constants in the {-P, ..., 0} representation instead of the standard
/// {0, ..., P} one. This saves several instructions later.
impl<const WIDTH: usize> Poseidon2ExternalPackedConstants<Mersenne31, WIDTH>
    for Poseidon2ExternalLayerMersenne31
{
    type ConstantsType = [__m256i; WIDTH];

    /// Convert elements from the standard form {0, ..., P} to {-P, ..., 0}.
    fn convert_from_field_array(external_constants: &[Mersenne31; WIDTH]) -> [__m256i; WIDTH] {
        external_constants
            .map(|external_constant| convert_to_vec_neg_form(external_constant.value as i32))
    }
}

/// Compute the map x -> 2^I x on Mersenne-31 field elements.
/// x must be represented as a value in {0..P}.
/// This requires 2 generic parameters, I and I_PRIME satisfying I + I_PRIME = 31.
/// If the inputs do not conform to this representations, the result is undefined.
#[inline(always)]
fn mul_2exp_i<const I: i32, const I_PRIME: i32>(val: PackedMersenne31AVX2) -> PackedMersenne31AVX2 {
    assert_eq!(I + I_PRIME, 31);
    unsafe {
        // Safety: If this code got compiled then AVX2 intrinsics are available.
        let input = val.to_vector();

        // In M31, multiplication by 2^n corresponds to a cyclic rotation which
        // is much faster than the naive multiplication method.

        // Shift the low bits up. This also shifts something unwanted into
        // the sign bit so we mark it dirty.
        let hi_bits_dirty = x86_64::_mm256_slli_epi32::<I>(input);

        // Shift the high bits down.
        let lo_bits = x86_64::_mm256_srli_epi32::<I_PRIME>(input);

        // Clear the sign bit.
        let hi_bits = x86_64::_mm256_and_si256(hi_bits_dirty, P_AVX2);

        // Combine the lo and high bits.
        let output = x86_64::_mm256_or_si256(lo_bits, hi_bits);
        PackedMersenne31AVX2::from_vector(output)
    }
}

/// We hard code multiplication by the diagonal minus 1 of our internal matrix (1 + D)
/// In the Mersenne31, WIDTH = 16 case, the diagonal minus 1 is:
/// [-2] + 1 << [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16]
/// i.e. The first entry is -2 and all other entires a power of 2.
#[inline(always)]
fn diagonal_mul_16(state: &mut [PackedMersenne31AVX2; 16]) {
    // The first three entries involve multiplication by -2, 1, 2 which are simple:

    state[2] = state[2] + state[2]; // add is 3 instructions whereas shift is 4.

    // For the remaining entires we use our fast shift code.
    state[3] = mul_2exp_i::<2, 29>(state[3]);
    state[4] = mul_2exp_i::<3, 28>(state[4]);
    state[5] = mul_2exp_i::<4, 27>(state[5]);
    state[6] = mul_2exp_i::<5, 26>(state[6]);
    state[7] = mul_2exp_i::<6, 25>(state[7]);
    state[8] = mul_2exp_i::<7, 24>(state[8]);
    state[9] = mul_2exp_i::<8, 23>(state[9]);
    state[10] = mul_2exp_i::<10, 21>(state[10]);
    state[11] = mul_2exp_i::<12, 19>(state[11]);
    state[12] = mul_2exp_i::<13, 18>(state[12]);
    state[13] = mul_2exp_i::<14, 17>(state[13]);
    state[14] = mul_2exp_i::<15, 16>(state[14]);
    state[15] = mul_2exp_i::<16, 15>(state[15]); // TODO: There is a faster method for 15.
}

/// We hard code multiplication by the diagonal minus 1 of our internal matrix (1 + D)
/// In the Mersenne31, WIDTH = 24 case, the diagonal minus 1 is:
/// [-2] + 1 << [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
/// i.e. The first entry is -2 and all other entires a power of 2.
#[inline(always)]
fn diagonal_mul_24(state: &mut [PackedMersenne31AVX2; 24]) {
    // The first three entries involve multiplication by -2, 1, 2 which are simple:

    state[2] = state[2] + state[2]; // add is 3 instructions whereas shift is 4.

    // For the remaining entires we use our fast shift code.
    state[3] = mul_2exp_i::<2, 29>(state[3]);
    state[4] = mul_2exp_i::<3, 28>(state[4]);
    state[5] = mul_2exp_i::<4, 27>(state[5]);
    state[6] = mul_2exp_i::<5, 26>(state[6]);
    state[7] = mul_2exp_i::<6, 25>(state[7]);
    state[8] = mul_2exp_i::<7, 24>(state[8]);
    state[9] = mul_2exp_i::<8, 23>(state[9]);
    state[10] = mul_2exp_i::<9, 22>(state[10]);
    state[11] = mul_2exp_i::<10, 21>(state[11]);
    state[12] = mul_2exp_i::<11, 20>(state[12]);
    state[13] = mul_2exp_i::<12, 19>(state[13]);
    state[14] = mul_2exp_i::<13, 18>(state[14]);
    state[15] = mul_2exp_i::<14, 17>(state[15]); // TODO: There is a faster method for 15.
    state[16] = mul_2exp_i::<15, 16>(state[16]);
    state[17] = mul_2exp_i::<16, 15>(state[17]);
    state[18] = mul_2exp_i::<17, 14>(state[18]);
    state[19] = mul_2exp_i::<18, 13>(state[19]);
    state[20] = mul_2exp_i::<19, 12>(state[20]);
    state[21] = mul_2exp_i::<20, 11>(state[21]);
    state[22] = mul_2exp_i::<21, 10>(state[22]);
    state[23] = mul_2exp_i::<22, 9>(state[23]);
}

/// Compute the map x -> (x + rc)^5 on Mersenne-31 field elements.
/// x must be represented as a value in {0..P}.
/// rc mut be represented as a value in {-P, ..., 0}.
/// If the inputs do not conform to these representations, the result is undefined.
/// The output will be represented as a value in {0..P}.
#[inline(always)]
fn add_rc_and_sbox(input: PackedMersenne31AVX2, rc: __m256i) -> PackedMersenne31AVX2 {
    unsafe {
        // Safety: If this code got compiled then AVX2 intrinsics are available.
        let input_vec = input.to_vector();
        let input_plus_rc = x86_64::_mm256_add_epi32(input_vec, rc);

        // Due to the representations of input and rc, input_plus_rc is in {-P, ..., P}.
        // This is exactly the required bound to apply sbox.
        let input_post_sbox = exp5(input_plus_rc);
        PackedMersenne31AVX2::from_vector(input_post_sbox)
    }
}

/// Compute a single Poseidon2 internal layer on a state of width 16.
#[inline(always)]
fn internal_16(state: &mut [PackedMersenne31AVX2; 16], rc: __m256i) {
    state[0] = add_rc_and_sbox(state[0], rc);
    let sum_non_0 = sum_15(&state[1..]);
    let sum = sum_non_0 + state[0];
    state[0] = sum_non_0 - state[0];
    diagonal_mul_16(state);
    state.iter_mut().skip(1).for_each(|x| *x += sum);
}

impl InternalLayer<PackedMersenne31AVX2, 16, 5> for Poseidon2InternalLayerMersenne31 {
    type InternalState = [PackedMersenne31AVX2; 16];

    /// Compute the full Poseidon2 internal layer on a state of width 16.
    fn permute_state(
        &self,
        state: &mut Self::InternalState,
        _internal_constants: &[Mersenne31],
        packed_internal_constants: &[__m256i],
    ) {
        packed_internal_constants
            .iter()
            .for_each(|&rc| internal_16(state, rc))
    }
}

/// Compute a single Poseidon2 internal layer on a state of width 24.
#[inline(always)]
fn internal_24(state: &mut [PackedMersenne31AVX2; 24], rc: __m256i) {
    state[0] = add_rc_and_sbox(state[0], rc);
    let sum_non_0 = sum_23(&state[1..]);
    let sum = sum_non_0 + state[0];
    state[0] = sum_non_0 - state[0];
    diagonal_mul_24(state);
    state.iter_mut().skip(1).for_each(|x| *x += sum);
}

impl InternalLayer<PackedMersenne31AVX2, 24, 5> for Poseidon2InternalLayerMersenne31 {
    type InternalState = [PackedMersenne31AVX2; 24];

    /// Compute the full Poseidon2 internal layer on a state of width 24.
    fn permute_state(
        &self,
        state: &mut Self::InternalState,
        _internal_constants: &[Mersenne31],
        packed_internal_constants: &[__m256i],
    ) {
        packed_internal_constants
            .iter()
            .for_each(|&rc| internal_24(state, rc))
    }
}

/// Compute a collection of Poseidon2 external layers.
/// One layer for every constant supplied.
#[inline]
fn external_rounds<const WIDTH: usize>(
    state: &mut [PackedMersenne31AVX2; WIDTH],
    packed_external_constants: &[[__m256i; WIDTH]],
) {
    packed_external_constants.iter().for_each(|round_consts| {
        state
            .iter_mut()
            .zip(round_consts.iter())
            .for_each(|(val, &rc)| *val = add_rc_and_sbox(*val, rc));
        mds_light_permutation(state, &MDSMat4);
    });
}

impl<const WIDTH: usize> ExternalLayer<PackedMersenne31AVX2, WIDTH, 5>
    for Poseidon2ExternalLayerMersenne31
{
    type InternalState = [PackedMersenne31AVX2; WIDTH];

    /// Compute the first half of the Poseidon2 external layers.
    fn permute_state_initial(
        &self,
        mut state: [PackedMersenne31AVX2; WIDTH],
        _initial_external_constants: &[[Mersenne31; WIDTH]],
        packed_initial_external_constants: &[[__m256i; WIDTH]],
    ) -> [PackedMersenne31AVX2; WIDTH] {
        mds_light_permutation(&mut state, &MDSMat4);
        external_rounds(&mut state, packed_initial_external_constants);
        state
    }

    /// Compute the second half of the Poseidon2 external layers.
    fn permute_state_final(
        &self,
        mut state: [PackedMersenne31AVX2; WIDTH],
        _final_external_constants: &[[Mersenne31; WIDTH]],
        packed_final_external_constants: &[[__m256i; WIDTH]],
    ) -> [PackedMersenne31AVX2; WIDTH] {
        external_rounds(&mut state, packed_final_external_constants);
        state
    }
}

#[cfg(test)]
mod tests {
    use p3_field::AbstractField;
    use p3_poseidon2::Poseidon2;
    use p3_symmetric::Permutation;
    use rand::Rng;

    use super::*;

    type F = Mersenne31;
    const D: u64 = 5;
    type Perm16 =
        Poseidon2<F, Poseidon2ExternalLayerMersenne31, Poseidon2InternalLayerMersenne31, 16, D>;
    type Perm24 =
        Poseidon2<F, Poseidon2ExternalLayerMersenne31, Poseidon2InternalLayerMersenne31, 24, D>;

    /// Test that the output is the same as the scalar version on a random input of length 16.
    #[test]
    fn test_avx2_poseidon2_width_16() {
        let mut rng = rand::thread_rng();

        // Our Poseidon2 implementation.
        let poseidon2 = Perm16::new_from_rng_128(
            Poseidon2ExternalLayerMersenne31,
            Poseidon2InternalLayerMersenne31,
            &mut rng,
        );

        let input: [F; 16] = rng.gen();

        let mut expected = input;
        poseidon2.permute_mut(&mut expected);

        let mut avx2_input = input.map(PackedMersenne31AVX2::from_f);
        poseidon2.permute_mut(&mut avx2_input);

        let avx2_output = avx2_input.map(|x| x.0[0]);

        assert_eq!(avx2_output, expected);
    }

    /// Test that the output is the same as the scalar version on a random input of length 24.
    #[test]
    fn test_avx2_poseidon2_width_24() {
        let mut rng = rand::thread_rng();

        // Our Poseidon2 implementation.
        let poseidon2 = Perm24::new_from_rng_128(
            Poseidon2ExternalLayerMersenne31,
            Poseidon2InternalLayerMersenne31,
            &mut rng,
        );

        let input: [F; 24] = rng.gen();

        let mut expected = input;
        poseidon2.permute_mut(&mut expected);

        let mut avx2_input = input.map(PackedMersenne31AVX2::from_f);
        poseidon2.permute_mut(&mut avx2_input);

        let avx2_output = avx2_input.map(|x| x.0[0]);

        assert_eq!(avx2_output, expected);
    }
}
