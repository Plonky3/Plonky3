use core::arch::x86_64::{self, __m256i};
use core::mem::transmute;
use p3_poseidon2::{
    internal_permute_state, matmul_internal, InternalLayer, Packed64bitM31Tensor,
    Poseidon2AVX2Helpers, Poseidon2AVX2Methods,
};

use crate::{
    monty_red, movehdup_epi32, DiffusionMatrixKoalaBear, KoalaBear, PackedKoalaBearAVX2,
    MONTY_INVERSE, POSEIDON2_INTERNAL_MATRIX_DIAG_16_KOALABEAR_MONTY,
    POSEIDON2_INTERNAL_MATRIX_DIAG_24_KOALABEAR_MONTY,
};

const POSEIDON2_INTERNAL_MATRIX_DIAG_16_MONTY_SHIFTS: [u64; 16] =
    [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15];

const POSEIDON2_INTERNAL_MATRIX_DIAG_24_MONTY_SHIFTS: [u64; 24] = [
    0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23,
];

// We need to change from the standard implementation as we are interpreting the matrix (1 + D(v)) as the monty form of the matrix not the raw form.
// matmul_internal internal performs a standard matrix multiplication so we need to additional rescale by the inverse monty constant.
// These will be removed once we have architecture specific implementations.

impl<const D: u64> InternalLayer<PackedKoalaBearAVX2, 16, D> for DiffusionMatrixKoalaBear {
    type InternalState = [PackedKoalaBearAVX2; 16];

    type InternalConstantsType = KoalaBear;

    fn permute_state(
        &self,
        state: &mut Self::InternalState,
        internal_constants: &[Self::InternalConstantsType],
    ) {
        internal_permute_state::<PackedKoalaBearAVX2, 16, D>(
            state,
            |state| {
                matmul_internal::<KoalaBear, PackedKoalaBearAVX2, 16>(
                    state,
                    POSEIDON2_INTERNAL_MATRIX_DIAG_16_KOALABEAR_MONTY,
                );
                state.iter_mut().for_each(|i| *i *= MONTY_INVERSE);
            },
            internal_constants,
        )
    }
}

impl<const D: u64> InternalLayer<PackedKoalaBearAVX2, 24, D> for DiffusionMatrixKoalaBear {
    type InternalState = [PackedKoalaBearAVX2; 24];

    type InternalConstantsType = KoalaBear;

    fn permute_state(
        &self,
        state: &mut Self::InternalState,
        internal_constants: &[Self::InternalConstantsType],
    ) {
        internal_permute_state::<PackedKoalaBearAVX2, 24, D>(
            state,
            |state| {
                matmul_internal::<KoalaBear, PackedKoalaBearAVX2, 24>(
                    state,
                    POSEIDON2_INTERNAL_MATRIX_DIAG_24_KOALABEAR_MONTY,
                );
                state.iter_mut().for_each(|i| *i *= MONTY_INVERSE);
            },
            internal_constants,
        )
    }
}

const P: u32 = 0x7f000001;
const P_4XU64: __m256i = unsafe { transmute::<[u64; 4], _>([P as u64; 4]) };
const MU: __m256i = unsafe { transmute::<[u32; 8], _>([0x81000001; 8]) };

/// A fast "mersenne-like" reduction which works for smallish inputs.
/// Input must be < 2^6 * 2^31.
/// Output will be < 2P.
#[inline]
fn partial_reduce(x: __m256i) -> __m256i {
    unsafe {
        // Safety: Each input must be < 2^6 * 2^31.
        // Given a larger input, the output will still represent the same value mod P but
        // may be larger than 2P.

        // Assume x < 2^37 and write x = x0 + 2^31 x1 with x0 < 2^31. Then
        // x = x0 + (2^24 - 1)*x1 mod P and provided x1 < 2^6, x0 + (2^24 - 1)*x1 < 2^31 + 2^30 < 2P.

        // Get the top 6 bits shifted down.
        let x1 = x86_64::_mm256_srli_epi64::<31>(x);

        // Another option would be to do a mul here to compute (2^24 - 1)*x1 immediately.
        // This saves one op as we just need a mul + add instead of slli + add + sub.
        // Timing however indicates this is slower likely due to the high latency of mul.
        let x1_24 = x86_64::_mm256_slli_epi64::<24>(x1);

        // Zero out the top 33 bits.
        const LOW31: __m256i = unsafe { transmute::<[u64; 4], _>([0x7fffffff; 4]) };
        let x0 = x86_64::_mm256_and_si256(x, LOW31);

        // Add the high bits back to the value and subtract x1.
        let sum = x86_64::_mm256_add_epi64(x0, x1_24);
        x86_64::_mm256_sub_epi64(sum, x1)
    }
}

/// Signed monty reduction is a slight modification on the standard algorithm which takes inputs in
/// [-P^2, P^2] and outputs values in [-P, P] stored in the top 32 bits of each 64 bit word.
#[inline]
#[must_use]
#[allow(non_snake_case)]
pub(crate) fn monty_red_signed(input: __m256i) -> __m256i {
    unsafe {
        // We only care about the bottom 32 bits of the result so epi32 and epu32 are interchangeable here.
        let q = x86_64::_mm256_mul_epi32(input, MU);

        // The epi32 is important here. q (mod 2^32) is interpreted as a value in [-2^31, 2^31).
        let q_P = x86_64::_mm256_mul_epi32(q, P_4XU64);

        x86_64::_mm256_sub_epi64(input, q_P)
    }
}

/// Compute x -> x^3 for each element of the vector.
/// The input must be in [0, 2P).
/// The output will be in (-P, P).
#[inline]
fn joint_sbox(x: __m256i) -> __m256i {
    unsafe {
        // Safety: If input is [0, 2P), no overflow will occur and the output will be in (-P, P).

        let x_sub_p = x86_64::_mm256_sub_epi64(x, P_4XU64);

        // Square x. As |x| < P, x^2 < P^2 < 2^62
        let x2 = x86_64::_mm256_mul_epi32(x_sub_p, x_sub_p);

        // Do a monty reduction to get an element in (-P, P) stored in the top 32 bits.
        let x2_red = monty_red(x2);
        // Copy values to low bits. We ignore the top bits so it's fine to leave them as is.
        let x2_low_32 = movehdup_epi32(x2_red);

        // Find cube of x. Note -P^2 < x3 < P^2.
        let x3 = x86_64::_mm256_mul_epi32(x2_low_32, x_sub_p);

        // Do a signed monty reduction to get an element in (-P, P) stored the top 32 bits.
        let x3_red = monty_red_signed(x3);
        // We need the high bits of the output to be 0 so we need a shift this time.

        x86_64::_mm256_srli_epi64::<32>(x3_red) // This shifts in 0's :(.
                                                // Would be much better to shift in sign bits if that was possible.
                                                // Seems like it should be possible in AVX512 but not AVX2.
    }
}

const fn expand_constant<const N: usize>(input: [u64; N]) -> [[u64; 4]; N] {
    let mut output = [[0; 4]; N];
    let mut acc = 0;
    loop {
        output[acc] = [input[acc]; 4];
        acc += 1;
        if acc == N {
            break;
        }
    }
    output
}

#[derive(Clone)]
pub struct Poseidon2DataKoalaBearAVX2();

impl Poseidon2AVX2Helpers for Poseidon2DataKoalaBearAVX2 {
    /// Given a vector of elements __m256i apply a monty reduction to each u64.
    /// Each u64 input must lie in [-P^2, P^2)
    /// Each output will be a u64 lying in [0, P)
    #[inline]
    fn monty_reduce_vec(state: __m256i) -> __m256i {
        unsafe {
            let red = monty_red_signed(state);
            let red_shift = x86_64::_mm256_srli_epi64::<32>(red); //srli shifts in 0's. Need to shift in sign bits. Not possible in AVX2 unfortunately.
            x86_64::_mm256_add_epi32(red_shift, Self::PRIME)
        }
    }

    /// Given a vector of elements __m256i apply a partial reduction to each u64
    /// Each u64 input must lie in [0, 2^{32}P)
    /// Each output will be a u64 lying in [0, 2P)
    /// Slightly cheaper than full_reduce
    #[inline]
    fn partial_reduce_vec(state: __m256i) -> __m256i {
        partial_reduce(state)
    }

    /// Apply the s-box: x -> x^s for some small s coprime to p - 1 to a vector __m256i.
    /// Input must be 4 u64's all in the range [0, P).
    /// Output will be 4 u64's all in the range [0, 2^{32}P).
    #[inline]
    fn joint_sbox_vec(state: __m256i) -> __m256i {
        joint_sbox(state)
    }

    /// Apply the s-box: x -> (x + rc)^5 to a vector __m256i.
    /// s0 is in [-P, P), rc is in [0, P)
    #[inline]
    fn internal_rc_sbox(s0: __m256i, rc: __m256i) -> __m256i {
        unsafe {
            // Need to get s0 into canonical form.
            let red_s0 = Self::final_reduce_pos_vec(s0);

            // Each entry of sum is <= 2P < 2^32.
            let sum = x86_64::_mm256_add_epi32(red_s0, rc);

            let sbox = joint_sbox(sum); // sbox is in (-P, P) as a u32 with all top bits 0.
            x86_64::_mm256_add_epi32(sbox, Self::PRIME) // Output is now in [0, 2P) so can be used as a 64 bit number.
        }
    }

    const PRIME: __m256i = unsafe { transmute([(P as u64); 4]) };
    const PACKED_8XPRIME: __m256i = unsafe { transmute([(P as u64) << 3; 4]) };
}

impl Poseidon2AVX2Methods<4, 16> for Poseidon2DataKoalaBearAVX2 {
    type PF = PackedKoalaBearAVX2;
    type InternalRep = [Packed64bitM31Tensor<4>; 2];

    const INTERNAL_SHIFTS: Packed64bitM31Tensor<4> = unsafe {
        transmute(expand_constant(
            POSEIDON2_INTERNAL_MATRIX_DIAG_16_MONTY_SHIFTS,
        ))
    };

    /// In memory, [PF; 16] = [[u32; 8]; 16] and we label the elements as:
    /// [[a_{0, 0}, ..., a_{0, 7}], ..., [a_{15, 0}, ..., a_{15, 7}]].
    /// We split each row in 2, expand each element to a u64 and then return vector of __mm256 elements arranged into a tensor.
    #[inline]
    fn from_input(input: [Self::PF; 16]) -> Self::InternalRep {
        unsafe {
            // Safety: Nothing unsafe to worry about.

            let zeros = x86_64::_mm256_setzero_si256();
            let vector_input: [__m256i; 16] = transmute(input);
            let mut output_0 = [zeros; 16];
            let mut output_1 = [zeros; 16];
            for i in 0..16 {
                output_0[i] = x86_64::_mm256_unpacklo_epi32(vector_input[i], zeros);
                output_1[i] = x86_64::_mm256_unpackhi_epi32(vector_input[i], zeros);
            }
            [transmute(output_0), transmute(output_1)]
        }
    }

    /// Essentially inverts from_input
    #[inline]
    fn to_output(input: Self::InternalRep) -> [Self::PF; 16] {
        unsafe {
            // Safety: Each __m256i must be made up of 4 values lying in [0, ... P).
            // Otherwise the result is undefined.

            let zeros = x86_64::_mm256_setzero_si256();
            let vector_input: [[__m256i; 16]; 2] = transmute(input);
            let mut output = [zeros; 16];

            for (i, item) in output.iter_mut().enumerate() {
                *item = transmute(x86_64::_mm256_shuffle_ps::<136>(
                    transmute(vector_input[0][i]),
                    transmute(vector_input[1][i]),
                ));
            }

            transmute(output)
        }
    }

    #[inline]
    fn manipulate_external_constants(input: [KoalaBear; 16]) -> Packed64bitM31Tensor<4> {
        unsafe { transmute(input.map(|x| [x.value as u64; 4])) }
    }

    #[inline]
    fn manipulate_internal_constants(input: KoalaBear) -> __m256i {
        unsafe { x86_64::_mm256_set1_epi64x(input.value as i64) }
    }

    /// Compute a single internal Poseidon2 round.
    /// Input must be < 2P < 2^32 but does not need to be canonical.
    /// Round Constant is assumed to be in canonical form.
    /// Output will be < 2P.
    #[inline]
    fn internal_round(state: &mut Packed64bitM31Tensor<4>, rc: __m256i) {
        unsafe {
            // We do two things simultaneously.
            // Take the first value, add rc and compute the s-box.
            // Do a matrix multiplication on the remaining elements.
            // We will then move the first element back in later.

            const ZEROS: __m256i = unsafe { transmute::<[u64; 4], _>([0; 4]) };

            let s0_post_sbox = Self::internal_rc_sbox(state.0[0][0], rc); // Need to do something different to the first element.

            state.0[0][0] = ZEROS;

            // Can do part of the sum vertically.
            let vec_sum = state.vec_sum();
            // still need to do the horizontal part of the sum but this can wait until later.

            // Doing the diagonal multiplication.
            state.left_shift(Self::INTERNAL_SHIFTS);

            // Need to multiply s0_post_sbox by -2. We are working with negatives so can do the obvious thing.
            let neg_s0 = x86_64::_mm256_sub_epi64(ZEROS, s0_post_sbox);
            let neg_2_s0 = x86_64::_mm256_add_epi64(neg_s0, neg_s0);

            state.0[0][0] = neg_2_s0;

            let total_sum = x86_64::_mm256_add_epi64(vec_sum, s0_post_sbox);

            for mat in state.0.iter_mut() {
                mat[0] = x86_64::_mm256_add_epi64(mat[0], total_sum);
                mat[1] = x86_64::_mm256_add_epi64(mat[1], total_sum);
                mat[2] = x86_64::_mm256_add_epi64(mat[2], total_sum);
                mat[3] = x86_64::_mm256_add_epi64(mat[3], total_sum);
            }

            Self::monty_reduce(state); // Output, non canonical in [-P, P].
        }
    }

    /// A single External Round.
    /// Note that we change the order to be mat_mul -> RC -> S-box (instead of RC -> S-box -> mat_mul in the paper).
    /// Input must be in [0, P).
    /// Output will be in [0, P).
    #[inline]
    fn rotated_external_round(
        state: &mut Packed64bitM31Tensor<4>,
        round_constant: &Packed64bitM31Tensor<4>,
    ) {
        // Assume input state < P < 2^31

        state.mat_mul_aes(); // state < 7 * 2^31
        state.right_mat_mul_i_plus_1(); // state < 35 * 2^31
        state.add(round_constant); // state < 36 * 2^31 < 64 * 2^31
        Self::partial_reduce(state); // We need to ensure state < 64 * 2^31 to use this. Output is < 2P
        Self::joint_sbox(state); // state is now in [-P, P).
        Self::final_reduce_signed(state) // state is now in [0, P).
    }

    /// The initial set of external rounds. This consists of rf/2 external rounds followed by a mat_mul
    /// Inputs will always be canonical though in principal only need to be < 2^56.
    /// Output will be < 2P.
    #[inline]
    fn initial_external_rounds(
        state: &mut Packed64bitM31Tensor<4>,
        round_constants: &[Packed64bitM31Tensor<4>],
    ) {
        for round_constant in round_constants {
            Self::rotated_external_round(state, round_constant)
        }

        state.mat_mul_aes();
        state.right_mat_mul_i_plus_1();
        Self::partial_reduce(state);
        Self::final_reduce_pos(state); // Input to internal_rounds needs to be [-P, P].
    }

    /// The initial set of external rounds. This consists of rf/2 external rounds followed by a mat_mul
    /// Inputs should be in [-P, P].
    /// Output will be < P.
    #[inline]
    fn internal_rounds(state: &mut Packed64bitM31Tensor<4>, round_constants: &[__m256i]) {
        for round_constant in round_constants {
            Self::internal_round(state, *round_constant)
        }
        Self::final_reduce_pos(state); // Input to final_external_rounds needs to be in [0, P].
    }

    /// The final set of external rounds. Due to an ordering change it starts by doing a "half round" and finish by a mat_mul.
    /// Output is returned reduced.
    #[inline]
    fn final_external_rounds(
        state: &mut Packed64bitM31Tensor<4>,
        round_constants: &[Packed64bitM31Tensor<4>],
    ) {
        state.add(&round_constants[0]);
        Self::joint_sbox(state);
        Self::final_reduce_signed(state);

        for round_constant in round_constants[1..].iter() {
            Self::rotated_external_round(state, round_constant)
        }

        state.mat_mul_aes();
        state.right_mat_mul_i_plus_1();
        Self::partial_reduce(state);
        Self::final_reduce_pos(state);
    }
}

impl Poseidon2AVX2Methods<6, 24> for Poseidon2DataKoalaBearAVX2 {
    type PF = PackedKoalaBearAVX2;
    type InternalRep = [Packed64bitM31Tensor<6>; 2];

    const INTERNAL_SHIFTS: Packed64bitM31Tensor<6> = unsafe {
        transmute(expand_constant(
            POSEIDON2_INTERNAL_MATRIX_DIAG_24_MONTY_SHIFTS,
        ))
    };

    /// In memory, [PF; 24] = [[u32; 8]; 24] and we label the elements as:
    /// [[a_{0, 0}, ..., a_{0, 7}], ..., [a_{23, 0}, ..., a_{23, 7}]].
    /// We split each row in 2, expand each element to a u64 and then return vector of __mm256 elements arranged into a tensor.
    #[inline]
    fn from_input(input: [Self::PF; 24]) -> Self::InternalRep {
        unsafe {
            // Safety: Nothing unsafe to worry about.

            let zeros = x86_64::_mm256_setzero_si256();
            let vector_input: [__m256i; 24] = transmute(input);
            let mut output_0 = [zeros; 24];
            let mut output_1 = [zeros; 24];
            for i in 0..24 {
                output_0[i] = x86_64::_mm256_unpacklo_epi32(vector_input[i], zeros);
                output_1[i] = x86_64::_mm256_unpackhi_epi32(vector_input[i], zeros);
            }
            [transmute(output_0), transmute(output_1)]
        }
    }

    /// Essentially inverts from_input
    #[inline]
    fn to_output(input: Self::InternalRep) -> [Self::PF; 24] {
        unsafe {
            // Safety: Each __m256i must be made up of 4 values lying in [0, ... P).
            // Otherwise the result is undefined.

            let zeros = x86_64::_mm256_setzero_si256();
            let vector_input: [[__m256i; 24]; 2] = transmute(input);
            let mut output = [zeros; 24];

            for (i, item) in output.iter_mut().enumerate() {
                *item = transmute(x86_64::_mm256_shuffle_ps::<136>(
                    transmute(vector_input[0][i]),
                    transmute(vector_input[1][i]),
                ));
            }

            transmute(output)
        }
    }

    #[inline]
    fn manipulate_external_constants(input: [KoalaBear; 24]) -> Packed64bitM31Tensor<6> {
        unsafe { transmute(input.map(|x| [x.value as u64; 4])) }
    }

    #[inline]
    fn manipulate_internal_constants(input: KoalaBear) -> __m256i {
        unsafe { x86_64::_mm256_set1_epi64x(input.value as i64) }
    }

    /// Compute a single internal Poseidon2 round.
    /// Input must be < 2P < 2^32 but does not need to be canonical.
    /// Round Constant is assumed to be in canonical form.
    /// Output will be < 2P.
    #[inline]
    fn internal_round(state: &mut Packed64bitM31Tensor<6>, rc: __m256i) {
        unsafe {
            // We do two things simultaneously.
            // Take the first value, add rc and compute the s-box.
            // Do a matrix multiplication on the remaining elements.
            // We will then move the first element back in later.

            const ZEROS: __m256i = unsafe { transmute::<[u64; 4], _>([0; 4]) };

            let s0_post_sbox = Self::internal_rc_sbox(state.0[0][0], rc); // Need to do something different to the first element.

            state.0[0][0] = ZEROS;

            // Can do part of the sum vertically.
            let vec_sum = state.vec_sum();
            // still need to do the horizontal part of the sum but this can wait until later.

            // Doing the diagonal multiplication.
            state.left_shift(Self::INTERNAL_SHIFTS);

            // Need to multiply s0_post_sbox by -2. We are working with negatives so can do the obvious thing.
            let neg_s0 = x86_64::_mm256_sub_epi64(ZEROS, s0_post_sbox);
            state.0[0][0] = x86_64::_mm256_add_epi64(neg_s0, neg_s0);

            let total_sum = x86_64::_mm256_add_epi64(vec_sum, s0_post_sbox);

            for mat in state.0.iter_mut() {
                mat[0] = x86_64::_mm256_add_epi64(mat[0], total_sum);
                mat[1] = x86_64::_mm256_add_epi64(mat[1], total_sum);
                mat[2] = x86_64::_mm256_add_epi64(mat[2], total_sum);
                mat[3] = x86_64::_mm256_add_epi64(mat[3], total_sum);
            }

            Self::monty_reduce(state); // Output, non canonical in [0, 2P].
        }
    }

    /// A single External Round.
    /// Note that we change the order to be mat_mul -> RC -> S-box (instead of RC -> S-box -> mat_mul in the paper).
    /// Input must be in [0, P).
    /// Output will be in [0, P).
    #[inline]
    fn rotated_external_round(
        state: &mut Packed64bitM31Tensor<6>,
        round_constant: &Packed64bitM31Tensor<6>,
    ) {
        // Assume input state < P < 2^31

        state.mat_mul_aes(); // state < 7 * 2^31
        state.right_mat_mul_i_plus_1(); // state < 35 * 2^31
        state.add(round_constant); // state < 36 * 2^31 < 64 * 2^31
        Self::partial_reduce(state); // We need to ensure state < 64 * 2^31 to use this. Output is < 2P
        Self::joint_sbox(state); // state is now in [-P, P).
        Self::final_reduce_signed(state) // state is now in [0, P).
    }

    /// The initial set of external rounds. This consists of rf/2 external rounds followed by a mat_mul
    /// Input must be in [0, P).
    /// Output will be in [0, P).
    #[inline]
    fn initial_external_rounds(
        state: &mut Packed64bitM31Tensor<6>,
        round_constants: &[Packed64bitM31Tensor<6>],
    ) {
        for round_constant in round_constants {
            Self::rotated_external_round(state, round_constant)
        }

        state.mat_mul_aes();
        state.right_mat_mul_i_plus_1();
        Self::partial_reduce(state);
    }

    /// The initial set of external rounds. This consists of rf/2 external rounds followed by a mat_mul
    /// Inputs should be in [-P, P].
    /// Output will be < P.
    #[inline]
    fn internal_rounds(state: &mut Packed64bitM31Tensor<6>, round_constants: &[__m256i]) {
        for round_constant in round_constants {
            Self::internal_round(state, *round_constant)
        }
        Self::final_reduce_pos(state); // Input to final_external_rounds needs to be in [0, P].
    }

    /// The final set of external rounds. Due to an ordering change it starts by doing a "half round" and finish by a mat_mul.
    /// Output is returned reduced.
    #[inline]
    fn final_external_rounds(
        state: &mut Packed64bitM31Tensor<6>,
        round_constants: &[Packed64bitM31Tensor<6>],
    ) {
        state.add(&round_constants[0]);
        Self::joint_sbox(state);
        Self::final_reduce_signed(state);

        for round_constant in round_constants[1..].iter() {
            Self::rotated_external_round(state, round_constant)
        }

        state.mat_mul_aes();
        state.right_mat_mul_i_plus_1();
        Self::partial_reduce(state);
        Self::final_reduce_pos(state);
    }
}

#[cfg(test)]
mod tests {
    use core::mem::transmute;

    use p3_field::{AbstractField, PrimeField32};
    use p3_poseidon2::{Poseidon2, Poseidon2AVX2};
    use p3_symmetric::Permutation;
    use rand::{Rng, SeedableRng};
    use rand_xoshiro::Xoroshiro128Plus;

    use crate::{
        DiffusionMatrixKoalaBear, KoalaBear, MDSLightPermutationKoalaBear, PackedKoalaBearAVX2,
        Poseidon2DataKoalaBearAVX2,
    };

    type F = KoalaBear;
    const D: u64 = 3;
    type Perm16 = Poseidon2<F, MDSLightPermutationKoalaBear, DiffusionMatrixKoalaBear, 16, D>;
    type Perm24 = Poseidon2<F, MDSLightPermutationKoalaBear, DiffusionMatrixKoalaBear, 24, D>;

    // A very simple function which performs a transpose.
    fn transpose<F, const N: usize, const M: usize>(input: [[F; N]; M]) -> [[F; M]; N]
    where
        F: PrimeField32,
    {
        let mut output = [[F::zero(); M]; N];
        #[allow(clippy::needless_range_loop)]
        for i in 0..N {
            for j in 0..M {
                output[i][j] = input[j][i]
            }
        }
        output
    }

    /// Test that the output is the same as the scalar version on a random input.
    #[test]
    fn test_avx2_poseidon2_width_16() {
        let mut rng = rand::thread_rng();

        // Our Poseidon2 implementation.
        let poseidon2 = Perm16::new_from_rng_128(
            MDSLightPermutationKoalaBear,
            DiffusionMatrixKoalaBear,
            &mut rng,
        );

        let input: [F; 16] = rng.gen();

        let mut expected = input;
        poseidon2.permute_mut(&mut expected);

        let mut avx2_input = input.map(PackedKoalaBearAVX2::from_f);
        poseidon2.permute_mut(&mut avx2_input);

        let avx2_output = avx2_input.map(|x| x.0[0]);

        assert_eq!(avx2_output, expected);
    }

    /// Test that the output is the same as the scalar version on a random input.
    #[test]
    fn test_avx2_poseidon2_width_24() {
        let mut rng = rand::thread_rng();

        // Our Poseidon2 implementation.
        let poseidon2 = Perm24::new_from_rng_128(
            MDSLightPermutationKoalaBear,
            DiffusionMatrixKoalaBear,
            &mut rng,
        );

        let input: [F; 24] = rng.gen();

        let mut expected = input;
        poseidon2.permute_mut(&mut expected);

        let mut avx2_input = input.map(PackedKoalaBearAVX2::from_f);
        poseidon2.permute_mut(&mut avx2_input);

        let avx2_output = avx2_input.map(|x| x.0[0]);

        assert_eq!(avx2_output, expected);
    }

    /// Test that the scalar and vectorized outputs are the same on a random input of length 64.
    #[test]
    fn test_avx2_vectorized_poseidon2_4_x_width_16_2() {
        let mut rng = Xoroshiro128Plus::seed_from_u64(0x123456789);

        // Our Poseidon2 implementation.
        let poseidon2 = Perm16::new_from_rng_128(
            MDSLightPermutationKoalaBear,
            DiffusionMatrixKoalaBear,
            &mut rng,
        );

        let input: [[F; 16]; 8] = rng.gen();

        let mut expected = input;
        for row in expected.iter_mut() {
            poseidon2.permute_mut(row);
        }

        let input_transpose: [[F; 8]; 16] = transpose(input);
        let avx2_input: [PackedKoalaBearAVX2; 16] = unsafe { transmute(input_transpose) };

        let mut rng = Xoroshiro128Plus::seed_from_u64(0x123456789);

        let vector_poseidon_2: Poseidon2AVX2<4, 16, Poseidon2DataKoalaBearAVX2> =
            Poseidon2AVX2::new_from_rng_128::<Xoroshiro128Plus, 3>(&mut rng);

        let avx2_output = vector_poseidon_2.permute(avx2_input);

        let output_transpose: [[F; 8]; 16] = unsafe { transmute(avx2_output) };
        let output = transpose(output_transpose);

        assert_eq!(output, expected)
    }

    /// Test that the scalar and vectorized outputs are the same on a random input of length 96.
    #[test]
    fn test_avx2_vectorized_poseidon2_6_x_width_24_2() {
        let mut rng = Xoroshiro128Plus::seed_from_u64(0x123456789);

        // Our Poseidon2 implementation.
        let poseidon2 = Perm24::new_from_rng_128(
            MDSLightPermutationKoalaBear,
            DiffusionMatrixKoalaBear,
            &mut rng,
        );

        let input: [[F; 24]; 8] = rng.gen();

        let mut expected = input;
        for row in expected.iter_mut() {
            poseidon2.permute_mut(row);
        }

        let input_transpose: [[F; 8]; 24] = transpose(input);
        let avx2_input: [PackedKoalaBearAVX2; 24] = unsafe { transmute(input_transpose) };

        let mut rng = Xoroshiro128Plus::seed_from_u64(0x123456789);

        let vector_poseidon_2: Poseidon2AVX2<6, 24, Poseidon2DataKoalaBearAVX2> =
            Poseidon2AVX2::new_from_rng_128::<Xoroshiro128Plus, 3>(&mut rng);

        let avx2_output = vector_poseidon_2.permute(avx2_input);

        let output_transpose: [[F; 8]; 24] = unsafe { transmute(avx2_output) };
        let output = transpose(output_transpose);

        assert_eq!(output, expected)
    }
}
