use core::arch::x86_64::{self, __m256i};
use core::mem::transmute;

use p3_poseidon2::{matmul_internal, DiffusionPermutation};
use p3_symmetric::Permutation;

use crate::{
    DiffusionMatrixMersenne31, Mersenne31, PackedMersenne31AVX2, Poseidon2ExternalMatrixMersenne31,
    POSEIDON2_INTERNAL_MATRIX_DIAG_16, POSEIDON2_INTERNAL_MATRIX_DIAG_24,
};

impl Permutation<[PackedMersenne31AVX2; 16]> for DiffusionMatrixMersenne31 {
    fn permute_mut(&self, state: &mut [PackedMersenne31AVX2; 16]) {
        matmul_internal::<Mersenne31, PackedMersenne31AVX2, 16>(
            state,
            POSEIDON2_INTERNAL_MATRIX_DIAG_16,
        );
    }
}

impl DiffusionPermutation<PackedMersenne31AVX2, 16> for DiffusionMatrixMersenne31 {}

impl Permutation<[PackedMersenne31AVX2; 24]> for DiffusionMatrixMersenne31 {
    fn permute_mut(&self, state: &mut [PackedMersenne31AVX2; 24]) {
        matmul_internal::<Mersenne31, PackedMersenne31AVX2, 24>(
            state,
            POSEIDON2_INTERNAL_MATRIX_DIAG_24,
        );
    }
}

impl DiffusionPermutation<PackedMersenne31AVX2, 24> for DiffusionMatrixMersenne31 {}

const P: u32 = 0x7fffffff;
const P_4XU64: __m256i = unsafe { transmute::<[u64; 4], _>([0x7fffffff; 4]) };

const POSEIDON2_INTERNAL_MATRIX_DIAG_16_SHIFTS: [u64; 16] =
    [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16];

const POSEIDON2_INTERNAL_MATRIX_DIAG_24_SHIFTS: [u64; 24] = [
    0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
];

/// Do a single round of M31 reduction on each element.
/// No restrictions on input size
/// Output is < max(2^{-30} * input, 2^32)
#[inline]
fn reduce(x: __m256i) -> __m256i {
    unsafe {
        // Get the top 33 bits shifted down.
        let high_bits = x86_64::_mm256_srli_epi64::<31>(x);

        // Zero out the top 33 bits.
        let low_bits = x86_64::_mm256_and_si256(x, P_4XU64);

        // Add the high bits back to the value
        x86_64::_mm256_add_epi64(low_bits, high_bits)
    }
}

/// Do a full reduction on each element, returning something in [0, 2^31 - 1]
/// Input are assumed to be < 2^62.
/// Output will be in canonical form.
#[inline]
fn full_reduce(x: __m256i) -> __m256i {
    unsafe {
        // Safety: Inputs must be < 2^62.

        // First we reduce to something in [0, 2^32 - 2].
        // Then we subtract P. If that subtraction overflows our reduced value is correct.
        // Otherwise, the new subtracted value is right.
        let x_red = reduce(x);
        let x_red_sub_p = x86_64::_mm256_sub_epi32(x_red, P_4XU64);

        // Note its fine to the use u32 version here as the top 32 bits of x_red are always 0.
        x86_64::_mm256_min_epu32(x_red, x_red_sub_p)
    }
}

/// Compute x -> x^5 for each element of the vector.
/// The input must be < 2P < 2^32.
/// The output will be < 2^34.
#[inline]
fn joint_sbox(x: __m256i) -> __m256i {
    unsafe {
        // Safety: If input is < 2P < 2^32, no overflow will occur and the output will be < 2^34.

        // Subtract p to get something in [-P, P].
        // This is unnecessary if the input already lies in [0, P], and so 1 operation could be saved there.
        let x_sub_p = x86_64::_mm256_sub_epi32(x, P_4XU64);

        // Square x_sub_p. As |x| < P, x^2 < P^2 < 2^62
        let x2 = x86_64::_mm256_mul_epi32(x_sub_p, x_sub_p);

        // Reduce and then subtract P. The result will then lie in (-2^31, 2^31).
        let x2_red = reduce(x2);
        let x2_red_sub_p = x86_64::_mm256_sub_epi32(x2_red, P_4XU64);

        // Square again. The result is again < 2^62.
        let x4 = x86_64::_mm256_mul_epi32(x2_red_sub_p, x2_red_sub_p);

        // Reduce again so the result is < 2^32
        let x4_red = reduce(x4);

        // Now when we multiply our result is < 2^64
        let x5 = x86_64::_mm256_mul_epu32(x, x4_red);

        // Now we reduce again and return the result which is < 2^34.
        reduce(x5)

        // Currently this requires 13 operations. (Each reduce expands to 3 ops.) Can we do better?
        // The MUL ops all have high latency
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

const fn expand_interleave_constant<const N: usize, const M: usize>(
    input: [u64; N],
) -> [[[u64; 2]; 2]; M] {
    let mut output = [[[0; 2]; 2]; M];
    let mut acc = 0;
    loop {
        output[acc / 2] = [[input[acc], input[acc + 4]]; 2];
        output[acc / 2 + 1] = [[input[acc + 1], input[acc + 5]]; 2];
        output[acc / 2 + 2] = [[input[acc + 2], input[acc + 6]]; 2];
        output[acc / 2 + 3] = [[input[acc + 3], input[acc + 7]]; 2];
        acc += 8;
        if acc == N {
            break;
        }
    }
    output
}

use p3_poseidon2::{Packed64bitM31Tensor, Poseidon2AVX2Helpers, Poseidon2AVX2Methods};

#[derive(Clone)]
pub struct Poseidon2DataM31AVX2();

impl Poseidon2AVX2Helpers for Poseidon2DataM31AVX2 {
    /// Given a vector of elements __m256i apply a monty reduction to each u64.
    /// Each u64 input must lie in [0, 2^{32}P)
    /// Each output will be a u64 lying in [0, P)
    #[inline]
    fn full_reduce_vec(state: __m256i) -> __m256i {
        full_reduce(state)
    }

    /// Given a vector of elements __m256i apply a partial monty reduction to each u64
    /// Each u64 input must lie in [0, 2^{32}P)
    /// Each output will be a u64 lying in [0, 2P)
    /// Slightly cheaper than full_reduce
    #[inline]
    fn partial_reduce_vec(state: __m256i) -> __m256i {
        reduce(state)
    }

    /// Apply the s-box: x -> x^s for some small s coprime to p - 1 to a vector __m256i.
    /// Input must be 4 u64's all in the range [0, P).
    /// Output will be 4 u64's all in the range [0, 2^{32}P).
    #[inline]
    fn joint_sbox_vec(state: __m256i) -> __m256i {
        joint_sbox(state)
    }

    /// Apply the s-box: x -> (x + rc)^5 to a vector __m256i.
    /// s0 is in [0, 2P], rc is in [0, P]
    #[inline]
    fn quad_internal_sbox(s0: __m256i, rc: u32) -> __m256i {
        unsafe {
            // We set all u32 registers but we will ignore the top ones later.
            // Seems to make compiler "slightly?" happier than using set1_epi64x.
            let constant = x86_64::_mm256_set1_epi32(rc as i32);

            // Need to get s0 into canonical form.
            let sub = x86_64::_mm256_sub_epi32(s0, P_4XU64);
            let red_s0 = x86_64::_mm256_min_epu32(s0, sub);

            // Each entry of sum is <= 2P.
            let sum = x86_64::_mm256_add_epi32(red_s0, constant);

            joint_sbox(sum)
        }
    }

    /// Apply the s-box: x -> (x + rc)^s to a vector __m256i where we only care about the first 2 u64's
    #[inline]
    fn double_internal_sbox(s0: __m256i, rc: u32) -> __m256i {
        unsafe {
            // For now we do something identical to quad_internal_sbox.
            // Long term there might be a better way.

            // This could/should be replaced by a blend.
            let full_broadcast = x86_64::_mm256_set1_epi32(rc as i32);
            let zeros = x86_64::_mm256_setzero_si256();
            let constant = x86_64::_mm256_blend_epi32::<0b01000100>(full_broadcast, zeros);

            // Need to get s0 into canonical form.
            let sub = x86_64::_mm256_sub_epi32(s0, P_4XU64);
            let red_s0 = x86_64::_mm256_min_epu32(s0, sub);

            // Each entry of sum is < 2P.
            let sum = x86_64::_mm256_add_epi64(red_s0, constant);

            joint_sbox(sum)
        }
    }

    const PACKED_8XPRIME: __m256i = unsafe { transmute([(P as u64) << 3; 4]) };
}

impl Poseidon2AVX2Methods<2, 16> for Poseidon2DataM31AVX2 {
    type PF = PackedMersenne31AVX2;
    type InternalRep = [Packed64bitM31Tensor<2>; 4];

    const INTERNAL_SHIFTS: Packed64bitM31Tensor<2> = unsafe {
        transmute(expand_interleave_constant::<16, 8>(
            POSEIDON2_INTERNAL_MATRIX_DIAG_16_SHIFTS,
        ))
    };

    /// In memory, [PF; 16] = [[u32; 8]; 16] and we label the elements as:
    /// [[a_{0, 0}, ..., a_{0, 7}], ..., [a_{15, 0}, ..., a_{15, 7}]].
    /// We split each row in 4, expand each element to a u64 and then return vector of __mm256 elements arranged into a tensor.
    fn from_input(input: [Self::PF; 16]) -> Self::InternalRep {
        unsafe {
            // Safety: Nothing unsafe to worry about.

            let zeros = x86_64::_mm256_setzero_si256();
            let vector_input: [__m256i; 16] = transmute(input);
            let mut output = [[[zeros; 4]; 2]; 4];
            for j in 0..2 {
                for i in 0..4 {
                    let splicedlhs = x86_64::_mm256_unpacklo_epi32(
                        vector_input[8 * j + i],
                        vector_input[8 * j + i + 4],
                    );
                    let splicedrhs = x86_64::_mm256_unpackhi_epi32(
                        vector_input[8 * j + i],
                        vector_input[8 * j + i + 4],
                    );
                    output[0][j][i] = x86_64::_mm256_unpacklo_epi32(splicedlhs, zeros);
                    output[1][j][i] = x86_64::_mm256_unpackhi_epi32(splicedlhs, zeros);
                    output[2][j][i] = x86_64::_mm256_unpacklo_epi32(splicedrhs, zeros);
                    output[3][j][i] = x86_64::_mm256_unpackhi_epi32(splicedrhs, zeros);
                }
            }

            transmute(output)
        }
    }

    /// Essentially inverts from_input
    fn to_output(input: Self::InternalRep) -> [Self::PF; 16] {
        unsafe {
            // Safety: Each __m256i must be made up of 4 values lying in [0, ... P).
            // Otherwise the result is undefined.

            let zeros = x86_64::_mm256_setzero_si256();
            let mut output = [zeros; 16];
            let tensor_input: [[[__m256i; 4]; 2]; 4] = transmute(input);

            for j in 0..2 {
                for i in 0..4 {
                    let comblhs = x86_64::_mm256_shuffle_ps::<0b10001000>(
                        transmute(tensor_input[0][j][i]),
                        transmute(tensor_input[1][j][i]),
                    );
                    let combrhs = x86_64::_mm256_shuffle_ps::<0b10001000>(
                        transmute(tensor_input[2][j][i]),
                        transmute(tensor_input[3][j][i]),
                    );
                    output[8 * j + i] =
                        transmute(x86_64::_mm256_shuffle_ps::<0b10001000>(comblhs, combrhs));
                    output[8 * j + i + 4] =
                        transmute(x86_64::_mm256_shuffle_ps::<0b11011101>(comblhs, combrhs));
                }
            }

            transmute(output)
        }
    }

    fn manipulate_external_constants(input: [Mersenne31; 16]) -> Packed64bitM31Tensor<2> {
        unsafe {
            let mut output = [[[0_u64; 4]; 4]; 2];
            for j in 0..2 {
                for i in 0..4 {
                    let firstval = input[8 * j + i].value as u64;
                    let secondval = input[8 * j + i + 4].value as u64;

                    output[j][i] = [firstval, secondval, firstval, secondval]
                }
            }

            transmute(output)
        }
    }

    fn manipulate_internal_constants(input: Mersenne31) -> u32 {
        input.value
    }
}

impl Poseidon2AVX2Methods<3, 24> for Poseidon2DataM31AVX2 {
    type PF = PackedMersenne31AVX2;
    type InternalRep = [Packed64bitM31Tensor<3>; 4];

    const INTERNAL_SHIFTS: Packed64bitM31Tensor<3> = unsafe {
        transmute(expand_interleave_constant::<24, 12>(
            POSEIDON2_INTERNAL_MATRIX_DIAG_24_SHIFTS,
        ))
    };

    /// In memory, [PF; 24] = [[u32; 8]; 24] and we label the elements as:
    /// [[a_{0, 0}, ..., a_{0, 7}], ..., [a_{23, 0}, ..., a_{23, 7}]].
    /// We split each row in 4, expand each element to a u64 and then return vector of __mm256 elements arranged into a tensor.
    fn from_input(input: [Self::PF; 24]) -> Self::InternalRep {
        unsafe {
            // Safety: Nothing unsafe to worry about.

            let zeros = x86_64::_mm256_setzero_si256();
            let vector_input: [__m256i; 24] = transmute(input);
            let mut output = [[[zeros; 4]; 3]; 4];
            for j in 0..3 {
                for i in 0..4 {
                    let splicedlhs = x86_64::_mm256_unpacklo_epi32(
                        vector_input[8 * j + i],
                        vector_input[8 * j + i + 4],
                    );
                    let splicedrhs = x86_64::_mm256_unpackhi_epi32(
                        vector_input[8 * j + i],
                        vector_input[8 * j + i + 4],
                    );
                    output[0][j][i] = x86_64::_mm256_unpacklo_epi32(splicedlhs, zeros);
                    output[1][j][i] = x86_64::_mm256_unpackhi_epi32(splicedlhs, zeros);
                    output[2][j][i] = x86_64::_mm256_unpacklo_epi32(splicedrhs, zeros);
                    output[3][j][i] = x86_64::_mm256_unpackhi_epi32(splicedrhs, zeros);
                }
            }

            transmute(output)
        }
    }

    /// Essentially inverts from_input
    fn to_output(input: Self::InternalRep) -> [Self::PF; 24] {
        unsafe {
            // Safety: Each __m256i must be made up of 4 values lying in [0, ... P).
            // Otherwise the result is undefined.

            let zeros = x86_64::_mm256_setzero_si256();
            let mut output = [zeros; 24];
            let tensor_input: [[[__m256i; 4]; 3]; 4] = transmute(input);

            for j in 0..3 {
                for i in 0..4 {
                    let comblhs = x86_64::_mm256_shuffle_ps::<0b10001000>(
                        transmute(tensor_input[0][j][i]),
                        transmute(tensor_input[1][j][i]),
                    );
                    let combrhs = x86_64::_mm256_shuffle_ps::<0b10001000>(
                        transmute(tensor_input[2][j][i]),
                        transmute(tensor_input[3][j][i]),
                    );
                    output[8 * j + i] =
                        transmute(x86_64::_mm256_shuffle_ps::<0b10001000>(comblhs, combrhs));
                    output[8 * j + i + 4] =
                        transmute(x86_64::_mm256_shuffle_ps::<0b11011101>(comblhs, combrhs));
                }
            }

            transmute(output)
        }
    }

    fn manipulate_external_constants(input: [Mersenne31; 24]) -> Packed64bitM31Tensor<3> {
        unsafe {
            let mut output = [[[0_u64; 4]; 4]; 3];
            for j in 0..3 {
                for i in 0..4 {
                    let firstval = input[8 * j + i].value as u64;
                    let secondval = input[8 * j + i + 4].value as u64;

                    output[j][i] = [firstval, secondval, firstval, secondval]
                }
            }

            transmute(output)
        }
    }

    fn manipulate_internal_constants(input: Mersenne31) -> u32 {
        input.value
    }
}

impl Poseidon2AVX2Methods<4, 16> for Poseidon2DataM31AVX2 {
    type PF = PackedMersenne31AVX2;
    type InternalRep = [Packed64bitM31Tensor<4>; 2];

    const INTERNAL_SHIFTS: Packed64bitM31Tensor<4> =
        unsafe { transmute(expand_constant(POSEIDON2_INTERNAL_MATRIX_DIAG_16_SHIFTS)) };

    /// In memory, [PF; 16] = [[u32; 8]; 16] and we label the elements as:
    /// [[a_{0, 0}, ..., a_{0, 7}], ..., [a_{15, 0}, ..., a_{15, 7}]].
    /// We split each row in 2, expand each element to a u64 and then return vector of __mm256 elements arranged into a tensor.
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
    fn to_output(input: Self::InternalRep) -> [Self::PF; 16] {
        unsafe {
            // Safety: Each __m256i must be made up of 4 values lying in [0, ... P).
            // Otherwise the result is undefined.

            let zeros = x86_64::_mm256_setzero_si256();
            let vector_input: [[__m256i; 16]; 2] = transmute(input);
            let mut output = [zeros; 16];

            for i in 0..16 {
                output[i] = transmute(x86_64::_mm256_shuffle_ps::<136>(
                    transmute(vector_input[0][i]),
                    transmute(vector_input[1][i]),
                ));
            }

            transmute(output)
        }
    }

    fn manipulate_external_constants(input: [Mersenne31; 16]) -> Packed64bitM31Tensor<4> {
        unsafe { transmute(input.map(|x| [x.value as u64; 4])) }
    }

    fn manipulate_internal_constants(input: Mersenne31) -> u32 {
        input.value
    }
}

impl Poseidon2AVX2Methods<6, 24> for Poseidon2DataM31AVX2 {
    type PF = PackedMersenne31AVX2;
    type InternalRep = [Packed64bitM31Tensor<6>; 2];

    const INTERNAL_SHIFTS: Packed64bitM31Tensor<6> =
        unsafe { transmute(expand_constant(POSEIDON2_INTERNAL_MATRIX_DIAG_24_SHIFTS)) };

    /// In memory, [PF; 24] = [[u32; 8]; 24] and we label the elements as:
    /// [[a_{0, 0}, ..., a_{0, 7}], ..., [a_{23, 0}, ..., a_{23, 7}]].
    /// We split each row in 2, expand each element to a u64 and then return vector of __mm256 elements arranged into a tensor.
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
    fn to_output(input: Self::InternalRep) -> [Self::PF; 24] {
        unsafe {
            // Safety: Each __m256i must be made up of 4 values lying in [0, ... P).
            // Otherwise the result is undefined.

            let zeros = x86_64::_mm256_setzero_si256();
            let vector_input: [[__m256i; 24]; 2] = transmute(input);
            let mut output = [zeros; 24];

            for i in 0..24 {
                output[i] = transmute(x86_64::_mm256_shuffle_ps::<136>(
                    transmute(vector_input[0][i]),
                    transmute(vector_input[1][i]),
                ));
            }

            transmute(output)
        }
    }

    fn manipulate_external_constants(input: [Mersenne31; 24]) -> Packed64bitM31Tensor<6> {
        unsafe { transmute(input.map(|x| [x.value as u64; 4])) }
    }

    fn manipulate_internal_constants(input: Mersenne31) -> u32 {
        input.value
    }
}

#[cfg(test)]
mod tests {
    use core::mem::transmute;

    use p3_field::{AbstractField, PrimeField32};
    use p3_poseidon2::{Poseidon2, Poseidon2AVX2, Poseidon2ExternalMatrixGeneral};
    use p3_symmetric::Permutation;
    use rand::{Rng, SeedableRng};
    use rand_xoshiro::Xoroshiro128Plus;

    use crate::{
        DiffusionMatrixMersenne31, Mersenne31, PackedMersenne31AVX2, Poseidon2DataM31AVX2,
    };

    type F = Mersenne31;
    const D: u64 = 5;
    type Perm16 = Poseidon2<F, Poseidon2ExternalMatrixGeneral, DiffusionMatrixMersenne31, 16, D>;
    type Perm24 = Poseidon2<F, Poseidon2ExternalMatrixGeneral, DiffusionMatrixMersenne31, 24, D>;

    // A very simple function which performs a transpose.
    fn transpose<F, const N: usize, const M: usize>(input: [[F; N]; M]) -> [[F; M]; N]
    where
        F: PrimeField32,
    {
        let mut output = [[F::zero(); M]; N];
        for i in 0..N {
            for j in 0..M {
                output[i][j] = input[j][i]
            }
        }
        output
    }

    /// Test that the output is the same as the scalar version on a random input of length 16.
    #[test]
    fn test_avx2_poseidon2_width_16() {
        let mut rng = rand::thread_rng();

        // Our Poseidon2 implementation.
        let poseidon2 = Perm16::new_from_rng_128(
            Poseidon2ExternalMatrixGeneral,
            DiffusionMatrixMersenne31,
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
            Poseidon2ExternalMatrixGeneral,
            DiffusionMatrixMersenne31,
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

    /// Test that the scalar and vectorized outputs are the same on a random input of length 32.
    #[test]
    fn test_avx2_vectorized_poseidon2_2_x_width_16_2() {
        let mut rng = Xoroshiro128Plus::seed_from_u64(0x123456789);

        // Our Poseidon2 implementation.
        let poseidon2 = Perm16::new_from_rng_128(
            Poseidon2ExternalMatrixGeneral,
            DiffusionMatrixMersenne31,
            &mut rng,
        );

        let input: [[F; 16]; 8] = rng.gen();

        let mut expected = input;
        for row in expected.iter_mut() {
            poseidon2.permute_mut(row);
        }

        let input_transpose: [[F; 8]; 16] = transpose(input);
        let avx2_input: [PackedMersenne31AVX2; 16] = unsafe { transmute(input_transpose) };

        let mut rng = Xoroshiro128Plus::seed_from_u64(0x123456789);

        let vector_poseidon_2: Poseidon2AVX2<2, 16, Poseidon2DataM31AVX2> =
            Poseidon2AVX2::new_from_rng_128::<Xoroshiro128Plus, 5>(&mut rng);

        let avx2_output = vector_poseidon_2.permute(avx2_input);

        let output_transpose: [[F; 8]; 16] = unsafe { transmute(avx2_output) };
        let output = transpose(output_transpose);

        assert_eq!(output, expected)
    }

    /// Test that the scalar and vectorized outputs are the same on a random input of length 48.
    #[test]
    fn test_avx2_vectorized_poseidon2_2_x_width_24() {
        let mut rng = Xoroshiro128Plus::seed_from_u64(0x123456789);

        // Our Poseidon2 implementation.
        let poseidon2 = Perm24::new_from_rng_128(
            Poseidon2ExternalMatrixGeneral,
            DiffusionMatrixMersenne31,
            &mut rng,
        );

        let input: [[F; 24]; 8] = rng.gen();

        let mut expected = input;
        for row in expected.iter_mut() {
            poseidon2.permute_mut(row);
        }

        let input_transpose: [[F; 8]; 24] = transpose(input);
        let avx2_input: [PackedMersenne31AVX2; 24] = unsafe { transmute(input_transpose) };

        let mut rng = Xoroshiro128Plus::seed_from_u64(0x123456789);

        let vector_poseidon_2: Poseidon2AVX2<3, 24, Poseidon2DataM31AVX2> =
            Poseidon2AVX2::new_from_rng_128::<Xoroshiro128Plus, 5>(&mut rng);

        let avx2_output = vector_poseidon_2.permute(avx2_input);

        let output_transpose: [[F; 8]; 24] = unsafe { transmute(avx2_output) };
        let output = transpose(output_transpose);

        assert_eq!(output, expected)
    }

    /// Test that the scalar and vectorized outputs are the same on a random input of length 64.
    #[test]
    fn test_avx2_vectorized_poseidon2_4_x_width_16_2() {
        let mut rng = Xoroshiro128Plus::seed_from_u64(0x123456789);

        // Our Poseidon2 implementation.
        let poseidon2 = Perm16::new_from_rng_128(
            Poseidon2ExternalMatrixGeneral,
            DiffusionMatrixMersenne31,
            &mut rng,
        );

        let input: [[F; 16]; 8] = rng.gen();

        let mut expected = input;
        for row in expected.iter_mut() {
            poseidon2.permute_mut(row);
        }

        let input_transpose: [[F; 8]; 16] = transpose(input);
        let avx2_input: [PackedMersenne31AVX2; 16] = unsafe { transmute(input_transpose) };

        let mut rng = Xoroshiro128Plus::seed_from_u64(0x123456789);

        let vector_poseidon_2: Poseidon2AVX2<4, 16, Poseidon2DataM31AVX2> =
            Poseidon2AVX2::new_from_rng_128::<Xoroshiro128Plus, 5>(&mut rng);

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
            Poseidon2ExternalMatrixGeneral,
            DiffusionMatrixMersenne31,
            &mut rng,
        );

        let input: [[F; 24]; 8] = rng.gen();

        let mut expected = input;
        for row in expected.iter_mut() {
            poseidon2.permute_mut(row);
        }

        let input_transpose: [[F; 8]; 24] = transpose(input);
        let avx2_input: [PackedMersenne31AVX2; 24] = unsafe { transmute(input_transpose) };

        let mut rng = Xoroshiro128Plus::seed_from_u64(0x123456789);

        let vector_poseidon_2: Poseidon2AVX2<6, 24, Poseidon2DataM31AVX2> =
            Poseidon2AVX2::new_from_rng_128::<Xoroshiro128Plus, 5>(&mut rng);

        let avx2_output = vector_poseidon_2.permute(avx2_input);

        let output_transpose: [[F; 8]; 24] = unsafe { transmute(avx2_output) };
        let output = transpose(output_transpose);

        assert_eq!(output, expected)
    }
}
