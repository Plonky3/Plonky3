use core::arch::x86_64::{self, __m256i};
use core::mem::transmute;
use p3_poseidon2::{matmul_internal, DiffusionPermutation};
use p3_symmetric::Permutation;

use crate::{
    DiffusionMatrixMersenne31, Mersenne31, PackedMersenne31AVX2, POSEIDON2_INTERNAL_MATRIX_DIAG_16,
    POSEIDON2_INTERNAL_MATRIX_DIAG_24,
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

/// A 4x4 Matrix of M31 elements with each element stored in 64-bits and each row saved as 256bit packed vector.
#[derive(Clone, Copy, Debug)]
pub struct Packed64bitM31Matrix(__m256i, __m256i, __m256i, __m256i);

/// Compute the transpose of the given matrix.
fn _transpose(x: Packed64bitM31Matrix) -> Packed64bitM31Matrix {
    unsafe {
        // Safety: If this code got compiled then AVX2 intrinsics are available.
        let i0 = x86_64::_mm256_unpacklo_epi64(x.0, x.1);
        let i1 = x86_64::_mm256_unpackhi_epi64(x.0, x.1);
        let i2 = x86_64::_mm256_unpacklo_epi64(x.2, x.3);
        let i3 = x86_64::_mm256_unpackhi_epi64(x.2, x.3);

        let y0 = x86_64::_mm256_permute2x128_si256::<0x20>(i0, i2);
        let y1 = x86_64::_mm256_permute2x128_si256::<0x20>(i1, i3);
        let y2 = x86_64::_mm256_permute2x128_si256::<0x31>(i0, i2);
        let y3 = x86_64::_mm256_permute2x128_si256::<0x31>(i1, i3);
        Packed64bitM31Matrix(y0, y1, y2, y3)
    }
}

/// Left Multiply by the AES matrix:
/// [ 2 3 1 1 ]
/// [ 1 2 3 1 ]
/// [ 1 1 2 3 ]
/// [ 3 1 1 2 ].
fn _mat_mul_aes(x: Packed64bitM31Matrix) -> Packed64bitM31Matrix {
    unsafe {
        // Safety: If the inputs are <= L, the outputs are <= 7L.
        // Hence if L < 2^61, overflow will not occur.
        let t01 = x86_64::_mm256_add_epi64(x.0, x.1);
        let t23 = x86_64::_mm256_add_epi64(x.2, x.3);
        let t0123 = x86_64::_mm256_add_epi64(t01, t23);
        let t01123 = x86_64::_mm256_add_epi64(t0123, x.1);
        let t01233 = x86_64::_mm256_add_epi64(t0123, x.3);

        let t00 = x86_64::_mm256_slli_epi64::<1>(x.0);
        let t22 = x86_64::_mm256_slli_epi64::<1>(x.2);

        let y0 = x86_64::_mm256_add_epi64(t01, t01123);
        let y1 = x86_64::_mm256_add_epi64(t22, t01123);
        let y2 = x86_64::_mm256_add_epi64(t23, t01233);
        let y3 = x86_64::_mm256_add_epi64(t00, t01233);
        Packed64bitM31Matrix(y0, y1, y2, y3)
    }
}

/// Left Multiply by the matrix I + 1:
/// [ 2 1 1 1 ]
/// [ 1 2 1 1 ]
/// [ 1 1 2 1 ]
/// [ 1 1 1 2 ].
fn _mat_mul_i_plus_1(x: Packed64bitM31Matrix) -> Packed64bitM31Matrix {
    unsafe {
        // Safety: If the inputs are <= L, the outputs are <= 5L.
        // Hence if L < 2^61, overflow will not occur.
        let t01 = x86_64::_mm256_add_epi64(x.0, x.1);
        let t23 = x86_64::_mm256_add_epi64(x.2, x.3);
        let t0123 = x86_64::_mm256_add_epi64(t01, t23);

        let y0 = x86_64::_mm256_add_epi64(x.0, t0123);
        let y1 = x86_64::_mm256_add_epi64(x.1, t0123);
        let y2 = x86_64::_mm256_add_epi64(x.2, t0123);
        let y3 = x86_64::_mm256_add_epi64(x.3, t0123);
        Packed64bitM31Matrix(y0, y1, y2, y3)
    }
}

/// Do a single round of M31 reduction on each element.
fn _reduce(x: __m256i) -> __m256i {
    unsafe {
        // Safety: If inputs are < L, output is < max(2^{-30}L, 2^32)

        // Get the high 31 bits shifted down
        let high_bits_lo = x86_64::_mm256_srli_epi64::<31>(x);

        // Add the high bits back to the value
        let input_plus_high_lo = x86_64::_mm256_add_epi64(x, high_bits_lo);

        // Clear the bottom 31 bits of the x's
        let high_bits_hi = x86_64::_mm256_srli_epi64::<31>(high_bits_lo);

        // subtract off the high bits
        x86_64::_mm256_sub_epi64(input_plus_high_lo, high_bits_hi)
    }
}

/// Computex x -> x^5 for each element of the vector.
/// The input must be in canoncial form.
/// The output will not be in canonical form.
fn _joint_sbox(x: __m256i) -> __m256i {
    unsafe {
        // Safety: If input is in canonical form, no overflow will occur and the output will be < 2^33.

        const P: __m256i = unsafe { transmute::<[u64; 4], _>([0x7fffffff; 4]) };

        // Square x. If it starts in canoncical form then x^2 < 2^62
        let x2 = x86_64::_mm256_mul_epu32(x, x);

        // Reduce and then subtract P. The result will then lie in (-2^31, 2^31).
        let x2_red = _reduce(x2);
        let x2_red_sub_p = x86_64::_mm256_sub_epi64(x2_red, P);

        // Square again. The result is again < 2^62.
        let x4 = x86_64::_mm256_mul_epi32(x2_red_sub_p, x2_red_sub_p);

        // Reduce again so the result is < 2^32
        let x4_red = _reduce(x4);

        // Now when we multiply our result is < 2^63
        let x5 = x86_64::_mm256_mul_epu32(x, x4_red);

        // Now we reduce again and return the result which is < 2^33.
        _reduce(x5)
    }
}

/// External Poseidon Rounds

#[cfg(test)]
mod tests {
    use p3_field::AbstractField;
    use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
    use p3_symmetric::Permutation;
    use rand::Rng;

    use crate::{DiffusionMatrixMersenne31, Mersenne31, PackedMersenne31AVX2};

    type F = Mersenne31;
    const D: u64 = 5;
    type Perm16 = Poseidon2<F, Poseidon2ExternalMatrixGeneral, DiffusionMatrixMersenne31, 16, D>;
    type Perm24 = Poseidon2<F, Poseidon2ExternalMatrixGeneral, DiffusionMatrixMersenne31, 24, D>;

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
}
