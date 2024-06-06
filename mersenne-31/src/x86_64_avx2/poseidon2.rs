use alloc::vec::Vec;
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

const P: u32 = 0x7fffffff;
const PSQ: i64 = (P as i64) * (P as i64);
const P_4XU64: __m256i = unsafe { transmute::<[u64; 4], _>([0x7fffffff; 4]) };
const MASK: __m256i = unsafe {
    transmute::<[u64; 4], _>([
        0,
        0xffffffffffffffff,
        0xffffffffffffffff,
        0xffffffffffffffff,
    ])
};

const INTERNAL_SHIFTS0: __m256i = unsafe { transmute::<[u64; 4], _>([0, 0, 1, 2]) };
const INTERNAL_SHIFTS1: __m256i = unsafe { transmute::<[u64; 4], _>([3, 4, 5, 6]) };
const INTERNAL_SHIFTS2: __m256i = unsafe { transmute::<[u64; 4], _>([7, 8, 10, 12]) };
const INTERNAL_SHIFTS3: __m256i = unsafe { transmute::<[u64; 4], _>([13, 14, 15, 16]) };

#[derive(Clone, Debug)]
pub struct Poseidon2AVX2M31 {
    pub rounds_f: usize,
    pub external_round_constants: Vec<Packed64bitM31Matrix>,
    pub rounds_p: usize,
    pub internal_round_constants: Vec<u32>,
}

/// A 4x4 Matrix of M31 elements with each element stored in 64-bits and each row saved as 256bit packed vector.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Packed64bitM31Matrix([__m256i; 4]);

impl Packed64bitM31Matrix {
    /// Compute the transpose of the given matrix.
    fn transpose(&mut self) {
        unsafe {
            // Safety: If this code got compiled then AVX2 intrinsics are available.
            let i0 = x86_64::_mm256_unpacklo_epi64(self.0[0], self.0[1]);
            let i1 = x86_64::_mm256_unpackhi_epi64(self.0[0], self.0[1]);
            let i2 = x86_64::_mm256_unpacklo_epi64(self.0[2], self.0[3]);
            let i3 = x86_64::_mm256_unpackhi_epi64(self.0[2], self.0[3]);

            self.0[0] = x86_64::_mm256_permute2x128_si256::<0x20>(i0, i2);
            self.0[1] = x86_64::_mm256_permute2x128_si256::<0x20>(i1, i3);
            self.0[2] = x86_64::_mm256_permute2x128_si256::<0x31>(i0, i2);
            self.0[3] = x86_64::_mm256_permute2x128_si256::<0x31>(i1, i3);
        }
    }

    /// Left Multiply by the AES matrix:
    /// [ 2 3 1 1 ]
    /// [ 1 2 3 1 ]
    /// [ 1 1 2 3 ]
    /// [ 3 1 1 2 ].
    fn mat_mul_aes(&mut self) {
        unsafe {
            // Safety: If the inputs are <= L, the outputs are <= 7L.
            // Hence if L < 2^61, overflow will not occur.
            let t01 = x86_64::_mm256_add_epi64(self.0[0], self.0[1]);
            let t23 = x86_64::_mm256_add_epi64(self.0[2], self.0[3]);
            let t0123 = x86_64::_mm256_add_epi64(t01, t23);
            let t01123 = x86_64::_mm256_add_epi64(t0123, self.0[1]);
            let t01233 = x86_64::_mm256_add_epi64(t0123, self.0[3]);

            let t00 = x86_64::_mm256_slli_epi64::<1>(self.0[0]);
            let t22 = x86_64::_mm256_slli_epi64::<1>(self.0[2]);

            self.0[0] = x86_64::_mm256_add_epi64(t01, t01123);
            self.0[1] = x86_64::_mm256_add_epi64(t22, t01123);
            self.0[2] = x86_64::_mm256_add_epi64(t23, t01233);
            self.0[3] = x86_64::_mm256_add_epi64(t00, t01233);
        }
    }

    /// Can probably rework this to do it without the transpose.
    /// Left Multiply by the matrix I + 1:
    /// [ 2 1 1 1 ]
    /// [ 1 2 1 1 ]
    /// [ 1 1 2 1 ]
    /// [ 1 1 1 2 ].
    fn mat_mul_i_plus_1(&mut self) {
        unsafe {
            // Safety: If the inputs are <= L, the outputs are <= 5L.
            // Hence if L < 2^61, overflow will not occur.
            let t01 = x86_64::_mm256_add_epi64(self.0[0], self.0[1]);
            let t23 = x86_64::_mm256_add_epi64(self.0[2], self.0[3]);
            let t0123 = x86_64::_mm256_add_epi64(t01, t23);

            self.0[0] = x86_64::_mm256_add_epi64(self.0[0], t0123);
            self.0[1] = x86_64::_mm256_add_epi64(self.0[1], t0123);
            self.0[2] = x86_64::_mm256_add_epi64(self.0[2], t0123);
            self.0[3] = x86_64::_mm256_add_epi64(self.0[3], t0123);
        }
    }

    /// Do a single round of M31 reduction on each element.
    fn reduce(&mut self) {
        self.0[0] = reduce(self.0[0]);
        self.0[1] = reduce(self.0[1]);
        self.0[2] = reduce(self.0[2]);
        self.0[3] = reduce(self.0[3]);
    }

    /// Do a single round of M31 reduction on each element.
    fn full_reduce(&mut self) {
        self.0[0] = full_reduce(self.0[0]);
        self.0[1] = full_reduce(self.0[1]);
        self.0[2] = full_reduce(self.0[2]);
        self.0[3] = full_reduce(self.0[3]);
    }

    /// Computex x -> x^5 for each element of the vector.
    /// The input must be in canoncial form.
    fn joint_sbox(&mut self) {
        self.0[0] = joint_sbox(self.0[0]);
        self.0[1] = joint_sbox(self.0[1]);
        self.0[2] = joint_sbox(self.0[2]);
        self.0[3] = joint_sbox(self.0[3]);
    }

    /// Add in round constants.
    fn add_rc(&mut self, rc: Packed64bitM31Matrix) {
        unsafe {
            // Safety: element of rc must be in canoncial form.
            // Elements of self should be small enough such that overflow is impossible.
            self.0[0] = x86_64::_mm256_add_epi64(self.0[0], rc.0[0]);
            self.0[1] = x86_64::_mm256_add_epi64(self.0[1], rc.0[1]);
            self.0[2] = x86_64::_mm256_add_epi64(self.0[2], rc.0[2]);
            self.0[3] = x86_64::_mm256_add_epi64(self.0[3], rc.0[3]);
        }
    }

    /// Compute a single internal Poseidon2 round.
    /// Assume inputs are < 2^32 - 1, but may not be canonical.
    /// Assume the round constant is given in canonical form.
    fn internal_round(&mut self, rc: u32) {
        unsafe {
            // We do two things simultaneously.
            // Take the first value, add rc and compute the cube.
            // Do a matrix multiplication on the remaining elements.
            // We will then move the first element back in later.

            let s0 = { transmute::<_, [u64; 4]>(self.0[0]) }[0] as u32; // Pull out the first element.
            self.0[0] = x86_64::_mm256_and_si256(self.0[0], MASK); // 0 the first element in the matrix.

            let t01 = x86_64::_mm256_add_epi64(self.0[0], self.0[1]);
            let t23 = x86_64::_mm256_add_epi64(self.0[2], self.0[3]);
            let t0123 = x86_64::_mm256_add_epi64(t01, t23);

            // Now need to sum t0123 horizontally.
            let total: u64 = { transmute::<_, [u64; 4]>(t0123) }.into_iter().sum();
            // IMPROVE: Suspect this is suboptimal and can be improved.

            // Doing the diagonal multiplication.
            self.0[0] = x86_64::_mm256_sllv_epi64(self.0[0], INTERNAL_SHIFTS0);
            self.0[1] = x86_64::_mm256_sllv_epi64(self.0[1], INTERNAL_SHIFTS1);
            self.0[2] = x86_64::_mm256_sllv_epi64(self.0[2], INTERNAL_SHIFTS2);
            self.0[3] = x86_64::_mm256_sllv_epi64(self.0[3], INTERNAL_SHIFTS3);

            // Need to compute s0 -> (s0 + rc)^5
            let (sum, over) = s0.overflowing_add(rc); // s0 + rc < 2^33 - 3, over detects if its >= 2^32.
            let sum_corr = sum.wrapping_sub(P << 1) as i32; // If over, sum_corr is in [0, 2^31 - 1].
            let sum_sub = sum.wrapping_sub(P) as i32; // If not over, sum_sub is in [-2^31 + 1, 2^31 - 1].
                                                      ////////////////////
                                                      // BUG!! sum_sub could be exactly 2^31.
                                                      // Need to fix before production.
                                                      ////////////////////
            let val = if over { sum_corr } else { sum_sub };

            let sq = (val as i64) * (val as i64); // Always positive as its a square.
            let sq_red = ((sq as u32 & P) + ((sq >> 31) as u32)).wrapping_sub(P) as i32; // Redcing to an element in [-2^31 + 1, 2^31 - 1]

            let quad = (sq_red as i64) * (sq_red as i64); // Always positive as its a square.
            let quad_red = ((quad as u32 & P) + ((quad >> 31) as u32)).wrapping_sub(P) as i32; // Redcing to an element in [-2^31 + 1, 2^31 - 1]

            let fifth = (((quad_red as i64) * (val as i64)) + PSQ) as u64; // This lies in [0, 2^63] as 2P^2 < 2^63.
            let fifth_red =
                ((fifth as u32 & P) + ((fifth >> 31) as u32 & P) + ((fifth >> 62) as u32)) as u64; // Note fifth_red < 2^32 as P + P + 1 < 2^32.

            // Need to mutiply self00 by -2.
            // Easiest to do 4P - self00 to get the negative, then shift left by 1.
            // only involves shifts.
            let s00 = (((P as u64) << 2) - fifth_red) << 1;

            let self00 = { transmute::<[u64; 4], __m256i>([s00, 0, 0, 0]) };
            self.0[0] = x86_64::_mm256_add_epi64(self.0[0], self00);

            let full_total = total + fifth_red;
            let shift = x86_64::_mm256_set1_epi64x(full_total as i64);

            self.0[0] = x86_64::_mm256_add_epi64(self.0[0], shift);
            self.0[1] = x86_64::_mm256_add_epi64(self.0[1], shift);
            self.0[2] = x86_64::_mm256_add_epi64(self.0[2], shift);
            self.0[3] = x86_64::_mm256_add_epi64(self.0[3], shift);

            self.reduce() // Output, non canonical in [0, 2^32 - 2].
        }
    }
}

/// Do a single round of M31 reduction on each element.
fn reduce(x: __m256i) -> __m256i {
    unsafe {
        // Safety: If inputs are < L, output is < max(2^{-30}L, 2^32)

        // Get the high 31 bits shifted down
        let high_bits_lo = x86_64::_mm256_srli_epi64::<31>(x);

        // Add the high bits back to the value
        let input_plus_high_lo = x86_64::_mm256_add_epi64(x, high_bits_lo);

        // Clear the bottom 31 bits of the x's
        let high_bits_hi = x86_64::_mm256_slli_epi64::<31>(high_bits_lo);

        // subtract off the high bits
        x86_64::_mm256_sub_epi64(input_plus_high_lo, high_bits_hi)
    }
}

/// Do a full reduction on each element, returning something in [0, 2^31 - 1]
fn full_reduce(x: __m256i) -> __m256i {
    unsafe {
        // Safety: Inputs must be < 2^62.

        // First we reduce to something in [0, 2^32 - 2].
        // Then we subtract P. If that subtraction overflows our reduced value is correct.
        // Otherwise, the new subtracted value is right.
        let x_red = reduce(x);
        let x_red_sub_p = x86_64::_mm256_sub_epi64(x_red, P_4XU64);

        // Note its fine to the use u32 version here as the top 32 bits of x_red are always 0.
        x86_64::_mm256_min_epu32(x_red, x_red_sub_p)
    }
}

/// Computex x -> x^5 for each element of the vector.
/// The input must be in canoncial form.
/// The output will not be in canonical form.
fn joint_sbox(x: __m256i) -> __m256i {
    unsafe {
        // Safety: If input is in canonical form, no overflow will occur and the output will be < 2^33.

        // Square x. If it starts in canoncical form then x^2 < 2^62
        let x2 = x86_64::_mm256_mul_epu32(x, x);

        // Reduce and then subtract P. The result will then lie in (-2^31, 2^31).
        let x2_red = reduce(x2);
        let x2_red_sub_p = x86_64::_mm256_sub_epi64(x2_red, P_4XU64);

        // Square again. The result is again < 2^62.
        let x4 = x86_64::_mm256_mul_epi32(x2_red_sub_p, x2_red_sub_p);

        // Reduce again so the result is < 2^32
        let x4_red = reduce(x4);

        // Now when we multiply our result is < 2^63
        let x5 = x86_64::_mm256_mul_epu32(x, x4_red);

        // Now we reduce again and return the result which is < 2^33.
        reduce(x5)
    }
}

/// External Poseidon Rounds
impl Poseidon2AVX2M31 {
    /// A single External Round.
    /// Note that we change the order to be mat_mul -> RC -> S-box (instead of RC -> S-box -> mat_mul in the paper).
    /// Input does not need to be in canonical form, < 2^50 is fine.
    /// Output will be < 2^33.
    fn rotated_external_round(&self, state: &mut Packed64bitM31Matrix, index: usize) {
        let round_constant = self.external_round_constants[index];
        state.transpose();
        state.mat_mul_aes();
        state.transpose();
        state.mat_mul_i_plus_1();
        state.add_rc(round_constant);
        state.full_reduce();
        state.joint_sbox();
    }

    /// The poseidon2 permutation.
    pub fn poseidon2(&self, state: &mut Packed64bitM31Matrix) {
        // We start by doing rf/2 external rounds followed by a mat_mul.
        let half_rf = self.rounds_f / 2;
        for index in 0..half_rf {
            self.rotated_external_round(state, index)
        }

        state.transpose();
        state.mat_mul_aes();
        state.transpose();
        state.mat_mul_i_plus_1();
        state.full_reduce();

        // TODO: Internal Rounds
        for index in 0..self.rounds_p {
            state.internal_round(self.internal_round_constants[index])
        }

        // We finish by doing rf/2 external rounds again. Due to an ordering change we start by doing a "half round" and finish by a matmul.
        state.add_rc(self.external_round_constants[half_rf]);
        state.full_reduce(); // Can possibly do something cheaper than full reduce here?
        state.joint_sbox();

        for index in (1 + half_rf)..self.rounds_f {
            self.rotated_external_round(state, index)
        }

        state.transpose();
        state.mat_mul_aes();
        state.transpose();
        state.mat_mul_i_plus_1();
        state.full_reduce();
    }

    pub fn poseidon2_non_mut(&self, state: Packed64bitM31Matrix) -> Packed64bitM31Matrix {
        let mut output = state;
        Self::poseidon2(self, &mut output);
        output
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use core::mem::transmute;

    use p3_field::AbstractField;
    use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
    use p3_symmetric::Permutation;
    use rand::distributions::Standard;
    use rand::{Rng, SeedableRng};
    use rand_xoshiro::Xoroshiro128Plus;

    use crate::{
        DiffusionMatrixMersenne31, Mersenne31, Packed64bitM31Matrix, PackedMersenne31AVX2,
        Poseidon2AVX2M31,
    };

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

    /// Test that the scalar and vectorized outputs are the same on a random input of length 16.
    #[test]
    fn test_avx2_vectorized_poseidon2_width_16() {
        let mut rng = Xoroshiro128Plus::seed_from_u64(0x123456789);
        const ROUNDS_F: usize = 8;
        const ROUNDS_P: usize = 14;
        const WIDTH: usize = 16;

        // Our Poseidon2 implementation.
        let poseidon2 = Perm16::new_from_rng_128(
            Poseidon2ExternalMatrixGeneral,
            DiffusionMatrixMersenne31,
            &mut rng,
        );

        let input: [F; 16] = rng.gen();

        let mut expected = input;
        poseidon2.permute_mut(&mut expected);

        let mut avx2_input: Packed64bitM31Matrix =
            unsafe { transmute(input.map(|x| x.value as u64)) };
        let mut rng = Xoroshiro128Plus::seed_from_u64(0x123456789);

        let (external_constants, internal_constants) = unsafe {
            let external_consts_f = (&mut rng)
                .sample_iter(Standard)
                .take(ROUNDS_F)
                .collect::<Vec<[F; WIDTH]>>();
            let internal_constants_f = (&mut rng)
                .sample_iter(Standard)
                .take(ROUNDS_P)
                .collect::<Vec<F>>();
            let ex_con_avx2 = external_consts_f
                .into_iter()
                .map(|arr| transmute(arr.map(|x| x.value as u64)))
                .collect::<Vec<Packed64bitM31Matrix>>();
            let in_con_avx2 = internal_constants_f
                .into_iter()
                .map(|elem| elem.value)
                .collect::<Vec<u32>>();
            (ex_con_avx2, in_con_avx2)
        };

        let p2 = Poseidon2AVX2M31 {
            rounds_f: ROUNDS_F,
            external_round_constants: external_constants,
            rounds_p: ROUNDS_P,
            internal_round_constants: internal_constants,
        };

        p2.poseidon2(&mut avx2_input);
        avx2_input.full_reduce();

        let output: [F; 16] = unsafe {
            transmute::<_, [u64; 16]>(avx2_input).map(|elem| Mersenne31 { value: elem as u32 })
        };

        assert_eq!(output, expected)
    }
}
