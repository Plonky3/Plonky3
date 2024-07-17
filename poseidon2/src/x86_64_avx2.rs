use alloc::vec::Vec;
use core::arch::x86_64::{self, __m256i};
use core::marker::PhantomData;
use core::mem::transmute;
use p3_field::{PackedField, PrimeField32};
use p3_symmetric::{CryptographicPermutation, Permutation};
use rand::distributions::{Distribution, Standard};
use rand::Rng;

use crate::poseidon2_round_numbers_128;

// We need the proof output to be independent of the architecture hence it's important to consider what the input will look like.
// As the nature of the vectorization code is to perform horizontal parallelization, if the scalar code inputs [F; 16], the
// packed field code will input [PF; 16] = [[F; N]; 16] and we should perform the poseidon2 permutation on the columns of length 16.

// Internally, we represent our state as a tensor of size 4x4x1, 4x4x2, 4x4x3, 4x4x4 corresponding respectively to a single Poseidon-16 instance, 2 instances of Poseidon16, 2 instances of Poseidon24, or 4 instances of Poseidon 16.
// This may be reduced in future once we determine which versions are fastest.
// Currently the mapping from the standard [F; 16], [[F; 16]; 2], [[F; 24]; 2] to the internal 4xN matrix form looks like:
//
// 2 Poseidon 16 instance:  [x0, x4, y0, y4] [x8, x12, y8, y12]
//                          [x1, x5, y1, y5] [x9, x13, y9, y13]
//                          [x2, x6, y2, y6] [x10, x14, y10, y14]
//                          [x3, x7, y3, y7] [x11, x15, y11, y15]
//
// 2 Poseidon 24 instance:  [x0, x4, y0, y4] [x8, x12, y8, y12]   [x16, x20, y16, y20]
//                          [x1, x5, y1, y5] [x9, x13, y9, y13]   [x17, x22, y17, y21]
//                          [x2, x6, y2, y6] [x10, x14, y10, y14] [x18, x22, y18, y22]
//                          [x3, x7, y3, y7] [x11, x15, y11, y15] [x19, x23, y19, y23]
//
// 4 Poseidon 16 instance:  [w0, x0, y0, z0] [w4, x4, y4, z4] [w8, x8, y8, z8]     [w12, x12, y12, z12]
//                          [w1, x1, y1, z1] [w5, x5, y5, z5] [w9, x9, y9, z9]     [w13, x13, y13, z13]
//                          [w2, x2, y2, z2] [w6, x6, y6, z6] [w10, x10, y10, z10] [w14, x14, y14, z14]
//                          [w3, x3, y3, z3] [w7, x7, y7, z7] [w11, x11, y11, z11] [w15, x15, y15, z15]
//
// 4 Poseidon 24 instance:  [w0, x0, y0, z0] [w4, x4, y4, z4] [w8, x8, y8, z8]     [w12, x12, y12, z12] [w16, x16, y16, z16] [w20, x20, y20, z20]
//                          [w1, x1, y1, z1] [w5, x5, y5, z5] [w9, x9, y9, z9]     [w13, x13, y13, z13] [w17, x17, y17, z17] [w21, x21, y21, z21]
//                          [w2, x2, y2, z2] [w6, x6, y6, z6] [w10, x10, y10, z10] [w14, x14, y14, z14] [w18, x18, y18, z18] [w22, x22, y22, z22]
//                          [w3, x3, y3, z3] [w7, x7, y7, z7] [w11, x11, y11, z11] [w15, x15, y15, z15] [w19, x19, y19, z19] [w23, x23, y23, z23]
// This necessitates some data manipulation.

/// A 4x4xN Matrix of 31-bit field elements with each element stored in 64-bits and each row saved as (multiple) 256bit packed vectors.
/// Used for the internal representations for vectorized AVX2 implementations for Poseidon2
/// Should only be called with N = 1, 2, 3, 4
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Packed64bitM31Tensor<const HEIGHT: usize>([[__m256i; 4]; HEIGHT]);

impl<const HEIGHT: usize> Packed64bitM31Tensor<HEIGHT> {
    /// Left Multiply by the AES matrix:
    /// [ 2 3 1 1 ]
    /// [ 1 2 3 1 ]
    /// [ 1 1 2 3 ]
    /// [ 3 1 1 2 ].
    #[inline]
    fn mat_mul_aes(&mut self) {
        unsafe {
            // Safety: If the inputs are <= L, the outputs are <= 7L.
            // Hence if L < 2^61, overflow will not occur.
            for matrix in self.0.iter_mut() {
                let t01 = x86_64::_mm256_add_epi64(matrix[0], matrix[1]);
                let t23 = x86_64::_mm256_add_epi64(matrix[2], matrix[3]);
                let t0123 = x86_64::_mm256_add_epi64(t01, t23);
                let t01123 = x86_64::_mm256_add_epi64(t0123, matrix[1]);
                let t01233 = x86_64::_mm256_add_epi64(t0123, matrix[3]);

                let t00 = x86_64::_mm256_add_epi64(matrix[0], matrix[0]);
                let t22 = x86_64::_mm256_add_epi64(matrix[2], matrix[2]);

                matrix[0] = x86_64::_mm256_add_epi64(t01, t01123);
                matrix[1] = x86_64::_mm256_add_epi64(t22, t01123);
                matrix[2] = x86_64::_mm256_add_epi64(t23, t01233);
                matrix[3] = x86_64::_mm256_add_epi64(t00, t01233);
            }
        }
    }

    /// Apply the map x_i -> x_i + (x_{i%4} + x_{4 + i%4} + x_{8 + i%4} + ...).
    /// Writing the state as:
    ///                         [x0 x4 ...]
    ///                         [x1 x5 ...]
    ///                         [x2 x6 ...]
    ///                         [x3 x7 ...]
    /// We are performing a right multiplication by the matrix I + 1.
    #[inline]
    fn right_mat_mul_i_plus_1(&mut self) {
        // The code looks slightly different for different heights.
        match HEIGHT {
            2 => right_mat_mul_i_plus_1_dim_2(self),
            3 => right_mat_mul_i_plus_1_dim_3(self),
            4 => right_mat_mul_i_plus_1_dim_4(self),
            6 => right_mat_mul_i_plus_1_dim_6(self),
            _ => unreachable!(),
        };
    }

    /// Add together two tensors element wise.
    #[inline]
    fn add(&mut self, rhs: Self) {
        unsafe {
            // Safety: element of rhs must be in canonical form.
            // Elements of self should be small enough such that overflow is impossible.
            for i in 0..HEIGHT {
                self.0[i][0] = x86_64::_mm256_add_epi64(self.0[i][0], rhs.0[i][0]);
                self.0[i][1] = x86_64::_mm256_add_epi64(self.0[i][1], rhs.0[i][1]);
                self.0[i][2] = x86_64::_mm256_add_epi64(self.0[i][2], rhs.0[i][2]);
                self.0[i][3] = x86_64::_mm256_add_epi64(self.0[i][3], rhs.0[i][3]);
            }
        }
    }

    /// Compute the sum of all vectors.
    #[inline]
    fn vec_sum(&self) -> __m256i {
        unsafe {
            // Safety: Elements of self should be small enough such that overflow is impossible.
            // If all inputs are < L, then outputs are <= 4*N*L
            let mut output = x86_64::_mm256_setzero_si256();
            for mat in self.0 {
                let t01 = x86_64::_mm256_add_epi64(mat[0], mat[1]);
                let t23 = x86_64::_mm256_add_epi64(mat[2], mat[3]);
                let t0123 = x86_64::_mm256_add_epi64(t01, t23);
                output = x86_64::_mm256_add_epi64(t0123, output);
            }
            output
        }
    }

    /// Left shift each element in self by the corresponding shift in HEIGHT;
    #[inline]
    fn left_shift(&mut self, shifts: Packed64bitM31Tensor<HEIGHT>) {
        unsafe {
            // Safety: Elements of self, shifts should be small enough such that overflow is impossible.
            for i in 0..HEIGHT {
                self.0[i][0] = x86_64::_mm256_sllv_epi64(self.0[i][0], shifts.0[i][0]);
                self.0[i][1] = x86_64::_mm256_sllv_epi64(self.0[i][1], shifts.0[i][1]);
                self.0[i][2] = x86_64::_mm256_sllv_epi64(self.0[i][2], shifts.0[i][2]);
                self.0[i][3] = x86_64::_mm256_sllv_epi64(self.0[i][3], shifts.0[i][3]);
            }
        }
    }
}

/// Compute the transpose of a m 4x4 matrix.
/// Used to get data into the right form.
#[inline]
pub fn transpose(input: [__m256i; 4]) -> [__m256i; 4] {
    unsafe {
        // Safety: If this code got compiled then AVX2 intrinsics are available.
        let i0 = x86_64::_mm256_unpacklo_epi64(input[0], input[1]);
        let i1 = x86_64::_mm256_unpackhi_epi64(input[0], input[1]);
        let i2 = x86_64::_mm256_unpacklo_epi64(input[2], input[3]);
        let i3 = x86_64::_mm256_unpackhi_epi64(input[2], input[3]);

        let out0 = x86_64::_mm256_permute2x128_si256::<0x20>(i0, i2);
        let out1 = x86_64::_mm256_permute2x128_si256::<0x20>(i1, i3);
        let out2 = x86_64::_mm256_permute2x128_si256::<0x31>(i0, i2);
        let out3 = x86_64::_mm256_permute2x128_si256::<0x31>(i1, i3);

        [out0, out1, out2, out3]
    }
}

#[inline]
fn right_mat_mul_i_plus_1_dim_2<const HEIGHT: usize>(input: &mut Packed64bitM31Tensor<HEIGHT>) {
    unsafe {
        // Safety: If the inputs are <= L, the outputs are <= 5L.
        for i in 0..4 {
            let acc01 = x86_64::_mm256_add_epi64(input.0[0][i], input.0[1][i]);
            let acc01_shuffle = x86_64::_mm256_castpd_si256(x86_64::_mm256_permute_pd::<0b0101>(
                x86_64::_mm256_castsi256_pd(acc01),
            ));
            let sum = x86_64::_mm256_add_epi64(acc01, acc01_shuffle);

            input.0[0][i] = x86_64::_mm256_add_epi64(input.0[0][i], sum);
            input.0[1][i] = x86_64::_mm256_add_epi64(input.0[1][i], sum);
        }
    }
}

// TODO: Merge this function with the one above.
#[inline]
fn right_mat_mul_i_plus_1_dim_3<const HEIGHT: usize>(input: &mut Packed64bitM31Tensor<HEIGHT>) {
    unsafe {
        // Safety: If the inputs are <= L, the outputs are <= 7L.
        for i in 0..4 {
            let acc01 = x86_64::_mm256_add_epi64(input.0[0][i], input.0[1][i]);
            let acc012 = x86_64::_mm256_add_epi64(acc01, input.0[2][i]);
            let acc012_shuffle = x86_64::_mm256_castpd_si256(x86_64::_mm256_permute_pd::<0b0101>(
                x86_64::_mm256_castsi256_pd(acc012),
            ));
            let sum = x86_64::_mm256_add_epi64(acc012, acc012_shuffle);

            input.0[0][i] = x86_64::_mm256_add_epi64(input.0[0][i], sum);
            input.0[1][i] = x86_64::_mm256_add_epi64(input.0[1][i], sum);
            input.0[2][i] = x86_64::_mm256_add_epi64(input.0[2][i], sum);
        }
    }
}

#[inline]
fn right_mat_mul_i_plus_1_dim_4<const HEIGHT: usize>(input: &mut Packed64bitM31Tensor<HEIGHT>) {
    unsafe {
        // Safety: If the inputs are <= L, the outputs are <= 5L.
        for i in 0..4 {
            let acc01 = x86_64::_mm256_add_epi64(input.0[0][i], input.0[1][i]);
            let acc23 = x86_64::_mm256_add_epi64(input.0[2][i], input.0[3][i]);
            let sum = x86_64::_mm256_add_epi64(acc01, acc23);

            input.0[0][i] = x86_64::_mm256_add_epi64(input.0[0][i], sum);
            input.0[1][i] = x86_64::_mm256_add_epi64(input.0[1][i], sum);
            input.0[2][i] = x86_64::_mm256_add_epi64(input.0[2][i], sum);
            input.0[3][i] = x86_64::_mm256_add_epi64(input.0[3][i], sum);
        }
    }
}

// TODO: Merge this function with the one above.
#[inline]
fn right_mat_mul_i_plus_1_dim_6<const HEIGHT: usize>(input: &mut Packed64bitM31Tensor<HEIGHT>) {
    unsafe {
        // Safety: If the inputs are <= L, the outputs are <= 5L.
        for i in 0..4 {
            let acc01 = x86_64::_mm256_add_epi64(input.0[0][i], input.0[1][i]);
            let acc23 = x86_64::_mm256_add_epi64(input.0[2][i], input.0[3][i]);
            let acc45 = x86_64::_mm256_add_epi64(input.0[4][i], input.0[5][i]);
            let sum0123 = x86_64::_mm256_add_epi64(acc01, acc23);
            let sum = x86_64::_mm256_add_epi64(sum0123, acc45);

            input.0[0][i] = x86_64::_mm256_add_epi64(input.0[0][i], sum);
            input.0[1][i] = x86_64::_mm256_add_epi64(input.0[1][i], sum);
            input.0[2][i] = x86_64::_mm256_add_epi64(input.0[2][i], sum);
            input.0[3][i] = x86_64::_mm256_add_epi64(input.0[3][i], sum);
            input.0[4][i] = x86_64::_mm256_add_epi64(input.0[4][i], sum);
            input.0[5][i] = x86_64::_mm256_add_epi64(input.0[5][i], sum);
        }
    }
}

/// Given the initial vector __m256i, split into 2 vectors as follows:
/// HEIGHT = 2, 3:   [x0, x4, y0, y4]  ->    [x0, 0, y0, 0],   [0, x4, 0, y4]
/// HEIGHT = 4:      [w0, x0, y0, z0]  ->    [w0, x0, y0, z0], [0, 0, 0, 0]
#[inline]
fn split<const HEIGHT: usize>(input: __m256i) -> (__m256i, __m256i) {
    unsafe {
        const ZEROS: __m256i = unsafe { transmute::<[u64; 4], _>([0; 4]) }; // TODO: Use same definition of ZEROS everywhere. Maybe make it a global constant?
        match HEIGHT {
            2 | 3 => {
                let initial_elems = x86_64::_mm256_blend_epi32::<0b11001100>(input, ZEROS);
                let remainder = x86_64::_mm256_blend_epi32::<0b00110011>(input, ZEROS);
                (initial_elems, remainder)
            }
            4 | 6 => (input, ZEROS),
            _ => unreachable!(),
        }
    }
}

/// Perform a horizontal sum:
/// HEIGHT = 2, 3:   [x0, x4, y0, y4]  ->    [x0 + x4, x0 + x4, y0 + y4, y0 + y4]
/// HEIGHT = 4:      [w0, x0, y0, z0]  ->    [w0, x0, y0, z0]
#[inline]
fn horizontal_sum<const HEIGHT: usize>(input: __m256i) -> __m256i {
    unsafe {
        match HEIGHT {
            2 | 3 => {
                let shuffled = x86_64::_mm256_castpd_si256(x86_64::_mm256_permute_pd::<0b0101>(
                    x86_64::_mm256_castsi256_pd(input),
                ));
                x86_64::_mm256_add_epi64(input, shuffled)
            }
            4 | 6 => input,
            _ => unreachable!(),
        }
    }
}

pub trait Poseidon2AVX2Helpers {
    /// Given a vector of elements __m256i apply a monty reduction to each u64.
    /// Each u64 input must lie in [0, 2^{32}P)
    /// Each output will be a u64 lying in [0, P)
    fn full_reduce_vec(state: __m256i) -> __m256i;

    /// Given a vector of elements __m256i apply a partial monty reduction to each u64
    /// Each u64 input must lie in [0, 2^{32}P)
    /// Each output will be a u64 lying in [0, 2P)
    /// Slightly cheaper than full_reduce
    fn partial_reduce_vec(state: __m256i) -> __m256i;

    /// Apply the s-box: x -> x^s for some small s coprime to p - 1 to a vector __m256i.
    /// Input must be 4 u64's all in the range [0, P).
    /// Output will be 4 u64's all in the range [0, 2^{32}P).
    fn joint_sbox_vec(state: __m256i) -> __m256i;

    /// Apply the s-box: x -> (x + rc)^s to a vector __m256i.
    fn quad_internal_sbox(s0: __m256i, rc: u32) -> __m256i;

    /// Apply the s-box: x -> (x + rc)^s to a vector __m256i where we only care about the first 2 u64's
    fn double_internal_sbox(s0: __m256i, rc: u32) -> __m256i;

    const PACKED_8XPRIME: __m256i;
}

pub trait Poseidon2AVX2Methods<const HEIGHT: usize, const WIDTH: usize>:
    Clone + Sync + Poseidon2AVX2Helpers
{
    // PackField = PF;
    type PF: PackedField;

    // InternalRep should be [Packed64bitM31Tensor<HEIGHT>; N] with N = (WIDTH * PF::WIDTH)/(16 * HEIGHT)
    type InternalRep: AsMut<[Packed64bitM31Tensor<HEIGHT>]>;

    // Convert data to and from [PF; WIDTH], Packed64bitM31Tensor<HEIGHT>.
    fn from_input(input: [Self::PF; WIDTH]) -> Self::InternalRep;
    fn to_output(output: Self::InternalRep) -> [Self::PF; WIDTH];

    // Convert a set of external constants [F; PERMWIDTH] into the right form.
    fn manipulate_external_constants(
        input: [<Self::PF as PackedField>::Scalar; WIDTH],
    ) -> Packed64bitM31Tensor<HEIGHT>;

    // Given a field element, pull out the u32 stored inside.
    fn manipulate_internal_constants(input: <Self::PF as PackedField>::Scalar) -> u32;

    /// Apply full_reduce_vec to every __m256i in the matrix.
    #[inline]
    fn full_reduce(state: &mut Packed64bitM31Tensor<HEIGHT>) {
        for mat in state.0.iter_mut() {
            mat[0] = Self::full_reduce_vec(mat[0]);
            mat[1] = Self::full_reduce_vec(mat[1]);
            mat[2] = Self::full_reduce_vec(mat[2]);
            mat[3] = Self::full_reduce_vec(mat[3]);
        }
    }

    /// Apply partial_reduce_vec to every __m256i in the matrix.
    #[inline]
    fn partial_reduce(state: &mut Packed64bitM31Tensor<HEIGHT>) {
        for mat in state.0.iter_mut() {
            mat[0] = Self::partial_reduce_vec(mat[0]);
            mat[1] = Self::partial_reduce_vec(mat[1]);
            mat[2] = Self::partial_reduce_vec(mat[2]);
            mat[3] = Self::partial_reduce_vec(mat[3]);
        }
    }

    /// Apply joint_sbox_vec to every __m256i in the matrix.
    #[inline]
    fn joint_sbox(state: &mut Packed64bitM31Tensor<HEIGHT>) {
        for mat in state.0.iter_mut() {
            mat[0] = Self::joint_sbox_vec(mat[0]);
            mat[1] = Self::joint_sbox_vec(mat[1]);
            mat[2] = Self::joint_sbox_vec(mat[2]);
            mat[3] = Self::joint_sbox_vec(mat[3]);
        }
    }

    fn internal_sbox(s0: __m256i, rc: u32) -> __m256i {
        match HEIGHT {
            2 | 3 => Self::double_internal_sbox(s0, rc),
            4 | 6 => Self::quad_internal_sbox(s0, rc),
            _ => unreachable!(),
        }
    }

    // Constants for the matrix used in the internal linear layer.
    // Gives the diagonal elements of the matrix arranged in the appropriate way.
    const INTERNAL_SHIFTS: Packed64bitM31Tensor<HEIGHT>;
}

/// Compute a single internal Poseidon2 round.
/// Input must be < 2P < 2^32 but does not need to be canonical.
/// Round Constant is assumed to be in canonical form.
/// Output will be < 2P.
#[inline]
fn internal_round<const HEIGHT: usize, const WIDTH: usize, P2AVX2>(
    state: &mut Packed64bitM31Tensor<HEIGHT>,
    rc: u32,
) where
    P2AVX2: Poseidon2AVX2Methods<HEIGHT, WIDTH>,
{
    unsafe {
        // We do two things simultaneously.
        // Take the first value, add rc and compute the s-box.
        // Do a matrix multiplication on the remaining elements.
        // We will then move the first element back in later.

        let (s0, rem) = split::<HEIGHT>(state.0[0][0]);

        let s0_post_sbox = P2AVX2::internal_sbox(s0, rc); // Need to do something different to the first element.

        state.0[0][0] = rem;

        // Can do part of the sum vertically.
        let vec_sum = state.vec_sum();
        // still need to do the horizontal part of the sum but this can wait until later.

        // Doing the diagonal multiplication.
        state.left_shift(P2AVX2::INTERNAL_SHIFTS);

        // Need to multiply s0_post_sbox by -2.
        // s0_post_sbox < 2^34 so the easiest will be to do
        // 8P - s0_post_sbox to get the negative, then shift left by 1.
        // This will add 16P to some other terms in state but this doesn't matter as we work mod P.
        let neg_s0 = x86_64::_mm256_sub_epi64(P2AVX2::PACKED_8XPRIME, s0_post_sbox);
        let neg_2_s0 = x86_64::_mm256_add_epi64(neg_s0, neg_s0);

        state.0[0][0] = x86_64::_mm256_add_epi64(neg_2_s0, state.0[0][0]);

        let total_sum = x86_64::_mm256_add_epi64(vec_sum, s0_post_sbox);
        let shift = horizontal_sum::<HEIGHT>(total_sum);

        for mat in state.0.iter_mut() {
            mat[0] = x86_64::_mm256_add_epi64(mat[0], shift);
            mat[1] = x86_64::_mm256_add_epi64(mat[1], shift);
            mat[2] = x86_64::_mm256_add_epi64(mat[2], shift);
            mat[3] = x86_64::_mm256_add_epi64(mat[3], shift);
        }

        P2AVX2::partial_reduce(state); // Output, non canonical in [0, 2^32 - 2].
    }
}

/// A single External Round.
/// Note that we change the order to be mat_mul -> RC -> S-box (instead of RC -> S-box -> mat_mul in the paper).
/// Input does not need to be in canonical form but must be < 2^56.
/// Output will be < 2^34.
#[inline]
fn rotated_external_round<const HEIGHT: usize, const WIDTH: usize, P2AVX2>(
    state: &mut Packed64bitM31Tensor<HEIGHT>,
    round_constant: &Packed64bitM31Tensor<HEIGHT>,
) where
    P2AVX2: Poseidon2AVX2Methods<HEIGHT, WIDTH>,
{
    // Assume input state < L

    state.mat_mul_aes(); // state < 7L
    state.right_mat_mul_i_plus_1(); // state < 49L (or 35L depending on HEIGHT)
    state.add(*round_constant); // state < 49L + (2^31 - 1)
    P2AVX2::partial_reduce(state); // We need state < 2^62 for this. => L < (2^62)/50 < 2^56.
    P2AVX2::joint_sbox(state); // state is now < 2^34.
}

/// The initial set of external rounds. This consists of rf/2 external rounds followed by a mat_mul
/// Inputs will always be canonical though in principal only need to be < 2^56.
/// Output will be < 2P.
#[inline]
pub fn initial_external_rounds<const HEIGHT: usize, const WIDTH: usize, P2AVX2>(
    state: &mut Packed64bitM31Tensor<HEIGHT>,
    round_constants: &[Packed64bitM31Tensor<HEIGHT>],
) where
    P2AVX2: Poseidon2AVX2Methods<HEIGHT, WIDTH>,
{
    for round_constant in round_constants.iter() {
        rotated_external_round::<HEIGHT, WIDTH, P2AVX2>(state, round_constant)
    }

    state.mat_mul_aes();
    state.right_mat_mul_i_plus_1();
    P2AVX2::partial_reduce(state); // Input to internal_rounds only needs to be < 2^32.
}

/// The initial set of external rounds. This consists of rf/2 external rounds followed by a mat_mul
/// Inputs should be < 2P.
/// Output will be < 2P.
#[inline]
pub fn internal_rounds<const HEIGHT: usize, const WIDTH: usize, P2AVX2>(
    state: &mut Packed64bitM31Tensor<HEIGHT>,
    round_constants: &[u32],
) where
    P2AVX2: Poseidon2AVX2Methods<HEIGHT, WIDTH>,
{
    for round_constant in round_constants.iter() {
        internal_round::<HEIGHT, WIDTH, P2AVX2>(state, *round_constant)
    }
}

/// The final set of external rounds. Due to an ordering change it starts by doing a "half round" and finish by a mat_mul.
#[inline]
pub fn final_external_rounds<const HEIGHT: usize, const WIDTH: usize, P2AVX2>(
    state: &mut Packed64bitM31Tensor<HEIGHT>,
    round_constants: &[Packed64bitM31Tensor<HEIGHT>],
) where
    P2AVX2: Poseidon2AVX2Methods<HEIGHT, WIDTH>,
{
    // Input is < 2P
    state.add(round_constants[0]);
    P2AVX2::partial_reduce(state);
    P2AVX2::joint_sbox(state);

    for round_constant in round_constants.iter().skip(1) {
        rotated_external_round::<HEIGHT, WIDTH, P2AVX2>(state, round_constant)
    }

    state.mat_mul_aes();
    state.right_mat_mul_i_plus_1();
    // Output is not reduced.
}

/// A Poseidon2 abstraction allowing for fast Packed Field implementations.
/// F is the field from which the input variables are drawn.
/// T is the internal working type.
/// TODO: THIS WILL NEED CHANGES AS IMPLEMENTATIONS DEMAND.
/// EVENTUALLY WILL REPLACE MAIN POSEIDON2 TYPE
#[derive(Clone, Debug)]
pub struct Poseidon2AVX2<const HEIGHT: usize, const WIDTH: usize, P2AVX2> {
    /// The external round constants for the initial set of external rounds.
    initial_external_constants: Vec<Packed64bitM31Tensor<HEIGHT>>,

    /// The internal round constants.
    internal_constants: Vec<u32>,

    /// The external round constants for the final set of external rounds.
    final_external_constants: Vec<Packed64bitM31Tensor<HEIGHT>>,

    _phantom: PhantomData<P2AVX2>,
}

impl<const HEIGHT: usize, const WIDTH: usize, P2AVX2, PF> Poseidon2AVX2<HEIGHT, WIDTH, P2AVX2>
where
    PF: PackedField,
    PF::Scalar: PrimeField32,
    P2AVX2: Poseidon2AVX2Methods<HEIGHT, WIDTH, PF = PF>,
{
    /// Create a new Poseidon2 configuration.
    pub fn new(
        initial_external_constants: Vec<Packed64bitM31Tensor<HEIGHT>>,
        internal_constants: Vec<u32>,
        final_external_constants: Vec<Packed64bitM31Tensor<HEIGHT>>,
    ) -> Self {
        // Need to determine supported widths later.
        // assert!(SUPPORTED_WIDTHS.contains(&WIDTH));
        Self {
            initial_external_constants,
            internal_constants,
            final_external_constants,
            _phantom: PhantomData,
        }
    }

    /// Create a new Poseidon2 configuration with 128 bit security and random rounds constants.
    pub fn new_from_rng_128<R: Rng, const D: u64>(rng: &mut R) -> Self
    where
        Standard: Distribution<PF> + Distribution<[PF; WIDTH]>,
        Standard: Distribution<PF::Scalar> + Distribution<[PF::Scalar; WIDTH]>,
    {
        let (rounds_f, rounds_p) = poseidon2_round_numbers_128::<PF::Scalar>(WIDTH, D);
        let half_f = rounds_f / 2;

        let initial_external_constants = rng
            .sample_iter(Standard)
            .take(half_f)
            .map(P2AVX2::manipulate_external_constants)
            .collect();

        let final_external_constants = rng
            .sample_iter(Standard)
            .take(half_f)
            .map(P2AVX2::manipulate_external_constants)
            .collect();

        let internal_constants = rng
            .sample_iter(Standard)
            .take(rounds_p)
            .map(P2AVX2::manipulate_internal_constants)
            .collect();

        Self::new(
            initial_external_constants,
            internal_constants,
            final_external_constants,
        )
    }
}

impl<const HEIGHT: usize, const WIDTH: usize, P2AVX2, PF> Permutation<[PF; WIDTH]>
    for Poseidon2AVX2<HEIGHT, WIDTH, P2AVX2>
where
    PF: PackedField,
    P2AVX2: Poseidon2AVX2Methods<HEIGHT, WIDTH, PF = PF>,
{
    fn permute(&self, state: [PF; WIDTH]) -> [PF; WIDTH] {
        let mut internal_reps = P2AVX2::from_input(state);
        for internal_rep in internal_reps.as_mut() {
            initial_external_rounds::<HEIGHT, WIDTH, P2AVX2>(
                internal_rep,
                &self.initial_external_constants,
            );
            internal_rounds::<HEIGHT, WIDTH, P2AVX2>(internal_rep, &self.internal_constants);
            final_external_rounds::<HEIGHT, WIDTH, P2AVX2>(
                internal_rep,
                &self.final_external_constants,
            );
            P2AVX2::full_reduce(internal_rep); // TODO: Can maybe do a simpler reduction than this?
        }

        P2AVX2::to_output(internal_reps)
    }

    fn permute_mut(&self, input: &mut [PF; WIDTH]) {
        let output = self.permute(*input);
        *input = output;
    }
}

impl<const HEIGHT: usize, const WIDTH: usize, P2AVX2, PF> CryptographicPermutation<[PF; WIDTH]>
    for Poseidon2AVX2<HEIGHT, WIDTH, P2AVX2>
where
    PF: PackedField,
    P2AVX2: Poseidon2AVX2Methods<HEIGHT, WIDTH, PF = PF>,
{
}
