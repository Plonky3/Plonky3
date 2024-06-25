use core::arch::x86_64::{self, __m256i};
use core::mem::transmute;

// Internally, we represent our state as a tensor of size 4x4x1, 4x4x2, 4x4x3, 4x4x4 corresponding respectively to a single Poseidon-16 instance, 2 instances of Poseidon16, 2 instances of Poseidon24, or 4 instances of Poseidon 16.
// This may be reduced in future once we determine which versions are fastest.
// Currently the mapping from the standard [F; 16], [[F; 16]; 2], [[F; 24]; 2] to the internal 4xN matrix form looks like:
//
// 1 Poseidon 16 instance:  [x0, x4, x8, x12]
//                          [x1, x5, x9, x13]
//                          [x2, x6, x10, x14]
//                          [x3, x7, x11, x15]
//
// 2 Poseidon 16 instance:  [x0, x4, y0, y4] [x8, x12, y8, y12]
//                          [x1, x5, y1, y5] [x9, x13, y9, y13]
//                          [x2, x6, y2, y6] [x10, x14, y10, y14]
//                          [x3, x7, y3, y7] [x11, x15, y11, y15]
//
// 2 Poseidon 24 instance:  [x0, x4, y0, y4] [x8, x12, y8, y12]   [x16, x17, y16, y17]
//                          [x1, x5, y1, y5] [x9, x13, y9, y13]   [x18, x19, y18, y19]
//                          [x2, x6, y2, y6] [x10, x14, y10, y14] [x20, x21, y20, y21]
//                          [x3, x7, y3, y7] [x11, x15, y11, y15] [x22, x23, y22, y23]
//
// 4 Poseidon 16 instance:  [w0, x0, y0, z0] [w4, x4, y4, z4] [w8, x8, y8, z8]     [w12, x12, y12, z12]
//                          [w1, x1, y1, z1] [w5, x5, y5, z5] [w9, x9, y9, z9]     [w13, x13, y13, z13]
//                          [w2, x2, y2, z2] [w6, x6, y6, z6] [w10, x10, y10, z10] [w14, x14, y14, z14]
//                          [w3, x3, y3, z3] [w7, x7, y7, z7] [w11, x11, y11, z11] [w15, x15, y15, z15]
// This necessitates some data manipulation. <Long term we can make this faster by instead assuming a more natural form for the matrices and letting the scalar code deal with the data manipulation.

// The design mentality is that that poseidon2.permute and transmute should commute for all of the following: [[F; 16]; 4], [[PF; 2]; 4], [[PF; 4]; 2], [PF; 8].

/// A 4x4xN Matrix of 31-bit field elements with each element stored in 64-bits and each row saved as (multiple) 256bit packed vectors.
/// Used for the internal representations for vectorized AVX2 implementations for Poseidon2
/// Should only be called with N = 1, 2, 3, 4
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Packed64bitM31Tensor<const HEIGHT: usize>([[__m256i; 4]; HEIGHT]);

impl<const HEIGHT: usize> Packed64bitM31Tensor<HEIGHT> {
    // Some of these need to be re-thought as to how they should apply in the tensor model:

    // /// Convert an array of packed field elements into a Packed64bitM31Matrix prepared for Poseidon2
    // #[inline]
    // pub fn from_packed_field_array(input: [PackedMersenne31AVX2; 2]) -> Packed64bitM31Matrix {
    //     unsafe {
    //         // Safety: `PackedMersenne31AVX2, Mersenne31, Packed64bitM31Matrix` are all `repr(transparent)`
    //         // Thus [PackedMersenne31AVX2; 2] can be transmuted to/from [u32; 16] and
    //         // Packed64bitM31Matrix can be transmuted to/from [u64; 16];
    //         let array_u32: [u32; 16] = transmute(input);
    //         let mut output: Packed64bitM31Matrix = transmute(array_u32.map(|x| x as u64));
    //         output.transpose(); // TODO: these should be removed, but it changes the interpretation of the permutation so will involve a change to the scalar version too.
    //         output
    //     }
    // }

    // /// Convert a Packed64bitM31Matrix back into an array of packed field elements.
    // /// The input may not be in canonical form.
    // #[inline]
    // pub fn to_packed_field_array(mut input: Packed64bitM31Matrix) -> [PackedMersenne31AVX2; 2] {
    //     unsafe {
    //         // Safety: `PackedMersenne31AVX2, Mersenne31, Packed64bitM31Matrix` are all `repr(transparent)`
    //         // Thus [PackedMersenne31AVX2; 2] can be transmuted to/from [u32; 16] and
    //         // Packed64bitM31Matrix can be transmuted to/from [u64; 16];
    //         input.full_reduce();
    //         input.transpose(); // TODO: these should be removed, but it changes the interpretation of the permutation so will involve a change to the scalar version too.
    //         let output: [u32; 16] = transmute::<_, [u64; 16]>(input).map(|x| x as u32);
    //         transmute(output)
    //     }
    // }

    // Convert data into the form expected by the Poseidon2 implementation
    #[inline]
    fn shuffle_data(&mut self) {
        match HEIGHT {
            1 => self.0[0] = transpose(self.0[0]),
            2 => {
                let mat0 = transpose([self.0[0][0], self.0[0][1], self.0[1][0], self.0[1][1]]);
                let mat1 = transpose([self.0[0][2], self.0[0][3], self.0[1][2], self.0[1][3]]);

                self.0[0] = mat0;
                self.0[1] = mat1;
            }
            3 => {
                let mat0 = transpose([self.0[0][0], self.0[0][1], self.0[1][2], self.0[1][3]]);
                let mat1 = transpose([self.0[0][2], self.0[0][3], self.0[2][0], self.0[2][1]]);
                let mat2 = transpose([self.0[1][0], self.0[1][1], self.0[2][2], self.0[2][3]]);

                self.0[0] = mat0;
                self.0[1] = mat1;
                self.0[2] = mat2;
            }
            4 => {
                let mat0 = transpose([self.0[0][0], self.0[1][0], self.0[2][0], self.0[3][0]]);
                let mat1 = transpose([self.0[0][1], self.0[1][1], self.0[2][1], self.0[3][1]]);
                let mat2 = transpose([self.0[0][2], self.0[1][2], self.0[2][2], self.0[3][2]]);
                let mat3 = transpose([self.0[0][3], self.0[1][3], self.0[2][3], self.0[3][3]]);

                self.0[0] = mat0;
                self.0[1] = mat1;
                self.0[1] = mat2;
                self.0[1] = mat3;
            }
            _ => unreachable!(),
        };
    }

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
            for mut matrix in self.0 {
                let t01 = x86_64::_mm256_add_epi64(matrix[0], matrix[1]);
                let t23 = x86_64::_mm256_add_epi64(matrix[2], matrix[3]);
                let t0123 = x86_64::_mm256_add_epi64(t01, t23);
                let t01123 = x86_64::_mm256_add_epi64(t0123, matrix[1]);
                let t01233 = x86_64::_mm256_add_epi64(t0123, matrix[3]);

                let t00 = x86_64::_mm256_slli_epi64::<1>(matrix[0]);
                let t22 = x86_64::_mm256_slli_epi64::<1>(matrix[2]);

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
            1 => right_mat_mul_i_plus_1_dim_1(&mut self.0[0]),
            2 => right_mat_mul_i_plus_1_dim_2(self),
            3 => right_mat_mul_i_plus_1_dim_3(self),
            4 => right_mat_mul_i_plus_1_dim_4(self),
            _ => unreachable!(),
        };
        // This basically boils down to needing to take the sum of each row.
    }

    /// Add in round constants.
    #[inline]
    fn add_rc(&mut self, rc: Self) {
        unsafe {
            // Safety: element of rc must be in canonical form.
            // Elements of self should be small enough such that overflow is impossible.
            for i in 0..HEIGHT {
                self.0[i][0] = x86_64::_mm256_add_epi64(self.0[i][0], rc.0[i][0]);
                self.0[i][1] = x86_64::_mm256_add_epi64(self.0[i][1], rc.0[i][1]);
                self.0[i][2] = x86_64::_mm256_add_epi64(self.0[i][2], rc.0[i][2]);
                self.0[i][3] = x86_64::_mm256_add_epi64(self.0[i][3], rc.0[i][3]);
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
fn right_mat_mul_i_plus_1_dim_1(mat: &mut [__m256i; 4]) {
    unsafe {
        // Safety: If the inputs are <= L, the outputs are <= 5L.
        mat[0] = x86_64::_mm256_add_epi64(mat[0], hsum(mat[0]));
        mat[1] = x86_64::_mm256_add_epi64(mat[1], hsum(mat[1]));
        mat[2] = x86_64::_mm256_add_epi64(mat[2], hsum(mat[2]));
        mat[3] = x86_64::_mm256_add_epi64(mat[3], hsum(mat[3]));
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
//    /// The following all need to be made specific to the field we are using.
//     #[inline]
//     fn reduce(?) { } // Reduce from ? to u32
//     #[inline]
//     fn full_reduce(?) { } // Reduce from ? to an element in [0, P)
//     #[inline]
//     fn joint_sbox(?) { } // Input a vector of u32's stored in u64's
//     #[inline]
//     fn double_sbox(?) { } // Input a vector of u32's stored in u64's. Only care about first 2 entries.
//     #[inline]
//     fn scalar_sbox(?) { } // Input a scaler u32 stored as a u64.
//
//     INTERNAL_SHIFTS0_T // Constants for internal linear layer.
//
//
//

//     /// Compute x -> x^5 for each element of the vector.
//     /// The input must be in canonical form.
//     #[inline]
//     fn joint_sbox(&mut self) {
//         self.0[0] = joint_sbox(self.0[0]);
//         self.0[1] = joint_sbox(self.0[1]);
//         self.0[2] = joint_sbox(self.0[2]);
//         self.0[3] = joint_sbox(self.0[3]);
//     }

//     /// Compute a single internal Poseidon2 round.
//     /// Assume inputs are < 2^32 - 1, but may not be canonical.
//     /// Assume the round constant is given in canonical form.
//     #[inline]
//     fn internal_round(&mut self, rc: u32) {
//         unsafe {
//             // We do two things simultaneously.
//             // Take the first value, add rc and compute the cube.
//             // Do a matrix multiplication on the remaining elements.
//             // We will then move the first element back in later.

//             let s0 = { transmute::<_, [u64; 4]>(self.0[0]) }[0] as u32; // Pull out the first element.

//             // Can do part of the sum vertically.
//             let t01 = x86_64::_mm256_add_epi64(self.0[0], self.0[1]);
//             let t23 = x86_64::_mm256_add_epi64(self.0[2], self.0[3]);
//             let t0123 = x86_64::_mm256_add_epi64(t01, t23);

//             // Now need to sum t0123 horizontally.
//             let t0123: [u64; 4] = transmute(t0123);
//             let total = t0123[0] + t0123[1] + t0123[2] + t0123[3] - (s0 as u64);
//             // IMPROVE: Suspect this is suboptimal and can be improved.

//             // Doing the diagonal multiplication.
//             self.0[0] = x86_64::_mm256_sllv_epi64(self.0[0], INTERNAL_SHIFTS0_T);
//             self.0[1] = x86_64::_mm256_sllv_epi64(self.0[1], INTERNAL_SHIFTS1_T);
//             self.0[2] = x86_64::_mm256_sllv_epi64(self.0[2], INTERNAL_SHIFTS2_T);
//             self.0[3] = x86_64::_mm256_sllv_epi64(self.0[3], INTERNAL_SHIFTS3_T);

//             // Need to compute s0 -> (s0 + rc)^5
//             let (sum, over) = s0.overflowing_add(rc); // s0 + rc <= 3P, over detects if its > 2^32 - 1 = 2P + 1.
//             let (sum_corr, under) = sum.overflowing_sub(P << 1); // If over, sum_corr is in [0, P].
//                                                                  // Under is used to flag the unique sum = 2P + 1 case.
//             let sum_sub = sum.wrapping_sub(P) as i32; // If not over and under, sum_sub is in [-P, P].

//             let val = if over | !under {
//                 sum_corr as i32
//             } else {
//                 sum_sub
//             }; // -P - 1 <= val <= P

//             let sq = (val as i64) * (val as i64); // 0 <= sq <= P^2
//             let sq_red = ((sq as u32 & P) + ((sq >> 31) as u32)).wrapping_sub(P) as i32; // -P <= sq_red <= P

//             let quad = (sq_red as i64) * (sq_red as i64); // 0 <= quad <= P^2
//             let quad_red = ((quad as u32 & P) + ((quad >> 31) as u32)).wrapping_sub(P) as i32; // -P <= quad_red <= P

//             let fifth = (((quad_red as i64) * (val as i64)) + PSQ) as u64; // 0 <= fifth <= 2P^2
//             let fifth_red =
//                 ((fifth as u32 & P) + ((fifth >> 31) as u32 & P) + ((fifth >> 62) as u32)) as u64; // Note fifth_red <= 2P + 1 < 2^32.

//             // Need to mutiply self00 by -2.
//             // Easiest to do 4P - self00 to get the negative, then shift left by 1.
//             // only involves shifts.
//             let s00 = (PX4 - fifth_red) << 1;

//             self.0[0] = x86_64::_mm256_insert_epi64::<0>(self.0[0], s00 as i64);

//             let full_total = total + fifth_red;
//             let shift = x86_64::_mm256_set1_epi64x(full_total as i64);

//             self.0[0] = x86_64::_mm256_add_epi64(self.0[0], shift);
//             self.0[1] = x86_64::_mm256_add_epi64(self.0[1], shift);
//             self.0[2] = x86_64::_mm256_add_epi64(self.0[2], shift);
//             self.0[3] = x86_64::_mm256_add_epi64(self.0[3], shift);

//             self.reduce() // Output, non canonical in [0, 2^32 - 2].
//         }
//     }

/// Compute the horizontal sum.
/// Outputs a constant __m256i vector with each element equal to the sum.
#[inline]
fn hsum(input: __m256i) -> __m256i {
    unsafe {
        let t0: [u64; 4] = transmute(input);
        let total0 = t0[0] + t0[1] + t0[2] + t0[3];
        x86_64::_mm256_set1_epi64x(total0 as i64)
    }
    // Another possible approach which doesn't pass to scalars:
    // let t0 = x86_64::_mm256_castpd_si256(x86_64::_mm256_permute_pd::<0b0101>(x86_64::_mm256_castsi256_pd(input)));
    // let part_sum = x86_64::_mm256_add_epi64(input, t0);
    // let part_sum_swap = x86_64::_mm256_permute4x64_epi64::<0b00001111>(part_sum);
    // x86_64::_mm256_add_epi64(part_sum, part_sum_swap)
}

/// A single External Round.
/// Note that we change the order to be mat_mul -> RC -> S-box (instead of RC -> S-box -> mat_mul in the paper).
/// Input does not need to be in canonical form, < 2^50 is fine.
/// Output will be < 2^33.
#[inline]
fn rotated_external_round<const HEIGHT: usize>(
    state: &mut Packed64bitM31Tensor<HEIGHT>,
    round_constant: &Packed64bitM31Tensor<HEIGHT>,
) {
    state.mat_mul_aes();
    state.right_mat_mul_i_plus_1();
    state.add_rc(*round_constant);
    state.full_reduce();
    state.joint_sbox();
}

/// The initial set of external rounds. This consists of rf/2 external rounds followed by a mat_mul
#[inline]
pub fn initial_external_rounds<const HEIGHT: usize>(
    state: &mut Packed64bitM31Tensor<HEIGHT>,
    round_constants: &[Packed64bitM31Tensor<HEIGHT>],
) {
    for round_constant in round_constants.iter() {
        rotated_external_round(state, round_constant)
    }

    state.mat_mul_aes();
    state.right_mat_mul_i_plus_1();
    state.full_reduce(); // Might be able to get away with not doing this.
}

/// The initial set of external rounds. This consists of rf/2 external rounds followed by a mat_mul
#[inline]
pub fn internal_rounds<const HEIGHT: usize>(
    state: &mut Packed64bitM31Tensor<HEIGHT>,
    round_constants: &[u32],
) {
    for round_constant in round_constants.iter() {
        state.internal_round(*round_constant)
    }
}

/// The final set of external rounds. Due to an ordering change it starts by doing a "half round" and finish by a mat_mul.
#[inline]
pub fn final_external_rounds<const HEIGHT: usize>(
    state: &mut Packed64bitM31Tensor<HEIGHT>,
    round_constants: &[Packed64bitM31Tensor<HEIGHT>],
) {
    state.add_rc(round_constants[0]);
    state.full_reduce(); // Can possibly do something cheaper than full reduce here?
    state.joint_sbox();

    for round_constant in round_constants.iter().skip(1) {
        rotated_external_round(state, round_constant)
    }

    state.mat_mul_aes();
    state.right_mat_mul_i_plus_1();
    // Output is not reduced.
}
