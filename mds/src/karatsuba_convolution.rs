use core::ops::{Add, Sub, AddAssign, SubAssign, Mul, MulAssign, Shr, Shl, ShrAssign, ShlAssign};
use alloc::vec;
use alloc::vec::Vec;
use itertools::Either;
use itertools::Itertools;
use p3_field::PrimeField64;

const MATRIX_CIRC_MDS_8_SML: [i64; 8] = [4, 1, 2, 9, 10, 5, 1, 1];

const MATRIX_CIRC_MDS_16_SML: [i64; 16] =
    [1, 1, 51, 1, 11, 17, 2, 1, 101, 63, 15, 2, 67, 22, 13, 3];

#[rustfmt::skip]
const MATRIX_CIRC_MDS_32_MERSENNE31: [i128; 32] = [
    0x1896DC78, 0x559D1E29, 0x04EBD732, 0x3FF449D7,
    0x2DB0E2CE, 0x26776B85, 0x76018E57, 0x1025FA13,
    0x06486BAB, 0x37706EBA, 0x25EB966B, 0x113C24E5,
    0x2AE20EC4, 0x5A27507C, 0x0CD38CF1, 0x761C10E5,
    0x19E3EF1A, 0x032C730F, 0x35D8AF83, 0x651DF13B,
    0x7EC3DB1A, 0x6A146994, 0x588F9145, 0x09B79455,
    0x7FDA05EC, 0x19FE71A8, 0x6988947A, 0x624F1D31,
    0x500BB628, 0x0B1428CE, 0x3A62E1D6, 0x77692387
];

// It will be handy for functions to be able to handle entries which are a combination of simple integer types
// In particular u64's, i64's, u128's and i128's so we make a general trait type here.
trait SimpleInteger:
    Sized
    + Default
    + Copy
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + MulAssign
    + ShrAssign<usize>
    + Shr<usize, Output = Self>
    + ShlAssign<usize>
    + Shl<usize, Output = Self>
    {}

impl<T> SimpleInteger for T where T: Sized
    + Default
    + Copy
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + MulAssign
    + ShrAssign<usize>
    + Shr<usize, Output = Self>
    + ShlAssign<usize>
    + Shl<usize, Output = Self>
    {}

// Let M be a circulant matrix with first column vec_col and first row vec_row. Then M.u is the convolution of vec_col and u.
// The vectors given here are the first rows of the respective circulant matrices NOT the first colums.
// Hence in order to do convolutions we need to compute the first column which is given by
// vec_col = [vec_row[0], vec_row[n - 1], vec_row[n - 2], ..., vec_row[1]]
// Clearly the following functions should be equal but I can't currently work out how to do it keeping everything const. (T::default() isn't const apparently...)
#[inline]
const fn row_to_col<T: SimpleInteger, const N: usize>(row: [T; N]) -> [T; N] {
    let mut col = [row[0]; N];
    let mut i = 1;
    loop {
        if i == N {
            break;
        }
        col[i] = row[N - i];
        i += 1
    }
    col
}

// Given a vector v, split it into its odd and even parts.
#[inline]
const fn split_half_eo<const N: usize, const HALF_N: usize>(
    vec: [i64; N],
) -> ([u64; HALF_N], [u64; HALF_N]) {
    let mut even = [0; HALF_N];
    let mut odd = [0; HALF_N];

    let mut i = 0;
    loop {
        if i == HALF_N {
            break;
        }
        even[i] = vec[2*i] as u64;
        odd[i] = vec[2*i + 1] as u64;
        i += 1
    }
    (even, odd)
}

// Several functions here to encode some simple vector addition and similar.

// Performs vector addition on slices saving the result in the first slice
#[inline]
fn add_mut<T: SimpleInteger>(lhs: &mut [T], rhs: &[T]) -> () {
    let n = rhs.len();
    for i in 0..n {
        lhs[i] += rhs[i]
    }
}

// Performs vector addition on slices returning a new array.
// This should be a const function but I can't work out how to do it.
// Need to say that Add and Default are constant. (I have a workaround for Default but not Add)
#[inline]
fn add_vec<T: SimpleInteger, const N: usize>(lhs: &[T], rhs: &[T]) -> [T; N] {
    let mut output:[T; N] = [T::default(); N];
    let mut i = 0;
    loop {
        output[i] = lhs[i] + rhs[i];
        i += 1;
        if i == N {
            break
        }
    }
    output
}

// Performs vector subtraction on slices saving the result in the first slice
#[inline]
fn sub_mut<T: SimpleInteger>(lhs: &mut [T], sub: &[T]) -> () {
    let n = sub.len();
    for i in 0..n {
        lhs[i] -= sub[i]
    }
}

// Performs vector addition on slices returning a new array.
// This should be a const function but I can't work out how to do it.
// Need to say that Sub and Default are constant. (I have a workaround for Default but not Sub)
#[inline]
fn sub_vec<T: SimpleInteger, const N: usize>(lhs: &[T], sub: &[T]) -> [T; N] {
    let mut output:[T; N] = [T::default(); N];
    let mut i = 0;
    loop {
        output[i] = lhs[i] - sub[i];
        i += 1;
        if i == N {
            break
        }
    }
    output
}

// Performs the vector operation lhs += val*rhs
#[inline]
fn add_mul_mut<T: SimpleInteger>(lhs: &mut [T], rhs: &[T], val: T) {
    let n = rhs.len();
    for i in 0..n {
        lhs[i] += val * rhs[i]
    }
}

// Takes the dot product of two vectors.
#[inline]
fn dot<T: SimpleInteger>(lhs: &[T], rhs: &[T]) -> T {
    let n = lhs.len();
    let mut sum = lhs[0]*rhs[0];
    for i in 1..n {
        sum += lhs[i]*rhs[i];
    }
    sum
}

// Given a vector v \in F^N, let v(x) \in F[X] denote the polynomial v_0 + v_1 x + ... + v_{N - 1} x^{N - 1}.
// Then w is equal to the convolution v * u if and only if w(x) = v(x)u(x) mod x^N - 1.
// Additionally, define the signed convolution by w(x) = v(x)u(x) mod x^N + 1.

// Using the chinese remainder theorem we can compute w(x) from:
//                      w_0 = v(x)u(x) mod x^{N/2} - 1
//                      w_1 = v(x)u(x) mod x^{N/2} + 1
// Via:
//                      w(x) = 1/2 (w_0(x) + w_1(x)) + x^{N/2}/2 (w_0(x) - w_1(x))

// To compute w_0 and w_1 we first compute
//                  v_0(x) = v(x) mod x^{N/2} - 1
//                  v_1(x) = v(x) mod x^{N/2} + 1
//                  u_0(x) = v(x) mod x^{N/2} - 1
//                  u_1(x) = v(x) mod x^{N/2} + 1

// Now w_0 is the convolution of v_0 and u_0 so this can be applied recursively.
// For w_1 we compute the signed convolution v_1(x)u_1(x) mod x^{N/2} + 1 using Karatsuba.              

// There are 2 possible approaches to this karatsuba which mirror the DIT vs DIF approaches to FFT's.

// Option 1: left/right decomposition.
// Write v = (v_l, v_r) so that v(x) = (v_l(x) + x^{N/2}v_r(x)).
// Then v(x)u(x) mod x^N + 1 = (v_l(x)u_l(x) - v_r(x)u_r(x)) + x^{N/2}((v_l(x) + v_r(x))(u_l(x) + u_r(x)) - (v_l(x)u_l(x) + v_r(x)u_r(x))) mod X^N + 1

// As v_l(x), v_r(x), u_l(x), u_r(x) all have degree < N/2 no product will have degree > N - 1.
// The only place we need to deal with the mod operation is after the multipication by x^{N/2} and this is easy to do.
// Thus this reduces the problem to 3 polynomial multiplications of size N/2 and we can use the standard karatsuba for this.

// Option 2: even/odd decomposition.
// Define the even v_e and odd v_o parts so that v(x) = (v_e(x^2) + xv_o(x^2)).
// Then v(x)u(x) = (v_e(x^2)u_e(x^2) + x^2 v_o(x^2)u_o(x^2)) + x ((v_e(x^2) + v_o(x^2))(u_e(x^2) + u_o(x^2)) - (v_e(x^2)u_e(x^2) + v_o(x^2)u_o(x^2))
// This reduces the problem to 3 signed convolutions of size N/2 and we can do this recursively.

// These algorithms do look a little different so it's not immediately obvious which is the right one to compute.
// Thus for now we will implement both.

// Once we get down to size 4/8 we use the O(n^2) approach.

/// Compute the product of two polynomials of degree 3.
/// output(x) = lhs(x)rhs(x)
#[inline]
fn prod4<T: SimpleInteger>(lhs: &[T; 4], rhs: &[T; 4]) -> [T; 7] {
    let mut output = [T::default(); 7];
    add_mul_mut(&mut output[..4], lhs, rhs[0]);
    add_mul_mut(&mut output[1..5], lhs, rhs[1]);
    add_mul_mut(&mut output[2..6], lhs, rhs[2]);
    add_mul_mut(&mut output[3..], lhs, rhs[3]);
    output
}

/// Compute the convolution product of two vectors of length 4.
/// output(x) = lhs(x)rhs(x) mod x^4 - 1
#[inline]
fn conv4<T: SimpleInteger>(lhs: &[T; 4], rhs: &[T; 4]) -> [T; 4] {
    // Even at this small size, doing the FFT decomposition seems to produce shorter compiled code using godbolt.
    // In particular testing the code produced for conv8.

    const N: usize = 4;
    const HALF: usize = N/2;
    let mut output = [T::default(); N];

    let lhs_p: [T; HALF] = add_vec(&lhs[..HALF], &lhs[HALF..]);     // v_0(x)
    let lhs_m: [T; HALF] = sub_vec(&lhs[..HALF], &lhs[HALF..]);     // v_1(x)

    // In most cases, rhs will actually be a constant but I'm unsure how to take advantage of this.
    let rhs_p: [T; HALF] = add_vec(&rhs[..HALF], &rhs[HALF..]);     // u_0(x)
    let rhs_m: [T; HALF] = sub_vec(&rhs[..HALF], &rhs[HALF..]);     // u_1(x)

    output[0] = lhs_p[0]*rhs_p[0] + lhs_p[1]*rhs_p[1];              
    output[1] = lhs_p[0]*rhs_p[1] + lhs_p[1]*rhs_p[0];              // output = v_0(x)u_0(x) mod x^2 - 1
    output[2] = output[0];
    output[3] = output[1];                                          // output = (1 + x^2)(v_0(x)u_0(x) mod x^2 - 1)

    let product_m = [lhs_m[0]*rhs_m[0] - lhs_m[1]*rhs_m[1], lhs_m[0]*rhs_m[1] + lhs_m[1]*rhs_m[0]];

    add_mut(&mut output[..HALF], &product_m);
    sub_mut(&mut output[HALF..], &product_m);                       // output = 2 (lhs * rhs)

    // Can maybe do this in place?
    output.map(|x| x >> 1)          // output = lhs * rhs

    // I leave the old N^2 code here for now as, with improvments
    // it may still end up as the better option.

    // let mut output = [T::default(); 4];
    // let rhs_rev_0 = [rhs[0], rhs[3], rhs[2], rhs[1]];
    // output[0] = dot(lhs, &rhs_rev_0);
    // let rhs_rev_1 = [rhs[1], rhs[0], rhs[3], rhs[2]];
    // output[1] = dot(lhs, &rhs_rev_1);
    // let rhs_rev_2 = [rhs[2], rhs[1], rhs[0], rhs[3]];
    // output[2] = dot(lhs, &rhs_rev_2);
    // let rhs_rev_3 = [rhs[3], rhs[2], rhs[1], rhs[0]];
    // output[3] = dot(lhs, &rhs_rev_3);
    // output

}

/// Compute the signed convolution of two vectors of length 4.
/// => output(x) = lhs(x)rhs(x) mod x^4 + 1
#[inline]
fn sign_conv4<T: SimpleInteger>(lhs: &[T; 4], rhs: &[T; 4]) -> [T; 4] {
    let mut output = [T::default(); 4];

    let rhs_rev = [rhs[3], rhs[2], rhs[1], rhs[0]];

    output[0] = lhs[0]*rhs[0] - dot(&lhs[1..], &rhs_rev[..3]);
    output[1] = dot(&lhs[..2], &rhs_rev[2..]) - dot(&lhs[2..], &rhs_rev[..2]);
    output[2] = dot(&lhs[..3], &rhs_rev[1..]) - lhs[3]*rhs[3];
    output[3] = dot(lhs, &rhs_rev);
    output

    // It may be possible to choose the MDS vector such that rhs = [1, 1, 1, 1] or some constant version of this.
    // In such cases, we can use faster code for sign_conv4. Just saving this here for now
    // let output1 = lhs[0] + lhs[1] - lhs[2] - lhs[3];
    // let output2 = output1 + (lhs[2] << 1);
    // [
    //     output1 - (lhs[1] << 1),
    //     output1,
    //     output2,
    //     output2 + (lhs[3] << 1),
    // ]
}

/// Assume that lhs is a vector of u64's and rhs is a vector of field elements.
/// Compute M(lhs).rhs where M is the circulant matrix with first row equal to lhs.
/// This uses the left/right decomposition.
/// Currently have not implmented the FFT trick to use signed convolutions.
pub fn apply_circulant_karat_left_right_field<F: PrimeField64, const N: usize>(
    lhs: [u64; N],
    rhs: [F; N],
) -> [F; N] {

    // We need the vector which is the first column of lhs not the first row.
    let lhs_u64 = {
        let mut lhs_u64 = lhs.into_iter().rev().collect::<Vec<_>>();
        lhs_u64.rotate_right(1);
        lhs_u64
    };
    let rhs_u64 = rhs.map(|x| x.as_canonical_u64());

    // Result is computed as a polynomial of degree 2N - 2.
    // We manually implement the reduction by X^N - 1.
    let mut result = karatsuba_polynomial_multiplication(&lhs_u64, &rhs_u64);

    for i in 0..(N - 1) {
        result[i] += result[i + N];
    }

    result[..N]
        .iter()
        .map(|x| F::from_wrapped_u64(*x))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

// Assume that lhs is a vector of u64's and rhs is a vector of field elements.
// This computes M(lhs).rhs where M is the circulant matrix with first row equal to lhs.
// It uses the even-odd decomposition.
/// Currently have not implmented the FFT trick to use signed convolutions.
pub fn apply_circulant_karat_even_odd_field<F: PrimeField64, const N: usize>(
    lhs: [u64; N],
    rhs: [F; N],
) -> [F; N] {

    // We need the vector which is the first column of lhs not the first row.
    let lhs_u64 = {
        let mut lhs_u64 = lhs.into_iter().rev().collect::<Vec<_>>();
        lhs_u64.rotate_right(1);
        lhs_u64
    };
    let rhs_u64 = rhs.map(|x| x.as_canonical_u64());

    let result = apply_circulant_karat_even_odd(&lhs_u64, &rhs_u64);

    result
        .into_iter()
        .map(|x| F::from_wrapped_u64(x))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

/// Computes the convolution of input and MATRIX_CIRC_MDS_8_SML.
/// Input must be an array of field elements of length 8.
/// Only works with Mersenne31 and Babybear31
pub fn apply_circulant_8_karat<F: PrimeField64>(input: [F; 8]) -> [F; 8] {

    // Flip MATRIX_CIRC_MDS_8_SML to get the first column of the circulant matrix.
    const MATRIX_CIRC_MDS_8_SML_I64: [i64; 8] = row_to_col(MATRIX_CIRC_MDS_8_SML);

    // The numbers we will encounter through our algorithm are bounded by
    // SUM(input.as_canonical_u64()) * SUM(MATRIX_CIRC_MDS_8_SML) <= (8 * 2**31) * 33 < 2**40 << 2**63
    // Hence we can work with i64's with garuntees of no overflow occuring.
    let input_i64 = input.map(|x| x.as_canonical_u64() as i64);

    // Compute the convolution.
    // Currently not taking full advantage of MATRIX_CIRC_MDS_8_SML_I64 being constant.
    // TODO: FIX this.
    let output = conv8(&input_i64, &MATRIX_CIRC_MDS_8_SML_I64);

    // Whilst some intermediate steps may be negative, as we started with 2 positive vectors
    // The output will always be positive so we can safley cast as u64's.
    output.map(|x| F::from_wrapped_u64(x as u64))
}

/// Computes the convolution of input and MATRIX_CIRC_MDS_16_SML.
/// Input must be an array of field elements of length 16.
/// Only works with Mersenne31 and Babybear31
pub fn apply_circulant_16_karat<F: PrimeField64>(input: [F; 16]) -> [F; 16] {

    // Flip MATRIX_CIRC_MDS_16_SML to get the first column of the circulant matrix.
    const MATRIX_CIRC_MDS_16_SML_I64: [i64; 16] = row_to_col(MATRIX_CIRC_MDS_16_SML);

    // The numbers we will encounter through our algorithm are bounded by
    // SUM(input.as_canonical_u64()) * SUM(MATRIX_CIRC_MDS_16_SML) <= (16 * 2**31) * 371 < 2**44 << 2**63
    // Hence we can work with i64's with garuntees of no overflow occuring.
    let input_i64 = input.map(|x| x.as_canonical_u64() as i64);

    // Compute the convolution.
    // Currently not taking full advantage of MATRIX_CIRC_MDS_16_SML_I64 being constant.
    // TODO: FIX this.
    let output = conv16(&MATRIX_CIRC_MDS_16_SML_I64, &input_i64);

    // Whilst some intermediate steps may be negative, as we started with 2 positive vectors
    // The output will always be positive so we can safley cast as u64's.
    output.map(|x| F::from_wrapped_u64(x as u64))
}

/// Computes the convolution of input and MATRIX_CIRC_MDS_32_MERSENNE31.
/// Input must be an array of Mersenne31 field elements of length 32.
pub fn apply_circulant_32_karat<F: PrimeField64>(input: [F; 32]) -> [F; 32] {
    // Flip MATRIX_CIRC_MDS_32_MERSENNE31 to get the first column of the circulant matrix.
    const MATRIX_CIRC_MDS_32_M31_I128: [i128; 32] = row_to_col(MATRIX_CIRC_MDS_32_MERSENNE31);

    // The numbers we will encounter through our algorithm are bounded by
    // SUM(input.as_canonical_u64()) * SUM(MATRIX_CIRC_MDS_32_MERSENNE31) <= (32 * 2**31)**2 < 2**72 << 2**127
    // Hence we can work with i128's with garuntees of no overflow occuring.
    let input_i128 = input.map(|x| x.as_canonical_u64() as i128);
    
    // Compute the convolution.
    // Currently not taking full advantage of MATRIX_CIRC_MDS_16_SML_I64 being constant.
    // TODO: FIX this.
    let output = conv32(&MATRIX_CIRC_MDS_32_M31_I128, &input_i128);

    // Whilst some intermediate steps may be negative, as we started with 2 positive vectors
    // The output will always be positive so we can safley cast as u64's.
    output.map(|x| F::from_wrapped_u128(x as u128))
}


/// Compute the product of two polynomials of degree 7.
/// output(x) = lhs(x)rhs(x)
/// We use Karatsuba as it should save a small number of operations (103 vs 113).
/// This should be checked agaisnt the standard approach.
#[inline]
fn prod8<T: SimpleInteger>(lhs: &[T; 8], rhs: &[T; 8]) -> [T; 15] {
    let mut output = [T::default(); 15];

    // The algorithm is simple:
    // v(x)u(x) = (v_l(x) + x^4v_h(x))(u_l(x) + x^4u_h(x))
    //          = v_l(x)u_l(x) + x^4((v_l(x) + v_h(x))(u_l(x) + u_h(x)) - v_l(x)u_l(x) - v_h(x)u_h(x)) + x^8v_h(x)u_h(x)

    // Split the inputs into the low and high parts.
    let mut lhs_low = [lhs[0], lhs[1], lhs[2], lhs[3]];     // v_l
    let lhs_high = [lhs[4], lhs[5], lhs[6], lhs[7]];        // v_h
    let mut rhs_low = [rhs[0], rhs[1], rhs[2], rhs[3]];     // v_l
    let rhs_high = [rhs[4], rhs[5], rhs[6], rhs[7]];        // v_h


    let prod_low = prod4(&lhs_low, &rhs_low);       // v_l(x)u_l(x)
    let prod_high = prod4(&lhs_high, &rhs_high);    // v_h(x)u_h(x)

    // Copy results into the output.
    output[..7].copy_from_slice(&prod_low);
    output[8..15].copy_from_slice(&prod_high);

    // Add the two halves together, storing the result in lhs_low.
    add_mut(&mut lhs_low, &lhs_high); // v_l + v_h
    add_mut(&mut rhs_low, &rhs_high); // u_l + u_h

    let prod_mix = prod4(&lhs_low, &rhs_low); // (v_l(x) + v_h(x))(u_l(x) + u_h(x))

    // Add the result to the output.
    add_mut(&mut output[4..11], &prod_mix);     // + x^4 (v_l(x) + v_h(x))(u_l(x) + u_h(x))
    sub_mut(&mut output[4..11], &prod_low);     // + x^4 ((v_l(x) + v_h(x))(u_l(x) + u_h(x)) - v_l(x)u_l(x))
    sub_mut(&mut output[4..11], &prod_high);    // + x^4 ((v_l(x) + v_h(x))(u_l(x) + u_h(x)) - v_l(x)u_l(x) - v_h(x)u_h(x))

    output
}

/// Compute the convolution of 2 vectors of length 8.
/// output(x) = lhs(x)rhs(x) mod x^8 - 1  <=>  output = lhs * rhs
/// Use the FFT Trick to split into a convolution of length 4 and a signed convolution of length 4.
#[inline]
fn conv8<T: SimpleInteger>(lhs: &[T; 8], rhs: &[T; 8]) -> [T; 8] {
    const N: usize = 8;
    const HALF: usize = N/2;
    let mut output = [T::default(); N];

    let lhs_p = add_vec(&lhs[..HALF], &lhs[HALF..]); // lhs(x) mod x^4 - 1
    let lhs_m = sub_vec(&lhs[..HALF], &lhs[HALF..]); // lhs(x) mod x^4 + 1

    // rhs will always be constant. Not sure how to tell the compiler this though.
    let rhs_p = add_vec(&rhs[..HALF], &rhs[HALF..]); // rhs(x) mod x^4 - 1   
    let rhs_m = sub_vec(&rhs[..HALF], &rhs[HALF..]); // rhs(x) mod x^4 + 1

    let prod_p = conv4(&lhs_p, &rhs_p);         // prod_p(x) = lhs(x)rhs(x) mod x^4 - 1
    let prod_m = sign_conv4(&lhs_m, &rhs_m);    // prod_m(x) = lhs(x)rhs(x) mod x^4 + 1

    output[..HALF].clone_from_slice(&prod_p);   
    output[HALF..].clone_from_slice(&prod_p);   // output = [prod_p, prod_p]

    add_mut(&mut output[..HALF], &prod_m);
    sub_mut(&mut output[HALF..], &prod_m);      // output = [prod_p + prod_m, prod_p - prod_m] = 2 (lhs * rhs)

    // Could also do this in place?
    output.map(|x| x >> 1) // output = (lhs * rhs)
}

/// Compute the signed convolution of 2 vectors of length 8.
/// output(x) = lhs(x)rhs(x) mod x^8 + 1
/// Use the Karatsuba Method to split into 3 degree 3 polynomial multiplications.
#[inline]
fn signed_conv8<T: SimpleInteger>(lhs: &[T; 8], rhs: &[T; 8]) -> [T; 8] {
    let mut output = [T::default(); 8];

    // The algorithm is simple:
    // v(x)u(x) mod x^8 + 1 = (v_l(x) + x^4v_h(x))(u_l(x) + x^4u_h(x)) mod x^8 + 1 
    //          = v_l(x)u_l(x) - v_h(x)u_h(x) + x^4((v_l(x) + v_h(x))(u_l(x) + u_h(x)) - v_l(x)u_l(x) - v_h(x)u_h(x))

    // There must be a better way to split [T; 8] into 2 [T; 4]'s...
    let mut lhs_low = [lhs[0], lhs[1], lhs[2], lhs[3]];     // v_l
    let lhs_high = [lhs[4], lhs[5], lhs[6], lhs[7]];        // v_h
    let mut rhs_low = [rhs[0], rhs[1], rhs[2], rhs[3]];     // u_l
    let rhs_high = [rhs[4], rhs[5], rhs[6], rhs[7]];        // u_h

    let prod_low = prod4(&lhs_low, &rhs_low);           // v_l(x)u_l(x)
    let prod_high = prod4(&lhs_high, &rhs_high);        // v_h(x)u_h(x)

    // Add the two halves together, storing the result in lhs_low/rhs_low.
    add_mut(&mut lhs_low, &lhs_high);   // v_l + v_h    
    add_mut(&mut rhs_low, &rhs_high);   // u_l + u_h 

    let mut prod_mix = prod4(&lhs_low, &rhs_low);   // (v_l(x) + v_h(x))(u_l(x) + u_h(x))
    sub_mut(&mut prod_mix, &prod_low);
    sub_mut(&mut prod_mix, &prod_high);             // (v_l(x) + v_h(x))(u_l(x) + u_h(x)) - v_l(x)u_l(x) - v_h(x)u_h(x))


    output[..7].copy_from_slice(&prod_low);         // output = v_l(x)u_l(x)
    sub_mut(&mut output[..7], &prod_high);          // output = v_l(x)u_l(x) - v_h(x)u_h(x)
    add_mut(&mut output[4..], &prod_mix[..4]);
    sub_mut(&mut output[..3], &prod_mix[4..]);      // Add prod_mix in 2 parts as x^4prod_mix(x) has monomials >= x^8.

    output
}

/// Compute the convolution of 2 vectors of length 8.
/// output(x) = lhs(x)rhs(x) mod x^8 - 1  <=>  output = lhs * rhs
/// Use the FFT Trick to split into a convolution of length 4 and a signed convolution of length 4.
#[inline]
fn conv16<T: SimpleInteger>(lhs: &[T; 16], rhs: &[T; 16]) -> [T; 16] {
    const N: usize = 16;
    const HALF: usize = N/2;
    let mut output = [T::default(); N];

    let lhs_p = add_vec(&lhs[..HALF], &lhs[HALF..]); // lhs(x) mod x^8 - 1
    let lhs_m = sub_vec(&lhs[..HALF], &lhs[HALF..]); // lhs(x) mod x^8 + 1

    // RHS will always be constant. Not sure how to tell the compiler this though.
    let rhs_p = add_vec(&rhs[..HALF], &rhs[HALF..]); // rhs(x) mod x^8 - 1   
    let rhs_m = sub_vec(&rhs[..HALF], &rhs[HALF..]); // rhs(x) mod x^8 + 1

    let prod_p = conv8(&lhs_p, &rhs_p);           // prod_p(x) = lhs(x)rhs(x) mod x^8 - 1
    let prod_m = signed_conv8(&lhs_m, &rhs_m);    // prod_m(x) = lhs(x)rhs(x) mod x^8 + 1

    output[..HALF].clone_from_slice(&prod_p);   
    output[HALF..].clone_from_slice(&prod_p);   // output = [prod_p, prod_p]

    add_mut(&mut output[..HALF], &prod_m);
    sub_mut(&mut output[HALF..], &prod_m);      // output = [prod_p + prod_m, prod_p - prod_m] = 2 (lhs * rhs)

    // Could also do this in place?
    output.map(|x| x >> 1) // output = (lhs * rhs)
}


/// Compute the signed convolution of 2 vectors of length 16.
/// output(x) = lhs(x)rhs(x) mod x^16 + 1
/// Use the Karatsuba Method to split into 3 degree 3 polynomial multiplications.
#[inline]
fn signed_conv16<T: SimpleInteger>(lhs: &[T; 16], rhs: &[T; 16]) -> [T; 16] {
    let mut output = [T::default(); 16];

    // The algorithm is simple:
    // v(x)u(x) mod x^8 + 1 = (v_l(x) + x^4v_h(x))(u_l(x) + x^4u_h(x)) mod x^8 + 1 
    //          = v_l(x)u_l(x) - v_h(x)u_h(x) + x^4((v_l(x) + v_h(x))(u_l(x) + u_h(x)) - v_l(x)u_l(x) - v_h(x)u_h(x))

    // There must be a better way to split [T; 16] into 2 [T; 8]'s...
    let mut lhs_low = [lhs[0], lhs[1], lhs[2], lhs[3], lhs[4], lhs[5], lhs[6], lhs[7]];     // v_l
    let lhs_high = [lhs[8], lhs[9], lhs[10], lhs[11], lhs[12], lhs[13], lhs[14], lhs[15]];  // v_h
    let mut rhs_low = [rhs[0], rhs[1], rhs[2], rhs[3], rhs[4], rhs[5], rhs[6], rhs[7]];     // u_l
    let rhs_high = [rhs[8], rhs[9], rhs[10], rhs[11], rhs[12], rhs[13], rhs[14], rhs[15]];  // u_h

    let prod_low = prod8(&lhs_low, &rhs_low);       // v_l(x)u_l(x)
    let prod_high = prod8(&lhs_high, &rhs_high);    // v_h(x)u_h(x)

    // Add the two halves together, storing the result in lhs_low.
    add_mut(&mut lhs_low, &lhs_high);   // v_l + v_h
    add_mut(&mut rhs_low, &rhs_high);   // u_l + u_h

    let mut prod_mix = prod8(&lhs_low, &rhs_low);   // (v_l(x) + v_h(x))(u_l(x) + u_h(x))
    sub_mut(&mut prod_mix, &prod_low);
    sub_mut(&mut prod_mix, &prod_high);             // (v_l(x) + v_h(x))(u_l(x) + u_h(x)) - v_l(x)u_l(x) - v_h(x)u_h(x))

    output[..15].copy_from_slice(&prod_low);        // output = v_l(x)u_l(x)     
    sub_mut(&mut output[..15], &prod_high);         // output = v_l(x)u_l(x) - v_h(x)u_h(x)
    add_mut(&mut output[8..], &prod_mix[..8]);
    sub_mut(&mut output[..7], &prod_mix[8..]);      // Add prod_mix in 2 parts as x^8prod_mix(x) has monomials >= x^16.

    output
}

/// Compute the convolution of 2 vectors of length 32.
/// output(x) = lhs(x)rhs(x) mod x^32 - 1  <=>  output = lhs * rhs
/// Use the FFT Trick to split into a convolution of length 16 and a signed convolution of length 16.
#[inline]
fn conv32<T: SimpleInteger>(lhs: &[T; 32], rhs: &[T; 32]) -> [T; 32] {
    const N: usize = 32;
    const HALF: usize = N/2;
    let mut output = [T::default(); N];

    let lhs_p = add_vec(&lhs[..HALF], &lhs[HALF..]);  // lhs(x) mod x^16 - 1
    let lhs_m = sub_vec(&lhs[..HALF], &lhs[HALF..]);  // lhs(x) mod x^16 + 1

    // RHS will always be constant. Not sure how to tell the compiler this though.
    let rhs_p = add_vec(&rhs[..HALF], &rhs[HALF..]);  // rhs(x) mod x^4 - 1  
    let rhs_m = sub_vec(&rhs[..HALF], &rhs[HALF..]);  // rhs(x) mod x^4 + 1

    let prod_p = conv16(&lhs_p, &rhs_p);            // prod_p(x) = lhs(x)rhs(x) mod x^16 - 1
    let prod_m = signed_conv16(&lhs_m, &rhs_m);     // prod_m(x) = lhs(x)rhs(x) mod x^16 + 1

    output[..HALF].clone_from_slice(&prod_p);
    output[HALF..].clone_from_slice(&prod_p);       // output = [prod_p, prod_p]

    add_mut(&mut output[..HALF], &prod_m);
    sub_mut(&mut output[HALF..], &prod_m);          // output = [prod_p + prod_m, prod_p - prod_m] = 2 (lhs * rhs)

    // Could also do this in place?
    output.map(|x| x >> 1) // output = (lhs * rhs)
}

///////////////////////

// Ignore functions below this point. They will be removed or massively changed soon.

///////////////////////


// TODO: UPDATE TO USE THE FFT TRICK
// TODO: MAKE ITERATIVE NOT RECURSIVE
// We start with generic implementations. It will likely be faster long term to specialise these.
// Given lhs (v) and rhs (u) compute the convolution of lhs and rhs recursively as smaller convolutions.
// Decompose v into its even (v_e) and odd (v_o) parts so that v(x) = (v_e(x^2) + xv_o(x^2)).
// Then v(x)u(x) = (v_e(x^2)u_e(x^2) + x^2 v_o(x^2)u_o(x^2))
//                  + x ((v_e(x^2) + v_o(x^2))(u_e(x^2) + u_o(x^2)) - (v_e(x^2)u_e(x^2) + v_o(x^2)u_o(x^2))
// This reduces the problem to 3 convolutions of size N/2 and we can do this recursively.
// Currently this only works for n a power of 2.
// We work with Vector's not arrays as the types at the limbs of match need to line up.
// Would be good to try and fix this.
fn apply_circulant_karat_even_odd(lhs: &[u64], rhs: &[u64]) -> Vec<u64> {
    let n = lhs.len();
    match n {
        1 => [lhs[0] * rhs[0]].try_into().unwrap(),
        2 => [
            lhs[0] * rhs[0] + lhs[1] * rhs[1],
            lhs[1] * rhs[0] + lhs[0] * rhs[1],
        ]
        .try_into()
        .unwrap(),
        4 => conv4(lhs.try_into().unwrap(), rhs.try_into().unwrap())
            .try_into()
            .unwrap(),
        _ => {
            let mut output = vec![0; n];
            let (mut lhs_even, lhs_odd): (Vec<_>, Vec<_>) =
                lhs.iter().enumerate().partition_map(|(ind, val)| {
                    if ind % 2 == 0 {
                        Either::Left(val)
                    } else {
                        Either::Right(val)
                    }
                });
            let (mut rhs_even, rhs_odd): (Vec<_>, Vec<_>) =
                rhs.iter().enumerate().partition_map(|(ind, val)| {
                    if ind % 2 == 0 {
                        Either::Left(val)
                    } else {
                        Either::Right(val)
                    }
                });

            let mut conv_even = apply_circulant_karat_even_odd(&lhs_even, &rhs_even);
            let conv_odd = apply_circulant_karat_even_odd(&lhs_odd, &rhs_odd);

            add_mut(&mut lhs_even, &lhs_odd);
            add_mut(&mut rhs_even, &rhs_odd);

            let mut output_odd = apply_circulant_karat_even_odd(&lhs_even, &rhs_even);
            sub_mut(&mut output_odd, &conv_odd);
            sub_mut(&mut output_odd, &conv_even);

            let half_min_1 = n / 2 - 1;

            add_mut(&mut conv_even[1..], &conv_odd[..half_min_1]);
            conv_even[0] += conv_odd[half_min_1];

            for i in 0..(n / 2) {
                output[2 * i] = conv_even[i];
                output[2 * i + 1] = output_odd[i];
            }
            output
        }
    }
}

// TODO: UPDATE TO USE THE FFT TRICK
// TODO: MAKE ITERATIVE NOT RECURSIVE
// Given lhs (v) and rhs (u) compute the polynomial product v(x)u(x) via the karatsuba method.
// Decompose v = (v_l, v_r) so that v(x) = (v_l(x) + x^{N/2}v_r(x)).
// Then v(x)u(x) = v_l(x)u_l(x) + x^Nv_r(x)u_r(x)
//                  + x^{N/2}((v_l(x) + v_r(x))(u_l(x) + u_r(x)) - (v_l(x)u_l(x) + v_r(x)u_r(x)).
// We work with Vector's not arrays as the types at the limbs of match need to line up.
// Would be good to try and fix this.
fn karatsuba_polynomial_multiplication(lhs: &[u64], rhs: &[u64]) -> Vec<u64> {
    let n = lhs.len();
    match n {
        1 => [lhs[0] * rhs[0]].try_into().unwrap(),
        2 => [
            lhs[0] * rhs[0],
            lhs[1] * rhs[0] + lhs[0] * rhs[1],
            lhs[1] * rhs[1],
        ]
        .try_into()
        .unwrap(),
        4 => prod4(lhs.try_into().unwrap(), rhs.try_into().unwrap())
            .try_into()
            .unwrap(),
        _ => {
            let half = n / 2;
            let mut output = vec![0; 2 * n - 1];

            // Can also do some unsafe rust here as these should never fail.
            let mut lhs_low: Vec<_> = lhs[..half].try_into().unwrap();
            let lhs_high: Vec<_> = lhs[half..].try_into().unwrap();
            let mut rhs_low: Vec<_> = rhs[..half].try_into().unwrap();
            let rhs_high: Vec<_> = rhs[half..].try_into().unwrap();

            let output_low = karatsuba_polynomial_multiplication(&lhs_low, &rhs_low);
            let output_high = karatsuba_polynomial_multiplication(&lhs_high, &rhs_high);

            add_mut(&mut lhs_low, &lhs_high);
            add_mut(&mut rhs_low, &rhs_high);

            let output_mid = karatsuba_polynomial_multiplication(&lhs_low, &rhs_low);

            add_mut(&mut output[..(n - 1)], &output_low);
            add_mut(&mut output[half..(half + n - 1)], &output_mid);
            sub_mut(&mut output[half..(half + n - 1)], &output_low);
            sub_mut(&mut output[half..(half + n - 1)], &output_high);
            add_mut(&mut output[n..], &output_high);

            output
        }
    }
}

// TODO: IMPLEMENT THE FFT TRICK IN THIS CASE.
// THIS FUNCTION WILL BE REMOVED OR GREATLY CHANGED SOON.
// Given inputs lhs and rhs computes the convolution lhs * rhs.
// Uses the odd even decomposition.
#[inline]
fn conv8_eo(lhs: &[u64; 8], rhs: &[u64; 8]) -> [u64; 8] {
    const N: usize = 8;
    let mut output = [0; N];

    let mut lhs_even = [lhs[0], lhs[2], lhs[4], lhs[6]];
    let lhs_odd = [lhs[1], lhs[3], lhs[5], lhs[7]];

    let mut rhs_even = [rhs[0], rhs[2], rhs[4], rhs[6]];
    let rhs_odd = [rhs[1], rhs[3], rhs[5], rhs[7]];

    let mut conv_even = conv4(&lhs_even, &rhs_even);
    let conv_odd = conv4(&lhs_odd, &rhs_odd);

    add_mut(&mut lhs_even, &lhs_odd);
    add_mut(&mut rhs_even, &rhs_odd);
    let mut conv_mix = conv4(&lhs_even, &rhs_even);

    sub_mut(&mut conv_mix, &conv_even);
    sub_mut(&mut conv_mix, &conv_odd);

    add_mut(&mut conv_even[1..], &conv_odd[..3]);
    conv_even[0] += conv_odd[3];

    output[0] = conv_even[0];
    output[1] = conv_mix[0];
    output[2] = conv_even[1];
    output[3] = conv_mix[1];
    output[4] = conv_even[2];
    output[5] = conv_mix[2];
    output[6] = conv_even[3];
    output[7] = conv_mix[3];

    output
}

// TODO: IMPLEMENT THE FFT TRICK IN THIS CASE.
// THIS FUNCTION WILL BE REMOVED OR GREATLY CHANGED SOON.
// Given an input v computes the convolution of v with the constant vector MATRIX_CIRC_MDS_16_SML.
/// Input must be an array of field elements of length 16.
/// Only works with Mersenne31 and Babybear31
/// Uses the odd even decomposition.
pub fn apply_circulant_16_karat_even_odd<F: PrimeField64>(input: [F; 16]) -> [F; 16] {
    const N: usize = 16;
    let mut output = [F::zero(); N];

    let mut lhs_even = [
        input[0].as_canonical_u64(),
        input[2].as_canonical_u64(),
        input[4].as_canonical_u64(),
        input[6].as_canonical_u64(),
        input[8].as_canonical_u64(),
        input[10].as_canonical_u64(),
        input[12].as_canonical_u64(),
        input[14].as_canonical_u64(),
    ];
    let lhs_odd = [
        input[1].as_canonical_u64(),
        input[3].as_canonical_u64(),
        input[5].as_canonical_u64(),
        input[7].as_canonical_u64(),
        input[9].as_canonical_u64(),
        input[11].as_canonical_u64(),
        input[13].as_canonical_u64(),
        input[15].as_canonical_u64(),
    ];

    const MATRIX_CIRC_MDS_16_SML_EVEN_ODD: ([u64; 8], [u64; 8]) = split_half_eo(row_to_col(MATRIX_CIRC_MDS_16_SML));

    // This is clearly a constant but add_vec is currently not a const function.
    let matrix_circ_mds_16_sml_eomix: [u64; 8] =
        add_vec(&MATRIX_CIRC_MDS_16_SML_EVEN_ODD.0, &MATRIX_CIRC_MDS_16_SML_EVEN_ODD.1);

    let mut conv_even = conv8_eo(&lhs_even, &MATRIX_CIRC_MDS_16_SML_EVEN_ODD.0);
    let conv_odd = conv8_eo(&lhs_odd, &MATRIX_CIRC_MDS_16_SML_EVEN_ODD.1);

    // No longer need lhs_even so can reuse the memory.
    add_mut(&mut lhs_even, &lhs_odd);

    let mut output_odd = conv8_eo(&lhs_even, &matrix_circ_mds_16_sml_eomix);
    sub_mut(&mut output_odd, &conv_even);
    sub_mut(&mut output_odd, &conv_odd);

    add_mut(&mut conv_even[1..], &conv_odd[..7]);
    conv_even[0] += conv_odd[7];

    output[0] = F::from_wrapped_u64(conv_even[0]);
    output[1] = F::from_wrapped_u64(output_odd[0]);
    output[2] = F::from_wrapped_u64(conv_even[1]);
    output[3] = F::from_wrapped_u64(output_odd[1]);
    output[4] = F::from_wrapped_u64(conv_even[2]);
    output[5] = F::from_wrapped_u64(output_odd[2]);
    output[6] = F::from_wrapped_u64(conv_even[3]);
    output[7] = F::from_wrapped_u64(output_odd[3]);
    output[8] = F::from_wrapped_u64(conv_even[4]);
    output[9] = F::from_wrapped_u64(output_odd[4]);
    output[10] = F::from_wrapped_u64(conv_even[5]);
    output[11] = F::from_wrapped_u64(output_odd[5]);
    output[12] = F::from_wrapped_u64(conv_even[6]);
    output[13] = F::from_wrapped_u64(output_odd[6]);
    output[14] = F::from_wrapped_u64(conv_even[7]);
    output[15] = F::from_wrapped_u64(output_odd[7]);

    output
}