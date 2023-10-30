use alloc::vec;
use alloc::vec::Vec;
use core::ops::{Add, AddAssign, Mul, MulAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign};
use itertools::Either;
use itertools::Itertools;
use p3_field::PrimeField64;

const MATRIX_CIRC_MDS_8_SML: [i64; 8] = [4, 1, 2, 9, 10, 5, 1, 1];

const MATRIX_CIRC_MDS_12_SML: [i64; 12] = [1, 1, 2, 1, 8, 9, 10, 7, 5, 9, 4, 10];

const MATRIX_CIRC_MDS_16_SML: [i64; 16] =
    [1, 1, 51, 1, 11, 17, 2, 1, 101, 63, 15, 2, 67, 22, 13, 3];

const MATRIX_CIRC_MDS_16_SML_U64: [u64; 16] =
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

const MATRIX_CIRC_MDS_64_MERSENNE31: [i128; 64] = [
    0x570227A5, 0x3702983F, 0x4B7B3B0A, 0x74F13DE3, 0x485314B0, 0x0157E2EC, 0x1AD2E5DE, 0x721515E3,
    0x5452ADA3, 0x0C74B6C1, 0x67DA9450, 0x33A48369, 0x3BDBEE06, 0x7C678D5E, 0x160F16D3, 0x54888B8C,
    0x666C7AA6, 0x113B89E2, 0x2A403CE2, 0x18F9DF42, 0x2A685E84, 0x49EEFDE5, 0x5D044806, 0x560A41F8,
    0x69EF1BD0, 0x2CD15786, 0x62E07766, 0x22A231E2, 0x3CFCF40C, 0x4E8F63D8, 0x69657A15, 0x466B4B2D,
    0x4194B4D2, 0x1E9A85EA, 0x39709C27, 0x4B030BF3, 0x655DCE1D, 0x251F8899, 0x5B2EA879, 0x1E10E42F,
    0x31F5BE07, 0x2AFBB7F9, 0x3E11021A, 0x5D97A17B, 0x6F0620BD, 0x5DBFC31D, 0x76C4761D, 0x21938559,
    0x33777473, 0x71F0E92C, 0x0B9872A1, 0x4C2411F9, 0x545B7C96, 0x20256BAF, 0x7B8B493E, 0x33AD525C,
    0x15EAEA1C, 0x6D2D1A21, 0x06A81D14, 0x3FACEB4F, 0x130EC21C, 0x3C84C4F5, 0x50FD67C0, 0x30FDD85A,
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
{
}

impl<T> SimpleInteger for T where
    T: Sized
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
{
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
    let mut output: [i64; 8] = [0; 8];
    conv8_mut(&input_i64, &MATRIX_CIRC_MDS_8_SML_I64, &mut output);

    // Whilst some intermediate steps may be negative, as we started with 2 positive vectors
    // The output will always be positive so we can safley cast as u64's.
    output.map(|x| F::from_wrapped_u64(x as u64))
}

pub fn apply_circulant_12_karat<F: PrimeField64>(input: [F; 12]) -> [F; 12] {
    // Flip MATRIX_CIRC_MDS_12_SML to get the first column of the circulant matrix.
    const MATRIX_CIRC_MDS_12_SML_I64: [i64; 12] = row_to_col(MATRIX_CIRC_MDS_12_SML);

    // The numbers we will encounter through our algorithm are bounded by
    // SUM(input.as_canonical_u64()) * SUM(MATRIX_CIRC_MDS_8_SML) <= (12 * 2**31) * 33 < 2**40 << 2**63
    // Hence we can work with i64's with garuntees of no overflow occuring.
    let input_i64 = input.map(|x| x.as_canonical_u64() as i64);

    // Compute the convolution.
    // Currently not taking full advantage of MATRIX_CIRC_MDS_8_SML_I64 being constant.
    // TODO: FIX this.
    let output = conv12(&input_i64, &MATRIX_CIRC_MDS_12_SML_I64);

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
    let mut output: [i64; 16] = [0; 16];
    conv16_mut(&MATRIX_CIRC_MDS_16_SML_I64, &input_i64, &mut output);

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
    let mut output: [i128; 32] = [0; 32];
    conv32_mut(&MATRIX_CIRC_MDS_32_M31_I128, &input_i128, &mut output);

    // Whilst some intermediate steps may be negative, as we started with 2 positive vectors
    // The output will always be positive so we can safley cast as u64's.
    output.map(|x| F::from_wrapped_u128(x as u128))
}

/// Computes the convolution of input and MATRIX_CIRC_MDS_64_MERSENNE31.
/// Input must be an array of Mersenne31 field elements of length 64.
pub fn apply_circulant_64_karat<F: PrimeField64>(input: [F; 64]) -> [F; 64] {
    // Flip MATRIX_CIRC_MDS_64_MERSENNE31 to get the first column of the circulant matrix.
    const MATRIX_CIRC_MDS_64_M31_I128: [i128; 64] = row_to_col(MATRIX_CIRC_MDS_64_MERSENNE31);

    // The numbers we will encounter through our algorithm are bounded by
    // SUM(input.as_canonical_u64()) * SUM(MATRIX_CIRC_MDS_64_MERSENNE31) < (64 * 2**31)**2 < 2**74 << 2**127
    // Hence we can work with i128's with garuntees of no overflow occuring.
    let input_i128 = input.map(|x| x.as_canonical_u64() as i128);

    // Compute the convolution.
    // Currently not taking full advantage of MATRIX_CIRC_MDS_16_SML_I64 being constant.
    // TODO: FIX this.
    let mut output: [i128; 64] = [0; 64];
    conv64_mut(&MATRIX_CIRC_MDS_64_M31_I128, &input_i128, &mut output);

    // Whilst some intermediate steps may be negative, as we started with 2 positive vectors
    // The output will always be positive so we can safley cast as u64's.
    output.map(|x| F::from_wrapped_u128(x as u128))
}

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

/// Given a vector v, split it into its odd and even parts.
/// vec = [v_e[0], v_o[0], v_e[1], ...]
/// inverse of interleave
#[inline]
const fn split_half_eo<const N: usize, const HALF_N: usize, T: SimpleInteger>(
    vec: &[T; N],
) -> ([T; HALF_N], [T; HALF_N]) {
    let mut even = [vec[0]; HALF_N];
    let mut odd = [vec[1]; HALF_N];

    let mut i = 1;
    loop {
        even[i] = vec[2 * i];
        odd[i] = vec[2 * i + 1];

        i += 1;
        if i == HALF_N {
            break;
        }
    }
    (even, odd)
}

/// Given two arrays v_e and v_o interleave them into an array twice the size
/// vec = [v_e[0], v_o[0], v_e[1], ...]
/// inverse of split_half_eo.
/// For some reason this is much slower than manually writing out the new array as the interleaving of the old array.
#[inline]
const fn _interleave<const N: usize, const HALF_N: usize, T: SimpleInteger>(
    even: [T; HALF_N],
    odd: [T; HALF_N],
) -> [T; N] {
    let mut vec = [even[0]; N];
    vec[1] = odd[0];

    let mut i = 1;
    loop {
        vec[2 * i] = even[i];
        vec[2 * i + 1] = odd[i];

        i += 1;
        if i == HALF_N {
            break;
        }
    }
    vec
}

// Several functions here to encode some simple vector addition and similar.

// Performs vector addition on slices saving the result in the first slice
#[inline]
fn add_mut<T: SimpleInteger>(lhs: &mut [T], rhs: &[T]) {
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
    let mut output: [T; N] = [T::default(); N];
    let mut i = 0;
    loop {
        output[i] = lhs[i] + rhs[i];
        i += 1;
        if i == N {
            break;
        }
    }
    output
}

// Performs vector subtraction on slices saving the result in the first slice
#[inline]
fn sub_mut<T: SimpleInteger>(lhs: &mut [T], sub: &[T]) {
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
    let mut output: [T; N] = [T::default(); N];
    let mut i = 0;
    loop {
        output[i] = lhs[i] - sub[i];
        i += 1;
        if i == N {
            break;
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
    let mut sum = lhs[0] * rhs[0];
    for i in 1..n {
        sum += lhs[i] * rhs[i];
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

// Option 2: seems to involve less total operations and so should be faster hence it is the once we have implmented for now.
// The main issue is that we are currently doing quite a bit of data manipulation. (Needing to split vectors into odd and even parts and recombining afterwards)
// Would be good to try and find a way to cut down on this.

// Once we get down to small sizes we use the O(n^2) approach.


/////////////////////////////////////////////////////////////////////////////////////////////////////
// Length 3


/// Compute the convolution of two vectors of length 3.
/// output(x) = lhs(x)rhs(x) mod x^3 - 1
#[inline]
fn conv3<T: SimpleInteger>(lhs: &[T; 3], rhs: &[T; 3]) -> [T; 3] {
    // This is small enough we just explicitely write down the answer.
    [
        lhs[0] * rhs[0] + lhs[1] * rhs[2] + lhs[2] * rhs[1],
        lhs[0] * rhs[1] + lhs[1] * rhs[0] + lhs[2] * rhs[2],
        lhs[0] * rhs[2] + lhs[1] * rhs[1] + lhs[2] * rhs[0],
    ]
}

/// Compute the signed convolution of two vectors of length 3.
/// output(x) = lhs(x)rhs(x) mod x^3 + 1
#[inline]
fn sign_conv3<T: SimpleInteger>(lhs: &[T; 3], rhs: &[T; 3]) -> [T; 3] {
    // This is small enough we just explicitely write down the answer.
    [
        lhs[0] * rhs[0] - lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[0] * rhs[1] + lhs[1] * rhs[0] - lhs[2] * rhs[2],
        lhs[0] * rhs[2] + lhs[1] * rhs[1] + lhs[2] * rhs[0],
    ]
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
// Length 4

/// Compute the convolution of two vectors of length 4.
/// output(x) = lhs(x)rhs(x) mod x^4 - 1
#[inline]
fn conv4_mut<T: SimpleInteger>(lhs: &[T; 4], rhs: &[T; 4], output: &mut[T]) {
    // Even at this small size, doing the FFT decomposition seems to produce shorter compiled code using godbolt.
    // In particular testing the code produced for conv8.

    let lhs_p = [lhs[0] + lhs[2], lhs[1] + lhs[3]]; // v_0(x)
    let lhs_m = [lhs[0] - lhs[2], lhs[1] - lhs[3]]; // v_1(x)

    // In most cases, rhs will actually be a constant but I'm unsure how to take advantage of this.
    let rhs_p = [rhs[0] + rhs[2], rhs[1] + rhs[3]]; // u_0(x)
    let rhs_m = [rhs[0] - rhs[2], rhs[1] - rhs[3]]; // u_1(x)


    // Might be worth trying to keep everything as a i64 up until this multiplication and
    // only here making things u128's. (Possible that the compiler has already worked this out though.)
    output[0] = lhs_m[0] * rhs_m[0] - lhs_m[1] * rhs_m[1];
    output[1] = lhs_m[0] * rhs_m[1] + lhs_m[1] * rhs_m[0];  // output[0, 1] = w_1 = v_1(x)u_1(x) mod x^2 + 1
    output[2] = lhs_p[0] * rhs_p[0] + lhs_p[1] * rhs_p[1];
    output[3] = lhs_p[0] * rhs_p[1] + lhs_p[1] * rhs_p[0];  // output[2, 3] = w_0 = v_0(x)u_0(x) mod x^2 - 1

    output[0] += output[2];
    output[1] += output[3]; // output[0, 1] = w_1 + w_0

    output[0] >>= 1;
    output[1] >>= 1; // output[0, 1] = (w_1 + w_0)/2)

    output[2] -= output[0];
    output[3] -= output[1]; // output[2, 3] = w_0 - (w_1 + w_0)/2) = (w_0 - w_1)/2

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
/// output(x) = lhs(x)rhs(x) mod x^4 + 1
#[inline]
fn signed_conv4<T: SimpleInteger>(lhs: &[T; 4], rhs: &[T; 4]) -> [T; 4] {
    let mut output = [T::default(); 4];

    // This might not be the best way to compute this.
    // Another approach is to define
    // [rhs[0], -rhs[3], -rhs[2], -rhs[1]]
    // [rhs[1], rhs[0], -rhs[3], -rhs[2]]
    // [rhs[2], rhs[1], rhs[0], -rhs[3]]
    // [rhs[3], rhs[2], rhs[1], rhs[0]]
    // And then take dot products.
    // Might also be other methods.

    let rhs_rev = [rhs[3], rhs[2], rhs[1], rhs[0]];

    output[0] = lhs[0] * rhs[0] - dot(&lhs[1..], &rhs_rev[..3]); // v_0u_0 - (v_1u_3 + v_2u_2 + v_3u_1)
    output[1] = dot(&lhs[..2], &rhs_rev[2..]) - dot(&lhs[2..], &rhs_rev[..2]); // v_0u_1 + v_1u_0 - (v_2u_3 + v_2u_3)
    output[2] = dot(&lhs[..3], &rhs_rev[1..]) - lhs[3] * rhs[3]; // v_0u_2 + v_1u_1 + v_2u_0 - v_3u_3
    output[3] = dot(lhs, &rhs_rev); // v_0u_3 + v_1u_2 + v_2u_1 + v_3u_0
    output

    // It may be possible to choose the MDS vector such that rhs = [1, 1, 1, 1] or some constant version of this.
    // In such cases, we can use faster code for signed_conv4. Just saving this here for now
    // let output1 = lhs[0] + lhs[1] - lhs[2] - lhs[3];
    // let output2 = output1 + (lhs[2] << 1);
    // [
    //     output1 - (lhs[1] << 1),
    //     output1,
    //     output2,
    //     output2 + (lhs[3] << 1),
    // ]
}

/// Compute the signed convolution of two vectors of length 4.
/// output(x) = lhs(x)rhs(x) mod x^4 + 1
/// Same as above but in place.
#[inline]
fn signed_conv4_mut<T: SimpleInteger>(lhs: &[T; 4], rhs: &[T; 4], output: &mut [T]) {

    let rhs_rev = [rhs[3], rhs[2], rhs[1], rhs[0]];

    output[0] = lhs[0] * rhs[0] - dot(&lhs[1..], &rhs_rev[..3]); // v_0u_0 - (v_1u_3 + v_2u_2 + v_3u_1)
    output[1] = dot(&lhs[..2], &rhs_rev[2..]) - dot(&lhs[2..], &rhs_rev[..2]); // v_0u_1 + v_1u_0 - (v_2u_3 + v_2u_3)
    output[2] = dot(&lhs[..3], &rhs_rev[1..]) - lhs[3] * rhs[3]; // v_0u_2 + v_1u_1 + v_2u_0 - v_3u_3
    output[3] = dot(lhs, &rhs_rev); // v_0u_3 + v_1u_2 + v_2u_1 + v_3u_0
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
// Length 6

/// Compute the convolution of two vectors of length 6.
/// output(x) = lhs(x)rhs(x) mod x^6 - 1
#[inline]
fn conv6<T: SimpleInteger>(lhs: &[T; 6], rhs: &[T; 6]) -> [T; 6] {
    // Even at this small size, doing the FFT decomposition seems to produce shorter compiled code using godbolt.
    // In particular testing the code produced for conv12 as that's what we really care about.

    const N: usize = 6;
    const HALF: usize = N / 2;
    let mut output = [T::default(); N];

    let lhs_p: [T; HALF] = add_vec(&lhs[..HALF], &lhs[HALF..]); // v_0(x)
    let lhs_m: [T; HALF] = sub_vec(&lhs[..HALF], &lhs[HALF..]); // v_1(x)

    // In most cases, rhs will actually be a constant but I'm unsure how to take advantage of this.
    let rhs_p: [T; HALF] = add_vec(&rhs[..HALF], &rhs[HALF..]); // u_0(x)
    let rhs_m: [T; HALF] = sub_vec(&rhs[..HALF], &rhs[HALF..]); // u_1(x)

    let prod_p = conv3(&lhs_p, &rhs_p); // prod_p(x) = lhs(x)rhs(x) mod x^4 - 1
    let prod_m = sign_conv3(&lhs_m, &rhs_m); // prod_m(x) = lhs(x)rhs(x) mod x^4 + 1

    output[..HALF].clone_from_slice(&prod_p);
    output[HALF..].clone_from_slice(&prod_p); // output = [prod_p, prod_p]

    add_mut(&mut output[..HALF], &prod_m);
    sub_mut(&mut output[HALF..], &prod_m); // output = [prod_p + prod_m, prod_p - prod_m] = 2 (lhs * rhs)

    // Can maybe do this in place?
    output.map(|x| x >> 1) // output = lhs * rhs
}

/// Compute the signed convolution of two vectors of length 6.
/// output(x) = lhs(x)rhs(x) mod x^6 + 1
#[inline]
fn sign_conv6<T: SimpleInteger>(lhs: &[T; 6], rhs: &[T; 6]) -> [T; 6] {
    let mut output = [T::default(); 6];

    // This might not be the best way to compute this.

    let rhs_rev = [rhs[5], rhs[4], rhs[3], rhs[2], rhs[1], rhs[0]];

    output[0] = lhs[0] * rhs[0] - dot(&lhs[1..], &rhs_rev[..5]);
    output[1] = dot(&lhs[..2], &rhs_rev[4..]) - dot(&lhs[2..], &rhs_rev[..4]);
    output[2] = dot(&lhs[..3], &rhs_rev[3..]) - dot(&lhs[3..], &rhs_rev[..3]);
    output[3] = dot(&lhs[..4], &rhs_rev[2..]) - dot(&lhs[4..], &rhs_rev[..2]);
    output[4] = dot(&lhs[..5], &rhs_rev[1..]) - lhs[5] * rhs[5];
    output[5] = dot(lhs, &rhs_rev);
    output
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
// Length 8

/// Compute the convolution of 2 vectors of length 8.
/// output(x) = lhs(x)rhs(x) mod x^8 - 1  <=>  output = lhs * rhs
/// Use the FFT Trick to split into a convolution of length 4 and a signed convolution of length 4.
#[inline]
fn conv8_mut<T: SimpleInteger>(lhs: &[T; 8], rhs: &[T; 8], output: &mut [T]) {
    const N: usize = 8;
    const HALF: usize = N / 2;

    let (lhs_left, lhs_right) = lhs.split_at(HALF);

    let lhs_p = add_vec(lhs_left, lhs_right); // lhs(x) mod x^4 - 1
    let lhs_m = sub_vec(lhs_left, lhs_right); // lhs(x) mod x^4 + 1

    // rhs will always be constant. Not sure how to tell the compiler this though.
    let (rhs_left, rhs_right) = rhs.split_at(HALF);

    let rhs_p = add_vec(rhs_left, rhs_right); // rhs(x) mod x^4 - 1
    let rhs_m = sub_vec(rhs_left, rhs_right); // rhs(x) mod x^4 + 1

    let (left, right) = output.split_at_mut(HALF);

    signed_conv4_mut(&lhs_m, &rhs_m, left); // left = w_1 = lhs*rhs mod x^4 + 1
    conv4_mut(&lhs_p, &rhs_p, right); // right = w_0 = lhs*rhs mod x^4 - 1

    for i in 0..HALF {
        left[i] += right[i]; // w_0 + w_1
        left[i] >>= 1; // (w_0 + w_1)/2
        right[i] -= left[i]; // (w_0 - w_1)/2
    }
}

/// Compute the signed convolution of 2 vectors of length 8.
/// output(x) = lhs(x)rhs(x) mod x^8 + 1
/// Use the Karatsuba Method to split into 3 degree 3 polynomial multiplications.
#[inline]
fn signed_conv8<T: SimpleInteger>(lhs: &[T; 8], rhs: &[T; 8]) -> [T; 8] {
    const N: usize = 8;
    const HALF: usize = N / 2;

    // The algorithm is relatively simple:
    // v(x)u(x) mod x^8 + 1 = (v_e(x^2) + xv_o(x^2))(u_e(x^2) + xu_o(x^2)) mod x^8 + 1
    //          = v_e(x^2)u_e(x^2) + x^2 v_o(x^2)u_o(x^2) + x((v_e(x^2) + v_o(x^2))(u_e(x^2) + u_o(x^2)) - v_e(x^2)u_e(x^2) - v_o(x^2)u_o(x^2))

    // Now computing v_e(x^2)u_e(x^2) mod x^8 + 1 is equivalent to computing v_e(x)u_e(x) mod x^4 + 1 and similarly for the other products.

    // Clearly there should be a cleaner way to get this decomposition but everything I've tried has been slower.
    // Also seems like we are doing quite a bit of data fiddiling. Would be nice to avoid this.
    let mut lhs_even = [lhs[0], lhs[2], lhs[4], lhs[6]]; // v_e
    let lhs_odd = [lhs[1], lhs[3], lhs[5], lhs[7]]; // v_o
    let mut rhs_even = [rhs[0], rhs[2], rhs[4], rhs[6]]; // u_e
    let rhs_odd = [rhs[1], rhs[3], rhs[5], rhs[7]]; // u_o

    let mut prod_even = signed_conv4(&lhs_even, &rhs_even); // v_e(x)u_e(x) mod x^4 + 1
    let prod_odd = signed_conv4(&lhs_odd, &rhs_odd); // v_o(x)u_o(x) mod x^4 + 1

    // Add the two halves together, storing the result in lhs_even/rhs_even.
    add_mut(&mut lhs_even, &lhs_odd); // v_e + v_o
    add_mut(&mut rhs_even, &rhs_odd); // u_e + u_o

    let mut prod_mix = signed_conv4(&lhs_even, &rhs_even); // (v_e(x) + v_o(x))(u_e(x) + u_o(x))
    sub_mut(&mut prod_mix, &prod_even);
    sub_mut(&mut prod_mix, &prod_odd); // (v_e(x) + v_o(x))(u_e(x) + u_o(x)) - v_e(x)u_e(x) - v_o(x)u_o(x)

    add_mut(&mut prod_even[1..], &prod_odd[..(HALF - 1)]);
    prod_even[0] -= prod_odd[HALF - 1]; // v_e(x)u_e(x) + xv_o(x)u_o(x) mod x^4 + 1

    [
        prod_even[0],
        prod_mix[0],
        prod_even[1],
        prod_mix[1],
        prod_even[2],
        prod_mix[2],
        prod_even[3],
        prod_mix[3],
    ] // Intertwining the result. Again this is some annoying data fiddiling. Must be a way to avoid some of this.
}


/////////////////////////////////////////////////////////////////////////////////////////////////////
// Length 12


/// Compute the convolution of 2 vectors of length 8.
/// output(x) = lhs(x)rhs(x) mod x^12 - 1  <=>  output = lhs * rhs
/// Use the FFT Trick to split into a convolution of length 6 and a signed convolution of length 6.
#[inline]
fn conv12<T: SimpleInteger>(lhs: &[T; 12], rhs: &[T; 12]) -> [T; 12] {
    const N: usize = 12;
    const HALF: usize = N / 2;
    let mut output = [T::default(); N];

    let lhs_p = add_vec(&lhs[..HALF], &lhs[HALF..]); // lhs(x) mod x^6 - 1
    let lhs_m = sub_vec(&lhs[..HALF], &lhs[HALF..]); // lhs(x) mod x^6 + 1

    // rhs will always be constant. Not sure how to tell the compiler this though.
    let rhs_p = add_vec(&rhs[..HALF], &rhs[HALF..]); // rhs(x) mod x^6 - 1
    let rhs_m = sub_vec(&rhs[..HALF], &rhs[HALF..]); // rhs(x) mod x^6 + 1

    let prod_p = conv6(&lhs_p, &rhs_p); // prod_p(x) = lhs(x)rhs(x) mod x^6 - 1
    let prod_m = sign_conv6(&lhs_m, &rhs_m); // prod_m(x) = lhs(x)rhs(x) mod x^6 + 1

    output[..HALF].clone_from_slice(&prod_p);
    output[HALF..].clone_from_slice(&prod_p); // output = [prod_p, prod_p]

    add_mut(&mut output[..HALF], &prod_m);
    sub_mut(&mut output[HALF..], &prod_m); // output = [prod_p + prod_m, prod_p - prod_m] = 2 (lhs * rhs)

    // Could also do this in place?
    output.map(|x| x >> 1) // output = (lhs * rhs)
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
// Length 16

/// Compute the convolution of 2 vectors of length 8.
/// output(x) = lhs(x)rhs(x) mod x^8 - 1  <=>  output = lhs * rhs
/// Use the FFT Trick to split into a convolution of length 4 and a signed convolution of length 4.
#[inline]
fn conv16_mut<T: SimpleInteger>(lhs: &[T; 16], rhs: &[T; 16], output: &mut [T]) {
    const N: usize = 16;
    const HALF: usize = N / 2;

    let (lhs_left, lhs_right) = lhs.split_at(HALF);

    let lhs_p = add_vec(lhs_left, lhs_right); // lhs(x) mod x^8 - 1
    let lhs_m = sub_vec(lhs_left, lhs_right); // lhs(x) mod x^8 + 1

    // rhs will always be constant. Not sure how to tell the compiler this though.
    let (rhs_left, rhs_right) = rhs.split_at(HALF);

    let rhs_p = add_vec(rhs_left, rhs_right); // rhs(x) mod x^8 - 1
    let rhs_m = sub_vec(rhs_left, rhs_right); // rhs(x) mod x^8 + 1

    let (left, right) = output.split_at_mut(HALF);
    left.clone_from_slice(&signed_conv8(&lhs_m, &rhs_m)); // left = w_1 = lhs*rhs mod x^8 + 1
    conv8_mut(&lhs_p, &rhs_p, right); // right = w_0 = lhs*rhs mod x^8 - 1
    for i in 0..HALF {
        left[i] += right[i]; // w_0 + w_1
        left[i] >>= 1; // (w_0 + w_1)/2
        right[i] -= left[i]; // (w_0 - w_1)/2
    }
}

/// Compute the signed convolution of 2 vectors of length 16.
/// output(x) = lhs(x)rhs(x) mod x^16 + 1
/// Use the Karatsuba Method to split into 3 degree 3 polynomial multiplications.
#[inline]
fn signed_conv16<T: SimpleInteger>(lhs: &[T; 16], rhs: &[T; 16]) -> [T; 16] {
    const N: usize = 16;
    const HALF: usize = N / 2;

    // The algorithm is relatively simple:
    // v(x)u(x) mod x^16 + 1 = (v_e(x^2) + xv_o(x^2))(u_e(x^2) + xu_o(x^2)) mod x^16 + 1
    //          = v_e(x^2)u_e(x^2) + x^2 v_o(x^2)u_o(x^2) + x((v_e(x^2) + v_o(x^2))(u_e(x^2) + u_o(x^2)) - v_e(x^2)u_e(x^2) - v_o(x^2)u_o(x^2))

    // Now computing v_e(x^2)u_e(x^2) mod x^16 + 1 is equivalent to computing v_e(x)u_e(x) mod x^8 + 1 and similarly for the other products.

    // Clearly there should be a cleaner way to get this decomposition but everything I've tried has been slower.
    // Also seems like we are doing quite a bit of data fiddiling. Would be nice to avoid this.
    let mut lhs_even = [
        lhs[0], lhs[2], lhs[4], lhs[6], lhs[8], lhs[10], lhs[12], lhs[14], // v_e
    ];
    let lhs_odd = [
        lhs[1], lhs[3], lhs[5], lhs[7], lhs[9], lhs[11], lhs[13], lhs[15], // v_o
    ];
    let mut rhs_even = [
        rhs[0], rhs[2], rhs[4], rhs[6], rhs[8], rhs[10], rhs[12], rhs[14], // u_e
    ];
    let rhs_odd = [
        rhs[1], rhs[3], rhs[5], rhs[7], rhs[9], rhs[11], rhs[13], rhs[15], // u_o
    ];

    let mut prod_even = signed_conv8(&lhs_even, &rhs_even); // v_e(x)u_e(x) mod x^8 + 1
    let prod_odd = signed_conv8(&lhs_odd, &rhs_odd); // v_o(x)u_o(x) mod x^8 + 1

    // Add the two halves together, storing the result in lhs_even/rhs_even.
    add_mut(&mut lhs_even, &lhs_odd); // v_e + v_o
    add_mut(&mut rhs_even, &rhs_odd); // u_e + u_o

    let mut prod_mix = signed_conv8(&lhs_even, &rhs_even); // (v_e(x) + v_o(x))(u_e(x) + u_o(x))
    sub_mut(&mut prod_mix, &prod_even);
    sub_mut(&mut prod_mix, &prod_odd); // (v_e(x) + v_o(x))(u_e(x) + u_o(x)) - v_e(x)u_e(x) - v_o(x)u_o(x)

    add_mut(&mut prod_even[1..], &prod_odd[..(HALF - 1)]);
    prod_even[0] -= prod_odd[HALF - 1]; // v_e(x)u_e(x) + xv_o(x)u_o(x) mod x^8 + 1

    [
        prod_even[0],
        prod_mix[0],
        prod_even[1],
        prod_mix[1],
        prod_even[2],
        prod_mix[2],
        prod_even[3],
        prod_mix[3],
        prod_even[4],
        prod_mix[4],
        prod_even[5],
        prod_mix[5],
        prod_even[6],
        prod_mix[6],
        prod_even[7],
        prod_mix[7],
    ] // Intertwining the result. Again this is some annoying data fiddiling. Must be a way to avoid some of this.
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
// Length 32

/// Compute the convolution of 2 vectors of length 32.
/// output(x) = lhs(x)rhs(x) mod x^32 - 1  <=>  output = lhs * rhs
/// Use the FFT Trick to split into a convolution of length 16 and a signed convolution of length 16.
#[inline]
fn conv32_mut<T: SimpleInteger>(lhs: &[T; 32], rhs: &[T; 32], output: &mut [T]) {
    const N: usize = 32;
    const HALF: usize = N / 2;

    let (lhs_left, lhs_right) = lhs.split_at(HALF);

    let lhs_p = add_vec(lhs_left, lhs_right); // lhs(x) mod x^16 - 1
    let lhs_m = sub_vec(lhs_left, lhs_right); // lhs(x) mod x^16 + 1

    // rhs will always be constant. Not sure how to tell the compiler this though.
    let (rhs_left, rhs_right) = rhs.split_at(HALF);

    let rhs_p = add_vec(rhs_left, rhs_right); // rhs(x) mod x^16 - 1
    let rhs_m = sub_vec(rhs_left, rhs_right); // rhs(x) mod x^16 + 1

    let (left, right) = output.split_at_mut(HALF);
    left.clone_from_slice(&signed_conv16(&lhs_m, &rhs_m)); // left = w_1 = lhs*rhs mod x^16 + 1
    conv16_mut(&lhs_p, &rhs_p, right); // right = w_0 = lhs*rhs mod x^16 - 1
    for i in 0..HALF {
        left[i] += right[i]; // w_0 + w_1
        left[i] >>= 1; // (w_0 + w_1)/2
        right[i] -= left[i]; // (w_0 - w_1)/2
    }
}

/// Compute the signed convolution of 2 vectors of length 16.
/// output(x) = lhs(x)rhs(x) mod x^16 + 1
/// Use the Karatsuba Method to split into 3 degree 3 polynomial multiplications.
#[inline]
fn signed_conv32<T: SimpleInteger>(lhs: &[T; 32], rhs: &[T; 32]) -> [T; 32] {
    const N: usize = 32;
    const HALF: usize = N / 2;

    // The algorithm is simple:
    // v(x)u(x) mod x^32 + 1 = (v_l(x) + x^4v_h(x))(u_l(x) + x^4u_h(x)) mod x^32 + 1
    //          = v_l(x)u_l(x) - v_h(x)u_h(x) + x^4((v_l(x) + v_h(x))(u_l(x) + u_h(x)) - v_l(x)u_l(x) - v_h(x)u_h(x))

    // Now computing v_e(x^2)u_e(x^2) mod x^32 + 1 is equivalent to computing v_e(x)u_e(x) mod x^16 + 1 and similarly for the other products.

    // Clearly there should be a cleaner way to get this decomposition but everything I've tried has been slower.
    // Also seems like we are doing quite a bit of data fiddiling. Would be nice to avoid this.
    let mut lhs_even = [
        lhs[0], lhs[2], lhs[4], lhs[6], lhs[8], lhs[10], lhs[12], lhs[14], lhs[16],
        lhs[18], // v_e
        lhs[20], lhs[22], lhs[24], lhs[26], lhs[28], lhs[30],
    ];
    let lhs_odd = [
        lhs[1], lhs[3], lhs[5], lhs[7], lhs[9], lhs[11], lhs[13], lhs[15], lhs[17],
        lhs[19], // v_o
        lhs[21], lhs[23], lhs[25], lhs[27], lhs[29], lhs[31],
    ];
    let mut rhs_even = [
        rhs[0], rhs[2], rhs[4], rhs[6], rhs[8], rhs[10], rhs[12], rhs[14], rhs[16],
        rhs[18], // u_e
        rhs[20], rhs[22], rhs[24], rhs[26], rhs[28], rhs[30],
    ];
    let rhs_odd = [
        rhs[1], rhs[3], rhs[5], rhs[7], rhs[9], rhs[11], rhs[13], rhs[15], rhs[17],
        rhs[19], // u_o
        rhs[21], rhs[23], rhs[25], rhs[27], rhs[29], rhs[31],
    ];

    let mut prod_even = signed_conv16(&lhs_even, &rhs_even); // v_e(x)u_e(x) mod x^16 + 1
    let prod_odd = signed_conv16(&lhs_odd, &rhs_odd); // v_o(x)u_o(x) mod x^16 + 1

    // Add the two halves together, storing the result in lhs_even/rhs_even.
    add_mut(&mut lhs_even, &lhs_odd); // v_e + v_o
    add_mut(&mut rhs_even, &rhs_odd); // u_e + u_o

    let mut prod_mix = signed_conv16(&lhs_even, &rhs_even); // (v_e(x) + v_o(x))(u_e(x) + u_o(x)) mod x^16 + 1
    sub_mut(&mut prod_mix, &prod_even);
    sub_mut(&mut prod_mix, &prod_odd); // (v_e(x) + v_o(x))(u_e(x) + u_o(x)) - v_e(x)u_e(x) - v_o(x)u_o(x)

    add_mut(&mut prod_even[1..], &prod_odd[..(HALF - 1)]);
    prod_even[0] -= prod_odd[HALF - 1]; // v_e(x)u_e(x) + xv_o(x)u_o(x) mod x^16 + 1

    [
        prod_even[0],
        prod_mix[0],
        prod_even[1],
        prod_mix[1],
        prod_even[2],
        prod_mix[2],
        prod_even[3],
        prod_mix[3],
        prod_even[4],
        prod_mix[4],
        prod_even[5],
        prod_mix[5],
        prod_even[6],
        prod_mix[6],
        prod_even[7],
        prod_mix[7],
        prod_even[8],
        prod_mix[8],
        prod_even[9],
        prod_mix[9],
        prod_even[10],
        prod_mix[10],
        prod_even[11],
        prod_mix[11],
        prod_even[12],
        prod_mix[12],
        prod_even[13],
        prod_mix[13],
        prod_even[14],
        prod_mix[14],
        prod_even[15],
        prod_mix[15],
    ] // Intertwining the result. Again this is some annoying data fiddiling. Must be a way to avoid some of this.
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
// Length 64

/// Compute the convolution of 2 vectors of length 64.
/// output(x) = lhs(x)rhs(x) mod x^64 - 1  <=>  output = lhs * rhs
/// Use the FFT Trick to split into a convolution of length 32 and a signed convolution of length 32.
#[inline]
fn conv64_mut<T: SimpleInteger>(lhs: &[T; 64], rhs: &[T; 64], output: &mut[T]) {
    const N: usize = 64;
    const HALF: usize = N / 2;

    let (lhs_left, lhs_right) = lhs.split_at(HALF);

    let lhs_p = add_vec(lhs_left, lhs_right); // lhs(x) mod x^32 - 1
    let lhs_m = sub_vec(lhs_left, lhs_right); // lhs(x) mod x^32 + 1

    // rhs will always be constant. Not sure how to tell the compiler this though.
    let (rhs_left, rhs_right) = rhs.split_at(HALF);

    let rhs_p = add_vec(rhs_left, rhs_right); // rhs(x) mod x^32 - 1
    let rhs_m = sub_vec(rhs_left, rhs_right); // rhs(x) mod x^32 + 1

    let (left, right) = output.split_at_mut(HALF);
    left.clone_from_slice(&signed_conv32(&lhs_m, &rhs_m)); // left = w_1 = lhs*rhs mod x^32 + 1
    conv32_mut(&lhs_p, &rhs_p, right); // right = w_0 = lhs*rhs mod x^32 - 1
    for i in 0..HALF {
        left[i] += right[i]; // w_0 + w_1
        left[i] >>= 1; // (w_0 + w_1)/2
        right[i] -= left[i]; // (w_0 - w_1)/2
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Ignore functions below this point. They will be removed or massively changed soon.

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

/// Leaving this here for the moment as it is used by the following function.
/// Will remove this eventually.
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

/// Compute the convolution of two vectors of length 4.
/// output(x) = lhs(x)rhs(x) mod x^4 - 1
#[inline]
fn conv4<T: SimpleInteger>(lhs: &[T; 4], rhs: &[T; 4]) -> [T; 4] {
    // Even at this small size, doing the FFT decomposition seems to produce shorter compiled code using godbolt.
    // In particular testing the code produced for conv8.

    const N: usize = 4;
    const HALF: usize = N / 2;
    let mut output = [T::default(); N];

    let lhs_p: [T; HALF] = add_vec(&lhs[..HALF], &lhs[HALF..]); // v_0(x)
    let lhs_m: [T; HALF] = sub_vec(&lhs[..HALF], &lhs[HALF..]); // v_1(x)

    // In most cases, rhs will actually be a constant but I'm unsure how to take advantage of this.
    let rhs_p: [T; HALF] = add_vec(&rhs[..HALF], &rhs[HALF..]); // u_0(x)
    let rhs_m: [T; HALF] = sub_vec(&rhs[..HALF], &rhs[HALF..]); // u_1(x)

    output[0] = lhs_p[0] * rhs_p[0] + lhs_p[1] * rhs_p[1];
    output[1] = lhs_p[0] * rhs_p[1] + lhs_p[1] * rhs_p[0]; // output = v_0(x)u_0(x) mod x^2 - 1
    output[2] = output[0];
    output[3] = output[1]; // output = (1 + x^2)(v_0(x)u_0(x) mod x^2 - 1)

    let product_m = [
        lhs_m[0] * rhs_m[0] - lhs_m[1] * rhs_m[1],
        lhs_m[0] * rhs_m[1] + lhs_m[1] * rhs_m[0],
    ];

    add_mut(&mut output[..HALF], &product_m);
    sub_mut(&mut output[HALF..], &product_m); // output = 2 (lhs * rhs)

    // Can maybe do this in place?
    output.map(|x| x >> 1) // output = lhs * rhs

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

    const MATRIX_CIRC_MDS_16_SML_EVEN_ODD: ([u64; 8], [u64; 8]) =
        split_half_eo(&row_to_col(MATRIX_CIRC_MDS_16_SML_U64));

    // This is clearly a constant but add_vec is currently not a const function.
    let matrix_circ_mds_16_sml_eomix: [u64; 8] = add_vec(
        &MATRIX_CIRC_MDS_16_SML_EVEN_ODD.0,
        &MATRIX_CIRC_MDS_16_SML_EVEN_ODD.1,
    );

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
