use alloc::vec;
use alloc::vec::Vec;
use itertools::Either;
use itertools::Itertools;
use p3_field::PrimeField64;

// const MATRIX_CIRC_MDS_8_SML: [u64; 8] = [4, 1, 1, 5, 10, 9, 2, 1];

// const MATRIX_CIRC_MDS_16_SML: [u64; 16] =
//     [1, 3, 13, 22, 67, 2, 15, 63, 101, 1, 2, 17, 11, 1, 51, 1];

const MATRIX_CIRC_MDS_8_SML: [u64; 8] = [4, 1, 2, 9, 10, 5, 1, 1];

const MATRIX_CIRC_MDS_16_SML: [u64; 16] =
    [1, 1, 51, 1, 11, 17, 2, 1, 101, 63, 15, 2, 67, 22, 13, 3];


// Let M be a circulant matrix with first column vec_col and first row vec_row. Then M.u is the convolution of vec_col and u.
// The vectors given here are the first rows of the respective circulant matrices NOT the first colums.
// Hence in order to do convolutions we need to compute the first column which is given by
// vec_col = [vec_row[0], vec_row[n - 1], vec_row[n - 2], ..., vec_row[1]]
#[inline]
const fn row_to_col<const N: usize>(row: [u64; N]) -> [u64; N] {
    let mut col = [0; N];
    col[0] = row[0];
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

// Given a vector v, split it into the left and right parts
#[inline]
const fn split_half_lr<const N: usize, const HALF_N: usize>(
    vec: [u64; N],
) -> ([u64; HALF_N], [u64; HALF_N]) {
    let mut lhs = [0; HALF_N];
    let mut rhs = [0; HALF_N];

    let mut i = 0;
    loop {
        if i == HALF_N {
            break;
        }
        lhs[i] = vec[i];
        rhs[i] = vec[i + HALF_N];
        i += 1
    }
    (lhs, rhs)
}

// Given a vector v, split it into its odd and even parts.
#[inline]
const fn split_half_eo<const N: usize, const HALF_N: usize>(
    vec: [u64; N],
) -> ([u64; HALF_N], [u64; HALF_N]) {
    let mut even = [0; HALF_N];
    let mut odd = [0; HALF_N];

    let mut i = 0;
    loop {
        if i == HALF_N {
            break;
        }
        even[i] = vec[2*i];
        odd[i] = vec[2*i + 1];
        i += 1
    }
    (even, odd)
}

// Several functions here to encode some simple vector addition and similar.

// Performs vector addition on slices saving the result in the first slice
#[inline]
fn add_mut(lhs: &mut [u64], rhs: &[u64]) {
    let n = rhs.len();
    for i in 0..n {
        lhs[i] += rhs[i]
    }
}

// Performs vector subtraction on slices saving the result in the first slice
#[inline]
fn sub_mut(lhs: &mut [u64], sub: &[u64]) {
    let n = sub.len();
    for i in 0..n {
        lhs[i] -= sub[i]
    }
}

// Performs the vector operation lhs += val*rhs
#[inline]
fn add_mul_mut(lhs: &mut [u64], rhs: &[u64], val: &u64) {
    let n = rhs.len();
    for i in 0..n {
        lhs[i] += val * rhs[i]
    }
}

// Takes the dot product of two vectors.
#[inline]
fn dot(lhs: &[u64], rhs: &[u64]) -> u64 {
    let n = lhs.len();
    let mut sum = lhs[0] * rhs[0];
    for i in 1..n {
        sum += lhs[i] * rhs[i];
    }
    sum
}

// Given a vector v \in F^N, let v(x) \in F[X] denote the polynomial v_0 + v_1 x + ... + v_{N - 1} x^{N - 1}.
// Then w is equal to the convolution v * u if and only if w(x) = v(x)u(x) in the ring F[X]/(x^N - 1).
// Hence we can use fast polynomial multiplication algorithms to compute convolutions.

// There are 2 possible approaches which mirror the DIT vs DIF approaches to FFT's.
// For now assume that v and u have length N = 2^n though these methids can be easily adapted for other N.

// Option 1: left/right decomposition.
// Write v = (v_l, v_r) so that v(x) = (v_l(x) + x^{N/2}v_r(x)).
// Then v(x)u(x) = (v_l(x)u_l(x) + v_r(x)u_r(x)) + x^{N/2}((v_l(x) + v_r(x))(u_l(x) + u_r(x)) - (v_l(x)u_l(x) + v_r(x)u_r(x))

// As v_l(x), v_r(x), u_l(x), u_r(x) all have degree < N/2 no product will have degree > N - 1.
// Hence this reduces the problem to 3 polynomial multiplications of size N/2 and we can use the standard karatsuba recursively for this.

// Option 2: even/odd decomposition.
// Define the even v_e and odd v_o parts so that v(x) = (v_e(x^2) + xv_o(x^2)).
// Then v(x)u(x) = (v_e(x^2)u_e(x^2) + x^2 v_o(x^2)u_o(x^2)) + x ((v_e(x^2) + v_o(x^2))(u_e(x^2) + u_o(x^2)) - (v_e(x^2)u_e(x^2) + v_o(x^2)u_o(x^2))
// This reduces the problem to 3 convolutions of size N/2 and we can do this recursively.

// These algorithms do look a little different so it's not immediately obvious which is the right one to compute.
// Thus for now we will implement both.

// Once we get down to size 4/8 we use the O(n^2) approach.

// Compute the product of two polynomials of degree 3.
// This will be one of our base cases.
#[inline]
fn prod4(lhs: &[u64; 4], rhs: &[u64; 4]) -> [u64; 7] {
    let mut output = [0; 7];
    add_mul_mut(&mut output[..4], lhs, &rhs[0]);
    add_mul_mut(&mut output[1..5], lhs, &rhs[1]);
    add_mul_mut(&mut output[2..6], lhs, &rhs[2]);
    add_mul_mut(&mut output[3..], lhs, &rhs[3]);
    output
}

// Compute the convolution product of two vectors of length 4.
// This will be one of our base cases.
fn conv4(lhs: &[u64; 4], rhs: &[u64; 4]) -> [u64; 4] {
    let mut output = [0; 4];
    let rhs_rev_0 = [rhs[0], rhs[3], rhs[2], rhs[1]];
    output[0] = dot(lhs, &rhs_rev_0);
    let rhs_rev_1 = [rhs[1], rhs[0], rhs[3], rhs[2]];
    output[1] = dot(lhs, &rhs_rev_1);
    let rhs_rev_2 = [rhs[2], rhs[1], rhs[0], rhs[3]];
    output[2] = dot(lhs, &rhs_rev_2);
    let rhs_rev_3 = [rhs[3], rhs[2], rhs[1], rhs[0]];
    output[3] = dot(lhs, &rhs_rev_3);
    output
}

// Assume that lhs is a vector of u64's and rhs is a vector of field elements.
// This computes M(lhs).rhs where M is the circulant matrix with first row equal to lhs.
// It uses the left/right decomposition.
pub fn apply_circulant_karat_left_right_field<F: PrimeField64, const N: usize>(
    lhs: [u64; N],
    rhs: [F; N],
) -> [F; N] {
    // We need the first vector which is the first column of lhs not the first row.
    let lhs_u64 = {
        let mut lhs_u64 = lhs.into_iter().rev().collect::<Vec<_>>();
        lhs_u64.rotate_right(1);
        lhs_u64
    };
    let rhs_u64 = rhs
        .into_iter()
        .map(|x| x.as_canonical_u64())
        .collect::<Vec<_>>();

    // Result is computed as a polynomial of degree 2N - 2.
    // We need to manually implement the reduction by X^N - 1.
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
pub fn apply_circulant_karat_even_odd_field<F: PrimeField64, const N: usize>(
    lhs: [u64; N],
    rhs: [F; N],
) -> [F; N] {
    // We need the first vector which is the first column of lhs not the first row.
    let lhs_u64 = {
        let mut lhs_u64 = lhs.into_iter().rev().collect::<Vec<_>>();
        lhs_u64.rotate_right(1);
        lhs_u64
    };
    let rhs_u64 = rhs
        .into_iter()
        .map(|x| x.as_canonical_u64())
        .collect::<Vec<_>>();

    let result = apply_circulant_karat_even_odd(&lhs_u64, &rhs_u64);

    result
        .into_iter()
        .map(|x| F::from_wrapped_u64(x))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

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

// Given an input v computes the convolution of v with the constant vector MATRIX_CIRC_MDS_8_SML.
pub fn apply_circulant_8_karat<F: PrimeField64>(input: [F; 8]) -> [F; 8] {
    const N: usize = 8;
    let mut output = [0; N];

    let lhs_low = [
        input[0].as_canonical_u64(),
        input[1].as_canonical_u64(),
        input[2].as_canonical_u64(),
        input[3].as_canonical_u64(),
    ];
    let lhs_high = [
        input[4].as_canonical_u64(),
        input[5].as_canonical_u64(),
        input[6].as_canonical_u64(),
        input[7].as_canonical_u64(),
    ];

    const MATRIX_CIRC_MDS_8_SML_LR: ([u64; 4], [u64; 4]) = split_half_lr(row_to_col(MATRIX_CIRC_MDS_8_SML));
    const MATRIX_CIRC_MDS_8_SML_MIX: [u64; 4] = add_n(&MATRIX_CIRC_MDS_8_SML_LR.0, &MATRIX_CIRC_MDS_8_SML_LR.1);

    let mut prod_conv = prod4(&lhs_low, &MATRIX_CIRC_MDS_8_SML_LR.0);
    let prod_high = prod4(&lhs_high, &MATRIX_CIRC_MDS_8_SML_LR.1);

    add_mut(&mut prod_conv, &prod_high);

    let lhs_mix = add_n(&lhs_low, &lhs_high);

    

    let mut prod_mix = prod4(&lhs_mix, &MATRIX_CIRC_MDS_8_SML_MIX);
    sub_mut(&mut prod_mix, &prod_conv);

    add_mut(&mut output[..7], &prod_conv);
    add_mut(&mut output[4..], &prod_mix[..4]);
    add_mut(&mut output[..3], &prod_mix[4..]);

    output
        .into_iter()
        .map(|x| F::from_wrapped_u64(x))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

// Given an input v computes the convolution of v with the constant vector MATRIX_CIRC_MDS_8_SML.
// Uses the odd even decomposition.
pub fn apply_circulant_8_karat_even_odd<F: PrimeField64>(input: [F; 8]) -> [F; 8] {
    const N: usize = 8;
    let mut output = [F::zero(); N];

    let lhs_even = [
        input[0].as_canonical_u64(),
        input[2].as_canonical_u64(),
        input[4].as_canonical_u64(),
        input[6].as_canonical_u64(),
    ];
    let lhs_odd = [
        input[1].as_canonical_u64(),
        input[3].as_canonical_u64(),
        input[5].as_canonical_u64(),
        input[7].as_canonical_u64(),
    ];

    const MATRIX_CIRC_MDS_8_SML_EVEN_ODD: ([u64; 4], [u64; 4]) = split_half_eo(row_to_col(MATRIX_CIRC_MDS_8_SML));
    const MATRIX_CIRC_MDS_8_SML_EOMIX: [u64; 4] =
        add_n(&MATRIX_CIRC_MDS_8_SML_EVEN_ODD.0, &MATRIX_CIRC_MDS_8_SML_EVEN_ODD.1);

    let mut conv_even = conv4(&lhs_even, &MATRIX_CIRC_MDS_8_SML_EVEN_ODD.0);
    let conv_odd = conv4(&lhs_odd, &MATRIX_CIRC_MDS_8_SML_EVEN_ODD.1);

    let lhs_mix = add_n(&lhs_even, &lhs_odd);

    let mut output_odd = conv4(&lhs_mix, &MATRIX_CIRC_MDS_8_SML_EOMIX);
    sub_mut(&mut output_odd, &conv_odd);
    sub_mut(&mut output_odd, &conv_even);

    add_mut(&mut conv_even[1..], &conv_odd[..3]);
    conv_even[0] += conv_odd[3];

    output[0] = F::from_wrapped_u64(conv_even[0]);
    output[1] = F::from_wrapped_u64(output_odd[0]);
    output[2] = F::from_wrapped_u64(conv_even[1]);
    output[3] = F::from_wrapped_u64(output_odd[1]);
    output[4] = F::from_wrapped_u64(conv_even[2]);
    output[5] = F::from_wrapped_u64(output_odd[2]);
    output[6] = F::from_wrapped_u64(conv_even[3]);
    output[7] = F::from_wrapped_u64(output_odd[3]);

    output
}

// Given an input v computes the convolution of v with the constant vector MATRIX_CIRC_MDS_16_SML.
pub fn apply_circulant_16_karat<F: PrimeField64>(input: [F; 16]) -> [F; 16] {
    const N: usize = 16;
    let mut output = [0; N];

    let lhs_low = [
        input[0].as_canonical_u64(),
        input[1].as_canonical_u64(),
        input[2].as_canonical_u64(),
        input[3].as_canonical_u64(),
        input[4].as_canonical_u64(),
        input[5].as_canonical_u64(),
        input[6].as_canonical_u64(),
        input[7].as_canonical_u64(),
    ];
    let lhs_high = [
        input[8].as_canonical_u64(),
        input[9].as_canonical_u64(),
        input[10].as_canonical_u64(),
        input[11].as_canonical_u64(),
        input[12].as_canonical_u64(),
        input[13].as_canonical_u64(),
        input[14].as_canonical_u64(),
        input[15].as_canonical_u64(),
    ];

    const MATRIX_CIRC_MDS_16_SML_LR: ([u64; 8], [u64; 8]) = split_half_lr(row_to_col(MATRIX_CIRC_MDS_16_SML));
    const MATRIX_CIRC_MDS_16_SML_MIX: [u64; 8] = add_n(&MATRIX_CIRC_MDS_16_SML_LR.0, &MATRIX_CIRC_MDS_16_SML_LR.1);

    let prod_low = prod8(MATRIX_CIRC_MDS_16_SML_LR.0, lhs_low);
    let prod_high = prod8(MATRIX_CIRC_MDS_16_SML_LR.1, lhs_high);
    let prod_conv = add_n(&prod_low, &prod_high);

    let lhs_mix = add_n(&lhs_low, &lhs_high);

    let mut prod_mix = prod8(MATRIX_CIRC_MDS_16_SML_MIX, lhs_mix);
    sub_mut(&mut prod_mix, &prod_conv);

    add_mut(&mut output[..15], &prod_conv);
    add_mut(&mut output[8..], &prod_mix[..8]);
    add_mut(&mut output[..7], &prod_mix[8..]);

    output
        .into_iter()
        .map(|x| F::from_wrapped_u64(x))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

// Given an input v computes the convolution of v with the constant vector MATRIX_CIRC_MDS_8_SML.
// Uses the odd even decomposition.
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
    const MATRIX_CIRC_MDS_16_SML_EOMIX: [u64; 8] =
        add_n(&MATRIX_CIRC_MDS_16_SML_EVEN_ODD.0, &MATRIX_CIRC_MDS_16_SML_EVEN_ODD.1);

    let mut conv_even = conv8(&lhs_even, &MATRIX_CIRC_MDS_16_SML_EVEN_ODD.0);
    let conv_odd = conv8(&lhs_odd, &MATRIX_CIRC_MDS_16_SML_EVEN_ODD.1);

    // No longer need lhs_even so can reuse the memory.
    add_mut(&mut lhs_even, &lhs_odd);

    let mut output_odd = conv8(&lhs_even, &MATRIX_CIRC_MDS_16_SML_EOMIX);
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

#[inline]
const fn add_n<const N: usize>(lhs: &[u64; N], rhs: &[u64; N]) -> [u64; N] {
    let mut output = [0u64; N];
    let mut i = 0;
    loop {
        if i == N {
            break;
        }
        output[i] = lhs[i] + rhs[i];
        i += 1;
    }
    output
}

// Given inputs lhs and rhs computes the convolution lhs * rhs.
// Uses the odd even decomposition.
#[inline]
fn conv8(lhs: &[u64; 8], rhs: &[u64; 8]) -> [u64; 8] {
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

// In principal we should save a small number of operations here by using karatsuba: (103, 113)
#[inline]
fn prod8(lhs: [u64; 8], rhs: [u64; 8]) -> [u64; 15] {
    let mut output = [0; 15];

    let mut lhs_low = [lhs[0], lhs[1], lhs[2], lhs[3]];
    let lhs_high = [lhs[4], lhs[5], lhs[6], lhs[7]];
    let mut rhs_low = [rhs[0], rhs[1], rhs[2], rhs[3]];
    let rhs_high = [rhs[4], rhs[5], rhs[6], rhs[7]];

    let prod_low = prod4(&lhs_low, &rhs_low);
    let prod_high = prod4(&lhs_high, &rhs_high);

    add_mut(&mut output[..7], &prod_low);
    add_mut(&mut output[8..15], &prod_high);

    // Add the two halves together, storing the result in lhs_low.
    add_mut(&mut lhs_low, &lhs_high);
    add_mut(&mut rhs_low, &rhs_high);

    let prod_mix = prod4(&lhs_low, &rhs_low);

    add_mut(&mut output[4..11], &prod_mix);
    sub_mut(&mut output[4..11], &prod_low);
    sub_mut(&mut output[4..11], &prod_high);

    output
}

#[inline]
const fn _rotate_right<const N: usize>(input: [u64; N], offset: usize) -> [u64; N] {
    let mut output = [0u64; N];
    let mut i = 0;
    loop {
        if i == N {
            break;
        }
        output[i] = input[(N - offset + i) % N];
        i += 1
    }
    output
}
