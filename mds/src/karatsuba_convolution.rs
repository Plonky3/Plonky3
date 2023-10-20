use p3_field::PrimeField64;

const _MATRIX_CIRC_MDS_8_SML: [u64; 8] = [4, 1, 1, 5, 10, 9, 2, 1];
const MATRIX_CIRC_MDS_8_SML_0: [u64; 4] = [4, 1, 1, 5];
const MATRIX_CIRC_MDS_8_SML_1: [u64; 4] = [10, 9, 2, 1];
const MATRIX_CIRC_MDS_8_SML_MIX: [u64; 4] = [14, 10, 3, 6];

const MATRIX_CIRC_MDS_8_SML_EVEN: [u64; 4] = [4, 1, 10, 2];
const MATRIX_CIRC_MDS_8_SML_ODD: [u64; 4] = [1, 5, 9, 1];
const MATRIX_CIRC_MDS_8_SML_EOMIX: [u64; 4] = [5, 6, 19, 3];

// const _MATRIX_CIRC_MDS_8_SML: [u64; 8] = [4, 1, 2, 9, 10, 5, 1, 1];
// const MATRIX_CIRC_MDS_8_SML_0: [u64; 4] = [4, 1, 2, 9];
// const MATRIX_CIRC_MDS_8_SML_1: [u64; 4] = [10, 5, 1, 13];
// const MATRIX_CIRC_MDS_8_SML_MIX: [u64; 4] = [14, 6, 3, 22];

// const _MATRIX_CIRC_MDS_12_SML: [u64; 12] = [1, 1, 2, 1, 8, 9, 10, 7, 5, 9, 4, 10];
// const _MATRIX_CIRC_MDS_12_SML_0: [u64; 6] = [1, 1, 2, 1, 8, 9];
// const _MATRIX_CIRC_MDS_12_SML_1: [u64; 6] = [10, 7, 5, 9, 4, 10];
// const _MATRIX_CIRC_MDS_12_SML_MIX: [u64; 6] = [11, 8, 7, 10, 12, 19];

const _MATRIX_CIRC_MDS_16_SML: [u64; 16] =
    [1, 3, 13, 22, 67, 2, 15, 63, 101, 1, 2, 17, 11, 1, 51, 1];
const MATRIX_CIRC_MDS_16_SML_0: [u64; 8] = [1, 3, 13, 22, 67, 2, 15, 63];
const MATRIX_CIRC_MDS_16_SML_1: [u64; 8] = [101, 1, 2, 17, 11, 1, 51, 1];
// const MATRIX_CIRC_MDS_16_SML_MIX: [u64; 8] = [102, 4, 15, 39, 78, 3, 66, 64];

// const MATRIX_CIRC_MDS_16_SML_EVEN: [u64; 8] = [1, 51, 11, 2, 101, 15, 67, 13];
// const MATRIX_CIRC_MDS_16_SML_ODD: [u64; 8] = [3, 1, 1, 17, 1, 63, 2, 22];
// const MATRIX_CIRC_MDS_16_SML_EOMIX: [u64; 8] = [4, 52, 12, 19, 102, 78, 69, 35];

const MATRIX_CIRC_MDS_16_SML_EVEN: [u64; 8] = [1, 13, 67, 15, 101, 2, 11, 51];
const MATRIX_CIRC_MDS_16_SML_ODD: [u64; 8] = [3, 22, 2, 63, 1, 17, 1, 1];

// const _MATRIX_CIRC_MDS_16_SML: [u64; 16]   = [1, 1, 51, 1, 11, 17, 2, 1, 101, 63, 15, 2, 67, 22, 13, 3];
// const MATRIX_CIRC_MDS_16_SML_0: [u64; 8]   = [1,    1, 51, 1, 11, 17, 2, 1];
// const MATRIX_CIRC_MDS_16_SML_1: [u64; 8]   = [101, 63, 15, 2, 67, 22, 13, 3];
// const MATRIX_CIRC_MDS_16_SML_MIX: [u64; 8] = [102, 64, 66, 3, 78, 39, 15, 4];

// Given an input v computes the convolution of v with the constant vector MATRIX_CIRC_MDS_8_SML.
pub fn apply_circulant_8_karat<F: PrimeField64>(input: [F; 8]) -> [F; 8] {
    const N: usize = 8;
    let mut output = [F::ZERO; N];

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

    let prod_low = prod4(&lhs_low, &MATRIX_CIRC_MDS_8_SML_0);
    let prod_high = prod4(&lhs_high, &MATRIX_CIRC_MDS_8_SML_1);
    let prod_conv = add_n(prod_low, prod_high);

    let lhs_mix = add_n(lhs_low, lhs_high);

    let prod_mix = prod4(&lhs_mix, &MATRIX_CIRC_MDS_8_SML_MIX);
    let prod_mid = sub_n(prod_mix, prod_conv);

    output[0] = F::from_wrapped_u64(prod_conv[0] + prod_mid[4]);
    output[1] = F::from_wrapped_u64(prod_conv[1] + prod_mid[5]);
    output[2] = F::from_wrapped_u64(prod_conv[2] + prod_mid[6]);
    output[3] = F::from_wrapped_u64(prod_conv[3]);
    output[4] = F::from_wrapped_u64(prod_conv[4] + prod_mid[0]);
    output[5] = F::from_wrapped_u64(prod_conv[5] + prod_mid[1]);
    output[6] = F::from_wrapped_u64(prod_conv[6] + prod_mid[2]);
    output[7] = F::from_wrapped_u64(prod_mid[3]);

    output
}

// Given an input v computes the convolution of v with the constant vector MATRIX_CIRC_MDS_8_SML.
// Uses the odd even decomposition.
pub fn apply_circulant_8_karat_even_odd<F: PrimeField64>(input: [F; 8]) -> [F; 8] {
    const N: usize = 8;
    let mut output = [F::ZERO; N];

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

    let conv_even = conv4(&lhs_even, &MATRIX_CIRC_MDS_8_SML_EVEN);
    let conv_odd = conv4(&lhs_odd, &MATRIX_CIRC_MDS_8_SML_ODD);

    let lhs_mix = add_n(lhs_even, lhs_odd);

    let conv_mix = conv4(&lhs_mix, &MATRIX_CIRC_MDS_8_SML_EOMIX);

    output[0] = F::from_wrapped_u64(conv_even[0] + conv_odd[3]);
    output[1] = F::from_wrapped_u64(conv_mix[0] - conv_even[0] - conv_odd[0]);
    output[2] = F::from_wrapped_u64(conv_even[1] + conv_odd[0]);
    output[3] = F::from_wrapped_u64(conv_mix[1] - conv_even[1] - conv_odd[1]);
    output[4] = F::from_wrapped_u64(conv_even[2] + conv_odd[1]);
    output[5] = F::from_wrapped_u64(conv_mix[2] - conv_even[2] - conv_odd[2]);
    output[6] = F::from_wrapped_u64(conv_even[3] + conv_odd[2]);
    output[7] = F::from_wrapped_u64(conv_mix[3] - conv_even[3] - conv_odd[3]);

    output
}

// Given an input v computes the convolution of v with the constant vector MATRIX_CIRC_MDS_16_SML.
pub fn apply_circulant_16_karat<F: PrimeField64>(input: [F; 16]) -> [F; 16] {
    const N: usize = 16;
    let mut output = [F::ZERO; N];

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

    let prod_low = prod8(MATRIX_CIRC_MDS_16_SML_0, lhs_low);
    let prod_high = prod8(MATRIX_CIRC_MDS_16_SML_1, lhs_high);
    let prod_conv = add_n(prod_low, prod_high);

    let lhs_mix = add_n(lhs_low, lhs_high);

    const MATRIX_CIRC_MDS_16_SML_MIX: [u64; 8] = add_n(MATRIX_CIRC_MDS_16_SML_0, MATRIX_CIRC_MDS_16_SML_1);

    let prod_mix = prod8(MATRIX_CIRC_MDS_16_SML_MIX, lhs_mix);
    let prod_mid = sub_n(prod_mix, prod_conv);

    output[0] = F::from_wrapped_u64(prod_conv[0] + prod_mid[8]);
    output[1] = F::from_wrapped_u64(prod_conv[1] + prod_mid[9]);
    output[2] = F::from_wrapped_u64(prod_conv[2] + prod_mid[10]);
    output[3] = F::from_wrapped_u64(prod_conv[3] + prod_mid[11]);
    output[4] = F::from_wrapped_u64(prod_conv[4] + prod_mid[12]);
    output[5] = F::from_wrapped_u64(prod_conv[5] + prod_mid[13]);
    output[6] = F::from_wrapped_u64(prod_conv[6] + prod_mid[14]);
    output[7] = F::from_wrapped_u64(prod_conv[7]);
    output[8] = F::from_wrapped_u64(prod_conv[8] + prod_mid[0]);
    output[9] = F::from_wrapped_u64(prod_conv[9] + prod_mid[1]);
    output[10] = F::from_wrapped_u64(prod_conv[10] + prod_mid[2]);
    output[11] = F::from_wrapped_u64(prod_conv[11] + prod_mid[3]);
    output[12] = F::from_wrapped_u64(prod_conv[12] + prod_mid[4]);
    output[13] = F::from_wrapped_u64(prod_conv[13] + prod_mid[5]);
    output[14] = F::from_wrapped_u64(prod_conv[14] + prod_mid[6]);
    output[15] = F::from_wrapped_u64(prod_mid[7]);

    output
}

// Given an input v computes the convolution of v with the constant vector MATRIX_CIRC_MDS_8_SML.
// Uses the odd even decomposition.
pub fn apply_circulant_16_karat_even_odd<F: PrimeField64>(input: [F; 16]) -> [F; 16] {
    const N: usize = 16;
    let mut output = [F::ZERO; N];

    let lhs_even = [
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

    let conv_even = conv8(&lhs_even, MATRIX_CIRC_MDS_16_SML_EVEN);
    let conv_odd = conv8(&lhs_odd, MATRIX_CIRC_MDS_16_SML_ODD);

    let lhs_mix = add_n(lhs_even, lhs_odd);

    const MATRIX_CIRC_MDS_16_SML_EOMIX: [u64; 8] = add_n(MATRIX_CIRC_MDS_16_SML_EVEN, MATRIX_CIRC_MDS_16_SML_ODD);

    let conv_mix = conv8(&lhs_mix, MATRIX_CIRC_MDS_16_SML_EOMIX);

    let conv_odd_rot = [conv_odd[7], conv_odd[0], conv_odd[1], conv_odd[2], conv_odd[3], conv_odd[4], conv_odd[5], conv_odd[6]];

    let output_even = add_n(conv_even, conv_odd_rot);
    let output_odd  = sub_n2(conv_mix, conv_even, conv_odd);

    output[0] = F::from_wrapped_u64(output_even[0]);
    output[1] = F::from_wrapped_u64(output_odd[0]);
    output[2] = F::from_wrapped_u64(output_even[1]);
    output[3] = F::from_wrapped_u64(output_odd[1]);
    output[4] = F::from_wrapped_u64(output_even[2]);
    output[5] = F::from_wrapped_u64(output_odd[2]);
    output[6] = F::from_wrapped_u64(output_even[3]);
    output[7] = F::from_wrapped_u64(output_odd[3]);
    output[8] = F::from_wrapped_u64(output_even[4]);
    output[9] = F::from_wrapped_u64(output_odd[4]);
    output[10] = F::from_wrapped_u64(output_even[5]);
    output[11] = F::from_wrapped_u64(output_odd[5]);
    output[12] = F::from_wrapped_u64(output_even[6]);
    output[13] = F::from_wrapped_u64(output_odd[6]);
    output[14] = F::from_wrapped_u64(output_even[7]);
    output[15] = F::from_wrapped_u64(output_odd[7]);

    output
}

#[inline]
const fn add_n<const N: usize>(lhs: [u64; N], rhs: [u64; N]) -> [u64; N] {
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

#[inline]
const fn sub_n<const N: usize>(lhs: [u64; N], sub1: [u64; N]) -> [u64; N] {
    let mut output = [0u64; N];
    let mut i = 0;
    loop {
        if i == N {
            break;
        }
        output[i] = lhs[i] - sub1[i];
        i += 1;
    }
    output
}

#[inline]
const fn sub_n2<const N: usize>(lhs: [u64; N], sub1: [u64; N], sub2: [u64; N]) -> [u64; N] {
    let mut output = [0u64; N];
    let mut i = 0;
    loop {
        if i == N {
            break;
        }
        output[i] = lhs[i] - sub1[i] - sub2[i];
        i += 1;
    }
    output
}

// No point doing the Karatsuba algorithm once we get down to vectors of length 4.
// It both takes more operations (27 vs 25) and is more complex.
// Hence this encodes the naive polynomial multiplication for polynomials of degree 3.
#[inline]
fn prod4(lhs: &[u64; 4], rhs: &[u64; 4]) -> [u64; 7] {
    [
        lhs[0] * rhs[0],
        lhs[0] * rhs[1] + lhs[1] * rhs[0],
        lhs[0] * rhs[2] + lhs[1] * rhs[1] + lhs[2] * rhs[0],
        lhs[0] * rhs[3] + lhs[1] * rhs[2] + lhs[2] * rhs[1] + lhs[3] * rhs[0],
        lhs[1] * rhs[3] + lhs[2] * rhs[2] + lhs[3] * rhs[1],
        lhs[2] * rhs[3] + lhs[3] * rhs[2],
        lhs[3] * rhs[3],
    ]
}

// No point doing the Karatsuba algorithm once we get down to vectors of length 4.
// It both takes more operations (27 vs 25) and is more complex.
// Hence this encodes the naive convolution for vectors of length 4.
#[inline]
fn conv4(lhs: &[u64; 4], rhs: &[u64; 4]) -> [u64; 4] {
    [
        lhs[0] * rhs[0] + lhs[1] * rhs[3] + lhs[2] * rhs[2] + lhs[3] * rhs[1],
        lhs[0] * rhs[1] + lhs[1] * rhs[0] + lhs[2] * rhs[3] + lhs[3] * rhs[2],
        lhs[0] * rhs[2] + lhs[1] * rhs[1] + lhs[2] * rhs[0] + lhs[3] * rhs[3],
        lhs[0] * rhs[3] + lhs[1] * rhs[2] + lhs[2] * rhs[1] + lhs[3] * rhs[0],
    ]
}

// Given an input v computes the convolution of v with the constant vector MATRIX_CIRC_MDS_8_SML.
// Uses the odd even decomposition.
#[inline]
fn conv8(lhs: &[u64; 8], rhs: [u64; 8]) -> [u64; 8] {
    // const N: usize = 8;
    // let mut output = [0; N];

    // output[0] = conv_combination_u64(lhs, rhs);
    // let rhs_1 = rotate_right(rhs, 1);
    // output[1] = conv_combination_u64(lhs, rhs_1);
    // let rhs_2 = rotate_right(rhs, 2);
    // output[2] = conv_combination_u64(lhs, rhs_2);
    // let rhs_3 = rotate_right(rhs, 3);
    // output[3] = conv_combination_u64(lhs, rhs_3);
    // let rhs_4 = rotate_right(rhs, 4);
    // output[4] = conv_combination_u64(lhs, rhs_4);
    // let rhs_5 = rotate_right(rhs, 5);
    // output[5] = conv_combination_u64(lhs, rhs_5);
    // let rhs_6 = rotate_right(rhs, 6);
    // output[6] = conv_combination_u64(lhs, rhs_6);
    // let rhs_7 = rotate_right(rhs, 7);
    // output[7] = conv_combination_u64(lhs, rhs_7);

    // output
    const N: usize = 8;
    let mut output = [0; N];

    let lhs_even = [lhs[0], lhs[2], lhs[4], lhs[6]];
    let lhs_odd = [lhs[1], lhs[3], lhs[5], lhs[7]];

    let rhs_even = [rhs[0], rhs[2], rhs[4], rhs[6]];
    let rhs_odd = [rhs[1], rhs[3], rhs[5], rhs[7]];

    let conv_even = conv4(&lhs_even, &rhs_even);
    let conv_odd = conv4(&lhs_odd, &rhs_odd);

    let lhs_mix = add_n(lhs_even, lhs_odd);
    let rhs_mix = add_n(rhs_even, rhs_odd);

    let conv_mix = conv4(&lhs_mix, &rhs_mix);

    output[0] = conv_even[0] + conv_odd[3];
    output[1] = conv_mix[0] - conv_even[0] - conv_odd[0];
    output[2] = conv_even[1] + conv_odd[0];
    output[3] = conv_mix[1] - conv_even[1] - conv_odd[1];
    output[4] = conv_even[2] + conv_odd[1];
    output[5] = conv_mix[2] - conv_even[2] - conv_odd[2];
    output[6] = conv_even[3] + conv_odd[2];
    output[7] = conv_mix[3] - conv_even[3] - conv_odd[3];

    output
}

// We save a small number of operations here by using karatsuba: (103, 113)
#[inline]
fn prod8(lhs: [u64; 8], rhs: [u64; 8]) -> [u64; 15] {
    let mut output = [0; 15];

    let lhs_low = [lhs[0], lhs[1], lhs[2], lhs[3]];
    let lhs_high = [lhs[4], lhs[5], lhs[6], lhs[7]];
    let rhs_low = [rhs[0], rhs[1], rhs[2], rhs[3]];
    let rhs_high = [rhs[4], rhs[5], rhs[6], rhs[7]];

    let prod_low = prod4(&lhs_low, &rhs_low);
    let prod_high = prod4(&lhs_high, &rhs_high);

    output[0] = prod_low[0];
    output[1] = prod_low[1];
    output[2] = prod_low[2];
    output[3] = prod_low[3];
    output[4] = prod_low[4];
    output[5] = prod_low[5];
    output[6] = prod_low[6];

    output[8] = prod_high[0];
    output[9] = prod_high[1];
    output[10] = prod_high[2];
    output[11] = prod_high[3];
    output[12] = prod_high[4];
    output[13] = prod_high[5];
    output[14] = prod_high[6];

    let lhs_mix = add_n(lhs_low, lhs_high);
    let rhs_mix = add_n(rhs_low, rhs_high);

    let prod_mix = prod4(&lhs_mix, &rhs_mix);
    let prod_mid = sub_n2(prod_mix, prod_low, prod_high);

    output[4] += prod_mid[0];
    output[5] += prod_mid[1];
    output[6] += prod_mid[2];
    output[7] += prod_mid[3];
    output[8] += prod_mid[4];
    output[9] += prod_mid[5];
    output[10] += prod_mid[6];

    output
}

#[inline]
fn _conv_combination_u64<const N: usize>(u: [u64; N], v: [u64; N]) -> u64 {
    // In order not to overflow a u64, we must have sum(u) <= 2^32.
    // debug_assert!(u.iter().sum::<u64>() <= (1u64 << 32));

    let mut dot = u[0] * v[0] as u64;
    for i in 1..N {
        dot += u[i] * v[i] as u64;
    }

    dot
}

#[inline]
const fn _rotate_right<const N: usize>(input: [u64; N], offset: usize) -> [u64; N] {
    let mut output = [0u64; N];
    let mut i = 0;
    loop {
        if i == N {
            break
        }
        output[i] = input[(N - offset + i) % N];
        i += 1
    }
    output
}