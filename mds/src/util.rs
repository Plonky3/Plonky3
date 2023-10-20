use alloc::vec::Vec;
use core::array;

use p3_dft::TwoAdicSubgroupDft;
use p3_field::{AbstractField, PrimeField64, TwoAdicField};

// NB: These four are MDS for M31, BabyBear and Goldilocks
//const MATRIX_CIRC_MDS_8_2EXP: [u64; 8] = [1, 1, 2, 1, 8, 32, 4, 256];
const MATRIX_CIRC_MDS_8_SML: [u64; 8] = [4, 1, 2, 9, 10, 5, 1, 1];
// Much smaller: [1, 1, -1, 2, 3, 8, 2, -3] but not sure how to deal with the -ve's

// const MATRIX_CIRC_MDS_12_2EXP: [u64; 12] = [1, 1, 2, 1, 8, 32, 2, 256, 4096, 8, 65536, 1024];
// const MATRIX_CIRC_MDS_12_SML: [u64; 12] = [9, 7, 4, 1, 16, 2, 256, 128, 3, 32, 1, 1];
const MATRIX_CIRC_MDS_12_SML: [u64; 12] = [1, 1, 2, 1, 8, 9, 10, 7, 5, 9, 4, 10];

// Trying to maximise the # of 1's in the vector.
// Not clear exatcly what we should be optimising here but that seems reasonable.
const MATRIX_CIRC_MDS_16_SML: [u64; 16] =
    [1, 1, 51, 1, 11, 17, 2, 1, 101, 63, 15, 2, 67, 22, 13, 3];

// 1, 1, 51, 52, 11, 63, 1, 2, 1, 2, 15, 67, 2, 22, 13, 3
// [1, 1, 2, 1, 8, 32, 2, 65, 77, 8, 91, 31, 3, 65, 32, 7];

/// Given the first row `circ_matrix` of an NxN circulant matrix, say
/// C, return the product `C*input`.
///
/// NB: This function is a naive implementation of the n²
/// evaluation. It is a placeholder until we have FFT implementations
/// for all combinations of field and size.
pub fn apply_circulant<AF: AbstractField, const N: usize>(
    circ_matrix: &[u64; N],
    input: [AF; N],
) -> [AF; N] {
    let mut matrix: [AF; N] = circ_matrix.map(AF::from_canonical_u64);

    let mut output = array::from_fn(|_| AF::zero());
    for out_i in output.iter_mut().take(N - 1) {
        *out_i = AF::dot_product(&matrix, &input);
        matrix.rotate_right(1);
    }
    output[N - 1] = AF::dot_product(&matrix, &input);
    output
}

/// Given an array `input` and an `offset`, return the array whose
/// elements are those of `input` shifted `offset` places to the
/// right.
///
/// NB: The algorithm is inefficient but simple enough that this
/// function can be declared `const`, and that is the intended use. In
/// non-`const` contexts you probably want `[T]::rotate_right()` from
/// the standard library.
pub(crate) const fn rotate_right<const N: usize>(input: [u64; N], offset: usize) -> [u64; N] {
    let mut output = [0u64; N];
    let mut i = 0;
    loop {
        if i == N {
            break;
        }
        output[i] = input[(N - offset + i) % N];
        i += 1;
    }
    output
}

/// As for `apply_circulant()` above, but with `circ_matrix` set to a
/// fixed 8x8 MDS matrix with small entries that satisfy the condition
/// on `PrimeField64::z_linear_combination_sml()`.
pub(crate) fn apply_circulant_8_sml<F: PrimeField64>(input: [F; 8]) -> [F; 8] {
    const N: usize = 8;
    let mut output = [F::zero(); N];

    const MAT_0: [u64; N] = MATRIX_CIRC_MDS_8_SML;
    output[0] = F::linear_combination_u64(MAT_0, &input);
    const MAT_1: [u64; N] = rotate_right(MATRIX_CIRC_MDS_8_SML, 1);
    output[1] = F::linear_combination_u64(MAT_1, &input);
    const MAT_2: [u64; N] = rotate_right(MATRIX_CIRC_MDS_8_SML, 2);
    output[2] = F::linear_combination_u64(MAT_2, &input);
    const MAT_3: [u64; N] = rotate_right(MATRIX_CIRC_MDS_8_SML, 3);
    output[3] = F::linear_combination_u64(MAT_3, &input);
    const MAT_4: [u64; N] = rotate_right(MATRIX_CIRC_MDS_8_SML, 4);
    output[4] = F::linear_combination_u64(MAT_4, &input);
    const MAT_5: [u64; N] = rotate_right(MATRIX_CIRC_MDS_8_SML, 5);
    output[5] = F::linear_combination_u64(MAT_5, &input);
    const MAT_6: [u64; N] = rotate_right(MATRIX_CIRC_MDS_8_SML, 6);
    output[6] = F::linear_combination_u64(MAT_6, &input);
    const MAT_7: [u64; N] = rotate_right(MATRIX_CIRC_MDS_8_SML, 7);
    output[7] = F::linear_combination_u64(MAT_7, &input);

    output
}

/// As for `apply_circulant()` above, but with `circ_matrix` set to a
/// fixed 12x12 MDS matrix with small entries that satisfy the condition
/// on `PrimeField64::z_linear_combination_sml()`.
pub(crate) fn apply_circulant_12_sml<F: PrimeField64>(input: [F; 12]) -> [F; 12] {
    const N: usize = 12;
    let mut output = [F::zero(); N];

    const MAT_0: [u64; N] = MATRIX_CIRC_MDS_12_SML;
    output[0] = F::linear_combination_u64(MAT_0, &input);
    const MAT_1: [u64; N] = rotate_right(MATRIX_CIRC_MDS_12_SML, 1);
    output[1] = F::linear_combination_u64(MAT_1, &input);
    const MAT_2: [u64; N] = rotate_right(MATRIX_CIRC_MDS_12_SML, 2);
    output[2] = F::linear_combination_u64(MAT_2, &input);
    const MAT_3: [u64; N] = rotate_right(MATRIX_CIRC_MDS_12_SML, 3);
    output[3] = F::linear_combination_u64(MAT_3, &input);
    const MAT_4: [u64; N] = rotate_right(MATRIX_CIRC_MDS_12_SML, 4);
    output[4] = F::linear_combination_u64(MAT_4, &input);
    const MAT_5: [u64; N] = rotate_right(MATRIX_CIRC_MDS_12_SML, 5);
    output[5] = F::linear_combination_u64(MAT_5, &input);
    const MAT_6: [u64; N] = rotate_right(MATRIX_CIRC_MDS_12_SML, 6);
    output[6] = F::linear_combination_u64(MAT_6, &input);
    const MAT_7: [u64; N] = rotate_right(MATRIX_CIRC_MDS_12_SML, 7);
    output[7] = F::linear_combination_u64(MAT_7, &input);
    const MAT_8: [u64; N] = rotate_right(MATRIX_CIRC_MDS_12_SML, 8);
    output[8] = F::linear_combination_u64(MAT_8, &input);
    const MAT_9: [u64; N] = rotate_right(MATRIX_CIRC_MDS_12_SML, 9);
    output[9] = F::linear_combination_u64(MAT_9, &input);
    const MAT_10: [u64; N] = rotate_right(MATRIX_CIRC_MDS_12_SML, 10);
    output[10] = F::linear_combination_u64(MAT_10, &input);
    const MAT_11: [u64; N] = rotate_right(MATRIX_CIRC_MDS_12_SML, 11);
    output[11] = F::linear_combination_u64(MAT_11, &input);

    output
}

/// As for `apply_circulant()` above, but with `circ_matrix` set to a
/// fixed 16x16 MDS matrix with small entries that satisfy the condition
/// on `PrimeField64::z_linear_combination_sml()`.
pub fn apply_circulant_16_sml<F: PrimeField64>(input: [F; 16]) -> [F; 16] {
    const N: usize = 16;
    let mut output = [F::ZERO; N];

    const MAT_0: [u64; N] = MATRIX_CIRC_MDS_16_SML;
    output[0] = F::linear_combination_u64(MAT_0, &input);
    const MAT_1: [u64; N] = rotate_right(MATRIX_CIRC_MDS_16_SML, 1);
    output[1] = F::linear_combination_u64(MAT_1, &input);
    const MAT_2: [u64; N] = rotate_right(MATRIX_CIRC_MDS_16_SML, 2);
    output[2] = F::linear_combination_u64(MAT_2, &input);
    const MAT_3: [u64; N] = rotate_right(MATRIX_CIRC_MDS_16_SML, 3);
    output[3] = F::linear_combination_u64(MAT_3, &input);
    const MAT_4: [u64; N] = rotate_right(MATRIX_CIRC_MDS_16_SML, 4);
    output[4] = F::linear_combination_u64(MAT_4, &input);
    const MAT_5: [u64; N] = rotate_right(MATRIX_CIRC_MDS_16_SML, 5);
    output[5] = F::linear_combination_u64(MAT_5, &input);
    const MAT_6: [u64; N] = rotate_right(MATRIX_CIRC_MDS_16_SML, 6);
    output[6] = F::linear_combination_u64(MAT_6, &input);
    const MAT_7: [u64; N] = rotate_right(MATRIX_CIRC_MDS_16_SML, 7);
    output[7] = F::linear_combination_u64(MAT_7, &input);
    const MAT_8: [u64; N] = rotate_right(MATRIX_CIRC_MDS_16_SML, 8);
    output[8] = F::linear_combination_u64(MAT_8, &input);
    const MAT_9: [u64; N] = rotate_right(MATRIX_CIRC_MDS_16_SML, 9);
    output[9] = F::linear_combination_u64(MAT_9, &input);
    const MAT_10: [u64; N] = rotate_right(MATRIX_CIRC_MDS_16_SML, 10);
    output[10] = F::linear_combination_u64(MAT_10, &input);
    const MAT_11: [u64; N] = rotate_right(MATRIX_CIRC_MDS_16_SML, 11);
    output[11] = F::linear_combination_u64(MAT_11, &input);
    const MAT_12: [u64; N] = rotate_right(MATRIX_CIRC_MDS_16_SML, 12);
    output[12] = F::linear_combination_u64(MAT_12, &input);
    const MAT_13: [u64; N] = rotate_right(MATRIX_CIRC_MDS_16_SML, 13);
    output[13] = F::linear_combination_u64(MAT_13, &input);
    const MAT_14: [u64; N] = rotate_right(MATRIX_CIRC_MDS_16_SML, 14);
    output[14] = F::linear_combination_u64(MAT_14, &input);
    const MAT_15: [u64; N] = rotate_right(MATRIX_CIRC_MDS_16_SML, 15);
    output[15] = F::linear_combination_u64(MAT_15, &input);

    output
}

/// Given the first row of a circulant matrix, return the first column
/// of that circulant matrix. For example, v = [0, 1, 2, 3, 4, 5],
/// then output = [0, 5, 4, 3, 2, 1], i.e. the first element is the
/// same and the other elements are reversed.
///
/// This is useful to prepare a circulant matrix for input to an FFT
/// algorithm, which expects the first column of the matrix rather
/// than the first row (as we normally store them).
///
/// NB: The algorithm is inefficient but simple enough that this
/// function can be declared `const`, and that is the intended context
/// for use.
pub(crate) const fn first_row_to_first_col<const N: usize>(v: &[u64; N]) -> [u64; N] {
    let mut output = [0u64; N];
    output[0] = v[0];
    let mut i = 1;
    loop {
        if i >= N {
            break;
        }
        output[i] = v[N - i];
        i += 1;
    }
    output
}

/// Use the convolution theorem to calculate the product of the given
/// circulant matrix and the given vector. The circulant matrix must
/// be specified by its first *column*, not its first row. If you have
/// the row as an array, you can obtain the column with `first_row_to_first_col()`.
#[inline]
pub(crate) fn apply_circulant_fft<F: TwoAdicField, const N: usize, FFT: TwoAdicSubgroupDft<F>>(
    fft: FFT,
    column: [u64; N],
    input: &[F; N],
) -> [F; N] {
    let column = column.map(F::from_canonical_u64).to_vec();
    let matrix = fft.dft(column);
    let input = fft.dft(input.to_vec());

    // point-wise product
    let product = matrix
        .iter()
        .zip(input)
        .map(|(&x, y)| x * y)
        .collect::<Vec<_>>();

    let output = fft.idft(product);
    output.try_into().unwrap()
}

#[cfg(test)]
mod tests {
    use super::rotate_right;

    #[test]
    fn rotation() {
        let input = [0, 1, 2, 3];
        let output = [
            [0, 1, 2, 3],
            [3, 0, 1, 2],
            [2, 3, 0, 1],
            [1, 2, 3, 0],
            [0, 1, 2, 3],
        ];

        for (i, &out_i) in output.iter().enumerate() {
            assert_eq!(rotate_right(input, i), out_i);
        }
    }
}
