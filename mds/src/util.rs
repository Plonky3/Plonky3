use p3_dft::TwoAdicSubgroupDft;
use p3_field::{AbstractField, PrimeField32, PrimeField64, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;

// NB: These four are MDS for M31, BabyBear and Goldilocks
//const MATRIX_CIRC_MDS_8_2EXP: [u64; 8] = [1, 1, 2, 1, 8, 32, 4, 256];
const MATRIX_CIRC_MDS_8_SML: [u64; 8] = [4, 1, 2, 9, 10, 5, 1, 1];

//const MATRIX_CIRC_MDS_12_2EXP: [u64; 12] = [1, 1, 2, 1, 8, 32, 2, 256, 4096, 8, 65536, 1024];
const MATRIX_CIRC_MDS_12_SML: [u64; 12] = [9, 7, 4, 1, 16, 2, 256, 128, 3, 32, 1, 1];

fn dot_vec<F: AbstractField, const N: usize>(u: &[F; N], v: &[F; N]) -> F {
    u.iter().zip(v).map(|(x, y)| x.clone() * y.clone()).sum()
}

pub(crate) fn apply_circulant<F: AbstractField, const N: usize>(
    circ_matrix: &[u64; N],
    input: [F; N],
) -> [F; N] {
    let mut matrix: [F; N] = circ_matrix.map(F::from_canonical_u64);

    let mut output = [F::ZERO; N];
    for out_i in output.iter_mut().take(N - 1) {
        *out_i = dot_vec(&input, &matrix);
        matrix.rotate_right(1);
    }
    output[N - 1] = dot_vec(&input, &matrix);
    output
}

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

#[inline]
fn dot_vec_prime32_8_sml<F: PrimeField32>(u: &[F; 8], v: [u64; 8]) -> F {
    // An unreduced dot product is bounded above by s * (2^32 - 1) < 2^38
    // where s = sum(MATRIX_CIRC_MDS_8_SML) = 33.
    let dot = u[0].as_noncanonical_u32() as u64 * v[0]
        + u[1].as_noncanonical_u32() as u64 * v[1]
        + u[2].as_noncanonical_u32() as u64 * v[2]
        + u[3].as_noncanonical_u32() as u64 * v[3]
        + u[4].as_noncanonical_u32() as u64 * v[4]
        + u[5].as_noncanonical_u32() as u64 * v[5]
        + u[6].as_noncanonical_u32() as u64 * v[6]
        + u[7].as_noncanonical_u32() as u64 * v[7];
    F::from_wrapped_u64(dot)
}

#[inline]
fn dot_vec_prime32_12_sml<F: PrimeField32>(u: &[F; 12], v: [u64; 12]) -> F {
    // An unreduced dot product is bounded above by s * (2^32 - 1) < 2^41
    // where s = sum(MATRIX_CIRC_MDS_12_SML) = 460.
    let dot = u[0].as_noncanonical_u32() as u64 * v[0]
        + u[1].as_noncanonical_u32() as u64 * v[1]
        + u[2].as_noncanonical_u32() as u64 * v[2]
        + u[3].as_noncanonical_u32() as u64 * v[3]
        + u[4].as_noncanonical_u32() as u64 * v[4]
        + u[5].as_noncanonical_u32() as u64 * v[5]
        + u[6].as_noncanonical_u32() as u64 * v[6]
        + u[7].as_noncanonical_u32() as u64 * v[7]
        + u[8].as_noncanonical_u32() as u64 * v[8]
        + u[9].as_noncanonical_u32() as u64 * v[9]
        + u[10].as_noncanonical_u32() as u64 * v[10]
        + u[11].as_noncanonical_u32() as u64 * v[11];
    F::from_wrapped_u64(dot)
}

pub(crate) fn apply_circulant_prime32_8_sml<F: PrimeField32>(input: [F; 8]) -> [F; 8] {
    const N: usize = 8;
    let mut output = [F::ZERO; N];

    const MAT_0: [u64; N] = MATRIX_CIRC_MDS_8_SML;
    output[0] = dot_vec_prime32_8_sml(&input, MAT_0);
    const MAT_1: [u64; N] = rotate_right(MATRIX_CIRC_MDS_8_SML, 1);
    output[1] = dot_vec_prime32_8_sml(&input, MAT_1);
    const MAT_2: [u64; N] = rotate_right(MATRIX_CIRC_MDS_8_SML, 2);
    output[2] = dot_vec_prime32_8_sml(&input, MAT_2);
    const MAT_3: [u64; N] = rotate_right(MATRIX_CIRC_MDS_8_SML, 3);
    output[3] = dot_vec_prime32_8_sml(&input, MAT_3);
    const MAT_4: [u64; N] = rotate_right(MATRIX_CIRC_MDS_8_SML, 4);
    output[4] = dot_vec_prime32_8_sml(&input, MAT_4);
    const MAT_5: [u64; N] = rotate_right(MATRIX_CIRC_MDS_8_SML, 5);
    output[5] = dot_vec_prime32_8_sml(&input, MAT_5);
    const MAT_6: [u64; N] = rotate_right(MATRIX_CIRC_MDS_8_SML, 6);
    output[6] = dot_vec_prime32_8_sml(&input, MAT_6);
    const MAT_7: [u64; N] = rotate_right(MATRIX_CIRC_MDS_8_SML, 7);
    output[7] = dot_vec_prime32_8_sml(&input, MAT_7);

    output
}

pub(crate) fn apply_circulant_prime32_12_sml<F: PrimeField32>(input: [F; 12]) -> [F; 12] {
    const N: usize = 12;
    let mut output = [F::ZERO; N];

    const MAT_0: [u64; N] = MATRIX_CIRC_MDS_12_SML;
    output[0] = dot_vec_prime32_12_sml(&input, MAT_0);
    const MAT_1: [u64; N] = rotate_right(MATRIX_CIRC_MDS_12_SML, 1);
    output[1] = dot_vec_prime32_12_sml(&input, MAT_1);
    const MAT_2: [u64; N] = rotate_right(MATRIX_CIRC_MDS_12_SML, 2);
    output[2] = dot_vec_prime32_12_sml(&input, MAT_2);
    const MAT_3: [u64; N] = rotate_right(MATRIX_CIRC_MDS_12_SML, 3);
    output[3] = dot_vec_prime32_12_sml(&input, MAT_3);
    const MAT_4: [u64; N] = rotate_right(MATRIX_CIRC_MDS_12_SML, 4);
    output[4] = dot_vec_prime32_12_sml(&input, MAT_4);
    const MAT_5: [u64; N] = rotate_right(MATRIX_CIRC_MDS_12_SML, 5);
    output[5] = dot_vec_prime32_12_sml(&input, MAT_5);
    const MAT_6: [u64; N] = rotate_right(MATRIX_CIRC_MDS_12_SML, 6);
    output[6] = dot_vec_prime32_12_sml(&input, MAT_6);
    const MAT_7: [u64; N] = rotate_right(MATRIX_CIRC_MDS_12_SML, 7);
    output[7] = dot_vec_prime32_12_sml(&input, MAT_7);
    const MAT_8: [u64; N] = rotate_right(MATRIX_CIRC_MDS_12_SML, 8);
    output[8] = dot_vec_prime32_12_sml(&input, MAT_8);
    const MAT_9: [u64; N] = rotate_right(MATRIX_CIRC_MDS_12_SML, 9);
    output[9] = dot_vec_prime32_12_sml(&input, MAT_9);
    const MAT_10: [u64; N] = rotate_right(MATRIX_CIRC_MDS_12_SML, 10);
    output[10] = dot_vec_prime32_12_sml(&input, MAT_10);
    const MAT_11: [u64; N] = rotate_right(MATRIX_CIRC_MDS_12_SML, 11);
    output[11] = dot_vec_prime32_12_sml(&input, MAT_11);

    output
}

#[inline]
fn dot_vec_prime64_8_sml<F: PrimeField64>(u: &[F; 8], v: [u64; 8]) -> F {
    // An unreduced dot product is bounded above by s * (2^64 - 1) < 2^70
    // where s = sum(MATRIX_CIRC_MDS_8_SML) = 33.
    let dot = u[0].as_noncanonical_u64() as u128 * v[0] as u128
        + u[1].as_noncanonical_u64() as u128 * v[1] as u128
        + u[2].as_noncanonical_u64() as u128 * v[2] as u128
        + u[3].as_noncanonical_u64() as u128 * v[3] as u128
        + u[4].as_noncanonical_u64() as u128 * v[4] as u128
        + u[5].as_noncanonical_u64() as u128 * v[5] as u128
        + u[6].as_noncanonical_u64() as u128 * v[6] as u128
        + u[7].as_noncanonical_u64() as u128 * v[7] as u128;
    F::from_wrapped_u128(dot)
}

#[inline]
fn dot_vec_prime64_12_sml<F: PrimeField64>(u: &[F; 12], v: [u64; 12]) -> F {
    // An unreduced dot product is bounded above by s * (2^64 - 1) < 2^73
    // where s = sum(MATRIX_CIRC_MDS_12_SML) = 460.
    let dot = u[0].as_noncanonical_u64() as u128 * v[0] as u128
        + u[1].as_noncanonical_u64() as u128 * v[1] as u128
        + u[2].as_noncanonical_u64() as u128 * v[2] as u128
        + u[3].as_noncanonical_u64() as u128 * v[3] as u128
        + u[4].as_noncanonical_u64() as u128 * v[4] as u128
        + u[5].as_noncanonical_u64() as u128 * v[5] as u128
        + u[6].as_noncanonical_u64() as u128 * v[6] as u128
        + u[7].as_noncanonical_u64() as u128 * v[7] as u128
        + u[8].as_noncanonical_u64() as u128 * v[8] as u128
        + u[9].as_noncanonical_u64() as u128 * v[9] as u128
        + u[10].as_noncanonical_u64() as u128 * v[10] as u128
        + u[11].as_noncanonical_u64() as u128 * v[11] as u128;
    F::from_wrapped_u128(dot)
}

pub(crate) fn apply_circulant_prime64_8_sml<F: PrimeField64>(input: [F; 8]) -> [F; 8] {
    const N: usize = 8;
    let mut output = [F::ZERO; N];

    const MAT_0: [u64; N] = MATRIX_CIRC_MDS_8_SML;
    output[0] = dot_vec_prime64_8_sml(&input, MAT_0);
    const MAT_1: [u64; N] = rotate_right(MATRIX_CIRC_MDS_8_SML, 1);
    output[1] = dot_vec_prime64_8_sml(&input, MAT_1);
    const MAT_2: [u64; N] = rotate_right(MATRIX_CIRC_MDS_8_SML, 2);
    output[2] = dot_vec_prime64_8_sml(&input, MAT_2);
    const MAT_3: [u64; N] = rotate_right(MATRIX_CIRC_MDS_8_SML, 3);
    output[3] = dot_vec_prime64_8_sml(&input, MAT_3);
    const MAT_4: [u64; N] = rotate_right(MATRIX_CIRC_MDS_8_SML, 4);
    output[4] = dot_vec_prime64_8_sml(&input, MAT_4);
    const MAT_5: [u64; N] = rotate_right(MATRIX_CIRC_MDS_8_SML, 5);
    output[5] = dot_vec_prime64_8_sml(&input, MAT_5);
    const MAT_6: [u64; N] = rotate_right(MATRIX_CIRC_MDS_8_SML, 6);
    output[6] = dot_vec_prime64_8_sml(&input, MAT_6);
    const MAT_7: [u64; N] = rotate_right(MATRIX_CIRC_MDS_8_SML, 7);
    output[7] = dot_vec_prime64_8_sml(&input, MAT_7);

    output
}

pub(crate) fn apply_circulant_prime64_12_sml<F: PrimeField64>(input: [F; 12]) -> [F; 12] {
    const N: usize = 12;
    let mut output = [F::ZERO; N];

    const MAT_0: [u64; N] = MATRIX_CIRC_MDS_12_SML;
    output[0] = dot_vec_prime64_12_sml(&input, MAT_0);
    const MAT_1: [u64; N] = rotate_right(MATRIX_CIRC_MDS_12_SML, 1);
    output[1] = dot_vec_prime64_12_sml(&input, MAT_1);
    const MAT_2: [u64; N] = rotate_right(MATRIX_CIRC_MDS_12_SML, 2);
    output[2] = dot_vec_prime64_12_sml(&input, MAT_2);
    const MAT_3: [u64; N] = rotate_right(MATRIX_CIRC_MDS_12_SML, 3);
    output[3] = dot_vec_prime64_12_sml(&input, MAT_3);
    const MAT_4: [u64; N] = rotate_right(MATRIX_CIRC_MDS_12_SML, 4);
    output[4] = dot_vec_prime64_12_sml(&input, MAT_4);
    const MAT_5: [u64; N] = rotate_right(MATRIX_CIRC_MDS_12_SML, 5);
    output[5] = dot_vec_prime64_12_sml(&input, MAT_5);
    const MAT_6: [u64; N] = rotate_right(MATRIX_CIRC_MDS_12_SML, 6);
    output[6] = dot_vec_prime64_12_sml(&input, MAT_6);
    const MAT_7: [u64; N] = rotate_right(MATRIX_CIRC_MDS_12_SML, 7);
    output[7] = dot_vec_prime64_12_sml(&input, MAT_7);
    const MAT_8: [u64; N] = rotate_right(MATRIX_CIRC_MDS_12_SML, 8);
    output[8] = dot_vec_prime64_12_sml(&input, MAT_8);
    const MAT_9: [u64; N] = rotate_right(MATRIX_CIRC_MDS_12_SML, 9);
    output[9] = dot_vec_prime64_12_sml(&input, MAT_9);
    const MAT_10: [u64; N] = rotate_right(MATRIX_CIRC_MDS_12_SML, 10);
    output[10] = dot_vec_prime64_12_sml(&input, MAT_10);
    const MAT_11: [u64; N] = rotate_right(MATRIX_CIRC_MDS_12_SML, 11);
    output[11] = dot_vec_prime64_12_sml(&input, MAT_11);

    output
}

/// Given the first row of a circulant matrix, return the first column
/// of that circulant matrix. For example, v = [0, 1, 2, 3, 4, 5],
/// then output = [0, 5, 4, 3, 2, 1], i.e. the first element is the
/// same and the other elements are reversed.
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
/// the row as an array, you can obtain the column with `first_row_to_first_column()`.
pub(crate) fn apply_circulant_fft<F: TwoAdicField, const N: usize, FFT: TwoAdicSubgroupDft<F>>(
    fft: FFT,
    column: [u64; N],
    input: &[F; N],
) -> [F; N] {
    let column = column.map(F::from_canonical_u64).to_vec();
    let matrix = fft.dft_batch(RowMajorMatrix::new_col(column));
    let input = fft.dft_batch(RowMajorMatrix::new_col(input.to_vec()));

    // point-wise product
    let product = matrix
        .values
        .iter()
        .zip(input.values)
        .map(|(&x, y)| x * y)
        .collect::<Vec<_>>();

    let output = fft.idft_batch(RowMajorMatrix::new_col(product));
    output.values.try_into().unwrap()
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

        for i in 0..output.len() {
            assert_eq!(rotate_right(input, i), output[i]);
        }
    }
}
