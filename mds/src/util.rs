use p3_dft::TwoAdicSubgroupDft;
use p3_field::{AbstractField, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;

// NB: These four are MDS for M31, BabyBear and Goldilocks
//const MATRIX_CIRC_MDS_8_2EXP: [u64; 8] = [1, 1, 2, 1, 8, 32, 4, 256];
pub(crate) const MATRIX_CIRC_MDS_8_SML: [u64; 8] = [4, 1, 2, 9, 10, 5, 1, 1];

//const MATRIX_CIRC_MDS_12_2EXP: [u64; 12] = [1, 1, 2, 1, 8, 32, 2, 256, 4096, 8, 65536, 1024];
pub(crate) const MATRIX_CIRC_MDS_12_SML: [u64; 12] = [9, 7, 4, 1, 16, 2, 256, 128, 3, 32, 1, 1];

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
