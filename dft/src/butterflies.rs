use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;

/// DIT butterfly.
#[inline]
pub fn dit_butterfly<F: Field>(
    mat: &mut RowMajorMatrix<F>,
    row_1: usize,
    row_2: usize,
    twiddle: F,
) {
    let RowMajorMatrix { values, width } = mat;
    for col in 0..*width {
        let idx_1 = row_1 * *width + col;
        let idx_2 = row_2 * *width + col;
        let val_1 = values[idx_1];
        let val_2 = values[idx_2] * twiddle;
        values[idx_1] = val_1 + val_2;
        values[idx_2] = val_1 - val_2;
    }
}

/// DIF butterfly.
#[inline]
pub fn dif_butterfly<F: Field>(
    mat: &mut RowMajorMatrix<F>,
    row_1: usize,
    row_2: usize,
    twiddle: F,
) {
    let RowMajorMatrix { values, width } = mat;
    for col in 0..*width {
        let idx_1 = row_1 * *width + col;
        let idx_2 = row_2 * *width + col;
        let val_1 = values[idx_1];
        let val_2 = values[idx_2];
        values[idx_1] = val_1 + val_2;
        values[idx_2] = (val_1 - val_2) * twiddle;
    }
}
