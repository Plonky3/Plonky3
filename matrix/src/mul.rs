use crate::dense::{RowMajorMatrix, RowMajorMatrixView, RowMajorMatrixViewMut};
use crate::sparse::CsrMatrix;
use crate::Matrix;
use alloc::vec;
use p3_field::Field;

/// Compute `C = A * B`, where `A` in a CSR matrix and `B` is a dense matrix.
///
/// # Panics
/// Panics if dimensions of input matrices don't match.
pub fn mul_csr_dense_v2<F: Field, B: Matrix<F>>(a: &CsrMatrix<F>, b: &B) -> RowMajorMatrix<F> {
    assert_eq!(a.width(), b.height(), "A, B dimensions don't match");
    let c_width = b.width();
    let c_height = a.height();
    let mut c = RowMajorMatrix::new(vec![F::ZERO; c_width * c_height], c_width);

    for a_row_idx in 0..a.height() {
        let c_row = c.row_mut(a_row_idx);
        for &(a_col_idx, a_val) in a.row(a_row_idx) {
            F::add_scaled_slice_in_place(c_row, b.row(a_col_idx), a_val);
        }
    }

    c
}
