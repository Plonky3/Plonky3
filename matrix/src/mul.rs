use crate::dense::{DenseMatrixView, DenseMatrixViewMut};
use crate::sparse::CsrMatrix;
use crate::Matrix;
use p3_field::field::Field;

/// Compute `C = A * B`, where `A` in a CSR matrix and `B` is a dense matrix.
/// This assumes that `C` is initially filled with zeros.
pub fn mul_csr_dense<F: Field>(
    a: &CsrMatrix<F>,
    b: DenseMatrixView<F>,
    c: &mut DenseMatrixViewMut<F>,
) {
    assert_eq!(a.width(), b.height(), "A, B dimensions don't match");
    assert_eq!(a.height(), c.height(), "A, C dimensions don't match");
    assert_eq!(b.width(), c.width(), "B, C dimensions don't match");

    for a_row_idx in 0..a.height() {
        let c_row = c.row_mut(a_row_idx);
        for &(a_col_idx, a_val) in a.row(a_row_idx) {
            F::add_scaled_slice_in_place(c_row, b.row(a_col_idx), a_val);
        }
    }
}
