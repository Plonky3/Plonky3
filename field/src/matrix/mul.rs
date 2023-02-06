use crate::field::Field;
use crate::matrix::dense::{DenseMatrixView, DenseMatrixViewMut};
use crate::matrix::sparse::CsrMatrix;
use crate::matrix::Matrix;

/// Compute `C = A * B`, where `A` in a CSR matrix and `B` is a dense matrix.
/// This assumes that `C` is initially filled with zeros.
pub fn mul_csr_dense<F: Field>(
    a: &CsrMatrix<F>,
    b: DenseMatrixView<F>,
    c: &mut DenseMatrixViewMut<F>,
) {
    debug_assert_eq!(b.width(), c.width());
    for a_row_idx in 0..a.height() {
        let c_row = c.row_mut(a_row_idx);
        for &(a_col_idx, a_val) in a.row(a_row_idx) {
            F::add_scaled_slice_in_place(c_row, b.row(a_col_idx), a_val);
        }
    }
}
