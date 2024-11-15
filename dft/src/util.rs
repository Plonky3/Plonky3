use core::borrow::BorrowMut;

use p3_field::{Field, PrimeField};
use p3_matrix::dense::{DenseMatrix, DenseStorage, RowMajorMatrix};
use p3_matrix::Matrix;
use p3_util::log2_strict_usize;
use tracing::instrument;

/// Divide each coefficient of the given matrix by its height.
///
/// We assume that the height is a power of 2.
#[instrument(skip_all, fields(dims = %mat.dimensions()))]
pub fn divide_by_height<F: Field, S: DenseStorage<F> + BorrowMut<[F]>>(
    mat: &mut DenseMatrix<F, S>,
) {
    let log_h = log2_strict_usize(mat.height());
    let h_inv = F::from_char(F::Char::inv_power_of_2(log_h));
    mat.scale(h_inv);
}

/// Multiply each element of row `i` of `mat` by `shift**i`.
pub(crate) fn coset_shift_cols<F: Field>(mat: &mut RowMajorMatrix<F>, shift: F) {
    mat.rows_mut()
        .zip(shift.powers())
        .for_each(|(row, weight)| {
            row.iter_mut().for_each(|coeff| {
                *coeff *= weight;
            })
        });
}
