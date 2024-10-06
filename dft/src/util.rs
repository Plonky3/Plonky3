use core::borrow::BorrowMut;

use p3_field::Field;
use p3_matrix::dense::{DenseMatrix, DenseStorage, RowMajorMatrix};
use p3_matrix::Matrix;
use p3_util::reverse_bits;
use tracing::instrument;

/// Divide each coefficient of the given matrix by its height.
#[instrument(skip_all, fields(dims = %mat.dimensions()))]
pub fn divide_by_height<F: Field, S: DenseStorage<F> + BorrowMut<[F]>>(
    mat: &mut DenseMatrix<F, S>,
) {
    mat.scale(F::from_canonical_usize(mat.height()).inverse())
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

/// Multiply each element of row `i` of `mat` by `shift**i`.
#[instrument(skip_all, fields(dims = %mat.dimensions()))]
pub(crate) fn coset_shift_cols_bitrev<F: Field>(mat: &mut RowMajorMatrix<F>, shift: F) {
    let mut shift_powers = shift.powers();
    let h = mat.height();
    for r in 0..h {
        let weight = shift_powers.next().unwrap();
        let r_bitrev = reverse_bits(r, h);
        mat.row_mut(r_bitrev)
            .iter_mut()
            .for_each(|coeff| *coeff *= weight);
    }
}
