use core::borrow::BorrowMut;

use p3_field::Field;
use p3_field::integers::QuotientMap;
use p3_matrix::Matrix;
use p3_matrix::dense::{DenseMatrix, DenseStorage, RowMajorMatrix};
use tracing::instrument;

/// Divide each coefficient of the given matrix by its height.
#[instrument(skip_all, fields(dims = %mat.dimensions()))]
pub fn divide_by_height<F: Field, S: DenseStorage<F> + BorrowMut<[F]>>(
    mat: &mut DenseMatrix<F, S>,
) {
    // If F isn't a PrimeField, (and is thus an extension field) it's much cheaper to
    // invert in F::PrimeSubfield.
    let h_inv_subfield = F::PrimeSubfield::from_int(mat.height()).inverse();
    mat.scale(F::from_prime_subfield(h_inv_subfield))
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
