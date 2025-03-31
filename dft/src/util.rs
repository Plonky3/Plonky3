use core::borrow::BorrowMut;

use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use p3_matrix::dense::{DenseMatrix, DenseStorage, RowMajorMatrix};
use p3_util::log2_strict_usize;
use tracing::instrument;

/// Divide each coefficient of the given matrix by its height.
///
/// # Panics
///
/// Panics if the height of the matrix is not a power of two.
#[instrument(skip_all, fields(dims = %mat.dimensions()))]
pub fn divide_by_height<F: Field, S: DenseStorage<F> + BorrowMut<[F]>>(
    mat: &mut DenseMatrix<F, S>,
) {
    let h = mat.height();
    let log_h = log2_strict_usize(h);
    // It's cheaper to use div_2exp_u64 as this usually avoids an inversion.
    // It's also cheaper to work in the PrimeSubfield whenever possible.
    let h_inv_subfield = F::PrimeSubfield::ONE.div_2exp_u64(log_h as u64);
    let h_inv = F::from_prime_subfield(h_inv_subfield);
    mat.scale(h_inv)
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
