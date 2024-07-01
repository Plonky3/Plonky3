use alloc::vec;
use core::borrow::BorrowMut;

use p3_field::Field;
use p3_matrix::dense::{DenseMatrix, DenseStorage, RowMajorMatrix};
use p3_matrix::Matrix;
use tracing::instrument;

/// Divide each coefficient of the given matrix by its height.
#[instrument(skip_all, fields(dims = %mat.dimensions()))]
pub fn divide_by_height<F: Field, S: DenseStorage<F> + BorrowMut<[F]>>(
    mat: &mut DenseMatrix<F, S>,
) {
    mat.scale(F::from_canonical_usize(mat.height()).inverse())
}

/// Append zeros to the "end" of the given matrix, except that the matrix is in bit-reversed order,
/// so in actuality we're interleaving zero rows.
#[inline]
pub fn bit_reversed_zero_pad<F: Field>(mat: &mut RowMajorMatrix<F>, added_bits: usize) {
    if added_bits == 0 {
        return;
    }

    // This is equivalent to:
    //     reverse_matrix_index_bits(mat);
    //     mat
    //         .values
    //         .resize(mat.values.len() << added_bits, F::zero());
    //     reverse_matrix_index_bits(mat);
    // But rather than implement it with bit reversals, we directly construct the resulting matrix,
    // whose rows are zero except for rows whose low `added_bits` bits are zero.

    let w = mat.width;
    let mut values = vec![F::zero(); mat.values.len() << added_bits];
    for i in (0..mat.values.len()).step_by(w) {
        values[(i << added_bits)..((i << added_bits) + w)].copy_from_slice(&mat.values[i..i + w]);
    }
    *mat = RowMajorMatrix::new(values, w);
}
