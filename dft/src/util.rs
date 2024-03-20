use alloc::vec;

use p3_field::{Field, PackedValue};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::MatrixRowSlicesMut;

/// Divide each coefficient of the given matrix by its height.
pub fn divide_by_height<F: Field>(mat: &mut impl MatrixRowSlicesMut<F>) {
    let h = mat.height();
    let h_inv = F::from_canonical_usize(h).inverse();
    for r in 0..h {
        let (packed, suffix) = F::Packing::pack_slice_with_suffix_mut(mat.row_slice_mut(r));
        packed.iter_mut().for_each(|x| *x *= h_inv);
        suffix.iter_mut().for_each(|x| *x *= h_inv);
    }
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
