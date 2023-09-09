use p3_field::Field;
use p3_matrix::dense::RowMajorMatrixViewMut;
use p3_matrix::Matrix;

/// DIT butterfly.
#[inline]
pub(crate) fn dit_butterfly<F: Field>(
    mat: &mut RowMajorMatrixViewMut<F>,
    row_1: usize,
    row_2: usize,
    twiddle: F,
) {
    let ((prefix_1, shorts_1, suffix_1), (prefix_2, shorts_2, suffix_2)) =
        mat.packing_aligned_rows(row_1, row_2);

    for (x_1, x_2) in prefix_1.iter_mut().zip(prefix_2) {
        let x_2_twiddle = *x_2 * twiddle;
        let sum = *x_1 + x_2_twiddle;
        let diff = *x_1 - x_2_twiddle;
        *x_1 = sum;
        *x_2 = diff;
    }
    for (x_1, x_2) in shorts_1.iter_mut().zip(shorts_2) {
        let x_2_twiddle = *x_2 * twiddle;
        let sum = *x_1 + x_2_twiddle;
        let diff = *x_1 - x_2_twiddle;
        *x_1 = sum;
        *x_2 = diff;
    }
    for (x_1, x_2) in suffix_1.iter_mut().zip(suffix_2) {
        let x_2_twiddle = *x_2 * twiddle;
        let sum = *x_1 + x_2_twiddle;
        let diff = *x_1 - x_2_twiddle;
        *x_1 = sum;
        *x_2 = diff;
    }
}

/// DIF butterfly.
#[inline]
pub(crate) fn dif_butterfly<F: Field>(
    mat: &mut RowMajorMatrixViewMut<F>,
    row_1: usize,
    row_2: usize,
    twiddle: F,
) {
    debug_assert!(row_1 < mat.height());
    debug_assert!(row_2 < mat.height());

    let ((prefix_1, shorts_1, suffix_1), (prefix_2, shorts_2, suffix_2)) =
        mat.packing_aligned_rows(row_1, row_2);

    for (x_1, x_2) in prefix_1.iter_mut().zip(prefix_2) {
        let sum = *x_1 + *x_2;
        let diff = *x_1 - *x_2;
        *x_1 = sum;
        *x_2 = diff * twiddle;
    }
    for (x_1, x_2) in shorts_1.iter_mut().zip(shorts_2) {
        let sum: F::Packing = *x_1 + *x_2;
        let diff: F::Packing = *x_1 - *x_2;
        *x_1 = sum;
        *x_2 = diff * twiddle;
    }
    for (x_1, x_2) in suffix_1.iter_mut().zip(suffix_2) {
        let sum = *x_1 + *x_2;
        let diff = *x_1 - *x_2;
        *x_1 = sum;
        *x_2 = diff * twiddle;
    }
}

/// Butterfly with twiddle factor 1 (works in either DIT or DIF).
#[inline]
pub(crate) fn twiddle_free_butterfly<F: Field>(
    mat: &mut RowMajorMatrixViewMut<F>,
    row_1: usize,
    row_2: usize,
) {
    let ((prefix_1, shorts_1, suffix_1), (prefix_2, shorts_2, suffix_2)) =
        mat.packing_aligned_rows(row_1, row_2);

    for (x_1, x_2) in prefix_1.iter_mut().zip(prefix_2) {
        let sum = *x_1 + *x_2;
        let diff = *x_1 - *x_2;
        *x_1 = sum;
        *x_2 = diff;
    }
    for (x_1, x_2) in shorts_1.iter_mut().zip(shorts_2) {
        let sum = *x_1 + *x_2;
        let diff = *x_1 - *x_2;
        *x_1 = sum;
        *x_2 = diff;
    }
    for (x_1, x_2) in suffix_1.iter_mut().zip(suffix_2) {
        let sum = *x_1 + *x_2;
        let diff = *x_1 - *x_2;
        *x_1 = sum;
        *x_2 = diff;
    }
}
