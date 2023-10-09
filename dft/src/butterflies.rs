use p3_field::Field;
use p3_matrix::dense::RowMajorMatrixViewMut;

/// DIF butterfly operation on rows.
#[inline]
pub(crate) fn dif_butterfly_on_rows<F: Field>(row_1: &mut [F], row_2: &mut [F], twiddle: F) {
    let ((prefix_1, shorts_1, suffix_1), (prefix_2, shorts_2, suffix_2)) = unsafe {
        (
            row_1.align_to_mut::<F::Packing>(),
            row_2.align_to_mut::<F::Packing>(),
        )
    };

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

/// Butterfly with twiddle factor 1 on rows (works in either DIT or DIF).
#[inline]
pub(crate) fn twiddle_free_butterfly_on_rows<F: Field>(row_1: &mut [F], row_2: &mut [F]) {
    let ((prefix_1, shorts_1, suffix_1), (prefix_2, shorts_2, suffix_2)) = unsafe {
        (
            row_1.align_to_mut::<F::Packing>(),
            row_2.align_to_mut::<F::Packing>(),
        )
    };

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

/// DIT butterfly operation on rows.
#[inline]
pub(crate) fn dit_butterfly_on_rows<F: Field>(row_1: &mut [F], row_2: &mut [F], twiddle: F) {
    let ((prefix_1, shorts_1, suffix_1), (prefix_2, shorts_2, suffix_2)) = unsafe {
        (
            row_1.align_to_mut::<F::Packing>(),
            row_2.align_to_mut::<F::Packing>(),
        )
    };
    dit(
        prefix_1, shorts_1, suffix_1, prefix_2, shorts_2, suffix_2, twiddle,
    )
}

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
    dit(
        prefix_1, shorts_1, suffix_1, prefix_2, shorts_2, suffix_2, twiddle,
    )
}

fn dit<F: Field>(
    prefix_1: &mut [F],
    shorts_1: &mut [<F as Field>::Packing],
    suffix_1: &mut [F],
    prefix_2: &mut [F],
    shorts_2: &mut [<F as Field>::Packing],
    suffix_2: &mut [F],
    twiddle: F,
) {
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
