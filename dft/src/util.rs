use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;

pub(crate) fn reverse_slice_index_bits<F>(vals: &mut [F]) {
    let n = vals.len();
    if n == 0 {
        return;
    }

    assert!(n.is_power_of_two());
    for i in 0..n {
        let j = reverse_bits(i, n);
        if i < j {
            vals.swap(i, j);
        }
    }
}

pub(crate) fn reverse_matrix_index_bits<F>(vals: &mut RowMajorMatrix<F>) {
    let h = vals.height();
    assert!(h.is_power_of_two());
    for i in 0..h {
        let j = reverse_bits(i, h);
        if i < j {
            swap_rows(vals, i, j);
        }
    }
}

/// Assumes `i < j`.
pub(crate) fn swap_rows<F>(mat: &mut RowMajorMatrix<F>, i: usize, j: usize) {
    let w = mat.width();
    let (upper, lower) = mat.values.split_at_mut(j * w);
    let row_i = &mut upper[i * w..(i + 1) * w];
    let row_j = &mut lower[..w];
    row_i.swap_with_slice(row_j);
}

#[inline]
pub(crate) fn reverse_bits(x: usize, n: usize) -> usize {
    debug_assert!(n.is_power_of_two());
    // NB: The only reason we need overflowing_shr() here as opposed
    // to plain '>>' is to accommodate the case n == num_bits == 0,
    // which would become `0 >> 64`. Rust thinks that any shift of 64
    // bits causes overflow, even when the argument is zero.
    x.reverse_bits()
        .overflowing_shr(usize::BITS - n.trailing_zeros())
        .0
}

/// Divide each coefficient of the given matrix by its height.
pub(crate) fn divide_by_height<F: Field>(mat: &mut RowMajorMatrix<F>) {
    let h = mat.height();
    let h_inv = F::from_canonical_usize(h).inverse();
    mat.values.iter_mut().for_each(|coeff| *coeff *= h_inv);
}
