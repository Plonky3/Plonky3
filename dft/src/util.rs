use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;

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

pub(crate) fn reverse_bits(x: usize, n: usize) -> usize {
    // NB: The only reason we need overflowing_shr() here as opposed
    // to plain '>>' is to accommodate the case n == num_bits == 0,
    // which would become `0 >> 64`. Rust thinks that any shift of 64
    // bits causes overflow, even when the argument is zero.
    x.reverse_bits()
        .overflowing_shr(usize::BITS - n.trailing_zeros())
        .0
}
