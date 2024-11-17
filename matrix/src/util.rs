use core::borrow::BorrowMut;

use p3_maybe_rayon::prelude::*;
use p3_util::{log2_strict_usize, reverse_bits_len, MutPtr};
use tracing::instrument;

use crate::dense::{DenseMatrix, DenseStorage, RowMajorMatrix};
use crate::Matrix;

#[instrument(level = "debug", skip_all)]
pub fn reverse_matrix_index_bits<'a, F, S>(mat: &mut DenseMatrix<F, S>)
where
    F: Clone + Send + Sync + 'a,
    S: DenseStorage<F> + BorrowMut<[F]>,
{
    let w = mat.width();
    let h = mat.height();
    let log_h = log2_strict_usize(h);
    let values = mat.values.borrow_mut().as_mut_ptr() as usize;

    (0..h).into_par_iter().for_each(|i| {
        let values = values as *mut F;
        let j = reverse_bits_len(i, log_h);
        if i < j {
            unsafe { swap_rows_raw(values, w, i, j) };
        }
    });
}

/// Assumes `start < stride`.
#[instrument(level = "debug", skip_all)]
pub fn reverse_matrix_index_bits_strided<'a, F>(
    mat: MutPtr<F>,
    w: usize,
    h: usize,
    start: usize,
    stride_bits: usize,
) where
    F: Clone + Send + Sync + 'a,
{
    let h = h >> stride_bits;
    let log_h = log2_strict_usize(h);

    (0..h).for_each(|i| {
        let j = reverse_bits_len(i, log_h);
        if i < j {
            let ii = start + (i << stride_bits);
            let jj = start + (j << stride_bits);
            unsafe { swap_rows_raw(mat.0, w, ii, jj) };
        }
    });
}

/// Assumes `i < j`.
#[inline]
pub fn swap_rows<F: Clone + Send + Sync>(mat: &mut RowMajorMatrix<F>, i: usize, j: usize) {
    let w = mat.width();
    let (upper, lower) = mat.values.split_at_mut(j * w);
    let row_i = &mut upper[i * w..(i + 1) * w];
    let row_j = &mut lower[..w];
    row_i.swap_with_slice(row_j);
}

/// Assumes `i < j`.
///
/// SAFETY: The caller must ensure `i < j < h`, where `h` is the height of the matrix.
#[inline]
pub(crate) unsafe fn swap_rows_raw<F>(mat: *mut F, w: usize, i: usize, j: usize) {
    // core::ptr::swap_nonoverlapping(mat.add(i * w), mat.add(j * w), w);
    let row_i = core::slice::from_raw_parts_mut(mat.add(i * w), w);
    let row_j = core::slice::from_raw_parts_mut(mat.add(j * w), w);
    row_i.swap_with_slice(row_j);
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use itertools::Itertools;

    use super::*;
    use crate::dense::RowMajorMatrix;

    #[test]
    fn test_reverse_matrix_index_bits_strided() {
        // We go from
        //      0  1  2  3
        //      4  5  6  7
        //      8  9 10 11
        //     12 13 14 15
        // to
        //      0  1  2  3
        //      4  9  6  7
        //      8  5 10 11
        //     12 13 14 15
        let values = (0..16).collect_vec();
        let mut mat = RowMajorMatrix::new(values, 1);
        reverse_matrix_index_bits_strided(MutPtr(mat.values.as_mut_ptr()), 1, 16, 1, 2);
        assert_eq!(
            mat.values,
            vec![0, 1, 2, 3, 4, 9, 6, 7, 8, 5, 10, 11, 12, 13, 14, 15,]
        );
    }
}
