use core::borrow::BorrowMut;

use p3_maybe_rayon::prelude::*;
use p3_util::{log2_strict_usize, reverse_bits_len};
use tracing::instrument;

use crate::Matrix;
use crate::dense::{DenseMatrix, DenseStorage, RowMajorMatrix};

/// Reverse the order of matrix rows based on the bit-reversal of their indices.
///
/// Given a matrix `mat` of height `h = 2^k`, this function rearranges its rows by
/// reversing the binary representation of each row index. For example, if `h = 8` (i.e., 3 bits):
///
/// ```text
/// Original Index  Binary   Reversed   Target Index
/// --------------  -------  ---------  -------------
///      0          000      000        0
///      1          001      100        4
///      2          010      010        2
///      3          011      110        6
///      4          100      001        1
///      5          101      101        5
///      6          110      011        3
///      7          111      111        7
/// ```
///
/// The transformation is performed in-place.
///
/// # Panics
/// Panics if the height of the matrix is not a power of two.
///
/// # Arguments
/// - `mat`: The matrix whose rows should be reordered.
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

    // SAFETY: Due to the i < j check, we are guaranteed that `swap_rows_raw
    // will never try and access a particular slice of data more than once
    // across all parallel threads. Hence the following code is safe and does
    // not trigger undefined behaviour.
    (0..h).into_par_iter().for_each(|i| {
        let values = values as *mut F;
        let j = reverse_bits_len(i, log_h);
        if i < j {
            unsafe { swap_rows_raw(values, w, i, j) };
        }
    });
}

/// Swap two rows `i` and `j` in a [`RowMajorMatrix`].
///
/// # Panics
/// Panics if the indices are out of bounds or not ordered as `i < j`.
///
/// # Arguments
/// - `mat`: The matrix to modify.
/// - `i`: The first row index (must be less than `j`).
/// - `j`: The second row index.
pub fn swap_rows<F: Clone + Send + Sync>(mat: &mut RowMajorMatrix<F>, i: usize, j: usize) {
    let w = mat.width();
    let (upper, lower) = mat.values.split_at_mut(j * w);
    let row_i = &mut upper[i * w..(i + 1) * w];
    let row_j = &mut lower[..w];
    row_i.swap_with_slice(row_j);
}

/// Swap two rows `i` and `j` in-place using raw pointer access.
///
/// This function is equivalent to [`swap_rows`] but uses unsafe raw pointer math for better performance.
///
/// # Safety
/// - The caller must ensure `i < j < h`, where `h` is the height of the matrix.
/// - The pointer must point to a vector corresponding to a matrix of width `w`.
///
/// # Arguments
/// - `mat`: A mutable pointer to the underlying matrix data.
/// - `w`: The matrix width (number of columns).
/// - `i`: The first row index.
/// - `j`: The second row index.
unsafe fn swap_rows_raw<F>(mat: *mut F, w: usize, i: usize, j: usize) {
    unsafe {
        let row_i = core::slice::from_raw_parts_mut(mat.add(i * w), w);
        let row_j = core::slice::from_raw_parts_mut(mat.add(j * w), w);
        row_i.swap_with_slice(row_j);
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::dense::RowMajorMatrix;

    #[test]
    fn test_swap_rows_basic() {
        let mut matrix = RowMajorMatrix::new(
            vec![
                1, 2, 3, // row 0
                4, 5, 6, // row 1
                7, 8, 9, // row 2
                10, 11, 12, // row 3
            ],
            3,
        );

        // Swap rows 0 and 2
        swap_rows(&mut matrix, 0, 2);

        assert_eq!(
            matrix.values,
            vec![
                7, 8, 9, // row 0 (was row 2)
                4, 5, 6, // row 1 (unchanged)
                1, 2, 3, // row 2 (was row 0)
                10, 11, 12, // row 3 (unchanged)
            ]
        );
    }

    #[test]
    fn test_swap_rows_raw_basic() {
        let mut matrix = RowMajorMatrix::new(
            vec![
                1, 2, 3, // row 0
                4, 5, 6, // row 1
                7, 8, 9, // row 2
            ],
            3,
        );
        let ptr = matrix.values.as_mut_ptr();
        unsafe {
            swap_rows_raw(ptr, matrix.width(), 0, 2);
        }

        assert_eq!(
            matrix.values,
            vec![
                7, 8, 9, // row 0 (was row 2)
                4, 5, 6, // row 1 (unchanged)
                1, 2, 3, // row 2 (was row 0)
            ]
        );
    }

    #[test]
    fn test_reverse_matrix_index_bits_pow2_height() {
        let mut matrix = RowMajorMatrix::new(
            vec![
                0, 1, // row 0
                2, 3, // row 1
                4, 5, // row 2
                6, 7, // row 3
                8, 9, // row 4
                10, 11, // row 5
                12, 13, // row 6
                14, 15, // row 7
            ],
            2,
        );

        reverse_matrix_index_bits(&mut matrix);

        assert_eq!(
            matrix.values,
            vec![
                0, 1, // row 0 → index 0b000 → stays at 0
                8, 9, // row 1 → index 0b001 → goes to index 4
                4, 5, // row 2 → index 0b010 → stays
                12, 13, // row 3 → index 0b011 → goes to index 6
                2, 3, // row 4 → index 0b100 → was row 1
                10, 11, // row 5 → index 0b101 → stays
                6, 7, // row 6 → index 0b110 → was row 3
                14, 15, // row 7 → index 0b111 → stays
            ]
        );
    }

    #[test]
    fn test_reverse_matrix_index_bits_height_1() {
        let mut matrix = RowMajorMatrix::new(
            vec![
                42, 43, // row 0
            ],
            2,
        );

        // Bit-reversing a height-1 matrix should do nothing.
        reverse_matrix_index_bits(&mut matrix);

        assert_eq!(
            matrix.values,
            vec![
                42, 43, // row 0 (unchanged)
            ]
        );
    }

    #[test]
    #[should_panic]
    fn test_reverse_matrix_index_bits_non_power_of_two_should_panic() {
        // height = 3 → not a power of two → should panic
        let mut matrix = RowMajorMatrix::new(
            vec![
                1, 2, // row 0
                3, 4, // row 1
                5, 6, // row 2
            ],
            2,
        );

        reverse_matrix_index_bits(&mut matrix);
    }
}
