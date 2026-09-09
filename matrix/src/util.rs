use core::borrow::BorrowMut;

use p3_maybe_rayon::prelude::*;
use p3_util::{log2_strict_usize, reverse_bits_len, reverse_slice_index_bits};
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
    let values = mat.values.borrow_mut();
    let total_bytes = core::mem::size_of_val(values);

    // Small elements benefit from the slice permutation's cache-aware decomposition.
    // On AArch64, retain parallel row swaps above 64 KiB when multiple workers are active.
    // Check the worker count because dependencies can enable Rayon through feature unification.
    let use_slice = w == 1 && core::mem::size_of::<F>() <= 8;
    #[cfg(target_arch = "aarch64")]
    let use_slice = use_slice && (total_bytes <= 64 * 1024 || current_num_threads() == 1);
    if use_slice {
        reverse_slice_index_bits(values);
        return;
    }

    let values = values.as_mut_ptr() as usize;

    // SAFETY: Due to the i < j check, we are guaranteed that `swap_rows_raw
    // will never try and access a particular slice of data more than once
    // across all parallel threads. Hence the following code is safe and does
    // not trigger undefined behaviour.
    let swap = |i| {
        let values = values as *mut F;
        let j = reverse_bits_len(i, log_h);
        if i < j {
            unsafe { swap_rows_raw(values, w, i, j) };
        }
    };
    // Total matrix bytes, not rows: avoid scheduling overhead for inputs up to 32 KiB.
    if total_bytes <= 32 * 1024 {
        (0..h).for_each(swap);
    } else {
        (0..h).into_par_iter().for_each(swap);
    }
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
    fn test_reverse_matrix_index_bits_strings() {
        use alloc::string::ToString;
        use alloc::vec::Vec;

        for width in [1, 3, 17] {
            for log_h in 0..=14 {
                let height = 1 << log_h;
                let original: Vec<_> = (0..height * width).map(|i| i.to_string()).collect();
                let expected: Vec<_> = (0..height)
                    .flat_map(|row| {
                        let start = reverse_bits_len(row, log_h) * width;
                        original[start..start + width].iter().cloned()
                    })
                    .collect();
                let mut matrix = RowMajorMatrix::new(original.clone(), width);
                reverse_matrix_index_bits(&mut matrix);
                assert_eq!(matrix.values, expected, "width={width}, log_h={log_h}");
                reverse_matrix_index_bits(&mut matrix);
                assert_eq!(
                    matrix.values, original,
                    "involution: width={width}, log_h={log_h}"
                );
            }
        }
    }

    #[test]
    fn test_reverse_matrix_index_bits_preserves_owners_without_cloning() {
        use alloc::vec::Vec;
        use core::sync::atomic::{AtomicUsize, Ordering};

        // Fits the width-one fast path while detecting cloning and lost or duplicate owners.
        struct Owner<'a>(&'a AtomicUsize);
        impl Clone for Owner<'_> {
            fn clone(&self) -> Self {
                panic!("bit reversal must not clone elements");
            }
        }
        impl Drop for Owner<'_> {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }

        for width in [1, 3, 17] {
            for log_h in 0..=15 {
                let height = 1 << log_h;
                let drops: Vec<_> = (0..height * width).map(|_| AtomicUsize::new(0)).collect();
                let mut matrix = RowMajorMatrix::new(drops.iter().map(Owner).collect(), width);
                reverse_matrix_index_bits(&mut matrix);
                for (i, value) in matrix.values.iter().enumerate() {
                    let source = reverse_bits_len(i / width, log_h) * width + i % width;
                    assert!(core::ptr::eq(value.0, &drops[source]));
                }
                reverse_matrix_index_bits(&mut matrix);
                for (value, count) in matrix.values.iter().zip(&drops) {
                    assert!(core::ptr::eq(value.0, count));
                    assert_eq!(count.load(Ordering::Relaxed), 0);
                }
                drop(matrix);
                assert!(drops.iter().all(|count| count.load(Ordering::Relaxed) == 1));
            }
        }
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
