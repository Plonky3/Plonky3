use p3_util::{log2_strict_usize, reverse_bits_len};

use crate::Matrix;
use crate::dense::{DenseMatrix, DenseStorage, RowMajorMatrix};
use crate::row_index_mapped::{RowIndexMap, RowIndexMappedView};
use crate::util::reverse_matrix_index_bits;

/// A trait for matrices that support *bit-reversed row reordering*.
///
/// Implementors of this trait can switch between row-major order and bit-reversed
/// row order (i.e., reversing the binary representation of each row index).
///
/// This trait allows interoperability between regular matrices and views
/// that access their rows in a bit-reversed order.
pub trait BitReversibleMatrix<T: Send + Sync>: Matrix<T> {
    /// The type returned when this matrix is viewed in bit-reversed order.
    type BitRev: BitReversibleMatrix<T>;

    /// Return a version of the matrix with its row order reversed by bit index.
    fn bit_reverse_rows(self) -> Self::BitRev;
}

/// A row index permutation that reorders rows according to bit-reversed index.
///
/// Used internally to implement `BitReversedMatrixView`.
#[derive(Debug)]
pub struct BitReversalPerm {
    /// The logarithm (base 2) of the matrix height. For height `h`, this is `log2(h)`.
    ///
    /// This must be exact, so the height must be a power of two.
    log_height: usize,
}

impl BitReversalPerm {
    /// Create a new bit-reversal view over the given matrix.
    ///
    /// # Panics
    /// Panics if the height of the matrix is not a power of two.
    ///
    /// # Arguments
    /// - `inner`: The matrix to wrap in a bit-reversed row view.
    ///
    /// # Returns
    /// A `BitReversedMatrixView` that wraps the input with row reordering.
    pub fn new_view<T: Send + Sync, Inner: Matrix<T>>(
        inner: Inner,
    ) -> BitReversedMatrixView<Inner> {
        RowIndexMappedView {
            index_map: Self {
                log_height: log2_strict_usize(inner.height()),
            },
            inner,
        }
    }
}

impl RowIndexMap for BitReversalPerm {
    fn height(&self) -> usize {
        1 << self.log_height
    }

    fn map_row_index(&self, r: usize) -> usize {
        reverse_bits_len(r, self.log_height)
    }

    // This might not be more efficient than the lazy generic impl
    // if we have a nested view.
    fn to_row_major_matrix<T: Clone + Send + Sync, Inner: Matrix<T>>(
        &self,
        inner: Inner,
    ) -> RowMajorMatrix<T> {
        let mut inner = inner.to_row_major_matrix();
        reverse_matrix_index_bits(&mut inner);
        inner
    }
}

/// A matrix view that reorders its rows using bit-reversal.
///
/// This type is produced by applying `BitReversibleMatrix::bit_reverse_rows()`
/// to a `DenseMatrix`.
pub type BitReversedMatrixView<Inner> = RowIndexMappedView<BitReversalPerm, Inner>;

impl<T: Clone + Send + Sync, S: DenseStorage<T>> BitReversibleMatrix<T>
    for BitReversedMatrixView<DenseMatrix<T, S>>
{
    type BitRev = DenseMatrix<T, S>;

    fn bit_reverse_rows(self) -> Self::BitRev {
        self.inner
    }
}

impl<T: Clone + Send + Sync, S: DenseStorage<T>> BitReversibleMatrix<T> for DenseMatrix<T, S> {
    type BitRev = BitReversedMatrixView<Self>;

    fn bit_reverse_rows(self) -> Self::BitRev {
        BitReversalPerm::new_view(self)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use super::*;

    #[test]
    fn test_bit_reversal_perm_map_index() {
        let perm = BitReversalPerm { log_height: 3 }; // height = 8
        assert_eq!(perm.map_row_index(0), 0); // 000 -> 000
        assert_eq!(perm.map_row_index(1), 4); // 001 -> 100
        assert_eq!(perm.map_row_index(2), 2); // 010 -> 010
        assert_eq!(perm.map_row_index(3), 6); // 011 -> 110
        assert_eq!(perm.map_row_index(4), 1); // 100 -> 001
        assert_eq!(perm.map_row_index(5), 5); // 101 -> 101
        assert_eq!(perm.map_row_index(6), 3); // 110 -> 011
        assert_eq!(perm.map_row_index(7), 7); // 111 -> 111
    }

    #[test]
    fn test_bit_reversal_perm_height() {
        let perm = BitReversalPerm { log_height: 3 };
        assert_eq!(perm.height(), 8); // 2^3
    }

    #[test]
    fn test_new_view_reverses_indices_correctly() {
        // Matrix with height = 8 (2^3), width = 1: [0,1,2,3,4,5,6,7]
        let matrix = RowMajorMatrix::new((0u32..8).collect::<Vec<_>>(), 1);
        let bitrev = BitReversalPerm::new_view(matrix);

        // Should map row indices via bit reversal
        let expected = [0, 4, 2, 6, 1, 5, 3, 7];
        for (i, &expected_row_idx) in expected.iter().enumerate() {
            let row: Vec<_> = bitrev.row(i).collect();
            assert_eq!(row, vec![expected_row_idx]);
        }
    }

    #[test]
    fn test_to_row_major_matrix_applies_reverse_matrix_index_bits() {
        let matrix = RowMajorMatrix::new((0u32..8).collect::<Vec<_>>(), 1);
        let perm = BitReversalPerm { log_height: 3 };

        let reordered = perm.to_row_major_matrix(matrix);
        let expected_values = vec![0, 4, 2, 6, 1, 5, 3, 7]; // bit-reversed row order
        assert_eq!(reordered.values, expected_values);
    }

    #[test]
    fn test_bit_reversible_matrix_trait_forward_reverse() {
        let original = RowMajorMatrix::new((0u32..8).collect::<Vec<_>>(), 1);
        let reversed_view = original.clone().bit_reverse_rows(); // -> BitReversedMatrixView
        let back_to_dense = reversed_view.bit_reverse_rows(); // -> back to DenseMatrix

        assert_eq!(original.values, back_to_dense.values);
        assert_eq!(original.width(), back_to_dense.width());
    }

    #[test]
    #[should_panic]
    fn test_new_view_panics_non_pow2_height() {
        // This matrix has height = 3 (not a power of two)
        let matrix = RowMajorMatrix::new(vec![1, 2, 3], 1);
        let _ = BitReversalPerm::new_view(matrix);
    }
}
