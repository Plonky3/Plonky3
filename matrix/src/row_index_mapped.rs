use core::ops::Deref;

use p3_field::PackedValue;

use crate::Matrix;
use crate::dense::RowMajorMatrix;

/// A trait for remapping row indices of a matrix.
///
/// Implementations can change the number of visible rows (`height`)
/// and define how a given logical row index maps to a physical one.
pub trait RowIndexMap: Send + Sync {
    /// Returns the number of rows exposed by the mapping.
    fn height(&self) -> usize;

    /// Maps a visible row index `r` to the corresponding row index in the underlying matrix.
    fn map_row_index(&self, r: usize) -> usize;

    /// Converts the mapped matrix into a dense row-major matrix.
    ///
    /// This default implementation iterates over all mapped rows,
    /// collects them in order, and builds a dense representation.
    fn to_row_major_matrix<T: Clone + Send + Sync, Inner: Matrix<T>>(
        &self,
        inner: Inner,
    ) -> RowMajorMatrix<T> {
        RowMajorMatrix::new(
            (0..self.height())
                .flat_map(|r| inner.row(self.map_row_index(r)))
                .collect(),
            inner.width(),
        )
    }
}

/// A matrix view that applies a row index mapping to an inner matrix.
///
/// The mapping changes which rows are visible and in what order.
/// The width remains unchanged.
#[derive(Copy, Clone, Debug)]
pub struct RowIndexMappedView<IndexMap, Inner> {
    /// A row index mapping that defines the number and order of visible rows.
    pub index_map: IndexMap,
    /// The inner matrix that holds actual data.
    pub inner: Inner,
}

impl<T: Send + Sync, IndexMap: RowIndexMap, Inner: Matrix<T>> Matrix<T>
    for RowIndexMappedView<IndexMap, Inner>
{
    fn width(&self) -> usize {
        self.inner.width()
    }

    fn height(&self) -> usize {
        self.index_map.height()
    }

    fn get(&self, r: usize, c: usize) -> T {
        self.inner.get(self.index_map.map_row_index(r), c)
    }

    type Row<'a>
        = Inner::Row<'a>
    where
        Self: 'a;

    // Override these methods so we use the potentially optimized inner methods instead of defaults.

    fn row(&self, r: usize) -> Self::Row<'_> {
        self.inner.row(self.index_map.map_row_index(r))
    }

    fn row_slice(&self, r: usize) -> impl Deref<Target = [T]> {
        self.inner.row_slice(self.index_map.map_row_index(r))
    }

    fn to_row_major_matrix(self) -> RowMajorMatrix<T>
    where
        Self: Sized,
        T: Clone,
    {
        // Use Perm's optimized permutation routine, if it has one.
        self.index_map.to_row_major_matrix(self.inner)
    }

    fn horizontally_packed_row<'a, P>(
        &'a self,
        r: usize,
    ) -> (
        impl Iterator<Item = P> + Send + Sync,
        impl Iterator<Item = T> + Send + Sync,
    )
    where
        P: PackedValue<Value = T>,
        T: Clone + 'a,
    {
        self.inner
            .horizontally_packed_row(self.index_map.map_row_index(r))
    }

    fn padded_horizontally_packed_row<'a, P>(
        &'a self,
        r: usize,
    ) -> impl Iterator<Item = P> + Send + Sync
    where
        P: PackedValue<Value = T>,
        T: Clone + Default + 'a,
    {
        self.inner
            .padded_horizontally_packed_row(self.index_map.map_row_index(r))
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::FieldArray;

    use super::*;
    use crate::dense::RowMajorMatrix;

    /// Mock implementation of RowIndexMap
    struct IdentityMap(usize);

    impl RowIndexMap for IdentityMap {
        fn height(&self) -> usize {
            self.0
        }

        fn map_row_index(&self, r: usize) -> usize {
            r
        }
    }

    /// Another mock implementation for reversing rows
    struct ReverseMap(usize);

    impl RowIndexMap for ReverseMap {
        fn height(&self) -> usize {
            self.0
        }

        fn map_row_index(&self, r: usize) -> usize {
            self.0 - 1 - r
        }
    }

    #[test]
    fn test_identity_row_index_map() {
        // Create an inner matrix.
        // The matrix will be:
        // [ 1  2  3 ]
        // [ 4  5  6 ]
        let inner = RowMajorMatrix::new(vec![1, 2, 3, 4, 5, 6], 3);

        // Create a mapped view using an `IdentityMap`, which does not alter row indices.
        let mapped_view = RowIndexMappedView {
            index_map: IdentityMap(inner.height()),
            inner,
        };

        // Check dimensions.
        assert_eq!(mapped_view.height(), 2);
        assert_eq!(mapped_view.width(), 3);

        // Check values.
        assert_eq!(mapped_view.get(0, 0), 1);
        assert_eq!(mapped_view.get(1, 2), 6);

        // Check rows.
        let rows: Vec<Vec<_>> = mapped_view.rows().map(|row| row.collect()).collect();
        assert_eq!(rows, vec![vec![1, 2, 3], vec![4, 5, 6]]);

        // Check dense matrix.
        let dense = mapped_view.to_row_major_matrix();
        assert_eq!(dense.values, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_reverse_row_index_map() {
        // Create an inner matrix.
        // The matrix will be:
        // [ 1  2  3 ]
        // [ 4  5  6 ]
        let inner = RowMajorMatrix::new(vec![1, 2, 3, 4, 5, 6], 3);

        // Create a mapped view using a ReverseMap, which reverses row indices.
        let mapped_view = RowIndexMappedView {
            index_map: ReverseMap(inner.height()),
            inner,
        };

        // Check dimensions.
        assert_eq!(mapped_view.height(), 2);
        assert_eq!(mapped_view.width(), 3);

        // Check the first element of the mapped view (originally the second row, first column).
        assert_eq!(mapped_view.get(0, 0), 4);
        // Check the last element of the mapped view (originally the first row, last column).
        assert_eq!(mapped_view.get(1, 2), 3);

        // Check rows.
        let rows: Vec<Vec<_>> = mapped_view.rows().map(|row| row.collect()).collect();
        assert_eq!(rows, vec![vec![4, 5, 6], vec![1, 2, 3]]);

        // Check dense matrix.
        let dense = mapped_view.to_row_major_matrix();
        assert_eq!(dense.values, vec![4, 5, 6, 1, 2, 3]);
    }

    #[test]
    fn test_horizontally_packed_row() {
        // Define the packed type with width 2
        type Packed = FieldArray<BabyBear, 2>;

        // Create an inner matrix of BabyBear elements.
        // Matrix layout:
        // [ 1  2 ]
        // [ 3  4 ]
        let inner = RowMajorMatrix::new(
            vec![
                BabyBear::new(1),
                BabyBear::new(2),
                BabyBear::new(3),
                BabyBear::new(4),
            ],
            2,
        );

        // Apply a reverse row index mapping.
        let mapped_view = RowIndexMappedView {
            index_map: ReverseMap(inner.height()),
            inner,
        };

        // Extract the packed and suffix iterators from row 0 (which is reversed row 1).
        let (packed_iter, suffix_iter) = mapped_view.horizontally_packed_row::<Packed>(0);

        // Collect iterators to concrete values.
        let packed: Vec<_> = packed_iter.collect();
        let suffix: Vec<_> = suffix_iter.collect();

        // Check the packed row values match reversed second row.
        assert_eq!(
            packed,
            &[Packed::from([BabyBear::new(3), BabyBear::new(4)])]
        );

        // Check there are no suffix leftovers.
        assert!(suffix.is_empty());
    }

    #[test]
    fn test_padded_horizontally_packed_row() {
        // Define a packed type with width 3
        type Packed = FieldArray<BabyBear, 3>;

        // Create a 2x2 matrix of BabyBear elements:
        // [ 1  2 ]
        // [ 3  4 ]
        let inner = RowMajorMatrix::new(
            vec![
                BabyBear::new(1),
                BabyBear::new(2),
                BabyBear::new(3),
                BabyBear::new(4),
            ],
            2,
        );

        // Use identity mapping (rows remain unchanged).
        let mapped_view = RowIndexMappedView {
            index_map: IdentityMap(inner.height()),
            inner,
        };

        // Pad the second row (row 1) into chunks of size 3.
        let packed: Vec<_> = mapped_view
            .padded_horizontally_packed_row::<Packed>(1)
            .collect();

        // Verify the packed result includes padding with zero at the end.
        assert_eq!(
            packed,
            vec![Packed::from([
                BabyBear::new(3),
                BabyBear::new(4),
                BabyBear::new(0),
            ])]
        );
    }

    #[test]
    fn test_row_slice() {
        // Create a 2x3 matrix of integers:
        // [ 10  20  30 ]
        // [ 40  50  60 ]
        let inner = RowMajorMatrix::new(vec![10, 20, 30, 40, 50, 60], 3);

        // Apply reverse row mapping (row 0 becomes 1, row 1 becomes 0).
        let mapped_view = RowIndexMappedView {
            index_map: ReverseMap(inner.height()),
            inner,
        };

        // Get row slices through dereferencing and verify content.
        assert_eq!(mapped_view.row_slice(0).deref(), &[40, 50, 60]); // was row 1
        assert_eq!(mapped_view.row_slice(1).deref(), &[10, 20, 30]); // was row 0
    }

    #[test]
    #[should_panic]
    fn test_out_of_bounds_row_access() {
        // Create a 2x2 matrix:
        // [ 1  2 ]
        // [ 3  4 ]
        let inner = RowMajorMatrix::new(vec![1, 2, 3, 4], 2);

        // Use identity mapping.
        let mapped_view = RowIndexMappedView {
            index_map: IdentityMap(inner.height()),
            inner,
        };

        // Attempt to access out-of-bounds row (index 2). Should panic.
        mapped_view.get(2, 1);
    }

    #[test]
    #[should_panic]
    fn test_out_of_bounds_col_access() {
        // Create a 2x2 matrix:
        // [ 1  2 ]
        // [ 3  4 ]
        let inner = RowMajorMatrix::new(vec![1, 2, 3, 4], 2);

        // Use identity mapping.
        let mapped_view = RowIndexMappedView {
            index_map: IdentityMap(inner.height()),
            inner,
        };

        // Attempt to access out-of-bounds column (index 20). Should panic.
        mapped_view.get(0, 20);
    }
}
