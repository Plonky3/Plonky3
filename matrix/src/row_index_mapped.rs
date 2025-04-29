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
    ///
    /// The input `r` is assumed to lie in the range `0..self.height()` and the output
    /// will lie in the range `0..self.inner.height()`.
    ///
    /// It is considered undefined behaviour to call `map_row_index` with `r >= self.height()`.
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
            unsafe {
                // Safety: The output of `map_row_index` is less than `inner.height()` for all inputs in the range `0..self.height()`.
                (0..self.height())
                    .flat_map(|r| inner.row_unchecked(self.map_row_index(r)))
                    .collect()
            },
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

impl<T: Send + Sync + Clone, IndexMap: RowIndexMap, Inner: Matrix<T>> Matrix<T>
    for RowIndexMappedView<IndexMap, Inner>
{
    fn width(&self) -> usize {
        self.inner.width()
    }

    fn height(&self) -> usize {
        self.index_map.height()
    }

    unsafe fn get_unchecked(&self, r: usize, c: usize) -> T {
        unsafe {
            // Safety: The caller must ensure that r < self.height() and c < self.width().
            self.inner.get_unchecked(self.index_map.map_row_index(r), c)
        }
    }

    unsafe fn row_unchecked(
        &self,
        r: usize,
    ) -> impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync> {
        unsafe {
            // Safety: The caller must ensure that r < self.height().
            self.inner.row_unchecked(self.index_map.map_row_index(r))
        }
    }

    unsafe fn row_subseq_unchecked(
        &self,
        r: usize,
        start: usize,
        end: usize,
    ) -> impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync> {
        unsafe {
            // Safety: The caller must ensure that r < self.height() and start <= end <= self.width().
            self.inner
                .row_subseq_unchecked(self.index_map.map_row_index(r), start, end)
        }
    }

    unsafe fn row_slice_unchecked(&self, r: usize) -> impl Deref<Target = [T]> {
        unsafe {
            // Safety: The caller must ensure that r < self.height().
            self.inner
                .row_slice_unchecked(self.index_map.map_row_index(r))
        }
    }

    unsafe fn row_subslice_unchecked(
        &self,
        r: usize,
        start: usize,
        end: usize,
    ) -> impl Deref<Target = [T]> {
        unsafe {
            // Safety: The caller must ensure that r < self.height() and start <= end <= self.width().
            self.inner
                .row_subslice_unchecked(self.index_map.map_row_index(r), start, end)
        }
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

    use itertools::Itertools;
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

    /// A final Mock implementation of RowIndexMap
    struct ConstantMap;

    impl RowIndexMap for ConstantMap {
        fn height(&self) -> usize {
            1
        }

        fn map_row_index(&self, _r: usize) -> usize {
            0
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
        assert_eq!(mapped_view.get(0, 0).unwrap(), 1);
        assert_eq!(mapped_view.get(1, 2).unwrap(), 6);

        unsafe {
            assert_eq!(mapped_view.get_unchecked(0, 1), 2);
            assert_eq!(mapped_view.get_unchecked(1, 0), 4);
        }

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
        assert_eq!(mapped_view.get(0, 0).unwrap(), 4);
        // Check the last element of the mapped view (originally the first row, last column).
        assert_eq!(mapped_view.get(1, 2).unwrap(), 3);

        unsafe {
            assert_eq!(mapped_view.get_unchecked(0, 1), 5);
            assert_eq!(mapped_view.get_unchecked(1, 0), 1);
        }

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
    fn test_row_and_row_slice_methods() {
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
        assert_eq!(mapped_view.row_slice(0).unwrap().deref(), &[40, 50, 60]); // was row 1
        assert_eq!(
            mapped_view.row(1).unwrap().into_iter().collect_vec(),
            vec![10, 20, 30]
        ); // was row 0

        unsafe {
            // Check unsafe row slices.
            assert_eq!(
                mapped_view.row_unchecked(0).into_iter().collect_vec(),
                vec![40, 50, 60]
            ); // was row 1
            assert_eq!(mapped_view.row_slice_unchecked(1).deref(), &[10, 20, 30]); // was row 0

            assert_eq!(
                mapped_view.row_subslice_unchecked(0, 1, 3).deref(),
                &[50, 60]
            ); // was row 1
            assert_eq!(
                mapped_view
                    .row_subseq_unchecked(1, 0, 2)
                    .into_iter()
                    .collect_vec(),
                vec![10, 20]
            ); // was row 0
        }

        assert!(mapped_view.row(2).is_none()); // Height out of bounds.
        assert!(mapped_view.row_slice(2).is_none()); // Height out of bounds.
    }

    #[test]
    fn test_out_of_bounds_access() {
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
        assert_eq!(mapped_view.get(2, 1), None);
        assert!(mapped_view.row(5).is_none());
        assert!(mapped_view.row_slice(11).is_none());
        assert_eq!(mapped_view.get(0, 20), None);
    }

    #[test]
    fn test_out_of_bounds_access_with_bad_map() {
        // Create a 2x2 matrix:
        // [ 1  2 ]
        // [ 3  4 ]
        let inner = RowMajorMatrix::new(vec![1, 2, 3, 4], 4);

        // Use identity mapping.
        let mapped_view = RowIndexMappedView {
            index_map: ConstantMap,
            inner,
        };

        assert_eq!(mapped_view.get(0, 2), Some(3));

        // Attempt to access out-of-bounds row (index 1). Should panic.
        assert_eq!(mapped_view.get(1, 0), None);
        assert!(mapped_view.row(1).is_none());
        assert!(mapped_view.row_slice(1).is_none());
    }
}
