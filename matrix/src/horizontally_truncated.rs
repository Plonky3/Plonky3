use core::marker::PhantomData;
use core::ops::Range;

use crate::Matrix;

/// A matrix wrapper that exposes a contiguous range of columns from an inner matrix.
///
/// This struct:
/// - wraps another matrix,
/// - restricts access to only the columns within the specified `column_range`.
pub struct HorizontallyTruncated<T, Inner> {
    /// The underlying full matrix being wrapped.
    inner: Inner,
    /// The range of columns to expose from the inner matrix.
    column_range: Range<usize>,
    /// Marker for the element type `T`, not used at runtime.
    _phantom: PhantomData<T>,
}

impl<T, Inner: Matrix<T>> HorizontallyTruncated<T, Inner>
where
    T: Send + Sync + Clone,
{
    /// Construct a new horizontally truncated view of a matrix.
    ///
    /// # Arguments
    /// - `inner`: The full inner matrix to be wrapped.
    /// - `truncated_width`: The number of columns to expose from the start (must be ≤ `inner.width()`).
    ///
    /// This is equivalent to `new_with_range(inner, 0..truncated_width)`.
    ///
    /// Returns `None` if `truncated_width` is greater than the width of the inner matrix.
    pub fn new(inner: Inner, truncated_width: usize) -> Option<Self> {
        Self::new_with_range(inner, 0..truncated_width)
    }

    /// Construct a new view exposing a specific column range of a matrix.
    ///
    /// # Arguments
    /// - `inner`: The full inner matrix to be wrapped.
    /// - `column_range`: The range of columns to expose (must satisfy `column_range.end <= inner.width()`).
    ///
    /// Returns `None` if the column range extends beyond the width of the inner matrix.
    pub fn new_with_range(inner: Inner, column_range: Range<usize>) -> Option<Self> {
        (column_range.end <= inner.width()).then(|| Self {
            inner,
            column_range,
            _phantom: PhantomData,
        })
    }
}

impl<T, Inner> Matrix<T> for HorizontallyTruncated<T, Inner>
where
    T: Send + Sync + Clone,
    Inner: Matrix<T>,
{
    /// Returns the number of columns exposed by the truncated matrix.
    #[inline(always)]
    fn width(&self) -> usize {
        self.column_range.len()
    }

    /// Returns the number of rows in the matrix (same as the inner matrix).
    #[inline(always)]
    fn height(&self) -> usize {
        self.inner.height()
    }

    #[inline(always)]
    unsafe fn get_unchecked(&self, r: usize, c: usize) -> T {
        unsafe {
            // Safety: The caller must ensure that `c < self.width()` and `r < self.height()`.
            //
            // We translate the column index by adding `column_range.start`.
            self.inner.get_unchecked(r, self.column_range.start + c)
        }
    }

    unsafe fn row_unchecked(
        &self,
        r: usize,
    ) -> impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync> {
        unsafe {
            // Safety: The caller must ensure that `r < self.height()`.
            self.inner
                .row_subseq_unchecked(r, self.column_range.start, self.column_range.end)
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
            //
            // We translate the column indices by adding `column_range.start`.
            self.inner.row_subseq_unchecked(
                r,
                self.column_range.start + start,
                self.column_range.start + end,
            )
        }
    }

    unsafe fn row_subslice_unchecked(
        &self,
        r: usize,
        start: usize,
        end: usize,
    ) -> impl core::ops::Deref<Target = [T]> {
        unsafe {
            // Safety: The caller must ensure that `r < self.height()` and `start <= end <= self.width()`.
            //
            // We translate the column indices by adding `column_range.start`.
            self.inner.row_subslice_unchecked(
                r,
                self.column_range.start + start,
                self.column_range.start + end,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use super::*;
    use crate::dense::RowMajorMatrix;

    #[test]
    fn test_truncate_width_by_one() {
        // Create a 3x4 matrix:
        // [ 1  2  3  4]
        // [ 5  6  7  8]
        // [ 9 10 11 12]
        let inner = RowMajorMatrix::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 4);

        // Truncate to width 3.
        let truncated = HorizontallyTruncated::new(inner, 3).unwrap();

        // Width should be 3.
        assert_eq!(truncated.width(), 3);

        // Height remains unchanged.
        assert_eq!(truncated.height(), 3);

        // Check individual elements.
        assert_eq!(truncated.get(0, 0), Some(1)); // row 0, col 0
        assert_eq!(truncated.get(1, 1), Some(6)); // row 1, col 1
        unsafe {
            assert_eq!(truncated.get_unchecked(0, 1), 2); // row 0, col 1
            assert_eq!(truncated.get_unchecked(2, 2), 11); // row 1, col 0
        }

        // Row 0: should return [1, 2, 3]
        let row0: Vec<_> = truncated.row(0).unwrap().into_iter().collect();
        assert_eq!(row0, vec![1, 2, 3]);
        unsafe {
            // Row 2: should return [5, 6, 7]
            let row1: Vec<_> = truncated.row_unchecked(1).into_iter().collect();
            assert_eq!(row1, vec![5, 6, 7]);

            // Row 3: is equal to return [9, 10, 11]
            let row3_subset: Vec<_> = truncated
                .row_subseq_unchecked(2, 1, 2)
                .into_iter()
                .collect();
            assert_eq!(row3_subset, vec![10]);
        }

        unsafe {
            let row1 = truncated.row_slice(1).unwrap();
            assert_eq!(&*row1, &[5, 6, 7]);

            let row2 = truncated.row_slice_unchecked(2);
            assert_eq!(&*row2, &[9, 10, 11]);

            let row0_subslice = truncated.row_subslice_unchecked(0, 0, 2);
            assert_eq!(&*row0_subslice, &[1, 2]);
        }

        assert!(truncated.get(0, 3).is_none()); // Width out of bounds
        assert!(truncated.get(3, 0).is_none()); // Height out of bounds
        assert!(truncated.row(3).is_none()); // Height out of bounds
        assert!(truncated.row_slice(3).is_none()); // Height out of bounds

        // Convert the truncated view to a RowMajorMatrix and check contents.
        let as_matrix = truncated.to_row_major_matrix();

        // The expected matrix after truncation:
        // [1  2  3]
        // [5  6  7]
        // [9 10 11]
        let expected = RowMajorMatrix::new(vec![1, 2, 3, 5, 6, 7, 9, 10, 11], 3);

        assert_eq!(as_matrix, expected);
    }

    #[test]
    fn test_no_truncation() {
        // 2x2 matrix:
        // [ 7  8 ]
        // [ 9 10 ]
        let inner = RowMajorMatrix::new(vec![7, 8, 9, 10], 2);

        // Truncate to full width (no change).
        let truncated = HorizontallyTruncated::new(inner, 2).unwrap();

        assert_eq!(truncated.width(), 2);
        assert_eq!(truncated.height(), 2);
        assert_eq!(truncated.get(0, 1).unwrap(), 8);
        assert_eq!(truncated.get(1, 0).unwrap(), 9);

        unsafe {
            assert_eq!(truncated.get_unchecked(0, 0), 7);
            assert_eq!(truncated.get_unchecked(1, 1), 10);
        }

        let row0: Vec<_> = truncated.row(0).unwrap().into_iter().collect();
        assert_eq!(row0, vec![7, 8]);

        let row1: Vec<_> = unsafe { truncated.row_unchecked(1).into_iter().collect() };
        assert_eq!(row1, vec![9, 10]);

        assert!(truncated.get(0, 2).is_none()); // Width out of bounds
        assert!(truncated.get(2, 0).is_none()); // Height out of bounds
        assert!(truncated.row(2).is_none()); // Height out of bounds
        assert!(truncated.row_slice(2).is_none()); // Height out of bounds
    }

    #[test]
    fn test_truncate_to_zero_width() {
        // 1x3 matrix: [11 12 13]
        let inner = RowMajorMatrix::new(vec![11, 12, 13], 3);

        // Truncate to width 0.
        let truncated = HorizontallyTruncated::new(inner, 0).unwrap();

        assert_eq!(truncated.width(), 0);
        assert_eq!(truncated.height(), 1);

        // Row should be empty.
        assert!(truncated.row(0).unwrap().into_iter().next().is_none());

        assert!(truncated.get(0, 0).is_none()); // Width out of bounds
        assert!(truncated.get(1, 0).is_none()); // Height out of bounds
        assert!(truncated.row(1).is_none()); // Height out of bounds
        assert!(truncated.row_slice(1).is_none()); // Height out of bounds
    }

    #[test]
    fn test_invalid_truncation_width() {
        // 2x2 matrix:
        // [1 2]
        // [3 4]
        let inner = RowMajorMatrix::new(vec![1, 2, 3, 4], 2);

        // Attempt to truncate beyond inner width (invalid).
        assert!(HorizontallyTruncated::new(inner, 5).is_none());
    }

    #[test]
    fn test_column_range_middle() {
        // Create a 3x5 matrix:
        // [ 1  2  3  4  5]
        // [ 6  7  8  9 10]
        // [11 12 13 14 15]
        let inner = RowMajorMatrix::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 5);

        // Select columns 1..4 (columns 1, 2, 3).
        let view = HorizontallyTruncated::new_with_range(inner, 1..4).unwrap();

        // Width should be 3 (columns 1, 2, 3).
        assert_eq!(view.width(), 3);

        // Height remains unchanged.
        assert_eq!(view.height(), 3);

        // Check individual elements (column indices are relative to the view).
        assert_eq!(view.get(0, 0), Some(2)); // row 0, col 0 -> inner col 1
        assert_eq!(view.get(0, 1), Some(3)); // row 0, col 1 -> inner col 2
        assert_eq!(view.get(0, 2), Some(4)); // row 0, col 2 -> inner col 3
        assert_eq!(view.get(1, 0), Some(7)); // row 1, col 0 -> inner col 1
        assert_eq!(view.get(2, 2), Some(14)); // row 2, col 2 -> inner col 3

        unsafe {
            assert_eq!(view.get_unchecked(1, 1), 8); // row 1, col 1 -> inner col 2
            assert_eq!(view.get_unchecked(2, 0), 12); // row 2, col 0 -> inner col 1
        }

        // Row 0: should return [2, 3, 4]
        let row0: Vec<_> = view.row(0).unwrap().into_iter().collect();
        assert_eq!(row0, vec![2, 3, 4]);

        // Row 1: should return [7, 8, 9]
        let row1: Vec<_> = view.row(1).unwrap().into_iter().collect();
        assert_eq!(row1, vec![7, 8, 9]);

        unsafe {
            // Row 2: should return [12, 13, 14]
            let row2: Vec<_> = view.row_unchecked(2).into_iter().collect();
            assert_eq!(row2, vec![12, 13, 14]);

            // Subsequence of row 1, cols 1..3 (view indices) -> [8, 9]
            let row1_subseq: Vec<_> = view.row_subseq_unchecked(1, 1, 3).into_iter().collect();
            assert_eq!(row1_subseq, vec![8, 9]);
        }

        // Out of bounds checks.
        assert!(view.get(0, 3).is_none()); // Width out of bounds
        assert!(view.get(3, 0).is_none()); // Height out of bounds

        // Convert the view to a RowMajorMatrix and check contents.
        let as_matrix = view.to_row_major_matrix();

        // The expected matrix after selecting columns 1..4:
        // [2  3  4]
        // [7  8  9]
        // [12 13 14]
        let expected = RowMajorMatrix::new(vec![2, 3, 4, 7, 8, 9, 12, 13, 14], 3);

        assert_eq!(as_matrix, expected);
    }

    #[test]
    fn test_column_range_end() {
        // Create a 2x4 matrix:
        // [1 2 3 4]
        // [5 6 7 8]
        let inner = RowMajorMatrix::new(vec![1, 2, 3, 4, 5, 6, 7, 8], 4);

        // Select columns 2..4 (columns 2, 3).
        let view = HorizontallyTruncated::new_with_range(inner, 2..4).unwrap();

        assert_eq!(view.width(), 2);
        assert_eq!(view.height(), 2);

        // Row 0: should return [3, 4]
        let row0: Vec<_> = view.row(0).unwrap().into_iter().collect();
        assert_eq!(row0, vec![3, 4]);

        // Row 1: should return [7, 8]
        let row1: Vec<_> = view.row(1).unwrap().into_iter().collect();
        assert_eq!(row1, vec![7, 8]);

        assert_eq!(view.get(0, 0), Some(3));
        assert_eq!(view.get(1, 1), Some(8));
    }

    #[test]
    fn test_column_range_single_column() {
        // Create a 3x4 matrix:
        // [1 2 3 4]
        // [5 6 7 8]
        // [9 10 11 12]
        let inner = RowMajorMatrix::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 4);

        // Select only column 2.
        let view = HorizontallyTruncated::new_with_range(inner, 2..3).unwrap();

        assert_eq!(view.width(), 1);
        assert_eq!(view.height(), 3);

        assert_eq!(view.get(0, 0), Some(3));
        assert_eq!(view.get(1, 0), Some(7));
        assert_eq!(view.get(2, 0), Some(11));

        // Row 0: should return [3]
        let row0: Vec<_> = view.row(0).unwrap().into_iter().collect();
        assert_eq!(row0, vec![3]);
    }

    #[test]
    fn test_column_range_empty() {
        // Create a 2x3 matrix:
        // [1 2 3]
        // [4 5 6]
        let inner = RowMajorMatrix::new(vec![1, 2, 3, 4, 5, 6], 3);

        // Select empty range (2..2).
        let view = HorizontallyTruncated::new_with_range(inner, 2..2).unwrap();

        assert_eq!(view.width(), 0);
        assert_eq!(view.height(), 2);

        // Row should be empty.
        assert!(view.row(0).unwrap().into_iter().next().is_none());
    }

    #[test]
    fn test_invalid_column_range() {
        // Create a 2x3 matrix:
        // [1 2 3]
        // [4 5 6]
        let inner = RowMajorMatrix::new(vec![1, 2, 3, 4, 5, 6], 3);

        // Attempt to select columns 1..5 (extends beyond width).
        assert!(HorizontallyTruncated::new_with_range(inner, 1..5).is_none());
    }
}
