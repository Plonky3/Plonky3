use core::marker::PhantomData;

use crate::Matrix;

/// A matrix wrapper that limits the number of columns visible from an inner matrix.
///
/// This struct wraps another matrix and restricts access to only the first `truncated_width` columns.
pub struct HorizontallyTruncated<T, Inner> {
    /// The underlying full matrix being wrapped.
    inner: Inner,
    /// The number of columns to expose from the inner matrix.
    truncated_width: usize,
    /// Marker for the element type `T`, not used at runtime.
    _phantom: PhantomData<T>,
}

impl<T, Inner: Matrix<T>> HorizontallyTruncated<T, Inner>
where
    T: Send + Sync,
{
    /// Construct a new horizontally truncated view of a matrix.
    ///
    /// # Arguments
    /// - `inner`: The full inner matrix to be wrapped.
    /// - `truncated_width`: The number of columns to expose (must be ≤ `inner.width()`).
    ///
    /// Returns `None` if `truncated_width` is greater than the width of the inner matrix.
    pub fn new(inner: Inner, truncated_width: usize) -> Option<Self> {
        (truncated_width <= inner.width()).then(|| Self {
            inner,
            truncated_width,
            _phantom: PhantomData,
        })
    }
}

impl<T, Inner> Matrix<T> for HorizontallyTruncated<T, Inner>
where
    T: Send + Sync,
    Inner: Matrix<T>,
{
    /// Returns the number of columns exposed by the truncated matrix.
    #[inline(always)]
    fn width(&self) -> usize {
        self.truncated_width
    }

    /// Returns the number of rows in the matrix (same as the inner matrix).
    #[inline(always)]
    fn height(&self) -> usize {
        self.inner.height()
    }

    /// Get the element at the given row and column.
    ///
    /// Returns None if `c >= truncated_width`, or if `r > self.height()`.
    #[inline(always)]
    fn get(&self, r: usize, c: usize) -> Option<T> {
        (c < self.truncated_width && r < self.height())
            .then(|| self.inner.get(r, c).expect("This should be unreachable without undefined behaviour. Most likely cause is inner.width() being incorrect."))
    }

    #[inline(always)]
    unsafe fn get_unchecked(&self, r: usize, c: usize) -> T {
        unsafe {
            // Safety: The caller must ensure that `c < truncated_width` and `r < self.height()`.
            self.inner.get_unchecked(r, c)
        }
    }

    fn row(
        &self,
        r: usize,
    ) -> Option<impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync>> {
        (r < self.height()).then(|| unsafe {
            // Safety: We just checked that `r < self.height()`.
            self.inner.row_subseq_unchecked(r, 0, self.truncated_width)
        })
    }

    unsafe fn row_unchecked(
        &self,
        r: usize,
    ) -> impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync> {
        unsafe {
            // Safety: The caller must ensure that `r < self.height()`.
            self.inner.row_subseq_unchecked(r, 0, self.truncated_width)
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
            self.inner.row_subseq_unchecked(r, start, end)
        }
    }

    fn row_slice(&self, r: usize) -> Option<impl core::ops::Deref<Target = [T]>> {
        (r < self.height()).then(|| unsafe {
            // Safety: We just checked that `r < self.height()`.
            self.inner
                .row_subslice_unchecked(r, 0, self.truncated_width)
        })
    }

    unsafe fn row_slice_unchecked(&self, r: usize) -> impl core::ops::Deref<Target = [T]> {
        unsafe {
            // Safety: The caller must ensure that `r < self.height()`.
            self.inner
                .row_subslice_unchecked(r, 0, self.truncated_width)
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
            self.inner.row_subslice_unchecked(r, start, end)
        }
    }
}

// TODO: Test row_subset_unchecked, row_slice, row_slice_unchecked, row_subslice_unchecked.

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
        let row: Vec<_> = truncated.row(0).unwrap().into_iter().collect();
        assert!(row.is_empty());

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
}
