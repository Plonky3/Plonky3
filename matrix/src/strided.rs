use crate::Matrix;
use crate::row_index_mapped::{RowIndexMap, RowIndexMappedView};

/// A vertical row-mapping strategy that selects every `stride`-th row from an inner matrix,
/// starting at a fixed `offset`.
///
/// This enables vertical striding like selecting rows: `offset`, `offset + stride`, etc.
#[derive(Debug)]
pub struct VerticallyStridedRowIndexMap {
    /// The number of rows in the resulting view.
    height: usize,
    /// The step size between selected rows in the inner matrix.
    stride: usize,
    /// The offset to start the stride from.
    offset: usize,
}

pub type VerticallyStridedMatrixView<Inner> =
    RowIndexMappedView<VerticallyStridedRowIndexMap, Inner>;

impl VerticallyStridedRowIndexMap {
    /// Create a new vertically strided view over a matrix.
    ///
    /// This selects rows in the inner matrix starting from `offset`, and then every `stride` rows after.
    /// Any choice of `offset` is valid.
    /// An `offset` at or past the inner height yields an empty view.
    ///
    /// # Arguments
    /// - `inner`: The inner matrix to view.
    /// - `stride`: The number of rows between each selected row.
    /// - `offset`: The inner row index of the first selected row.
    ///
    /// # Panics
    /// Panics if `stride` is zero.
    pub fn new_view<T: Send + Sync + Clone, Inner: Matrix<T>>(
        inner: Inner,
        stride: usize,
        offset: usize,
    ) -> VerticallyStridedMatrixView<Inner> {
        // View row i maps to inner row offset + i * stride, valid while < h.
        // Count of valid rows: ceil((h - offset) / stride), saturating to 0 when offset >= h.
        let height = inner.height().saturating_sub(offset).div_ceil(stride);
        RowIndexMappedView {
            index_map: Self {
                height,
                stride,
                offset,
            },
            inner,
        }
    }
}

impl RowIndexMap for VerticallyStridedRowIndexMap {
    fn height(&self) -> usize {
        self.height
    }

    fn map_row_index(&self, r: usize) -> usize {
        r * self.stride + self.offset
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::{Matrix, RowMajorMatrix};

    fn sample_matrix() -> RowMajorMatrix<i32> {
        // A 5x3 matrix:
        // [10, 11, 12]
        // [20, 21, 22]
        // [30, 31, 32]
        // [40, 41, 42]
        // [50, 51, 52]
        RowMajorMatrix::new(
            vec![10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42, 50, 51, 52],
            3,
        )
    }

    #[test]
    fn test_vertically_strided_view_stride_1_offset_0() {
        let matrix = sample_matrix();
        let view = VerticallyStridedRowIndexMap::new_view(matrix, 1, 0);

        assert_eq!(view.height(), 5);
        assert_eq!(view.width(), 3);

        assert_eq!(view.get(0, 0), Some(10));
        assert_eq!(view.get(1, 1), Some(21));
        unsafe {
            assert_eq!(view.get_unchecked(4, 2), 52);
        }
        assert_eq!(view.get(5, 0), None); // out of bounds
        assert_eq!(view.get(0, 3), None); // out of bounds
    }

    #[test]
    fn test_vertically_strided_view_stride_2_offset_0() {
        let matrix = sample_matrix();
        let view = VerticallyStridedRowIndexMap::new_view(matrix, 2, 0);

        assert_eq!(view.height(), 3);
        assert_eq!(view.get(0, 0), Some(10)); // row 0
        unsafe {
            assert_eq!(view.get_unchecked(1, 1), 31); // row 2
            assert_eq!(view.get_unchecked(2, 2), 52); // row 4
        }
        assert_eq!(view.get(0, 3), None); // out of bounds
    }

    #[test]
    fn test_vertically_strided_view_stride_2_offset_1() {
        let matrix = sample_matrix();
        let view = VerticallyStridedRowIndexMap::new_view(matrix, 2, 1);

        assert_eq!(view.height(), 2);
        assert_eq!(view.get(0, 0), Some(20)); // row 1
        unsafe {
            assert_eq!(view.get_unchecked(1, 1), 41);
        } // row 3
    }

    #[test]
    fn test_vertically_strided_view_stride_3_offset_0() {
        let matrix = sample_matrix();
        let view = VerticallyStridedRowIndexMap::new_view(matrix, 3, 0);

        assert_eq!(view.height(), 2);
        assert_eq!(view.get(0, 0), Some(10)); // row 0
        assert_eq!(view.get(1, 1), Some(41)); // row 3
    }

    #[test]
    fn test_vertically_strided_view_stride_3_offset_1() {
        let matrix = sample_matrix();
        let view = VerticallyStridedRowIndexMap::new_view(matrix, 3, 1);

        assert_eq!(view.height(), 2);
        unsafe {
            assert_eq!(view.get_unchecked(0, 0), 20); // row 1
            assert_eq!(view.get_unchecked(1, 1), 51); // row 4
        }
    }

    #[test]
    fn test_vertically_strided_view_stride_3_offset_2() {
        let matrix = sample_matrix();
        let view = VerticallyStridedRowIndexMap::new_view(matrix, 3, 2);

        assert_eq!(view.height(), 1);
        assert_eq!(view.get(0, 2), Some(32)); // row 2
    }

    #[test]
    fn test_vertically_strided_view_stride_greater_than_height() {
        let matrix = sample_matrix();
        let view = VerticallyStridedRowIndexMap::new_view(matrix, 10, 0);

        assert_eq!(view.height(), 1);
        assert_eq!(view.get(0, 0), Some(10)); // row 0
    }

    #[test]
    fn test_vertically_strided_view_stride_greater_than_height_with_valid_offset() {
        let matrix = sample_matrix(); // height = 5
        let view = VerticallyStridedRowIndexMap::new_view(matrix, 10, 4);

        // offset == 4 < height == 5 → view selects row 4
        assert_eq!(view.height(), 1);
        assert_eq!(view.get(0, 2), Some(52)); // row 4
    }

    #[test]
    fn test_vertically_strided_view_stride_greater_than_height_with_offset_beyond_height() {
        let matrix = sample_matrix(); // height = 5
        let view = VerticallyStridedRowIndexMap::new_view(matrix, 10, 6);

        // offset == 6 > height == 5 → no valid row
        assert_eq!(view.height(), 0);
        assert_eq!(view.get(0, 0), None); // out of bounds
    }

    #[test]
    fn test_vertically_strided_view_offset_greater_than_stride() {
        // Regression: with offset >= stride the old height formula over-reported,
        // letting view rows map past the inner height.
        //
        //     h = 5, stride = 2, offset = 3 → inner rows 3, 5, 7, ... → only row 3 in bounds
        let matrix = sample_matrix();
        let view = VerticallyStridedRowIndexMap::new_view(matrix, 2, 3);

        assert_eq!(view.height(), 1);
        // The single view row is inner row 3.
        assert_eq!(view.get(0, 0), Some(40));
        // View row 1 would be inner row 5, past the inner height → rejected.
        assert_eq!(view.get(1, 0), None);
    }

    #[test]
    fn test_vertically_strided_view_offset_equal_to_stride() {
        // h = 6, stride = 2, offset = 2 → inner rows 2, 4, 6, ... → rows 2 and 4 in bounds.
        let matrix = RowMajorMatrix::new(vec![10, 20, 30, 40, 50, 60], 1);
        let view = VerticallyStridedRowIndexMap::new_view(matrix, 2, 2);

        assert_eq!(view.height(), 2);
        assert_eq!(view.get(0, 0), Some(30)); // inner row 2
        assert_eq!(view.get(1, 0), Some(50)); // inner row 4
        // View row 2 would be inner row 6, past the inner height → rejected.
        assert_eq!(view.get(2, 0), None);
    }

    #[test]
    fn test_vertically_strided_view_exhaustive_height_and_bounds() {
        // Invariant: for any (stride, offset), the view height matches a brute-force count
        // and every view row reads the expected inner row.
        for stride in 1..=7usize {
            // Sweep offsets covering offset < stride, stride <= offset < h, and offset >= h.
            for offset in 0..=12usize {
                let matrix = sample_matrix();
                let h = matrix.height();
                let view = VerticallyStridedRowIndexMap::new_view(matrix, stride, offset);

                // Brute-force reference: count inner rows offset + i * stride < h.
                let expected_height = (offset..h).step_by(stride).count();
                assert_eq!(
                    view.height(),
                    expected_height,
                    "height mismatch for stride={stride}, offset={offset}"
                );

                // Every view row must read back the matching inner row,
                // proving the mapped index stays in bounds.
                for (i, inner_row) in (offset..h).step_by(stride).enumerate() {
                    assert_eq!(
                        view.get(i, 0),
                        Some(10 * (inner_row as i32 + 1)),
                        "wrong row for stride={stride}, offset={offset}, view row {i}"
                    );
                }

                // The first row past the reported height must be rejected.
                assert_eq!(view.get(expected_height, 0), None);
            }
        }
    }
}
