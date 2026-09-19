//! Validated geometry for a capacity-free jagged commitment.

use alloc::vec::Vec;

use p3_util::log2_ceil_usize;

use super::JaggedLayoutError;

/// A sparse column layout and its contiguous dense representation.
///
/// Every column occupies one consecutive interval in the dense witness.
///
/// The final power-of-two suffix is virtual zero padding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JaggedLayout {
    /// Number of variables addressing a row in the sparse view.
    row_variables: usize,
    /// Number of variables addressing the dense power-of-two envelope.
    dense_variables: usize,
    /// Start of every column followed by the total live area.
    cumulative_heights: Vec<usize>,
}

impl JaggedLayout {
    /// Builds a layout from the live height of every sparse column.
    ///
    /// Columns may be empty and heights need not be powers of two.
    ///
    /// # Errors
    ///
    /// - The column count is zero or not a power of two.
    /// - The row bound or dense area cannot be represented by `usize`.
    /// - A column height exceeds the declared row bound.
    pub fn new(row_variables: usize, heights: &[usize]) -> Result<Self, JaggedLayoutError> {
        // A column point names one vertex of a Boolean cube.
        if heights.is_empty() {
            return Err(JaggedLayoutError::NoColumns);
        }
        if !heights.len().is_power_of_two() {
            return Err(JaggedLayoutError::ColumnCountNotPowerOfTwo {
                columns: heights.len(),
            });
        }

        // The row bound is used for checked height validation and allocation.
        let row_shift =
            u32::try_from(row_variables).map_err(|_| JaggedLayoutError::RowVariablesOverflow {
                variables: row_variables,
            })?;
        let row_bound =
            1usize
                .checked_shl(row_shift)
                .ok_or(JaggedLayoutError::RowVariablesOverflow {
                    variables: row_variables,
                })?;

        // Prefix sums are the sparse-to-dense bijection.
        //
        //     column y  ->  [prefix[y], prefix[y + 1])
        let mut cumulative_heights = Vec::with_capacity(heights.len() + 1);
        cumulative_heights.push(0usize);
        for (column, &height) in heights.iter().enumerate() {
            if height > row_bound {
                return Err(JaggedLayoutError::HeightExceedsRowBound {
                    column,
                    height,
                    maximum: row_bound,
                });
            }
            let area = cumulative_heights[column]
                .checked_add(height)
                .ok_or(JaggedLayoutError::AreaOverflow { column })?;
            cumulative_heights.push(area);
        }

        // The dense multilinear has the smallest power-of-two domain covering all live cells.
        // An empty trace still has one constant zero evaluation.
        let area = cumulative_heights[heights.len()];
        let capacity = area
            .max(1)
            .checked_next_power_of_two()
            .ok_or(JaggedLayoutError::DenseAreaOverflow { area })?;
        let dense_variables = log2_ceil_usize(capacity);

        Ok(Self {
            row_variables,
            dense_variables,
            cumulative_heights,
        })
    }

    /// Returns the number of variables addressing a sparse row.
    #[must_use]
    pub const fn row_variables(&self) -> usize {
        // The value was validated when the layout was built.
        self.row_variables
    }

    /// Returns the number of variables addressing a sparse column.
    #[must_use]
    pub const fn column_variables(&self) -> usize {
        // Construction requires a nonzero power-of-two column count.
        self.num_columns().trailing_zeros() as usize
    }

    /// Returns the number of variables addressing the dense witness.
    #[must_use]
    pub const fn dense_variables(&self) -> usize {
        // The value was derived from the checked dense capacity.
        self.dense_variables
    }

    /// Returns the number of sparse columns.
    #[must_use]
    pub const fn num_columns(&self) -> usize {
        // The final entry is the total rather than a column start.
        self.cumulative_heights.len() - 1
    }

    /// Returns the number of live witness cells.
    #[must_use]
    pub fn area(&self) -> usize {
        // Construction always stores one terminal prefix sum.
        self.cumulative_heights[self.num_columns()]
    }

    /// Returns the power-of-two size of the dense multilinear.
    #[must_use]
    pub const fn dense_capacity(&self) -> usize {
        // Construction rejects an exponent that does not fit in a machine index.
        1usize << self.dense_variables
    }

    /// Returns the live height of one sparse column.
    ///
    /// # Panics
    ///
    /// Panics when the column index is outside the layout.
    #[must_use]
    pub fn column_height(&self, column: usize) -> usize {
        // Adjacent prefix sums delimit exactly one column.
        self.cumulative_heights[column + 1] - self.cumulative_heights[column]
    }

    /// Returns every column boundary followed by the total live area.
    #[must_use]
    pub fn cumulative_heights(&self) -> &[usize] {
        // The stored prefix table includes both endpoints of every column.
        &self.cumulative_heights
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    #[test]
    fn mixed_heights_form_contiguous_dense_intervals() {
        // Fixture state:
        //
        //     heights   [3, 0, 5, 1]
        //     prefixes  [0, 3, 3, 8, 9]
        //     capacity  16 cells
        let layout = JaggedLayout::new(3, &[3, 0, 5, 1]).unwrap();

        assert_eq!(layout.cumulative_heights(), &[0, 3, 3, 8, 9]);
        assert_eq!(layout.area(), 9);
        assert_eq!(layout.dense_variables(), 4);
        assert_eq!(layout.dense_capacity(), 16);
        assert_eq!(layout.column_height(1), 0);
    }

    #[test]
    fn malformed_geometry_is_rejected_at_construction() {
        // No Boolean point addresses three columns.
        assert_eq!(
            JaggedLayout::new(2, &[1, 1, 1]),
            Err(JaggedLayoutError::ColumnCountNotPowerOfTwo { columns: 3 })
        );

        // A two-variable row point addresses only four rows.
        assert_eq!(
            JaggedLayout::new(2, &[5]),
            Err(JaggedLayoutError::HeightExceedsRowBound {
                column: 0,
                height: 5,
                maximum: 4,
            })
        );

        // An empty height list has no sparse coordinate space.
        assert_eq!(JaggedLayout::new(0, &[]), Err(JaggedLayoutError::NoColumns));

        // The empty trace is the constant zero polynomial over one dense cell.
        let empty = JaggedLayout::new(3, &[0, 0]).unwrap();
        assert_eq!(empty.area(), 0);
        assert_eq!(empty.dense_capacity(), 1);
        assert_eq!(empty.dense_variables(), 0);

        // Keep the allocation alive until every assertion has read it.
        drop(vec![empty]);
    }
}
