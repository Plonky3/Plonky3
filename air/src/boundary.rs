//! Public inputs named by trace position rather than asserted by the AIR.
//!
//! An AIR lists the cells whose values are public:
//!
//! ```text
//!     (column, end) holds public_values[public_value]
//! ```
//!
//! A backend that supports the declaration binds each listed cell itself.
//! The AIR then writes no boundary constraint of its own.

use thiserror::Error;

/// Which end of the trace a boundary cell lives on.
///
/// These are the two rows a first-row and a last-row selector single out.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BoundaryEnd {
    /// The first trace row, index `0`.
    First,
    /// The last trace row, index `height - 1`.
    Last,
}

/// One main-trace cell whose value is a public input, named by its position.
///
/// A cell pairs a trace position with one of the AIR's public values:
///
/// ```text
///     (column, end) holds public_values[public_value]
/// ```
///
/// This is a declaration, not a constraint.
/// Binding the cell to the value is the proving backend's job.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BoundaryPublic {
    /// Main-trace column holding the cell.
    pub column: usize,
    /// Trace end the cell sits on.
    pub end: BoundaryEnd,
    /// Index into the AIR's public values supplying the cell's value.
    pub public_value: usize,
}

impl BoundaryPublic {
    /// Bundle a column, a trace end, and a public-value index into a boundary cell.
    pub const fn new(column: usize, end: BoundaryEnd, public_value: usize) -> Self {
        Self {
            column,
            end,
            public_value,
        }
    }

    /// Row index this cell sits on in a trace of `height` rows.
    ///
    /// ```text
    ///     first end -> 0
    ///     last  end -> height - 1
    /// ```
    ///
    /// # Panics
    ///
    /// Panics when `height` is zero.
    /// An empty trace has no boundary row to name.
    #[must_use]
    pub const fn row(&self, height: usize) -> usize {
        // A zero-height trace has no row to address at all.
        assert!(height > 0, "a boundary cell needs at least one trace row");

        // The two ends are the low and high rows of the trace.
        match self.end {
            BoundaryEnd::First => 0,
            BoundaryEnd::Last => height - 1,
        }
    }
}

/// Reasons a public boundary declaration cannot be applied to an AIR.
///
/// Each variant is a statement about the declaration alone.
/// No trace is involved.
#[derive(Copy, Clone, Debug, Error, PartialEq, Eq)]
pub enum BoundaryIoError {
    /// A declared cell names a column the AIR does not have.
    #[error("boundary-IO column {column} is out of range for main width {width}")]
    ColumnOutOfRange {
        /// Column the declaration names.
        column: usize,
        /// Number of main columns the AIR declares.
        width: usize,
    },
    /// A declared cell names a public value the AIR does not have.
    #[error(
        "boundary-IO public value {index} is out of range for {num_public_values} public values"
    )]
    PublicValueOutOfRange {
        /// Public-value index the declaration names.
        index: usize,
        /// Number of public values the AIR declares.
        num_public_values: usize,
    },
    /// Two declared cells name the same trace cell.
    ///
    /// One cell would then be pinned to two different public values.
    /// Even an honest proof fails whenever those two values differ.
    #[error("boundary-IO declares column {column} twice on the {end:?} row")]
    DuplicateCell {
        /// Column named by both declarations.
        column: usize,
        /// Trace end both declarations name.
        end: BoundaryEnd,
    },
}

/// Check that a declaration addresses only cells and values the AIR has.
///
/// A declaration is fixed by the AIR, never by the witness.
/// Each party checks the AIR it is handed, since nothing else ties the two together.
///
/// # Arguments
///
/// - `cells`: the declaration, as [`BaseAir::public_boundary_io`] returns it.
/// - `width`: main-trace columns the AIR declares.
/// - `num_public_values`: public values the AIR declares.
///
/// # Errors
///
/// - A cell names a column outside the main trace.
/// - A cell names a public value the AIR does not declare.
/// - Two cells name the same trace cell.
///
/// [`BaseAir::public_boundary_io`]: crate::BaseAir::public_boundary_io
pub fn validate(
    cells: &[BoundaryPublic],
    width: usize,
    num_public_values: usize,
) -> Result<(), BoundaryIoError> {
    for (index, cell) in cells.iter().enumerate() {
        // A pin reads the trace by this number with no further check.
        if cell.column >= width {
            return Err(BoundaryIoError::ColumnOutOfRange {
                column: cell.column,
                width,
            });
        }

        // A pin reads the public values by this number with no further check.
        if cell.public_value >= num_public_values {
            return Err(BoundaryIoError::PublicValueOutOfRange {
                index: cell.public_value,
                num_public_values,
            });
        }

        // A repeat of an earlier cell pins one trace cell to two public values.
        // A pairwise scan over the accepted prefix avoids allocating a set.
        if cells[..index]
            .iter()
            .any(|earlier| earlier.column == cell.column && earlier.end == cell.end)
        {
            return Err(BoundaryIoError::DuplicateCell {
                column: cell.column,
                end: cell.end,
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_reads_each_end_of_the_trace() {
        // The two ends are the low and high rows.
        let first = BoundaryPublic::new(0, BoundaryEnd::First, 0);
        let last = BoundaryPublic::new(0, BoundaryEnd::Last, 0);

        assert_eq!(first.row(8), 0);
        assert_eq!(last.row(8), 7);

        // A single-row trace collapses both ends onto row zero.
        assert_eq!(first.row(1), 0);
        assert_eq!(last.row(1), 0);
    }

    #[test]
    fn validate_accepts_distinct_in_range_cells() {
        // Invariant: one column may carry a cell on each end.
        //
        //     (column 1, first) and (column 1, last) are two different trace cells
        let cells = [
            BoundaryPublic::new(1, BoundaryEnd::First, 0),
            BoundaryPublic::new(1, BoundaryEnd::Last, 1),
        ];

        assert_eq!(validate(&cells, 2, 2), Ok(()));
    }

    #[test]
    fn validate_accepts_an_empty_declaration() {
        // An AIR that lists nothing is trivially well-formed, whatever its shape.
        assert_eq!(validate(&[], 0, 0), Ok(()));
    }

    #[test]
    fn validate_rejects_column_past_the_main_width() {
        // Mutation: name column 2 of a width-2 AIR.
        //
        //     columns present: 0, 1
        //     column named   : 2
        //                      → out of range
        let cells = [BoundaryPublic::new(2, BoundaryEnd::First, 0)];

        assert_eq!(
            validate(&cells, 2, 1),
            Err(BoundaryIoError::ColumnOutOfRange {
                column: 2,
                width: 2
            })
        );
    }

    #[test]
    fn validate_rejects_public_value_past_the_declared_count() {
        // Mutation: name public value 3 of an AIR declaring one public value.
        //
        //     public values present: 0
        //     public value named   : 3
        //                            → out of range
        let cells = [BoundaryPublic::new(0, BoundaryEnd::First, 3)];

        assert_eq!(
            validate(&cells, 2, 1),
            Err(BoundaryIoError::PublicValueOutOfRange {
                index: 3,
                num_public_values: 1
            })
        );
    }

    #[test]
    fn validate_rejects_two_cells_on_one_trace_cell() {
        // Mutation: point two cells at column 0's first row.
        //
        //     pinned to public value 0, and also to public value 1
        //                                → no trace satisfies both
        let cells = [
            BoundaryPublic::new(0, BoundaryEnd::First, 0),
            BoundaryPublic::new(0, BoundaryEnd::First, 1),
        ];

        assert_eq!(
            validate(&cells, 2, 2),
            Err(BoundaryIoError::DuplicateCell {
                column: 0,
                end: BoundaryEnd::First
            })
        );
    }
}
