//! Errors returned by the jagged sparse-to-dense reduction.

use thiserror::Error;

use crate::SumcheckError;

/// A malformed sparse layout.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum JaggedLayoutError {
    /// The layout has no column to address.
    #[error("a jagged layout needs at least one column")]
    NoColumns,
    /// The column count cannot be addressed by a Boolean point.
    #[error("the jagged column count must be a power of two, got {columns}")]
    ColumnCountNotPowerOfTwo {
        /// Number of columns supplied by the caller.
        columns: usize,
    },
    /// The row bound cannot be represented by a machine index.
    #[error("the row-variable count {variables} does not fit in a machine index")]
    RowVariablesOverflow {
        /// Number of row variables supplied by the caller.
        variables: usize,
    },
    /// A live column extends past the declared row space.
    #[error("column {column} has height {height}, above the row bound {maximum}")]
    HeightExceedsRowBound {
        /// Index of the invalid column.
        column: usize,
        /// Number of live entries in the invalid column.
        height: usize,
        /// Maximum number of rows described by the row variables.
        maximum: usize,
    },
    /// The sum of live column lengths overflowed a machine index.
    #[error("the jagged trace area overflows a machine index at column {column}")]
    AreaOverflow {
        /// Column whose height made the running area overflow.
        column: usize,
    },
    /// The padded dense area cannot be represented by a machine index.
    #[error("the jagged trace area {area} has no representable power-of-two envelope")]
    DenseAreaOverflow {
        /// Sum of all live column lengths.
        area: usize,
    },
}

/// A malformed prover input or rejected jagged proof.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum JaggedError {
    /// The row point does not match the public row bound.
    #[error("the row point has {actual} coordinates, expected {expected}")]
    RowPointWidthMismatch {
        /// Number of coordinates required by the layout.
        expected: usize,
        /// Number of coordinates supplied by the caller.
        actual: usize,
    },
    /// The column point cannot address the public column count.
    #[error("the column point has {actual} coordinates, expected {expected}")]
    ColumnPointWidthMismatch {
        /// Number of coordinates required by the layout.
        expected: usize,
        /// Number of coordinates supplied by the caller.
        actual: usize,
    },
    /// The dense witness does not fill the power-of-two envelope exactly.
    #[error("the dense witness has {actual} cells, expected {expected}")]
    DenseLengthMismatch {
        /// Envelope size fixed by the layout.
        expected: usize,
        /// Number of cells supplied by the prover.
        actual: usize,
    },
    /// The witness does not take the value the caller asked to have proved.
    #[error("the dense witness does not evaluate to the claimed sparse value")]
    ClaimMismatch,
    /// The delegated quadratic sumcheck rejected.
    #[error(transparent)]
    Sumcheck(#[from] SumcheckError),
    /// The terminal product does not equal the sumcheck claim.
    #[error("the terminal jagged relation is inconsistent")]
    TerminalMismatch,
}
