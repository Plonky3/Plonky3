//! Errors returned while materializing bus product leaves.

use thiserror::Error;

/// Invalid declarations rejected before any product proof is built.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum BusLeafError {
    /// The fingerprint point describes a tuple width that cannot fit in memory.
    #[error("bus tuple width overflows usize")]
    TupleWidthOverflow,
    /// A declaration has a different tuple width from the fingerprint challenge.
    #[error("bus declaration {declaration} has width {actual}, expected {expected}")]
    TupleWidthMismatch {
        /// Position of the malformed declaration.
        declaration: usize,
        /// Width fixed by the challenge point.
        expected: usize,
        /// Width carried by the declaration.
        actual: usize,
    },
    /// Tuple columns within one declaration have different row counts.
    #[error("bus declaration {declaration} column {column} has {actual} rows, expected {expected}")]
    ColumnHeightMismatch {
        /// Position of the malformed declaration.
        declaration: usize,
        /// Position of the malformed column.
        column: usize,
        /// Height fixed by the first column.
        expected: usize,
        /// Height carried by the malformed column.
        actual: usize,
    },
    /// A selector does not cover the same rows as its tuple columns.
    #[error("bus declaration {declaration} selector has {actual} rows, expected {expected}")]
    SelectorHeightMismatch {
        /// Position of the malformed declaration.
        declaration: usize,
        /// Height fixed by the tuple columns.
        expected: usize,
        /// Height carried by the selector.
        actual: usize,
    },
    /// A selected-row marker is not zero or one.
    #[error("bus declaration {declaration} selector row {row} is not Boolean")]
    NonBooleanSelector {
        /// Position of the malformed declaration.
        declaration: usize,
        /// Position of the malformed row.
        row: usize,
    },
}
