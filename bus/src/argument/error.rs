use thiserror::Error;

use crate::{BusDirection, ProductGkrError};

/// Failure to construct or verify a planned bus reduction.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum BusArgumentError {
    /// One materialized side has a statement-inconsistent leaf count.
    #[error("binary-bus {direction:?} leaf count is {actual}, expected {expected}")]
    LeafCountMismatch {
        /// Side whose materialization has the wrong length.
        direction: BusDirection,
        /// Count derived from the public bus plan.
        expected: usize,
        /// Count supplied by materialization.
        actual: usize,
    },
    /// Honest-prover inputs do not satisfy the multiset equality.
    #[error("binary-bus push and pull products differ")]
    UnbalancedProducts,
    /// A reduction output does not contain one terminal value per multiset side.
    #[error("binary-bus reduction has {actual} terminal values, expected {expected}")]
    TerminalValueCount {
        /// Number of values fixed by the push-and-pull statement.
        expected: usize,
        /// Number of values supplied by the reduction output.
        actual: usize,
    },
    /// The product reduction is malformed or inconsistent.
    #[error(transparent)]
    Product(#[from] ProductGkrError),
}
