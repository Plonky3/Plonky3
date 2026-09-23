use p3_bus::{BusEvaluationError, BusPlanError};
use thiserror::Error;

/// Failure to plan, evaluate, or authenticate a bus statement.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum BusBindingError {
    /// AIR metadata and public trace heights describe different instance counts.
    #[error("binary-bus received {airs} AIRs but {heights} trace heights")]
    InstanceCountMismatch {
        /// Number of AIR descriptions.
        airs: usize,
        /// Number of public trace heights.
        heights: usize,
    },
    /// Symbolic declarations do not define a supported statement.
    #[error(transparent)]
    Plan(#[from] BusPlanError),
    /// A symbolic declaration cannot be evaluated from its supplied values.
    #[error(transparent)]
    Evaluation(#[from] BusEvaluationError),
    /// A declaration reads a periodic column, which this backend does not evaluate.
    #[error("binary-bus declaration {declaration} of AIR {air} reads a periodic column")]
    PeriodicColumn {
        /// AIR position in the statement.
        air: usize,
        /// Declaration position within that AIR.
        declaration: usize,
    },
    /// ProductGKR returned a terminal point of the wrong dimension.
    #[error("binary-bus ProductGKR point has dimension {actual}, expected {expected}")]
    ProductPointDimension {
        /// Dimension fixed by the public product-tree shape.
        expected: usize,
        /// Dimension returned by the reduction.
        actual: usize,
    },
    /// The shared sumcheck point is too short to address the tallest bus table.
    #[error("binary-bus composition point has dimension {actual}, expected at least {expected}")]
    CompositionPointDimension {
        /// Dimension fixed by the tallest participating table.
        expected: usize,
        /// Dimension of the supplied point.
        actual: usize,
    },
}
