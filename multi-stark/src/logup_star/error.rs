//! Reasons the reduction rejects a proof.

use p3_sumcheck::generic_degree::GenericDegreeError;
use thiserror::Error;

use crate::fractional_gkr::FractionGkrError;

/// Reasons the reduction rejects a proof.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum LogupStarError {
    /// A pushforward carries a different number of entries than its table has.
    #[error("table {table} pushforward has {actual} entries, expected {expected}")]
    PushforwardWidth {
        /// Position of the table in statement order.
        table: usize,
        /// Entry count the statement describes.
        expected: usize,
        /// Entry count the proof carries.
        actual: usize,
    },
    /// The proof carries a different number of pushforwards than there are tables.
    #[error("proof carries {actual} pushforwards, expected {expected}")]
    PushforwardCount {
        /// Table count the statement describes.
        expected: usize,
        /// Pushforward count the proof carries.
        actual: usize,
    },
    /// The proof carries a different number of position-column values than there are readers.
    #[error("proof carries {actual} position claims, expected {expected}")]
    PositionClaimCount {
        /// Reader count the statement describes.
        expected: usize,
        /// Value count the proof carries.
        actual: usize,
    },
    /// The proof carries a different number of column values than the tables have columns.
    #[error("proof carries column claims for {actual} tables, expected {expected}")]
    ColumnClaimShape {
        /// Table count the statement describes.
        expected: usize,
        /// Table count the proof carries.
        actual: usize,
    },
    /// One table's column values do not match its declared width.
    #[error("table {table} carries {actual} column claims, expected {expected}")]
    ColumnClaimCount {
        /// Position of the table in statement order.
        table: usize,
        /// Column count the statement describes.
        expected: usize,
        /// Value count the proof carries.
        actual: usize,
    },
    /// The fraction reduction failed its own consistency checks.
    #[error("fraction reduction: {0}")]
    FractionGkr(#[from] FractionGkrError),
    /// The reduction's numerator opening disagrees with the statement's own weights.
    ///
    /// Every weight on that side is public, so only a broken reduction lands here.
    #[error("fraction reduction numerator does not match the statement")]
    LeafNumerator,
    /// The reduction's denominator opening disagrees with the claimed position values.
    ///
    /// This is where a lookup identity that does not hold surfaces.
    #[error("fraction reduction denominator does not match the claimed positions")]
    LeafDenominator,
    /// The product sumcheck failed its own consistency checks.
    #[error("product sumcheck: {0}")]
    ProductSumcheck(#[from] GenericDegreeError),
    /// The product sumcheck starts from a sum the statement does not ask for.
    #[error("product sumcheck claims a sum the statement does not")]
    ProductClaimedSum,
    /// The product sumcheck reduces to a value the claimed columns do not reproduce.
    #[error("product sumcheck does not close on the claimed table columns")]
    ProductFinalValue,
}
