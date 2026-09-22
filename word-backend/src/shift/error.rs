//! Errors returned by the word shift reduction.

use p3_sumcheck::generic_degree::GenericDegreeError;
use p3_word::Segment;
use thiserror::Error;

/// A malformed statement or failed shift reduction.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ShiftReductionError {
    /// A statement point has a different arity from the checked system shape.
    #[error("{axis} point has length {actual}, expected {expected}")]
    PointLength {
        /// Human-readable axis name.
        axis: &'static str,
        /// Shape-derived point length.
        expected: usize,
        /// Supplied point length.
        actual: usize,
    },
    /// A public or committed word segment has the wrong length.
    #[error("{segment:?} segment has length {actual}, expected {expected}")]
    SegmentLength {
        /// Segment whose shape differs.
        segment: Segment,
        /// Shape-derived number of words.
        expected: usize,
        /// Supplied number of words.
        actual: usize,
    },
    /// One delegated quadratic sumcheck is malformed or inconsistent.
    #[error("shift reduction sumcheck failed: {0}")]
    Sumcheck(#[from] GenericDegreeError),
    /// The bit phase does not start from the statement claim less the public share.
    #[error("the bit sumcheck disagrees with the claim the statement enters it with")]
    EnteringClaim,
    /// The two reduction phases do not meet at the same intermediate claim.
    #[error("bit and word sumchecks disagree on their shared claim")]
    IntermediateClaim,
    /// The public wiring vanishes at the sampled word point, so no opening is pinned.
    #[error("shift reduction wiring vanishes and determines no trace opening")]
    DegenerateWiring,
    /// The final sumcheck value does not equal the claimed trace and wiring evaluations.
    #[error("shift reduction does not close against the trace opening")]
    FinalClaim,
}
