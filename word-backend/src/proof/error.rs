//! Errors returned by the complete word-level proof.

use p3_sumcheck::generic_degree::GenericDegreeError;
use p3_word::Segment;
use thiserror::Error;

use crate::ShiftReductionError;

/// A malformed statement or a failed word-level proof.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum WordProofError<E> {
    /// The commitment is too narrow to hold the padded bit trace.
    #[error("the commitment spans {actual} variables, short of the {expected} the trace needs")]
    TraceShape {
        /// Variables the padded word trace needs.
        expected: usize,
        /// Variables the supplied commitment covers.
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
    /// The sampled batching coefficient leaves one relation family unchecked.
    #[error("the sampled relation batching coefficient vanishes and checks no bitwise relation")]
    DegenerateBatching,
    /// The delegated vanishing check is malformed or inconsistent.
    #[error("relation vanishing check failed: {0}")]
    Zerocheck(GenericDegreeError),
    /// The proof claims a nonzero sum for a check whose whole point is that it vanishes.
    #[error("the batched relation sum is claimed nonzero, so some relation fails on the cube")]
    RelationSum,
    /// The vanishing check does not close against the supplied operand evaluations.
    #[error("the relation vanishing check does not close against the claimed operands")]
    RelationClaim,
    /// The shift reduction rejected the statement or the proof.
    #[error("shift reduction failed: {0}")]
    Shift(ShiftReductionError),
    /// The Boolean commitment refused the trace, the opening, or its proof.
    #[error("boolean commitment failed: {0}")]
    Commitment(E),
    /// The opened trace value differs from the value the reduction pinned.
    #[error("the opened trace value differs from the reduced claim")]
    OpeningValue,
}

impl<E> From<GenericDegreeError> for WordProofError<E> {
    fn from(value: GenericDegreeError) -> Self {
        Self::Zerocheck(value)
    }
}

impl<E> From<ShiftReductionError> for WordProofError<E> {
    fn from(value: ShiftReductionError) -> Self {
        Self::Shift(value)
    }
}
