//! Errors returned by the complete word-level proof.

use p3_sumcheck::generic_degree::GenericDegreeError;
use p3_word::Segment;
use thiserror::Error;

use crate::{IntegerMulError, ShiftReductionError};

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
    /// The proof carries a multiplication record exactly when the statement declares no product.
    #[error("the multiplication record does not match the statement's product relations")]
    ProductRecord,
    /// The multiplication reduction rejected the field or the proof.
    #[error("multiplication reduction failed: {0}")]
    IntegerMul(IntegerMulError),
    /// The delegated vanishing check is malformed or inconsistent.
    #[error("relation vanishing check failed: {0}")]
    Zerocheck(GenericDegreeError),
    /// The proof claims a batched sum other than the one the product claims fix.
    ///
    /// Without products that sum is zero, since every local relation vanishes on the cube.
    #[error("the batched relation sum differs from the one the statement fixes")]
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

impl<E> From<IntegerMulError> for WordProofError<E> {
    fn from(value: IntegerMulError) -> Self {
        Self::IntegerMul(value)
    }
}

impl<E> From<ShiftReductionError> for WordProofError<E> {
    fn from(value: ShiftReductionError) -> Self {
        Self::Shift(value)
    }
}
