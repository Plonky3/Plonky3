//! Error types for batch-STARK verification.

use core::fmt::Debug;

use p3_lookup::LookupError;
use p3_uni_stark::{InvalidProofShapeError, VerificationError};
use thiserror::Error;

/// Failure returned when verifying a batch proof.
///
/// A batch proof layers a lookup argument on top of a generic multi-AIR STARK.
/// Verification can therefore fail in two independent ways.
///
/// - The generic STARK part rejects the proof shape, an opening, or an evaluation.
/// - The lookup argument rejects the auxiliary trace or the cross-AIR balance.
///
/// Splitting the two keeps lookup concerns out of the base STARK error type.
#[derive(Debug, Error)]
pub enum BatchVerificationError<PcsErr>
where
    PcsErr: Debug,
{
    /// A generic STARK verification failure.
    #[error(transparent)]
    Verification(#[from] VerificationError<PcsErr>),
    /// A lookup-argument verification failure.
    #[error(transparent)]
    Lookup(#[from] LookupError),
}

impl<PcsErr: Debug> From<InvalidProofShapeError> for BatchVerificationError<PcsErr> {
    fn from(err: InvalidProofShapeError) -> Self {
        Self::Verification(VerificationError::InvalidProofShape(err))
    }
}
