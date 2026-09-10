//! Error types for batch-STARK verification.

use core::fmt::Debug;

use p3_lookup::LookupError;
use p3_uni_stark::{InvalidProofShapeError, PeriodicColumnError, VerificationError};
use thiserror::Error;

use crate::transcript::BatchTranscriptFailure;

/// Failure returned when verifying a batch proof.
///
/// A batch proof layers a lookup argument on top of a generic multi-AIR STARK.
/// Verification can therefore fail in three independent ways.
///
/// - The generic STARK part rejects the proof shape, an opening, or an evaluation.
/// - The lookup argument rejects the auxiliary trace or the cross-AIR balance.
/// - A described transcript step rejects the value the proof carries for it.
///
/// Splitting them keeps lookup and transcript concerns out of the base STARK error type.
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
    /// A described transcript step the proof failed to satisfy.
    ///
    /// Both grinding sites report here.
    ///
    /// A rejection means the witness was forged.
    /// It also means the two sides may have been configured with different difficulties.
    #[error(transparent)]
    Transcript(#[from] BatchTranscriptFailure),
}

impl<PcsErr: Debug> From<InvalidProofShapeError> for BatchVerificationError<PcsErr> {
    fn from(err: InvalidProofShapeError) -> Self {
        Self::Verification(VerificationError::InvalidProofShape(err))
    }
}

impl<PcsErr: Debug> From<PeriodicColumnError> for BatchVerificationError<PcsErr> {
    fn from(err: PeriodicColumnError) -> Self {
        Self::Verification(VerificationError::PeriodicColumn(err))
    }
}
