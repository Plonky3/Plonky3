//! Error types for batch-STARK verification.

use core::fmt::Debug;

use p3_lookup::LookupError;
use p3_uni_stark::{InvalidProofShapeError, PeriodicColumnShapeError, VerificationError};
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
#[non_exhaustive]
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
    /// The lookup grinding difficulty cannot be sampled from the base field.
    ///
    /// A grind of b bits needs b below the word size and 2^b below the field order p.
    ///
    /// It is checked only when the batch declares a lookup, since only then does the grind run.
    #[error("{bits}-bit lookup grinding difficulty does not fit the field order")]
    UnsupportedLookupPowBits {
        /// The configured difficulty.
        bits: usize,
    },
}

impl<PcsErr: Debug> From<InvalidProofShapeError> for BatchVerificationError<PcsErr> {
    fn from(err: InvalidProofShapeError) -> Self {
        Self::Verification(VerificationError::InvalidProofShape(err))
    }
}

impl<PcsErr: Debug> From<PeriodicColumnShapeError> for BatchVerificationError<PcsErr> {
    fn from(err: PeriodicColumnShapeError) -> Self {
        Self::Verification(VerificationError::PeriodicColumn(err))
    }
}
