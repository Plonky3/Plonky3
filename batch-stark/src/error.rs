//! Error types for batch-STARK verification.

use core::fmt::Debug;

use p3_lookup::LookupError;
use p3_uni_stark::{InvalidProofShapeError, PeriodicColumnError, VerificationError};
use thiserror::Error;

use crate::transcript::InvalidLookupPow;

/// Failure returned when verifying a batch proof.
///
/// A batch proof layers a lookup argument on top of a generic multi-AIR STARK.
/// Verification can therefore fail in two independent ways.
///
/// - The generic STARK part rejects the proof shape, an opening, or an evaluation.
/// - The lookup argument rejects the auxiliary trace, the cross-AIR balance, or
///   the proof of work guarding its challenges.
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
    /// The proof of work guarding the lookup challenges is invalid.
    ///
    /// Either the witness was forged, the prover and verifier disagree on
    /// [`p3_uni_stark::StarkGenericConfig::lookup_proof_of_work_bits`], or the
    /// proof disagrees with the batch about whether lookups exist at all.
    #[error("invalid proof-of-work witness for the lookup challenges: {0:?}")]
    InvalidLookupPow(InvalidLookupPow),
    /// The proof of work guarding the out-of-domain point is invalid.
    ///
    /// Either the witness was forged, or the prover and verifier disagree on
    /// [`p3_uni_stark::StarkGenericConfig::deep_proof_of_work_bits`].
    #[error("invalid proof-of-work witness for the out-of-domain point")]
    InvalidDeepPowWitness,
}

impl<PcsErr: Debug> From<InvalidLookupPow> for BatchVerificationError<PcsErr> {
    fn from(err: InvalidLookupPow) -> Self {
        Self::InvalidLookupPow(err)
    }
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
