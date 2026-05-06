//! Verifier error types.

use alloc::string::String;

use thiserror::Error;

use crate::fiat_shamir::errors::FiatShamirError;
use crate::sumcheck::SumcheckError;

/// Errors during WHIR proof verification.
#[derive(Error, Debug)]
pub enum VerifierError {
    /// Merkle proof verification failed.
    #[error("Merkle proof verification failed at position {position}: {reason}")]
    MerkleProofInvalid { position: usize, reason: String },

    /// Sumcheck polynomial evaluation mismatch.
    #[error("Sumcheck verification failed at round {round}: expected {expected}, got {actual}")]
    SumcheckFailed {
        round: usize,
        expected: String,
        actual: String,
    },

    /// STIR challenge response is invalid.
    #[error("STIR challenge {challenge_id} verification failed: {details}")]
    StirChallengeFailed {
        challenge_id: usize,
        details: String,
    },

    /// The proof carries the wrong number of opening evaluation batches.
    ///
    /// Raised by the adapter before any sumcheck or Merkle work.
    #[error("expected {expected} opening evaluation batches, got {actual}")]
    OpeningBatchCountMismatch { expected: usize, actual: usize },

    /// One opening batch has the wrong number of evaluations for its column list.
    ///
    /// Raised by the adapter before any sumcheck or Merkle work.
    #[error("table {table_idx} opening expected {expected} evaluations, got {actual}")]
    OpeningBatchSizeMismatch {
        table_idx: usize,
        expected: usize,
        actual: usize,
    },

    /// Sumcheck verification error.
    #[error(transparent)]
    Sumcheck(#[from] SumcheckError),

    /// Fiat-Shamir transcript error.
    #[error(transparent)]
    FiatShamir(#[from] FiatShamirError),

    /// Invalid round index.
    #[error("Invalid round index: {index}")]
    InvalidRoundIndex { index: usize },

    /// Proof-of-work witness verification failed.
    #[error("Invalid proof-of-work witness")]
    InvalidPowWitness,
}
