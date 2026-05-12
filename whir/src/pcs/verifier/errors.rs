//! Verifier error types.

use alloc::string::String;

use thiserror::Error;

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

    /// Invalid round index.
    #[error("Invalid round index: {index}")]
    InvalidRoundIndex { index: usize },

    /// Proof-of-work witness verification failed.
    #[error("Invalid proof-of-work witness")]
    InvalidPowWitness,

    /// Proof is missing the Merkle commitment for a round.
    #[error("Proof is missing the Merkle commitment for round {round}")]
    MissingRoundCommitment { round: usize },

    /// Proof contains an unexpected number of rounds.
    #[error("Proof has {actual} rounds, expected {expected}")]
    RoundCountMismatch { expected: usize, actual: usize },

    /// Proof is missing the final polynomial evaluations.
    #[error("Proof is missing the final polynomial evaluations")]
    MissingFinalPoly,
}
