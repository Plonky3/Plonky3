//! Failure modes and shape checks of the HVZK base case.

use thiserror::Error;

/// Failure modes of the HVZK base case.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum BaseCaseZkError {
    /// A revealed vector has the wrong length.
    #[error("blinded {kind} length mismatch: expected {expected}, got {actual}")]
    BlindedLengthMismatch {
        kind: &'static str,
        expected: usize,
        actual: usize,
    },

    /// The number of revealed masks or commitments is wrong.
    #[error("mask count mismatch: expected {expected}, got {actual}")]
    MaskCountMismatch { expected: usize, actual: usize },

    /// The number of spot-check openings is wrong.
    #[error("{kind} opening count mismatch: expected {expected}, got {actual}")]
    OpeningCountMismatch {
        kind: &'static str,
        expected: usize,
        actual: usize,
    },

    /// The proof-of-work witness failed.
    #[error("invalid proof-of-work witness")]
    InvalidPowWitness,

    /// The joint linear target check failed.
    #[error("base-case target check failed")]
    TargetCheckFailed,

    /// A source spot-check equation failed.
    #[error("source spot check failed at position {position}")]
    SourceSpotCheckFailed { position: usize },

    /// A mask spot-check equation failed.
    #[error("mask group {group} spot check failed at position {position}")]
    MaskSpotCheckFailed { group: usize, position: usize },

    /// A Merkle opening failed to verify.
    #[error("merkle verification failed for {kind} at position {position}")]
    MerkleVerificationFailed { kind: &'static str, position: usize },

    /// A source opening could not be resolved by the caller.
    #[error("source opening at position {position} rejected")]
    SourceOpeningRejected { position: usize },
}
