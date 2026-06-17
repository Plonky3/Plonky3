//! Error type for the generic-degree sumcheck verifier.

use thiserror::Error;

/// Errors raised by the generic-degree sumcheck verifier.
#[derive(Debug, Clone, Eq, PartialEq, Error)]
pub enum GenericDegreeError {
    /// Proof contains the wrong number of rounds.
    #[error("sumcheck round count mismatch: expected {expected}, got {actual}")]
    RoundCountMismatch {
        /// Number of rounds the verifier expected.
        expected: usize,
        /// Number of rounds present in the proof.
        actual: usize,
    },
    /// A round polynomial does not contain the expected number of evaluations.
    #[error("sumcheck round {round}: polynomial has {actual} evals, expected {expected}")]
    PolyEvalCountMismatch {
        /// Round index where the mismatch was found.
        round: usize,
        /// Expected number of evaluations per round.
        expected: usize,
        /// Actual number of evaluations.
        actual: usize,
    },
    /// PoW witness count does not match the round count when grinding is enabled.
    #[error("sumcheck pow witness count mismatch: expected {expected}, got {actual}")]
    PowWitnessCountMismatch {
        /// Expected witness count.
        expected: usize,
        /// Actual witness count.
        actual: usize,
    },
    /// A PoW witness failed to validate.
    #[error("sumcheck round {round}: invalid pow witness")]
    InvalidPowWitness {
        /// Round in which validation failed.
        round: usize,
    },
    /// Per-variable degree must be at least one to carry any information.
    #[error("sumcheck degree must be at least one, got {degree}")]
    InvalidDegree {
        /// Rejected degree value.
        degree: usize,
    },
}
