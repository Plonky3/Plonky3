use thiserror::Error;

/// Errors from sumcheck protocol verification.
#[derive(Error, Debug)]
pub enum SumcheckError {
    /// The proof contains a different number of rounds than expected.
    #[error("Sumcheck round count mismatch: expected {expected}, got {actual}")]
    RoundCountMismatch { expected: usize, actual: usize },

    /// The proof is missing sumcheck data when rounds > 0.
    #[error("Missing sumcheck data for {expected_rounds} expected rounds")]
    MissingSumcheckData { expected_rounds: usize },

    /// Proof-of-work witness verification failed.
    #[error("Invalid proof-of-work witness")]
    InvalidPowWitness,

    /// The proof carries fewer PoW witnesses than sumcheck rounds.
    #[error("Sumcheck PoW witness count mismatch: expected {expected}, got {actual}")]
    PowWitnessCountMismatch { expected: usize, actual: usize },
}
