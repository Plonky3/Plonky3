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

    /// HVZK sumcheck: a per-round wire payload had the wrong number of field
    /// elements. Each round must carry `max(ℓ_zk - 1, 2)` elements (after the
    /// linear coefficient is skipped per Lemma 6.4 / paper §6).
    #[error("HVZK round {round}: wire size mismatch, expected {expected}, got {actual}")]
    WireSizeMismatch {
        round: usize,
        expected: usize,
        actual: usize,
    },

    /// HVZK sumcheck: the number of supplied mask commitments does not match
    /// the folding factor `k`.
    #[error("HVZK mask commitment count mismatch: expected {expected}, got {actual}")]
    MaskCommitmentCountMismatch { expected: usize, actual: usize },
}
