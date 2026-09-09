use thiserror::Error;

/// Errors from sumcheck protocol verification.
#[derive(Error, Debug, PartialEq, Eq)]
pub enum SumcheckError {
    /// The proof contains a different number of rounds than expected.
    #[error("Sumcheck round count mismatch: expected {expected}, got {actual}")]
    RoundCountMismatch {
        /// Round count the configuration fixes.
        expected: usize,
        /// Round count the proof carries.
        actual: usize,
    },

    /// The proof is missing sumcheck data when rounds > 0.
    #[error("Missing sumcheck data for {expected_rounds} expected rounds")]
    MissingSumcheckData {
        /// Round count the configuration fixes.
        expected_rounds: usize,
    },

    /// Proof-of-work witness verification failed.
    #[error("Sumcheck round {round}: pow witness does not meet {difficulty} bits")]
    InvalidPowWitness {
        /// Round whose challenge the rejected grinding step guards.
        round: usize,
        /// Difficulty in bits that the step requires.
        difficulty: usize,
    },

    /// Grinding is enabled but a round carries no witness to replay its step with.
    #[error("Sumcheck round {round}: missing the pow witness for {difficulty} bits")]
    MissingPowWitness {
        /// Round whose challenge the absent grinding step guards.
        round: usize,
        /// Difficulty in bits that the step requires.
        difficulty: usize,
    },

    /// The proof carries fewer PoW witnesses than sumcheck rounds.
    #[error("Sumcheck PoW witness count mismatch: expected {expected}, got {actual}")]
    PowWitnessCountMismatch {
        /// Witness count the configuration fixes.
        expected: usize,
        /// Witness count the proof carries.
        actual: usize,
    },

    /// HVZK sumcheck: a per-round wire payload had the wrong number of field elements.
    ///
    /// Each round carries `max(ell_zk, 3) - 1` of them.
    ///
    /// The linear coefficient is the one left off, per Lemma 6.4 and paper §6.
    #[error("HVZK round {round}: wire size mismatch, expected {expected}, got {actual}")]
    WireSizeMismatch {
        /// Round index where the mismatch was found, counted from zero.
        round: usize,
        /// Element count the configuration fixes.
        expected: usize,
        /// Element count the proof carries.
        actual: usize,
    },

    /// HVZK sumcheck: the base field has characteristic two.
    ///
    /// Lemma 6.4 inverts the endpoint identity, which divides by two.
    #[error("HVZK sumcheck: Lemma 6.4 requires char(F) != 2")]
    EvenCharacteristic,

    /// HVZK sumcheck: the mask code message is too short to hide a round.
    ///
    /// A mask of degree `ell_zk - 1` must cover the degree-2 plain piece.
    #[error("HVZK sumcheck: mask message length {ell_zk} is below the required 3")]
    MaskTooShort {
        /// Mask code message length the configuration fixes.
        ell_zk: usize,
    },

    /// HVZK sumcheck: a masked batch was configured with no rounds.
    ///
    /// Such a batch commits no mask and reduces no claim.
    #[error("HVZK sumcheck: a masked batch must run at least one round")]
    NoRounds,

    /// An opening claim's evaluations do not match the requested column shape.
    ///
    /// The evaluations come from the proof, so a malformed proof must be
    /// rejected here rather than aborting the verifier.
    #[error(
        "Opening shape mismatch for table {table_idx}: requested {expected_current} current and {expected_next} next, got {actual_current} current and {actual_next} next"
    )]
    OpeningShapeMismatch {
        /// Table whose opening claim was rejected.
        table_idx: usize,
        /// Current-column count the request fixes.
        expected_current: usize,
        /// Next-column count the request fixes.
        expected_next: usize,
        /// Current-column count the claim carries.
        actual_current: usize,
        /// Next-column count the claim carries.
        actual_next: usize,
    },
}
