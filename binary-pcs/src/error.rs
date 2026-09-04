//! Typed reasons an opening proof was rejected.

use thiserror::Error;

/// Why an opening proof was rejected.
///
/// Derives [`thiserror::Error`] and [`Debug`] only — not `PartialEq`/`Eq`. [`p3_commit::Mmcs`]
/// carries no such bound on `Error` for a general MMCS (`commit/src/mmcs.rs`), so the derive
/// cannot apply here.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum BinaryPcsError<MmcsError> {
    /// The proof carries a different number of intermediate folding rounds than the config
    /// derives.
    ///
    /// Checked before any indexing into `rounds`, so a malformed proof is rejected rather
    /// than desyncing the round loop that walks it.
    #[error("expected {expected} folding rounds, proof carries {actual}")]
    RoundCountMismatch { expected: usize, actual: usize },

    /// A round opened a different number of rows than the sampled query count demands.
    ///
    /// Checked before any indexing into the opened rows, so a short or padded opening list
    /// is rejected instead of being read out of bounds or silently truncated.
    #[error("round {round} opened {actual} rows, expected {expected}")]
    OpeningCountMismatch {
        round: usize,
        expected: usize,
        actual: usize,
    },

    /// An opened row had the wrong width.
    ///
    /// Every committed round holds a width-1 codeword; a row of any other width would desync
    /// the fold if it were read as-is rather than rejected up front.
    #[error("round {round} row {query} has width {actual}, expected {expected}")]
    RowWidthMismatch {
        round: usize,
        query: usize,
        expected: usize,
        actual: usize,
    },

    /// The final codeword's length disagrees with the fold schedule.
    ///
    /// Checked before any query reads from it, so an under- or over-sized final codeword is
    /// rejected instead of being indexed out of bounds.
    #[error("final codeword has {actual} symbols, expected {expected}")]
    FinalCodewordLengthMismatch { expected: usize, actual: usize },

    /// A Merkle multiproof did not verify.
    #[error("Merkle opening failed in round {round}")]
    MerkleFailed {
        round: usize,
        #[source]
        source: MmcsError,
    },

    /// Folding the queried pair at one round does not reproduce the value read at the next
    /// round (or, at the last round, in the final codeword).
    ///
    /// This is what ties every committed round to its neighbours; without it, a prover could
    /// commit to codewords with no relation to one another at all.
    #[error("round {round} query {query} is not the fold of the previous round")]
    FoldMismatch { round: usize, query: usize },

    /// The final codeword does not encode the value the sumcheck ended at.
    #[error("the final codeword does not encode the sumcheck's final value")]
    FinalCheck,

    /// The grinding witness did not meet the demanded difficulty.
    #[error("proof-of-work witness rejected")]
    InvalidPowWitness,

    /// The proof carries a different number of opening-protocol evaluation batches than the
    /// public protocol schedules.
    ///
    /// Checked before any claim is registered, so a malformed proof is rejected before it can
    /// desync the transcript one claim at a time.
    #[error("expected {expected} opening batches, proof carries {actual}")]
    OpeningBatchCountMismatch { expected: usize, actual: usize },

    /// One opening batch has the wrong number of evaluations for its column list.
    #[error("table {table_idx} opening expected {expected} evaluations, got {actual}")]
    OpeningBatchSizeMismatch {
        table_idx: usize,
        expected: usize,
        actual: usize,
    },

    /// The sumcheck transcript did not verify.
    #[error(transparent)]
    Sumcheck(#[from] p3_sumcheck::SumcheckError),

    /// The proof's sumcheck data carries PoW witnesses.
    ///
    /// Every fold round runs at `pow_bits = 0` (see `prover::fold_rounds`), and the verifier
    /// replays each round with a freshly built, always-empty `pow_witnesses` vector, so
    /// whatever the proof carries here is read by nothing and bound to nothing: a third party
    /// could mutate it and keep a valid proof. `p3_sumcheck::ring_switch` rejects a non-empty
    /// `pow_witnesses` for the same reason.
    #[error("the sumcheck data carries {actual} PoW witnesses, expected none")]
    NonEmptyPowWitnesses { actual: usize },
}
