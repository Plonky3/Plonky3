//! Error types for the WARP accumulation scheme.

use alloc::string::String;

use thiserror::Error;

/// Errors raised by [`WarpVerifier::verify`](crate::protocol::WarpVerifier).
#[derive(Error, Debug)]
pub enum VerifierError {
    /// A sumcheck round-polynomial did not satisfy `h(0) + h(1) == claimed_sum`.
    #[error("sumcheck consistency failed at round {round} (phase {phase})")]
    SumcheckConsistency { phase: &'static str, round: usize },

    /// A sumcheck round-polynomial had the wrong number of coefficients.
    #[error("sumcheck round {round} (phase {phase}) had {got} coefficients, expected {expected}")]
    SumcheckDegree {
        phase: &'static str,
        round: usize,
        got: usize,
        expected: usize,
    },

    /// The §6.3 final-claim oracle check failed:
    /// `h_last(γ_last) != eq(τ, γ) · (ν₀ + ω · η)`.
    #[error("twin-constraint final claim mismatch")]
    TwinConstraintFinalClaim,

    /// The §8.2 final-claim oracle check failed:
    /// `h_last(α_last) != eq*(α) · µ`.
    #[error("multilinear-batching final claim mismatch")]
    MultilinearBatchingFinalClaim,

    /// A Merkle multi-opening proof failed to verify.
    #[error("merkle proof failed at index {index}: {reason}")]
    MerkleProof { index: usize, reason: String },

    /// The number of shift-query opening proofs did not match the configured count.
    #[error("expected {expected} shift-query openings, got {got}")]
    ShiftQueryCount { expected: usize, got: usize },

    /// The new accumulator-instance fields don't match what the verifier reconstructed.
    #[error("accumulator instance mismatch: {field}")]
    AccumulatorMismatch { field: &'static str },
}

/// Errors raised by [`WarpDecider::decide`](crate::protocol::WarpDecider).
#[derive(Error, Debug)]
pub enum DeciderError {
    /// The Merkle root of the recomputed codeword does not match `acc.x.rt`.
    #[error("decider: merkle root mismatch")]
    MerkleRoot,

    /// The multilinear extension of the codeword evaluated at α does not equal µ.
    #[error("decider: multilinear extension f̂(α) != µ")]
    MlEval,

    /// The bundled PESAT evaluation Pb(β, w) does not equal η.
    #[error("decider: bundled PESAT Pb(β, w) != η")]
    BundledPesat,

    /// The encoded witness `C(w)` does not equal the codeword `f`.
    #[error("decider: encoded witness != codeword")]
    EncodingMismatch,
}

/// Errors raised by [`Finalizer`](crate::finalize::Finalizer) impls.
#[derive(Error, Debug)]
pub enum FinalizerError {
    /// A finalizer that doesn't produce a transmissible proof was asked to
    /// verify one. The caller should run the local decider instead.
    #[error("finalizer does not support transmissible verification")]
    NoTransmissibleProof,

    /// Wrap a decider failure raised during finalization or verification.
    #[error(transparent)]
    Decider(#[from] DeciderError),

    /// Wrap a verifier failure raised during finalization-proof verification.
    #[error(transparent)]
    Verifier(#[from] VerifierError),

    /// A PCS opening proof failed.
    #[error("finalizer opening proof failed: {0}")]
    OpeningProof(String),

    /// The finalizer can't be applied to this configuration. The string is
    /// a static, human-readable explanation.
    #[error("finalizer unsupported: {0}")]
    Unsupported(&'static str),
}

/// Top-level error type for the `warp` crate.
#[derive(Error, Debug)]
pub enum WarpError {
    /// Wrap a verifier error.
    #[error(transparent)]
    Verifier(#[from] VerifierError),
    /// Wrap a decider error.
    #[error(transparent)]
    Decider(#[from] DeciderError),
    /// Wrap a finalizer error.
    #[error(transparent)]
    Finalizer(#[from] FinalizerError),
    /// Configuration is inconsistent (e.g. ℓ not a power of two, missing prior accs).
    #[error("invalid configuration: {0}")]
    Config(&'static str),
}
