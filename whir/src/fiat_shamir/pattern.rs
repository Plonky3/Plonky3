//! Transcript pattern labels for the Fiat-Shamir domain separator.

use p3_field::Field;

/// Top-level classification of a transcript operation.
///
/// Each step in the protocol transcript is one of three kinds:
/// - Prover sends data (observed into the sponge).
/// - Verifier draws randomness (sampled from the sponge).
/// - Non-binding auxiliary data (hints, not absorbed).
///
/// The discriminant is encoded as a field element and combined with a
/// sub-label and count to form a single domain separator entry.
#[derive(Debug, Clone, Copy)]
pub enum Pattern {
    /// Verifier samples randomness from the transcript.
    Sample,
    /// Prover observes (absorbs) data into the transcript.
    Observe,
    /// Prover supplies non-binding auxiliary data.
    Hint,
}

impl Pattern {
    /// Convert to a field element using the enum discriminant.
    #[must_use]
    pub fn as_field_element<F: Field>(self) -> F {
        F::from_u8(self as u8)
    }
}

/// Sub-labels for sampled (verifier-drawn) transcript items.
///
/// Each variant identifies the semantic role of a challenge or
/// randomness drawn from the sponge during the protocol.
#[derive(Debug, Clone, Copy)]
pub enum Sample {
    /// Randomness for the initial linear combination of constraints.
    InitialCombinationRandomness,
    /// Per-round folding challenge in the sumcheck protocol.
    FoldingRandomness,
    /// Randomness for combining quotient polynomials after each round.
    CombinationRandomness,
    /// Byte-encoded query positions for STIR proximity tests.
    StirQueries,
    /// Byte-encoded query positions for the final proximity test.
    FinalQueries,
    /// Challenge bytes for proof-of-work grinding.
    PowQueries,
    /// Out-of-domain evaluation point.
    OodQuery,
    /// Dummy single-element sample used as a transcript checkpoint.
    ///
    /// Drawn between proof-of-work and query generation in non-final rounds
    /// to keep the domain separator synchronized with the actual prover/verifier
    /// transcript.
    TranscriptCheckpoint,
}

impl Sample {
    /// Convert to a field element using the enum discriminant.
    #[must_use]
    pub fn as_field_element<F: Field>(self) -> F {
        F::from_u8(self as u8)
    }
}

/// Sub-labels for observed (prover-absorbed) transcript items.
///
/// Each variant identifies the semantic role of data the prover
/// commits into the sponge.
#[derive(Debug, Clone, Copy)]
pub enum Observe {
    /// Merkle tree root digest for a committed polynomial.
    MerkleDigest,
    /// Evaluations at out-of-domain points.
    OodAnswers,
    /// Coefficients of the degree-2 sumcheck round polynomial.
    ///
    /// Only c_0 and c_2 are sent; c_1 is derived by the verifier
    /// as `claimed_sum - c_0`.
    SumcheckPoly,
    /// Evaluation answers at STIR query positions.
    StirAnswers,
    /// Coefficients of the final folded polynomial.
    FinalCoeffs,
    /// Proof-of-work nonce solving the grinding challenge.
    PowNonce,
    /// A public protocol parameter value.
    ///
    /// Used in the domain separator header to bind the transcript
    /// to the specific protocol configuration. Each parameter is
    /// encoded as a (marker, value) pair.
    ProtocolParam,
}

impl Observe {
    /// Convert to a field element using the enum discriminant.
    #[must_use]
    pub fn as_field_element<F: Field>(self) -> F {
        F::from_u8(self as u8)
    }
}

/// Sub-labels for hint (non-binding auxiliary) transcript items.
///
/// Hints are data the prover sends to help the verifier reconstruct
/// information but are not absorbed into the sponge. They do not
/// affect soundness.
#[derive(Debug, Clone, Copy)]
pub enum Hint {
    /// Indices and positions of STIR query points.
    StirQueries,
    /// Evaluation values at STIR query positions.
    StirAnswers,
    /// Authentication paths for Merkle tree opening proofs.
    MerkleProof,
    /// Precomputed weight evaluations deferred from earlier rounds.
    DeferredWeightEvaluations,
}

impl Hint {
    /// Convert to a field element using the enum discriminant.
    #[must_use]
    pub fn as_field_element<F: Field>(self) -> F {
        F::from_u8(self as u8)
    }
}
