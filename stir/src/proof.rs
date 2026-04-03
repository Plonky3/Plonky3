//! STIR proof types.

use alloc::vec::Vec;

use p3_commit::Mmcs;
use p3_field::Field;
use serde::{Deserialize, Serialize};

/// A complete STIR proof.
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "Witness: Serialize",
    deserialize = "Witness: Deserialize<'de>",
))]
pub struct StirProof<EF: Field, M: Mmcs<EF>, Witness> {
    /// Commitment to the initial codeword on L_0.
    pub initial_commitment: M::Commitment,

    /// One proof per intermediate STIR round (excluding the final send).
    pub round_proofs: Vec<StirRoundProof<EF, M, Witness>>,

    /// Coefficients of the final polynomial, sent directly to the verifier.
    pub final_polynomial: Vec<EF>,

    /// Proof-of-work witness for the final folding step.
    pub final_folding_pow_witness: Witness,

    /// Proof-of-work witness for the final query phase.
    pub final_pow_witness: Witness,

    /// Merkle openings for the final consistency queries against the last committed codeword.
    pub final_query_proofs: Vec<StirFinalQueryProof<EF, M>>,

    /// Deduplicated fold-domain indices queried in the first round (or the final round when
    /// there are no intermediate rounds). Used by the PCS layer for the input-binding check.
    pub first_round_query_indices: Vec<usize>,
}

/// Proof for a single STIR round.
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "Witness: Serialize",
    deserialize = "Witness: Deserialize<'de>",
))]
pub struct StirRoundProof<EF: Field, M: Mmcs<EF>, Witness> {
    /// Commitment to the next round's codeword.
    pub commitment: M::Commitment,

    /// Proof-of-work witness for the folding step.
    pub folding_pow_witness: Witness,

    // TODO: we don't need to send this in theory
    /// Coefficients of the folded polynomial `g_i`.
    pub fold_polynomial: Vec<EF>,

    /// Proof-of-work witness for the STIR query phase of this round.
    pub pow_witness: Witness,

    /// Evaluations of the FOLDED polynomial at each OOD point.
    pub ood_answers: Vec<EF>,

    /// Shake polynomial coefficients.
    ///
    /// `S(X) = sum_{y in P} (Ans(X) - Ans(y)) / (X - y)` where `P` is the set of all OOD +
    /// queried points. Sent by the prover and checked by the verifier at a random evaluation
    /// point.
    pub shake_polynomial: Vec<EF>,

    /// Merkle openings for each STIR query.
    pub query_proofs: Vec<StirQueryProof<EF, M>>,

    /// Openings from the next-round commitment at the sampled shift-query points.
    ///
    /// These bind the committed next-round oracle to the current round's
    /// degree-corrected quotient state.
    pub next_query_proofs: Vec<StirNextQueryProof<EF, M>>,
}

/// Merkle opening and fiber evaluations for a single STIR query.
///
/// Contains openings from the current round's commitment (for the fold computation).
/// In Construction 5.2, the next-round oracle is the degree-corrected quotient of the
/// fold polynomial, so fold-consistency is not checked directly here; it is established
/// via the shake-polynomial argument instead.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound = "")]
pub struct StirQueryProof<EF: Field, M: Mmcs<EF>> {
    /// The `k = 2^log_folding_factor` evaluations of the current polynomial at the fiber.
    pub fiber_evals: Vec<EF>,

    /// Merkle opening proof authenticating `fiber_evals` against the current committed codeword.
    pub opening_proof: M::Proof,
}

/// Merkle opening for the next-round commitment at a sampled current-round query point.
///
/// The next-round codeword is committed as a fiber matrix, so opening a single queried value
/// reveals the entire corresponding row.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound = "")]
pub struct StirNextQueryProof<EF: Field, M: Mmcs<EF>> {
    /// The opened row from the next-round commitment's fiber matrix.
    pub row_evals: Vec<EF>,

    /// Merkle opening proof authenticating `row_evals` against the next-round commitment.
    pub opening_proof: M::Proof,
}

/// Merkle opening and fiber evaluations for a final-round query.
///
/// Used to verify consistency between the last committed codeword and the final polynomial.
/// Unlike [`StirQueryProof`] there is no subsequent commitment to link to.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound = "")]
pub struct StirFinalQueryProof<EF: Field, M: Mmcs<EF>> {
    /// The `k = 2^log_folding_factor` evaluations of the last committed codeword at the fiber.
    pub fiber_evals: Vec<EF>,

    /// Merkle opening proof authenticating `fiber_evals` against the last committed codeword.
    pub opening_proof: M::Proof,
}
