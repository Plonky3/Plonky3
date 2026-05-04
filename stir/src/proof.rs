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
    /// Commitment to the folded oracle `g_i` on the next round's domain.
    pub commitment: M::Commitment,

    /// Proof-of-work witness for the folding step.
    pub folding_pow_witness: Witness,

    /// Evaluations of the FOLDED polynomial at each OOD point.
    pub ood_answers: Vec<EF>,

    /// Proof-of-work witness for the STIR query phase of this round.
    pub pow_witness: Witness,

    /// Answer polynomial coefficients.
    ///
    /// The unique polynomial `Ans(X)` of degree `< |P|` interpolating `(y_i, v_i)` for every
    /// `y_i ∈ P` (OOD + queried points) and its claimed value `v_i`. Sent by the prover so the
    /// verifier avoids the O(|P|²) Newton interpolation; correctness against the queried/OOD
    /// values is enforced together with `shake_polynomial` at a random `rho`.
    pub ans_polynomial: Vec<EF>,

    /// Shake polynomial coefficients.
    ///
    /// `S(X) = sum_{y in P} (Ans(X) - Ans(y)) / (X - y)` where `P` is the set of all OOD +
    /// queried points. Sent by the prover and checked by the verifier at a random evaluation
    /// point: `S(rho) == sum_i (Ans(rho) - v_i) / (rho - y_i)`. Both `ans_polynomial` and
    /// `shake_polynomial` are observed in the transcript before `rho` is sampled, so a
    /// malicious prover cannot fit `Ans` to a known `rho`.
    pub shake_polynomial: Vec<EF>,

    /// Merkle openings for each STIR query.
    pub query_proofs: Vec<StirQueryProof<EF, M>>,
}

/// Merkle opening and fiber evaluations for a single STIR query.
///
/// The opened row is taken from the CURRENT commitment:
/// - in round 0 this is the current oracle itself,
/// - in later rounds this is the previous round's folded oracle `g_i`.
///
/// The verifier translates the opened row into current-round oracle values before
/// applying the fold.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound = "")]
pub struct StirQueryProof<EF: Field, M: Mmcs<EF>> {
    /// The opened row from the current commitment's fiber matrix.
    pub row_evals: Vec<EF>,

    /// Merkle opening proof authenticating `row_evals` against the current commitment.
    pub opening_proof: M::Proof,
}

/// Merkle opening and fiber evaluations for a final-round query.
///
/// Used to verify consistency between the last virtual oracle and the final polynomial.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound = "")]
pub struct StirFinalQueryProof<EF: Field, M: Mmcs<EF>> {
    /// The opened row from the last commitment's fiber matrix.
    pub row_evals: Vec<EF>,

    /// Merkle opening proof authenticating `row_evals` against the last committed codeword.
    pub opening_proof: M::Proof,
}
