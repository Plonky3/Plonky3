//! Proof payload for the HVZK-WHIR pipeline.

use alloc::vec::Vec;

use p3_commit::Mmcs;
use p3_sumcheck::zk::ZkSumcheckData;
use serde::{Deserialize, Serialize};

use crate::pcs::proof::QueryOpening;

/// Complete HVZK-WHIR opening proof.
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "F: Serialize, EF: Serialize, MT::Commitment: Serialize, MT::Proof: Serialize",
    deserialize = "F: Deserialize<'de>, EF: Deserialize<'de>, MT::Commitment: Deserialize<'de>, MT::Proof: Deserialize<'de>"
))]
pub struct ZkWhirProof<F: Send + Sync + Clone, EF, MT: Mmcs<F>> {
    /// Claimed opening evaluations, one per requested point.
    pub evals: Vec<EF>,
    /// Masked sumcheck transcripts, one per fold batch (`n_rounds + 1` batches).
    pub sumchecks: Vec<ZkSumcheckData<F, EF>>,
    /// Interleaved sumcheck mask commitment per fold batch.
    pub sumcheck_mask_commitments: Vec<MT::Commitment>,
    /// Per code-switching round payload.
    pub rounds: Vec<ZkRoundProof<F, EF, MT>>,
    /// Final masked base-case payload.
    pub base_case: BaseCaseZkProof<F, EF, MT>,
}

/// One HVZK code-switching round (Construction 9.7).
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "F: Serialize, EF: Serialize, MT::Commitment: Serialize, MT::Proof: Serialize",
    deserialize = "F: Deserialize<'de>, EF: Deserialize<'de>, MT::Commitment: Deserialize<'de>, MT::Proof: Deserialize<'de>"
))]
pub struct ZkRoundProof<F: Send + Sync + Clone, EF, MT: Mmcs<F>> {
    /// Commitment to the next interleaved ZK oracle.
    pub commitment: MT::Commitment,
    /// Commitment to the fresh code-switch mask `(folded randomness || pad)`.
    pub mask_commitment: MT::Commitment,
    /// Private out-of-domain answers `y_j = ze*(rho_j) . (message || randomness || pad)`.
    pub ood_answers: Vec<EF>,
    /// PoW witness after the commitments.
    pub pow_witness: F,
    /// STIR query openings against the previous oracle.
    pub queries: Vec<QueryOpening<F, EF, MT::Proof>>,
}

/// The masked base case (Construction 7.2).
///
/// - `f* = g~ + gamma * f` and the per-mask analogues are revealed in the clear.
/// - The fresh masks act as one-time pads.
/// - Nothing about the folded witness or the carried masks leaks.
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "F: Serialize, EF: Serialize, MT::Commitment: Serialize, MT::Proof: Serialize",
    deserialize = "F: Deserialize<'de>, EF: Deserialize<'de>, MT::Commitment: Deserialize<'de>, MT::Proof: Deserialize<'de>"
))]
pub struct BaseCaseZkProof<F: Send + Sync + Clone, EF, MT: Mmcs<F>> {
    /// Commitment to the fresh main mask `g`, in the folded source code.
    pub fresh_main_commitment: MT::Commitment,
    /// Commitments to the fresh blinds, one per carried mask group, in chronological group order.
    pub fresh_mask_commitments: Vec<MT::Commitment>,
    /// Fresh-side claim `mu_g = <g~, W> + sum_i <s~'_i, u_i>`.
    pub masked_claim: EF,
    /// Blinded source message `f* = g~ + gamma . f`.
    pub blinded_message: Vec<EF>,
    /// Blinded source encoding randomness `r* = r_g + gamma . r`.
    pub blinded_randomness: Vec<EF>,
    /// Per carried mask: blinded message and blinded encoding randomness.
    pub blinded_masks: Vec<BlindedMask<EF>>,
    /// PoW witness before the spot checks.
    pub pow_witness: F,
    /// Spot-check openings of the (virtual) source oracle, i.e. leaves of the last committed oracle.
    pub source_queries: Vec<QueryOpening<F, EF, MT::Proof>>,
    /// Spot-check openings of the fresh main mask `g` at the same positions.
    pub fresh_main_queries: Vec<QueryOpening<F, EF, MT::Proof>>,
    /// Per group: paired openings of the carried oracle and its fresh blind.
    /// Each opened row spans the whole group.
    pub mask_queries: Vec<Vec<MaskOpeningPair<F, EF, MT>>>,
}

/// One mask oracle's blinded reveal.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BlindedMask<EF> {
    /// `xi*_i = s~'_i + gamma . xi_i`.
    pub message: Vec<EF>,
    /// `r*_i = r'_i + gamma . r_i`.
    pub randomness: Vec<EF>,
}

/// Paired openings of a carried mask and its fresh blind at one position.
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "F: Serialize, EF: Serialize, MT::Proof: Serialize",
    deserialize = "F: Deserialize<'de>, EF: Deserialize<'de>, MT::Proof: Deserialize<'de>"
))]
pub struct MaskOpeningPair<F: Send + Sync + Clone, EF, MT: Mmcs<F>> {
    /// Opening of the carried mask oracle `xi_i`.
    pub carried: QueryOpening<F, EF, MT::Proof>,
    /// Opening of the fresh blind `s'_i`.
    pub fresh: QueryOpening<F, EF, MT::Proof>,
}
