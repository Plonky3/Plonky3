//! Proof types produced by the commit-fold-query pipeline.

use alloc::vec::Vec;

use p3_binary_field::BinaryField128;
use p3_commit::Mmcs;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::{OpeningEvals, SumcheckData};
use serde::{Deserialize, Serialize};

/// One intermediate folding round: the commitment to that round's folded codeword, and its
/// query openings.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "MT::Commitment: Serialize, MT::MultiProof: Serialize",
    deserialize = "MT::Commitment: Deserialize<'de>, MT::MultiProof: Deserialize<'de>"
))]
pub struct RoundProof<MT: Mmcs<BinaryField128>> {
    /// Merkle root of this round's folded codeword.
    pub commitment: MT::Commitment,
    /// Opened rows, two per query — the low- then high-indexed symbol of that query's fold
    /// pair at this round — in the order the query indices were sampled.
    pub opened_values: Vec<Vec<BinaryField128>>,
    /// One multiproof authenticating every opened row of this round together.
    pub multi_proof: MT::MultiProof,
}

/// A full opening proof.
///
/// The last folding round has no [`RoundProof`]: its codeword has already shrunk to
/// `2^log_inv_rate` symbols, so it is sent in full as `final_codeword` instead of committed —
/// a Merkle path over it would only repeat what the verifier can already read directly. Every
/// earlier round appears in `rounds`, committed and opened at the paired positions each query
/// needs.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "MT::Commitment: Serialize, MT::MultiProof: Serialize",
    deserialize = "MT::Commitment: Deserialize<'de>, MT::MultiProof: Deserialize<'de>"
))]
pub struct BinaryPcsProof<MT: Mmcs<BinaryField128>> {
    /// Every sumcheck round message, in round order.
    pub sumcheck: SumcheckData<BinaryField128, BinaryField128>,
    /// One entry per intermediate folding round, i.e. every folding round except the last.
    pub rounds: Vec<RoundProof<MT>>,
    /// Openings of the base commitment: two width-1 rows per query — the pair's low- then
    /// high-indexed symbol — in the order query indices were sampled.
    pub base_opened_values: Vec<Vec<BinaryField128>>,
    /// Multiproof for the base commitment's queried rows.
    pub base_multi_proof: MT::MultiProof,
    /// The final folded codeword, sent in full.
    pub final_codeword: Poly<BinaryField128>,
    /// Witness for the single grind before the query phase.
    pub pow_witness: BinaryField128,
    /// Opening-protocol claimed evaluations, in schedule order: one batch per
    /// [`p3_sumcheck::OpeningProtocol`] entry, each holding the current-point evaluations
    /// separately from the repeat-last successor-point evaluations.
    pub evals: Vec<OpeningEvals<BinaryField128>>,
}
