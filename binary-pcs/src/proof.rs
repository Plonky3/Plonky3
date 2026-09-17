//! Proof types produced by the commit-fold-query pipeline.

use alloc::vec::Vec;

use p3_commit::Mmcs;
use p3_field::Field;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::{OpeningEvals, SumcheckData};
use serde::{Deserialize, Serialize};

/// One intermediate fold batch: the commitment to that round's folded codeword, and its
/// query openings.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "MX::Commitment: Serialize, MX::MultiProof: Serialize",
    deserialize = "MX::Commitment: Deserialize<'de>, MX::MultiProof: Deserialize<'de>"
))]
pub struct RoundProof<EF: Field, MX: Mmcs<EF>> {
    /// Merkle root of this round's folded codeword.
    pub commitment: MX::Commitment,
    /// Every symbol of each queried coset, one width-1 row per symbol. Coset width is
    /// `2^arity` for the next fold batch, derived from the verifier's configuration.
    /// Cosets follow sampled query order; symbols inside a coset are ascending.
    pub opened_values: Vec<Vec<EF>>,
    /// One multiproof authenticating every opened row of this round together.
    pub multi_proof: MX::MultiProof,
}

/// A full opening proof.
///
/// The last fold batch has no round entry.
/// Its codeword has already shrunk to `2^log_inv_rate` symbols, so it is sent in full.
///
/// A Merkle path over it would only repeat what the verifier can already read.
/// Every earlier batch is committed and opened at every coset position a query needs.
///
/// The base commitment holds the committed alphabet, every folded one the challenge field.
/// The first fold of the schedule is what crosses between the two.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "MT::Commitment: Serialize, MT::MultiProof: Serialize, MX::Commitment: Serialize, MX::MultiProof: Serialize",
    deserialize = "MT::Commitment: Deserialize<'de>, MT::MultiProof: Deserialize<'de>, MX::Commitment: Deserialize<'de>, MX::MultiProof: Deserialize<'de>"
))]
pub struct BinaryPcsProof<F: Field, EF: Field, MT: Mmcs<F>, MX: Mmcs<EF>> {
    /// Every sumcheck round message, in round order.
    pub sumcheck: SumcheckData<F, EF>,
    /// One entry per intermediate fold batch, excluding the final batch.
    pub rounds: Vec<RoundProof<EF, MX>>,
    /// All symbols of the sampled base cosets, in query order with ascending coset offsets.
    /// The first batch's arity determines the number of width-1 rows per query.
    pub base_opened_values: Vec<Vec<F>>,
    /// Multiproof for the base commitment's queried rows.
    pub base_multi_proof: MT::MultiProof,
    /// The final folded codeword, sent in full and absorbed before query grinding/sampling.
    pub final_codeword: Poly<EF>,
    /// Witness for the single grind before the query phase.
    pub pow_witness: F,
    /// Claimed evaluations, in the opening protocol's own schedule order.
    ///
    /// One batch per protocol entry.
    /// Each batch holds the current-point values separately from the successor-point ones.
    pub evals: Vec<OpeningEvals<EF>>,
}
