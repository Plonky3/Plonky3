use alloc::vec::Vec;

use p3_commit::Mmcs;
use p3_field::Field;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "Witness: Serialize, InputProof: Serialize",
    deserialize = "Witness: Deserialize<'de>, InputProof: Deserialize<'de>"
))]
pub struct FriProof<F: Field, M: Mmcs<F>, Witness, InputProof> {
    pub commit_phase_commits: Vec<M::Commitment>,
    pub commit_pow_witnesses: Vec<Witness>,
    /// Openings of the input commitments at every query index, one entry per
    /// input batch, each covering all queries with a single shared proof.
    pub input_openings: InputProof,
    /// For each commit phase commitment, the openings of the commit phase
    /// codeword at every queried location, all authenticated by one shared
    /// proof per round.
    pub commit_phase_openings: Vec<CommitPhaseMultiStep<F, M>>,
    pub final_poly: Vec<F>,
    pub query_pow_witness: Witness,
}

/// All queries' openings of one commit-phase codeword, sharing one proof.
///
/// The per-query equivalent shipped one full authentication path per query;
/// queries into the same tree overlap heavily, so shared sibling digests are
/// deduplicated by the multiproof.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(bound = "")]
pub struct CommitPhaseMultiStep<F: Field, M: Mmcs<F>> {
    /// The log2 of the folding arity used for this round.
    ///
    /// The schedule is a protocol-wide constant, so it lives once per round
    /// rather than once per query.
    pub log_arity: u8,
    /// For each query, the openings of the commit phase codeword at the
    /// sibling locations. For arity k, each entry contains k-1 sibling values.
    pub sibling_values: Vec<Vec<F>>,
    /// One shared proof authenticating every query's row in this round's tree.
    pub opening_proof: M::MultiProof,
}

impl<F: Field, M: Mmcs<F>> CommitPhaseMultiStep<F, M> {
    /// Convert `log_arity` to `usize` and enforce the protocol bounds.
    ///
    /// Returns `None` when `log_arity` is zero or exceeds `max_log_arity`.
    #[inline]
    pub(crate) fn checked_log_arity(&self, max_log_arity: usize) -> Option<usize> {
        let log_arity = self.log_arity as usize;
        (1..=max_log_arity)
            .contains(&log_arity)
            .then_some(log_arity)
    }
}

/// All queries' openings of one input batch commitment, sharing one proof.
///
/// The multi-opening analogue of [`p3_commit::BatchOpening`].
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "T: Serialize",
    deserialize = "T: serde::de::DeserializeOwned"
))]
pub struct BatchMultiOpening<T: Send + Sync + Clone, InputMmcs: Mmcs<T>> {
    /// The opened row values: `opened_values[q][m]` is the row of matrix `m`
    /// at query `q`'s (reduced) index.
    pub opened_values: Vec<Vec<Vec<T>>>,
    /// One shared proof authenticating every query's rows.
    pub opening_proof: InputMmcs::MultiProof,
}
