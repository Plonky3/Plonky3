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
    /// Proof of work for the phase before the opening-batching challenge.
    ///
    /// This witness belongs to the commitment scheme wrapped around the low-degree test.
    ///
    /// The challenge it guards is drawn before the reduced openings even exist.
    /// The two witness sets below guard challenges drawn inside the low-degree test.
    ///
    /// It travels in this proof because that is where the verifier meets it.
    /// The replay happens outside the bracket the low-degree test runs in.
    pub batch_pow_witness: Witness,
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
/// The per-query equivalent shipped one full authentication path per query.
///
/// Queries into the same tree overlap heavily.
///
/// Shared sibling digests are therefore deduplicated by the multiproof.
///
/// # Why no arity is carried here
///
/// The arity of every round follows from the committed heights and the parameters.
///
/// Both sides hold those before a proof exists.
///
/// A round carrying its own arity would declare a length the verifier already knows.
///
/// A length declared twice is a length that can disagree.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(bound = "")]
pub struct CommitPhaseMultiStep<F: Field, M: Mmcs<F>> {
    /// For each query, the openings of the commit phase codeword at the
    /// sibling locations. For arity k, each entry contains k-1 sibling values.
    pub sibling_values: Vec<Vec<F>>,
    /// One shared proof authenticating every query's row in this round's tree.
    pub opening_proof: M::MultiProof,
}

/// All queries' openings of one input batch commitment, sharing one proof.
///
/// The multi-opening analogue of a single-query batch opening.
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
