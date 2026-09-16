use alloc::vec::Vec;

use p3_commit::Mmcs;
use p3_field::Field;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "Witness: Serialize, InputProof: Serialize",
    deserialize = "Witness: Deserialize<'de>, InputProof: Deserialize<'de>"
))]
pub struct CircleFriProof<F: Field, M: Mmcs<F>, Witness, InputProof> {
    pub commit_phase_commits: Vec<M::Commitment>,
    pub commit_pow_witnesses: Vec<Witness>,
    /// Openings of the input commitments at every query index, sharing one proof
    /// per committed tree.
    pub input_openings: InputProof,
    /// For each commit phase commitment, the openings of the commit phase codeword
    /// at every queried location, all authenticated by one shared proof per round.
    pub commit_phase_openings: Vec<CircleCommitPhaseMultiStep<F, M>>,
    // This could become Vec<FC::Challenge> if this library was generalized to support non-constant
    // final polynomials.
    pub final_poly: F,
    pub pow_witness: Witness,
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
/// Circle folding halves the domain and nothing else.
///
/// So every round folds by two.
///
/// The round count follows from the tallest claimed height.
///
/// Both sides hold that height before a proof exists.
///
/// A round carrying its own arity would declare a length the verifier already knows.
///
/// A length declared twice is a length that can disagree.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(bound = "")]
pub struct CircleCommitPhaseMultiStep<F: Field, M: Mmcs<F>> {
    /// For each query, the openings of the commit phase codeword at the sibling
    /// locations. For arity k, each entry contains k-1 sibling values.
    pub sibling_values: Vec<Vec<F>>,
    /// One shared proof authenticating every query's row in this round's tree.
    pub opening_proof: M::MultiProof,
}
