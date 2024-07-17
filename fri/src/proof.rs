use alloc::vec::Vec;

use p3_commit::Mmcs;
use p3_field::Field;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "PowWitness: Serialize, InputProof: Serialize",
    deserialize = "PowWitness: Deserialize<'de>, InputProof: Deserialize<'de>"
))]
pub struct FriProof<F: Field, M: Mmcs<F>, PowWitness, InputProof> {
    pub commit_phase_commits: Vec<M::Commitment>,
    pub final_poly: Vec<F>,
    pub pow_witness: PowWitness,
    pub query_proofs: Vec<QueryProof<F, M, InputProof>>,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "InputProof: Serialize",
    deserialize = "InputProof: Deserialize<'de>",
))]
pub struct QueryProof<F: Field, M: Mmcs<F>, InputProof> {
    pub input_proof: InputProof,
    /// For each commit phase commitment, this contains openings of a commit phase codeword at the
    /// queried location, along with an opening proof.
    pub commit_phase_openings: Vec<CommitPhaseProofStep<F, M>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(bound = "")]
pub struct CommitPhaseProofStep<F: Field, M: Mmcs<F>> {
    pub openings: Vec<Vec<F>>,
    pub proof: M::Proof,
}
