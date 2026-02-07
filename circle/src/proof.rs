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
    pub query_proofs: Vec<CircleQueryProof<F, M, InputProof>>,
    // This could become Vec<FC::Challenge> if this library was generalized to support non-constant
    // final polynomials.
    pub final_poly: F,
    pub pow_witness: Witness,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "InputProof: Serialize",
    deserialize = "InputProof: Deserialize<'de>",
))]
pub struct CircleQueryProof<F: Field, M: Mmcs<F>, InputProof> {
    pub input_proof: InputProof,
    /// For each commit phase commitment, this contains openings of a commit phase codeword at the
    /// queried location, along with an opening proof.
    pub commit_phase_openings: Vec<CircleCommitPhaseProofStep<F, M>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(bound = "")]
pub struct CircleCommitPhaseProofStep<F: Field, M: Mmcs<F>> {
    /// The log2 of the folding arity used for this step.
    pub log_arity: u8,
    /// The openings of the commit phase codeword at the sibling locations.
    /// For arity k, this contains k-1 sibling values.
    pub sibling_values: Vec<F>,

    pub opening_proof: M::Proof,
}
