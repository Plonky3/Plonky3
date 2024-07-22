use std::fmt::Debug;

use alloc::vec::Vec;

use p3_commit::Mmcs;
use p3_field::Field;
use p3_util::log2_strict_usize;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "PowWitness: Serialize",
    deserialize = "PowWitness: Deserialize<'de>"
))]
pub struct FriProof<F: Field, M: Mmcs<F>, PowWitness> {
    pub commit_phase_commits: Vec<M::Commitment>,
    pub final_polys: Vec<Vec<F>>,
    pub pow_witness: PowWitness,
    pub query_proofs: Vec<QueryProof<F, M>>,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(bound = "")]
pub struct QueryProof<F: Field, M: Mmcs<F>> {
    /// For each commit phase commitment, this contains openings of a commit phase codeword at the
    /// queried location, along with an opening proof.
    pub commit_phase_openings: Vec<CommitPhaseProofStep<F, M>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(bound = "")]
pub struct CommitPhaseProofStep<F: Field, M: Mmcs<F>> {
    pub siblings: Vec<Vec<F>>,
    pub proof: M::Proof,
}

impl<F: Field, M: Mmcs<F>> QueryProof<F, M> {
    pub fn log_folding_arities(&self) -> Vec<usize> {
        self.commit_phase_openings
            .iter()
            .map(|step| step.log_folding_arity())
            .collect()
    }
}

impl<F: Field, M: Mmcs<F>> CommitPhaseProofStep<F, M> {
    pub fn log_folding_arity(&self) -> usize {
        self.siblings
            .iter()
            .map(|sibs| log2_strict_usize(sibs.len() + 1))
            .max()
            .unwrap()
    }
}
