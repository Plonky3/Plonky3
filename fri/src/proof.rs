use alloc::vec::Vec;

use p3_challenger::GrindingChallenger;
use p3_commit::Mmcs;
use serde::{Deserialize, Serialize};

use crate::FriConfig;

#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct FriProof<FC: FriConfig> {
    pub(crate) commit_phase_commits: Vec<<FC::CommitPhaseMmcs as Mmcs<FC::Challenge>>::Commitment>,
    pub(crate) query_proofs: Vec<QueryProof<FC>>,
    // This could become Vec<FC::Challenge> if this library was generalized to support non-constant
    // final polynomials.
    pub(crate) final_poly: FC::Challenge,
    pub(crate) pow_witness: <FC::Challenger as GrindingChallenger>::Witness,
}

#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct QueryProof<FC: FriConfig> {
    /// For each input commitment, this contains openings of each matrix at the queried location,
    /// along with an opening proof.
    // pub(crate) input_openings: Vec<InputOpening<FC>>,

    /// For each commit phase commitment, this contains openings of a commit phase codeword at the
    /// queried location, along with an opening proof.
    pub(crate) commit_phase_openings: Vec<CommitPhaseProofStep<FC>>,
}

/// Openings of each input codeword at the queried location, along with an opening proof, for a
/// single commitment round.
/*
#[derive(Serialize, Deserialize)]
pub struct InputOpening<FC: FriConfig> {
    /// The opening of each input codeword at the queried location.
    pub(crate) opened_values: Vec<Vec<FC::Val>>,

    pub(crate) opening_proof: <FC::InputMmcs as Mmcs<FC::Val>>::Proof,
}
*/

#[derive(Serialize, Deserialize)]
pub struct CommitPhaseProofStep<FC: FriConfig> {
    /// The opening of the commit phase codeword at the sibling location.
    // This may change to Vec<FC::Challenge> if the library is generalized to support other FRI
    // folding arities besides 2, meaning that there can be multiple siblings.
    pub(crate) sibling_value: FC::Challenge,

    pub(crate) opening_proof: <FC::CommitPhaseMmcs as Mmcs<FC::Challenge>>::Proof,
}
