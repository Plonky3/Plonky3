use alloc::vec::Vec;

use p3_commit::Mmcs;

use crate::FriConfig;

#[allow(dead_code)] // TODO: fields should be used soon
pub struct FriProof<FC: FriConfig> {
    pub(crate) query_proofs: Vec<QueryProof<FC>>,
}

#[allow(dead_code)] // TODO: fields should be used soon
#[allow(clippy::type_complexity)]
pub struct QueryProof<FC: FriConfig> {
    /// For each input commitment, this contains openings of each matrix at the queried location,
    /// along with an opening proof.
    pub(crate) input_openings: Vec<(
        Vec<Vec<FC::Domain>>,
        <FC::InputMmcs as Mmcs<FC::Domain>>::Proof,
    )>,

    /// For each commit phase commitment, this contains openings of each matrix at the queried
    /// location, along with an opening proof.
    // TODO: There should be two opened values for each matrix, as in `[FC::Challenge; 2]`.
    pub(crate) commit_phase_openings: Vec<(
        Vec<Vec<FC::Challenge>>,
        <FC::CommitPhaseMmcs as Mmcs<FC::Challenge>>::Proof,
    )>,
}
