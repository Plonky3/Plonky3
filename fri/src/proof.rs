use alloc::vec::Vec;

use p3_commit::MMCS;

use crate::FriConfig;

#[allow(dead_code)] // TODO: fields should be used soon
pub struct FriProof<FC: FriConfig> {
    pub(crate) query_proofs: Vec<QueryProof<FC>>,
}

#[allow(dead_code)] // TODO: fields should be used soon
pub struct QueryProof<FC: FriConfig> {
    /// For each input commitment, this contains openings of each matrix at the queried location,
    /// along with an opening proof.
    pub(crate) input_openings: Vec<(Vec<Vec<FC::Val>>, <FC::InputMmcs as MMCS<FC::Val>>::Proof)>,
    /// For each commit phase commitment, this contains openings of each matrix at the queried
    /// location, along with an opening proof.
    pub(crate) commit_phase_openings: Vec<(
        Vec<Vec<FC::Challenge>>,
        <FC::CommitPhaseMmcs as MMCS<FC::Challenge>>::Proof,
    )>,
}
