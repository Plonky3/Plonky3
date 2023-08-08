use alloc::vec::Vec;

use p3_commit::{DirectMMCS, MMCS};
use p3_field::{ExtensionField, Field};

#[allow(dead_code)] // TODO: fields should be used soon
pub struct FriProof<F, Challenge, M, MC>
where
    F: Field,
    Challenge: ExtensionField<F>,
    M: MMCS<F>,
    MC: DirectMMCS<Challenge>,
{
    pub(crate) query_proofs: Vec<QueryProof<F, Challenge, M, MC>>,
}

#[allow(dead_code)] // TODO: fields should be used soon
pub struct QueryProof<F, Challenge, M, MC>
where
    F: Field,
    Challenge: ExtensionField<F>,
    M: MMCS<F>,
    MC: DirectMMCS<Challenge>,
{
    /// For each MMCS commitment, this contains openings of each matrix at the queried location,
    /// along with an opening proof.
    pub(crate) input_openings: Vec<(Vec<Vec<F>>, M::Proof)>,
    pub(crate) commit_phase_openings: Vec<(Vec<Vec<Challenge>>, MC::Proof)>,
}
