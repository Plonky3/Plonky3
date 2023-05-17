use alloc::vec::Vec;
use p3_commit::{DirectMMCS, MMCS};
use p3_field::FieldExtension;

pub struct FriProof<FE, M, MC>
where
    FE: FieldExtension,
    M: MMCS<FE::Base>,
    MC: DirectMMCS<FE::Base>,
{
    _queries: Vec<QueryProof<FE, M, MC>>,
}

pub struct QueryProof<FE, M, MC>
where
    FE: FieldExtension,
    M: MMCS<FE::Base>,
    MC: DirectMMCS<FE::Base>,
{
    /// An opened row of each matrix that was part of this batch-FRI proof.
    _leaves: Vec<Vec<FE::Base>>,
    _leaf_opening_proofs: Vec<M::Proof>,
    _steps: Vec<QueryStepProof<FE, MC>>,
}

pub struct QueryStepProof<FE, MC>
where
    FE: FieldExtension,
    MC: DirectMMCS<FE::Base>,
{
    /// An opened row of each matrix that was part of this batch-FRI proof.
    _leaves: FE::Extension,
    _leaf_opening_proofs: MC::Proof,
}
