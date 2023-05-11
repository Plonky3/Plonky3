use alloc::vec::Vec;
use p3_commit::{DirectMMCS, MMCS};
use p3_field::Field;

pub struct FriProof<F, FE, M, MC>
where
    F: Field,
    FE: Field<Base = F>,
    M: MMCS<F>,
    MC: DirectMMCS<F>,
{
    queries: Vec<QueryProof<F, FE, M, MC>>,
}

pub struct QueryProof<F, FE, M, MC>
where
    F: Field,
    FE: Field<Base = F>,
    M: MMCS<F>,
    MC: DirectMMCS<F>,
{
    /// An opened row of each matrix that was part of this batch-FRI proof.
    leaves: Vec<Vec<F>>,
    leaf_opening_proofs: Vec<M::Proof>,
    steps: Vec<QueryStepProof<F, FE, MC>>,
}

pub struct QueryStepProof<F, FE, MC>
where
    F: Field,
    FE: Field<Base = F>,
    MC: DirectMMCS<F>,
{
    /// An opened row of each matrix that was part of this batch-FRI proof.
    leaves: FE,
    leaf_opening_proofs: MC::Proof,
}
