use alloc::vec::Vec;
use p3_commit::{DirectMMCS, MMCS};
use p3_field::Field;

pub struct FriProof<F, M, MC>
where
    F: Field,
    M: MMCS<F::Base>,
    MC: DirectMMCS<F::Base>,
{
    queries: Vec<QueryProof<F, M, MC>>,
}

pub struct QueryProof<F, M, MC>
where
    F: Field,
    M: MMCS<F::Base>,
    MC: DirectMMCS<F::Base>,
{
    /// An opened row of each matrix that was part of this batch-FRI proof.
    leaves: Vec<Vec<F::Base>>,
    leaf_opening_proofs: Vec<M::Proof>,
    steps: Vec<QueryStepProof<F, MC>>,
}

pub struct QueryStepProof<F, MC>
where
    F: Field,
    MC: DirectMMCS<F::Base>,
{
    /// An opened row of each matrix that was part of this batch-FRI proof.
    leaves: F,
    leaf_opening_proofs: MC::Proof,
}
