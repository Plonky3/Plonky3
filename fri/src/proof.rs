use alloc::vec::Vec;
use p3_commit::mmcs::{DirectMMCS, MMCS};
use p3_field::field::{AbstractFieldExtension, Field};

pub struct FriProof<F, FE, M, MC>
where
    F: Field,
    FE: AbstractFieldExtension<F>,
    M: MMCS<F>,
    MC: DirectMMCS<F>,
{
    queries: Vec<QueryProof<F, FE, M, MC>>,
}

pub struct QueryProof<F, FE, M, MC>
where
    F: Field,
    FE: AbstractFieldExtension<F>,
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
    FE: AbstractFieldExtension<F>,
    MC: DirectMMCS<F>,
{
    /// An opened row of each matrix that was part of this batch-FRI proof.
    leaves: FE,
    leaf_opening_proofs: MC::Proof,
}
