use alloc::vec::Vec;
use p3_commit::{DirectMMCS, MMCS};
use p3_field::{AbstractFieldExtension, Field};

pub struct FriProof<F, EF, M, MC>
where
    F: Field,
    EF: AbstractFieldExtension<F>,
    M: MMCS<F>,
    MC: DirectMMCS<F>,
{
    queries: Vec<QueryProof<F, EF, M, MC>>,
}

pub struct QueryProof<F, EF, M, MC>
where
    F: Field,
    EF: AbstractFieldExtension<F>,
    M: MMCS<F>,
    MC: DirectMMCS<F>,
{
    /// An opened row of each matrix that was part of this batch-FRI proof.
    leaves: Vec<Vec<F>>,
    leaf_opening_proofs: Vec<M::Proof>,
    steps: Vec<QueryStepProof<F, EF, MC>>,
}

pub struct QueryStepProof<F, EF, MC>
where
    F: Field,
    EF: AbstractFieldExtension<F>,
    MC: DirectMMCS<F>,
{
    /// An opened row of each matrix that was part of this batch-FRI proof.
    leaves: EF,
    leaf_opening_proofs: MC::Proof,
}
