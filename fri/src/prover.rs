use crate::FriProof;
use p3_commit::mmcs::MMCS;
use p3_field::field::{Field, FieldExtension};

pub(crate) fn prove<F, FE, O>(_codewords: &[O::ProverData]) -> FriProof<F, FE, O>
where
    F: Field,
    FE: FieldExtension<F>,
    O: MMCS<F>,
{
    todo!()
}
