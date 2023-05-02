use crate::FriProof;
use p3_commit::mmcs::MMCS;
use p3_field::field::Field;

pub(crate) fn prove<F, O>(_codewords: &[O::ProverData]) -> FriProof<F, O>
where
    F: Field,
    O: MMCS<F>,
{
    todo!()
}
