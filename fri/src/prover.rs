use crate::FriProof;
use p3_commit::mmcs::MMCS;
use p3_field::field::FieldExtension;

pub(crate) fn prove<FE, O>(_codewords: &[O::ProverData]) -> FriProof<FE, O>
where
    FE: FieldExtension,
    O: MMCS<FE::Base>,
{
    todo!()
}
