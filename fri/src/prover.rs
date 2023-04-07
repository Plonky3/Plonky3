use crate::FriProof;
use p3_commit::oracle::Oracle;
use p3_field::field::FieldExtension;

pub(crate) fn prove<FE, O>(_codewords: &[O::Commitment]) -> FriProof<FE, O>
where
    FE: FieldExtension,
    O: Oracle<FE::Base>,
{
    todo!()
}
