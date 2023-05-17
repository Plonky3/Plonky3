use crate::FriProof;
use p3_challenger::Challenger;
use p3_commit::{DirectMMCS, MMCS};
use p3_field::FieldExtension;

pub(crate) fn verify<FE, M, MC, Chal>(
    _proof: &FriProof<FE, M, MC>,
    _challenger: &mut Chal,
) -> Result<(), ()>
where
    FE: FieldExtension,
    M: MMCS<FE::Base>,
    MC: DirectMMCS<FE::Base>,
    Chal: Challenger<FE::Base>,
{
    todo!()
}
