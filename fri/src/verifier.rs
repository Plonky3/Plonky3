use crate::FriProof;
use p3_challenger::Challenger;
use p3_commit::{DirectMMCS, MMCS};
use p3_field::Field;

pub(crate) fn verify<F, M, MC, Chal>(
    _proof: &FriProof<F, M, MC>,
    _challenger: &mut Chal,
) -> Result<(), ()>
where
    F: Field,
    M: MMCS<F::Base>,
    MC: DirectMMCS<F::Base>,
    Chal: Challenger<F::Base>,
{
    todo!()
}
