use p3_challenger::Challenger;
use p3_commit::{DirectMMCS, MMCS};
use p3_field::{ExtensionField, Field};

use crate::FriProof;

pub(crate) fn verify<F, Challenge, M, MC, Chal>(
    _proof: &FriProof<F, Challenge, M, MC>,
    _challenger: &mut Chal,
) -> Result<(), ()>
where
    F: Field,
    Challenge: ExtensionField<F>,
    M: MMCS<F>,
    MC: DirectMMCS<Challenge>,
    Chal: Challenger<F>,
{
    todo!()
}
