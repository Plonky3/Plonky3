use crate::FriProof;
use p3_challenger::Challenger;
use p3_commit::{DirectMMCS, MMCS};
use p3_field::{AbstractFieldExtension, Field};

pub(crate) fn verify<F, EF, M, MC, Chal>(
    _proof: &FriProof<F, EF, M, MC>,
    _challenger: &mut Chal,
) -> Result<(), ()>
where
    F: Field,
    EF: AbstractFieldExtension<F>,
    M: MMCS<F>,
    MC: DirectMMCS<F>,
    Chal: Challenger<F>,
{
    todo!()
}
