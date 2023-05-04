use crate::FriProof;
use p3_challenger::Challenger;
use p3_commit::mmcs::{DirectMMCS, MMCS};
use p3_field::field::{Field, FieldExtension};

pub(crate) fn verify<F, FE, M, MC, Chal>(
    _proof: &FriProof<F, FE, M, MC>,
    _challenger: &mut Chal,
) -> Result<(), ()>
where
    F: Field,
    FE: FieldExtension<F>,
    M: MMCS<F>,
    MC: DirectMMCS<F>,
    Chal: Challenger<F>,
{
    todo!()
}
