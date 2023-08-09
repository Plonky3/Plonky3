use p3_challenger::Challenger;

use crate::{FriConfig, FriProof};

pub(crate) fn verify<FC: FriConfig, Chal: Challenger<FC::Val>>(
    _proof: &FriProof<FC>,
    _challenger: &mut Chal,
) -> Result<(), ()> {
    todo!()
}
