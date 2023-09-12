use crate::{FriConfig, FriProof};

pub(crate) fn verify<FC: FriConfig>(
    _proof: &FriProof<FC>,
    _challenger: &mut FC::Challenger,
) -> Result<(), ()> {
    Ok(()) // TODO
}
