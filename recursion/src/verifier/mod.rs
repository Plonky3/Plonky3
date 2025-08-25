use p3_uni_stark::{PcsError, StarkGenericConfig, VerificationError, verify as base_verify};

use crate::prover::RecursiveProof;

pub fn verify<SC: StarkGenericConfig>(
    config: &SC,
    proof: RecursiveProof<SC>,
) -> Result<(), VerificationError<PcsError<SC>>> {
    base_verify(config, &proof.add_air, &proof.add_proof, &vec![])?;
    base_verify(config, &proof.sub_air, &proof.sub_proof, &vec![])
}
