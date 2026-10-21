//! The Flock verifier: bind the statement and the commitment, replay the PIOP, check the opening.

use p3_binary_pcs::whir::preset::keccak64;
use tracing::instrument;

use crate::Batch;
use crate::config::{FlockChallenger, FlockConfig};
use crate::error::VerificationError;
use crate::piop::{self, Placement};
use crate::ports::PublicPorts;
use crate::proof::FlockProof;

/// Verify that a proof's committed witness satisfies every instance of a batch and holds its public port words.
///
/// Without public words the statement is existential: some witness satisfies every instance.
///
/// # Errors
///
/// Returns the first phase that rejects.
#[instrument(skip_all, name = "flock verify", fields(log_n = batch.log_n()))]
pub fn verify(
    config: &FlockConfig,
    batch: &Batch<'_>,
    public: &PublicPorts,
    proof: &FlockProof,
    challenger: &mut FlockChallenger,
) -> Result<(), VerificationError> {
    // Bind the statement, then the commitment, where the prover did.
    public.bind(batch, challenger)?;
    let pcs = keccak64::witness_commitment(batch.log_n(), &config.pcs, 1)?;
    pcs.observe(&proof.commitment, challenger);

    // Replay the reductions down to one witness claim, then check its opening.
    let claim = piop::verify(
        config.pow_bits,
        batch,
        public,
        &proof.piop,
        &Placement::whole(&pcs),
        challenger,
    )?;
    pcs.verify(&proof.commitment, &[claim], &proof.opening, challenger)?;
    Ok(())
}
