//! The Flock verifier: bind the statement and the commitment, replay the zerocheck and the lincheck, check the opening.

use tracing::instrument;

use crate::config::{FlockChallenger, FlockConfig};
use crate::error::VerificationError;
use crate::opening::WitnessPcs;
use crate::ports::PublicPorts;
use crate::proof::FlockProof;
use crate::{Batch, lincheck, zerocheck};

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
    let pcs = WitnessPcs::new(batch.log_n(), &config.pcs)?;
    pcs.observe(&proof.commitment, challenger);

    // Replay the reductions down to one witness claim, then check its opening.
    let claims = zerocheck::verify(batch, &proof.zerocheck, config.pow_bits, challenger)?;
    let claim = lincheck::verify(batch, &claims, public, &proof.lincheck, challenger)?;
    pcs.verify(&proof.commitment, &[claim], &proof.opening, challenger)?;
    Ok(())
}
