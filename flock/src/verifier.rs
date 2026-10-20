//! The Flock verifier: bind the commitment, replay the zerocheck and the lincheck, check the opening.

use tracing::instrument;

use crate::config::{FlockChallenger, FlockConfig};
use crate::error::VerificationError;
use crate::opening::WitnessPcs;
use crate::proof::FlockProof;
use crate::{Batch, lincheck, zerocheck};

/// Verify that a proof's committed witness satisfies every instance of a batch.
///
/// No input or output is checked: the statement is the circuit and the batch alone.
///
/// # Errors
///
/// Returns the first phase that rejects.
#[instrument(skip_all, name = "flock verify", fields(log_n = batch.log_n()))]
pub fn verify(
    config: &FlockConfig,
    batch: &Batch<'_>,
    proof: &FlockProof,
    challenger: &mut FlockChallenger,
) -> Result<(), VerificationError> {
    // Bind the commitment where the prover did.
    let pcs = WitnessPcs::new(batch.log_n(), &config.pcs)?;
    pcs.observe(&proof.commitment, challenger);

    // Replay the reductions down to one witness claim, then check its opening.
    let claims = zerocheck::verify(batch, &proof.zerocheck, config.pow_bits, challenger)?;
    let claim = lincheck::verify(batch, &claims, &proof.lincheck, challenger)?;
    pcs.verify(&proof.commitment, &[claim], &proof.opening, challenger)?;
    Ok(())
}
