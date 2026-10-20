//! The Flock prover: commit, zerocheck, lincheck, open.

use tracing::instrument;

use crate::config::{FlockChallenger, FlockConfig};
use crate::error::ProvingError;
use crate::opening::WitnessPcs;
use crate::proof::FlockProof;
use crate::{Batch, Witness, lincheck, zerocheck};

/// Prove that a witness satisfies every instance of a batch.
///
/// # Errors
///
/// - The parameters admit no commitment schedule, or the witness is not the batch's.
#[instrument(skip_all, name = "flock prove", fields(log_n = batch.log_n()))]
pub fn prove(
    config: &FlockConfig,
    batch: &Batch<'_>,
    witness: &Witness,
    challenger: &mut FlockChallenger,
) -> Result<FlockProof, ProvingError> {
    // Commit first, so every challenge depends on the witness.
    let pcs = WitnessPcs::new(batch.log_n(), &config.pcs)?;
    let (commitment, data) = pcs.commit(witness, challenger)?;

    // a * b = c everywhere, reduced to claims at one point.
    let (zerocheck, claims) = zerocheck::prove(batch, witness, config.pow_bits, challenger);

    // Those claims, reduced to one claim on the committed bits.
    let (lincheck, claim) = lincheck::prove(batch, witness, &claims, challenger);

    // That claim, opened.
    let opening = pcs.open(data, &[claim], challenger)?;
    Ok(FlockProof {
        commitment,
        zerocheck,
        lincheck,
        opening,
    })
}
