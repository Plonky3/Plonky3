//! The Flock prover: bind the statement, commit, run the PIOP on the whole commitment, open.

use p3_binary_pcs::whir::preset::hash64;
use tracing::instrument;

use crate::config::{FlockChallenger, FlockConfig, FlockHash};
use crate::error::ProvingError;
use crate::piop::{self, Placement};
use crate::ports::PublicPorts;
use crate::proof::FlockProof;
use crate::{Batch, Witness};

/// Prove that a witness satisfies every instance of a batch and holds its public port words.
///
/// # Errors
///
/// - The public words do not fit the batch, or the witness disagrees with them.
/// - The parameters admit no commitment schedule, or the witness is not the batch's.
#[instrument(skip_all, name = "flock prove", fields(log_n = batch.log_n()))]
pub fn prove<H: FlockHash>(
    config: &FlockConfig,
    batch: &Batch<'_>,
    public: &PublicPorts,
    mut witness: Witness,
    challenger: &mut FlockChallenger<H>,
) -> Result<FlockProof<H>, ProvingError> {
    // The statement first, so every challenge depends on it.
    public.bind(batch, challenger)?;

    // Then the commitment, so every challenge depends on the witness.
    let pcs = hash64::witness_commitment::<H>(batch.log_n(), &config.pcs, 1)?;
    // The words move into the commitment, which hands them back to the PIOP.
    let (commitment, data) = pcs.commit(witness.take_z(), challenger)?;

    // The circuit and the public words, reduced to one claim on the committed bits.
    let (piop, claim) = piop::prove(
        config.pow_bits,
        batch,
        public,
        &witness,
        &Placement::whole(&pcs),
        &data,
        challenger,
    )?;

    // That claim, opened.
    let opening = pcs.open(data, &[claim.claim], &[claim.split.as_ref()], challenger)?;
    Ok(FlockProof {
        commitment,
        piop,
        opening,
    })
}
