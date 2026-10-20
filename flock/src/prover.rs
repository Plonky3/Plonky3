//! The Flock prover: bind the statement, commit, zerocheck, lincheck, open.

use tracing::instrument;

use crate::config::{FlockChallenger, FlockConfig};
use crate::error::ProvingError;
use crate::opening::WitnessPcs;
use crate::ports::PublicPorts;
use crate::proof::FlockProof;
use crate::{Batch, Witness, lincheck, zerocheck};

/// Prove that a witness satisfies every instance of a batch and holds its public port words.
///
/// # Errors
///
/// - The public words do not fit the batch, or the witness disagrees with them.
/// - The parameters admit no commitment schedule, or the witness is not the batch's.
#[instrument(skip_all, name = "flock prove", fields(log_n = batch.log_n()))]
pub fn prove(
    config: &FlockConfig,
    batch: &Batch<'_>,
    public: &PublicPorts,
    witness: &Witness,
    challenger: &mut FlockChallenger,
) -> Result<FlockProof, ProvingError> {
    // The statement first, so every challenge depends on it.
    public.check_witness(batch, witness)?;
    public.bind(batch, challenger)?;

    // Then the commitment, so every challenge depends on the witness.
    let pcs = WitnessPcs::new(batch.log_n(), &config.pcs)?;
    let (commitment, data) = pcs.commit(witness, challenger)?;

    // a * b = c everywhere, reduced to claims at one point.
    let (zerocheck, claims) = zerocheck::prove(batch, witness, config.pow_bits, challenger);

    // Those claims and the public words, reduced to one claim on the committed bits.
    let (lincheck, claim, folded) =
        lincheck::prove_folded(batch, witness, &claims, public, challenger);

    // That claim, opened, its element split from the folded witness rather than from the commitment.
    let split = pcs.suffix_tensors(&folded, &claim.point, batch.log_instances());
    let opening = pcs.open(data, &[claim], &[split.as_ref()], challenger)?;
    Ok(FlockProof {
        commitment,
        zerocheck,
        lincheck,
        opening,
    })
}
