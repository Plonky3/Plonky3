//! The Flock PIOP on one region of a shared bit commitment: zerocheck and lincheck down to one claim the caller opens.
//!
//! The caller binds the statement before committing and opens the claim after, batched with any other claim on the commitment.

#[cfg(test)]
mod tests;

use p3_binary_pcs::whir::preset::keccak64::{ProverData, WitnessCommitment};
use p3_binary_pcs::whir::{Region, WeightedClaim};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::instrument;

use crate::config::FlockChallenger;
use crate::error::{ProvingError, VerificationError};
use crate::lincheck::LincheckProof;
use crate::point::QuirkyClaim;
use crate::ports::PublicPorts;
use crate::zerocheck::ZerocheckProof;
use crate::{Batch, Challenge, Val, Witness, lincheck, zerocheck};

/// The reductions of a Flock proof, from the circuit to one claim on the committed witness.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PiopProof {
    /// Zerocheck of `A z o B z = z`.
    pub zerocheck: ZerocheckProof,
    /// Reduction of the zerocheck claims and the public words to one witness claim.
    pub lincheck: LincheckProof,
}

/// Where a batch's witness lies: a region of a shared bit commitment.
#[derive(Clone, Copy)]
pub struct Placement<'a> {
    /// The shared commitment.
    pub commitment: &'a WitnessCommitment,
    /// The region holding the witness bits.
    pub region: Region,
}

/// The claim a prover leaves on its region, with its element split for the deferred ring switch.
pub type RegionClaim = p3_binary_pcs::whir::RegionClaim<Val, Challenge>;

/// Why a placement cannot hold a batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
#[error("{region:?} is not an aligned run of 2^{log_n} bits inside 2^{num_variables}")]
pub struct RegionError {
    /// The region asked for.
    pub region: Region,
    /// The batch's witness variables.
    pub log_n: usize,
    /// The commitment's variables.
    pub num_variables: usize,
}

impl<'a> Placement<'a> {
    /// The whole commitment as the region.
    #[must_use]
    pub const fn whole(commitment: &'a WitnessCommitment) -> Self {
        Self {
            commitment,
            region: Region {
                log_bits: commitment.num_variables(),
                offset_bits: 0,
            },
        }
    }

    /// Check that the region holds exactly the batch's bits, aligned inside the commitment.
    ///
    /// # Errors
    ///
    /// The region has another size, is misaligned, or lies outside the commitment.
    pub const fn check(&self, batch: &Batch<'_>) -> Result<(), RegionError> {
        let num_variables = self.commitment.num_variables();
        if self.region.log_bits == batch.log_n() && self.region.lies_within(num_variables) {
            Ok(())
        } else {
            Err(RegionError {
                region: self.region,
                log_n: batch.log_n(),
                num_variables,
            })
        }
    }

    /// A claim on the region's bits as the same claim on the whole commitment.
    fn lift(&self, claim: QuirkyClaim) -> WeightedClaim<Challenge> {
        let selector = self.region.selector(self.commitment.num_variables());
        let mut claim = claim.into_weighted();
        claim.opening = claim.opening.within(&selector);
        claim
    }
}

/// Prove that the witness in a placement satisfies every instance of a batch and holds its public port words.
///
/// The statement is bound and the commitment observed before; the returned claim is opened after.
///
/// # Errors
///
/// - The placement does not hold the batch.
/// - The public words do not fit the batch, or the committed words disagree with them.
///
/// # Panics
///
/// - The prover data is not the placement commitment's, or the witness's `A z`, `B z` are not the batch's.
#[instrument(skip_all, name = "flock piop prove", fields(log_n = batch.log_n()))]
pub fn prove(
    pow_bits: usize,
    batch: &Batch<'_>,
    public: &PublicPorts,
    witness: &Witness,
    placement: &Placement<'_>,
    data: &ProverData,
    challenger: &mut FlockChallenger,
) -> Result<(PiopProof, RegionClaim), ProvingError> {
    placement.check(batch)?;
    let z = placement
        .region
        .words(WitnessCommitment::committed_words(data));
    public.check_words(batch, z)?;

    // a * b = c everywhere, reduced to claims at one point.
    let (zerocheck, claims) = zerocheck::prove(batch, witness, pow_bits, challenger);

    // Those claims and the public words, reduced to one claim on the region's bits.
    let (lincheck, claim, folded) = lincheck::prove_folded(batch, z, &claims, public, challenger);

    // The selector's eq is one on the region, so the fold over the instances is the fold over the commitment.
    let split =
        placement
            .commitment
            .suffix_tensors(&folded, &claim.point.rest, batch.log_instances());
    let claim = placement.lift(claim);
    Ok((
        PiopProof {
            zerocheck,
            lincheck,
        },
        RegionClaim { claim, split },
    ))
}

/// Replay a batch's reductions on a placement, returning the claim the caller must still open.
///
/// The statement is bound and the commitment observed before, where the prover did.
///
/// # Errors
///
/// The placement does not hold the batch, or the zerocheck, the lincheck or the public words reject.
#[instrument(skip_all, name = "flock piop verify", fields(log_n = batch.log_n()))]
pub fn verify(
    pow_bits: usize,
    batch: &Batch<'_>,
    public: &PublicPorts,
    proof: &PiopProof,
    placement: &Placement<'_>,
    challenger: &mut FlockChallenger,
) -> Result<WeightedClaim<Challenge>, VerificationError> {
    placement.check(batch)?;
    let claims = zerocheck::verify(batch, &proof.zerocheck, pow_bits, challenger)?;
    let claim = lincheck::verify(batch, &claims, public, &proof.lincheck, challenger)?;
    Ok(placement.lift(claim))
}
