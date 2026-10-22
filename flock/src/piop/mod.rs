//! The Flock PIOP on one region of a shared bit commitment: zerocheck and lincheck down to one claim the caller opens.
//!
//! The caller binds the statement before committing and opens the claim after, batched with any other claim on the commitment.

#[cfg(test)]
mod tests;

use p3_binary_pcs::whir::preset::hash64::{ProverData, WitnessCommitment};
use p3_binary_pcs::whir::{Region, SuffixTensors, WeightedClaim};
use p3_challenger::{CanObserve, CanSample, GrindingChallenger};
use p3_field::Algebra;
use p3_keccak::Keccak256Hash;
use p3_verifier_arith::Native;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::instrument;

use crate::config::{FlockChallenger, FlockHash};
use crate::error::{ProvingError, VerificationError};
use crate::lincheck::LincheckProof;
use crate::point::QuirkyClaim;
use crate::ports::PublicPorts;
use crate::zerocheck::ZerocheckProof;
use crate::{Batch, Challenge, FlockArith, Val, Witness, lincheck, zerocheck};

/// The reductions of a Flock proof, from the circuit to one claim on the committed witness.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PiopProof {
    /// Zerocheck of `A z o B z = z`.
    pub zerocheck: ZerocheckProof,
    /// Reduction of the zerocheck claims and the public words to one witness claim.
    pub lincheck: LincheckProof,
}

/// Where a batch's witness lies: a region of a shared bit commitment.
pub struct Placement<'a, H: FlockHash = Keccak256Hash> {
    /// The shared commitment.
    pub commitment: &'a WitnessCommitment<H>,
    /// The region holding the witness bits.
    pub region: Region,
}

/// The claim a prover leaves on its region, with its element split for the deferred ring switch.
#[must_use = "the claim is sound only once opened against the commitment"]
#[derive(Clone, Debug)]
pub struct RegionClaim {
    /// The claim on the whole commitment.
    pub claim: WeightedClaim<Challenge>,
    /// The claim's element, split from the folded witness rather than from the commitment.
    pub split: Option<SuffixTensors<Challenge, Val>>,
}

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

impl<H: FlockHash> Clone for Placement<'_, H> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<H: FlockHash> Copy for Placement<'_, H> {}

impl<'a, H: FlockHash> Placement<'a, H> {
    /// The whole commitment as the region.
    #[must_use]
    pub const fn whole(commitment: &'a WitnessCommitment<H>) -> Self {
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
    fn lift<E: Algebra<Challenge>>(&self, claim: QuirkyClaim<E>) -> WeightedClaim<E> {
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
pub fn prove<H: FlockHash>(
    pow_bits: usize,
    batch: &Batch<'_>,
    public: &PublicPorts,
    witness: &Witness,
    placement: &Placement<'_, H>,
    data: &ProverData<H>,
    challenger: &mut FlockChallenger<H>,
) -> Result<(PiopProof, RegionClaim), ProvingError> {
    placement.check(batch)?;
    let z = placement
        .region
        .words(WitnessCommitment::<H>::committed_words(data));
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
pub fn verify<H: FlockHash>(
    pow_bits: usize,
    batch: &Batch<'_>,
    public: &PublicPorts,
    proof: &PiopProof,
    placement: &Placement<'_, H>,
    challenger: &mut FlockChallenger<H>,
) -> Result<WeightedClaim<Challenge>, VerificationError> {
    verify_in(
        &mut Native::new(),
        pow_bits,
        batch,
        public,
        proof,
        placement,
        challenger,
    )
}

/// As [`verify`], within an arithmetic.
///
/// # Errors
///
/// As [`verify`].
#[instrument(skip_all, name = "flock piop verify", fields(log_n = batch.log_n()))]
pub fn verify_in<A, C, H: FlockHash>(
    arith: &mut A,
    pow_bits: usize,
    batch: &Batch<'_>,
    public: &PublicPorts,
    proof: &PiopProof,
    placement: &Placement<'_, H>,
    challenger: &mut C,
) -> Result<WeightedClaim<A::EF>, VerificationError>
where
    A: FlockArith,
    C: CanObserve<Val> + CanSample<Val> + GrindingChallenger<Witness = Val>,
{
    placement.check(batch)?;
    let claims = zerocheck::verify_in(arith, batch, &proof.zerocheck, pow_bits, challenger)?;
    let claim = lincheck::verify_in(arith, batch, &claims, public, &proof.lincheck, challenger)?;
    Ok(placement.lift(claim))
}
