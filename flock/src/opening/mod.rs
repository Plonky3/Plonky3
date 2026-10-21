//! The witness commitment: packed bits under WHIR, opened at quirky claims through the ring switch.
//!
//! A quirky claim weighs the six absorbed bits by the skip domain's Lagrange basis instead of an equality table, as in DP ring switching.

#[cfg(test)]
mod tests;

use alloc::vec::Vec;

use p3_binary_pcs::whir::preset::keccak64::{self, WitnessCommitment};
use p3_binary_pcs::whir::{CommitmentError, SuffixTensors, WeightedClaim, WeightedOpening, preset};
use p3_multilinear_util::point::Point;

use crate::config::{FlockChallenger, PcsParams};
use crate::point::{QuirkyClaim, QuirkyPoint};
use crate::skip::{SKIP_SIZE, lagrange_weights};
use crate::{Challenge, Val};

/// Merkle commitment to committed elements.
pub type FlockMmcs = keccak64::Mmcs;

/// A witness commitment.
pub type Commitment = keccak64::Commitment;

/// What the prover keeps between committing and opening.
pub type ProverData = keccak64::ProverData;

/// One opening proof for every claim.
pub type OpeningProof = keccak64::OpeningProof;

/// Why committing or opening fails.
pub type OpeningError = CommitmentError;

/// The bit commitment of a batch's witness: 64 bits per element, WHIR over the additive domain.
pub struct WitnessPcs(
    /// The shared bit commitment, opened at quirky claims.
    WitnessCommitment,
);

impl WitnessPcs {
    /// The bit commitment for a witness of `2^num_variables` bits.
    ///
    /// The bound is the one opening of one claim, which is what a Flock proof opens.
    ///
    /// # Errors
    ///
    /// The witness is narrower than one element, the parameters admit no schedule, or no target reaches the bound.
    pub fn new(num_variables: usize, params: &PcsParams) -> Result<Self, OpeningError> {
        preset::keccak64::witness_commitment(num_variables, params, 1).map(Self)
    }

    /// Commit to a witness's packed words, taking them over, and observe the commitment.
    ///
    /// [`Self::committed_words`] reads them back from the prover data.
    ///
    /// # Errors
    ///
    /// The words do not have the committed variables.
    pub fn commit(
        &self,
        z: Vec<u64>,
        challenger: &mut FlockChallenger,
    ) -> Result<(Commitment, ProverData), OpeningError> {
        self.0.commit(z, challenger)
    }

    /// The witness words a commitment holds.
    #[must_use]
    pub fn committed_words(data: &ProverData) -> &[u64] {
        WitnessCommitment::committed_words(data)
    }

    /// Observe a commitment on the verifier side, as committing does on the prover side.
    pub fn observe(&self, commitment: &Commitment, challenger: &mut FlockChallenger) {
        self.0.observe(commitment, challenger);
    }

    /// A claim's element split over the trailing coordinates the commitment's first fold binds.
    ///
    /// Built from the witness folded over the instances, as the lincheck holds it before its rounds.
    ///
    /// # Returns
    ///
    /// Nothing when the inner coordinates past the skip are fewer than the first fold binds.
    #[must_use]
    pub fn suffix_tensors(
        &self,
        folded: &[Challenge],
        point: &QuirkyPoint,
        log_instances: usize,
    ) -> Option<SuffixTensors<Challenge, Val>> {
        self.0.suffix_tensors(folded, &point.rest, log_instances)
    }

    /// Open the committed witness at every claim, in one proof.
    ///
    /// Entry `i` of `splits`, when present, is claim `i`'s element split by [`Self::suffix_tensors`].
    ///
    /// # Errors
    ///
    /// A claim of the wrong width, a claim the witness does not satisfy, or a refused opening.
    pub fn open(
        &self,
        data: ProverData,
        claims: &[QuirkyClaim],
        splits: &[Option<&SuffixTensors<Challenge, Val>>],
        challenger: &mut FlockChallenger,
    ) -> Result<OpeningProof, OpeningError> {
        self.0.open(data, &weighted(claims), splits, challenger)
    }

    /// Check one opening proof against every claim.
    ///
    /// # Errors
    ///
    /// A claim of the wrong width, or any refusal of the ring switch or WHIR.
    pub fn verify(
        &self,
        commitment: &Commitment,
        claims: &[QuirkyClaim],
        proof: &OpeningProof,
        challenger: &mut FlockChallenger,
    ) -> Result<(), OpeningError> {
        self.0
            .verify(commitment, &weighted(claims), proof, challenger)
    }
}

/// Quirky claims as weighted claims: Lagrange weights at the skip coordinate on the absorbed bits.
fn weighted(claims: &[QuirkyClaim]) -> Vec<WeightedClaim<Challenge>> {
    claims
        .iter()
        .map(|claim| WeightedClaim {
            opening: WeightedOpening {
                high: Point::new(claim.point.rest.clone()),
                low_weights: lagrange_weights(claim.point.skip, SKIP_SIZE),
            },
            value: claim.value,
        })
        .collect()
}
