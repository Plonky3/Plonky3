//! The witness commitment: packed bits under WHIR, opened at quirky claims through the ring switch.
//!
//! A quirky claim weighs the six absorbed bits by the skip domain's Lagrange basis instead of an equality table, as in DP ring switching.

#[cfg(test)]
mod tests;

use alloc::vec::Vec;

use p3_binary_pcs::whir::preset::{hash64, keccak64};
use p3_binary_pcs::whir::{CommitmentError, SuffixTensors, WeightedClaim};
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::{ExtensionMmcs, Mmcs, MmcsArith};
use p3_field::Algebra;
use p3_keccak::Keccak256Hash;

use crate::config::{FlockChallenger, FlockHash, PcsParams};
use crate::point::{QuirkyClaim, QuirkyPoint};
use crate::{Challenge, FlockArith, Val};

/// Merkle commitment to committed elements: leaves and nodes hashed by `H`, arity two.
pub type FlockMmcs<H = Keccak256Hash> = hash64::Mmcs<H>;

/// A witness commitment, the same type for every hash.
pub type Commitment = keccak64::Commitment;

/// An opening of several Merkle rows.
type MultiProof<H> = <FlockMmcs<H> as Mmcs<Val>>::MultiProof;

/// What the prover keeps between committing and opening.
pub type ProverData<H = Keccak256Hash> = hash64::ProverData<H>;

/// One opening proof for every claim.
pub type OpeningProof<H = Keccak256Hash> = hash64::OpeningProof<H>;

/// Why committing or opening fails.
pub type OpeningError = CommitmentError;

/// The bit commitment of a batch's witness: 64 bits per element, WHIR over the additive domain.
pub struct WitnessPcs<H: FlockHash = Keccak256Hash>(
    /// The shared bit commitment, opened at quirky claims.
    hash64::WitnessCommitment<H>,
);

impl<H: FlockHash> WitnessPcs<H> {
    /// The bit commitment for a witness of `2^num_variables` bits.
    ///
    /// The bound is the one opening of one claim, which is what a Flock proof opens.
    ///
    /// # Errors
    ///
    /// The witness is narrower than one element, the parameters admit no schedule, or no target reaches the bound.
    pub fn new(num_variables: usize, params: &PcsParams) -> Result<Self, OpeningError> {
        Self::with_claims(num_variables, params, 1)
    }

    /// The bit commitment for a witness of `2^num_variables` bits, one opening batching up to `claims` claims.
    ///
    /// # Errors
    ///
    /// The witness is narrower than one element, the parameters admit no schedule, or no target reaches the bound.
    pub fn with_claims(
        num_variables: usize,
        params: &PcsParams,
        claims: usize,
    ) -> Result<Self, OpeningError> {
        hash64::witness_commitment(num_variables, params, claims).map(Self)
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
        challenger: &mut FlockChallenger<H>,
    ) -> Result<(Commitment, ProverData<H>), OpeningError> {
        self.0.commit(z, challenger)
    }

    /// The witness words a commitment holds.
    #[must_use]
    pub fn committed_words(data: &ProverData<H>) -> &[u64] {
        hash64::WitnessCommitment::<H>::committed_words(data)
    }

    /// Observe a commitment on the verifier side, as committing does on the prover side.
    pub fn observe(&self, commitment: &Commitment, challenger: &mut FlockChallenger<H>) {
        self.0.observe(commitment, challenger);
    }

    /// As [`Self::observe`], within an arithmetic, `target` being the commitment there.
    pub fn observe_in<A, Ch>(
        &self,
        arith: &mut A,
        commitment: &Commitment,
        target: &<FlockMmcs<H> as MmcsArith<A, A::F>>::Target,
        challenger: &mut Ch,
    ) where
        A: FlockArith,
        Ch: CanObserve<Val> + CanObserve<Commitment>,
        FlockMmcs<H>: MmcsArith<A, A::F>,
    {
        self.0.observe_in(arith, commitment, target, challenger);
    }

    /// The Merkle scheme every round commits with.
    #[must_use]
    pub const fn mmcs(&self) -> &FlockMmcs<H> {
        self.0.mmcs()
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
        data: ProverData<H>,
        claims: &[QuirkyClaim],
        splits: &[Option<&SuffixTensors<Challenge, Val>>],
        challenger: &mut FlockChallenger<H>,
    ) -> Result<OpeningProof<H>, OpeningError> {
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
        proof: &OpeningProof<H>,
        challenger: &mut FlockChallenger<H>,
    ) -> Result<(), OpeningError> {
        self.0
            .verify(commitment, &weighted(claims), proof, challenger)
    }

    /// As [`Self::verify`], within an arithmetic.
    ///
    /// # Errors
    ///
    /// As [`Self::verify`].
    pub fn verify_in<'s, A, Ch>(
        &'s self,
        arith: &mut A,
        commitment: &<FlockMmcs<H> as MmcsArith<A, A::F>>::Target,
        claims: &[QuirkyClaim<A::EF>],
        proof: &OpeningProof<H>,
        challenger: &mut Ch,
    ) -> Result<(), OpeningError>
    where
        A: FlockArith,
        Ch: FieldChallenger<Val>
            + GrindingChallenger<Witness = Val>
            + CanSampleUniformBits<Val>
            + CanObserve<Commitment>,
        FlockMmcs<H>: MmcsArith<A, A::F, Commitment = Commitment, MultiProof = MultiProof<H>>,
        ExtensionMmcs<Val, Challenge, &'s FlockMmcs<H>>: MmcsArith<
                A,
                A::EF,
                MultiProof = MultiProof<H>,
                Target = <FlockMmcs<H> as MmcsArith<A, A::F>>::Target,
            >,
    {
        self.0
            .verify_in(arith, commitment, &weighted(claims), proof, challenger)
    }
}

/// Quirky claims as weighted claims.
fn weighted<E: Algebra<Challenge>>(claims: &[QuirkyClaim<E>]) -> Vec<WeightedClaim<E>> {
    claims
        .iter()
        .map(|claim| claim.clone().into_weighted())
        .collect()
}
