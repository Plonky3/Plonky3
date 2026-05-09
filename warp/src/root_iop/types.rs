//! Public data shapes for WARP root IOP transcripts.
//!
//! These are the typed objects that bridge WARP and the WHIR compiler. WARP
//! records oracle commitments and openings as deterministic claim ids; the
//! compiler later proves that all those claims are consistent with the committed
//! Reed-Solomon oracles.

use alloc::vec::Vec;

use p3_challenger::{CanObserve, FieldChallenger};
use p3_field::{Field, PrimeCharacteristicRing};
use serde::{Deserialize, Serialize};

use crate::protocol::{ExternalCommitmentObserver, ExternalCommittedCodeword};

/// Field carried by a root IOP oracle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RootIopOracleField {
    /// Base-field fresh WARP codeword.
    Base,
    /// Extension-field accumulator codeword.
    Extension,
}

/// Verifier-visible commitment placeholder for a WARP root IOP oracle.
///
/// This is intentionally small and deterministic. The final WHIR-style
/// compiler must bind this oracle id/shape to the actual oracle commitment it
/// proves. WARP itself only needs an object to absorb into the Fiat-Shamir
/// transcript in the correct order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RootIopCommitment {
    /// Monotone id assigned when the oracle is created.
    pub oracle_id: usize,
    /// `log_2` of the oracle length.
    pub log_len: usize,
    /// Field of the oracle entries.
    pub field: RootIopOracleField,
}

/// Verifier-visible commitment for the sound root IOP recorder.
///
/// Unlike [`RootIopCommitment`], this includes the actual commitment generated
/// by a Plonky3 commitment backend. The Fiat-Shamir transcript absorbs both
/// the WARP oracle metadata and this backend commitment before any challenge
/// that may depend on the oracle is sampled.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "Comm: Serialize + serde::de::DeserializeOwned")]
pub struct RootIopBoundCommitment<Comm> {
    /// Monotone id assigned when the oracle is created.
    pub oracle_id: usize,
    /// `log_2` of the oracle length.
    pub log_len: usize,
    /// Field of the oracle entries.
    pub field: RootIopOracleField,
    /// Real backend commitment binding the oracle values.
    pub commitment: Comm,
}

impl<Comm> RootIopBoundCommitment<Comm> {
    /// Absorb the WARP oracle metadata and real backend commitment.
    pub fn observe_into<F, Challenger>(&self, challenger: &mut Challenger)
    where
        F: Field + PrimeCharacteristicRing,
        Comm: Clone,
        Challenger: FieldChallenger<F> + CanObserve<Comm>,
    {
        challenger.observe(F::from_usize(self.oracle_id));
        challenger.observe(F::from_usize(self.log_len));
        challenger.observe(F::from_usize(match self.field {
            RootIopOracleField::Base => 0,
            RootIopOracleField::Extension => 1,
        }));
        challenger.observe(self.commitment.clone());
    }
}

impl RootIopCommitment {
    /// Absorb this virtual commitment into a field challenger.
    pub fn observe_into<F, Challenger>(&self, challenger: &mut Challenger)
    where
        F: Field + PrimeCharacteristicRing,
        Challenger: FieldChallenger<F>,
    {
        challenger.observe(F::from_usize(self.oracle_id));
        challenger.observe(F::from_usize(self.log_len));
        challenger.observe(F::from_usize(match self.field {
            RootIopOracleField::Base => 0,
            RootIopOracleField::Extension => 1,
        }));
    }
}

/// Point opened by the root IOP.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "EF: Serialize + serde::de::DeserializeOwned")]
pub enum RootIopOpeningPoint<EF> {
    /// Boolean-hypercube row index for generic linear-oracle claims.
    Index(usize),
    /// Reed-Solomon codeword index.
    ///
    /// This is the WARP/WHIR-native claim kind. It records an opening of the
    /// RS codeword `u = C(w)` at the smooth-domain position `omega_n^index`.
    /// The WHIR compiler may then authenticate it with the same RS code used
    /// by WHIR's initial oracle, rather than treating it as an unrelated
    /// multilinear-table row claim.
    RsCodewordIndex(usize),
    /// Multilinear extension point.
    Mle(Vec<EF>),
}

/// Value opened by the root IOP.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "F: Serialize + serde::de::DeserializeOwned,
                  EF: Serialize + serde::de::DeserializeOwned")]
pub enum RootIopOpeningValue<F, EF> {
    /// Base-field fresh value.
    Base(F),
    /// Extension-field accumulator value.
    Extension(EF),
}

/// One authenticated-opening claim that the final root proof must cover.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "F: Serialize + serde::de::DeserializeOwned,
                  EF: Serialize + serde::de::DeserializeOwned")]
pub struct RootIopOpeningClaim<F, EF> {
    /// Stable claim id used in the WARP proof body.
    pub claim_id: usize,
    /// Oracle being opened.
    pub oracle_id: usize,
    /// Opened point.
    pub point: RootIopOpeningPoint<EF>,
    /// Claimed value.
    pub value: RootIopOpeningValue<F, EF>,
}

/// Opening proof placeholder carried inside WARP proofs.
///
/// A proof system that compiles the whole root IOP should replace the list of
/// individual PCS proofs with these claim ids, then authenticate all recorded
/// ids with one final proof.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RootIopOpeningProof {
    /// Claim ids, in the same order as the opened indices/points.
    pub claim_ids: Vec<usize>,
}

/// Root proof shape when both WARP step openings and final decider openings
/// are recorded as root IOP claim ids.
pub type RootIopWarpRootProof<F, EF, FinalProof> = crate::root::WarpExternalRootProofBatched<
    F,
    EF,
    RootIopCommitment,
    RootIopCommitment,
    RootIopOpeningProof,
    RootIopOpeningProof,
    FinalProof,
>;

/// Finalizer proof shape produced by using [`super::RootIopProver`] as the
/// accumulator point-opening backend.
pub type RootIopWarpFinalizerProof<EF> =
    crate::finalize::WhirWarpFinalizerProof<EF, RootIopOpeningProof>;

/// Fresh codeword already registered in a root IOP transcript.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "F: Serialize + serde::de::DeserializeOwned")]
pub struct RootIopCommittedCodeword<F> {
    commitment: RootIopCommitment,
    codeword: Vec<F>,
    witness: Vec<F>,
}

/// Fresh codeword registered by the sound root IOP recorder.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "F: Serialize + serde::de::DeserializeOwned,
                  Comm: Serialize + serde::de::DeserializeOwned")]
pub struct RootIopBoundCommittedCodeword<F, Comm> {
    commitment: RootIopBoundCommitment<Comm>,
    codeword: Vec<F>,
    witness: Vec<F>,
}

impl<F, Comm> RootIopBoundCommittedCodeword<F, Comm> {
    /// Create a registered fresh codeword from explicit parts.
    pub const fn new(
        commitment: RootIopBoundCommitment<Comm>,
        codeword: Vec<F>,
        witness: Vec<F>,
    ) -> Self {
        Self {
            commitment,
            codeword,
            witness,
        }
    }
}

impl<F, Comm> ExternalCommittedCodeword<F> for RootIopBoundCommittedCodeword<F, Comm>
where
    F: Field + Serialize + serde::de::DeserializeOwned,
    Comm: Clone + Serialize + serde::de::DeserializeOwned,
{
    type Commitment = RootIopBoundCommitment<Comm>;

    fn commitment(&self) -> Self::Commitment {
        self.commitment.clone()
    }

    fn codeword(&self) -> &[F] {
        &self.codeword
    }

    fn witness(&self) -> &[F] {
        &self.witness
    }
}

impl<F, Comm, Challenger> ExternalCommitmentObserver<F, Challenger>
    for RootIopBoundCommittedCodeword<F, Comm>
where
    F: Field + PrimeCharacteristicRing + Serialize + serde::de::DeserializeOwned,
    Comm: Clone + Serialize + serde::de::DeserializeOwned,
    Challenger: FieldChallenger<F> + CanObserve<Comm>,
{
    fn observe_commitment(&self, challenger: &mut Challenger) {
        self.commitment.observe_into::<F, _>(challenger);
    }
}

impl<F> RootIopCommittedCodeword<F> {
    /// Create a registered fresh codeword from explicit parts.
    pub const fn new(commitment: RootIopCommitment, codeword: Vec<F>, witness: Vec<F>) -> Self {
        Self {
            commitment,
            codeword,
            witness,
        }
    }
}

impl<F> ExternalCommittedCodeword<F> for RootIopCommittedCodeword<F>
where
    F: Field + Serialize + serde::de::DeserializeOwned,
{
    type Commitment = RootIopCommitment;

    fn commitment(&self) -> Self::Commitment {
        self.commitment
    }

    fn codeword(&self) -> &[F] {
        &self.codeword
    }

    fn witness(&self) -> &[F] {
        &self.witness
    }
}

impl<F, Challenger> ExternalCommitmentObserver<F, Challenger> for RootIopCommittedCodeword<F>
where
    F: Field + PrimeCharacteristicRing + Serialize + serde::de::DeserializeOwned,
    Challenger: FieldChallenger<F>,
{
    fn observe_commitment(&self, challenger: &mut Challenger) {
        self.commitment.observe_into::<F, _>(challenger);
    }
}

/// Prover-side accumulator oracle data.
#[derive(Clone, Debug)]
pub struct RootIopAccumulatorProverData<EF> {
    pub(super) commitment: RootIopCommitment,
    pub(super) codeword: Vec<EF>,
}

/// Prover-side accumulator oracle data for the committed root IOP recorder.
#[derive(Clone, Debug)]
pub struct RootIopBoundAccumulatorProverData<EF, Comm> {
    pub(super) commitment: RootIopBoundCommitment<Comm>,
    pub(super) codeword: Vec<EF>,
}

/// Oracle payload recorded by the root IOP prover.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "F: Serialize + serde::de::DeserializeOwned,
                  EF: Serialize + serde::de::DeserializeOwned")]
pub enum RootIopOracleValues<F, EF> {
    /// Base-field codeword.
    Base(Vec<F>),
    /// Extension-field codeword.
    Extension(Vec<EF>),
}

/// Complete recorded root IOP transcript.
///
/// This is linear-size witness data. It is useful for tests and for the next
/// compiler stage, but it should not be treated as the final succinct proof.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "F: Serialize + serde::de::DeserializeOwned,
                  EF: Serialize + serde::de::DeserializeOwned")]
pub struct RootIopTranscript<F, EF> {
    /// All oracles registered by the prover.
    pub oracles: Vec<(RootIopCommitment, RootIopOracleValues<F, EF>)>,
    /// All opening claims produced by WARP and the final decider.
    pub claims: Vec<RootIopOpeningClaim<F, EF>>,
}

/// Complete recorded root IOP transcript with real backend commitments.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "F: Serialize + serde::de::DeserializeOwned,
                  EF: Serialize + serde::de::DeserializeOwned,
                  Comm: Serialize + serde::de::DeserializeOwned")]
pub struct RootIopBoundTranscript<F, EF, Comm> {
    /// All oracles registered by the prover.
    pub oracles: Vec<(RootIopBoundCommitment<Comm>, RootIopOracleValues<F, EF>)>,
    /// All opening claims produced by WARP and the final decider.
    pub claims: Vec<RootIopOpeningClaim<F, EF>>,
}

impl<F, EF, Comm> Default for RootIopBoundTranscript<F, EF, Comm> {
    fn default() -> Self {
        Self {
            oracles: Vec::new(),
            claims: Vec::new(),
        }
    }
}

impl<F, EF> Default for RootIopTranscript<F, EF> {
    fn default() -> Self {
        Self {
            oracles: Vec::new(),
            claims: Vec::new(),
        }
    }
}
