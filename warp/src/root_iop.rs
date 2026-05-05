//! Root IOP opening transcript for a single WARP proof.
//!
//! WARP's root protocol is an IOR/IOP transcript: commitments are bound before
//! the verifier samples challenges, and later verifier queries open those
//! committed oracles. The slow WHIR-backed prototype used a complete WHIR PCS
//! proof for every such opening. This module isolates the reusable layer we
//! need instead: WARP records all openings as typed claims, so a single
//! compiler/proximity proof can authenticate those claims together.
//!
//! The recorder in this module is deliberately not the final succinct proof.
//! It is the boundary between the WARP root IOP and a WHIR-style compiler as
//! in WHIR Construction 7.4: first record the linear IOP transcript, then prove
//! the recorded oracle claims with one batched proximity layer.

use alloc::vec::Vec;
use core::cell::RefCell;
use core::fmt::Debug;

use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{ExtensionMmcs, Mmcs, MultilinearOpenedValues};
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use serde::{Deserialize, Serialize};

use crate::finalize::AccumulatorPointOpeningBackend;
use crate::protocol::{
    AccumulatorBatchOpeningBackend, AccumulatorCommitmentBackend,
    ExternalCodewordBatchOpeningProver, ExternalCodewordBatchOpeningVerifier,
    ExternalCodewordOpeningProver, ExternalCodewordOpeningVerifier, ExternalCommitmentObserver,
    ExternalCommittedCodeword,
};

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
    /// Boolean-hypercube row index.
    Index(usize),
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

/// Finalizer proof shape produced by using [`RootIopProver`] as the
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
    commitment: RootIopCommitment,
    codeword: Vec<EF>,
}

/// Prover-side accumulator oracle data for the committed root IOP recorder.
#[derive(Clone, Debug)]
pub struct RootIopBoundAccumulatorProverData<EF, Comm> {
    commitment: RootIopBoundCommitment<Comm>,
    codeword: Vec<EF>,
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

impl<F, EF, Comm> RootIopBoundTranscript<F, EF, Comm>
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// Check recorded claim values against the recorded oracle values.
    ///
    /// This is a witness check for tests and compiler inputs. Succinct
    /// verification must replace it with a WHIR-style proof.
    pub fn verify_witnessed_claim_values(&self) -> Result<(), RootIopError> {
        for claim in &self.claims {
            let (_, oracle) = self
                .oracles
                .iter()
                .find(|(commitment, _)| commitment.oracle_id == claim.oracle_id)
                .ok_or(RootIopError::UnknownOracle(claim.oracle_id))?;
            verify_claim_against_oracle::<F, EF>(claim, oracle)?;
        }
        Ok(())
    }
}

/// Proof-system interface for authenticating a recorded WARP root IOP.
///
/// The intended production implementation is a WHIR-style compiler that proves
/// all recorded claims with one batched proximity argument. The trait is kept
/// independent of WARP's step prover so different compilers can consume the
/// same transcript.
pub trait RootIopProofSystem<F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// Transmissible proof object.
    type Proof;
    /// Proving/verification error.
    type Error: Debug;

    /// Produce a proof for the recorded transcript.
    fn prove(transcript: &RootIopTranscript<F, EF>) -> Result<Self::Proof, Self::Error>;

    /// Verify that `proof` authenticates every expected WARP claim.
    fn verify(
        expected_claims: &[RootIopOpeningClaim<F, EF>],
        proof: &Self::Proof,
    ) -> Result<(), Self::Error>;
}

/// Proof-system interface for the commitment-bound root IOP.
///
/// This is the target interface for the single WHIR-style proof: replay WARP
/// with [`RootIopBoundVerifier`] to obtain `expected_commitments` and
/// `expected_claims`, then verify one proof against that whole set.
pub trait RootIopBoundProofSystem<F, EF, Comm>
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// Transmissible proof object.
    type Proof;
    /// Proving/verification error.
    type Error: Debug;

    /// Produce a proof for the recorded transcript.
    fn prove(transcript: &RootIopBoundTranscript<F, EF, Comm>) -> Result<Self::Proof, Self::Error>;

    /// Verify that `proof` authenticates the WARP-derived commitments and claims.
    fn verify(
        expected_commitments: &[RootIopBoundCommitment<Comm>],
        expected_claims: &[RootIopOpeningClaim<F, EF>],
        proof: &Self::Proof,
    ) -> Result<(), Self::Error>;
}

/// Linear-size root IOP proof used as a correctness baseline.
///
/// This is not the final succinct proof. It exists so the WARP root IOP can be
/// tested end to end while the WHIR compiler is implemented behind the same
/// [`RootIopProofSystem`] interface.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "F: Serialize + serde::de::DeserializeOwned,
                  EF: Serialize + serde::de::DeserializeOwned")]
pub struct WitnessRootIopProof<F, EF> {
    /// Full recorded transcript, including oracle values.
    pub transcript: RootIopTranscript<F, EF>,
}

/// Linear-size bound root IOP proof used as a correctness baseline.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "F: Serialize + serde::de::DeserializeOwned,
                  EF: Serialize + serde::de::DeserializeOwned,
                  Comm: Serialize + serde::de::DeserializeOwned")]
pub struct WitnessRootIopBoundProof<F, EF, Comm> {
    /// Full recorded transcript, including oracle values.
    pub transcript: RootIopBoundTranscript<F, EF, Comm>,
}

impl<F, EF> RootIopProofSystem<F, EF> for WitnessRootIopProof<F, EF>
where
    F: Field + PartialEq + Clone,
    EF: ExtensionField<F> + PartialEq + Clone,
{
    type Proof = Self;
    type Error = RootIopError;

    fn prove(transcript: &RootIopTranscript<F, EF>) -> Result<Self::Proof, Self::Error> {
        transcript.verify_witnessed_claim_values()?;
        Ok(Self {
            transcript: transcript.clone(),
        })
    }

    fn verify(
        expected_claims: &[RootIopOpeningClaim<F, EF>],
        proof: &Self::Proof,
    ) -> Result<(), Self::Error> {
        proof.transcript.verify_witnessed_claim_values()?;
        for expected in expected_claims {
            let actual = proof
                .transcript
                .claims
                .get(expected.claim_id)
                .ok_or(RootIopError::UnknownClaim(expected.claim_id))?;
            if actual != expected {
                return Err(RootIopError::ClaimMetadataMismatch(expected.claim_id));
            }
        }
        Ok(())
    }
}

impl<F, EF, Comm> RootIopBoundProofSystem<F, EF, Comm> for WitnessRootIopBoundProof<F, EF, Comm>
where
    F: Field + PartialEq + Clone,
    EF: ExtensionField<F> + PartialEq + Clone,
    Comm: PartialEq + Clone,
{
    type Proof = Self;
    type Error = RootIopError;

    fn prove(transcript: &RootIopBoundTranscript<F, EF, Comm>) -> Result<Self::Proof, Self::Error> {
        transcript.verify_witnessed_claim_values()?;
        Ok(Self {
            transcript: transcript.clone(),
        })
    }

    fn verify(
        expected_commitments: &[RootIopBoundCommitment<Comm>],
        expected_claims: &[RootIopOpeningClaim<F, EF>],
        proof: &Self::Proof,
    ) -> Result<(), Self::Error> {
        proof.transcript.verify_witnessed_claim_values()?;
        for expected in expected_commitments {
            let Some((actual, _)) = proof
                .transcript
                .oracles
                .iter()
                .find(|(commitment, _)| commitment.oracle_id == expected.oracle_id)
            else {
                return Err(RootIopError::UnknownOracle(expected.oracle_id));
            };
            if actual != expected {
                return Err(RootIopError::CommitmentMismatch(expected.oracle_id));
            }
        }
        for expected in expected_claims {
            let actual = proof
                .transcript
                .claims
                .get(expected.claim_id)
                .ok_or(RootIopError::UnknownClaim(expected.claim_id))?;
            if actual != expected {
                return Err(RootIopError::ClaimMetadataMismatch(expected.claim_id));
            }
        }
        Ok(())
    }
}

impl<F, EF> RootIopTranscript<F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// Check that every recorded claim is consistent with the recorded oracle
    /// values. This is a linear witness check, not a succinct proof verifier.
    pub fn verify_witnessed_claim_values(&self) -> Result<(), RootIopError> {
        for claim in &self.claims {
            let (_, oracle) = self
                .oracles
                .iter()
                .find(|(commitment, _)| commitment.oracle_id == claim.oracle_id)
                .ok_or(RootIopError::UnknownOracle(claim.oracle_id))?;
            verify_claim_against_oracle::<F, EF>(claim, oracle)?;
        }
        Ok(())
    }
}

fn verify_claim_against_oracle<F, EF>(
    claim: &RootIopOpeningClaim<F, EF>,
    oracle: &RootIopOracleValues<F, EF>,
) -> Result<(), RootIopError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    match (oracle, &claim.point, &claim.value) {
        (
            RootIopOracleValues::Base(values),
            RootIopOpeningPoint::Index(index),
            RootIopOpeningValue::Base(value),
        ) => {
            let actual = values.get(*index).ok_or(RootIopError::IndexOutOfBounds {
                oracle_id: claim.oracle_id,
                index: *index,
            })?;
            if actual != value {
                return Err(RootIopError::ClaimValueMismatch(claim.claim_id));
            }
        }
        (
            RootIopOracleValues::Extension(values),
            RootIopOpeningPoint::Index(index),
            RootIopOpeningValue::Extension(value),
        ) => {
            let actual = values.get(*index).ok_or(RootIopError::IndexOutOfBounds {
                oracle_id: claim.oracle_id,
                index: *index,
            })?;
            if actual != value {
                return Err(RootIopError::ClaimValueMismatch(claim.claim_id));
            }
        }
        (
            RootIopOracleValues::Extension(values),
            RootIopOpeningPoint::Mle(point),
            RootIopOpeningValue::Extension(value),
        ) => {
            let actual = Poly::<EF>::new(values.clone()).eval_ext::<F>(&Point::new(point.clone()));
            if &actual != value {
                return Err(RootIopError::ClaimValueMismatch(claim.claim_id));
            }
        }
        _ => return Err(RootIopError::OracleFieldMismatch),
    }
    Ok(())
}

#[derive(Clone, Debug, Default)]
struct RootIopState<F, EF> {
    transcript: RootIopTranscript<F, EF>,
}

#[derive(Clone, Debug)]
struct RootIopBoundState<F, EF, Comm> {
    transcript: RootIopBoundTranscript<F, EF, Comm>,
}

impl<F, EF, Comm> Default for RootIopBoundState<F, EF, Comm> {
    fn default() -> Self {
        Self {
            transcript: RootIopBoundTranscript::default(),
        }
    }
}

impl<F, EF> RootIopState<F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    fn push_oracle(
        &mut self,
        field: RootIopOracleField,
        values: RootIopOracleValues<F, EF>,
    ) -> Result<RootIopCommitment, RootIopError> {
        let len = match &values {
            RootIopOracleValues::Base(values) => values.len(),
            RootIopOracleValues::Extension(values) => values.len(),
        };
        let log_len = checked_log2_len(len)?;
        let commitment = RootIopCommitment {
            oracle_id: self.transcript.oracles.len(),
            log_len,
            field,
        };
        self.transcript.oracles.push((commitment, values));
        Ok(commitment)
    }

    fn push_claim(
        &mut self,
        oracle_id: usize,
        point: RootIopOpeningPoint<EF>,
        value: RootIopOpeningValue<F, EF>,
    ) -> usize {
        let claim_id = self.transcript.claims.len();
        self.transcript.claims.push(RootIopOpeningClaim {
            claim_id,
            oracle_id,
            point,
            value,
        });
        claim_id
    }
}

impl<F, EF, Comm> RootIopBoundState<F, EF, Comm>
where
    F: Field,
    EF: ExtensionField<F>,
    Comm: Clone,
{
    fn push_oracle(
        &mut self,
        field: RootIopOracleField,
        commitment: Comm,
        values: RootIopOracleValues<F, EF>,
    ) -> Result<RootIopBoundCommitment<Comm>, RootIopError> {
        let len = match &values {
            RootIopOracleValues::Base(values) => values.len(),
            RootIopOracleValues::Extension(values) => values.len(),
        };
        let log_len = checked_log2_len(len)?;
        let commitment = RootIopBoundCommitment {
            oracle_id: self.transcript.oracles.len(),
            log_len,
            field,
            commitment,
        };
        self.transcript.oracles.push((commitment.clone(), values));
        Ok(commitment)
    }

    fn push_claim(
        &mut self,
        oracle_id: usize,
        point: RootIopOpeningPoint<EF>,
        value: RootIopOpeningValue<F, EF>,
    ) -> usize {
        let claim_id = self.transcript.claims.len();
        self.transcript.claims.push(RootIopOpeningClaim {
            claim_id,
            oracle_id,
            point,
            value,
        });
        claim_id
    }
}

/// Prover-side recorder for the WARP root IOP.
#[derive(Debug)]
pub struct RootIopProver<F, EF> {
    log_codeword_len: usize,
    state: RefCell<RootIopState<F, EF>>,
}

impl<F, EF> RootIopProver<F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// Create an empty root IOP recorder for codewords of length `2^log_codeword_len`.
    pub fn new(log_codeword_len: usize) -> Self {
        Self {
            log_codeword_len,
            state: RefCell::new(RootIopState::default()),
        }
    }

    /// Register a fresh base-field codeword and witness.
    pub fn commit_fresh_codeword(
        &self,
        codeword: Vec<F>,
        witness: Vec<F>,
    ) -> Result<RootIopCommittedCodeword<F>, RootIopError> {
        if codeword.len() != (1 << self.log_codeword_len) {
            return Err(RootIopError::ShapeMismatch);
        }
        let commitment = self.state.borrow_mut().push_oracle(
            RootIopOracleField::Base,
            RootIopOracleValues::Base(codeword.clone()),
        )?;
        Ok(RootIopCommittedCodeword::new(commitment, codeword, witness))
    }

    /// Return the complete recorded transcript.
    pub fn transcript(&self) -> RootIopTranscript<F, EF>
    where
        F: Clone,
        EF: Clone,
    {
        self.state.borrow().transcript.clone()
    }
}

/// Verifier-side recorder for expected WARP root IOP claims.
#[derive(Debug)]
pub struct RootIopVerifier<F, EF> {
    log_codeword_len: usize,
    expected_claims: RefCell<Vec<RootIopOpeningClaim<F, EF>>>,
}

/// Prover-side root IOP recorder backed by real Plonky3 commitments.
#[derive(Debug)]
pub struct RootIopBoundProver<F, EF, MT>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
{
    mmcs: MT,
    ext_mmcs: ExtensionMmcs<F, EF, MT>,
    log_codeword_len: usize,
    state: RefCell<RootIopBoundState<F, EF, MT::Commitment>>,
}

impl<F, EF, MT> RootIopBoundProver<F, EF, MT>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
{
    /// Create an empty committed root IOP recorder.
    pub fn new(mmcs: MT, log_codeword_len: usize) -> Self {
        let ext_mmcs = ExtensionMmcs::new(mmcs.clone());
        Self {
            mmcs,
            ext_mmcs,
            log_codeword_len,
            state: RefCell::new(RootIopBoundState::default()),
        }
    }

    /// Register and commit a fresh base-field codeword.
    pub fn commit_fresh_codeword(
        &self,
        codeword: Vec<F>,
        witness: Vec<F>,
    ) -> Result<RootIopBoundCommittedCodeword<F, MT::Commitment>, RootIopError> {
        if codeword.len() != (1 << self.log_codeword_len) {
            return Err(RootIopError::ShapeMismatch);
        }
        let (backend_commitment, _td) = self
            .mmcs
            .commit_matrix(RowMajorMatrix::new_col(codeword.clone()));
        let commitment = self.state.borrow_mut().push_oracle(
            RootIopOracleField::Base,
            backend_commitment,
            RootIopOracleValues::Base(codeword.clone()),
        )?;
        Ok(RootIopBoundCommittedCodeword::new(
            commitment, codeword, witness,
        ))
    }

    /// Return the complete recorded transcript.
    pub fn transcript(&self) -> RootIopBoundTranscript<F, EF, MT::Commitment>
    where
        F: Clone,
        EF: Clone,
        MT::Commitment: Clone,
    {
        self.state.borrow().transcript.clone()
    }
}

/// Verifier-side recorder for a committed root IOP transcript.
#[derive(Debug)]
pub struct RootIopBoundVerifier<F, EF, Comm> {
    log_codeword_len: usize,
    expected_commitments: RefCell<Vec<RootIopBoundCommitment<Comm>>>,
    expected_claims: RefCell<Vec<RootIopOpeningClaim<F, EF>>>,
}

impl<F, EF, Comm> RootIopBoundVerifier<F, EF, Comm>
where
    F: Field,
    EF: ExtensionField<F>,
    Comm: Clone,
{
    /// Create an empty verifier recorder.
    pub fn new(log_codeword_len: usize) -> Self {
        Self {
            log_codeword_len,
            expected_commitments: RefCell::new(Vec::new()),
            expected_claims: RefCell::new(Vec::new()),
        }
    }

    /// Return expected opening claims.
    pub fn expected_claims(&self) -> Vec<RootIopOpeningClaim<F, EF>>
    where
        F: Clone,
        EF: Clone,
    {
        self.expected_claims.borrow().clone()
    }

    /// Return commitments observed by the WARP verifier.
    pub fn expected_commitments(&self) -> Vec<RootIopBoundCommitment<Comm>> {
        self.expected_commitments.borrow().clone()
    }

    /// Check that expected commitments and claims are present in a transcript.
    pub fn verify_against_transcript(
        &self,
        transcript: &RootIopBoundTranscript<F, EF, Comm>,
    ) -> Result<(), RootIopError>
    where
        F: PartialEq,
        EF: PartialEq,
        Comm: PartialEq,
    {
        for expected in self.expected_commitments.borrow().iter() {
            let Some((actual, _)) = transcript
                .oracles
                .iter()
                .find(|(commitment, _)| commitment.oracle_id == expected.oracle_id)
            else {
                return Err(RootIopError::UnknownOracle(expected.oracle_id));
            };
            if actual != expected {
                return Err(RootIopError::CommitmentMismatch(expected.oracle_id));
            }
        }
        for expected in self.expected_claims.borrow().iter() {
            let actual = transcript
                .claims
                .get(expected.claim_id)
                .ok_or(RootIopError::UnknownClaim(expected.claim_id))?;
            if actual != expected {
                return Err(RootIopError::ClaimMetadataMismatch(expected.claim_id));
            }
        }
        Ok(())
    }

    fn record_commitment(&self, commitment: &RootIopBoundCommitment<Comm>) {
        let mut expected = self.expected_commitments.borrow_mut();
        if !expected
            .iter()
            .any(|known| known.oracle_id == commitment.oracle_id)
        {
            expected.push(commitment.clone());
        }
    }

    fn record_expected_claim(
        &self,
        proof_claim_id: usize,
        oracle_id: usize,
        point: RootIopOpeningPoint<EF>,
        value: RootIopOpeningValue<F, EF>,
    ) {
        self.expected_claims.borrow_mut().push(RootIopOpeningClaim {
            claim_id: proof_claim_id,
            oracle_id,
            point,
            value,
        });
    }
}

impl<F, EF> RootIopVerifier<F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// Create an empty verifier recorder for codewords of length `2^log_codeword_len`.
    pub fn new(log_codeword_len: usize) -> Self {
        Self {
            log_codeword_len,
            expected_claims: RefCell::new(Vec::new()),
        }
    }

    /// Return the claims that the WARP verifier expected the final proof to authenticate.
    pub fn expected_claims(&self) -> Vec<RootIopOpeningClaim<F, EF>>
    where
        F: Clone,
        EF: Clone,
    {
        self.expected_claims.borrow().clone()
    }

    /// Check that the expected verifier claims are included in a prover transcript.
    pub fn verify_against_transcript(
        &self,
        transcript: &RootIopTranscript<F, EF>,
    ) -> Result<(), RootIopError>
    where
        F: PartialEq,
        EF: PartialEq,
    {
        for expected in self.expected_claims.borrow().iter() {
            let actual = transcript
                .claims
                .get(expected.claim_id)
                .ok_or(RootIopError::UnknownClaim(expected.claim_id))?;
            if actual != expected {
                return Err(RootIopError::ClaimMetadataMismatch(expected.claim_id));
            }
        }
        Ok(())
    }

    fn record_expected_claim(
        &self,
        proof_claim_id: usize,
        oracle_id: usize,
        point: RootIopOpeningPoint<EF>,
        value: RootIopOpeningValue<F, EF>,
    ) {
        self.expected_claims.borrow_mut().push(RootIopOpeningClaim {
            claim_id: proof_claim_id,
            oracle_id,
            point,
            value,
        });
    }
}

impl<F, EF, C> ExternalCodewordOpeningProver<F, C> for RootIopProver<F, EF>
where
    F: Field + Serialize + serde::de::DeserializeOwned,
    EF: ExtensionField<F>,
    C: ExternalCommittedCodeword<F, Commitment = RootIopCommitment>,
{
    type Proof = RootIopOpeningProof;
    type Error = RootIopError;

    fn open(&self, committed: &C, index: usize) -> Result<(F, Self::Proof), Self::Error> {
        let value = *committed
            .codeword()
            .get(index)
            .ok_or(RootIopError::IndexOutOfBounds {
                oracle_id: committed.commitment().oracle_id,
                index,
            })?;
        let claim_id = self.state.borrow_mut().push_claim(
            committed.commitment().oracle_id,
            RootIopOpeningPoint::Index(index),
            RootIopOpeningValue::Base(value),
        );
        Ok((
            value,
            RootIopOpeningProof {
                claim_ids: alloc::vec![claim_id],
            },
        ))
    }
}

impl<F, EF, C> ExternalCodewordBatchOpeningProver<F, C> for RootIopProver<F, EF>
where
    F: Field + Serialize + serde::de::DeserializeOwned,
    EF: ExtensionField<F>,
    C: ExternalCommittedCodeword<F, Commitment = RootIopCommitment>,
{
    type BatchProof = RootIopOpeningProof;

    fn open_batch(
        &self,
        committed: &C,
        indices: &[usize],
    ) -> Result<(Vec<F>, Self::BatchProof), Self::Error> {
        let mut values = Vec::with_capacity(indices.len());
        let mut claim_ids = Vec::with_capacity(indices.len());
        for &index in indices {
            let value = *committed
                .codeword()
                .get(index)
                .ok_or(RootIopError::IndexOutOfBounds {
                    oracle_id: committed.commitment().oracle_id,
                    index,
                })?;
            let claim_id = self.state.borrow_mut().push_claim(
                committed.commitment().oracle_id,
                RootIopOpeningPoint::Index(index),
                RootIopOpeningValue::Base(value),
            );
            values.push(value);
            claim_ids.push(claim_id);
        }
        Ok((values, RootIopOpeningProof { claim_ids }))
    }
}

impl<F, EF, Challenger> ExternalCodewordOpeningVerifier<F, Challenger> for RootIopVerifier<F, EF>
where
    F: Field + PrimeCharacteristicRing + Serialize + serde::de::DeserializeOwned,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F>,
{
    type Commitment = RootIopCommitment;
    type Proof = RootIopOpeningProof;
    type Error = RootIopError;

    fn observe_commitment(&self, challenger: &mut Challenger, commitment: &Self::Commitment) {
        commitment.observe_into::<F, _>(challenger);
    }

    fn verify_opening(
        &self,
        commitment: &Self::Commitment,
        log_codeword_len: usize,
        index: usize,
        value: F,
        proof: &Self::Proof,
    ) -> Result<(), Self::Error> {
        if commitment.field != RootIopOracleField::Base || commitment.log_len != log_codeword_len {
            return Err(RootIopError::ShapeMismatch);
        }
        let claim_id = *proof
            .claim_ids
            .first()
            .ok_or(RootIopError::OpeningArityMismatch)?;
        if proof.claim_ids.len() != 1 {
            return Err(RootIopError::OpeningArityMismatch);
        }
        self.record_expected_claim(
            claim_id,
            commitment.oracle_id,
            RootIopOpeningPoint::Index(index),
            RootIopOpeningValue::Base(value),
        );
        Ok(())
    }
}

impl<F, EF, Challenger> ExternalCodewordBatchOpeningVerifier<F, Challenger>
    for RootIopVerifier<F, EF>
where
    F: Field + PrimeCharacteristicRing + Serialize + serde::de::DeserializeOwned,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F>,
{
    type BatchProof = RootIopOpeningProof;

    fn verify_batch_opening(
        &self,
        commitment: &Self::Commitment,
        log_codeword_len: usize,
        indices: &[usize],
        values: &[F],
        proof: &Self::BatchProof,
    ) -> Result<(), Self::Error> {
        if commitment.field != RootIopOracleField::Base || commitment.log_len != log_codeword_len {
            return Err(RootIopError::ShapeMismatch);
        }
        if indices.len() != values.len() || indices.len() != proof.claim_ids.len() {
            return Err(RootIopError::OpeningArityMismatch);
        }
        for ((&index, &value), &claim_id) in indices
            .iter()
            .zip(values.iter())
            .zip(proof.claim_ids.iter())
        {
            self.record_expected_claim(
                claim_id,
                commitment.oracle_id,
                RootIopOpeningPoint::Index(index),
                RootIopOpeningValue::Base(value),
            );
        }
        Ok(())
    }
}

impl<F, EF, Challenger> AccumulatorCommitmentBackend<F, EF, Challenger> for RootIopProver<F, EF>
where
    F: Field,
    EF: ExtensionField<F> + Serialize + serde::de::DeserializeOwned,
    Challenger: FieldChallenger<F>,
{
    type Commitment = RootIopCommitment;
    type ProverData = RootIopAccumulatorProverData<EF>;
    type Proof = RootIopOpeningProof;
    type Error = RootIopError;

    fn commit(
        &self,
        codeword: Vec<EF>,
    ) -> Result<(Self::Commitment, Self::ProverData), Self::Error> {
        if codeword.len() != (1 << self.log_codeword_len) {
            return Err(RootIopError::ShapeMismatch);
        }
        let commitment = self.state.borrow_mut().push_oracle(
            RootIopOracleField::Extension,
            RootIopOracleValues::Extension(codeword.clone()),
        )?;
        Ok((
            commitment,
            RootIopAccumulatorProverData {
                commitment,
                codeword,
            },
        ))
    }

    fn open(
        &self,
        prover_data: &Self::ProverData,
        index: usize,
    ) -> Result<(EF, Self::Proof), Self::Error> {
        let value = *prover_data
            .codeword
            .get(index)
            .ok_or(RootIopError::IndexOutOfBounds {
                oracle_id: prover_data.commitment.oracle_id,
                index,
            })?;
        let claim_id = self.state.borrow_mut().push_claim(
            prover_data.commitment.oracle_id,
            RootIopOpeningPoint::Index(index),
            RootIopOpeningValue::Extension(value),
        );
        Ok((
            value,
            RootIopOpeningProof {
                claim_ids: alloc::vec![claim_id],
            },
        ))
    }

    fn observe_commitment(&self, challenger: &mut Challenger, commitment: &Self::Commitment) {
        commitment.observe_into::<F, _>(challenger);
    }

    fn verify_opening(
        &self,
        _commitment: &Self::Commitment,
        _log_codeword_len: usize,
        _index: usize,
        _value: EF,
        _proof: &Self::Proof,
    ) -> Result<(), Self::Error> {
        Err(RootIopError::ProverUsedAsVerifier)
    }
}

impl<F, EF, Challenger> AccumulatorBatchOpeningBackend<F, EF, Challenger> for RootIopProver<F, EF>
where
    F: Field,
    EF: ExtensionField<F> + Serialize + serde::de::DeserializeOwned,
    Challenger: FieldChallenger<F>,
{
    type BatchProof = RootIopOpeningProof;

    fn open_batch(
        &self,
        prover_data: &Self::ProverData,
        indices: &[usize],
    ) -> Result<(Vec<EF>, Self::BatchProof), Self::Error> {
        let mut values = Vec::with_capacity(indices.len());
        let mut claim_ids = Vec::with_capacity(indices.len());
        for &index in indices {
            let value = *prover_data
                .codeword
                .get(index)
                .ok_or(RootIopError::IndexOutOfBounds {
                    oracle_id: prover_data.commitment.oracle_id,
                    index,
                })?;
            let claim_id = self.state.borrow_mut().push_claim(
                prover_data.commitment.oracle_id,
                RootIopOpeningPoint::Index(index),
                RootIopOpeningValue::Extension(value),
            );
            values.push(value);
            claim_ids.push(claim_id);
        }
        Ok((values, RootIopOpeningProof { claim_ids }))
    }

    fn verify_batch_opening(
        &self,
        _commitment: &Self::Commitment,
        _log_codeword_len: usize,
        _indices: &[usize],
        _values: &[EF],
        _proof: &Self::BatchProof,
    ) -> Result<(), Self::Error> {
        Err(RootIopError::ProverUsedAsVerifier)
    }
}

impl<F, EF, Challenger> AccumulatorCommitmentBackend<F, EF, Challenger> for RootIopVerifier<F, EF>
where
    F: Field + PrimeCharacteristicRing,
    EF: ExtensionField<F> + Serialize + serde::de::DeserializeOwned,
    Challenger: FieldChallenger<F>,
{
    type Commitment = RootIopCommitment;
    type ProverData = RootIopAccumulatorProverData<EF>;
    type Proof = RootIopOpeningProof;
    type Error = RootIopError;

    fn commit(
        &self,
        _codeword: Vec<EF>,
    ) -> Result<(Self::Commitment, Self::ProverData), Self::Error> {
        Err(RootIopError::VerifierUsedAsProver)
    }

    fn open(
        &self,
        _prover_data: &Self::ProverData,
        _index: usize,
    ) -> Result<(EF, Self::Proof), Self::Error> {
        Err(RootIopError::VerifierUsedAsProver)
    }

    fn observe_commitment(&self, challenger: &mut Challenger, commitment: &Self::Commitment) {
        commitment.observe_into::<F, _>(challenger);
    }

    fn verify_opening(
        &self,
        commitment: &Self::Commitment,
        log_codeword_len: usize,
        index: usize,
        value: EF,
        proof: &Self::Proof,
    ) -> Result<(), Self::Error> {
        if commitment.field != RootIopOracleField::Extension
            || commitment.log_len != log_codeword_len
        {
            return Err(RootIopError::ShapeMismatch);
        }
        let claim_id = *proof
            .claim_ids
            .first()
            .ok_or(RootIopError::OpeningArityMismatch)?;
        if proof.claim_ids.len() != 1 {
            return Err(RootIopError::OpeningArityMismatch);
        }
        self.record_expected_claim(
            claim_id,
            commitment.oracle_id,
            RootIopOpeningPoint::Index(index),
            RootIopOpeningValue::Extension(value),
        );
        Ok(())
    }
}

impl<F, EF, Challenger> AccumulatorBatchOpeningBackend<F, EF, Challenger> for RootIopVerifier<F, EF>
where
    F: Field + PrimeCharacteristicRing,
    EF: ExtensionField<F> + Serialize + serde::de::DeserializeOwned,
    Challenger: FieldChallenger<F>,
{
    type BatchProof = RootIopOpeningProof;

    fn open_batch(
        &self,
        _prover_data: &Self::ProverData,
        _indices: &[usize],
    ) -> Result<(Vec<EF>, Self::BatchProof), Self::Error> {
        Err(RootIopError::VerifierUsedAsProver)
    }

    fn verify_batch_opening(
        &self,
        commitment: &Self::Commitment,
        log_codeword_len: usize,
        indices: &[usize],
        values: &[EF],
        proof: &Self::BatchProof,
    ) -> Result<(), Self::Error> {
        if commitment.field != RootIopOracleField::Extension
            || commitment.log_len != log_codeword_len
        {
            return Err(RootIopError::ShapeMismatch);
        }
        if indices.len() != values.len() || indices.len() != proof.claim_ids.len() {
            return Err(RootIopError::OpeningArityMismatch);
        }
        for ((&index, &value), &claim_id) in indices
            .iter()
            .zip(values.iter())
            .zip(proof.claim_ids.iter())
        {
            self.record_expected_claim(
                claim_id,
                commitment.oracle_id,
                RootIopOpeningPoint::Index(index),
                RootIopOpeningValue::Extension(value),
            );
        }
        Ok(())
    }
}

impl<F, EF, Challenger> AccumulatorPointOpeningBackend<F, EF, Challenger> for RootIopProver<F, EF>
where
    F: Field,
    EF: ExtensionField<F> + Serialize + serde::de::DeserializeOwned,
    Challenger: FieldChallenger<F>,
{
    type PointProof = RootIopOpeningProof;
    type PointError = RootIopError;

    fn num_vars(&self) -> usize {
        self.log_codeword_len
    }

    fn prove_points(
        &self,
        prover_data: &Self::ProverData,
        opening_points: &[Vec<Point<EF>>],
    ) -> Result<(MultilinearOpenedValues<EF>, Self::PointProof), Self::PointError> {
        if opening_points.len() != 1 {
            return Err(RootIopError::OpeningArityMismatch);
        }
        let poly = Poly::<EF>::new(prover_data.codeword.clone());
        let mut values = Vec::with_capacity(opening_points[0].len());
        let mut claim_ids = Vec::with_capacity(opening_points[0].len());
        for point in &opening_points[0] {
            let value = poly.eval_ext::<F>(point);
            let claim_id = self.state.borrow_mut().push_claim(
                prover_data.commitment.oracle_id,
                RootIopOpeningPoint::Mle(point.as_slice().to_vec()),
                RootIopOpeningValue::Extension(value),
            );
            values.push(value);
            claim_ids.push(claim_id);
        }
        Ok((alloc::vec![values], RootIopOpeningProof { claim_ids }))
    }

    fn verify_points(
        &self,
        _commitment: &Self::Commitment,
        _opening_claims: &[Vec<(Point<EF>, EF)>],
        _proof: &Self::PointProof,
    ) -> Result<(), Self::PointError> {
        Err(RootIopError::ProverUsedAsVerifier)
    }
}

impl<F, EF, Challenger> AccumulatorPointOpeningBackend<F, EF, Challenger> for RootIopVerifier<F, EF>
where
    F: Field + PrimeCharacteristicRing,
    EF: ExtensionField<F> + Serialize + serde::de::DeserializeOwned,
    Challenger: FieldChallenger<F>,
{
    type PointProof = RootIopOpeningProof;
    type PointError = RootIopError;

    fn num_vars(&self) -> usize {
        self.log_codeword_len
    }

    fn prove_points(
        &self,
        _prover_data: &Self::ProverData,
        _opening_points: &[Vec<Point<EF>>],
    ) -> Result<(MultilinearOpenedValues<EF>, Self::PointProof), Self::PointError> {
        Err(RootIopError::VerifierUsedAsProver)
    }

    fn verify_points(
        &self,
        commitment: &Self::Commitment,
        opening_claims: &[Vec<(Point<EF>, EF)>],
        proof: &Self::PointProof,
    ) -> Result<(), Self::PointError> {
        if commitment.field != RootIopOracleField::Extension {
            return Err(RootIopError::ShapeMismatch);
        }
        if opening_claims.len() != 1 || opening_claims[0].len() != proof.claim_ids.len() {
            return Err(RootIopError::OpeningArityMismatch);
        }
        for ((point, value), &claim_id) in opening_claims[0].iter().zip(proof.claim_ids.iter()) {
            self.record_expected_claim(
                claim_id,
                commitment.oracle_id,
                RootIopOpeningPoint::Mle(point.as_slice().to_vec()),
                RootIopOpeningValue::Extension(*value),
            );
        }
        Ok(())
    }
}

impl<F, EF, MT, C> ExternalCodewordOpeningProver<F, C> for RootIopBoundProver<F, EF, MT>
where
    F: Field + Serialize + serde::de::DeserializeOwned,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
    C: ExternalCommittedCodeword<F, Commitment = RootIopBoundCommitment<MT::Commitment>>,
{
    type Proof = RootIopOpeningProof;
    type Error = RootIopError;

    fn open(&self, committed: &C, index: usize) -> Result<(F, Self::Proof), Self::Error> {
        let commitment = committed.commitment();
        let value = *committed
            .codeword()
            .get(index)
            .ok_or(RootIopError::IndexOutOfBounds {
                oracle_id: commitment.oracle_id,
                index,
            })?;
        let claim_id = self.state.borrow_mut().push_claim(
            commitment.oracle_id,
            RootIopOpeningPoint::Index(index),
            RootIopOpeningValue::Base(value),
        );
        Ok((
            value,
            RootIopOpeningProof {
                claim_ids: alloc::vec![claim_id],
            },
        ))
    }
}

impl<F, EF, MT, C> ExternalCodewordBatchOpeningProver<F, C> for RootIopBoundProver<F, EF, MT>
where
    F: Field + Serialize + serde::de::DeserializeOwned,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
    C: ExternalCommittedCodeword<F, Commitment = RootIopBoundCommitment<MT::Commitment>>,
{
    type BatchProof = RootIopOpeningProof;

    fn open_batch(
        &self,
        committed: &C,
        indices: &[usize],
    ) -> Result<(Vec<F>, Self::BatchProof), Self::Error> {
        let commitment = committed.commitment();
        let mut values = Vec::with_capacity(indices.len());
        let mut claim_ids = Vec::with_capacity(indices.len());
        for &index in indices {
            let value = *committed
                .codeword()
                .get(index)
                .ok_or(RootIopError::IndexOutOfBounds {
                    oracle_id: commitment.oracle_id,
                    index,
                })?;
            let claim_id = self.state.borrow_mut().push_claim(
                commitment.oracle_id,
                RootIopOpeningPoint::Index(index),
                RootIopOpeningValue::Base(value),
            );
            values.push(value);
            claim_ids.push(claim_id);
        }
        Ok((values, RootIopOpeningProof { claim_ids }))
    }
}

impl<F, EF, Comm, Challenger> ExternalCodewordOpeningVerifier<F, Challenger>
    for RootIopBoundVerifier<F, EF, Comm>
where
    F: Field + PrimeCharacteristicRing + Serialize + serde::de::DeserializeOwned,
    EF: ExtensionField<F>,
    Comm: Clone + Serialize + serde::de::DeserializeOwned,
    Challenger: FieldChallenger<F> + CanObserve<Comm>,
{
    type Commitment = RootIopBoundCommitment<Comm>;
    type Proof = RootIopOpeningProof;
    type Error = RootIopError;

    fn observe_commitment(&self, challenger: &mut Challenger, commitment: &Self::Commitment) {
        self.record_commitment(commitment);
        commitment.observe_into::<F, _>(challenger);
    }

    fn verify_opening(
        &self,
        commitment: &Self::Commitment,
        log_codeword_len: usize,
        index: usize,
        value: F,
        proof: &Self::Proof,
    ) -> Result<(), Self::Error> {
        if commitment.field != RootIopOracleField::Base || commitment.log_len != log_codeword_len {
            return Err(RootIopError::ShapeMismatch);
        }
        let claim_id = *proof
            .claim_ids
            .first()
            .ok_or(RootIopError::OpeningArityMismatch)?;
        if proof.claim_ids.len() != 1 {
            return Err(RootIopError::OpeningArityMismatch);
        }
        self.record_expected_claim(
            claim_id,
            commitment.oracle_id,
            RootIopOpeningPoint::Index(index),
            RootIopOpeningValue::Base(value),
        );
        Ok(())
    }
}

impl<F, EF, Comm, Challenger> ExternalCodewordBatchOpeningVerifier<F, Challenger>
    for RootIopBoundVerifier<F, EF, Comm>
where
    F: Field + PrimeCharacteristicRing + Serialize + serde::de::DeserializeOwned,
    EF: ExtensionField<F>,
    Comm: Clone + Serialize + serde::de::DeserializeOwned,
    Challenger: FieldChallenger<F> + CanObserve<Comm>,
{
    type BatchProof = RootIopOpeningProof;

    fn verify_batch_opening(
        &self,
        commitment: &Self::Commitment,
        log_codeword_len: usize,
        indices: &[usize],
        values: &[F],
        proof: &Self::BatchProof,
    ) -> Result<(), Self::Error> {
        if commitment.field != RootIopOracleField::Base || commitment.log_len != log_codeword_len {
            return Err(RootIopError::ShapeMismatch);
        }
        if indices.len() != values.len() || indices.len() != proof.claim_ids.len() {
            return Err(RootIopError::OpeningArityMismatch);
        }
        for ((&index, &value), &claim_id) in indices
            .iter()
            .zip(values.iter())
            .zip(proof.claim_ids.iter())
        {
            self.record_expected_claim(
                claim_id,
                commitment.oracle_id,
                RootIopOpeningPoint::Index(index),
                RootIopOpeningValue::Base(value),
            );
        }
        Ok(())
    }
}

impl<F, EF, MT, Challenger> AccumulatorCommitmentBackend<F, EF, Challenger>
    for RootIopBoundProver<F, EF, MT>
where
    F: Field + PrimeCharacteristicRing,
    EF: ExtensionField<F> + Serialize + serde::de::DeserializeOwned,
    MT: Mmcs<F>,
    Challenger: FieldChallenger<F> + CanObserve<MT::Commitment>,
{
    type Commitment = RootIopBoundCommitment<MT::Commitment>;
    type ProverData = RootIopBoundAccumulatorProverData<EF, MT::Commitment>;
    type Proof = RootIopOpeningProof;
    type Error = RootIopError;

    fn commit(
        &self,
        codeword: Vec<EF>,
    ) -> Result<(Self::Commitment, Self::ProverData), Self::Error> {
        if codeword.len() != (1 << self.log_codeword_len) {
            return Err(RootIopError::ShapeMismatch);
        }
        let (backend_commitment, _td) = self
            .ext_mmcs
            .commit_matrix(RowMajorMatrix::new_col(codeword.clone()));
        let commitment = self.state.borrow_mut().push_oracle(
            RootIopOracleField::Extension,
            backend_commitment,
            RootIopOracleValues::Extension(codeword.clone()),
        )?;
        Ok((
            commitment.clone(),
            RootIopBoundAccumulatorProverData {
                commitment,
                codeword,
            },
        ))
    }

    fn open(
        &self,
        prover_data: &Self::ProverData,
        index: usize,
    ) -> Result<(EF, Self::Proof), Self::Error> {
        let value = *prover_data
            .codeword
            .get(index)
            .ok_or(RootIopError::IndexOutOfBounds {
                oracle_id: prover_data.commitment.oracle_id,
                index,
            })?;
        let claim_id = self.state.borrow_mut().push_claim(
            prover_data.commitment.oracle_id,
            RootIopOpeningPoint::Index(index),
            RootIopOpeningValue::Extension(value),
        );
        Ok((
            value,
            RootIopOpeningProof {
                claim_ids: alloc::vec![claim_id],
            },
        ))
    }

    fn observe_commitment(&self, challenger: &mut Challenger, commitment: &Self::Commitment) {
        commitment.observe_into::<F, _>(challenger);
    }

    fn verify_opening(
        &self,
        _commitment: &Self::Commitment,
        _log_codeword_len: usize,
        _index: usize,
        _value: EF,
        _proof: &Self::Proof,
    ) -> Result<(), Self::Error> {
        Err(RootIopError::ProverUsedAsVerifier)
    }
}

impl<F, EF, MT, Challenger> AccumulatorBatchOpeningBackend<F, EF, Challenger>
    for RootIopBoundProver<F, EF, MT>
where
    F: Field + PrimeCharacteristicRing,
    EF: ExtensionField<F> + Serialize + serde::de::DeserializeOwned,
    MT: Mmcs<F>,
    Challenger: FieldChallenger<F> + CanObserve<MT::Commitment>,
{
    type BatchProof = RootIopOpeningProof;

    fn open_batch(
        &self,
        prover_data: &Self::ProverData,
        indices: &[usize],
    ) -> Result<(Vec<EF>, Self::BatchProof), Self::Error> {
        let mut values = Vec::with_capacity(indices.len());
        let mut claim_ids = Vec::with_capacity(indices.len());
        for &index in indices {
            let value = *prover_data
                .codeword
                .get(index)
                .ok_or(RootIopError::IndexOutOfBounds {
                    oracle_id: prover_data.commitment.oracle_id,
                    index,
                })?;
            let claim_id = self.state.borrow_mut().push_claim(
                prover_data.commitment.oracle_id,
                RootIopOpeningPoint::Index(index),
                RootIopOpeningValue::Extension(value),
            );
            values.push(value);
            claim_ids.push(claim_id);
        }
        Ok((values, RootIopOpeningProof { claim_ids }))
    }

    fn verify_batch_opening(
        &self,
        _commitment: &Self::Commitment,
        _log_codeword_len: usize,
        _indices: &[usize],
        _values: &[EF],
        _proof: &Self::BatchProof,
    ) -> Result<(), Self::Error> {
        Err(RootIopError::ProverUsedAsVerifier)
    }
}

impl<F, EF, Comm, Challenger> AccumulatorCommitmentBackend<F, EF, Challenger>
    for RootIopBoundVerifier<F, EF, Comm>
where
    F: Field + PrimeCharacteristicRing,
    EF: ExtensionField<F> + Serialize + serde::de::DeserializeOwned,
    Comm: Clone + Serialize + serde::de::DeserializeOwned,
    Challenger: FieldChallenger<F> + CanObserve<Comm>,
{
    type Commitment = RootIopBoundCommitment<Comm>;
    type ProverData = RootIopBoundAccumulatorProverData<EF, Comm>;
    type Proof = RootIopOpeningProof;
    type Error = RootIopError;

    fn commit(
        &self,
        _codeword: Vec<EF>,
    ) -> Result<(Self::Commitment, Self::ProverData), Self::Error> {
        Err(RootIopError::VerifierUsedAsProver)
    }

    fn open(
        &self,
        _prover_data: &Self::ProverData,
        _index: usize,
    ) -> Result<(EF, Self::Proof), Self::Error> {
        Err(RootIopError::VerifierUsedAsProver)
    }

    fn observe_commitment(&self, challenger: &mut Challenger, commitment: &Self::Commitment) {
        self.record_commitment(commitment);
        commitment.observe_into::<F, _>(challenger);
    }

    fn verify_opening(
        &self,
        commitment: &Self::Commitment,
        log_codeword_len: usize,
        index: usize,
        value: EF,
        proof: &Self::Proof,
    ) -> Result<(), Self::Error> {
        if commitment.field != RootIopOracleField::Extension
            || commitment.log_len != log_codeword_len
        {
            return Err(RootIopError::ShapeMismatch);
        }
        let claim_id = *proof
            .claim_ids
            .first()
            .ok_or(RootIopError::OpeningArityMismatch)?;
        if proof.claim_ids.len() != 1 {
            return Err(RootIopError::OpeningArityMismatch);
        }
        self.record_expected_claim(
            claim_id,
            commitment.oracle_id,
            RootIopOpeningPoint::Index(index),
            RootIopOpeningValue::Extension(value),
        );
        Ok(())
    }
}

impl<F, EF, Comm, Challenger> AccumulatorBatchOpeningBackend<F, EF, Challenger>
    for RootIopBoundVerifier<F, EF, Comm>
where
    F: Field + PrimeCharacteristicRing,
    EF: ExtensionField<F> + Serialize + serde::de::DeserializeOwned,
    Comm: Clone + Serialize + serde::de::DeserializeOwned,
    Challenger: FieldChallenger<F> + CanObserve<Comm>,
{
    type BatchProof = RootIopOpeningProof;

    fn open_batch(
        &self,
        _prover_data: &Self::ProverData,
        _indices: &[usize],
    ) -> Result<(Vec<EF>, Self::BatchProof), Self::Error> {
        Err(RootIopError::VerifierUsedAsProver)
    }

    fn verify_batch_opening(
        &self,
        commitment: &Self::Commitment,
        log_codeword_len: usize,
        indices: &[usize],
        values: &[EF],
        proof: &Self::BatchProof,
    ) -> Result<(), Self::Error> {
        if commitment.field != RootIopOracleField::Extension
            || commitment.log_len != log_codeword_len
        {
            return Err(RootIopError::ShapeMismatch);
        }
        if indices.len() != values.len() || indices.len() != proof.claim_ids.len() {
            return Err(RootIopError::OpeningArityMismatch);
        }
        for ((&index, &value), &claim_id) in indices
            .iter()
            .zip(values.iter())
            .zip(proof.claim_ids.iter())
        {
            self.record_expected_claim(
                claim_id,
                commitment.oracle_id,
                RootIopOpeningPoint::Index(index),
                RootIopOpeningValue::Extension(value),
            );
        }
        Ok(())
    }
}

impl<F, EF, MT, Challenger> AccumulatorPointOpeningBackend<F, EF, Challenger>
    for RootIopBoundProver<F, EF, MT>
where
    F: Field + PrimeCharacteristicRing,
    EF: ExtensionField<F> + Serialize + serde::de::DeserializeOwned,
    MT: Mmcs<F>,
    Challenger: FieldChallenger<F> + CanObserve<MT::Commitment>,
{
    type PointProof = RootIopOpeningProof;
    type PointError = RootIopError;

    fn num_vars(&self) -> usize {
        self.log_codeword_len
    }

    fn prove_points(
        &self,
        prover_data: &Self::ProverData,
        opening_points: &[Vec<Point<EF>>],
    ) -> Result<(MultilinearOpenedValues<EF>, Self::PointProof), Self::PointError> {
        if opening_points.len() != 1 {
            return Err(RootIopError::OpeningArityMismatch);
        }
        let poly = Poly::<EF>::new(prover_data.codeword.clone());
        let mut values = Vec::with_capacity(opening_points[0].len());
        let mut claim_ids = Vec::with_capacity(opening_points[0].len());
        for point in &opening_points[0] {
            let value = poly.eval_ext::<F>(point);
            let claim_id = self.state.borrow_mut().push_claim(
                prover_data.commitment.oracle_id,
                RootIopOpeningPoint::Mle(point.as_slice().to_vec()),
                RootIopOpeningValue::Extension(value),
            );
            values.push(value);
            claim_ids.push(claim_id);
        }
        Ok((alloc::vec![values], RootIopOpeningProof { claim_ids }))
    }

    fn verify_points(
        &self,
        _commitment: &Self::Commitment,
        _opening_claims: &[Vec<(Point<EF>, EF)>],
        _proof: &Self::PointProof,
    ) -> Result<(), Self::PointError> {
        Err(RootIopError::ProverUsedAsVerifier)
    }
}

impl<F, EF, Comm, Challenger> AccumulatorPointOpeningBackend<F, EF, Challenger>
    for RootIopBoundVerifier<F, EF, Comm>
where
    F: Field + PrimeCharacteristicRing,
    EF: ExtensionField<F> + Serialize + serde::de::DeserializeOwned,
    Comm: Clone + Serialize + serde::de::DeserializeOwned,
    Challenger: FieldChallenger<F> + CanObserve<Comm>,
{
    type PointProof = RootIopOpeningProof;
    type PointError = RootIopError;

    fn num_vars(&self) -> usize {
        self.log_codeword_len
    }

    fn prove_points(
        &self,
        _prover_data: &Self::ProverData,
        _opening_points: &[Vec<Point<EF>>],
    ) -> Result<(MultilinearOpenedValues<EF>, Self::PointProof), Self::PointError> {
        Err(RootIopError::VerifierUsedAsProver)
    }

    fn verify_points(
        &self,
        commitment: &Self::Commitment,
        opening_claims: &[Vec<(Point<EF>, EF)>],
        proof: &Self::PointProof,
    ) -> Result<(), Self::PointError> {
        if commitment.field != RootIopOracleField::Extension {
            return Err(RootIopError::ShapeMismatch);
        }
        if opening_claims.len() != 1 || opening_claims[0].len() != proof.claim_ids.len() {
            return Err(RootIopError::OpeningArityMismatch);
        }
        for ((point, value), &claim_id) in opening_claims[0].iter().zip(proof.claim_ids.iter()) {
            self.record_expected_claim(
                claim_id,
                commitment.oracle_id,
                RootIopOpeningPoint::Mle(point.as_slice().to_vec()),
                RootIopOpeningValue::Extension(*value),
            );
        }
        Ok(())
    }
}

/// Root IOP recorder error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RootIopError {
    /// Codeword length must be a non-zero power of two.
    InvalidLength(usize),
    /// Codeword/commitment shape mismatch.
    ShapeMismatch,
    /// Oracle id was not found.
    UnknownOracle(usize),
    /// Claim id was not found.
    UnknownClaim(usize),
    /// Opened index is outside the oracle.
    IndexOutOfBounds { oracle_id: usize, index: usize },
    /// Claim id list length did not match opening list length.
    OpeningArityMismatch,
    /// Claim metadata disagreed between WARP and the recorded transcript.
    ClaimMetadataMismatch(usize),
    /// Commitment metadata disagreed between WARP and the recorded transcript.
    CommitmentMismatch(usize),
    /// Claim value disagreed with the recorded oracle.
    ClaimValueMismatch(usize),
    /// Base/extension opening was used against the wrong oracle field.
    OracleFieldMismatch,
    /// A prover recorder was asked to perform verifier-only work.
    ProverUsedAsVerifier,
    /// A verifier recorder was asked to perform prover-only work.
    VerifierUsedAsProver,
}

fn checked_log2_len(len: usize) -> Result<usize, RootIopError> {
    if len == 0 || !len.is_power_of_two() {
        return Err(RootIopError::InvalidLength(len));
    }
    Ok(len.trailing_zeros() as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_baby_bear::BabyBear;
    use p3_challenger::{CanObserve, CanSample, CanSampleBits};
    use p3_commit::{BatchOpening, BatchOpeningRef, Mmcs};
    use p3_field::extension::BinomialExtensionField;
    use p3_matrix::{Dimensions, Matrix};
    use serde::{Deserialize, Serialize};

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    #[derive(Clone, Copy, Debug)]
    struct DummyChallenger;

    impl CanObserve<F> for DummyChallenger {
        fn observe(&mut self, _value: F) {}
    }

    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    struct ToyCommitment(Vec<F>);

    impl CanObserve<ToyCommitment> for DummyChallenger {
        fn observe(&mut self, _value: ToyCommitment) {}
    }

    #[derive(Clone, Debug)]
    struct ToyMmcs;

    impl Mmcs<F> for ToyMmcs {
        type ProverData<M> = Vec<M>;
        type Commitment = ToyCommitment;
        type Proof = ();
        type Error = RootIopError;

        fn commit<M: Matrix<F>>(&self, inputs: Vec<M>) -> (Self::Commitment, Self::ProverData<M>) {
            let mut digest = Vec::new();
            for input in &inputs {
                digest.push(F::from_usize(input.height()));
                digest.push(F::from_usize(input.width()));
                for r in 0..input.height() {
                    for c in 0..input.width() {
                        digest.push(input.get(r, c).unwrap());
                    }
                }
            }
            (ToyCommitment(digest), inputs)
        }

        fn open_batch<M: Matrix<F>>(
            &self,
            index: usize,
            prover_data: &Self::ProverData<M>,
        ) -> BatchOpening<F, Self> {
            let opened_values = prover_data
                .iter()
                .map(|matrix| {
                    let row = index.min(matrix.height() - 1);
                    (0..matrix.width())
                        .map(|col| matrix.get(row, col).unwrap())
                        .collect()
                })
                .collect();
            BatchOpening::new(opened_values, ())
        }

        fn get_matrices<'a, M: Matrix<F>>(
            &self,
            prover_data: &'a Self::ProverData<M>,
        ) -> Vec<&'a M> {
            prover_data.iter().collect()
        }

        fn verify_batch(
            &self,
            _commit: &Self::Commitment,
            _dimensions: &[Dimensions],
            _index: usize,
            _batch_opening: BatchOpeningRef<'_, F, Self>,
        ) -> Result<(), Self::Error> {
            Ok(())
        }
    }

    impl CanSample<F> for DummyChallenger {
        fn sample(&mut self) -> F {
            F::ZERO
        }
    }

    impl CanSampleBits<usize> for DummyChallenger {
        fn sample_bits(&mut self, _bits: usize) -> usize {
            0
        }
    }

    impl FieldChallenger<F> for DummyChallenger {}

    #[test]
    fn records_and_checks_base_and_extension_claims() {
        let prover = RootIopProver::<F, EF>::new(2);
        let fresh = prover
            .commit_fresh_codeword(
                alloc::vec![
                    F::from_u64(3),
                    F::from_u64(4),
                    F::from_u64(5),
                    F::from_u64(6)
                ],
                alloc::vec![F::from_u64(7), F::from_u64(8)],
            )
            .unwrap();
        let (_, fresh_proof) = <RootIopProver<F, EF> as ExternalCodewordBatchOpeningProver<
            F,
            RootIopCommittedCodeword<F>,
        >>::open_batch(&prover, &fresh, &[1, 3])
        .unwrap();
        assert_eq!(fresh_proof.claim_ids, alloc::vec![0, 1]);

        let (acc_commitment, acc_data) =
            <RootIopProver<F, EF> as AccumulatorCommitmentBackend<F, EF, DummyChallenger>>::commit(
                &prover,
                alloc::vec![
                    EF::from_u64(11),
                    EF::from_u64(12),
                    EF::from_u64(13),
                    EF::from_u64(14),
                ],
            )
            .unwrap();
        let (_, acc_proof) = <RootIopProver<F, EF> as AccumulatorBatchOpeningBackend<
            F,
            EF,
            DummyChallenger,
        >>::open_batch(&prover, &acc_data, &[0, 2])
        .unwrap();
        assert_eq!(acc_commitment.oracle_id, 1);
        assert_eq!(acc_proof.claim_ids, alloc::vec![2, 3]);

        let transcript = prover.transcript();
        transcript.verify_witnessed_claim_values().unwrap();
    }

    #[test]
    fn verifier_expected_claims_match_prover_transcript() {
        let prover = RootIopProver::<F, EF>::new(2);
        let fresh = prover
            .commit_fresh_codeword(
                alloc::vec![
                    F::from_u64(1),
                    F::from_u64(2),
                    F::from_u64(3),
                    F::from_u64(4)
                ],
                alloc::vec![F::from_u64(1)],
            )
            .unwrap();
        let (values, proof) = <RootIopProver<F, EF> as ExternalCodewordBatchOpeningProver<
            F,
            RootIopCommittedCodeword<F>,
        >>::open_batch(&prover, &fresh, &[0, 2])
        .unwrap();

        let verifier = RootIopVerifier::<F, EF>::new(2);
        <RootIopVerifier<F, EF> as ExternalCodewordBatchOpeningVerifier<
            F,
            DummyChallenger,
        >>::verify_batch_opening(&verifier, &fresh.commitment(), 2, &[0, 2], &values, &proof)
        .unwrap();

        let transcript = prover.transcript();
        verifier.verify_against_transcript(&transcript).unwrap();
        transcript.verify_witnessed_claim_values().unwrap();

        let proof = WitnessRootIopProof::<F, EF>::prove(&transcript).unwrap();
        WitnessRootIopProof::<F, EF>::verify(&verifier.expected_claims(), &proof).unwrap();
    }

    #[test]
    fn bound_recorder_binds_real_commitments_and_claim_ids() {
        let prover = RootIopBoundProver::<F, EF, ToyMmcs>::new(ToyMmcs, 2);
        let fresh = prover
            .commit_fresh_codeword(
                alloc::vec![
                    F::from_u64(1),
                    F::from_u64(2),
                    F::from_u64(3),
                    F::from_u64(4)
                ],
                alloc::vec![F::from_u64(1)],
            )
            .unwrap();
        let (fresh_values, fresh_proof) =
            <RootIopBoundProver<F, EF, ToyMmcs> as ExternalCodewordBatchOpeningProver<
                F,
                RootIopBoundCommittedCodeword<F, ToyCommitment>,
            >>::open_batch(&prover, &fresh, &[1, 3])
            .unwrap();

        let (acc_commitment, acc_data) =
            <RootIopBoundProver<F, EF, ToyMmcs> as AccumulatorCommitmentBackend<
                F,
                EF,
                DummyChallenger,
            >>::commit(
                &prover,
                alloc::vec![
                    EF::from_u64(11),
                    EF::from_u64(12),
                    EF::from_u64(13),
                    EF::from_u64(14),
                ],
            )
            .unwrap();
        let (acc_values, acc_proof) =
            <RootIopBoundProver<F, EF, ToyMmcs> as AccumulatorBatchOpeningBackend<
                F,
                EF,
                DummyChallenger,
            >>::open_batch(&prover, &acc_data, &[0, 2])
            .unwrap();

        let verifier = RootIopBoundVerifier::<F, EF, ToyCommitment>::new(2);
        let mut challenger = DummyChallenger;
        <RootIopBoundVerifier<F, EF, ToyCommitment> as ExternalCodewordOpeningVerifier<
            F,
            DummyChallenger,
        >>::observe_commitment(&verifier, &mut challenger, &fresh.commitment());
        <RootIopBoundVerifier<F, EF, ToyCommitment> as ExternalCodewordBatchOpeningVerifier<
            F,
            DummyChallenger,
        >>::verify_batch_opening(
            &verifier,
            &fresh.commitment(),
            2,
            &[1, 3],
            &fresh_values,
            &fresh_proof,
        )
        .unwrap();
        <RootIopBoundVerifier<F, EF, ToyCommitment> as AccumulatorCommitmentBackend<
            F,
            EF,
            DummyChallenger,
        >>::observe_commitment(&verifier, &mut challenger, &acc_commitment);
        <RootIopBoundVerifier<F, EF, ToyCommitment> as AccumulatorBatchOpeningBackend<
            F,
            EF,
            DummyChallenger,
        >>::verify_batch_opening(
            &verifier,
            &acc_commitment,
            2,
            &[0, 2],
            &acc_values,
            &acc_proof,
        )
        .unwrap();

        let transcript = prover.transcript();
        transcript.verify_witnessed_claim_values().unwrap();
        verifier.verify_against_transcript(&transcript).unwrap();
        let proof = WitnessRootIopBoundProof::<F, EF, ToyCommitment>::prove(&transcript).unwrap();
        WitnessRootIopBoundProof::<F, EF, ToyCommitment>::verify(
            &verifier.expected_commitments(),
            &verifier.expected_claims(),
            &proof,
        )
        .unwrap();
        assert_eq!(verifier.expected_commitments().len(), 2);
    }

    #[test]
    fn witnessed_claim_check_rejects_tampered_value() {
        let prover = RootIopProver::<F, EF>::new(1);
        let fresh = prover
            .commit_fresh_codeword(
                alloc::vec![F::from_u64(1), F::from_u64(2)],
                alloc::vec![F::from_u64(1)],
            )
            .unwrap();
        let _ = <RootIopProver<F, EF> as ExternalCodewordOpeningProver<
            F,
            RootIopCommittedCodeword<F>,
        >>::open(&prover, &fresh, 1)
        .unwrap();
        let mut transcript = prover.transcript();
        transcript.claims[0].value = RootIopOpeningValue::Base(F::from_u64(9));
        assert_eq!(
            transcript.verify_witnessed_claim_values(),
            Err(RootIopError::ClaimValueMismatch(0))
        );
    }
}
