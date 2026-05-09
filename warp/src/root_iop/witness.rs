//! Linear witness proofs for the root IOP recorder.
//!
//! These types are intentionally not the final succinct WARP proof. They are
//! a transparent, linear-size checker for the transcript boundary: every
//! claimed root opening is recomputed directly from the recorded oracle values.
//! Tests use this layer to make sure WARP records the right claims before the
//! WHIR compiler replaces the witness with a succinct proximity proof.

use core::fmt::Debug;

use p3_field::{ExtensionField, Field};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use serde::{Deserialize, Serialize};

use super::{
    RootIopBoundCommitment, RootIopBoundTranscript, RootIopError, RootIopOpeningClaim,
    RootIopOpeningPoint, RootIopOpeningValue, RootIopOracleValues, RootIopTranscript,
};

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
/// with [`super::RootIopBoundVerifier`] to obtain `expected_commitments` and
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
            RootIopOpeningPoint::Index(index) | RootIopOpeningPoint::RsCodewordIndex(index),
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
            RootIopOpeningPoint::Index(index) | RootIopOpeningPoint::RsCodewordIndex(index),
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
