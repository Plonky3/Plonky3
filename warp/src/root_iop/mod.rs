//! Root IOP opening transcript for a single WARP proof.
//!
//! WARP's root protocol is an IOR/IOP transcript: commitments are bound before
//! the verifier samples challenges, and later verifier queries open those
//! committed oracles. Earlier experimental paths authenticated each opening
//! separately. This module isolates the reusable layer we need instead: WARP
//! records all openings as typed claims, so a single compiler/proximity proof
//! can authenticate those claims together.
//!
//! The recorder in this module is deliberately not the final succinct proof.
//! It is the boundary between the WARP root IOP and a WHIR-style compiler as
//! in WHIR Construction 7.4: first record the linear IOP transcript, then prove
//! the recorded oracle claims with one batched proximity layer.

use alloc::vec::Vec;

use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Mmcs, MultilinearOpenedValues};
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use serde::Serialize;

use crate::finalize::AccumulatorPointOpeningBackend;
use crate::protocol::{
    AccumulatorBatchOpeningBackend, AccumulatorCommitmentBackend,
    ExternalCodewordBatchOpeningProver, ExternalCodewordBatchOpeningVerifier,
    ExternalCodewordOpeningProver, ExternalCodewordOpeningVerifier, ExternalCommittedCodeword,
};

mod error;
mod recorders;
mod types;
mod witness;

use error::checked_log2_len;

pub use error::RootIopError;
pub use recorders::{RootIopBoundProver, RootIopBoundVerifier, RootIopProver, RootIopVerifier};
pub use types::*;
pub use witness::{
    RootIopBoundProofSystem, RootIopProofSystem, WitnessRootIopBoundProof, WitnessRootIopProof,
};

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
            RootIopOpeningPoint::RsCodewordIndex(index),
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
                RootIopOpeningPoint::RsCodewordIndex(index),
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
            RootIopOpeningPoint::RsCodewordIndex(index),
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
                RootIopOpeningPoint::RsCodewordIndex(index),
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
            RootIopOpeningPoint::RsCodewordIndex(index),
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
                RootIopOpeningPoint::RsCodewordIndex(index),
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
            RootIopOpeningPoint::RsCodewordIndex(index),
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
                RootIopOpeningPoint::RsCodewordIndex(index),
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
            RootIopOpeningPoint::RsCodewordIndex(index),
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
                RootIopOpeningPoint::RsCodewordIndex(index),
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
            RootIopOpeningPoint::RsCodewordIndex(index),
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
                RootIopOpeningPoint::RsCodewordIndex(index),
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
            RootIopOpeningPoint::RsCodewordIndex(index),
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
                RootIopOpeningPoint::RsCodewordIndex(index),
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
            RootIopOpeningPoint::RsCodewordIndex(index),
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
                RootIopOpeningPoint::RsCodewordIndex(index),
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

#[cfg(test)]
mod tests;
