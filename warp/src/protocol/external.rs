//! External committed-codeword interface for WARP fresh inputs.
//!
//! WARP's alphabet-`F` variant only needs three things from each fresh input:
//! the committed RS codeword, the underlying witness/message, and authenticated
//! openings at sampled shift-query indices. This module isolates those
//! operations from Plonky3's [`Mmcs`] so upstream PCS implementations can feed
//! WARP without being coerced into the wrong commitment abstraction.

use alloc::vec;
use alloc::vec::Vec;
use core::convert::Infallible;
use core::fmt::Debug;

use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{BatchOpeningRef, ExtensionMmcs, Mmcs};
use p3_field::{ExtensionField, Field};
use p3_matrix::Dimensions;
use p3_matrix::dense::RowMajorMatrix;
use serde::Serialize;
use serde::de::DeserializeOwned;

use super::prover::CommittedCodeword;

/// Verifier adapter for fresh inputs committed with a Plonky3 [`Mmcs`].
#[derive(Clone, Copy, Debug)]
pub struct MmcsExternalOpeningVerifier<'a, MT> {
    mmcs: &'a MT,
}

impl<'a, MT> MmcsExternalOpeningVerifier<'a, MT> {
    /// Create a verifier adapter over an existing `Mmcs`.
    pub fn new(mmcs: &'a MT) -> Self {
        Self { mmcs }
    }
}

/// A fresh WARP codeword that has already been committed by an upstream PCS.
pub trait ExternalCommittedCodeword<F: Field> {
    /// Verifier-visible commitment/claim for this codeword.
    type Commitment: Clone + Serialize + DeserializeOwned;

    /// Return the commitment/claim that must be bound before WARP samples
    /// challenges.
    fn commitment(&self) -> Self::Commitment;

    /// Committed RS codeword, flattened as the vector used by WARP.
    fn codeword(&self) -> &[F];

    /// Underlying witness/message matched to `codeword`.
    fn witness(&self) -> &[F];
}

/// Transcript binding for an external commitment.
///
/// This is separate from [`ExternalCommittedCodeword`] because different
/// commitment systems bind different verifier-visible metadata. A plain
/// Plonky3 Merkle commitment only observes the root/cap; other systems may
/// also bind matrix shape, trace metadata, and public values.
pub trait ExternalCommitmentObserver<F, Challenger>: ExternalCommittedCodeword<F>
where
    F: Field,
    Challenger: FieldChallenger<F>,
{
    /// Absorb this codeword's verifier-visible claim into the transcript.
    fn observe_commitment(&self, challenger: &mut Challenger);
}

/// Prover-side opening backend for an external committed codeword.
pub trait ExternalCodewordOpeningProver<F, C>
where
    F: Field,
    C: ExternalCommittedCodeword<F>,
{
    /// Opening proof produced by the external PCS.
    type Proof: Clone + Serialize + DeserializeOwned;
    /// Opening error.
    type Error: Debug;

    /// Open the flattened codeword at `index`, returning the scalar WARP
    /// answer and the external PCS proof authenticating it.
    fn open(&self, committed: &C, index: usize) -> Result<(F, Self::Proof), Self::Error>;
}

/// Verifier-side opening backend for an external committed codeword.
pub trait ExternalCodewordOpeningVerifier<F, Challenger>
where
    F: Field,
    Challenger: FieldChallenger<F>,
{
    /// Verifier-visible commitment/claim for the external PCS.
    type Commitment: Clone + Serialize + DeserializeOwned;
    /// Opening proof produced by the external PCS.
    type Proof: Clone + Serialize + DeserializeOwned;
    /// Verification error.
    type Error: Debug;

    /// Absorb `commitment` into the transcript in the same order used by the
    /// prover.
    fn observe_commitment(&self, challenger: &mut Challenger, commitment: &Self::Commitment);

    /// Verify that `value` is the flattened-codeword entry at `index`.
    fn verify_opening(
        &self,
        commitment: &Self::Commitment,
        log_codeword_len: usize,
        index: usize,
        value: F,
        proof: &Self::Proof,
    ) -> Result<(), Self::Error>;
}

/// Prover-side backend for opening one external codeword at many shift
/// indices in one proof.
pub trait ExternalCodewordBatchOpeningProver<F, C>: ExternalCodewordOpeningProver<F, C>
where
    F: Field,
    C: ExternalCommittedCodeword<F>,
{
    /// Batched opening proof produced by the external PCS.
    type BatchProof: Clone + Serialize + DeserializeOwned;

    /// Open the flattened codeword at every `index`, returning the scalar WARP
    /// answers in the same order and one external PCS proof authenticating all
    /// of them.
    fn open_batch(
        &self,
        committed: &C,
        indices: &[usize],
    ) -> Result<(Vec<F>, Self::BatchProof), Self::Error>;

    /// Open an owned committed codeword at every `index`.
    ///
    /// Backends whose proof data is expensive to clone can override this to
    /// move that proof data into the PCS prover. The default preserves the
    /// by-reference behavior.
    fn open_batch_owned(
        &self,
        committed: C,
        indices: &[usize],
    ) -> Result<(Vec<F>, Self::BatchProof), Self::Error> {
        self.open_batch(&committed, indices)
    }
}

/// Verifier-side backend for checking many openings of one external codeword
/// in one proof.
pub trait ExternalCodewordBatchOpeningVerifier<F, Challenger>:
    ExternalCodewordOpeningVerifier<F, Challenger>
where
    F: Field,
    Challenger: FieldChallenger<F>,
{
    /// Batched opening proof produced by the external PCS.
    type BatchProof: Clone + Serialize + DeserializeOwned;

    /// Verify that `values[i]` is the flattened-codeword entry at `indices[i]`.
    fn verify_batch_opening(
        &self,
        commitment: &Self::Commitment,
        log_codeword_len: usize,
        indices: &[usize],
        values: &[F],
        proof: &Self::BatchProof,
    ) -> Result<(), Self::Error>;
}

/// Commitment/opening backend for WARP accumulator codewords.
///
/// Fresh inputs may arrive from one PCS while the merged WARP accumulator uses
/// another commitment format. This trait isolates the accumulator side so the
/// same WARP algebra can use either Plonky3's local [`ExtensionMmcs`] or a
/// backend supplied by the root proof compiler.
pub trait AccumulatorCommitmentBackend<F, EF, Challenger>
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F>,
{
    /// Verifier-visible accumulator commitment/claim.
    type Commitment: Clone + Serialize + DeserializeOwned;
    /// Prover-side data needed to reopen the accumulator codeword.
    type ProverData;
    /// Opening proof for one accumulator codeword entry.
    type Proof: Clone + Serialize + DeserializeOwned;
    /// Backend error.
    type Error: Debug;

    /// Commit the merged accumulator codeword.
    fn commit(
        &self,
        codeword: Vec<EF>,
    ) -> Result<(Self::Commitment, Self::ProverData), Self::Error>;

    /// Commit the merged accumulator codeword when the corresponding RS message
    /// is already available to the prover.
    ///
    /// The default implementation preserves existing backends. WHIR-backed
    /// single-RS root compilers can override this to avoid decoding a systematic
    /// message from the same codeword again.
    fn commit_with_message(
        &self,
        codeword: Vec<EF>,
        _message: &[EF],
    ) -> Result<(Self::Commitment, Self::ProverData), Self::Error> {
        self.commit(codeword)
    }

    /// Open the committed accumulator codeword at `index`.
    fn open(
        &self,
        prover_data: &Self::ProverData,
        index: usize,
    ) -> Result<(EF, Self::Proof), Self::Error>;

    /// Absorb this accumulator commitment into the Fiat-Shamir transcript.
    fn observe_commitment(&self, challenger: &mut Challenger, commitment: &Self::Commitment);

    /// Verify one accumulator opening.
    fn verify_opening(
        &self,
        commitment: &Self::Commitment,
        log_codeword_len: usize,
        index: usize,
        value: EF,
        proof: &Self::Proof,
    ) -> Result<(), Self::Error>;
}

/// Batched opening backend for accumulator codewords.
pub trait AccumulatorBatchOpeningBackend<F, EF, Challenger>:
    AccumulatorCommitmentBackend<F, EF, Challenger>
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F>,
{
    /// Batched opening proof for one accumulator codeword at many indices.
    type BatchProof: Clone + Serialize + DeserializeOwned;

    /// Open the committed accumulator codeword at all `indices`.
    fn open_batch(
        &self,
        prover_data: &Self::ProverData,
        indices: &[usize],
    ) -> Result<(Vec<EF>, Self::BatchProof), Self::Error>;

    /// Verify all openings of one accumulator codeword.
    fn verify_batch_opening(
        &self,
        commitment: &Self::Commitment,
        log_codeword_len: usize,
        indices: &[usize],
        values: &[EF],
        proof: &Self::BatchProof,
    ) -> Result<(), Self::Error>;
}

impl<F, MT> ExternalCommittedCodeword<F> for CommittedCodeword<F, MT>
where
    F: Field,
    MT: Mmcs<F>,
{
    type Commitment = MT::Commitment;

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

impl<F, EF, MT, Challenger> AccumulatorCommitmentBackend<F, EF, Challenger>
    for ExtensionMmcs<F, EF, MT>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
    Challenger: FieldChallenger<F> + CanObserve<MT::Commitment>,
{
    type Commitment = MT::Commitment;
    type ProverData = <ExtensionMmcs<F, EF, MT> as Mmcs<EF>>::ProverData<RowMajorMatrix<EF>>;
    type Proof = MT::Proof;
    type Error = MT::Error;

    fn commit(
        &self,
        codeword: Vec<EF>,
    ) -> Result<(Self::Commitment, Self::ProverData), Self::Error> {
        Ok(self.commit_matrix(RowMajorMatrix::new(codeword, 1)))
    }

    fn open(
        &self,
        prover_data: &Self::ProverData,
        index: usize,
    ) -> Result<(EF, Self::Proof), Self::Error> {
        let opening = <ExtensionMmcs<F, EF, MT> as Mmcs<EF>>::open_batch(self, index, prover_data);
        let (mut vals, proof) = opening.unpack();
        Ok((vals.swap_remove(0)[0], proof))
    }

    fn observe_commitment(&self, challenger: &mut Challenger, commitment: &Self::Commitment) {
        challenger.observe(commitment.clone());
    }

    fn verify_opening(
        &self,
        commitment: &Self::Commitment,
        log_codeword_len: usize,
        index: usize,
        value: EF,
        proof: &Self::Proof,
    ) -> Result<(), Self::Error> {
        let dims = vec![Dimensions {
            height: 1usize << log_codeword_len,
            width: 1,
        }];
        let opened = vec![vec![value]];
        self.verify_batch(
            commitment,
            &dims,
            index,
            BatchOpeningRef::new(&opened, proof),
        )
    }
}

impl<F, EF, MT, Challenger> AccumulatorBatchOpeningBackend<F, EF, Challenger>
    for ExtensionMmcs<F, EF, MT>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
    MT::Proof: Clone + Serialize + DeserializeOwned,
    Challenger: FieldChallenger<F> + CanObserve<MT::Commitment>,
{
    type BatchProof = Vec<MT::Proof>;

    fn open_batch(
        &self,
        prover_data: &Self::ProverData,
        indices: &[usize],
    ) -> Result<(Vec<EF>, Self::BatchProof), Self::Error> {
        let mut values = Vec::with_capacity(indices.len());
        let mut proofs = Vec::with_capacity(indices.len());
        for &index in indices {
            let (value, proof) = <Self as AccumulatorCommitmentBackend<F, EF, Challenger>>::open(
                self,
                prover_data,
                index,
            )?;
            values.push(value);
            proofs.push(proof);
        }
        Ok((values, proofs))
    }

    fn verify_batch_opening(
        &self,
        commitment: &Self::Commitment,
        log_codeword_len: usize,
        indices: &[usize],
        values: &[EF],
        proof: &Self::BatchProof,
    ) -> Result<(), Self::Error> {
        assert_eq!(indices.len(), values.len());
        assert_eq!(indices.len(), proof.len());
        for ((&index, &value), proof) in indices.iter().zip(values.iter()).zip(proof.iter()) {
            <Self as AccumulatorCommitmentBackend<F, EF, Challenger>>::verify_opening(
                self,
                commitment,
                log_codeword_len,
                index,
                value,
                proof,
            )?;
        }
        Ok(())
    }
}

impl<F, MT, Challenger> ExternalCommitmentObserver<F, Challenger> for CommittedCodeword<F, MT>
where
    F: Field,
    MT: Mmcs<F>,
    Challenger: FieldChallenger<F> + CanObserve<MT::Commitment>,
{
    fn observe_commitment(&self, challenger: &mut Challenger) {
        challenger.observe(self.commitment.clone());
    }
}

impl<F, MT> ExternalCodewordOpeningProver<F, CommittedCodeword<F, MT>> for MT
where
    F: Field,
    MT: Mmcs<F>,
{
    type Proof = MT::Proof;
    type Error = Infallible;

    fn open(
        &self,
        committed: &CommittedCodeword<F, MT>,
        index: usize,
    ) -> Result<(F, Self::Proof), Self::Error> {
        let opening = <MT as Mmcs<F>>::open_batch(self, index, &committed.prover_data);
        let (mut vals, proof) = opening.unpack();
        Ok((vals.swap_remove(0)[0], proof))
    }
}

impl<F, MT> ExternalCodewordBatchOpeningProver<F, CommittedCodeword<F, MT>> for MT
where
    F: Field,
    MT: Mmcs<F>,
    MT::Proof: Clone + Serialize + DeserializeOwned,
{
    type BatchProof = Vec<MT::Proof>;

    fn open_batch(
        &self,
        committed: &CommittedCodeword<F, MT>,
        indices: &[usize],
    ) -> Result<(Vec<F>, Self::BatchProof), Self::Error> {
        let mut values = Vec::with_capacity(indices.len());
        let mut proofs = Vec::with_capacity(indices.len());
        for &index in indices {
            let (value, proof) = <Self as ExternalCodewordOpeningProver<
                F,
                CommittedCodeword<F, MT>,
            >>::open(self, committed, index)?;
            values.push(value);
            proofs.push(proof);
        }
        Ok((values, proofs))
    }
}

impl<F, MT, Challenger> ExternalCodewordOpeningVerifier<F, Challenger>
    for MmcsExternalOpeningVerifier<'_, MT>
where
    F: Field,
    MT: Mmcs<F>,
    Challenger: FieldChallenger<F> + CanObserve<MT::Commitment>,
{
    type Commitment = MT::Commitment;
    type Proof = MT::Proof;
    type Error = MT::Error;

    fn observe_commitment(&self, challenger: &mut Challenger, commitment: &Self::Commitment) {
        challenger.observe(commitment.clone());
    }

    fn verify_opening(
        &self,
        commitment: &Self::Commitment,
        log_codeword_len: usize,
        index: usize,
        value: F,
        proof: &Self::Proof,
    ) -> Result<(), Self::Error> {
        let dims = vec![Dimensions {
            height: 1usize << log_codeword_len,
            width: 1,
        }];
        let opened = vec![vec![value]];
        self.mmcs.verify_batch(
            commitment,
            &dims,
            index,
            BatchOpeningRef::new(&opened, proof),
        )
    }
}

impl<F, MT, Challenger> ExternalCodewordBatchOpeningVerifier<F, Challenger>
    for MmcsExternalOpeningVerifier<'_, MT>
where
    F: Field,
    MT: Mmcs<F>,
    MT::Proof: Clone + Serialize + DeserializeOwned,
    Challenger: FieldChallenger<F> + CanObserve<MT::Commitment>,
{
    type BatchProof = Vec<MT::Proof>;

    fn verify_batch_opening(
        &self,
        commitment: &Self::Commitment,
        log_codeword_len: usize,
        indices: &[usize],
        values: &[F],
        proof: &Self::BatchProof,
    ) -> Result<(), Self::Error> {
        assert_eq!(indices.len(), values.len());
        assert_eq!(indices.len(), proof.len());
        for ((&index, &value), proof) in indices.iter().zip(values.iter()).zip(proof.iter()) {
            <Self as ExternalCodewordOpeningVerifier<F, Challenger>>::verify_opening(
                self,
                commitment,
                log_codeword_len,
                index,
                value,
                proof,
            )?;
        }
        Ok(())
    }
}
