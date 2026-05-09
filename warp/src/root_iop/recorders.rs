//! Prover/verifier recorders for WARP's root IOP transcript.
//!
//! The recorders are deliberately thin state machines. Prover-side recorders
//! append committed oracle values and claim ids, while verifier-side recorders
//! replay WARP and collect the commitments/claims that the final WHIR compiler
//! must authenticate.

use alloc::vec::Vec;
use core::cell::RefCell;

use p3_commit::{ExtensionMmcs, Mmcs};
use p3_field::{ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrix;

use super::{
    RootIopBoundCommitment, RootIopBoundCommittedCodeword, RootIopBoundTranscript,
    RootIopCommitment, RootIopCommittedCodeword, RootIopError, RootIopOpeningClaim,
    RootIopOpeningPoint, RootIopOpeningValue, RootIopOracleField, RootIopOracleValues,
    RootIopTranscript, checked_log2_len,
};

#[derive(Clone, Debug, Default)]
pub(super) struct RootIopState<F, EF> {
    pub(super) transcript: RootIopTranscript<F, EF>,
}

#[derive(Clone, Debug)]
pub(super) struct RootIopBoundState<F, EF, Comm> {
    pub(super) transcript: RootIopBoundTranscript<F, EF, Comm>,
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
    pub(super) fn push_oracle(
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

    pub(super) fn push_claim(
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
    pub(super) fn push_oracle(
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

    pub(super) fn push_claim(
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
    pub(super) log_codeword_len: usize,
    pub(super) state: RefCell<RootIopState<F, EF>>,
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
    pub(super) log_codeword_len: usize,
    pub(super) expected_claims: RefCell<Vec<RootIopOpeningClaim<F, EF>>>,
}

/// Prover-side root IOP recorder backed by real Plonky3 commitments.
#[derive(Debug)]
pub struct RootIopBoundProver<F, EF, MT>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
{
    pub(super) mmcs: MT,
    pub(super) ext_mmcs: ExtensionMmcs<F, EF, MT>,
    pub(super) log_codeword_len: usize,
    pub(super) state: RefCell<RootIopBoundState<F, EF, MT::Commitment>>,
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
    pub(super) log_codeword_len: usize,
    pub(super) expected_commitments: RefCell<Vec<RootIopBoundCommitment<Comm>>>,
    pub(super) expected_claims: RefCell<Vec<RootIopOpeningClaim<F, EF>>>,
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

    pub(super) fn record_commitment(&self, commitment: &RootIopBoundCommitment<Comm>) {
        let mut expected = self.expected_commitments.borrow_mut();
        if !expected
            .iter()
            .any(|known| known.oracle_id == commitment.oracle_id)
        {
            expected.push(commitment.clone());
        }
    }

    pub(super) fn record_expected_claim(
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

    pub(super) fn record_expected_claim(
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
