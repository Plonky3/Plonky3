//! BitZ as a polynomial commitment of its own: commit to bits, prove linear claims over `F` on them.

use alloc::vec::Vec;

use p3_binary_pcs::whir::Region;
use p3_binary_pcs::whir::preset::keccak64::{self, Commitment, ProverData, WitnessCommitment};
use p3_field::PrimeField64;
use tracing::instrument;

use crate::Challenger;
use crate::claim::{ClaimBinding, TensorClaim};
use crate::error::{ParamsError, ProvingError, VerificationError};
use crate::params::{BitShape, BitzParams};
use crate::proof::BitzProof;
use crate::reduce::Bitz;

/// The bits of one shape under their own commitment, opened at public linear claims over `F`.
pub struct BitzPcs<F> {
    /// The reduction to one claim on the commitment.
    bitz: Bitz<F>,
    /// The commitment, opening that one claim.
    commitment: WitnessCommitment,
}

impl<F: PrimeField64> BitzPcs<F> {
    /// The commitment for a shape over `F`, checked to reach `params.commitment.security_bits` for one claim.
    ///
    /// # Errors
    ///
    /// A fold could overflow the exponent, the commitment admits no schedule, or the composed error misses the target.
    pub fn new(shape: BitShape, params: &BitzParams) -> Result<Self, ParamsError> {
        let pcs = Self {
            bitz: Bitz::new(shape, params.inner_pow_bits)?,
            commitment: keccak64::witness_commitment(shape.num_variables(), &params.commitment, 1)?,
        };
        let target = params.commitment.security_bits;
        let achieved = pcs.security_bits(1);
        if achieved < target as f64 {
            return Err(ParamsError::Security { target, achieved });
        }
        Ok(pcs)
    }

    /// The reduction.
    #[must_use]
    pub const fn bitz(&self) -> &Bitz<F> {
        &self.bitz
    }

    /// The commitment.
    #[must_use]
    pub const fn commitment(&self) -> &WitnessCommitment {
        &self.commitment
    }

    /// Bits of the union bound over the opening and the reduction of `claims` claims.
    ///
    /// Every reduction draw follows the commitment, so each is charged over the candidates it leaves (Remark A.2).
    #[must_use]
    pub fn security_bits(&self, claims: usize) -> f64 {
        self.commitment
            .pcs()
            .readings_security(1, false)
            .map_or(0.0, |mut security| {
                for term in self.bitz.security_terms(claims) {
                    security.charge_reduction(term);
                }
                security.error().bits()
            })
    }

    /// Commit to the packed words of the shape, bit `j` of word `w` being bit `64 w + j`, and observe the commitment.
    ///
    /// # Errors
    ///
    /// The words do not hold the shape's bits.
    pub fn commit(
        &self,
        words: Vec<u64>,
        challenger: &mut Challenger,
    ) -> Result<(Commitment, ProverData), ProvingError> {
        Ok(self.commitment.commit(words, challenger)?)
    }

    /// Prove every claim on the committed bits.
    ///
    /// # Errors
    ///
    /// A claim does not fit the shape, or the bits do not satisfy it.
    #[instrument(skip_all, name = "bitz prove")]
    pub fn prove(
        &self,
        data: ProverData,
        claims: &[TensorClaim<F>],
        challenger: &mut Challenger,
    ) -> Result<BitzProof, ProvingError> {
        let (reduction, claim) = self.bitz.reduce(
            claims,
            ClaimBinding::Public,
            &self.commitment,
            self.whole(),
            &data,
            challenger,
        )?;
        let opening = self.commitment.open(
            data,
            core::slice::from_ref(&claim.claim),
            &[claim.split.as_ref()],
            challenger,
        )?;
        Ok(BitzProof { reduction, opening })
    }

    /// Check a proof of every claim against a commitment, observing it first as committing did.
    ///
    /// # Errors
    ///
    /// Any check of the reduction or the opening fails.
    #[instrument(skip_all, name = "bitz verify")]
    pub fn verify(
        &self,
        commitment: &Commitment,
        claims: &[TensorClaim<F>],
        proof: &BitzProof,
        challenger: &mut Challenger,
    ) -> Result<(), VerificationError> {
        self.commitment.observe(commitment, challenger);
        let claim = self.bitz.verify_reduction(
            claims,
            ClaimBinding::Public,
            &self.commitment,
            self.whole(),
            &proof.reduction,
            challenger,
        )?;
        Ok(self
            .commitment
            .verify(commitment, &[claim], &proof.opening, challenger)?)
    }

    /// The whole commitment as the region.
    const fn whole(&self) -> Region {
        Region {
            log_bits: self.commitment.num_variables(),
            offset_bits: 0,
        }
    }
}
