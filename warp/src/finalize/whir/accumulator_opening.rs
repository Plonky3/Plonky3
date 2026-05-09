//! Final accumulator opening protocol for WHIR-backed WARP finalization.
//!
//! This module proves the WARP decider claim `f_hat(alpha) = mu` against the
//! accumulator commitment already produced by the accumulation chain. It is
//! intentionally fail-closed: proving recomputes the PCS commitment to the
//! final accumulator witness and requires it to equal the public `instance.rt`.
//! A proof for a fresh unrelated commitment is rejected before any opening is
//! returned.

use super::*;

/// Precommitted opening prover/verifier for the final accumulator codeword.
///
/// The type is generic over the PCS because the soundness requirement is about
/// the trait contract, not a particular implementation detail: `commit(f)`
/// must produce the same public commitment type and layout as `acc.x.rt`.
/// Once `p3-whir` exposes that precommitted layout for WARP's RS oracle, it can
/// instantiate this layer directly.
pub struct WhirAccumulatorOpeningProtocol<'a, F, EF, Pcs, Challenger, Dft>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    Dft: TwoAdicSubgroupDft<F>,
{
    pcs: &'a Pcs,
    code: &'a ReedSolomonCode<F, Dft>,
    challenger_seed: Challenger,
    _ph: PhantomData<EF>,
}

impl<'a, F, EF, Pcs, Challenger, Dft>
    WhirAccumulatorOpeningProtocol<'a, F, EF, Pcs, Challenger, Dft>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    Dft: TwoAdicSubgroupDft<F>,
{
    /// Create a new opening protocol from a compatible PCS and transcript seed.
    ///
    /// `challenger_seed` is cloned for proving and verification, so callers
    /// should pass the same transcript state that should prefix this final
    /// opening protocol.
    pub const fn new(
        pcs: &'a Pcs,
        code: &'a ReedSolomonCode<F, Dft>,
        challenger_seed: Challenger,
    ) -> Self {
        Self {
            pcs,
            code,
            challenger_seed,
            _ph: PhantomData,
        }
    }
}

impl<'a, F, EF, Pcs, Challenger, Dft>
    WhirAccumulatorOpeningProtocol<'a, F, EF, Pcs, Challenger, Dft>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Pcs: MultilinearPcs<EF, Challenger, Val = EF>,
    Pcs::Commitment: PartialEq,
    Challenger: Clone,
    Dft: TwoAdicSubgroupDft<F>,
{
    /// Prove `f_hat(alpha) = mu` for the accumulator's committed final
    /// codeword.
    ///
    /// This is fail-closed:
    /// - `code` must be systematic, since the later PESAT terminal openings
    ///   depend on the same message-subspace layout.
    /// - `pcs.num_vars()` must match the codeword MLE dimension.
    /// - `pcs.commit(witness.f)` must equal `instance.rt`.
    /// - the opened PCS value must equal `instance.mu`.
    pub fn prove<ProverData>(
        &self,
        instance: &AccumulatorInstance<EF, Pcs::Commitment>,
        witness: &AccumulatorWitness<EF, ProverData>,
    ) -> Result<WhirAccumulatorOpeningProof<Pcs::Proof>, FinalizerError> {
        self.validate_instance_shape(instance)?;
        if witness.f.len() != self.code.codeword_len() {
            return Err(FinalizerError::Decider(DeciderError::EncodingMismatch));
        }

        let opening_points = self.opening_points(instance);
        let mut challenger = self.challenger_seed.clone();
        let evaluations = RowMajorMatrix::new(witness.f.clone(), 1);
        let (commitment, prover_data) =
            self.pcs
                .commit(evaluations, &opening_points, &mut challenger);
        if commitment != instance.rt {
            return Err(FinalizerError::Decider(DeciderError::MerkleRoot));
        }

        let (opened_values, pcs_proof) = self.pcs.open(prover_data, &mut challenger);
        let opened_mu = opened_values
            .first()
            .and_then(|poly_values| poly_values.first())
            .copied()
            .ok_or(FinalizerError::Unsupported(
                "PCS did not return the accumulator opening value",
            ))?;
        if opened_mu != instance.mu {
            return Err(FinalizerError::Decider(DeciderError::MlEval));
        }

        Ok(WhirAccumulatorOpeningProof { pcs_proof })
    }

    /// Verify the accumulator codeword opening proof against `acc.x.rt`.
    pub fn verify(
        &self,
        instance: &AccumulatorInstance<EF, Pcs::Commitment>,
        proof: &WhirAccumulatorOpeningProof<Pcs::Proof>,
    ) -> Result<(), FinalizerError> {
        self.validate_instance_shape(instance)?;
        let opening_claims = [vec![(Point::new(instance.alpha.clone()), instance.mu)]];
        let mut challenger = self.challenger_seed.clone();
        self.pcs
            .verify(
                &instance.rt,
                &opening_claims,
                &proof.pcs_proof,
                &mut challenger,
            )
            .map_err(|err| FinalizerError::OpeningProof(format!("{err:?}")))
    }

    fn validate_instance_shape(
        &self,
        instance: &AccumulatorInstance<EF, Pcs::Commitment>,
    ) -> Result<(), FinalizerError> {
        if !self.code.is_systematic() {
            return Err(FinalizerError::Unsupported(
                "WHIR-native WARP finalization requires systematic RS encoding",
            ));
        }
        if self.pcs.num_vars() != self.code.log_codeword_len() {
            return Err(FinalizerError::Unsupported(
                "PCS variable count must match the WARP codeword MLE dimension",
            ));
        }
        if instance.alpha.len() != self.code.log_codeword_len() {
            return Err(FinalizerError::Decider(DeciderError::MlEval));
        }
        Ok(())
    }

    fn opening_points(
        &self,
        instance: &AccumulatorInstance<EF, Pcs::Commitment>,
    ) -> [Vec<Point<EF>>; 1] {
        [vec![Point::new(instance.alpha.clone())]]
    }
}
