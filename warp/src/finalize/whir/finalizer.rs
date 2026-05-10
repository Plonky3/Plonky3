//! WHIR-facing finalizer protocols for WARP decider checks.
//!
//! This module proves the two final WARP conditions against one accumulator
//! commitment: the accumulator opening `f_hat(alpha) = mu` and the Boolean
//! PESAT decider claim `Pb(beta, C^{-1}(f)) = eta`. The Boolean path relies on
//! systematic RS encoding, so the witness `C^{-1}(f)` is obtained from the
//! message subspace of the same committed RS codeword used by WHIR.

use super::*;

/// WHIR-native decider proof for the direct Boolean PESAT relation.
pub struct WhirBooleanPesatProtocol<'a, F, EF, Pcs, Challenger, Dft>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    Dft: TwoAdicSubgroupDft<F>,
{
    pcs: &'a Pcs,
    code: &'a ReedSolomonCode<F, Dft>,
    pesat: &'a BooleanPesat<F, EF>,
    challenger_seed: Challenger,
}

impl<'a, F, EF, Pcs, Challenger, Dft> WhirBooleanPesatProtocol<'a, F, EF, Pcs, Challenger, Dft>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    Dft: TwoAdicSubgroupDft<F>,
{
    /// Create a Boolean PESAT finalizer from a compatible PCS.
    pub const fn new(
        pcs: &'a Pcs,
        code: &'a ReedSolomonCode<F, Dft>,
        pesat: &'a BooleanPesat<F, EF>,
        challenger_seed: Challenger,
    ) -> Self {
        Self {
            pcs,
            code,
            pesat,
            challenger_seed,
        }
    }
}

impl<'a, F, EF, Pcs, Challenger, Dft> WhirBooleanPesatProtocol<'a, F, EF, Pcs, Challenger, Dft>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Pcs: MultilinearPcs<EF, Challenger, Val = EF>,
    Pcs::Commitment: Clone + PartialEq,
    Challenger:
        FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<Pcs::Commitment> + Clone,
    Dft: TwoAdicSubgroupDft<F>,
{
    /// Prove `sum_i eq(beta, i) * w_i * (w_i - 1) = eta`.
    pub fn prove<ProverData>(
        &self,
        instance: &AccumulatorInstance<EF, Pcs::Commitment>,
        witness: &AccumulatorWitness<EF, ProverData>,
    ) -> Result<WhirPesatProof<EF, Pcs::Proof>, FinalizerError> {
        self.validate_pesat_shape(instance)?;
        if witness.f.len() != self.code.codeword_len() {
            return Err(FinalizerError::Decider(DeciderError::EncodingMismatch));
        }
        let committed_witness = self.systematic_witness_from_codeword(&witness.f)?;

        let mut challenger = self.challenger_seed.clone();
        self.observe_statement(instance, &mut challenger);
        let (sumcheck, point, final_claim, terminal_value) =
            self.prove_boolean_sumcheck(instance, &committed_witness, &mut challenger);
        self.check_terminal_claim(instance, point.as_slice(), terminal_value, final_claim)?;

        challenger.observe_algebra_element(terminal_value);
        let opening_points = [vec![self.opening_point(point.as_slice())]];
        let evaluations = RowMajorMatrix::new(witness.f.clone(), 1);
        let (commitment, prover_data) =
            self.pcs
                .commit(evaluations, &opening_points, &mut challenger);
        if commitment != instance.rt {
            return Err(FinalizerError::Decider(DeciderError::MerkleRoot));
        }
        let (opened_values, pcs_proof) = self.pcs.open(prover_data, &mut challenger);
        let opened_values = opened_values.first().ok_or(FinalizerError::Unsupported(
            "PCS did not return terminal opening",
        ))?;
        if opened_values != &vec![terminal_value] {
            return Err(FinalizerError::Decider(DeciderError::EncodingMismatch));
        }

        Ok(WhirPesatProof {
            decider_sumcheck: sumcheck,
            terminal_values: vec![terminal_value],
            pcs_proof,
        })
    }

    /// Verify the direct Boolean PESAT decider proof.
    pub fn verify(
        &self,
        instance: &AccumulatorInstance<EF, Pcs::Commitment>,
        proof: &WhirPesatProof<EF, Pcs::Proof>,
    ) -> Result<(), FinalizerError> {
        self.validate_pesat_shape(instance)?;
        if proof.terminal_values.len() != 1 {
            return Err(FinalizerError::Decider(DeciderError::EncodingMismatch));
        }

        let mut challenger = self.challenger_seed.clone();
        self.observe_statement(instance, &mut challenger);
        let (point, final_claim) = verify_sumcheck::<F, EF, _>(
            &proof.decider_sumcheck,
            &mut challenger,
            instance.eta,
            instance.beta.len(),
            3,
            "whir-boolean-pesat",
        )?;
        let terminal_value = proof.terminal_values[0];
        self.check_terminal_claim(instance, point.as_slice(), terminal_value, final_claim)?;

        challenger.observe_algebra_element(terminal_value);
        let opening_claims = [vec![(self.opening_point(point.as_slice()), terminal_value)]];
        self.pcs
            .verify(
                &instance.rt,
                &opening_claims,
                &proof.pcs_proof,
                &mut challenger,
            )
            .map_err(|err| FinalizerError::OpeningProof(format!("{err:?}")))
    }

    fn validate_pesat_shape(
        &self,
        instance: &AccumulatorInstance<EF, Pcs::Commitment>,
    ) -> Result<(), FinalizerError> {
        if !self.code.is_systematic() {
            return Err(FinalizerError::Unsupported(
                "WHIR-native Boolean PESAT proof requires systematic RS encoding",
            ));
        }
        if self.pcs.num_vars() != self.code.log_codeword_len() {
            return Err(FinalizerError::Unsupported(
                "PCS variable count must match the WARP codeword MLE dimension",
            ));
        }
        if self.pesat.shape().witness_len() != self.code.msg_len() {
            return Err(FinalizerError::Decider(DeciderError::EncodingMismatch));
        }
        if instance.beta.len() != self.pesat.shape().log_constraints {
            return Err(FinalizerError::Decider(DeciderError::BundledPesat));
        }
        Ok(())
    }

    fn observe_statement(
        &self,
        instance: &AccumulatorInstance<EF, Pcs::Commitment>,
        challenger: &mut Challenger,
    ) {
        challenger.observe(instance.rt.clone());
        challenger.observe_algebra_slice(&instance.beta);
        challenger.observe_algebra_element(instance.eta);
    }

    fn prove_boolean_sumcheck(
        &self,
        instance: &AccumulatorInstance<EF, Pcs::Commitment>,
        witness: &[EF],
        challenger: &mut Challenger,
    ) -> (SumcheckProof<EF>, Point<EF>, EF, EF) {
        let rounds = instance.beta.len();
        let mut proof = SumcheckProof::<EF>::new();
        let mut prefix = Vec::with_capacity(rounds);
        let mut prefix_eq = EF::ONE;
        let mut folded = witness.to_vec();
        let mut claim = instance.eta;

        for round in 0..rounds {
            let suffix_bits = rounds - round - 1;
            let half_len = 1usize << suffix_bits;
            let suffix_eq = Poly::<EF>::new_from_point(&instance.beta[round + 1..], prefix_eq);
            let beta = instance.beta[round];
            let eq_const = EF::ONE - beta;
            let eq_linear = beta + beta - EF::ONE;
            let mut coeffs = vec![EF::ZERO; 4];

            for suffix in 0..half_len {
                let lo = folded[suffix];
                let hi = folded[suffix + half_len];
                let diff = hi - lo;
                let q0 = lo * (lo - EF::ONE);
                let q1 = diff * (lo + lo - EF::ONE);
                let q2 = diff * diff;
                let weight = suffix_eq.as_slice()[suffix];
                coeffs[0] += weight * eq_const * q0;
                coeffs[1] += weight * (eq_const * q1 + eq_linear * q0);
                coeffs[2] += weight * (eq_const * q2 + eq_linear * q1);
                coeffs[3] += weight * eq_linear * q2;
            }

            debug_assert_eq!(coeffs[0] + coeffs.iter().copied().sum::<EF>(), claim);
            let challenge = observe_and_sample::<F, EF, _>(&mut proof, challenger, coeffs.clone());
            claim = coeffs
                .iter()
                .rev()
                .fold(EF::ZERO, |acc, &c| acc * challenge + c);
            prefix.push(challenge);
            prefix_eq *= beta * challenge + (EF::ONE - beta) * (EF::ONE - challenge);

            for suffix in 0..half_len {
                let lo = folded[suffix];
                let hi = folded[suffix + half_len];
                folded[suffix] = lo + challenge * (hi - lo);
            }
            folded.truncate(half_len);
        }

        debug_assert_eq!(folded.len(), 1);
        (proof, Point::new(prefix), claim, folded[0])
    }

    fn check_terminal_claim(
        &self,
        instance: &AccumulatorInstance<EF, Pcs::Commitment>,
        point: &[EF],
        terminal_value: EF,
        final_claim: EF,
    ) -> Result<(), FinalizerError> {
        let expected =
            eval_eq_ext(&instance.beta, point) * terminal_value * (terminal_value - EF::ONE);
        if expected != final_claim {
            return Err(FinalizerError::Decider(DeciderError::BundledPesat));
        }
        Ok(())
    }

    fn systematic_witness_from_codeword(&self, codeword: &[EF]) -> Result<Vec<EF>, FinalizerError> {
        if codeword.len() != self.code.codeword_len() {
            return Err(FinalizerError::Decider(DeciderError::EncodingMismatch));
        }
        Ok((0..self.code.msg_len())
            .map(|index| codeword[self.code.systematic_codeword_index(index)])
            .collect())
    }

    fn opening_point(&self, point: &[EF]) -> Point<EF> {
        self.code.systematic_message_point(point)
    }
}

/// Composed WHIR-native finalizer for direct Boolean PESAT.
pub struct WhirBooleanWarpFinalizerProtocol<'a, F, EF, Pcs, Challenger, Dft>
where
    F: TwoAdicField + PrimeCharacteristicRing,
    EF: ExtensionField<F>,
    Dft: TwoAdicSubgroupDft<F>,
{
    pcs: &'a Pcs,
    code: &'a ReedSolomonCode<F, Dft>,
    pesat: &'a BooleanPesat<F, EF>,
    challenger_seed: Challenger,
}

impl<'a, F, EF, Pcs, Challenger, Dft>
    WhirBooleanWarpFinalizerProtocol<'a, F, EF, Pcs, Challenger, Dft>
where
    F: TwoAdicField + PrimeCharacteristicRing,
    EF: ExtensionField<F>,
    Dft: TwoAdicSubgroupDft<F>,
{
    /// Create a composed Boolean WHIR finalizer from a compatible PCS.
    pub const fn new(
        pcs: &'a Pcs,
        code: &'a ReedSolomonCode<F, Dft>,
        pesat: &'a BooleanPesat<F, EF>,
        challenger_seed: Challenger,
    ) -> Self {
        Self {
            pcs,
            code,
            pesat,
            challenger_seed,
        }
    }
}

impl<'a, F, EF, Pcs, Challenger, Dft>
    WhirBooleanWarpFinalizerProtocol<'a, F, EF, Pcs, Challenger, Dft>
where
    F: TwoAdicField + PrimeCharacteristicRing,
    EF: ExtensionField<F> + TwoAdicField,
    Pcs: MultilinearPcs<EF, Challenger, Val = EF>,
    Pcs::Commitment: Clone + PartialEq,
    Challenger:
        FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<Pcs::Commitment> + Clone,
    Dft: TwoAdicSubgroupDft<F>,
{
    /// Prove the final accumulator opening and direct Boolean PESAT claim.
    pub fn prove<ProverData>(
        &self,
        instance: &AccumulatorInstance<EF, Pcs::Commitment>,
        witness: &AccumulatorWitness<EF, ProverData>,
    ) -> Result<WhirWarpFinalizerProof<EF, Pcs::Proof>, FinalizerError> {
        let opening_protocol = WhirAccumulatorOpeningProtocol::<F, EF, Pcs, Challenger, Dft>::new(
            self.pcs,
            self.code,
            self.tagged_challenger(domain::WHIR_WARP_OPENING),
        );
        let boolean_protocol = WhirBooleanPesatProtocol::<F, EF, Pcs, Challenger, Dft>::new(
            self.pcs,
            self.code,
            self.pesat,
            self.tagged_challenger(domain::WHIR_WARP_PESAT),
        );

        let accumulator_opening = opening_protocol.prove(instance, witness)?;
        let pesat = boolean_protocol.prove(instance, witness)?;

        Ok(WhirWarpFinalizerProof {
            accumulator_opening,
            pesat,
        })
    }

    /// Verify both final WARP decider subclaims against the same commitment.
    pub fn verify(
        &self,
        instance: &AccumulatorInstance<EF, Pcs::Commitment>,
        proof: &WhirWarpFinalizerProof<EF, Pcs::Proof>,
    ) -> Result<(), FinalizerError> {
        let opening_protocol = WhirAccumulatorOpeningProtocol::<F, EF, Pcs, Challenger, Dft>::new(
            self.pcs,
            self.code,
            self.tagged_challenger(domain::WHIR_WARP_OPENING),
        );
        let boolean_protocol = WhirBooleanPesatProtocol::<F, EF, Pcs, Challenger, Dft>::new(
            self.pcs,
            self.code,
            self.pesat,
            self.tagged_challenger(domain::WHIR_WARP_PESAT),
        );

        opening_protocol.verify(instance, &proof.accumulator_opening)?;
        boolean_protocol.verify(instance, &proof.pesat)
    }

    fn tagged_challenger(&self, tag: u64) -> Challenger {
        let mut challenger = self.challenger_seed.clone();
        challenger.observe(F::from_u64(tag));
        challenger
    }
}

impl<'a, F, EF, Pcs, Challenger, Dft, ProverData>
    crate::finalize::AccumulatorFinalizer<F, EF, Pcs::Commitment, ProverData>
    for WhirBooleanWarpFinalizerProtocol<'a, F, EF, Pcs, Challenger, Dft>
where
    F: TwoAdicField + PrimeCharacteristicRing,
    EF: ExtensionField<F> + TwoAdicField,
    Pcs: MultilinearPcs<EF, Challenger, Val = EF>,
    Pcs::Commitment: Clone + PartialEq,
    Challenger:
        FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<Pcs::Commitment> + Clone,
    Dft: TwoAdicSubgroupDft<F>,
{
    type Proof = WhirWarpFinalizerProof<EF, Pcs::Proof>;

    fn finalize(
        &self,
        instance: &AccumulatorInstance<EF, Pcs::Commitment>,
        witness: &AccumulatorWitness<EF, ProverData>,
    ) -> Result<Self::Proof, FinalizerError> {
        self.prove(instance, witness)
    }

    fn verify(
        &self,
        instance: &AccumulatorInstance<EF, Pcs::Commitment>,
        proof: &Self::Proof,
    ) -> Result<(), FinalizerError> {
        WhirBooleanWarpFinalizerProtocol::verify(self, instance, proof)
    }
}

/// Precommitted Boolean finalizer over an accumulator opening backend.
pub struct WhirPrecommittedBooleanWarpFinalizerProtocol<'a, F, EF, Backend, Challenger, Dft>
where
    F: TwoAdicField + PrimeCharacteristicRing,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F>,
    Backend: AccumulatorPointOpeningBackend<F, EF, Challenger>,
    Dft: TwoAdicSubgroupDft<F>,
{
    backend: &'a Backend,
    code: &'a ReedSolomonCode<F, Dft>,
    pesat: &'a BooleanPesat<F, EF>,
    challenger_seed: Challenger,
}

impl<'a, F, EF, Backend, Challenger, Dft>
    WhirPrecommittedBooleanWarpFinalizerProtocol<'a, F, EF, Backend, Challenger, Dft>
where
    F: TwoAdicField + PrimeCharacteristicRing,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F>,
    Backend: AccumulatorPointOpeningBackend<F, EF, Challenger>,
    Dft: TwoAdicSubgroupDft<F>,
{
    /// Create a finalizer over an already committed accumulator.
    pub const fn new(
        backend: &'a Backend,
        code: &'a ReedSolomonCode<F, Dft>,
        pesat: &'a BooleanPesat<F, EF>,
        challenger_seed: Challenger,
    ) -> Self {
        Self {
            backend,
            code,
            pesat,
            challenger_seed,
        }
    }
}

impl<'a, F, EF, Backend, Challenger, Dft>
    WhirPrecommittedBooleanWarpFinalizerProtocol<'a, F, EF, Backend, Challenger, Dft>
where
    F: TwoAdicField + PrimeCharacteristicRing,
    EF: ExtensionField<F> + TwoAdicField,
    Backend: AccumulatorPointOpeningBackend<F, EF, Challenger>,
    Backend::Commitment: Clone + PartialEq,
    Challenger: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanObserve<Backend::Commitment>
        + Clone,
    Dft: TwoAdicSubgroupDft<F>,
{
    /// Prove both final decider claims against `instance.rt`.
    pub fn prove(
        &self,
        instance: &AccumulatorInstance<EF, Backend::Commitment>,
        witness: &AccumulatorWitness<EF, Backend::ProverData>,
    ) -> Result<WhirWarpFinalizerProof<EF, Backend::PointProof>, FinalizerError> {
        let pcs = PrecommittedAccumulatorPcs::<F, EF, Backend, Challenger>::prover(
            self.backend,
            &instance.rt,
            &witness.td,
        );
        let finalizer = WhirBooleanWarpFinalizerProtocol::<F, EF, _, Challenger, Dft>::new(
            &pcs,
            self.code,
            self.pesat,
            self.challenger_seed.clone(),
        );
        finalizer.prove(instance, witness)
    }

    /// Verify both final decider claims against `instance.rt`.
    pub fn verify(
        &self,
        instance: &AccumulatorInstance<EF, Backend::Commitment>,
        proof: &WhirWarpFinalizerProof<EF, Backend::PointProof>,
    ) -> Result<(), FinalizerError> {
        let pcs = PrecommittedAccumulatorPcs::<F, EF, Backend, Challenger>::verifier(self.backend);
        let finalizer = WhirBooleanWarpFinalizerProtocol::<F, EF, _, Challenger, Dft>::new(
            &pcs,
            self.code,
            self.pesat,
            self.challenger_seed.clone(),
        );
        finalizer.verify(instance, proof)
    }
}

impl<'a, F, EF, Backend, Challenger, Dft>
    crate::finalize::AccumulatorFinalizer<F, EF, Backend::Commitment, Backend::ProverData>
    for WhirPrecommittedBooleanWarpFinalizerProtocol<'a, F, EF, Backend, Challenger, Dft>
where
    F: TwoAdicField + PrimeCharacteristicRing,
    EF: ExtensionField<F> + TwoAdicField,
    Backend: AccumulatorPointOpeningBackend<F, EF, Challenger>,
    Backend::Commitment: Clone + PartialEq,
    Challenger: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanObserve<Backend::Commitment>
        + Clone,
    Dft: TwoAdicSubgroupDft<F>,
{
    type Proof = WhirWarpFinalizerProof<EF, Backend::PointProof>;

    fn finalize(
        &self,
        instance: &AccumulatorInstance<EF, Backend::Commitment>,
        witness: &AccumulatorWitness<EF, Backend::ProverData>,
    ) -> Result<Self::Proof, FinalizerError> {
        self.prove(instance, witness)
    }

    fn verify(
        &self,
        instance: &AccumulatorInstance<EF, Backend::Commitment>,
        proof: &Self::Proof,
    ) -> Result<(), FinalizerError> {
        WhirPrecommittedBooleanWarpFinalizerProtocol::verify(self, instance, proof)
    }
}

fn eval_eq_ext<EF: Field>(lhs: &[EF], rhs: &[EF]) -> EF {
    debug_assert_eq!(lhs.len(), rhs.len());
    lhs.iter()
        .zip(rhs.iter())
        .map(|(&l, &r)| l * r + (EF::ONE - l) * (EF::ONE - r))
        .product()
}
