//! WARP verifier: replays the prover transcript, checks every algebraic
//! identity, and rejects if anything is off.
//!
//! The verifier produces no new accumulator state on its own; the prover
//! sends the new `acc.x` and the verifier validates it against the proof
//! transcript.

use alloc::vec;
use alloc::vec::Vec;
use alloc::{format, string::ToString};
use core::marker::PhantomData;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{BatchOpeningRef, ExtensionMmcs, Mmcs};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::Dimensions;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use tracing::instrument;

use crate::accumulator::{
    AccumulatorInstance, WarpProof, WarpProofCommitted, WarpProofExternal, WarpProofExternalBatched,
};
use crate::code::ReedSolomonCode;
use crate::error::VerifierError;
use crate::relation::BundledPesat;
use crate::sumcheck::verify_sumcheck;
use crate::transcript::{bind_protocol, sample_indices};

use super::WarpParams;
use super::prover::boolean_point;
use super::{
    AccumulatorBatchOpeningBackend, AccumulatorCommitmentBackend,
    ExternalCodewordBatchOpeningVerifier, ExternalCodewordOpeningVerifier,
    MmcsExternalOpeningVerifier,
};

/// WARP verifier bound to a specific PESAT, RS code, and Mmcs.
pub struct WarpVerifier<'a, F, EF, MT, Dft, Pesat>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    MT: Mmcs<F>,
    Dft: TwoAdicSubgroupDft<F>,
    Pesat: BundledPesat<F, EF>,
{
    pub mmcs: &'a MT,
    pub code: &'a ReedSolomonCode<F, Dft>,
    pub pesat: &'a Pesat,
    pub params: WarpParams,
    _ph: PhantomData<EF>,
}

impl<'a, F, EF, MT, Dft, Pesat> WarpVerifier<'a, F, EF, MT, Dft, Pesat>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    MT: Mmcs<F> + Sync,
    Dft: TwoAdicSubgroupDft<F>,
    Pesat: BundledPesat<F, EF>,
{
    /// Create a verifier.
    pub fn new(
        mmcs: &'a MT,
        code: &'a ReedSolomonCode<F, Dft>,
        pesat: &'a Pesat,
        params: WarpParams,
    ) -> Self {
        assert_eq!(
            pesat.shape().explicit_len,
            0,
            "p3-warp v1 supports instance-free PESAT only (κ = 0)"
        );
        assert_eq!(
            code.msg_len(),
            pesat.shape().witness_len(),
            "RS message length must equal PESAT witness length"
        );
        Self {
            mmcs,
            code,
            pesat,
            params,
            _ph: PhantomData,
        }
    }

    /// Verify one accumulation step.
    ///
    /// # Arguments
    ///
    /// - `challenger` — Fiat-Shamir state, must be in identical state to the
    ///   prover's challenger at protocol start.
    /// - `num_fresh` — number of fresh PESAT instances `ℓ_1` that the prover
    ///   accumulated this step.
    /// - `prior_instances` — verifier-visible parts of the `ℓ_2` prior
    ///   accumulators that were folded in.
    /// - `new_instance` — verifier-visible part of the new accumulator
    ///   produced by the prover.
    /// - `proof` — the prover's transcript record.
    #[instrument(skip_all, name = "warp::verify")]
    pub fn verify<Challenger>(
        &self,
        challenger: &mut Challenger,
        num_fresh: usize,
        prior_instances: &[AccumulatorInstance<EF, MT::Commitment>],
        new_instance: &AccumulatorInstance<EF, MT::Commitment>,
        proof: &WarpProof<F, EF, MT::Commitment, MT::Proof>,
    ) -> Result<(), VerifierError>
    where
        Challenger:
            FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<MT::Commitment>,
    {
        let l1 = num_fresh;
        let l2 = prior_instances.len();
        let l = l1 + l2;
        if !(l >= 2 && l.is_power_of_two()) {
            return Err(VerifierError::AccumulatorMismatch { field: "ℓ" });
        }
        let log_l = l.trailing_zeros() as usize;
        let shape = self.pesat.shape();
        let log_m = shape.log_constraints;
        let log_n = self.code.log_codeword_len();
        let log_h = log_n - self.code.log_inv_rate();
        let beta_len = shape.beta_len();
        if new_instance.alpha.len() != log_n {
            return Err(VerifierError::AccumulatorMismatch { field: "alpha" });
        }
        if new_instance.beta.len() != beta_len {
            return Err(VerifierError::AccumulatorMismatch { field: "beta" });
        }
        for inst in prior_instances {
            if inst.alpha.len() != log_n {
                return Err(VerifierError::AccumulatorMismatch {
                    field: "prior alpha",
                });
            }
            if inst.beta.len() != beta_len {
                return Err(VerifierError::AccumulatorMismatch {
                    field: "prior beta",
                });
            }
        }
        if proof.mu_fresh.len() != l1 {
            return Err(VerifierError::AccumulatorMismatch {
                field: "mu_fresh count",
            });
        }
        if proof.fresh_shift_answers.len() != self.params.num_shift_queries
            || proof.fresh_merkle_proofs.len() != self.params.num_shift_queries
        {
            return Err(VerifierError::ShiftQueryCount {
                expected: self.params.num_shift_queries,
                got: proof.fresh_shift_answers.len(),
            });
        }
        if proof.acc_shift_answers.len() != l2 || proof.acc_merkle_proofs.len() != l2 {
            return Err(VerifierError::AccumulatorMismatch {
                field: "acc shift answers length",
            });
        }
        for (j, answers) in proof.acc_shift_answers.iter().enumerate() {
            if answers.len() != self.params.num_shift_queries {
                return Err(VerifierError::ShiftQueryCount {
                    expected: self.params.num_shift_queries,
                    got: answers.len(),
                });
            }
            if proof.acc_merkle_proofs[j].len() != self.params.num_shift_queries {
                return Err(VerifierError::ShiftQueryCount {
                    expected: self.params.num_shift_queries,
                    got: proof.acc_merkle_proofs[j].len(),
                });
            }
        }
        if proof.nu_ood.len() != self.params.num_ood {
            return Err(VerifierError::AccumulatorMismatch { field: "nu_ood" });
        }

        // ---------- 1. Bind protocol parameters into the transcript. ----------
        bind_protocol::<F, _>(
            challenger,
            &self.pesat.description(),
            l1,
            l2,
            self.params.num_ood,
            self.params.num_shift_queries,
            log_n,
            log_h,
        );

        // ---------- 2. Replay observe/sample of rt_0, fresh µ_i, prior accs. ----------
        challenger.observe(proof.rt_0.clone());
        for &mu_i in &proof.mu_fresh {
            challenger.observe_algebra_element(mu_i);
        }
        for inst in prior_instances {
            challenger.observe(inst.rt.clone());
            for &a in &inst.alpha {
                challenger.observe_algebra_element(a);
            }
            challenger.observe_algebra_element(inst.mu);
            for &b in &inst.beta {
                challenger.observe_algebra_element(b);
            }
            challenger.observe_algebra_element(inst.eta);
        }

        // ---------- 3. Re-sample fresh τ_i and reconstruct (α_i, β_i, η_i). ----------
        let fresh_taus: Vec<Vec<EF>> = (0..l1)
            .map(|_| {
                (0..log_m)
                    .map(|_| challenger.sample_algebra_element())
                    .collect()
            })
            .collect();

        let mut all_alphas: Vec<Vec<EF>> = (0..l1).map(|_| vec![EF::ZERO; log_n]).collect();
        let mut all_betas: Vec<Vec<EF>> = fresh_taus.clone();
        let mut all_mus: Vec<EF> = proof.mu_fresh.clone();
        let mut all_etas: Vec<EF> = vec![EF::ZERO; l1];
        for inst in prior_instances {
            all_alphas.push(inst.alpha.clone());
            all_betas.push(inst.beta.clone());
            all_mus.push(inst.mu);
            all_etas.push(inst.eta);
        }

        // ---------- 4. Sample (ω, τ); compute initial sum σ⁽¹⁾. ----------
        let omega: EF = challenger.sample_algebra_element();
        let tau: Vec<EF> = (0..log_l)
            .map(|_| challenger.sample_algebra_element())
            .collect();
        let tau_eq = Poly::<EF>::new_from_point(&tau, EF::ONE);
        let sigma_1: EF = (0..l)
            .map(|i| tau_eq.as_slice()[i] * (all_mus[i] + omega * all_etas[i]))
            .sum();

        // ---------- 5. Verify §6.3 sumcheck. ----------
        let d1 = 1 + (log_n + 1).max(log_m + shape.max_degree);
        let (gamma_pt, twin_final_claim) = verify_sumcheck::<F, EF, _>(
            &proof.twin_constraint_sumcheck,
            challenger,
            sigma_1,
            log_l,
            d1,
            "twin-constraint",
        )?;
        let gamma = gamma_pt.as_slice().to_vec();

        // ---------- 6. Read merged commitment + ν₀ + η. ----------
        challenger.observe(new_instance.rt.clone());
        challenger.observe_algebra_element(proof.nu_0);
        challenger.observe_algebra_element(proof.eta);

        // ---------- 7. Twin-constraint final-claim oracle check:
        //              h_last(γ_last) == eq(τ, γ) · (ν_0 + ω · η). ----------
        let eq_tau_at_gamma: EF = Point::eval_eq(&tau, &gamma);
        let expected_twin_final = eq_tau_at_gamma * (proof.nu_0 + omega * proof.eta);
        if expected_twin_final != twin_final_claim {
            return Err(VerifierError::TwinConstraintFinalClaim);
        }

        // ---------- 8. Verify accumulator (β, η). ----------
        // β = B̂(γ) = Σ_i eq(γ, i) · β_i.
        let gamma_eq = Poly::<EF>::new_from_point(&gamma, EF::ONE);
        let mut expected_beta = vec![EF::ZERO; beta_len];
        for (i, betas_i) in all_betas.iter().enumerate() {
            let eq_i = gamma_eq.as_slice()[i];
            for (slot, &b) in expected_beta.iter_mut().zip(betas_i.iter()) {
                *slot += eq_i * b;
            }
        }
        if expected_beta != new_instance.beta {
            return Err(VerifierError::AccumulatorMismatch { field: "beta" });
        }
        if new_instance.eta != proof.eta {
            return Err(VerifierError::AccumulatorMismatch { field: "eta" });
        }

        // ---------- 9. Re-sample OOD points + observe ν_k. ----------
        let mut zetas: Vec<Vec<EF>> = Vec::with_capacity(self.params.r());
        let mut nus: Vec<EF> = Vec::with_capacity(self.params.r());
        // ζ_0 = Â(γ) = Σ_i eq(γ, i) · α_i.
        let mut zeta_0: Vec<EF> = vec![EF::ZERO; log_n];
        for (i, alphas_i) in all_alphas.iter().enumerate() {
            let eq_i: EF = gamma_eq.as_slice()[i];
            for (slot, &a) in zeta_0.iter_mut().zip(alphas_i.iter()) {
                *slot += eq_i * a;
            }
        }
        zetas.push(zeta_0);
        nus.push(proof.nu_0);

        for k in 0..self.params.num_ood {
            let zeta_k: Vec<EF> = (0..log_n)
                .map(|_| challenger.sample_algebra_element())
                .collect();
            challenger.observe_algebra_element(proof.nu_ood[k]);
            zetas.push(zeta_k);
            nus.push(proof.nu_ood[k]);
        }

        // ---------- 10. Sample shift indices and verify Merkle openings. ----------
        let shift_indices =
            sample_indices::<F, _>(challenger, log_n, self.params.num_shift_queries);

        // Verify rt_0 openings (one per shift index, each leaf is Vec<F> of length ℓ_1).
        let fresh_dims = vec![Dimensions {
            height: 1usize << log_n,
            width: l1.max(1),
        }];
        for (k_idx, &x_k) in shift_indices.iter().enumerate() {
            let leaf = &proof.fresh_shift_answers[k_idx];
            if leaf.len() != l1 {
                return Err(VerifierError::MerkleProof {
                    index: x_k,
                    reason: format!("expected ℓ_1 = {l1} leaf values, got {}", leaf.len()),
                });
            }
            let opened = vec![leaf.clone()];
            self.mmcs
                .verify_batch(
                    &proof.rt_0,
                    &fresh_dims,
                    x_k,
                    BatchOpeningRef::new(&opened, &proof.fresh_merkle_proofs[k_idx]),
                )
                .map_err(|_| VerifierError::MerkleProof {
                    index: x_k,
                    reason: "rt_0 opening failed".to_string(),
                })?;
        }

        // Verify each prior acc rt opening (one per shift index, each leaf is Vec<EF> of length 1).
        let ext_mmcs = ExtensionMmcs::<F, EF, MT>::new(self.mmcs.clone());
        let acc_dims = vec![Dimensions {
            height: 1usize << log_n,
            width: 1,
        }];
        for (j, inst) in prior_instances.iter().enumerate() {
            for (k_idx, &x_k) in shift_indices.iter().enumerate() {
                let leaf = &proof.acc_shift_answers[j][k_idx];
                if leaf.len() != 1 {
                    return Err(VerifierError::MerkleProof {
                        index: x_k,
                        reason: format!("expected single-element leaf, got {}", leaf.len()),
                    });
                }
                let opened = vec![leaf.clone()];
                ext_mmcs
                    .verify_batch(
                        &inst.rt,
                        &acc_dims,
                        x_k,
                        BatchOpeningRef::new(&opened, &proof.acc_merkle_proofs[j][k_idx]),
                    )
                    .map_err(|_| VerifierError::MerkleProof {
                        index: x_k,
                        reason: format!("prior acc {j} opening failed"),
                    })?;
            }
        }

        // ---------- 11. Compute shift answers ν_{s+k} = Σ eq(γ, i) · f_i(x_k). ----------
        for (k_idx, &x_k) in shift_indices.iter().enumerate() {
            let mut nu_sk = EF::ZERO;
            for i in 0..l1 {
                nu_sk += gamma_eq.as_slice()[i] * EF::from(proof.fresh_shift_answers[k_idx][i]);
            }
            for j in 0..l2 {
                nu_sk += gamma_eq.as_slice()[l1 + j] * proof.acc_shift_answers[j][k_idx][0];
            }
            zetas.push(boolean_point::<EF>(x_k, log_n));
            nus.push(nu_sk);
        }

        // ---------- 12. Sample ξ; compute σ⁽²⁾ + verify §8.2 sumcheck. ----------
        let xi: Vec<EF> = (0..self.params.log_r())
            .map(|_| challenger.sample_algebra_element())
            .collect();
        let xi_eq_full = Poly::<EF>::new_from_point(&xi, EF::ONE);
        let r = self.params.r();
        let xi_eq: Vec<EF> = xi_eq_full.as_slice()[..r].to_vec();
        let sigma_2: EF = (0..r).map(|j| xi_eq[j] * nus[j]).sum();

        let (alpha_pt, batching_final_claim) = verify_sumcheck::<F, EF, _>(
            &proof.batching_sumcheck,
            challenger,
            sigma_2,
            log_n,
            2,
            "multilinear-batching",
        )?;
        challenger.observe_algebra_element(proof.mu_final);

        // ---------- 13. §8.2 final-claim oracle check:
        //              h_last(α_last) == µ · eq*(α). ----------
        let alpha_slice = alpha_pt.as_slice();
        // eq*(α) = Σ_j ξ_j · eq(ζ_j, α).
        let mut eq_star_at_alpha = EF::ZERO;
        for (j, zeta_j) in zetas.iter().enumerate() {
            eq_star_at_alpha += xi_eq[j] * Point::eval_eq(zeta_j, alpha_slice);
        }
        let expected_batching_final = proof.mu_final * eq_star_at_alpha;
        if expected_batching_final != batching_final_claim {
            return Err(VerifierError::MultilinearBatchingFinalClaim);
        }

        // ---------- 14. Validate accumulator α and µ. ----------
        if new_instance.alpha != alpha_slice.to_vec() {
            return Err(VerifierError::AccumulatorMismatch { field: "alpha" });
        }
        if new_instance.mu != proof.mu_final {
            return Err(VerifierError::AccumulatorMismatch { field: "mu" });
        }

        Ok(())
    }

    /// Verify one accumulation step produced by
    /// [`WarpProver::prove_with_committed`](crate::protocol::WarpProver::prove_with_committed).
    ///
    /// Mirrors [`Self::verify`] but consumes:
    /// - `fresh_commitments: &[MT::Commitment]` (length `ℓ_1`) — the
    ///   externally-known per-fresh Merkle commitments. These are
    ///   observed into the FS transcript individually (in input order)
    ///   rather than as a single stacked `rt_0`.
    /// - `proof: &WarpProofCommitted<...>` — drops `rt_0` and uses
    ///   `Vec<Vec<Proof>>` for `fresh_merkle_proofs` (one path per
    ///   `(shift, fresh)` pair).
    ///
    /// Verifies each fresh Merkle path against the corresponding
    /// `fresh_commitments[fresh_idx]` directly via `mmcs.verify_batch`.
    #[instrument(skip_all, name = "warp::verify_with_committed")]
    #[allow(clippy::too_many_arguments)]
    pub fn verify_with_committed<Challenger>(
        &self,
        challenger: &mut Challenger,
        fresh_commitments: &[MT::Commitment],
        prior_instances: &[AccumulatorInstance<EF, MT::Commitment>],
        new_instance: &AccumulatorInstance<EF, MT::Commitment>,
        proof: &WarpProofCommitted<F, EF, MT::Commitment, MT::Proof>,
    ) -> Result<(), VerifierError>
    where
        Challenger:
            FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<MT::Commitment>,
    {
        let external_proof = WarpProofExternal {
            mu_fresh: proof.mu_fresh.clone(),
            twin_constraint_sumcheck: proof.twin_constraint_sumcheck.clone(),
            nu_0: proof.nu_0,
            eta: proof.eta,
            nu_ood: proof.nu_ood.clone(),
            batching_sumcheck: proof.batching_sumcheck.clone(),
            mu_final: proof.mu_final,
            fresh_shift_answers: proof.fresh_shift_answers.clone(),
            fresh_opening_proofs: proof.fresh_merkle_proofs.clone(),
            acc_shift_answers: proof.acc_shift_answers.clone(),
            acc_merkle_proofs: proof.acc_merkle_proofs.clone(),
            _ph: PhantomData,
        };
        let fresh_verifier = MmcsExternalOpeningVerifier::new(self.mmcs);
        self.verify_with_external_committed(
            challenger,
            &fresh_verifier,
            fresh_commitments,
            prior_instances,
            new_instance,
            &external_proof,
        )
    }

    /// Verify one accumulation step whose fresh inputs were committed by an
    /// arbitrary external PCS backend.
    #[instrument(skip_all, name = "warp::verify_with_external_committed")]
    #[allow(clippy::too_many_arguments)]
    pub fn verify_with_external_committed<Challenger, FreshVerifier>(
        &self,
        challenger: &mut Challenger,
        fresh_verifier: &FreshVerifier,
        fresh_commitments: &[FreshVerifier::Commitment],
        prior_instances: &[AccumulatorInstance<EF, MT::Commitment>],
        new_instance: &AccumulatorInstance<EF, MT::Commitment>,
        proof: &WarpProofExternal<F, EF, MT::Commitment, FreshVerifier::Proof, MT::Proof>,
    ) -> Result<(), VerifierError>
    where
        Challenger:
            FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<MT::Commitment>,
        FreshVerifier: ExternalCodewordOpeningVerifier<F, Challenger>,
    {
        let acc_backend = ExtensionMmcs::<F, EF, MT>::new(self.mmcs.clone());
        self.verify_with_external_committed_accumulator(
            challenger,
            fresh_verifier,
            &acc_backend,
            fresh_commitments,
            prior_instances,
            new_instance,
            proof,
        )
    }

    /// Verify one accumulation step whose fresh inputs and accumulator
    /// codewords use caller-provided external commitment/opening backends.
    #[instrument(skip_all, name = "warp::verify_with_external_committed_accumulator")]
    #[allow(clippy::too_many_arguments)]
    pub fn verify_with_external_committed_accumulator<Challenger, FreshVerifier, AccBackend>(
        &self,
        challenger: &mut Challenger,
        fresh_verifier: &FreshVerifier,
        acc_backend: &AccBackend,
        fresh_commitments: &[FreshVerifier::Commitment],
        prior_instances: &[AccumulatorInstance<EF, AccBackend::Commitment>],
        new_instance: &AccumulatorInstance<EF, AccBackend::Commitment>,
        proof: &WarpProofExternal<
            F,
            EF,
            AccBackend::Commitment,
            FreshVerifier::Proof,
            AccBackend::Proof,
        >,
    ) -> Result<(), VerifierError>
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
        FreshVerifier: ExternalCodewordOpeningVerifier<F, Challenger>,
        AccBackend: AccumulatorCommitmentBackend<F, EF, Challenger>,
    {
        let l1 = fresh_commitments.len();
        let l2 = prior_instances.len();
        let l = l1 + l2;
        if !(l >= 2 && l.is_power_of_two()) {
            return Err(VerifierError::AccumulatorMismatch { field: "ℓ" });
        }
        let log_l = l.trailing_zeros() as usize;
        let shape = self.pesat.shape();
        let log_m = shape.log_constraints;
        let log_n = self.code.log_codeword_len();
        let log_h = log_n - self.code.log_inv_rate();
        let beta_len = shape.beta_len();
        if new_instance.alpha.len() != log_n {
            return Err(VerifierError::AccumulatorMismatch { field: "alpha" });
        }
        if new_instance.beta.len() != beta_len {
            return Err(VerifierError::AccumulatorMismatch { field: "beta" });
        }
        for inst in prior_instances {
            if inst.alpha.len() != log_n {
                return Err(VerifierError::AccumulatorMismatch {
                    field: "prior alpha",
                });
            }
            if inst.beta.len() != beta_len {
                return Err(VerifierError::AccumulatorMismatch {
                    field: "prior beta",
                });
            }
        }
        if proof.mu_fresh.len() != l1 {
            return Err(VerifierError::AccumulatorMismatch {
                field: "mu_fresh count",
            });
        }
        if proof.fresh_shift_answers.len() != self.params.num_shift_queries
            || proof.fresh_opening_proofs.len() != self.params.num_shift_queries
        {
            return Err(VerifierError::ShiftQueryCount {
                expected: self.params.num_shift_queries,
                got: proof.fresh_shift_answers.len(),
            });
        }
        if proof.acc_shift_answers.len() != l2 || proof.acc_merkle_proofs.len() != l2 {
            return Err(VerifierError::AccumulatorMismatch {
                field: "acc shift answers length",
            });
        }
        for (j, answers) in proof.acc_shift_answers.iter().enumerate() {
            if answers.len() != self.params.num_shift_queries {
                return Err(VerifierError::ShiftQueryCount {
                    expected: self.params.num_shift_queries,
                    got: answers.len(),
                });
            }
            if proof.acc_merkle_proofs[j].len() != self.params.num_shift_queries {
                return Err(VerifierError::ShiftQueryCount {
                    expected: self.params.num_shift_queries,
                    got: proof.acc_merkle_proofs[j].len(),
                });
            }
        }
        if proof.nu_ood.len() != self.params.num_ood {
            return Err(VerifierError::AccumulatorMismatch { field: "nu_ood" });
        }

        // ---------- 1. Bind protocol parameters. ----------
        bind_protocol::<F, _>(
            challenger,
            &self.pesat.description(),
            l1,
            l2,
            self.params.num_ood,
            self.params.num_shift_queries,
            log_n,
            log_h,
        );

        // ---------- 2. VARIANT: observe each fresh commitment individually. ----------
        for c in fresh_commitments {
            fresh_verifier.observe_commitment(challenger, c);
        }
        for &mu_i in &proof.mu_fresh {
            challenger.observe_algebra_element(mu_i);
        }
        for inst in prior_instances {
            acc_backend.observe_commitment(challenger, &inst.rt);
            for &a in &inst.alpha {
                challenger.observe_algebra_element(a);
            }
            challenger.observe_algebra_element(inst.mu);
            for &b in &inst.beta {
                challenger.observe_algebra_element(b);
            }
            challenger.observe_algebra_element(inst.eta);
        }

        // ---------- 3. Re-sample fresh τ_i and reconstruct (α_i, β_i, η_i). ----------
        let fresh_taus: Vec<Vec<EF>> = (0..l1)
            .map(|_| {
                (0..log_m)
                    .map(|_| challenger.sample_algebra_element())
                    .collect()
            })
            .collect();

        let mut all_alphas: Vec<Vec<EF>> = (0..l1).map(|_| vec![EF::ZERO; log_n]).collect();
        let mut all_betas: Vec<Vec<EF>> = fresh_taus.clone();
        let mut all_mus: Vec<EF> = proof.mu_fresh.clone();
        let mut all_etas: Vec<EF> = vec![EF::ZERO; l1];
        for inst in prior_instances {
            all_alphas.push(inst.alpha.clone());
            all_betas.push(inst.beta.clone());
            all_mus.push(inst.mu);
            all_etas.push(inst.eta);
        }

        // ---------- 4. Sample (ω, τ); compute σ⁽¹⁾. ----------
        let omega: EF = challenger.sample_algebra_element();
        let tau: Vec<EF> = (0..log_l)
            .map(|_| challenger.sample_algebra_element())
            .collect();
        let tau_eq = Poly::<EF>::new_from_point(&tau, EF::ONE);
        let sigma_1: EF = (0..l)
            .map(|i| tau_eq.as_slice()[i] * (all_mus[i] + omega * all_etas[i]))
            .sum();

        // ---------- 5. Verify §6.3 sumcheck. ----------
        let d1 = 1 + (log_n + 1).max(log_m + shape.max_degree);
        let (gamma_pt, twin_final_claim) = verify_sumcheck::<F, EF, _>(
            &proof.twin_constraint_sumcheck,
            challenger,
            sigma_1,
            log_l,
            d1,
            "twin-constraint",
        )?;
        let gamma = gamma_pt.as_slice().to_vec();

        // ---------- 6. Read merged commitment + ν₀ + η. ----------
        acc_backend.observe_commitment(challenger, &new_instance.rt);
        challenger.observe_algebra_element(proof.nu_0);
        challenger.observe_algebra_element(proof.eta);

        // ---------- 7. Twin-constraint final-claim oracle check. ----------
        let eq_tau_at_gamma: EF = Point::eval_eq(&tau, &gamma);
        let expected_twin_final = eq_tau_at_gamma * (proof.nu_0 + omega * proof.eta);
        if expected_twin_final != twin_final_claim {
            return Err(VerifierError::TwinConstraintFinalClaim);
        }

        // ---------- 8. Verify accumulator (β, η). ----------
        let gamma_eq = Poly::<EF>::new_from_point(&gamma, EF::ONE);
        let mut expected_beta = vec![EF::ZERO; beta_len];
        for (i, betas_i) in all_betas.iter().enumerate() {
            let eq_i = gamma_eq.as_slice()[i];
            for (slot, &b) in expected_beta.iter_mut().zip(betas_i.iter()) {
                *slot += eq_i * b;
            }
        }
        if expected_beta != new_instance.beta {
            return Err(VerifierError::AccumulatorMismatch { field: "beta" });
        }
        if new_instance.eta != proof.eta {
            return Err(VerifierError::AccumulatorMismatch { field: "eta" });
        }

        // ---------- 9. Re-sample OOD points + observe ν_k. ----------
        let mut zetas: Vec<Vec<EF>> = Vec::with_capacity(self.params.r());
        let mut nus: Vec<EF> = Vec::with_capacity(self.params.r());
        let mut zeta_0: Vec<EF> = vec![EF::ZERO; log_n];
        for (i, alphas_i) in all_alphas.iter().enumerate() {
            let eq_i: EF = gamma_eq.as_slice()[i];
            for (slot, &a) in zeta_0.iter_mut().zip(alphas_i.iter()) {
                *slot += eq_i * a;
            }
        }
        zetas.push(zeta_0);
        nus.push(proof.nu_0);

        for k in 0..self.params.num_ood {
            let zeta_k: Vec<EF> = (0..log_n)
                .map(|_| challenger.sample_algebra_element())
                .collect();
            challenger.observe_algebra_element(proof.nu_ood[k]);
            zetas.push(zeta_k);
            nus.push(proof.nu_ood[k]);
        }

        // ---------- 10. Sample shift indices and verify Merkle openings. ----------
        let shift_indices =
            sample_indices::<F, _>(challenger, log_n, self.params.num_shift_queries);

        // VARIANT: verify each fresh codeword's external opening proof
        // individually against its external commitment.
        for (k_idx, &x_k) in shift_indices.iter().enumerate() {
            let answers_for_xk = &proof.fresh_shift_answers[k_idx];
            let proofs_for_xk = &proof.fresh_opening_proofs[k_idx];
            if answers_for_xk.len() != l1 || proofs_for_xk.len() != l1 {
                return Err(VerifierError::MerkleProof {
                    index: x_k,
                    reason: format!(
                        "expected ℓ_1 = {l1} per-fresh values/paths, got {}/{}",
                        answers_for_xk.len(),
                        proofs_for_xk.len()
                    ),
                });
            }
            for (i, commitment_i) in fresh_commitments.iter().enumerate() {
                fresh_verifier
                    .verify_opening(
                        commitment_i,
                        log_n,
                        x_k,
                        answers_for_xk[i],
                        &proofs_for_xk[i],
                    )
                    .map_err(|err| VerifierError::MerkleProof {
                        index: x_k,
                        reason: format!("fresh commitment {i} opening failed: {err:?}"),
                    })?;
            }
        }

        // Verify each prior accumulator opening against the configured
        // accumulator backend.
        for (j, inst) in prior_instances.iter().enumerate() {
            for (k_idx, &x_k) in shift_indices.iter().enumerate() {
                let leaf = &proof.acc_shift_answers[j][k_idx];
                if leaf.len() != 1 {
                    return Err(VerifierError::MerkleProof {
                        index: x_k,
                        reason: format!("expected single-element leaf, got {}", leaf.len()),
                    });
                }
                acc_backend
                    .verify_opening(
                        &inst.rt,
                        log_n,
                        x_k,
                        leaf[0],
                        &proof.acc_merkle_proofs[j][k_idx],
                    )
                    .map_err(|err| VerifierError::MerkleProof {
                        index: x_k,
                        reason: format!("prior acc {j} opening failed: {err:?}"),
                    })?;
            }
        }

        // ---------- 11. Compute shift answers ν_{s+k}. ----------
        for (k_idx, &x_k) in shift_indices.iter().enumerate() {
            let mut nu_sk = EF::ZERO;
            for i in 0..l1 {
                nu_sk += gamma_eq.as_slice()[i] * EF::from(proof.fresh_shift_answers[k_idx][i]);
            }
            for j in 0..l2 {
                nu_sk += gamma_eq.as_slice()[l1 + j] * proof.acc_shift_answers[j][k_idx][0];
            }
            zetas.push(boolean_point::<EF>(x_k, log_n));
            nus.push(nu_sk);
        }

        // ---------- 12. Sample ξ; verify §8.2 sumcheck. ----------
        let xi: Vec<EF> = (0..self.params.log_r())
            .map(|_| challenger.sample_algebra_element())
            .collect();
        let xi_eq_full = Poly::<EF>::new_from_point(&xi, EF::ONE);
        let r = self.params.r();
        let xi_eq: Vec<EF> = xi_eq_full.as_slice()[..r].to_vec();
        let sigma_2: EF = (0..r).map(|j| xi_eq[j] * nus[j]).sum();

        let (alpha_pt, batching_final_claim) = verify_sumcheck::<F, EF, _>(
            &proof.batching_sumcheck,
            challenger,
            sigma_2,
            log_n,
            2,
            "multilinear-batching",
        )?;
        challenger.observe_algebra_element(proof.mu_final);

        // ---------- 13. §8.2 final-claim oracle check. ----------
        let alpha_slice = alpha_pt.as_slice();
        let mut eq_star_at_alpha = EF::ZERO;
        for (j, zeta_j) in zetas.iter().enumerate() {
            eq_star_at_alpha += xi_eq[j] * Point::eval_eq(zeta_j, alpha_slice);
        }
        let expected_batching_final = proof.mu_final * eq_star_at_alpha;
        if expected_batching_final != batching_final_claim {
            return Err(VerifierError::MultilinearBatchingFinalClaim);
        }

        // ---------- 14. Validate accumulator α and µ. ----------
        if new_instance.alpha != alpha_slice.to_vec() {
            return Err(VerifierError::AccumulatorMismatch { field: "alpha" });
        }
        if new_instance.mu != proof.mu_final {
            return Err(VerifierError::AccumulatorMismatch { field: "mu" });
        }

        Ok(())
    }

    /// Verify one externally committed WARP step whose shift openings are
    /// batched per fresh codeword and per prior accumulator.
    #[instrument(
        skip_all,
        name = "warp::verify_with_external_committed_accumulator_batched"
    )]
    #[allow(clippy::too_many_arguments)]
    pub fn verify_with_external_committed_accumulator_batched<
        Challenger,
        FreshVerifier,
        AccBackend,
    >(
        &self,
        challenger: &mut Challenger,
        fresh_verifier: &FreshVerifier,
        acc_backend: &AccBackend,
        fresh_commitments: &[FreshVerifier::Commitment],
        prior_instances: &[AccumulatorInstance<EF, AccBackend::Commitment>],
        new_instance: &AccumulatorInstance<EF, AccBackend::Commitment>,
        proof: &WarpProofExternalBatched<
            F,
            EF,
            AccBackend::Commitment,
            FreshVerifier::BatchProof,
            AccBackend::BatchProof,
        >,
    ) -> Result<(), VerifierError>
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
        FreshVerifier: ExternalCodewordBatchOpeningVerifier<F, Challenger>,
        AccBackend: AccumulatorBatchOpeningBackend<F, EF, Challenger>,
    {
        let l1 = fresh_commitments.len();
        let l2 = prior_instances.len();
        let l = l1 + l2;
        if !(l >= 2 && l.is_power_of_two()) {
            return Err(VerifierError::AccumulatorMismatch { field: "ℓ" });
        }
        let log_l = l.trailing_zeros() as usize;
        let shape = self.pesat.shape();
        let log_m = shape.log_constraints;
        let log_n = self.code.log_codeword_len();
        let log_h = log_n - self.code.log_inv_rate();
        let beta_len = shape.beta_len();
        if new_instance.alpha.len() != log_n {
            return Err(VerifierError::AccumulatorMismatch { field: "alpha" });
        }
        if new_instance.beta.len() != beta_len {
            return Err(VerifierError::AccumulatorMismatch { field: "beta" });
        }
        for inst in prior_instances {
            if inst.alpha.len() != log_n {
                return Err(VerifierError::AccumulatorMismatch {
                    field: "prior alpha",
                });
            }
            if inst.beta.len() != beta_len {
                return Err(VerifierError::AccumulatorMismatch {
                    field: "prior beta",
                });
            }
        }
        if proof.mu_fresh.len() != l1 {
            return Err(VerifierError::AccumulatorMismatch {
                field: "mu_fresh count",
            });
        }
        if proof.fresh_shift_answers.len() != self.params.num_shift_queries {
            return Err(VerifierError::ShiftQueryCount {
                expected: self.params.num_shift_queries,
                got: proof.fresh_shift_answers.len(),
            });
        }
        if proof.fresh_opening_proofs.len() != l1 {
            return Err(VerifierError::AccumulatorMismatch {
                field: "fresh batch proof count",
            });
        }
        for answers in &proof.fresh_shift_answers {
            if answers.len() != l1 {
                return Err(VerifierError::AccumulatorMismatch {
                    field: "fresh shift answer width",
                });
            }
        }
        if proof.acc_shift_answers.len() != l2 || proof.acc_merkle_proofs.len() != l2 {
            return Err(VerifierError::AccumulatorMismatch {
                field: "acc shift answers length",
            });
        }
        for answers in &proof.acc_shift_answers {
            if answers.len() != self.params.num_shift_queries {
                return Err(VerifierError::ShiftQueryCount {
                    expected: self.params.num_shift_queries,
                    got: answers.len(),
                });
            }
            for leaf in answers {
                if leaf.len() != 1 {
                    return Err(VerifierError::AccumulatorMismatch {
                        field: "acc shift answer leaf width",
                    });
                }
            }
        }
        if proof.nu_ood.len() != self.params.num_ood {
            return Err(VerifierError::AccumulatorMismatch { field: "nu_ood" });
        }

        bind_protocol::<F, _>(
            challenger,
            &self.pesat.description(),
            l1,
            l2,
            self.params.num_ood,
            self.params.num_shift_queries,
            log_n,
            log_h,
        );

        for c in fresh_commitments {
            fresh_verifier.observe_commitment(challenger, c);
        }
        for &mu_i in &proof.mu_fresh {
            challenger.observe_algebra_element(mu_i);
        }
        for inst in prior_instances {
            acc_backend.observe_commitment(challenger, &inst.rt);
            for &a in &inst.alpha {
                challenger.observe_algebra_element(a);
            }
            challenger.observe_algebra_element(inst.mu);
            for &b in &inst.beta {
                challenger.observe_algebra_element(b);
            }
            challenger.observe_algebra_element(inst.eta);
        }

        let fresh_taus: Vec<Vec<EF>> = (0..l1)
            .map(|_| {
                (0..log_m)
                    .map(|_| challenger.sample_algebra_element())
                    .collect()
            })
            .collect();

        let mut all_alphas: Vec<Vec<EF>> = (0..l1).map(|_| vec![EF::ZERO; log_n]).collect();
        let mut all_betas: Vec<Vec<EF>> = fresh_taus.clone();
        let mut all_mus: Vec<EF> = proof.mu_fresh.clone();
        let mut all_etas: Vec<EF> = vec![EF::ZERO; l1];
        for inst in prior_instances {
            all_alphas.push(inst.alpha.clone());
            all_betas.push(inst.beta.clone());
            all_mus.push(inst.mu);
            all_etas.push(inst.eta);
        }

        let omega: EF = challenger.sample_algebra_element();
        let tau: Vec<EF> = (0..log_l)
            .map(|_| challenger.sample_algebra_element())
            .collect();
        let tau_eq = Poly::<EF>::new_from_point(&tau, EF::ONE);
        let sigma_1: EF = (0..l)
            .map(|i| tau_eq.as_slice()[i] * (all_mus[i] + omega * all_etas[i]))
            .sum();

        let d1 = 1 + (log_n + 1).max(log_m + shape.max_degree);
        let (gamma_pt, twin_final_claim) = verify_sumcheck::<F, EF, _>(
            &proof.twin_constraint_sumcheck,
            challenger,
            sigma_1,
            log_l,
            d1,
            "twin-constraint",
        )?;
        let gamma = gamma_pt.as_slice().to_vec();

        acc_backend.observe_commitment(challenger, &new_instance.rt);
        challenger.observe_algebra_element(proof.nu_0);
        challenger.observe_algebra_element(proof.eta);

        let eq_tau_at_gamma: EF = Point::eval_eq(&tau, &gamma);
        let expected_twin_final = eq_tau_at_gamma * (proof.nu_0 + omega * proof.eta);
        if expected_twin_final != twin_final_claim {
            return Err(VerifierError::TwinConstraintFinalClaim);
        }

        let gamma_eq = Poly::<EF>::new_from_point(&gamma, EF::ONE);
        let mut expected_beta = vec![EF::ZERO; beta_len];
        for (i, betas_i) in all_betas.iter().enumerate() {
            let eq_i = gamma_eq.as_slice()[i];
            for (slot, &b) in expected_beta.iter_mut().zip(betas_i.iter()) {
                *slot += eq_i * b;
            }
        }
        if expected_beta != new_instance.beta {
            return Err(VerifierError::AccumulatorMismatch { field: "beta" });
        }
        if new_instance.eta != proof.eta {
            return Err(VerifierError::AccumulatorMismatch { field: "eta" });
        }

        let mut zetas: Vec<Vec<EF>> = Vec::with_capacity(self.params.r());
        let mut nus: Vec<EF> = Vec::with_capacity(self.params.r());
        let mut zeta_0: Vec<EF> = vec![EF::ZERO; log_n];
        for (i, alphas_i) in all_alphas.iter().enumerate() {
            let eq_i: EF = gamma_eq.as_slice()[i];
            for (slot, &a) in zeta_0.iter_mut().zip(alphas_i.iter()) {
                *slot += eq_i * a;
            }
        }
        zetas.push(zeta_0);
        nus.push(proof.nu_0);

        for k in 0..self.params.num_ood {
            let zeta_k: Vec<EF> = (0..log_n)
                .map(|_| challenger.sample_algebra_element())
                .collect();
            challenger.observe_algebra_element(proof.nu_ood[k]);
            zetas.push(zeta_k);
            nus.push(proof.nu_ood[k]);
        }

        let shift_indices =
            sample_indices::<F, _>(challenger, log_n, self.params.num_shift_queries);

        for (i, commitment_i) in fresh_commitments.iter().enumerate() {
            let values = proof
                .fresh_shift_answers
                .iter()
                .map(|answers| answers[i])
                .collect::<Vec<_>>();
            fresh_verifier
                .verify_batch_opening(
                    commitment_i,
                    log_n,
                    &shift_indices,
                    &values,
                    &proof.fresh_opening_proofs[i],
                )
                .map_err(|err| VerifierError::MerkleProof {
                    index: *shift_indices.first().unwrap_or(&0),
                    reason: format!("fresh commitment {i} batch opening failed: {err:?}"),
                })?;
        }

        for (j, inst) in prior_instances.iter().enumerate() {
            let values = proof.acc_shift_answers[j]
                .iter()
                .map(|leaf| leaf[0])
                .collect::<Vec<_>>();
            acc_backend
                .verify_batch_opening(
                    &inst.rt,
                    log_n,
                    &shift_indices,
                    &values,
                    &proof.acc_merkle_proofs[j],
                )
                .map_err(|err| VerifierError::MerkleProof {
                    index: *shift_indices.first().unwrap_or(&0),
                    reason: format!("prior acc {j} batch opening failed: {err:?}"),
                })?;
        }

        for (k_idx, &x_k) in shift_indices.iter().enumerate() {
            let mut nu_sk = EF::ZERO;
            for i in 0..l1 {
                nu_sk += gamma_eq.as_slice()[i] * EF::from(proof.fresh_shift_answers[k_idx][i]);
            }
            for j in 0..l2 {
                nu_sk += gamma_eq.as_slice()[l1 + j] * proof.acc_shift_answers[j][k_idx][0];
            }
            zetas.push(boolean_point::<EF>(x_k, log_n));
            nus.push(nu_sk);
        }

        let xi: Vec<EF> = (0..self.params.log_r())
            .map(|_| challenger.sample_algebra_element())
            .collect();
        let xi_eq_full = Poly::<EF>::new_from_point(&xi, EF::ONE);
        let r = self.params.r();
        let xi_eq: Vec<EF> = xi_eq_full.as_slice()[..r].to_vec();
        let sigma_2: EF = (0..r).map(|j| xi_eq[j] * nus[j]).sum();

        let (alpha_pt, batching_final_claim) = verify_sumcheck::<F, EF, _>(
            &proof.batching_sumcheck,
            challenger,
            sigma_2,
            log_n,
            2,
            "multilinear-batching",
        )?;
        challenger.observe_algebra_element(proof.mu_final);

        let alpha_slice = alpha_pt.as_slice();
        let mut eq_star_at_alpha = EF::ZERO;
        for (j, zeta_j) in zetas.iter().enumerate() {
            eq_star_at_alpha += xi_eq[j] * Point::eval_eq(zeta_j, alpha_slice);
        }
        let expected_batching_final = proof.mu_final * eq_star_at_alpha;
        if expected_batching_final != batching_final_claim {
            return Err(VerifierError::MultilinearBatchingFinalClaim);
        }

        if new_instance.alpha != alpha_slice.to_vec() {
            return Err(VerifierError::AccumulatorMismatch { field: "alpha" });
        }
        if new_instance.mu != proof.mu_final {
            return Err(VerifierError::AccumulatorMismatch { field: "mu" });
        }

        Ok(())
    }
}
