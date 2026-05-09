//! WARP prover: implements one accumulation step.
//!
//! Pipeline (Construction 10.4 specialised to RS):
//!
//! 1. **PESAT reduction (§5.10).** For each fresh witness `w_i ∈ F^k`,
//!    encode `f_i = C(w_i)`, set `µ_i = f̂_i(0)`, `α_i = 0`, `β_i = (τ_i, x_i) = τ_i`
//!    (κ = 0 in v1), `η_i = 0`. Stack the `ℓ_1` codewords as one Merkle
//!    commitment `rt_0`.
//!
//! 2. **Twin-constraint pseudo-batching (§6.3).** Sample `ω, τ`. Run
//!    `log ℓ` rounds of the high-degree sumcheck folding the `(F̂, ŵ, Â, B̂)`
//!    tables into a single merged `(f, w, ζ_0, β)`.
//!
//! 3. **Codeword + multilinear batching (§7.2 + §8.2).** Commit `f` over
//!    `EF`. Sample `s` OOD points `ζ_k` and answer with `ν_k = f̂(ζ_k)`.
//!    Sample `t` shift queries `x_k`, open the `ℓ` original codewords at
//!    each. Sample `ξ`, run `log n` rounds of degree-2 sumcheck on
//!    `eq*(X) · f̂(X) = σ⁽²⁾`.

use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{ExtensionMmcs, Mmcs};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{
    ExtensionField, Field, PackedFieldExtension, PackedValue, PrimeCharacteristicRing, TwoAdicField,
};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::eq_batch::eval_eq_batch;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use tracing::instrument;

use crate::accumulator::{
    Accumulator, AccumulatorInstance, AccumulatorWitness, WarpProof, WarpProofCommitted,
    WarpProofExternal, WarpProofExternalBatched,
};
use crate::code::ReedSolomonCode;
use crate::relation::claim_6_5::{
    fold_claim_6_5_packed_round, fold_claim_6_5_scalar_round, packed_ext_scalar_with_scratch,
    unpack_packed_coeffs_to_scalar,
};
use crate::relation::{BundledPesat, Claim65Scratch};
use crate::sumcheck::{SumcheckProof, observe_and_sample};
use crate::transcript::{bind_protocol, sample_indices};

use super::WarpParams;
use super::{
    AccumulatorBatchOpeningBackend, AccumulatorCommitmentBackend,
    ExternalCodewordBatchOpeningProver, ExternalCodewordOpeningProver, ExternalCommitmentObserver,
    ExternalCommittedCodeword,
};

/// Match WHIR's sumcheck split threshold: below this, Rayon overhead usually
/// dominates; above it, large linear folds benefit from parallel lanes.
const PAR_THRESHOLD: usize = 1 << 14;

struct OwnedPriorAccumulator<EF, Comm, ProverData> {
    instance: AccumulatorInstance<EF, Comm>,
    td: ProverData,
    f: Vec<EF>,
    w: Vec<EF>,
}

/// WARP prover bound to a specific PESAT, RS code, and Mmcs.
pub struct WarpProver<'a, F, EF, MT, Dft, Pesat>
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

/// A pre-committed fresh input for [`WarpProver::prove_with_committed`].
///
/// Carries one Reed-Solomon codeword along with its Merkle commitment and
/// prover-data, plus the underlying message/witness `w`. This realises the
/// **alphabet-`F` variant** of WARP Construction 10.4 (paper Theorem 10.3
/// proof, lines 3911-3915): the `ℓ_1` fresh codewords are committed
/// individually rather than via one stacked alphabet-`F^{ℓ_1}` tree.
///
/// Suitable when fresh codewords arrive pre-committed from upstream
/// pipelines, for example batched STARK proofs whose trace commitments are
/// RS-code Merkle roots.
///
/// Building one from a raw witness is a one-liner via
/// [`WarpProver::commit_witness`].
pub struct CommittedCodeword<F, MT>
where
    F: Field,
    MT: Mmcs<F>,
{
    /// The codeword `f = C(w)` as a `Vec<F>` of length `n`.
    pub codeword: Vec<F>,
    /// The witness `w` as a `Vec<F>` of length `k`.
    pub witness: Vec<F>,
    /// The Merkle commitment to the codeword (alphabet `F`).
    pub commitment: MT::Commitment,
    /// Merkle prover-data needed to open the codeword at shift indices.
    pub prover_data: MT::ProverData<RowMajorMatrix<F>>,
}

impl<'a, F, EF, MT, Dft, Pesat> WarpProver<'a, F, EF, MT, Dft, Pesat>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    MT: Mmcs<F> + Sync,
    Dft: TwoAdicSubgroupDft<F>,
    Pesat: BundledPesat<F, EF>,
{
    /// Create a prover.
    pub fn new(
        mmcs: &'a MT,
        code: &'a ReedSolomonCode<F, Dft>,
        pesat: &'a Pesat,
        params: WarpParams,
    ) -> Self {
        // Sanity: the RS message length equals the PESAT witness length.
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

    /// Prove one accumulation step.
    ///
    /// # Arguments
    ///
    /// - `challenger` — Fiat-Shamir state. Must be in identical state as the
    ///   verifier's challenger (typically: freshly initialised).
    /// - `fresh_witnesses` — `ℓ_1` PESAT witnesses, each of length `k`.
    /// - `prior_accumulators` — `ℓ_2` prior accumulators (their `rt` and `f`
    ///   are folded into the new acc).
    ///
    /// # Returns
    ///
    /// `((acc_instance, acc_witness), proof)`. The new `acc` is produced
    /// from the merged codeword, and `proof` is a record of the transcript
    /// that lets the verifier re-derive `acc_instance` independently.
    ///
    /// # Panics
    ///
    /// - `ℓ = ℓ_1 + ℓ_2` must be ≥ 2 and a power of 2.
    /// - Each `fresh_witnesses[i]` must have length `k` (= RS message length).
    /// - Each `prior_accumulators[j].witness.f` must have length `n` (RS
    ///   codeword length) and `.w` must have length `k`.
    #[instrument(skip_all, name = "warp::prove")]
    pub fn prove<Challenger>(
        &self,
        challenger: &mut Challenger,
        fresh_witnesses: &[Vec<F>],
        prior_accumulators: &[Accumulator<EF, MT::Commitment, ExtProverData<F, EF, MT>>],
    ) -> (
        Accumulator<EF, MT::Commitment, ExtProverData<F, EF, MT>>,
        WarpProof<F, EF, MT::Commitment, MT::Proof>,
    )
    where
        Challenger:
            FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<MT::Commitment>,
    {
        let l1 = fresh_witnesses.len();
        let l2 = prior_accumulators.len();
        let l = l1 + l2;
        assert!(
            l >= 2 && l.is_power_of_two(),
            "ℓ = ℓ_1 + ℓ_2 = {l} must be ≥ 2 and a power of two"
        );
        let log_l = l.trailing_zeros() as usize;

        let shape = self.pesat.shape();
        let log_m = shape.log_constraints;
        let log_n = self.code.log_codeword_len();
        let log_h = log_n - self.code.log_inv_rate();
        let n = self.code.codeword_len();
        let k = self.code.msg_len();
        let beta_len = shape.beta_len();
        assert_eq!(beta_len, log_m + shape.explicit_len);

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

        // ---------- 2. PESAT reduction (§5.10) on fresh witnesses. ----------
        for w in fresh_witnesses {
            assert_eq!(w.len(), k, "fresh witness length must be k = {k}");
        }
        let fresh_matrix = if l1 == 0 {
            RowMajorMatrix::new(Vec::new(), 1)
        } else {
            self.code.encode_batch(fresh_witnesses)
        };

        // µ_i = f̂_i(0^{log n}) = f_i[0].
        let mu_fresh: Vec<EF> = (0..l1).map(|i| EF::from(fresh_matrix.values[i])).collect();

        // Keep per-codeword columns for the twin-constraint folding tables.
        let fresh_codewords: Vec<Vec<F>> = (0..l1)
            .map(|i| {
                (0..n)
                    .map(|k_idx| fresh_matrix.values[k_idx * l1 + i])
                    .collect()
            })
            .collect();
        let (rt_0, td_0) = self.mmcs.commit_matrix(fresh_matrix);

        // ---------- 3. Observe rt_0, fresh µ_i, and prior accumulator instances. ----------
        challenger.observe(rt_0.clone());
        for &mu_i in &mu_fresh {
            challenger.observe_algebra_element(mu_i);
        }
        for acc in prior_accumulators {
            challenger.observe(acc.instance.rt.clone());
            for &a in &acc.instance.alpha {
                challenger.observe_algebra_element(a);
            }
            challenger.observe_algebra_element(acc.instance.mu);
            for &b in &acc.instance.beta {
                challenger.observe_algebra_element(b);
            }
            challenger.observe_algebra_element(acc.instance.eta);
        }

        // ---------- 4. Sample τ_i for each fresh PESAT (zerocheck randomness). ----------
        // Per §5.10: τ_i ∈ EF^{log M}, β_i = τ_i (κ = 0), η_i = 0, α_i = 0^{log n}.
        let fresh_taus: Vec<Vec<EF>> = (0..l1)
            .map(|_| {
                (0..log_m)
                    .map(|_| challenger.sample_algebra_element())
                    .collect()
            })
            .collect();

        // ---------- 5. Build the F̂, ŵ, Â, B̂ tables, lifting fresh F → EF. ----------
        let mut f_table: Vec<Vec<EF>> = Vec::with_capacity(l);
        let mut w_table: Vec<Vec<EF>> = Vec::with_capacity(l);
        let mut a_table: Vec<Vec<EF>> = Vec::with_capacity(l);
        let mut b_table: Vec<Vec<EF>> = Vec::with_capacity(l);
        for i in 0..l1 {
            f_table.push(fresh_codewords[i].iter().map(|&x| EF::from(x)).collect());
            w_table.push(fresh_witnesses[i].iter().map(|&x| EF::from(x)).collect());
            a_table.push(vec![EF::ZERO; log_n]); // α_i = 0
            b_table.push(fresh_taus[i].clone()); // β_i = τ_i (κ = 0)
        }
        for acc in prior_accumulators {
            assert_eq!(acc.witness.f.len(), n, "prior acc f has wrong length");
            assert_eq!(acc.witness.w.len(), k, "prior acc w has wrong length");
            assert_eq!(acc.instance.alpha.len(), log_n, "prior α length");
            assert_eq!(acc.instance.beta.len(), beta_len, "prior β length");
            f_table.push(acc.witness.f.clone());
            w_table.push(acc.witness.w.clone());
            a_table.push(acc.instance.alpha.clone());
            b_table.push(acc.instance.beta.clone());
        }

        // ---------- 6. Sample (ω, τ) for the §6.3 sumcheck. ----------
        let omega: EF = challenger.sample_algebra_element();
        let tau: Vec<EF> = (0..log_l)
            .map(|_| challenger.sample_algebra_element())
            .collect();

        // Initial sum σ⁽¹⁾ = Σ_i eq(τ, i) · (µ_i + ω · η_i).
        let mut all_mus: Vec<EF> = mu_fresh.clone();
        let mut all_etas: Vec<EF> = vec![EF::ZERO; l1];
        for acc in prior_accumulators {
            all_mus.push(acc.instance.mu);
            all_etas.push(acc.instance.eta);
        }
        let tau_eq_table = Poly::<EF>::new_from_point(&tau, EF::ONE);
        let mut current_eq: Vec<EF> = tau_eq_table.as_slice().to_vec();
        let _sigma_1: EF = (0..l)
            .map(|i| current_eq[i] * (all_mus[i] + omega * all_etas[i]))
            .sum();

        // ---------- 7. §6.3 twin-constraint sumcheck. ----------
        let d1 = self.twin_round_poly_degree(log_n, log_m, shape.max_degree);
        let mut twin_proof = SumcheckProof::<EF>::new();
        let mut gamma = Vec::with_capacity(log_l);
        for _ in 0..log_l {
            let coeffs = self.compute_twin_round_coeffs(
                d1,
                &f_table,
                &w_table,
                &a_table,
                &b_table,
                &current_eq,
                omega,
                log_m,
            );
            let g = observe_and_sample::<F, EF, _>(&mut twin_proof, challenger, coeffs);
            gamma.push(g);
            // Fold all tables and current_eq at γ.
            f_table = fold_table(f_table, g);
            w_table = fold_table(w_table, g);
            a_table = fold_table(a_table, g);
            b_table = fold_table(b_table, g);
            current_eq = fold_eq(current_eq, g);
        }
        debug_assert_eq!(f_table.len(), 1);

        // ---------- 8. Extract merged codeword, witness, and final claims. ----------
        let f_merged = f_table.into_iter().next().unwrap();
        let w_merged = w_table.into_iter().next().unwrap();
        let zeta_0 = a_table.into_iter().next().unwrap();
        let beta_final = b_table.into_iter().next().unwrap();

        let f_poly = Poly::<EF>::new(f_merged.clone());
        let zeta_0_pt = Point::<EF>::new(zeta_0.clone());
        let nu_0: EF = f_poly.eval_ext::<F>(&zeta_0_pt);

        let beta_tau = &beta_final[..log_m];
        let beta_x = &beta_final[log_m..];
        let mut z_for_eta = Vec::with_capacity(beta_x.len() + w_merged.len());
        z_for_eta.extend_from_slice(beta_x);
        z_for_eta.extend_from_slice(&w_merged);
        let beta_tau_eq = Poly::<EF>::new_from_point(beta_tau, EF::ONE);
        let eta = self
            .pesat
            .evaluate_bundled(beta_tau_eq.as_slice(), &z_for_eta);

        // ---------- 9. Commit the merged f over EF. ----------
        let ext_mmcs = ExtensionMmcs::<F, EF, MT>::new(self.mmcs.clone());
        let f_matrix = RowMajorMatrix::new(f_merged.clone(), 1);
        let (rt_merged, td_merged) = ext_mmcs.commit_matrix(f_matrix);

        challenger.observe(rt_merged.clone());
        challenger.observe_algebra_element(nu_0);
        challenger.observe_algebra_element(eta);

        // ---------- 10. §7.2 OOD samples + answers. ----------
        let s = self.params.num_ood;
        let t = self.params.num_shift_queries;
        let mut zetas: Vec<Vec<EF>> = Vec::with_capacity(1 + s + t);
        let mut nus: Vec<EF> = Vec::with_capacity(1 + s + t);
        zetas.push(zeta_0.clone());
        nus.push(nu_0);

        let mut nu_ood = Vec::with_capacity(s);
        for _ in 0..s {
            let zeta_k: Vec<EF> = (0..log_n)
                .map(|_| challenger.sample_algebra_element())
                .collect();
            let zeta_k_pt = Point::<EF>::new(zeta_k.clone());
            let nu_k: EF = f_poly.eval_ext::<F>(&zeta_k_pt);
            challenger.observe_algebra_element(nu_k);
            zetas.push(zeta_k);
            nus.push(nu_k);
            nu_ood.push(nu_k);
        }

        // ---------- 11. §7.2 shift queries (Boolean ζ_k from binary(x_k)). ----------
        let shift_indices = sample_indices::<F, _>(challenger, log_n, t);
        // Open rt_0 and each prior rt at each shift index.
        let mut fresh_shift_answers: Vec<Vec<F>> = Vec::with_capacity(t);
        let mut fresh_merkle_proofs: Vec<MT::Proof> = Vec::with_capacity(t);
        for &x_k in &shift_indices {
            let opening = self.mmcs.open_batch(x_k, &td_0);
            let (mut opened_values, opening_proof) = opening.unpack();
            // Single matrix → single Vec<F> of length ℓ_1.
            fresh_shift_answers.push(opened_values.swap_remove(0));
            fresh_merkle_proofs.push(opening_proof);
        }

        let mut acc_shift_answers: Vec<Vec<Vec<EF>>> = Vec::with_capacity(l2);
        let mut acc_merkle_proofs: Vec<Vec<MT::Proof>> = Vec::with_capacity(l2);
        for acc in prior_accumulators {
            let mut answers_for_acc = Vec::with_capacity(t);
            let mut proofs_for_acc = Vec::with_capacity(t);
            for &x_k in &shift_indices {
                let opening = <ExtensionMmcs<F, EF, MT> as Mmcs<EF>>::open_batch(
                    &ext_mmcs,
                    x_k,
                    &acc.witness.td,
                );
                let (mut vals, proof) = opening.unpack();
                answers_for_acc.push(vals.swap_remove(0));
                proofs_for_acc.push(proof);
            }
            acc_shift_answers.push(answers_for_acc);
            acc_merkle_proofs.push(proofs_for_acc);
        }

        // Compute ν_{s+k} = Σ_i eq(γ, i) · f_i(x_k) using the opened values.
        let gamma_eq = Poly::<EF>::new_from_point(&gamma, EF::ONE);
        let gamma_eq_slice = gamma_eq.as_slice();
        for (k_idx, &x_k) in shift_indices.iter().enumerate() {
            // Sum contributions from fresh codewords, then prior accumulators.
            let mut nu_sk = EF::ZERO;
            for i in 0..l1 {
                nu_sk += gamma_eq_slice[i] * EF::from(fresh_shift_answers[k_idx][i]);
            }
            for j in 0..l2 {
                // Prior acc's merged f at this position is acc_shift_answers[j][k_idx][0].
                nu_sk += gamma_eq_slice[l1 + j] * acc_shift_answers[j][k_idx][0];
            }
            // Also build the corresponding ζ_{s+k} = binary(x_k) ∈ {0,1}^{log n}.
            let zeta_sk = boolean_point::<EF>(x_k, log_n);
            zetas.push(zeta_sk);
            nus.push(nu_sk);
        }

        // ---------- 12. §8.2 multilinear constraint batching sumcheck. ----------
        let xi: Vec<EF> = (0..self.params.log_r())
            .map(|_| challenger.sample_algebra_element())
            .collect();
        // Truncate xi_eq to the first r entries (xi_eq table has length 2^log_r ≥ r).
        let xi_eq_full = Poly::<EF>::new_from_point(&xi, EF::ONE);
        let r = self.params.r();
        let xi_eq: Vec<EF> = xi_eq_full.as_slice()[..r].to_vec();

        // Build eq*(X) over {0,1}^{log n}: Σ_{j ∈ [r]} ξ_j · eq(ζ_j, X).
        // This is exactly the WARP §8.2 batching polynomial. Use Plonky3's
        // batched equality kernel instead of materializing one equality table
        // per ζ_j.
        let eq_star = weighted_eq_poly::<F, EF>(&zetas[..r], &xi_eq, log_n);
        // σ⁽²⁾ = Σ_j ξ_j · ν_j.
        let sigma_2: EF = (0..r).map(|j| xi_eq[j] * nus[j]).sum();

        // Run a degree-2 sumcheck on f̂(a) · eq*(a). Both sides are EF Polys.
        //
        // SIMD strategy: pack `f` and `eq*` into `EF::ExtensionPacking` form,
        // run rounds with packed arithmetic until the packed polynomial has
        // exactly one element (i.e., `log_n − log_W` rounds), then transition
        // to scalar for the final `log_W` rounds. Mirrors WHIR's
        // `ProductPolynomial::transition` pattern.
        let mut batching_proof = SumcheckProof::<EF>::new();
        let mut alpha_challenges = Vec::with_capacity(log_n);
        let pack_w = F::Packing::WIDTH;

        // Bail to scalar for tiny `n` where packing has more overhead than benefit.
        if f_poly.num_evals() < pack_w * 2 || f_poly.num_evals() % pack_w != 0 {
            let mut f_remaining = f_poly.clone();
            let mut eq_star_remaining = eq_star;
            for _ in 0..log_n {
                let evals = degree2_round_evals(&f_remaining, &eq_star_remaining);
                let coeffs = degree2_coeffs_from_evals(evals);
                let alpha = observe_and_sample::<F, EF, _>(&mut batching_proof, challenger, coeffs);
                alpha_challenges.push(alpha);
                f_remaining.fix_prefix_var_mut(alpha);
                eq_star_remaining.fix_prefix_var_mut(alpha);
            }
            debug_assert_eq!(f_remaining.num_evals(), 1);
            let mu_final = f_remaining.as_slice()[0];
            challenger.observe_algebra_element(mu_final);
            // Build acc + proof below.
            let acc_instance = AccumulatorInstance {
                rt: rt_merged.clone(),
                alpha: alpha_challenges,
                mu: mu_final,
                beta: beta_final,
                eta,
            };
            let acc_witness = AccumulatorWitness {
                td: td_merged,
                f: f_merged,
                w: w_merged,
            };
            let proof = WarpProof {
                rt_0,
                mu_fresh,
                twin_constraint_sumcheck: twin_proof,
                nu_0,
                eta,
                nu_ood,
                batching_sumcheck: batching_proof,
                mu_final,
                fresh_shift_answers,
                fresh_merkle_proofs,
                acc_shift_answers,
                acc_merkle_proofs,
            };
            let _ = (sigma_2, _sigma_1);
            let _ = nus;
            return (
                Accumulator {
                    instance: acc_instance,
                    witness: acc_witness,
                },
                proof,
            );
        }

        // Packed path: pack both polys, run packed rounds, then transition.
        let log_pack = pack_w.trailing_zeros() as usize;
        let n_packed_rounds = log_n.saturating_sub(log_pack);
        let mut f_packed: Vec<EF::ExtensionPacking> = pack_ef_slice::<F, EF>(f_poly.as_slice());
        let mut eq_packed: Vec<EF::ExtensionPacking> = pack_ef_slice::<F, EF>(eq_star.as_slice());

        for _ in 0..n_packed_rounds {
            let evals = degree2_round_evals_packed::<F, EF>(&f_packed, &eq_packed);
            let coeffs = degree2_coeffs_from_evals(evals);
            let alpha = observe_and_sample::<F, EF, _>(&mut batching_proof, challenger, coeffs);
            alpha_challenges.push(alpha);
            fold_packed_in_place::<F, EF>(&mut f_packed, alpha);
            fold_packed_in_place::<F, EF>(&mut eq_packed, alpha);
        }

        // Transition: unpack the single remaining packed element into `pack_w`
        // scalar `EF` values. Continue scalar rounds.
        debug_assert_eq!(f_packed.len(), 1);
        let f_scalar: Vec<EF> =
            EF::ExtensionPacking::to_ext_iter(f_packed.iter().copied()).collect();
        let eq_scalar: Vec<EF> =
            EF::ExtensionPacking::to_ext_iter(eq_packed.iter().copied()).collect();
        let mut f_remaining = Poly::<EF>::new(f_scalar);
        let mut eq_star_remaining = Poly::<EF>::new(eq_scalar);

        for _ in 0..log_pack {
            let evals = degree2_round_evals(&f_remaining, &eq_star_remaining);
            let coeffs = degree2_coeffs_from_evals(evals);
            let alpha = observe_and_sample::<F, EF, _>(&mut batching_proof, challenger, coeffs);
            alpha_challenges.push(alpha);
            f_remaining.fix_prefix_var_mut(alpha);
            eq_star_remaining.fix_prefix_var_mut(alpha);
        }
        debug_assert_eq!(f_remaining.num_evals(), 1);
        let mu_final = f_remaining.as_slice()[0];
        challenger.observe_algebra_element(mu_final);

        // ---------- 13. Build new accumulator + proof. ----------
        let acc_instance = AccumulatorInstance {
            rt: rt_merged.clone(),
            alpha: alpha_challenges,
            mu: mu_final,
            beta: beta_final,
            eta,
        };
        let acc_witness = AccumulatorWitness {
            td: td_merged,
            f: f_merged,
            w: w_merged,
        };
        let proof = WarpProof {
            rt_0,
            mu_fresh,
            twin_constraint_sumcheck: twin_proof,
            nu_0,
            eta,
            nu_ood,
            batching_sumcheck: batching_proof,
            mu_final,
            fresh_shift_answers,
            fresh_merkle_proofs,
            acc_shift_answers,
            acc_merkle_proofs,
        };
        // Suppress unused warnings for derivation-only state.
        let _ = (sigma_2, _sigma_1);
        let _ = nus;

        (
            Accumulator {
                instance: acc_instance,
                witness: acc_witness,
            },
            proof,
        )
    }

    /// Encode a raw witness and commit the resulting codeword via `mmcs`,
    /// returning a [`CommittedCodeword`] suitable as input to
    /// [`Self::prove_with_committed`].
    ///
    /// This is just `code.encode(w)` followed by
    /// `mmcs.commit_matrix(RowMajorMatrix::new(codeword, 1))`. Provided
    /// for convenience and to make the symmetry with [`Self::prove`]
    /// obvious in tests:
    ///
    /// ```ignore
    /// // The two are equivalent (modulo FS sequencing — see below):
    /// let (acc1, proof1) = prover.prove(&mut ch, &[w], &[]);
    ///
    /// let committed = vec![prover.commit_witness(w)];
    /// let (acc2, proof2) = prover.prove_with_committed(&mut ch, committed, &[]);
    /// ```
    ///
    /// In precommitted usage the `CommittedCodeword` may come directly from an
    /// upstream PCS. This helper exists for benches and tests where the input
    /// starts as a raw `Vec<F>` witness.
    pub fn commit_witness(&self, witness: Vec<F>) -> CommittedCodeword<F, MT> {
        let k = self.code.msg_len();
        assert_eq!(witness.len(), k, "witness length must equal k = {k}");
        let codeword = self.code.encode(&witness);
        let codeword_matrix = RowMajorMatrix::new(codeword.clone(), 1);
        let (commitment, prover_data) = self.mmcs.commit_matrix(codeword_matrix);
        CommittedCodeword {
            codeword,
            witness,
            commitment,
            prover_data,
        }
    }

    /// Prove one accumulation step using **pre-committed** fresh inputs.
    ///
    /// This realises the **alphabet-`F` variant** of WARP Construction 10.4
    /// (paper Theorem 10.3 proof, lines 3911-3915): the `ℓ_1` fresh
    /// codewords are committed individually rather than via one stacked
    /// alphabet-`F^{ℓ_1}` tree.
    ///
    /// Compared to [`Self::prove`]:
    /// - **No fresh-codeword Merkle commit is built by this function.** The
    ///   caller supplies the commitments inside [`CommittedCodeword`]. This
    ///   eliminates one of the two Merkle trees per step (the dominant
    ///   per-step prover cost — see `warp/docs/zkvm-pipeline.md`). Saves
    ///   ~50% of step Merkle work in pipelines that already pre-commit
    ///   their inputs.
    /// - The proof type is [`WarpProofCommitted`]: it omits `rt_0` (the
    ///   verifier holds the `ℓ_1` external commitments) and carries one
    ///   Merkle path **per `(shift, fresh)` pair** instead of one path
    ///   per shift into the stacked tree.
    ///
    /// FS observation order: `bind` → each fresh `commitment_i` (in input
    /// order) → each `µ_i` → priors → sample `τ`s. Same overall sequence
    /// as [`Self::prove`], with the single stacked-root observation
    /// replaced by `ℓ_1` individual root observations.
    ///
    /// # Panics
    ///
    /// - `ℓ = ℓ_1 + ℓ_2` must be ≥ 2 and a power of 2.
    /// - Each `fresh_committed[i].codeword.len()` must equal `n`.
    /// - Each `fresh_committed[i].witness.len()` must equal `k`.
    #[instrument(skip_all, name = "warp::prove_with_committed")]
    pub fn prove_with_committed<Challenger>(
        &self,
        challenger: &mut Challenger,
        fresh_committed: Vec<CommittedCodeword<F, MT>>,
        prior_accumulators: &[Accumulator<EF, MT::Commitment, ExtProverData<F, EF, MT>>],
    ) -> (
        Accumulator<EF, MT::Commitment, ExtProverData<F, EF, MT>>,
        WarpProofCommitted<F, EF, MT::Commitment, MT::Proof>,
    )
    where
        Challenger:
            FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<MT::Commitment>,
    {
        let (acc, proof) = self.prove_with_external_committed(
            challenger,
            self.mmcs,
            fresh_committed,
            prior_accumulators,
        );
        (
            acc,
            WarpProofCommitted {
                mu_fresh: proof.mu_fresh,
                twin_constraint_sumcheck: proof.twin_constraint_sumcheck,
                nu_0: proof.nu_0,
                eta: proof.eta,
                nu_ood: proof.nu_ood,
                batching_sumcheck: proof.batching_sumcheck,
                mu_final: proof.mu_final,
                fresh_shift_answers: proof.fresh_shift_answers,
                fresh_merkle_proofs: proof.fresh_opening_proofs,
                acc_shift_answers: proof.acc_shift_answers,
                acc_merkle_proofs: proof.acc_merkle_proofs,
                _ph: PhantomData,
            },
        )
    }

    /// Prove one accumulation step using fresh inputs committed by an
    /// arbitrary external PCS backend.
    ///
    /// The external backend is responsible for opening and later verifying
    /// the fresh codewords. The merged accumulator commitment remains the
    /// local Plonky3 `Mmcs`; use
    /// [`prove_with_external_committed_accumulator`](Self::prove_with_external_committed_accumulator)
    /// when the accumulator itself must use an external commitment layout.
    #[instrument(skip_all, name = "warp::prove_with_external_committed")]
    pub fn prove_with_external_committed<Challenger, Fresh, FreshOpenings>(
        &self,
        challenger: &mut Challenger,
        fresh_openings: &FreshOpenings,
        fresh_committed: Vec<Fresh>,
        prior_accumulators: &[Accumulator<EF, MT::Commitment, ExtProverData<F, EF, MT>>],
    ) -> (
        Accumulator<EF, MT::Commitment, ExtProverData<F, EF, MT>>,
        WarpProofExternal<F, EF, MT::Commitment, FreshOpenings::Proof, MT::Proof>,
    )
    where
        Challenger:
            FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<MT::Commitment>,
        Fresh: ExternalCommittedCodeword<F> + ExternalCommitmentObserver<F, Challenger>,
        FreshOpenings: ExternalCodewordOpeningProver<F, Fresh>,
    {
        let acc_backend = ExtensionMmcs::<F, EF, MT>::new(self.mmcs.clone());
        self.prove_with_external_committed_accumulator(
            challenger,
            fresh_openings,
            &acc_backend,
            fresh_committed,
            prior_accumulators,
        )
    }

    /// Prove one accumulation step using external PCS backends for both fresh
    /// inputs and accumulated WARP codewords.
    ///
    /// This entry point lets fresh codewords and every new accumulator use
    /// caller-provided commitment backends so subsequent WARP steps and the
    /// root proof see the same canonical layout.
    #[instrument(skip_all, name = "warp::prove_with_external_committed_accumulator")]
    pub fn prove_with_external_committed_accumulator<Challenger, Fresh, FreshOpenings, AccBackend>(
        &self,
        challenger: &mut Challenger,
        fresh_openings: &FreshOpenings,
        acc_backend: &AccBackend,
        fresh_committed: Vec<Fresh>,
        prior_accumulators: &[Accumulator<EF, AccBackend::Commitment, AccBackend::ProverData>],
    ) -> (
        Accumulator<EF, AccBackend::Commitment, AccBackend::ProverData>,
        WarpProofExternal<F, EF, AccBackend::Commitment, FreshOpenings::Proof, AccBackend::Proof>,
    )
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
        Fresh: ExternalCommittedCodeword<F> + ExternalCommitmentObserver<F, Challenger>,
        FreshOpenings: ExternalCodewordOpeningProver<F, Fresh>,
        AccBackend: AccumulatorCommitmentBackend<F, EF, Challenger>,
    {
        let l1 = fresh_committed.len();
        let l2 = prior_accumulators.len();
        let l = l1 + l2;
        assert!(
            l >= 2 && l.is_power_of_two(),
            "ℓ = ℓ_1 + ℓ_2 = {l} must be ≥ 2 and a power of two"
        );
        let log_l = l.trailing_zeros() as usize;

        let shape = self.pesat.shape();
        let log_m = shape.log_constraints;
        let log_n = self.code.log_codeword_len();
        let log_h = log_n - self.code.log_inv_rate();
        let n = self.code.codeword_len();
        let k = self.code.msg_len();
        let beta_len = shape.beta_len();
        assert_eq!(beta_len, log_m + shape.explicit_len);

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

        // ---------- 2. PESAT reduction (§5.10) on fresh inputs. ----------
        // VARIANT vs `prove`: no encode + no stacked commit; the codewords
        // and per-fresh commitments are supplied by the caller.
        for c in &fresh_committed {
            assert_eq!(
                c.codeword().len(),
                n,
                "fresh codeword length must be n = {n}"
            );
            assert_eq!(c.witness().len(), k, "fresh witness length must be k = {k}");
        }

        // µ_i = f̂_i(0^{log n}) = f_i[0].
        let mu_fresh: Vec<EF> = fresh_committed
            .iter()
            .map(|c| EF::from(c.codeword()[0]))
            .collect();

        // ---------- 3. Observe per-fresh commitments + µ_i + priors. ----------
        // VARIANT: ℓ_1 individual fresh commitments instead of one stacked rt_0.
        for c in &fresh_committed {
            c.observe_commitment(challenger);
        }
        for &mu_i in &mu_fresh {
            challenger.observe_algebra_element(mu_i);
        }
        for acc in prior_accumulators {
            acc_backend.observe_commitment(challenger, &acc.instance.rt);
            for &a in &acc.instance.alpha {
                challenger.observe_algebra_element(a);
            }
            challenger.observe_algebra_element(acc.instance.mu);
            for &b in &acc.instance.beta {
                challenger.observe_algebra_element(b);
            }
            challenger.observe_algebra_element(acc.instance.eta);
        }

        // ---------- 4. Sample τ_i for each fresh PESAT. ----------
        let fresh_taus: Vec<Vec<EF>> = (0..l1)
            .map(|_| {
                (0..log_m)
                    .map(|_| challenger.sample_algebra_element())
                    .collect()
            })
            .collect();

        // ---------- 5. Build the F̂, ŵ, Â, B̂ tables, lifting fresh F → EF. ----------
        let mut f_table: Vec<Vec<EF>> = Vec::with_capacity(l);
        let mut w_table: Vec<Vec<EF>> = Vec::with_capacity(l);
        let mut a_table: Vec<Vec<EF>> = Vec::with_capacity(l);
        let mut b_table: Vec<Vec<EF>> = Vec::with_capacity(l);
        for (i, c) in fresh_committed.iter().enumerate() {
            f_table.push(c.codeword().iter().map(|&x| EF::from(x)).collect());
            w_table.push(c.witness().iter().map(|&x| EF::from(x)).collect());
            a_table.push(vec![EF::ZERO; log_n]); // α_i = 0
            b_table.push(fresh_taus[i].clone()); // β_i = τ_i (κ = 0)
        }
        for acc in prior_accumulators {
            assert_eq!(acc.witness.f.len(), n, "prior acc f has wrong length");
            assert_eq!(acc.witness.w.len(), k, "prior acc w has wrong length");
            assert_eq!(acc.instance.alpha.len(), log_n, "prior α length");
            assert_eq!(acc.instance.beta.len(), beta_len, "prior β length");
            f_table.push(acc.witness.f.clone());
            w_table.push(acc.witness.w.clone());
            a_table.push(acc.instance.alpha.clone());
            b_table.push(acc.instance.beta.clone());
        }

        // ---------- 6. Sample (ω, τ) for the §6.3 sumcheck. ----------
        let omega: EF = challenger.sample_algebra_element();
        let tau: Vec<EF> = (0..log_l)
            .map(|_| challenger.sample_algebra_element())
            .collect();

        // Initial sum σ⁽¹⁾ = Σ_i eq(τ, i) · (µ_i + ω · η_i).
        let mut all_mus: Vec<EF> = mu_fresh.clone();
        let mut all_etas: Vec<EF> = vec![EF::ZERO; l1];
        for acc in prior_accumulators {
            all_mus.push(acc.instance.mu);
            all_etas.push(acc.instance.eta);
        }
        let tau_eq_table = Poly::<EF>::new_from_point(&tau, EF::ONE);
        let mut current_eq: Vec<EF> = tau_eq_table.as_slice().to_vec();
        let _sigma_1: EF = (0..l)
            .map(|i| current_eq[i] * (all_mus[i] + omega * all_etas[i]))
            .sum();

        // ---------- 7. §6.3 twin-constraint sumcheck. ----------
        let d1 = self.twin_round_poly_degree(log_n, log_m, shape.max_degree);
        let mut twin_proof = SumcheckProof::<EF>::new();
        let mut gamma = Vec::with_capacity(log_l);
        for _ in 0..log_l {
            let coeffs = self.compute_twin_round_coeffs(
                d1,
                &f_table,
                &w_table,
                &a_table,
                &b_table,
                &current_eq,
                omega,
                log_m,
            );
            let g = observe_and_sample::<F, EF, _>(&mut twin_proof, challenger, coeffs);
            gamma.push(g);
            f_table = fold_table(f_table, g);
            w_table = fold_table(w_table, g);
            a_table = fold_table(a_table, g);
            b_table = fold_table(b_table, g);
            current_eq = fold_eq(current_eq, g);
        }
        debug_assert_eq!(f_table.len(), 1);

        // ---------- 8. Extract merged codeword, witness, and final claims. ----------
        let f_merged = f_table.into_iter().next().unwrap();
        let w_merged = w_table.into_iter().next().unwrap();
        let zeta_0 = a_table.into_iter().next().unwrap();
        let beta_final = b_table.into_iter().next().unwrap();

        let f_poly = Poly::<EF>::new(f_merged.clone());
        let zeta_0_pt = Point::<EF>::new(zeta_0.clone());
        let nu_0: EF = f_poly.eval_ext::<F>(&zeta_0_pt);

        let beta_tau = &beta_final[..log_m];
        let beta_x = &beta_final[log_m..];
        let mut z_for_eta = Vec::with_capacity(beta_x.len() + w_merged.len());
        z_for_eta.extend_from_slice(beta_x);
        z_for_eta.extend_from_slice(&w_merged);
        let beta_tau_eq = Poly::<EF>::new_from_point(beta_tau, EF::ONE);
        let eta = self
            .pesat
            .evaluate_bundled(beta_tau_eq.as_slice(), &z_for_eta);

        // ---------- 9. Commit the merged f over EF (this is the ONLY new
        //               accumulator commit per step in this variant). ----------
        let (rt_merged, td_merged) = acc_backend
            .commit_with_message(f_merged.clone(), &w_merged)
            .unwrap_or_else(|err| panic!("accumulator commitment failed: {err:?}"));

        acc_backend.observe_commitment(challenger, &rt_merged);
        challenger.observe_algebra_element(nu_0);
        challenger.observe_algebra_element(eta);

        // ---------- 10. §7.2 OOD samples + answers. ----------
        let s = self.params.num_ood;
        let t = self.params.num_shift_queries;
        let mut zetas: Vec<Vec<EF>> = Vec::with_capacity(1 + s + t);
        let mut nus: Vec<EF> = Vec::with_capacity(1 + s + t);
        zetas.push(zeta_0.clone());
        nus.push(nu_0);

        let mut nu_ood = Vec::with_capacity(s);
        for _ in 0..s {
            let zeta_k: Vec<EF> = (0..log_n)
                .map(|_| challenger.sample_algebra_element())
                .collect();
            let zeta_k_pt = Point::<EF>::new(zeta_k.clone());
            let nu_k: EF = f_poly.eval_ext::<F>(&zeta_k_pt);
            challenger.observe_algebra_element(nu_k);
            zetas.push(zeta_k);
            nus.push(nu_k);
            nu_ood.push(nu_k);
        }

        // ---------- 11. §7.2 shift queries. ----------
        let shift_indices = sample_indices::<F, _>(challenger, log_n, t);

        // VARIANT: open each fresh codeword's individual tree separately at
        // each shift index, producing one Merkle path per (shift, fresh)
        // pair. Reuses `mmcs.open_batch` directly on the per-fresh
        // `prover_data`.
        let mut fresh_shift_answers: Vec<Vec<F>> = Vec::with_capacity(t);
        let mut fresh_opening_proofs: Vec<Vec<FreshOpenings::Proof>> = Vec::with_capacity(t);
        for &x_k in &shift_indices {
            let mut answers_for_xk: Vec<F> = Vec::with_capacity(l1);
            let mut proofs_for_xk: Vec<FreshOpenings::Proof> = Vec::with_capacity(l1);
            for c in &fresh_committed {
                let (cell, proof) = fresh_openings
                    .open(c, x_k)
                    .unwrap_or_else(|err| panic!("external fresh opening failed: {err:?}"));
                answers_for_xk.push(cell);
                proofs_for_xk.push(proof);
            }
            fresh_shift_answers.push(answers_for_xk);
            fresh_opening_proofs.push(proofs_for_xk);
        }

        // Prior accumulator openings use the same accumulator backend that
        // committed previous WARP steps.
        let mut acc_shift_answers: Vec<Vec<Vec<EF>>> = Vec::with_capacity(l2);
        let mut acc_merkle_proofs: Vec<Vec<AccBackend::Proof>> = Vec::with_capacity(l2);
        for acc in prior_accumulators {
            let mut answers_for_acc = Vec::with_capacity(t);
            let mut proofs_for_acc = Vec::with_capacity(t);
            for &x_k in &shift_indices {
                let (cell, proof) = acc_backend
                    .open(&acc.witness.td, x_k)
                    .unwrap_or_else(|err| panic!("accumulator opening failed: {err:?}"));
                answers_for_acc.push(vec![cell]);
                proofs_for_acc.push(proof);
            }
            acc_shift_answers.push(answers_for_acc);
            acc_merkle_proofs.push(proofs_for_acc);
        }

        // Compute ν_{s+k} using the opened values.
        let gamma_eq = Poly::<EF>::new_from_point(&gamma, EF::ONE);
        let gamma_eq_slice = gamma_eq.as_slice();
        for (k_idx, &x_k) in shift_indices.iter().enumerate() {
            let mut nu_sk = EF::ZERO;
            for i in 0..l1 {
                nu_sk += gamma_eq_slice[i] * EF::from(fresh_shift_answers[k_idx][i]);
            }
            for j in 0..l2 {
                nu_sk += gamma_eq_slice[l1 + j] * acc_shift_answers[j][k_idx][0];
            }
            let zeta_sk = boolean_point::<EF>(x_k, log_n);
            zetas.push(zeta_sk);
            nus.push(nu_sk);
        }

        // ---------- 12. §8.2 multilinear constraint batching sumcheck. ----------
        // Identical to `prove` from here.
        let xi: Vec<EF> = (0..self.params.log_r())
            .map(|_| challenger.sample_algebra_element())
            .collect();
        let xi_eq_full = Poly::<EF>::new_from_point(&xi, EF::ONE);
        let r = self.params.r();
        let xi_eq: Vec<EF> = xi_eq_full.as_slice()[..r].to_vec();

        let eq_star = weighted_eq_poly::<F, EF>(&zetas[..r], &xi_eq, log_n);
        let sigma_2: EF = (0..r).map(|j| xi_eq[j] * nus[j]).sum();

        let mut batching_proof = SumcheckProof::<EF>::new();
        let mut alpha_challenges = Vec::with_capacity(log_n);
        let pack_w = F::Packing::WIDTH;

        if f_poly.num_evals() < pack_w * 2 || f_poly.num_evals() % pack_w != 0 {
            // Scalar fallback — same as `prove`.
            let mut f_remaining = f_poly.clone();
            let mut eq_star_remaining = eq_star;
            for _ in 0..log_n {
                let evals = degree2_round_evals(&f_remaining, &eq_star_remaining);
                let coeffs = degree2_coeffs_from_evals(evals);
                let alpha = observe_and_sample::<F, EF, _>(&mut batching_proof, challenger, coeffs);
                alpha_challenges.push(alpha);
                f_remaining.fix_prefix_var_mut(alpha);
                eq_star_remaining.fix_prefix_var_mut(alpha);
            }
            debug_assert_eq!(f_remaining.num_evals(), 1);
            let mu_final = f_remaining.as_slice()[0];
            challenger.observe_algebra_element(mu_final);
            let acc_instance = AccumulatorInstance {
                rt: rt_merged.clone(),
                alpha: alpha_challenges,
                mu: mu_final,
                beta: beta_final,
                eta,
            };
            let acc_witness = AccumulatorWitness {
                td: td_merged,
                f: f_merged,
                w: w_merged,
            };
            let proof = WarpProofExternal {
                mu_fresh,
                twin_constraint_sumcheck: twin_proof,
                nu_0,
                eta,
                nu_ood,
                batching_sumcheck: batching_proof,
                mu_final,
                fresh_shift_answers,
                fresh_opening_proofs,
                acc_shift_answers,
                acc_merkle_proofs,
                _ph: PhantomData,
            };
            let _ = (sigma_2, _sigma_1);
            let _ = nus;
            return (
                Accumulator {
                    instance: acc_instance,
                    witness: acc_witness,
                },
                proof,
            );
        }

        // Packed §8.2 path — same as `prove`.
        let log_pack = pack_w.trailing_zeros() as usize;
        let n_packed_rounds = log_n.saturating_sub(log_pack);
        let mut f_packed: Vec<EF::ExtensionPacking> = pack_ef_slice::<F, EF>(f_poly.as_slice());
        let mut eq_packed: Vec<EF::ExtensionPacking> = pack_ef_slice::<F, EF>(eq_star.as_slice());

        for _ in 0..n_packed_rounds {
            let evals = degree2_round_evals_packed::<F, EF>(&f_packed, &eq_packed);
            let coeffs = degree2_coeffs_from_evals(evals);
            let alpha = observe_and_sample::<F, EF, _>(&mut batching_proof, challenger, coeffs);
            alpha_challenges.push(alpha);
            fold_packed_in_place::<F, EF>(&mut f_packed, alpha);
            fold_packed_in_place::<F, EF>(&mut eq_packed, alpha);
        }

        debug_assert_eq!(f_packed.len(), 1);
        let f_scalar: Vec<EF> =
            EF::ExtensionPacking::to_ext_iter(f_packed.iter().copied()).collect();
        let eq_scalar: Vec<EF> =
            EF::ExtensionPacking::to_ext_iter(eq_packed.iter().copied()).collect();
        let mut f_remaining = Poly::<EF>::new(f_scalar);
        let mut eq_star_remaining = Poly::<EF>::new(eq_scalar);

        for _ in 0..log_pack {
            let evals = degree2_round_evals(&f_remaining, &eq_star_remaining);
            let coeffs = degree2_coeffs_from_evals(evals);
            let alpha = observe_and_sample::<F, EF, _>(&mut batching_proof, challenger, coeffs);
            alpha_challenges.push(alpha);
            f_remaining.fix_prefix_var_mut(alpha);
            eq_star_remaining.fix_prefix_var_mut(alpha);
        }
        debug_assert_eq!(f_remaining.num_evals(), 1);
        let mu_final = f_remaining.as_slice()[0];
        challenger.observe_algebra_element(mu_final);

        // ---------- 13. Build new accumulator + proof. ----------
        let acc_instance = AccumulatorInstance {
            rt: rt_merged.clone(),
            alpha: alpha_challenges,
            mu: mu_final,
            beta: beta_final,
            eta,
        };
        let acc_witness = AccumulatorWitness {
            td: td_merged,
            f: f_merged,
            w: w_merged,
        };
        let proof = WarpProofExternal {
            mu_fresh,
            twin_constraint_sumcheck: twin_proof,
            nu_0,
            eta,
            nu_ood,
            batching_sumcheck: batching_proof,
            mu_final,
            fresh_shift_answers,
            fresh_opening_proofs,
            acc_shift_answers,
            acc_merkle_proofs,
            _ph: PhantomData,
        };
        let _ = (sigma_2, _sigma_1);
        let _ = nus;

        (
            Accumulator {
                instance: acc_instance,
                witness: acc_witness,
            },
            proof,
        )
    }

    /// Prove one accumulation step using batched external PCS openings for
    /// the sampled WARP shift queries.
    ///
    /// This follows the same transcript as
    /// [`Self::prove_with_external_committed_accumulator`], but asks each
    /// backend for one proof over all sampled shift indices. With WHIR this
    /// maps to one `open_deferred` call per fresh codeword and one per prior
    /// accumulator limb, instead of one call per `(shift, codeword)` pair.
    #[instrument(
        skip_all,
        name = "warp::prove_with_external_committed_accumulator_batched"
    )]
    pub fn prove_with_external_committed_accumulator_batched<
        Challenger,
        Fresh,
        FreshOpenings,
        AccBackend,
    >(
        &self,
        challenger: &mut Challenger,
        fresh_openings: &FreshOpenings,
        acc_backend: &AccBackend,
        fresh_committed: Vec<Fresh>,
        prior_accumulators: Vec<Accumulator<EF, AccBackend::Commitment, AccBackend::ProverData>>,
    ) -> (
        Accumulator<EF, AccBackend::Commitment, AccBackend::ProverData>,
        WarpProofExternalBatched<
            F,
            EF,
            AccBackend::Commitment,
            FreshOpenings::BatchProof,
            AccBackend::BatchProof,
        >,
    )
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
        Fresh: ExternalCommittedCodeword<F> + ExternalCommitmentObserver<F, Challenger>,
        FreshOpenings: ExternalCodewordBatchOpeningProver<F, Fresh>,
        AccBackend: AccumulatorBatchOpeningBackend<F, EF, Challenger>,
    {
        let l1 = fresh_committed.len();
        let mut prior_accumulators = prior_accumulators
            .into_iter()
            .map(|acc| OwnedPriorAccumulator {
                instance: acc.instance,
                td: acc.witness.td,
                f: acc.witness.f,
                w: acc.witness.w,
            })
            .collect::<Vec<_>>();
        let l2 = prior_accumulators.len();
        let l = l1 + l2;
        assert!(
            l >= 2 && l.is_power_of_two(),
            "ℓ = ℓ_1 + ℓ_2 = {l} must be ≥ 2 and a power of two"
        );
        let log_l = l.trailing_zeros() as usize;

        let shape = self.pesat.shape();
        let log_m = shape.log_constraints;
        let log_n = self.code.log_codeword_len();
        let log_h = log_n - self.code.log_inv_rate();
        let n = self.code.codeword_len();
        let k = self.code.msg_len();
        let beta_len = shape.beta_len();
        assert_eq!(beta_len, log_m + shape.explicit_len);

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

        for c in &fresh_committed {
            assert_eq!(
                c.codeword().len(),
                n,
                "fresh codeword length must be n = {n}"
            );
            assert_eq!(c.witness().len(), k, "fresh witness length must be k = {k}");
        }

        let mu_fresh: Vec<EF> = fresh_committed
            .iter()
            .map(|c| EF::from(c.codeword()[0]))
            .collect();

        for c in &fresh_committed {
            c.observe_commitment(challenger);
        }
        for &mu_i in &mu_fresh {
            challenger.observe_algebra_element(mu_i);
        }
        for acc in &prior_accumulators {
            acc_backend.observe_commitment(challenger, &acc.instance.rt);
            for &a in &acc.instance.alpha {
                challenger.observe_algebra_element(a);
            }
            challenger.observe_algebra_element(acc.instance.mu);
            for &b in &acc.instance.beta {
                challenger.observe_algebra_element(b);
            }
            challenger.observe_algebra_element(acc.instance.eta);
        }

        let fresh_taus: Vec<Vec<EF>> = (0..l1)
            .map(|_| {
                (0..log_m)
                    .map(|_| challenger.sample_algebra_element())
                    .collect()
            })
            .collect();

        let mut f_table: Vec<Vec<EF>> = Vec::with_capacity(l);
        let mut w_table: Vec<Vec<EF>> = Vec::with_capacity(l);
        let mut a_table: Vec<Vec<EF>> = Vec::with_capacity(l);
        let mut b_table: Vec<Vec<EF>> = Vec::with_capacity(l);
        for (i, c) in fresh_committed.iter().enumerate() {
            f_table.push(c.codeword().iter().map(|&x| EF::from(x)).collect());
            w_table.push(c.witness().iter().map(|&x| EF::from(x)).collect());
            a_table.push(vec![EF::ZERO; log_n]);
            b_table.push(fresh_taus[i].clone());
        }
        for acc in &mut prior_accumulators {
            assert_eq!(acc.f.len(), n, "prior acc f has wrong length");
            assert_eq!(acc.w.len(), k, "prior acc w has wrong length");
            assert_eq!(acc.instance.alpha.len(), log_n, "prior α length");
            assert_eq!(acc.instance.beta.len(), beta_len, "prior β length");
            f_table.push(core::mem::take(&mut acc.f));
            w_table.push(core::mem::take(&mut acc.w));
            a_table.push(acc.instance.alpha.clone());
            b_table.push(acc.instance.beta.clone());
        }

        let omega: EF = challenger.sample_algebra_element();
        let tau: Vec<EF> = (0..log_l)
            .map(|_| challenger.sample_algebra_element())
            .collect();

        let mut all_mus: Vec<EF> = mu_fresh.clone();
        let mut all_etas: Vec<EF> = vec![EF::ZERO; l1];
        for acc in &prior_accumulators {
            all_mus.push(acc.instance.mu);
            all_etas.push(acc.instance.eta);
        }
        let tau_eq_table = Poly::<EF>::new_from_point(&tau, EF::ONE);
        let mut current_eq: Vec<EF> = tau_eq_table.as_slice().to_vec();
        let _sigma_1: EF = (0..l)
            .map(|i| current_eq[i] * (all_mus[i] + omega * all_etas[i]))
            .sum();

        let d1 = self.twin_round_poly_degree(log_n, log_m, shape.max_degree);
        let mut twin_proof = SumcheckProof::<EF>::new();
        let mut gamma = Vec::with_capacity(log_l);
        for _ in 0..log_l {
            let coeffs = self.compute_twin_round_coeffs(
                d1,
                &f_table,
                &w_table,
                &a_table,
                &b_table,
                &current_eq,
                omega,
                log_m,
            );
            let g = observe_and_sample::<F, EF, _>(&mut twin_proof, challenger, coeffs);
            gamma.push(g);
            f_table = fold_table(f_table, g);
            w_table = fold_table(w_table, g);
            a_table = fold_table(a_table, g);
            b_table = fold_table(b_table, g);
            current_eq = fold_eq(current_eq, g);
        }
        debug_assert_eq!(f_table.len(), 1);

        let f_merged = f_table.into_iter().next().unwrap();
        let w_merged = w_table.into_iter().next().unwrap();
        let zeta_0 = a_table.into_iter().next().unwrap();
        let beta_final = b_table.into_iter().next().unwrap();

        let f_poly = Poly::<EF>::new(f_merged.clone());
        let zeta_0_pt = Point::<EF>::new(zeta_0.clone());
        let nu_0: EF = f_poly.eval_ext::<F>(&zeta_0_pt);

        let beta_tau = &beta_final[..log_m];
        let beta_x = &beta_final[log_m..];
        let mut z_for_eta = Vec::with_capacity(beta_x.len() + w_merged.len());
        z_for_eta.extend_from_slice(beta_x);
        z_for_eta.extend_from_slice(&w_merged);
        let beta_tau_eq = Poly::<EF>::new_from_point(beta_tau, EF::ONE);
        let eta = self
            .pesat
            .evaluate_bundled(beta_tau_eq.as_slice(), &z_for_eta);

        let (rt_merged, td_merged) = acc_backend
            .commit_with_message(f_merged.clone(), &w_merged)
            .unwrap_or_else(|err| panic!("accumulator commitment failed: {err:?}"));

        acc_backend.observe_commitment(challenger, &rt_merged);
        challenger.observe_algebra_element(nu_0);
        challenger.observe_algebra_element(eta);

        let s = self.params.num_ood;
        let t = self.params.num_shift_queries;
        let mut zetas: Vec<Vec<EF>> = Vec::with_capacity(1 + s + t);
        let mut nus: Vec<EF> = Vec::with_capacity(1 + s + t);
        zetas.push(zeta_0.clone());
        nus.push(nu_0);

        let mut nu_ood = Vec::with_capacity(s);
        for _ in 0..s {
            let zeta_k: Vec<EF> = (0..log_n)
                .map(|_| challenger.sample_algebra_element())
                .collect();
            let zeta_k_pt = Point::<EF>::new(zeta_k.clone());
            let nu_k: EF = f_poly.eval_ext::<F>(&zeta_k_pt);
            challenger.observe_algebra_element(nu_k);
            zetas.push(zeta_k);
            nus.push(nu_k);
            nu_ood.push(nu_k);
        }

        let shift_indices = sample_indices::<F, _>(challenger, log_n, t);

        let mut fresh_shift_answers: Vec<Vec<F>> = (0..t).map(|_| Vec::with_capacity(l1)).collect();
        let mut fresh_opening_proofs = Vec::with_capacity(l1);
        for c in fresh_committed {
            let (cells, proof) = fresh_openings
                .open_batch_owned(c, &shift_indices)
                .unwrap_or_else(|err| panic!("external fresh batch opening failed: {err:?}"));
            assert_eq!(
                cells.len(),
                t,
                "external fresh batch opening returned wrong number of values"
            );
            for (k_idx, cell) in cells.into_iter().enumerate() {
                fresh_shift_answers[k_idx].push(cell);
            }
            fresh_opening_proofs.push(proof);
        }

        let mut acc_shift_answers: Vec<Vec<Vec<EF>>> = Vec::with_capacity(l2);
        let mut acc_merkle_proofs = Vec::with_capacity(l2);
        for acc in &prior_accumulators {
            let (cells, proof) = acc_backend
                .open_batch(&acc.td, &shift_indices)
                .unwrap_or_else(|err| panic!("accumulator batch opening failed: {err:?}"));
            assert_eq!(
                cells.len(),
                t,
                "accumulator batch opening returned wrong number of values"
            );
            acc_shift_answers.push(cells.into_iter().map(|cell| vec![cell]).collect());
            acc_merkle_proofs.push(proof);
        }

        let gamma_eq = Poly::<EF>::new_from_point(&gamma, EF::ONE);
        let gamma_eq_slice = gamma_eq.as_slice();
        for (k_idx, &x_k) in shift_indices.iter().enumerate() {
            let mut nu_sk = EF::ZERO;
            for i in 0..l1 {
                nu_sk += gamma_eq_slice[i] * EF::from(fresh_shift_answers[k_idx][i]);
            }
            for j in 0..l2 {
                nu_sk += gamma_eq_slice[l1 + j] * acc_shift_answers[j][k_idx][0];
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

        let eq_star = weighted_eq_poly::<F, EF>(&zetas[..r], &xi_eq, log_n);
        let sigma_2: EF = (0..r).map(|j| xi_eq[j] * nus[j]).sum();

        let mut batching_proof = SumcheckProof::<EF>::new();
        let mut alpha_challenges = Vec::with_capacity(log_n);
        let pack_w = F::Packing::WIDTH;

        if f_poly.num_evals() < pack_w * 2 || f_poly.num_evals() % pack_w != 0 {
            let mut f_remaining = f_poly.clone();
            let mut eq_star_remaining = eq_star;
            for _ in 0..log_n {
                let evals = degree2_round_evals(&f_remaining, &eq_star_remaining);
                let coeffs = degree2_coeffs_from_evals(evals);
                let alpha = observe_and_sample::<F, EF, _>(&mut batching_proof, challenger, coeffs);
                alpha_challenges.push(alpha);
                f_remaining.fix_prefix_var_mut(alpha);
                eq_star_remaining.fix_prefix_var_mut(alpha);
            }
            debug_assert_eq!(f_remaining.num_evals(), 1);
            let mu_final = f_remaining.as_slice()[0];
            challenger.observe_algebra_element(mu_final);
            let acc_instance = AccumulatorInstance {
                rt: rt_merged.clone(),
                alpha: alpha_challenges,
                mu: mu_final,
                beta: beta_final,
                eta,
            };
            let acc_witness = AccumulatorWitness {
                td: td_merged,
                f: f_merged,
                w: w_merged,
            };
            let proof = WarpProofExternalBatched {
                mu_fresh,
                twin_constraint_sumcheck: twin_proof,
                nu_0,
                eta,
                nu_ood,
                batching_sumcheck: batching_proof,
                mu_final,
                fresh_shift_answers,
                fresh_opening_proofs,
                acc_shift_answers,
                acc_merkle_proofs,
                _ph: PhantomData,
            };
            let _ = (sigma_2, _sigma_1);
            let _ = nus;
            return (
                Accumulator {
                    instance: acc_instance,
                    witness: acc_witness,
                },
                proof,
            );
        }

        let log_pack = pack_w.trailing_zeros() as usize;
        let n_packed_rounds = log_n.saturating_sub(log_pack);
        let mut f_packed: Vec<EF::ExtensionPacking> = pack_ef_slice::<F, EF>(f_poly.as_slice());
        let mut eq_packed: Vec<EF::ExtensionPacking> = pack_ef_slice::<F, EF>(eq_star.as_slice());

        for _ in 0..n_packed_rounds {
            let evals = degree2_round_evals_packed::<F, EF>(&f_packed, &eq_packed);
            let coeffs = degree2_coeffs_from_evals(evals);
            let alpha = observe_and_sample::<F, EF, _>(&mut batching_proof, challenger, coeffs);
            alpha_challenges.push(alpha);
            fold_packed_in_place::<F, EF>(&mut f_packed, alpha);
            fold_packed_in_place::<F, EF>(&mut eq_packed, alpha);
        }

        debug_assert_eq!(f_packed.len(), 1);
        let f_scalar: Vec<EF> =
            EF::ExtensionPacking::to_ext_iter(f_packed.iter().copied()).collect();
        let eq_scalar: Vec<EF> =
            EF::ExtensionPacking::to_ext_iter(eq_packed.iter().copied()).collect();
        let mut f_remaining = Poly::<EF>::new(f_scalar);
        let mut eq_star_remaining = Poly::<EF>::new(eq_scalar);

        for _ in 0..log_pack {
            let evals = degree2_round_evals(&f_remaining, &eq_star_remaining);
            let coeffs = degree2_coeffs_from_evals(evals);
            let alpha = observe_and_sample::<F, EF, _>(&mut batching_proof, challenger, coeffs);
            alpha_challenges.push(alpha);
            f_remaining.fix_prefix_var_mut(alpha);
            eq_star_remaining.fix_prefix_var_mut(alpha);
        }
        debug_assert_eq!(f_remaining.num_evals(), 1);
        let mu_final = f_remaining.as_slice()[0];
        challenger.observe_algebra_element(mu_final);

        let acc_instance = AccumulatorInstance {
            rt: rt_merged.clone(),
            alpha: alpha_challenges,
            mu: mu_final,
            beta: beta_final,
            eta,
        };
        let acc_witness = AccumulatorWitness {
            td: td_merged,
            f: f_merged,
            w: w_merged,
        };
        let proof = WarpProofExternalBatched {
            mu_fresh,
            twin_constraint_sumcheck: twin_proof,
            nu_0,
            eta,
            nu_ood,
            batching_sumcheck: batching_proof,
            mu_final,
            fresh_shift_answers,
            fresh_opening_proofs,
            acc_shift_answers,
            acc_merkle_proofs,
            _ph: PhantomData,
        };
        let _ = (sigma_2, _sigma_1);
        let _ = nus;

        (
            Accumulator {
                instance: acc_instance,
                witness: acc_witness,
            },
            proof,
        )
    }

    /// §6.3 round-polynomial degree: `1 + max{log n + 1, log M + d}`.
    fn twin_round_poly_degree(&self, log_n: usize, log_m: usize, d: usize) -> usize {
        1 + (log_n + 1).max(log_m + d)
    }

    /// Compute the §6.3 round polynomial in **coefficient form** via Lemma 6.4
    /// of the WARP paper (eprint 2025/753, lines 2090–2222). The polynomial is
    ///
    /// ```text
    ///     ĥ_j(X) = Σ_{i' ∈ [half)} eq_τ_{i'}(X) · ( û_{i'}(X)  +  ω · pb_{i'}(X) )
    /// ```
    ///
    /// where, with `c_lerp(X) = (1 − X) · c_lo + X · c_hi`,
    /// - `eq_τ_{i'}(X) = (1 − X) · eq_table[i'] + X · eq_table[i' + half]`
    /// - `û_{i'}(X)   = Σ_{b ∈ [n)} eq(A_{i'}(X), b) · F̂_{i', b}(X)`
    ///   (Claim 6.5: `m = log n`, `d = 1`, cost `O(n)` per `i'`).
    /// - `pb_{i'}(X)  = Σ_{c ∈ [M)} eq(B_τ_{i'}(X), c) · p̂_c(B_x_{i'}(X), W_{i'}(X))`
    ///   (Claim 6.5: `m = log M`, `d = PESAT degree`, cost `O(M · d)` per `i'` —
    ///   delegated to [`BundledPesat::bundled_round_poly`]).
    ///
    /// # Cost
    ///
    /// `O(half · (n + M · d))` field ops per round (paper Lemma 6.4
    /// asymptote), versus `O(half · (D₁+1) · (n + M))` for the previous
    /// "evaluate at `D₁+1` integer α points + Lagrange-interpolate" path.
    /// The win factor is `~D₁ / 1 = log n + log M + d`.
    ///
    /// The codeword side uses Claim 6.5 directly. High-order variables are
    /// folded over `EF::ExtensionPacking`, then the final SIMD-lane variables
    /// are folded scalar after unpacking. This preserves the MSB-first
    /// hypercube convention used by `Poly::new_from_point` and avoids the old
    /// `(log n + 2)` evaluations plus interpolation.
    #[allow(clippy::too_many_arguments)]
    fn compute_twin_round_coeffs(
        &self,
        d1: usize,
        f_table: &[Vec<EF>],
        w_table: &[Vec<EF>],
        a_table: &[Vec<EF>],
        b_table: &[Vec<EF>],
        eq_table: &[EF],
        omega: EF,
        _log_m: usize,
    ) -> Vec<EF> {
        let half = f_table.len() / 2;
        let n = f_table[0].len();

        // Reduce the per-i local round-poly contributions into a single
        // round polynomial via element-wise sum. The fold accumulator owns
        // reusable Claim 6.5 work tables, so the hot path accumulates the
        // codeword and constraint contributions directly into `acc` instead
        // of allocating one round-polynomial Vec per paired row.
        //
        // Mirrors Plonky3 WHIR's `par_fold_reduce` pattern
        // (`whir/src/sumcheck/strategy.rs:55-64`): parallel above the
        // scalar-overhead threshold, sequential below.
        //
        // We parallelise across `i` (the bundling axis) instead of within a
        // single `i`'s inner work — each `i` is a coarse-grained task with
        // O(n + M·d) work, comfortably above the rayon split-overhead floor.
        // PAR_THRESHOLD chosen analogously to whir (`PAR_THRESHOLD = 1 << 14`):
        // each `i` does ~`(log n + 2) · (n / W) + M · d` field ops; we go
        // parallel once `half · per_i_work` clears ~2^14 effective ops, which
        // for our scales (`n ≥ 1024`) means `half ≥ 2` is already worth it.
        if half >= 2 && n >= 1024 {
            (0..half)
                .into_par_iter()
                .par_fold_reduce(
                    || TwinRoundScratch::<F, EF>::new(d1),
                    |mut scratch, i| {
                        add_twin_round_contribution::<F, EF, Pesat>(
                            self.pesat,
                            d1,
                            i,
                            half,
                            f_table,
                            w_table,
                            a_table,
                            b_table,
                            eq_table,
                            omega,
                            &mut scratch,
                        );
                        scratch
                    },
                    |mut a, b| {
                        for (lhs, rhs) in a.acc.iter_mut().zip(b.acc.into_iter()) {
                            *lhs += rhs;
                        }
                        a
                    },
                )
                .acc
        } else {
            let mut scratch = TwinRoundScratch::<F, EF>::new(d1);
            for i in 0..half {
                add_twin_round_contribution::<F, EF, Pesat>(
                    self.pesat,
                    d1,
                    i,
                    half,
                    f_table,
                    w_table,
                    a_table,
                    b_table,
                    eq_table,
                    omega,
                    &mut scratch,
                );
            }
            scratch.acc
        }
    }
}

struct TwinRoundScratch<F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    codeword: Claim65Scratch<F, EF>,
    constraint: Claim65Scratch<F, EF>,
    codeword_coeffs: Vec<EF>,
    constraint_coeffs: Vec<EF>,
    acc: Vec<EF>,
}

impl<F, EF> TwinRoundScratch<F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    fn new(degree: usize) -> Self {
        Self {
            codeword: Claim65Scratch::new(),
            constraint: Claim65Scratch::new(),
            codeword_coeffs: Vec::new(),
            constraint_coeffs: Vec::new(),
            acc: vec![EF::ZERO; degree + 1],
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn add_twin_round_contribution<F, EF, Pesat>(
    pesat: &Pesat,
    d1: usize,
    i: usize,
    half: usize,
    f_table: &[Vec<EF>],
    w_table: &[Vec<EF>],
    a_table: &[Vec<EF>],
    b_table: &[Vec<EF>],
    eq_table: &[EF],
    omega: EF,
    scratch: &mut TwinRoundScratch<F, EF>,
) where
    F: Field,
    EF: ExtensionField<F>,
    Pesat: BundledPesat<F, EF>,
{
    codeword_claim_6_5_coeffs_into::<F, EF>(
        &f_table[i],
        &f_table[i + half],
        &a_table[i],
        &a_table[i + half],
        &mut scratch.codeword_coeffs,
        &mut scratch.codeword,
    );

    pesat.bundled_round_poly_into(
        &b_table[i],
        &b_table[i + half],
        &w_table[i],
        &w_table[i + half],
        &mut scratch.constraint_coeffs,
        &mut scratch.constraint,
    );

    // Build g_i(X) = û_i(X) + ω · pb_i(X), then multiply by linear
    // eq_τ_i(X) = eq_lo + (eq_hi − eq_lo) · X. This writes directly into
    // the fold accumulator. The order and coefficients are identical to the
    // previous local-vector path; only allocation strategy changes.
    let eq_lo = eq_table[i];
    let eq_diff = eq_table[i + half] - eq_lo;
    let g_len = scratch
        .codeword_coeffs
        .len()
        .max(scratch.constraint_coeffs.len());
    debug_assert!(g_len <= d1);
    for k in 0..=g_len {
        let mut g_k = EF::ZERO;
        if k < scratch.codeword_coeffs.len() {
            g_k += scratch.codeword_coeffs[k];
        }
        if k < scratch.constraint_coeffs.len() {
            g_k += omega * scratch.constraint_coeffs[k];
        }

        let mut g_k_minus_1 = EF::ZERO;
        if k > 0 {
            let km1 = k - 1;
            if km1 < scratch.codeword_coeffs.len() {
                g_k_minus_1 += scratch.codeword_coeffs[km1];
            }
            if km1 < scratch.constraint_coeffs.len() {
                g_k_minus_1 += omega * scratch.constraint_coeffs[km1];
            }
        }

        scratch.acc[k] += g_k * eq_lo + g_k_minus_1 * eq_diff;
    }
}

/// Codeword-side Claim 6.5 kernel for WARP §6.3.
///
/// Computes
///
/// ```text
///     Σ_b eq(A_lo + X(A_hi - A_lo), b) · (F_lo[b] + X(F_hi[b] - F_lo[b]))
/// ```
///
/// in coefficient form. This is the `m = log n, d = 1` instance of the
/// WARP Claim 6.5 composition algorithm. The packed path folds the prefix
/// variables over SIMD lanes, then finishes the lane-local suffix variables
/// scalar.
#[cfg(test)]
fn codeword_claim_6_5_coeffs<F, EF>(f_lo: &[EF], f_hi: &[EF], a_lo: &[EF], a_hi: &[EF]) -> Vec<EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let mut out = Vec::new();
    let mut scratch = Claim65Scratch::<F, EF>::new();
    codeword_claim_6_5_coeffs_into::<F, EF>(f_lo, f_hi, a_lo, a_hi, &mut out, &mut scratch);
    out
}

fn codeword_claim_6_5_coeffs_into<F, EF>(
    f_lo: &[EF],
    f_hi: &[EF],
    a_lo: &[EF],
    a_hi: &[EF],
    out: &mut Vec<EF>,
    scratch: &mut Claim65Scratch<F, EF>,
) where
    F: Field,
    EF: ExtensionField<F>,
{
    debug_assert_eq!(f_lo.len(), f_hi.len());
    debug_assert!(f_lo.len().is_power_of_two());
    debug_assert_eq!(a_lo.len(), f_lo.len().ilog2() as usize);
    debug_assert_eq!(a_lo.len(), a_hi.len());

    // Fresh WARP inputs have α = 0^{log n}. When both endpoints are fresh
    // descendants, eq(A(X), b) is the fixed selector eq(0, b), so the full
    // Claim 6.5 codeword composition collapses to the first codeword entry:
    // f_lo[0] + X · (f_hi[0] - f_lo[0]). This preserves the exact same
    // polynomial while avoiding an O(n) scan for fresh/fresh pairs.
    if a_lo.iter().all(|&x| x == EF::ZERO) && a_hi.iter().all(|&x| x == EF::ZERO) {
        out.clear();
        out.push(f_lo[0]);
        out.push(f_hi[0] - f_lo[0]);
        return;
    }

    let pack_w = F::Packing::WIDTH;
    if f_lo.len() >= pack_w * 2 && f_lo.len().is_multiple_of(pack_w) {
        codeword_claim_6_5_coeffs_packed_prefix_into::<F, EF>(f_lo, f_hi, a_lo, a_hi, out, scratch);
    } else {
        codeword_claim_6_5_coeffs_scalar_into::<F, EF>(f_lo, f_hi, a_lo, a_hi, out, scratch);
    }
}

#[cfg(test)]
fn codeword_claim_6_5_coeffs_scalar<F, EF>(
    f_lo: &[EF],
    f_hi: &[EF],
    a_lo: &[EF],
    a_hi: &[EF],
) -> Vec<EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let mut out = Vec::new();
    let mut scratch = Claim65Scratch::<F, EF>::new();
    codeword_claim_6_5_coeffs_scalar_into::<F, EF>(f_lo, f_hi, a_lo, a_hi, &mut out, &mut scratch);
    out
}

fn codeword_claim_6_5_coeffs_scalar_into<F, EF>(
    f_lo: &[EF],
    f_hi: &[EF],
    a_lo: &[EF],
    a_hi: &[EF],
    out: &mut Vec<EF>,
    scratch: &mut Claim65Scratch<F, EF>,
) where
    F: Field,
    EF: ExtensionField<F>,
{
    let mut width = 2;
    let mut len = f_lo.len();
    scratch.scalar_current.clear();
    for (&lo, &hi) in f_lo.iter().zip(f_hi) {
        scratch.scalar_current.push(lo);
        scratch.scalar_current.push(hi - lo);
    }

    for (&c0, &c_at_one) in a_lo.iter().zip(a_hi) {
        let c1 = c_at_one - c0;
        let next_len = len / 2;
        let next_width = width + 1;
        scratch.scalar_next.clear();
        scratch.scalar_next.resize(next_len * next_width, EF::ZERO);
        fold_claim_6_5_scalar_round(
            &scratch.scalar_current,
            width,
            len,
            c0,
            c1,
            &mut scratch.scalar_next,
        );
        core::mem::swap(&mut scratch.scalar_current, &mut scratch.scalar_next);
        width = next_width;
        len = next_len;
    }

    debug_assert_eq!(len, 1);
    out.clear();
    out.extend_from_slice(&scratch.scalar_current);
}

fn codeword_claim_6_5_coeffs_packed_prefix_into<F, EF>(
    f_lo: &[EF],
    f_hi: &[EF],
    a_lo: &[EF],
    a_hi: &[EF],
    out: &mut Vec<EF>,
    scratch: &mut Claim65Scratch<F, EF>,
) where
    F: Field,
    EF: ExtensionField<F>,
{
    let pack_w = F::Packing::WIDTH;
    let log_pack = pack_w.trailing_zeros() as usize;
    let log_n = f_lo.len().trailing_zeros() as usize;
    let packed_rounds = log_n.saturating_sub(log_pack);

    let mut width = 2;
    let mut len = f_lo.len() / pack_w;
    scratch.packed_current.clear();
    scratch.lane_buf.clear();
    scratch.lane_buf.resize(pack_w, EF::ZERO);
    for (lo_chunk, hi_chunk) in f_lo.chunks_exact(pack_w).zip(f_hi.chunks_exact(pack_w)) {
        scratch
            .packed_current
            .push(<EF::ExtensionPacking as PackedFieldExtension<F, EF>>::from_ext_slice(lo_chunk));
        for lane in 0..pack_w {
            scratch.lane_buf[lane] = hi_chunk[lane] - lo_chunk[lane];
        }
        scratch.packed_current.push(
            <EF::ExtensionPacking as PackedFieldExtension<F, EF>>::from_ext_slice(
                &scratch.lane_buf,
            ),
        );
    }

    for (&c0, &c_at_one) in a_lo.iter().zip(a_hi).take(packed_rounds) {
        let c1 = c_at_one - c0;
        let c0_packed = packed_ext_scalar_with_scratch::<F, EF>(c0, &mut scratch.broadcast_buf);
        let c1_packed = packed_ext_scalar_with_scratch::<F, EF>(c1, &mut scratch.broadcast_buf);
        let next_len = len / 2;
        let next_width = width + 1;
        scratch.packed_next.clear();
        scratch
            .packed_next
            .resize(next_len * next_width, EF::ExtensionPacking::ZERO);
        fold_claim_6_5_packed_round::<F, EF>(
            &scratch.packed_current,
            width,
            len,
            c0_packed,
            c1_packed,
            &mut scratch.packed_next,
        );
        core::mem::swap(&mut scratch.packed_current, &mut scratch.packed_next);
        width = next_width;
        len = next_len;
    }
    debug_assert_eq!(len, 1);

    unpack_packed_coeffs_to_scalar::<F, EF>(
        &scratch.packed_current,
        width,
        &mut scratch.scalar_current,
    );
    let mut scalar_width = width;
    let mut scalar_len = pack_w;
    for (&c0, &c_at_one) in a_lo.iter().zip(a_hi).skip(packed_rounds) {
        let c1 = c_at_one - c0;
        let next_len = scalar_len / 2;
        let next_width = scalar_width + 1;
        scratch.scalar_next.clear();
        scratch.scalar_next.resize(next_len * next_width, EF::ZERO);
        fold_claim_6_5_scalar_round(
            &scratch.scalar_current,
            scalar_width,
            scalar_len,
            c0,
            c1,
            &mut scratch.scalar_next,
        );
        core::mem::swap(&mut scratch.scalar_current, &mut scratch.scalar_next);
        scalar_width = next_width;
        scalar_len = next_len;
    }

    debug_assert_eq!(scalar_len, 1);
    debug_assert_eq!(scalar_width, log_n + 2);
    out.clear();
    out.extend_from_slice(&scratch.scalar_current);
}

/// Compute `[h(0), h(1), h(2)]` of the §8.2 round polynomial
/// `h(X) = Σ_b f̂(X, b) · eq*(X, b)` using SIMD-packed extension
/// arithmetic. The two input slices are packed `EF` polynomials over
/// the suffix hypercube; the routine reduces the round computation to
/// three packed dot-products over `n / W` packed elements, then folds
/// SIMD lanes with `to_ext_iter`.
///
/// Mirrors the WHIR `new_classic_packed` pattern but specialised to the
/// degree-2 round-poly case.
#[inline]
pub(crate) fn degree2_round_evals_packed<F, EF>(
    f: &[EF::ExtensionPacking],
    eq_star: &[EF::ExtensionPacking],
) -> [EF; 3]
where
    F: Field,
    EF: ExtensionField<F>,
{
    debug_assert_eq!(f.len(), eq_star.len());
    debug_assert!(f.len() >= 2);
    let half = f.len() / 2;
    let f_lo = &f[..half];
    let f_hi = &f[half..];
    let eq_lo = &eq_star[..half];
    let eq_hi = &eq_star[half..];

    let mut h_0 = EF::ExtensionPacking::ZERO;
    let mut h_1 = EF::ExtensionPacking::ZERO;
    let mut h_2 = EF::ExtensionPacking::ZERO;
    for i in 0..half {
        h_0 += f_lo[i] * eq_lo[i];
        h_1 += f_hi[i] * eq_hi[i];
        let f_two = f_hi[i].double() - f_lo[i];
        let eq_two = eq_hi[i].double() - eq_lo[i];
        h_2 += f_two * eq_two;
    }

    [
        EF::ExtensionPacking::to_ext_iter([h_0]).sum(),
        EF::ExtensionPacking::to_ext_iter([h_1]).sum(),
        EF::ExtensionPacking::to_ext_iter([h_2]).sum(),
    ]
}

#[inline]
fn degree2_coeffs_from_evals<EF: Field>(evals: [EF; 3]) -> Vec<EF> {
    let [h_0, h_1, h_2] = evals;
    let inv_two = (EF::ONE + EF::ONE).inverse();
    let c_0 = h_0;
    let c_2 = (h_2 - h_1 - h_1 + h_0) * inv_two;
    let c_1 = h_1 - h_0 - c_2;
    alloc::vec![c_0, c_1, c_2]
}

/// In-place fold of a packed-extension table along its first variable
/// (MSB-first convention): `out[i] = lo[i] + α · (hi[i] - lo[i])`,
/// where `α` is broadcast across SIMD lanes.
///
/// After folding, the table's length halves; the trailing half is
/// truncated.
#[inline]
pub(crate) fn fold_packed_in_place<F, EF>(table: &mut Vec<EF::ExtensionPacking>, alpha: EF)
where
    F: Field,
    EF: ExtensionField<F>,
{
    let half = table.len() / 2;
    // Broadcast scalar α into every SIMD lane without allocating a temporary
    // lane vector. `PackedFieldExtension` is an algebra over `EF`.
    let alpha_packed: EF::ExtensionPacking = alpha.into();
    for i in 0..half {
        let lo = table[i];
        let hi = table[i + half];
        table[i] = lo + alpha_packed * (hi - lo);
    }
    table.truncate(half);
}

/// Pack a flat `Vec<EF>` of `n` evaluations into `n / W` SIMD-packed
/// extension-field elements, where `W = F::Packing::WIDTH`.
///
/// Each packed element holds `W` consecutive `EF` values across SIMD lanes —
/// this is exactly the layout [`Poly::eval_packed`] expects, so the result
/// can be wrapped in [`Poly::new`] and evaluated as a multilinear polynomial
/// over `log_2(n) − log_2(W)` packed-axis variables (with `log_2(W)` more
/// implicit variables absorbed into the lanes).
///
/// # Panics
///
/// `slice.len()` must be a positive multiple of `F::Packing::WIDTH`.
#[inline]
pub(crate) fn pack_ef_slice<F, EF>(slice: &[EF]) -> Vec<EF::ExtensionPacking>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let w = F::Packing::WIDTH;
    debug_assert!(
        slice.len() % w == 0,
        "slice length must be multiple of WIDTH"
    );
    slice
        .chunks_exact(w)
        .map(<EF::ExtensionPacking as PackedFieldExtension<F, EF>>::from_ext_slice)
        .collect()
}

/// Type alias for the `ExtensionMmcs` prover-data over an `EF` codeword
/// matrix of width 1. This expands to
/// `MT::ProverData<FlatMatrixView<F, EF, RowMajorMatrix<EF>>>` —
/// i.e., the base Mmcs's prover data with a single `FlatMatrixView` wrapper
/// reinterpreting the EF matrix as a wider F matrix.
pub type ExtProverData<F, EF, MT> =
    <ExtensionMmcs<F, EF, MT> as Mmcs<EF>>::ProverData<RowMajorMatrix<EF>>;

/// Linear interpolation `(1 − α) · lo + α · hi`.
#[inline]
fn lerp<EF: Field>(lo: EF, hi: EF, alpha: EF) -> EF {
    lo + alpha * (hi - lo)
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;

    use super::*;
    use crate::relation::lagrange_interpolate_int_points;

    type TestF = BabyBear;
    type TestEF = BinomialExtensionField<TestF, 4>;

    fn codeword_claim_reference(
        f_lo: &[TestEF],
        f_hi: &[TestEF],
        a_lo: &[TestEF],
        a_hi: &[TestEF],
    ) -> Vec<TestEF> {
        let log_n = f_lo.len().ilog2() as usize;
        let degree = log_n + 1;
        let evals = (0..=degree)
            .map(|alpha_idx| {
                let alpha = TestEF::from_u64(alpha_idx as u64);
                let a_alpha = a_lo
                    .iter()
                    .zip(a_hi)
                    .map(|(&lo, &hi)| lo + alpha * (hi - lo))
                    .collect::<Vec<_>>();
                let eq_a = Poly::<TestEF>::new_from_point(&a_alpha, TestEF::ONE);
                f_lo.iter()
                    .zip(f_hi)
                    .zip(eq_a.iter())
                    .map(|((&lo, &hi), &eq)| (lo + alpha * (hi - lo)) * eq)
                    .sum()
            })
            .collect::<Vec<_>>();
        lagrange_interpolate_int_points(&evals)
    }

    fn deterministic_vec(len: usize, offset: u64) -> Vec<TestEF> {
        (0..len)
            .map(|i| TestEF::from_u64(offset + (i as u64) * 17))
            .collect()
    }

    #[test]
    fn weighted_eq_poly_matches_table_sum_reference() {
        let log_n = 5;
        let points = (0..4)
            .map(|j| deterministic_vec(log_n, 11 + 37 * j as u64))
            .collect::<Vec<_>>();
        let scalars = deterministic_vec(points.len(), 101);

        let actual = weighted_eq_poly::<TestF, TestEF>(&points, &scalars, log_n);

        let mut expected = Poly::<TestEF>::zero(log_n);
        for (point, &scale) in points.iter().zip(&scalars) {
            let scaled = Poly::<TestEF>::new_from_point(point, scale);
            for (acc, &value) in expected.as_mut_slice().iter_mut().zip(scaled.iter()) {
                *acc += value;
            }
        }

        assert_eq!(actual, expected);
    }

    #[test]
    fn codeword_claim_6_5_scalar_matches_interpolation_reference() {
        let n = 4;
        let f_lo = deterministic_vec(n, 3);
        let f_hi = deterministic_vec(n, 29);
        let a_lo = deterministic_vec(n.ilog2() as usize, 101);
        let a_hi = deterministic_vec(n.ilog2() as usize, 211);
        let expected = codeword_claim_reference(&f_lo, &f_hi, &a_lo, &a_hi);
        let actual = codeword_claim_6_5_coeffs_scalar::<TestF, TestEF>(&f_lo, &f_hi, &a_lo, &a_hi);
        assert_eq!(actual, expected);
    }

    #[test]
    fn codeword_claim_6_5_packed_prefix_matches_interpolation_reference() {
        let n = <TestF as Field>::Packing::WIDTH * 4;
        let f_lo = deterministic_vec(n, 5);
        let f_hi = deterministic_vec(n, 41);
        let a_lo = deterministic_vec(n.ilog2() as usize, 109);
        let a_hi = deterministic_vec(n.ilog2() as usize, 307);
        let expected = codeword_claim_reference(&f_lo, &f_hi, &a_lo, &a_hi);
        let actual = codeword_claim_6_5_coeffs::<TestF, TestEF>(&f_lo, &f_hi, &a_lo, &a_hi);
        assert_eq!(actual, expected);
    }

    #[test]
    fn codeword_claim_6_5_zero_alpha_selects_first_codeword_entry() {
        let n = <TestF as Field>::Packing::WIDTH * 4;
        let f_lo = deterministic_vec(n, 13);
        let f_hi = deterministic_vec(n, 97);
        let a_lo = vec![TestEF::ZERO; n.ilog2() as usize];
        let a_hi = vec![TestEF::ZERO; n.ilog2() as usize];
        let actual = codeword_claim_6_5_coeffs::<TestF, TestEF>(&f_lo, &f_hi, &a_lo, &a_hi);
        assert_eq!(actual, vec![f_lo[0], f_hi[0] - f_lo[0]]);
    }
}

/// Componentwise linear interpolation into the left-hand vector.
#[inline]
fn lerp_vec_in_place<EF>(lo: &mut [EF], hi: &[EF], alpha: EF)
where
    EF: Field + Send + Sync,
{
    debug_assert_eq!(lo.len(), hi.len());
    if lo.len() > PAR_THRESHOLD {
        lo.par_iter_mut()
            .zip(hi.par_iter())
            .for_each(|(l, &r)| *l = lerp(*l, r, alpha));
    } else {
        for (l, &r) in lo.iter_mut().zip(hi) {
            *l = lerp(*l, r, alpha);
        }
    }
}

/// Fold a `Vec<Vec<EF>>` table along its first axis at challenge `γ`,
/// returning a half-length table whose `i`-th entry is
/// `(1 − γ) · table[i] + γ · table[i + half]` componentwise.
fn fold_table<EF>(mut table: Vec<Vec<EF>>, gamma: EF) -> Vec<Vec<EF>>
where
    EF: Field + Send + Sync,
{
    let half = table.len() / 2;
    {
        let (lo_rows, hi_rows) = table.split_at_mut(half);
        for (lo, hi) in lo_rows.iter_mut().zip(hi_rows.iter()) {
            lerp_vec_in_place(lo, hi, gamma);
        }
    }
    table.truncate(half);
    table
}

/// Same as [`fold_table`] but for a flat `Vec<EF>` (the `eq` table).
fn fold_eq<EF: Field>(mut eq: Vec<EF>, gamma: EF) -> Vec<EF> {
    let half = eq.len() / 2;
    for i in 0..half {
        eq[i] = lerp(eq[i], eq[i + half], gamma);
    }
    eq.truncate(half);
    eq
}

/// Build the WARP §8.2 batching polynomial
/// `eq*(X) = Σ_j scalars[j] · eq(points[j], X)` over `{0,1}^{log_n}`.
///
/// This is algebraically identical to summing `Poly::new_from_point` tables,
/// but uses Plonky3's batched equality kernel, the same primitive WHIR uses
/// for equality constraints. The row-major matrix has variables as rows and
/// batching points as columns, matching `eval_eq_batch`'s convention.
pub(crate) fn weighted_eq_poly<F, EF>(points: &[Vec<EF>], scalars: &[EF], log_n: usize) -> Poly<EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    assert_eq!(points.len(), scalars.len());
    assert!(!points.is_empty());
    assert!(points.iter().all(|point| point.len() == log_n));

    let width = points.len();
    let mut point_matrix = Vec::with_capacity(log_n * width);
    for var_idx in 0..log_n {
        point_matrix.extend(points.iter().map(|point| point[var_idx]));
    }

    let mut values = vec![EF::ZERO; 1 << log_n];
    eval_eq_batch::<F, EF, false>(
        RowMajorMatrixView::new(&point_matrix, width),
        &mut values,
        scalars,
    );
    Poly::new(values)
}

/// Build a Boolean point `binary(x) ∈ {0,1}^{log n}` for a shift query.
///
/// The bit ordering matches the Plonky3 multilinear-extension convention:
/// `point[0]` corresponds to the most-significant bit of the codeword
/// index, so `f̂(boolean_point(x)) = f[x]` for a length-`n` codeword.
pub(crate) fn boolean_point<EF: Field>(x: usize, log_n: usize) -> Vec<EF> {
    (0..log_n)
        .map(|i| {
            // point[i] is the (log_n − 1 − i)-th bit of x, i.e., MSB-first.
            if (x >> (log_n - 1 - i)) & 1 == 1 {
                EF::ONE
            } else {
                EF::ZERO
            }
        })
        .collect()
}

/// Compute the three evaluations `[h(0), h(1), h(2)]` of the §8.2
/// round polynomial `h(X) = Σ_b f̂(X, b) · eq*(X, b)` over the suffix
/// hypercube `b ∈ {0,1}^{n−1}`.
///
/// This is the same quadratic-sumcheck round computation used by WHIR's
/// `sumcheck_coefficients_prefix` helper.
pub(crate) fn degree2_round_evals<EF: Field>(f: &Poly<EF>, eq_star: &Poly<EF>) -> [EF; 3] {
    debug_assert_eq!(f.num_evals(), eq_star.num_evals());
    debug_assert!(f.num_evals() >= 2);
    let half = f.num_evals() / 2;
    let f_lo = &f.as_slice()[..half];
    let f_hi = &f.as_slice()[half..];
    let eq_lo = &eq_star.as_slice()[..half];
    let eq_hi = &eq_star.as_slice()[half..];

    let h_0: EF = f_lo.iter().zip(eq_lo).map(|(&f, &e)| f * e).sum();
    let h_1: EF = f_hi.iter().zip(eq_hi).map(|(&f, &e)| f * e).sum();
    // h(2) = Σ ((1-2)·f_lo + 2·f_hi) · ((1-2)·eq_lo + 2·eq_hi) = Σ (2·f_hi − f_lo)(2·eq_hi − eq_lo).
    let h_2: EF = f_lo
        .iter()
        .zip(f_hi)
        .zip(eq_lo.iter().zip(eq_hi))
        .map(|((&fl, &fh), (&el, &eh))| (fh.double() - fl) * (eh.double() - el))
        .sum();
    [h_0, h_1, h_2]
}
