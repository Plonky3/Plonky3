//! STIR prover implementation (Construction 5.2).
//!
//! Codewords are stored in natural order: `codeword[j] = f(shift * g^j)` where
//! `g = two_adic_generator(log_domain_size)` and `shift` is the domain's coset shift.
//! Before committing, the codeword is arranged as a `(new_height × arity)` matrix where
//! row `j` contains the fiber, allowing a single MMCS opening to reveal the entire fiber.

use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{
    BasedVectorSpace, ExtensionField, Field, TwoAdicField, batch_multiplicative_inverse,
};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_util::log2_ceil_usize;
use tracing::instrument;

use crate::config::StirConfig;
use crate::proof::{StirProof, StirQueryOpenings, StirRoundProof};
use crate::utils::{
    compute_shake_polynomial, eval_poly_parallel, fold_codeword, fold_domain_params,
    interpolate_poly, next_domain_shift, sample_ood_points, vanishing_poly_from_roots,
};

/// Prove that a polynomial (given in coefficient form over `EF`) has low degree,
/// using the STIR proximity testing protocol.
///
/// The initial codeword commitment is observed in the challenger internally; callers must
/// NOT pre-commit the initial codeword.
///
/// Returns `(proof, first_round_query_indices)`. The second component is the deduplicated
/// fold-domain indices the prover queried in the first round (or in the final round, when
/// there are no intermediate rounds). It is the prover-side hint the PCS layer uses to bind
/// input commitments at the matching positions; it is NOT part of the verifier-checked proof
/// (the verifier re-derives these indices from the Fiat-Shamir transcript).
#[instrument(skip_all)]
pub fn prove_stir<F, EF, Dft, M, Challenger>(
    config: &StirConfig<F, EF, M, Challenger>,
    poly_coeffs: Vec<EF>,
    dft: &Dft,
    challenger: &mut Challenger,
) -> (StirProof<EF, M, Challenger::Witness>, Vec<usize>)
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField + BasedVectorSpace<F>,
    Dft: TwoAdicSubgroupDft<F>,
    M: Mmcs<EF>,
    Challenger: FieldChallenger<F>
        + CanObserve<M::Commitment>
        + GrindingChallenger<Witness = F>
        + CanSampleUniformBits<F>,
{
    let initial_shift = F::GENERATOR;
    let log_initial_domain = config.log_starting_domain_size();
    let initial_domain_size = 1 << log_initial_domain;
    let mut coeffs = poly_coeffs;
    coeffs.resize(initial_domain_size, EF::ZERO);
    let initial_codeword = codeword_from_coeffs(dft, coeffs, initial_shift, log_initial_domain);

    prove_stir_from_codeword(config, initial_codeword, dft, challenger)
}

/// Prove low degree from an initial codeword that the caller has already bound.
///
/// Identical to [`prove_stir_from_codeword`] except that the initial oracle is never committed:
/// no Merkle tree is built over it, its commitment is absent from the proof, and the queries
/// that read it ship no rows. The verifier side is [`verify_stir_with_external_initial`], whose
/// caller supplies the queried fibers.
///
/// # Soundness requirement
///
/// The caller MUST guarantee that the initial codeword is uniquely determined by data already
/// absorbed into `challenger` before this call — for the PCS layer, the input commitments, the
/// claimed opening values, and the batching challenge derived from them. STIR draws the round-0
/// folding challenge after this point, so a prover still free to choose the initial codeword
/// afterwards would break the proximity argument. A commitment adds nothing once the codeword
/// is already pinned, which is exactly why it can be dropped.
///
/// [`verify_stir_with_external_initial`]: crate::verifier::verify_stir_with_external_initial
#[instrument(skip_all)]
pub fn prove_stir_from_external_codeword<F, EF, Dft, M, Challenger>(
    config: &StirConfig<F, EF, M, Challenger>,
    initial_codeword: Vec<EF>,
    dft: &Dft,
    challenger: &mut Challenger,
) -> (StirProof<EF, M, Challenger::Witness>, Vec<usize>)
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField + BasedVectorSpace<F>,
    Dft: TwoAdicSubgroupDft<F>,
    M: Mmcs<EF>,
    Challenger: FieldChallenger<F>
        + CanObserve<M::Commitment>
        + GrindingChallenger<Witness = F>
        + CanSampleUniformBits<F>,
{
    prove_stir_inner(config, initial_codeword, dft, challenger, false)
}

/// Prove low degree from an initial natural-order codeword on STIR's starting domain.
///
/// This avoids an inverse DFT followed by a forward DFT when a caller already has the
/// codeword. The codeword must contain exactly `2^config.log_starting_domain_size()` values
/// on `F::GENERATOR * H` in natural order.
#[instrument(skip_all)]
pub fn prove_stir_from_codeword<F, EF, Dft, M, Challenger>(
    config: &StirConfig<F, EF, M, Challenger>,
    initial_codeword: Vec<EF>,
    dft: &Dft,
    challenger: &mut Challenger,
) -> (StirProof<EF, M, Challenger::Witness>, Vec<usize>)
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField + BasedVectorSpace<F>,
    Dft: TwoAdicSubgroupDft<F>,
    M: Mmcs<EF>,
    Challenger: FieldChallenger<F>
        + CanObserve<M::Commitment>
        + GrindingChallenger<Witness = F>
        + CanSampleUniformBits<F>,
{
    prove_stir_inner(config, initial_codeword, dft, challenger, true)
}

/// Output of proving one intermediate STIR round.
struct RoundOutput<F, EF: Field, M: Mmcs<EF>, Witness> {
    proof: StirRoundProof<EF, M, Witness>,
    next_codeword: Vec<EF>,
    next_commit_data: M::ProverData<RowMajorMatrix<EF>>,
    next_shift: F,
    next_log_domain: usize,
    /// Deduplicated, ascending fold-domain query indices this round drew. Only consumed by
    /// the caller when this is round 0.
    seen_query_indices: Vec<usize>,
}

/// One instance's in-flight intermediate round, advanced in the order the transcript demands.
///
/// The round alternates between blocks of squeezed challenges and blocks of absorbed prover
/// messages, and every grind sits immediately before a challenge block with no prover message
/// in between — the property that lets a grind bind the challenge that follows it. Holding the
/// round open across those boundaries lets a caller drive several instances through the same
/// boundaries, so one grind can cover every instance's challenges at that site.
///
/// The phases, in order: \[grind\] `fold_and_commit` -> absorb commitment -> squeeze OOD points
/// -> `ood_answers` -> absorb answers -> \[grind\] -> squeeze `r_comb` and query indices ->
/// `record_queries` -> `ans_shake` -> absorb Ans/shake -> squeeze rho -> `finish`.
struct RoundProver<'a, F, EF: Field, Dft, M: Mmcs<EF>, Challenger> {
    config: &'a StirConfig<F, EF, M, Challenger>,
    round: usize,
    dft: &'a Dft,

    log_arity: usize,
    fold_log_domain: usize,
    fold_shift: F,
    next_log_domain: usize,
    next_shift: F,

    folded_codeword: Vec<EF>,
    fold_coeffs: Vec<EF>,
    next_commit_codeword: Vec<EF>,
    new_data: Option<M::ProverData<RowMajorMatrix<EF>>>,

    ood_points: Vec<EF>,
    ood_answers: Vec<EF>,

    query_indices: Vec<usize>,
    query_points: Vec<EF>,
    query_answers: Vec<EF>,
    seen_query_indices: Vec<usize>,

    ans_poly: Vec<EF>,
    shake_poly: Vec<EF>,
    all_points: Vec<EF>,
}

impl<'a, F, EF, Dft, M, Challenger> RoundProver<'a, F, EF, Dft, M, Challenger>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField + BasedVectorSpace<F>,
    Dft: TwoAdicSubgroupDft<F>,
    M: Mmcs<EF>,
{
    /// Fix the round's domain geometry. Touches no transcript state.
    fn new(
        config: &'a StirConfig<F, EF, M, Challenger>,
        round: usize,
        dft: &'a Dft,
        current_shift: F,
        current_log_domain: usize,
    ) -> Self {
        let log_arity = config.round_configs[round].log_folding_factor;
        let (fold_log_domain, fold_shift) =
            fold_domain_params(current_shift, current_log_domain, log_arity);
        Self {
            config,
            round,
            dft,
            log_arity,
            fold_log_domain,
            fold_shift,
            next_log_domain: current_log_domain - 1,
            next_shift: next_domain_shift(current_shift, log_arity),
            folded_codeword: Vec::new(),
            fold_coeffs: Vec::new(),
            next_commit_codeword: Vec::new(),
            new_data: None,
            ood_points: Vec::new(),
            ood_answers: Vec::new(),
            query_indices: Vec::new(),
            query_points: Vec::new(),
            query_answers: Vec::new(),
            seen_query_indices: Vec::new(),
            ans_poly: Vec::new(),
            shake_poly: Vec::new(),
            all_points: Vec::new(),
        }
    }

    /// Fold at `gamma` and commit the folded oracle. The returned commitment is the round's
    /// first prover message and must be absorbed by the caller.
    fn fold_and_commit(
        &mut self,
        current_oracle_codeword: &[EF],
        current_shift: F,
        current_log_domain: usize,
        gamma: EF,
    ) -> M::Commitment {
        // `fold_codeword` interpolates at subgroup coordinates `g^{·}`. The codeword
        // lives on the coset `current_shift · <g>`, so passing `gamma / current_shift`
        // yields exactly Construction 4.5's coset fold at challenge `gamma`.
        let fold_beta = gamma * EF::from(current_shift.inverse());
        self.folded_codeword = tracing::debug_span!("fold_codeword").in_scope(|| {
            fold_codeword::<F, EF>(
                current_oracle_codeword,
                fold_beta,
                self.log_arity,
                current_log_domain,
            )
        });
        self.fold_coeffs = coeffs_from_codeword(self.dft, &self.folded_codeword, self.fold_shift);

        self.next_commit_codeword = codeword_from_coeffs(
            self.dft,
            self.fold_coeffs.clone(),
            self.next_shift,
            self.next_log_domain,
        );
        let (new_commit, new_data) =
            tracing::debug_span!("commit_as_fiber_matrix").in_scope(|| {
                commit_as_fiber_matrix(
                    &self.config.mmcs,
                    &self.next_commit_codeword,
                    self.config.log_folding_factor,
                )
            });
        self.new_data = Some(new_data);
        new_commit
    }

    /// The domains an OOD point must avoid: the current and next witness domains, and the
    /// fold-query domain. Excluding the fold domain prevents an honest-prover failure where an
    /// OOD point coincides with a sampled query point and the later interpolation hits
    /// duplicate roots.
    const fn ood_excluded_domains(
        &self,
        current_shift: F,
        current_log_domain: usize,
    ) -> [(F, usize); 3] {
        [
            (current_shift, current_log_domain),
            (self.next_shift, self.next_log_domain),
            (self.fold_shift, self.fold_log_domain),
        ]
    }

    /// Evaluate the folded polynomial at the sampled OOD points. The answers are the round's
    /// second prover message and must be absorbed by the caller.
    fn ood_answers(&mut self, ood_points: Vec<EF>) -> &[EF] {
        // `fold_coeffs` is padded to the next round's full domain size, but the folded
        // polynomial's true degree is bounded by the round's degree schedule (a fixed
        // factor smaller); evaluating only the non-trivially-zero prefix cuts Horner's
        // work by that same factor.
        let rc = &self.config.round_configs[self.round];
        let folded_degree_bound = 1usize << (rc.log_degree - self.log_arity);
        let truncated = &self.fold_coeffs[..folded_degree_bound.min(self.fold_coeffs.len())];

        self.ood_points = ood_points;
        self.ood_answers = self
            .ood_points
            .iter()
            .map(|&z| eval_poly_parallel(truncated, z))
            .collect();
        &self.ood_answers
    }

    /// Record the sampled query indices and the folded values they read.
    fn record_queries(&mut self, query_indices: Vec<usize>) {
        let fold_gen = F::two_adic_generator(self.fold_log_domain);
        let mut seen: alloc::collections::BTreeSet<usize> = alloc::collections::BTreeSet::new();

        for &j in &query_indices {
            if seen.insert(j) {
                self.query_points
                    .push(EF::from(self.fold_shift) * EF::from(fold_gen.exp_u64(j as u64)));
                self.query_answers.push(self.folded_codeword[j]);
            }
        }
        self.query_indices = query_indices;
        self.seen_query_indices = seen.into_iter().collect();
    }

    /// Interpolate the answer polynomial and derive the shake polynomial. Both are prover
    /// messages and must be absorbed by the caller before rho is squeezed — otherwise a
    /// malicious prover could fit Ans to satisfy the shake identity at a known rho.
    fn ans_shake(&mut self) -> (&[EF], &[EF]) {
        self.all_points = self
            .ood_points
            .iter()
            .chain(self.query_points.iter())
            .copied()
            .collect();
        let all_values: Vec<EF> = self
            .ood_answers
            .iter()
            .chain(self.query_answers.iter())
            .copied()
            .collect();

        self.ans_poly = interpolate_poly(&self.all_points, &all_values);
        self.shake_poly = compute_shake_polynomial(&self.ans_poly, &self.all_points);
        (&self.ans_poly, &self.shake_poly)
    }

    /// Evaluate the next virtual witness on the next round's domain. Touches no transcript
    /// state, so it may run after the caller has moved on to other instances.
    fn finish(mut self, r_comb: EF) -> RoundFinish<F, EF, M> {
        // Construction 5.2: f_{i+1} = DegCor((g_i − Ans_i) / Z_{G_i}).
        //
        // DegCor(x) = (1 - (r_comb*x)^{gap+1}) / (1 - r_comb*x) is geometric in x over the
        // coset (base-field ratio, EF-valued start), so it is evaluated pointwise via two
        // `Powers` sweeps instead of a third DFT; its `(1 - r_comb*x)` denominator is folded
        // into the vanishing-polynomial batch inversion below (one inversion, not two). Ans
        // and the vanishing polynomial still need evaluating — their roots are this round's
        // arbitrary interpolation points — but both are tiny next to the domain, so they go
        // through the low-degree coset evaluation rather than a full-size DFT each.
        let all_points = core::mem::take(&mut self.all_points);
        let next_shift = self.next_shift;
        let next_log_domain = self.next_log_domain;
        let num_answers = all_points.len();

        let vanishing_coeffs = vanishing_poly_from_roots(&all_points);
        // Ans interpolates `num_answers` points and the vanishing polynomial has exactly
        // `num_answers + 1` coefficients.
        let log_answer_len = log2_ceil_usize(num_answers + 1).min(next_log_domain);
        let (ans_evals, vanishing_evals) = tracing::debug_span!("eval_low_degree_pair_on_coset")
            .in_scope(|| {
                eval_low_degree_pair_on_coset(
                    self.dft,
                    &self.ans_poly,
                    &vanishing_coeffs,
                    next_shift,
                    next_log_domain,
                    log_answer_len,
                )
            });

        // x_j = next_shift * g^j, so step_j = r_comb * x_j = step_start * g^j, and the degree
        // correction's numerator sweeps the same coset at stride `num_answers + 1`. Both are
        // geometric with a base-field ratio, so each chunk seeds one exponentiation and then
        // advances by a base-field multiply: no power tables, and no ext-by-ext products.
        const POWER_CHUNK: usize = 1 << 12;
        let g_next = F::two_adic_generator(next_log_domain);
        let g_next_hi = g_next.exp_u64((num_answers + 1) as u64);
        let step_start = r_comb * next_shift;
        let step_start_hi = step_start.exp_u64((num_answers + 1) as u64);

        // The quotient denominators are the vanishing evaluations scaled by the degree
        // correction's `(1 - step)` denominator, so they are formed in place.
        let next_oracle_codeword = tracing::debug_span!("quotient_sweep").in_scope(|| {
            let mut combined_denoms = vanishing_evals;
            combined_denoms
                .par_chunks_mut(POWER_CHUNK)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let mut step = step_start * g_next.exp_u64((chunk_idx * POWER_CHUNK) as u64);
                    for denom in chunk.iter_mut() {
                        *denom *= EF::ONE - step;
                        step *= g_next;
                    }
                });
            let combined_inverses = batch_multiplicative_inverse(&combined_denoms);
            drop(combined_denoms);

            // `commit_as_fiber_matrix` has already copied the committed codeword into the
            // Merkle tree, so the next oracle is formed in place over it.
            let mut next_oracle_codeword = core::mem::take(&mut self.next_commit_codeword);
            next_oracle_codeword
                .par_chunks_mut(POWER_CHUNK)
                .zip(ans_evals.par_chunks(POWER_CHUNK))
                .zip(combined_inverses.par_chunks(POWER_CHUNK))
                .enumerate()
                .for_each(|(chunk_idx, ((chunk, ans_chunk), inverse_chunk))| {
                    let mut numerator_step =
                        step_start_hi * g_next_hi.exp_u64((chunk_idx * POWER_CHUNK) as u64);
                    for ((value, &ans), &inverse) in
                        chunk.iter_mut().zip(ans_chunk).zip(inverse_chunk)
                    {
                        *value = (*value - ans) * inverse * (EF::ONE - numerator_step);
                        numerator_step *= g_next_hi;
                    }
                });
            next_oracle_codeword
        });

        RoundFinish {
            next_codeword: next_oracle_codeword,
            next_commit_data: self.new_data.take().expect("set by `fold_and_commit`"),
            next_shift,
            next_log_domain,
            seen_query_indices: core::mem::take(&mut self.seen_query_indices),
            ans_poly: core::mem::take(&mut self.ans_poly),
            shake_poly: core::mem::take(&mut self.shake_poly),
        }
    }
}

/// Everything a finished round hands back once its transcript phases are done.
struct RoundFinish<F, EF: Field, M: Mmcs<EF>> {
    next_codeword: Vec<EF>,
    next_commit_data: M::ProverData<RowMajorMatrix<EF>>,
    next_shift: F,
    next_log_domain: usize,
    seen_query_indices: Vec<usize>,
    ans_poly: Vec<EF>,
    shake_poly: Vec<EF>,
}

/// Prove one intermediate STIR round (Construction 5.2): fold the current oracle, sample OOD
/// and query points, answer them, commit the folded oracle, and evaluate the next virtual
/// witness directly on the next round's domain.
#[allow(clippy::too_many_arguments)]
#[instrument(skip_all)]
fn prove_round<F, EF, Dft, M, Challenger>(
    config: &StirConfig<F, EF, M, Challenger>,
    round: usize,
    dft: &Dft,
    challenger: &mut Challenger,
    current_oracle_codeword: &[EF],
    current_shift: F,
    current_log_domain: usize,
    current_commit_data: Option<&M::ProverData<RowMajorMatrix<EF>>>,
) -> RoundOutput<F, EF, M, Challenger::Witness>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField + BasedVectorSpace<F>,
    Dft: TwoAdicSubgroupDft<F>,
    M: Mmcs<EF>,
    Challenger: FieldChallenger<F>
        + CanObserve<M::Commitment>
        + GrindingChallenger<Witness = F>
        + CanSampleUniformBits<F>,
{
    let rc = &config.round_configs[round];
    let arity = 1 << rc.log_folding_factor;

    let mut state = RoundProver::new(config, round, dft, current_shift, current_log_domain);

    // Step 1: fold. Derive gamma after folding PoW.
    let folding_pow_witness = challenger.grind(rc.folding_pow_bits);
    let gamma: EF = challenger.sample_algebra_element();

    let new_commit = state.fold_and_commit(
        current_oracle_codeword,
        current_shift,
        current_log_domain,
        gamma,
    );
    challenger.observe(new_commit.clone());

    // Step 2: OOD sampling.
    let ood_points = sample_ood_points(
        challenger,
        state.ood_excluded_domains(current_shift, current_log_domain),
        rc.num_ood_samples,
    );
    let ood_answers = state.ood_answers(ood_points).to_vec();
    challenger.observe_algebra_slice(&ood_answers);

    // Step 3: query-phase PoW. It protects the immediately following combination challenge
    // and query indices. It does not strengthen the earlier OOD samples or the later shake
    // challenge, which is separated from this grind by prover-controlled messages.
    let pow_witness = challenger.grind(rc.pow_bits);

    // Step 4: Query sampling.
    let r_comb: EF = challenger.sample_algebra_element();

    let query_indices: Vec<usize> = (0..rc.num_queries)
        .map(|_| {
            // `RESAMPLE = true`: the challenger loops on field-side rejection internally,
            // so this `expect` is unreachable for every challenger in this workspace.
            // Unbiased sampling is required because `sample_bits` carries a per-draw modular
            // bias of `2^fold_log_domain / |F|`, which is non-negligible over 31-bit fields.
            challenger
                .sample_uniform_bits::<true>(state.fold_log_domain)
                .expect("RESAMPLE = true: rejection loops internally, never errors")
        })
        .collect();
    state.record_queries(query_indices);

    // One shared, pruned multi-opening proof for every query drawn this round. Absent when
    // this round reads an external initial oracle, whose fibers the verifier rebuilds.
    let query_openings = current_commit_data
        .as_ref()
        .map(|data| open_fiber_rows(&config.mmcs, &state.query_indices, data));
    debug_assert!(
        query_openings
            .iter()
            .all(|o| o.row_evals.iter().all(|row| row.len() == arity))
    );

    // Step 4: Answer polynomial, shake polynomial, and shake-check challenge.
    let (ans_poly, shake_poly) = state.ans_shake();
    // Bind ans_poly into the transcript BEFORE rho is sampled — otherwise a malicious prover
    // could fit Ans to satisfy the shake identity at a known rho.
    challenger.observe_algebra_slice(ans_poly);
    challenger.observe_algebra_slice(shake_poly);

    // Sample and discard the shake-check challenge so the transcript state
    // stays consistent with the verifier.
    let _rho: EF = challenger.sample_algebra_element();

    // Step 5: Construction 5.2 — evaluate the next virtual witness directly on L_{i+1}:
    // f_{i+1} = DegCor((g_i − Ans_i) / Z_{G_i}).
    //
    // DegCor(x) = (1 - (r_comb*x)^{gap+1}) / (1 - r_comb*x) is geometric in x over the
    // coset (base-field ratio, EF-valued start), so it is evaluated pointwise via two
    // `Powers` sweeps instead of a third DFT; its `(1 - r_comb*x)` denominator is folded
    // into the vanishing-polynomial batch inversion below (one inversion, not two). Ans
    // and the vanishing polynomial still need evaluating — their roots are this round's
    // arbitrary interpolation points — but both are tiny next to the domain, so they go
    // through the low-degree coset evaluation rather than a full-size DFT each.
    let finish = state.finish(r_comb);

    RoundOutput {
        proof: StirRoundProof {
            commitment: new_commit,
            folding_pow_witness,
            ood_answers,
            pow_witness,
            ans_polynomial: finish.ans_poly,
            shake_polynomial: finish.shake_poly,
            query_openings,
        },
        next_codeword: finish.next_codeword,
        next_commit_data: finish.next_commit_data,
        next_shift: finish.next_shift,
        next_log_domain: finish.next_log_domain,
        seen_query_indices: finish.seen_query_indices,
    }
}

/// Output of proving the final STIR round.
struct FinalRoundOutput<EF: Field, M: Mmcs<EF>, Witness> {
    final_polynomial: Vec<EF>,
    final_folding_pow_witness: Witness,
    final_pow_witness: Witness,
    final_query_openings: Option<StirQueryOpenings<EF, M>>,
    /// Deduplicated, ascending fold-domain query indices this round drew. Only consumed by
    /// the caller when there were no intermediate rounds.
    seen_query_indices: Vec<usize>,
}

/// One instance's in-flight final round, advanced in the order the transcript demands.
///
/// The phases, in order: \[grind\] `fold_and_derive` -> absorb final polynomial -> \[grind\] ->
/// squeeze query indices -> `record_queries` -> `finish`.
struct FinalRoundProver<'a, F, EF: Field, Dft, M: Mmcs<EF>, Challenger> {
    config: &'a StirConfig<F, EF, M, Challenger>,
    dft: &'a Dft,

    current_shift: F,
    current_log_domain: usize,
    final_new_log_domain: usize,
    final_new_shift: F,

    final_codeword: Vec<EF>,
    final_poly: Vec<EF>,

    query_indices: Vec<usize>,
    seen_query_indices: Vec<usize>,
}

impl<'a, F, EF, Dft, M, Challenger> FinalRoundProver<'a, F, EF, Dft, M, Challenger>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField + BasedVectorSpace<F>,
    Dft: TwoAdicSubgroupDft<F>,
    M: Mmcs<EF>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    /// Fix the final round's domain geometry. Touches no transcript state.
    fn new(
        config: &'a StirConfig<F, EF, M, Challenger>,
        dft: &'a Dft,
        current_shift: F,
        current_log_domain: usize,
    ) -> Self {
        let final_log_arity = config.log_folding_factor;
        let (final_new_log_domain, final_new_shift) =
            fold_domain_params(current_shift, current_log_domain, final_log_arity);
        Self {
            config,
            dft,
            current_shift,
            current_log_domain,
            final_new_log_domain,
            final_new_shift,
            final_codeword: Vec::new(),
            final_poly: Vec::new(),
            query_indices: Vec::new(),
            seen_query_indices: Vec::new(),
        }
    }

    /// Fold at `final_gamma` and derive the final polynomial. The returned coefficients are
    /// the round's only prover message and must be absorbed by the caller.
    fn fold_and_derive(&mut self, current_oracle_codeword: &[EF], final_gamma: EF) -> &[EF] {
        let final_log_arity = self.config.log_folding_factor;
        // See the round-fold note in `RoundProver::fold_and_commit`: `gamma / current_shift`
        // over subgroup coordinates is the paper's coset fold at challenge `gamma`.
        let final_fold_beta = final_gamma * EF::from(self.current_shift.inverse());
        self.final_codeword = tracing::debug_span!("fold_codeword").in_scope(|| {
            fold_codeword::<F, EF>(
                current_oracle_codeword,
                final_fold_beta,
                final_log_arity,
                self.current_log_domain,
            )
        });
        // The final polynomial has only `final_len` coefficients, far fewer than
        // `final_codeword`'s full domain size. Rather than run a full-size iDFT and discard
        // the (necessarily zero) high coefficients, gather a `final_len`-sized coset — every
        // `stride`-th natural-order point, which is exactly the subgroup coset of that size —
        // and run the small iDFT directly on it.
        let final_len = self.config.final_poly_len();
        let stride = self.final_codeword.len() / final_len;
        let final_poly_evals: Vec<EF> = (0..final_len)
            .map(|i| self.final_codeword[i * stride])
            .collect();
        self.final_poly = coeffs_from_codeword(self.dft, &final_poly_evals, self.final_new_shift);
        &self.final_poly
    }

    /// Record the sampled query indices.
    fn record_queries(&mut self, query_indices: Vec<usize>) {
        let mut seen: alloc::collections::BTreeSet<usize> = alloc::collections::BTreeSet::new();
        for &j in &query_indices {
            seen.insert(j);
        }
        self.query_indices = query_indices;
        self.seen_query_indices = seen.into_iter().collect();
    }

    /// Touches no transcript state, so it may run after the caller has moved on to other
    /// instances.
    fn finish(self) -> FinalRoundFinish<EF> {
        FinalRoundFinish {
            final_poly: self.final_poly,
            seen_query_indices: self.seen_query_indices,
        }
    }
}

/// Everything a finished final round hands back once its transcript phases are done.
struct FinalRoundFinish<EF: Field> {
    final_poly: Vec<EF>,
    seen_query_indices: Vec<usize>,
}

/// Prove the final STIR round: fold the last committed codeword and send the resulting
/// low-degree polynomial directly, rather than committing and querying it again.
#[instrument(skip_all)]
fn prove_final_round<F, EF, Dft, M, Challenger>(
    config: &StirConfig<F, EF, M, Challenger>,
    dft: &Dft,
    challenger: &mut Challenger,
    current_oracle_codeword: &[EF],
    current_shift: F,
    current_log_domain: usize,
    current_commit_data: Option<&M::ProverData<RowMajorMatrix<EF>>>,
) -> FinalRoundOutput<EF, M, Challenger::Witness>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField + BasedVectorSpace<F>,
    Dft: TwoAdicSubgroupDft<F>,
    M: Mmcs<EF>,
    Challenger: FieldChallenger<F>
        + CanObserve<M::Commitment>
        + GrindingChallenger<Witness = F>
        + CanSampleUniformBits<F>,
{
    let final_arity = 1usize << config.log_folding_factor;
    let mut state = FinalRoundProver::new(config, dft, current_shift, current_log_domain);

    let final_folding_pow_witness = challenger.grind(config.final_folding_pow_bits);
    let final_gamma: EF = challenger.sample_algebra_element();

    let final_poly = state.fold_and_derive(current_oracle_codeword, final_gamma);
    challenger.observe_algebra_slice(final_poly);

    let final_pow_witness = challenger.grind(config.final_pow_bits);

    let mut final_query_indices = Vec::with_capacity(config.final_queries);
    for _ in 0..config.final_queries {
        let j = challenger
            .sample_uniform_bits::<true>(state.final_new_log_domain)
            .expect("RESAMPLE = true: rejection loops internally, never errors");
        final_query_indices.push(j);
    }
    state.record_queries(final_query_indices);

    let final_query_openings = current_commit_data
        .as_ref()
        .map(|data| open_fiber_rows(&config.mmcs, &state.query_indices, data));
    debug_assert!(
        final_query_openings
            .iter()
            .all(|o| o.row_evals.iter().all(|row| row.len() == final_arity))
    );

    let finish = state.finish();

    FinalRoundOutput {
        final_polynomial: finish.final_poly,
        final_folding_pow_witness,
        final_pow_witness,
        final_query_openings,
        seen_query_indices: finish.seen_query_indices,
    }
}

/// Shared body of [`prove_stir_from_codeword`] and [`prove_stir_from_external_codeword`].
///
/// `commit_initial` selects whether the initial oracle is committed and opened by STIR.
fn prove_stir_inner<F, EF, Dft, M, Challenger>(
    config: &StirConfig<F, EF, M, Challenger>,
    initial_codeword: Vec<EF>,
    dft: &Dft,
    challenger: &mut Challenger,
    commit_initial: bool,
) -> (StirProof<EF, M, Challenger::Witness>, Vec<usize>)
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField + BasedVectorSpace<F>,
    Dft: TwoAdicSubgroupDft<F>,
    M: Mmcs<EF>,
    Challenger: FieldChallenger<F>
        + CanObserve<M::Commitment>
        + GrindingChallenger<Witness = F>
        + CanSampleUniformBits<F>,
{
    let num_rounds = config.num_rounds();
    let initial_shift = F::GENERATOR;
    let log_initial_domain = config.log_starting_domain_size();
    let initial_domain_size = 1 << log_initial_domain;
    assert_eq!(
        initial_codeword.len(),
        initial_domain_size,
        "initial STIR codeword length must match the configured starting domain"
    );

    // Commit before moving the codeword into the round state, avoiding a full clone.
    let (initial_commit, initial_data) = if commit_initial {
        let (commit, data) =
            commit_as_fiber_matrix(&config.mmcs, &initial_codeword, config.log_folding_factor);
        challenger.observe(commit.clone());
        (Some(commit), Some(data))
    } else {
        (None, None)
    };

    let mut current_oracle_codeword = initial_codeword;
    let mut current_shift = initial_shift;
    let mut current_log_domain = log_initial_domain;

    let mut current_commit_data = initial_data;

    let mut round_proofs = Vec::with_capacity(num_rounds);

    // Collect first-round query fold-domain indices (for PCS binding).
    let mut first_round_query_indices = Vec::new();

    // Intermediate rounds (Construction 5.2).
    for round in 0..num_rounds {
        let output = prove_round(
            config,
            round,
            dft,
            challenger,
            &current_oracle_codeword,
            current_shift,
            current_log_domain,
            current_commit_data.as_ref(),
        );

        if round == 0 {
            first_round_query_indices = output.seen_query_indices;
        }

        round_proofs.push(output.proof);
        current_oracle_codeword = output.next_codeword;
        current_commit_data = Some(output.next_commit_data);
        current_shift = output.next_shift;
        current_log_domain = output.next_log_domain;
    }

    // Final round: fold the last committed codeword and send the resulting polynomial.
    let final_output = prove_final_round(
        config,
        dft,
        challenger,
        &current_oracle_codeword,
        current_shift,
        current_log_domain,
        current_commit_data.as_ref(),
    );

    // When there are no intermediate rounds the final queries target the
    // initial codeword.  Expose them for PCS input binding.
    if num_rounds == 0 {
        first_round_query_indices = final_output.seen_query_indices;
    }

    let proof = StirProof {
        initial_commitment: initial_commit,
        round_proofs,
        final_polynomial: final_output.final_polynomial,
        final_folding_pow_witness: final_output.final_folding_pow_witness,
        final_pow_witness: final_output.final_pow_witness,
        final_query_openings: final_output.final_query_openings,
    };
    (proof, first_round_query_indices)
}

/// One `(proof, first_round_query_indices)` pair per instance, in `configs` order.
type StirMultiOutput<EF, M, Witness> = Vec<(StirProof<EF, M, Witness>, Vec<usize>)>;

/// Per-instance mutable state threaded through [`prove_stir_multi_inner`]'s round loop.
struct MultiInstanceState<F, EF: Field, M: Mmcs<EF>> {
    codeword: Vec<EF>,
    shift: F,
    log_domain: usize,
    commit_data: Option<M::ProverData<RowMajorMatrix<EF>>>,
    initial_commitment: Option<M::Commitment>,
    round_proofs: Vec<StirRoundProof<EF, M, F>>,
    first_round_query_indices: Vec<usize>,
}

/// Prove low degree for `B` polynomials of possibly different degrees in lockstep, sharing
/// every grind across the instances active at that site.
///
/// Instances are right-aligned: instance `i`, with `Mi = configs[i].num_rounds()`
/// intermediate rounds, runs its local rounds at global rounds `max_m - Mi .. max_m`
/// (`max_m = maxᵢ Mi`), so every instance reaches its final round on the same global step. At
/// each global round the shared grind bits are the max, over the instances active at that
/// round, of THAT instance's local round's bits: the round error is `maxᵢ εᵢ`, not `Σᵢ εᵢ`,
/// because the statement is false in specific instances, fixed before the grind, and the
/// adversary wins only if every doomed instance's fresh challenge lands lucky. Widening the
/// active set adds hurdles, not chances, so no union-bound penalty applies.
///
/// The shared witness is replicated verbatim into every active instance's own proof slot;
/// `verifier::verify_stir_multi_inner` checks the copies agree before checking the grind once.
///
/// Returns one `(proof, first_round_query_indices)` pair per instance, in `configs` order.
fn prove_stir_multi_inner<F, EF, Dft, M, Challenger>(
    configs: &[&StirConfig<F, EF, M, Challenger>],
    initial_codewords: Vec<Vec<EF>>,
    dft: &Dft,
    challenger: &mut Challenger,
    commit_initial: bool,
) -> StirMultiOutput<EF, M, Challenger::Witness>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField + BasedVectorSpace<F>,
    Dft: TwoAdicSubgroupDft<F>,
    M: Mmcs<EF>,
    Challenger: FieldChallenger<F>
        + CanObserve<M::Commitment>
        + GrindingChallenger<Witness = F>
        + CanSampleUniformBits<F>,
{
    let b = configs.len();
    assert_eq!(
        initial_codewords.len(),
        b,
        "one initial codeword per instance"
    );

    let mut states: Vec<MultiInstanceState<F, EF, M>> = Vec::with_capacity(b);
    for (i, codeword) in initial_codewords.into_iter().enumerate() {
        let log_initial_domain = configs[i].log_starting_domain_size();
        assert_eq!(
            codeword.len(),
            1 << log_initial_domain,
            "initial STIR codeword length must match the configured starting domain"
        );
        let (initial_commitment, commit_data) = if commit_initial {
            let (commit, data) =
                commit_as_fiber_matrix(&configs[i].mmcs, &codeword, configs[i].log_folding_factor);
            challenger.observe(commit.clone());
            (Some(commit), Some(data))
        } else {
            (None, None)
        };
        states.push(MultiInstanceState {
            codeword,
            shift: F::GENERATOR,
            log_domain: log_initial_domain,
            commit_data,
            initial_commitment,
            round_proofs: Vec::with_capacity(configs[i].num_rounds()),
            first_round_query_indices: Vec::new(),
        });
    }

    let max_m = configs.iter().map(|c| c.num_rounds()).max().unwrap_or(0);
    let offset = |i: usize| max_m - configs[i].num_rounds();

    for r in 0..max_m {
        let active: Vec<usize> = (0..b).filter(|&i| offset(i) <= r).collect();

        // [grind folding_pow_bits], shared across every active instance's local round.
        let shared_folding_bits = active
            .iter()
            .map(|&i| configs[i].round_configs[r - offset(i)].folding_pow_bits)
            .max()
            .expect("`active` is non-empty for r < max_m");
        let folding_pow_witness = challenger.grind(shared_folding_bits);

        // Phase 1: per-instance folding challenge, fold, commit, and absorb the commitment.
        struct Phase1<'a, F, EF: Field, Dft, M: Mmcs<EF>, Challenger> {
            rp: RoundProver<'a, F, EF, Dft, M, Challenger>,
            commit: M::Commitment,
        }
        let phase1: Vec<Phase1<'_, F, EF, Dft, M, Challenger>> = active
            .iter()
            .map(|&i| {
                let local_r = r - offset(i);
                let mut rp = RoundProver::new(
                    configs[i],
                    local_r,
                    dft,
                    states[i].shift,
                    states[i].log_domain,
                );
                let gamma: EF = challenger.sample_algebra_element();
                let commit = rp.fold_and_commit(
                    &states[i].codeword,
                    states[i].shift,
                    states[i].log_domain,
                    gamma,
                );
                challenger.observe(commit.clone());
                Phase1 { rp, commit }
            })
            .collect();

        // Phase 2: per-instance OOD sampling and answer absorb.
        struct Phase2<'a, F, EF: Field, Dft, M: Mmcs<EF>, Challenger> {
            rp: RoundProver<'a, F, EF, Dft, M, Challenger>,
            commit: M::Commitment,
            ood_answers: Vec<EF>,
        }
        let phase2: Vec<Phase2<'_, F, EF, Dft, M, Challenger>> = active
            .iter()
            .zip(phase1)
            .map(|(&i, p)| {
                let rc = &configs[i].round_configs[r - offset(i)];
                let mut rp = p.rp;
                let ood_points = sample_ood_points(
                    challenger,
                    rp.ood_excluded_domains(states[i].shift, states[i].log_domain),
                    rc.num_ood_samples,
                );
                let ood_answers = rp.ood_answers(ood_points).to_vec();
                challenger.observe_algebra_slice(&ood_answers);
                Phase2 {
                    rp,
                    commit: p.commit,
                    ood_answers,
                }
            })
            .collect();

        // [grind pow_bits], shared across every active instance's local round.
        let shared_pow_bits = active
            .iter()
            .map(|&i| configs[i].round_configs[r - offset(i)].pow_bits)
            .max()
            .expect("`active` is non-empty for r < max_m");
        let pow_witness = challenger.grind(shared_pow_bits);

        // Phase 3: per-instance combination challenge, query sampling, and query openings.
        struct Phase3<'a, F, EF: Field, Dft, M: Mmcs<EF>, Challenger> {
            rp: RoundProver<'a, F, EF, Dft, M, Challenger>,
            commit: M::Commitment,
            ood_answers: Vec<EF>,
            query_openings: Option<StirQueryOpenings<EF, M>>,
            r_comb: EF,
        }
        let phase3: Vec<Phase3<'_, F, EF, Dft, M, Challenger>> = active
            .iter()
            .zip(phase2)
            .map(|(&i, p)| {
                let rc = &configs[i].round_configs[r - offset(i)];
                let mut rp = p.rp;
                let r_comb: EF = challenger.sample_algebra_element();
                let query_indices: Vec<usize> = (0..rc.num_queries)
                    .map(|_| {
                        challenger
                            .sample_uniform_bits::<true>(rp.fold_log_domain)
                            .expect("RESAMPLE = true: rejection loops internally, never errors")
                    })
                    .collect();
                rp.record_queries(query_indices);

                let query_openings = states[i]
                    .commit_data
                    .as_ref()
                    .map(|data| open_fiber_rows(&configs[i].mmcs, &rp.query_indices, data));
                debug_assert!(query_openings.iter().all(|o| {
                    o.row_evals
                        .iter()
                        .all(|row| row.len() == 1usize << rp.log_arity)
                }));

                Phase3 {
                    rp,
                    commit: p.commit,
                    ood_answers: p.ood_answers,
                    query_openings,
                    r_comb,
                }
            })
            .collect();

        // Phase 4: per-instance answer/shake polynomial absorb and shake-check challenge.
        struct Phase4<'a, F, EF: Field, Dft, M: Mmcs<EF>, Challenger> {
            rp: RoundProver<'a, F, EF, Dft, M, Challenger>,
            commit: M::Commitment,
            ood_answers: Vec<EF>,
            query_openings: Option<StirQueryOpenings<EF, M>>,
            r_comb: EF,
        }
        let phase4: Vec<Phase4<'_, F, EF, Dft, M, Challenger>> = phase3
            .into_iter()
            .map(|p| {
                let mut rp = p.rp;
                let (ans, shake) = rp.ans_shake();
                let ans = ans.to_vec();
                let shake = shake.to_vec();
                challenger.observe_algebra_slice(&ans);
                challenger.observe_algebra_slice(&shake);
                let _rho: EF = challenger.sample_algebra_element();
                Phase4 {
                    rp,
                    commit: p.commit,
                    ood_answers: p.ood_answers,
                    query_openings: p.query_openings,
                    r_comb: p.r_comb,
                }
            })
            .collect();

        // Phase 5 (finish): touches no transcript state, so instance order no longer matters.
        for (&i, p) in active.iter().zip(phase4) {
            let finish = p.rp.finish(p.r_comb);
            states[i].round_proofs.push(StirRoundProof {
                commitment: p.commit,
                folding_pow_witness,
                ood_answers: p.ood_answers,
                pow_witness,
                ans_polynomial: finish.ans_poly,
                shake_polynomial: finish.shake_poly,
                query_openings: p.query_openings,
            });
            if r == offset(i) {
                states[i].first_round_query_indices = finish.seen_query_indices;
            }
            states[i].codeword = finish.next_codeword;
            states[i].commit_data = Some(finish.next_commit_data);
            states[i].shift = finish.next_shift;
            states[i].log_domain = finish.next_log_domain;
        }
    }

    // Final round: every instance reaches it on this same global step (right-alignment).
    let shared_final_folding_bits = configs
        .iter()
        .map(|c| c.final_folding_pow_bits)
        .max()
        .unwrap_or(0);
    let final_folding_pow_witness = challenger.grind(shared_final_folding_bits);

    let mut final_provers: Vec<FinalRoundProver<'_, F, EF, Dft, M, Challenger>> =
        Vec::with_capacity(b);
    for i in 0..b {
        let mut frp = FinalRoundProver::new(configs[i], dft, states[i].shift, states[i].log_domain);
        let final_gamma: EF = challenger.sample_algebra_element();
        let final_poly = frp.fold_and_derive(&states[i].codeword, final_gamma);
        challenger.observe_algebra_slice(final_poly);
        final_provers.push(frp);
    }

    let shared_final_pow_bits = configs.iter().map(|c| c.final_pow_bits).max().unwrap_or(0);
    let final_pow_witness = challenger.grind(shared_final_pow_bits);

    let mut results = Vec::with_capacity(b);
    for (i, mut frp) in final_provers.into_iter().enumerate() {
        let final_arity = 1usize << configs[i].log_folding_factor;
        let query_indices: Vec<usize> = (0..configs[i].final_queries)
            .map(|_| {
                challenger
                    .sample_uniform_bits::<true>(frp.final_new_log_domain)
                    .expect("RESAMPLE = true: rejection loops internally, never errors")
            })
            .collect();
        frp.record_queries(query_indices);

        let final_query_openings = states[i]
            .commit_data
            .as_ref()
            .map(|data| open_fiber_rows(&configs[i].mmcs, &frp.query_indices, data));
        debug_assert!(
            final_query_openings
                .iter()
                .all(|o| o.row_evals.iter().all(|row| row.len() == final_arity))
        );

        let finish = frp.finish();
        if configs[i].num_rounds() == 0 {
            states[i].first_round_query_indices = finish.seen_query_indices;
        }

        let proof = StirProof {
            initial_commitment: states[i].initial_commitment.clone(),
            round_proofs: core::mem::take(&mut states[i].round_proofs),
            final_polynomial: finish.final_poly,
            final_folding_pow_witness,
            final_pow_witness,
            final_query_openings,
        };
        results.push((
            proof,
            core::mem::take(&mut states[i].first_round_query_indices),
        ));
    }

    results
}

/// Prove low degree for `B` polynomials of possibly different degrees (given in coefficient
/// form over `EF`), sharing every grind across the STIR instances active at that grind's site.
///
/// See `prove_stir_multi_inner` for the shared-grind soundness argument.
#[instrument(skip_all)]
pub fn prove_stir_multi<F, EF, Dft, M, Challenger>(
    configs: &[&StirConfig<F, EF, M, Challenger>],
    poly_coeffs: Vec<Vec<EF>>,
    dft: &Dft,
    challenger: &mut Challenger,
) -> StirMultiOutput<EF, M, Challenger::Witness>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField + BasedVectorSpace<F>,
    Dft: TwoAdicSubgroupDft<F>,
    M: Mmcs<EF>,
    Challenger: FieldChallenger<F>
        + CanObserve<M::Commitment>
        + GrindingChallenger<Witness = F>
        + CanSampleUniformBits<F>,
{
    assert_eq!(
        configs.len(),
        poly_coeffs.len(),
        "one polynomial per instance"
    );
    let initial_codewords = configs
        .iter()
        .zip(poly_coeffs)
        .map(|(config, coeffs)| {
            let log_initial_domain = config.log_starting_domain_size();
            let mut coeffs = coeffs;
            coeffs.resize(1 << log_initial_domain, EF::ZERO);
            codeword_from_coeffs(dft, coeffs, F::GENERATOR, log_initial_domain)
        })
        .collect();
    prove_stir_multi_inner(configs, initial_codewords, dft, challenger, true)
}

/// Prove low degree for `B` initial natural-order codewords on each instance's starting
/// domain, sharing every grind across the STIR instances active at that grind's site.
///
/// See [`prove_stir_from_codeword`] for the codeword layout requirement, applied per instance,
/// and `prove_stir_multi_inner` for the shared-grind soundness argument.
#[instrument(skip_all)]
pub fn prove_stir_multi_from_codewords<F, EF, Dft, M, Challenger>(
    configs: &[&StirConfig<F, EF, M, Challenger>],
    initial_codewords: Vec<Vec<EF>>,
    dft: &Dft,
    challenger: &mut Challenger,
) -> StirMultiOutput<EF, M, Challenger::Witness>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField + BasedVectorSpace<F>,
    Dft: TwoAdicSubgroupDft<F>,
    M: Mmcs<EF>,
    Challenger: FieldChallenger<F>
        + CanObserve<M::Commitment>
        + GrindingChallenger<Witness = F>
        + CanSampleUniformBits<F>,
{
    prove_stir_multi_inner(configs, initial_codewords, dft, challenger, true)
}

/// Prove low degree for `B` initial codewords the caller has already bound (see
/// [`prove_stir_from_external_codeword`]), sharing every grind across the STIR instances
/// active at that grind's site.
///
/// # Soundness requirement
///
/// See [`prove_stir_from_external_codeword`]: each caller-supplied codeword must already be
/// pinned by data absorbed into `challenger` before this call, and independently for each
/// instance.
///
/// See `prove_stir_multi_inner` for the shared-grind soundness argument.
#[instrument(skip_all)]
pub fn prove_stir_multi_from_external_codewords<F, EF, Dft, M, Challenger>(
    configs: &[&StirConfig<F, EF, M, Challenger>],
    initial_codewords: Vec<Vec<EF>>,
    dft: &Dft,
    challenger: &mut Challenger,
) -> StirMultiOutput<EF, M, Challenger::Witness>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField + BasedVectorSpace<F>,
    Dft: TwoAdicSubgroupDft<F>,
    M: Mmcs<EF>,
    Challenger: FieldChallenger<F>
        + CanObserve<M::Commitment>
        + GrindingChallenger<Witness = F>
        + CanSampleUniformBits<F>,
{
    prove_stir_multi_inner(configs, initial_codewords, dft, challenger, false)
}

/// Evaluate two polynomials of at most `2^log_len` coefficients on the coset `shift * <g>` of
/// size `2^log_size`, returning both codewords in **natural order**.
///
/// A full-size DFT would spend all `log_size` butterfly layers on an input that is zero past
/// its first `2^log_len` coefficients. Instead, split the coset into
/// `m = 2^(log_size - log_len)` cosets of the size-`2^log_len` subgroup: writing the natural
/// index as `i = a + m*b` with `a < m` and `b < 2^log_len`,
///
/// ```text
/// x_i = shift * g^(a + m*b) = (shift * g^a) * (g^m)^b
/// ```
///
/// and `g^m` generates that subgroup. So evaluating on the whole coset is `m` independent
/// size-`2^log_len` DFTs, one per `a`, of the coefficients pre-scaled by `(shift * g^a)^c` —
/// `O(N * log_len)` instead of `O(N * log_size)`. The `m` transforms are the columns of one
/// batched call, and each output row lands on a contiguous natural-order block.
///
/// Coefficients past index `2^log_size` are ignored, matching evaluation of the truncation.
fn eval_low_degree_pair_on_coset<F, EF, Dft>(
    dft: &Dft,
    first: &[EF],
    second: &[EF],
    shift: F,
    log_size: usize,
    log_len: usize,
) -> (Vec<EF>, Vec<EF>)
where
    F: TwoAdicField,
    EF: ExtensionField<F> + BasedVectorSpace<F>,
    Dft: TwoAdicSubgroupDft<F>,
{
    debug_assert!(log_len <= log_size);
    let size = 1usize << log_size;
    let num_cosets = size >> log_len;
    let generator = F::two_adic_generator(log_size);

    // Row `c` holds the pair of degree-`c` coefficients scaled by `(shift * g^a)^c`, which as
    // `a` varies is the geometric sequence starting at `shift^c` with ratio `g^c`.
    let mut scaled = EF::zero_vec((1usize << log_len) * 2 * num_cosets);
    scaled
        .par_chunks_mut(2 * num_cosets)
        .enumerate()
        .for_each(|(c, row)| {
            let first_c = first.get(c).copied().unwrap_or(EF::ZERO);
            let second_c = second.get(c).copied().unwrap_or(EF::ZERO);
            let ratio = generator.exp_u64(c as u64);
            let mut scale = shift.exp_u64(c as u64);
            for pair in row.chunks_exact_mut(2) {
                pair[0] = first_c * scale;
                pair[1] = second_c * scale;
                scale *= ratio;
            }
        });

    let transformed = dft
        .dft_algebra_batch(RowMajorMatrix::new(scaled, 2 * num_cosets))
        .values;

    // Transform row `b` holds the evaluations at `i = a + num_cosets * b` for every `a`, i.e.
    // the natural-order block `[num_cosets * b, num_cosets * (b + 1))`.
    let mut first_evals = EF::zero_vec(size);
    let mut second_evals = EF::zero_vec(size);
    first_evals
        .par_chunks_mut(num_cosets)
        .zip(second_evals.par_chunks_mut(num_cosets))
        .zip(transformed.par_chunks_exact(2 * num_cosets))
        .for_each(|((first_block, second_block), row)| {
            for ((first_slot, second_slot), pair) in first_block
                .iter_mut()
                .zip(second_block.iter_mut())
                .zip(row.chunks_exact(2))
            {
                *first_slot = pair[0];
                *second_slot = pair[1];
            }
        });

    (first_evals, second_evals)
}

/// Evaluate a polynomial (coefficients in `EF`) on a coset `shift * <g>` of size
/// `2^log_size`, returning the codeword in **natural order**.
pub fn codeword_from_coeffs<F, EF, Dft>(
    dft: &Dft,
    coeffs: Vec<EF>,
    shift: F,
    log_size: usize,
) -> Vec<EF>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + BasedVectorSpace<F>,
    Dft: TwoAdicSubgroupDft<F>,
{
    let size = 1 << log_size;
    let mut padded = coeffs;
    padded.resize(size, EF::ZERO);

    let mat = RowMajorMatrix::new_col(padded);
    let result = dft.coset_dft_algebra_batch(mat, shift);
    result.values
}

/// Recover polynomial coefficients from a natural-order codeword on coset `shift * <g>`.
///
/// The returned vector has length `codeword.len()`. Trailing zero coefficients are not
/// stripped: callers downstream (`add_polys`, `codeword_from_coeffs`) either handle
/// variable-length inputs or explicitly resize, and a content-dependent length here
/// would make the contract brittle against future refactors.
pub fn coeffs_from_codeword<F, EF, Dft>(dft: &Dft, codeword: &[EF], shift: F) -> Vec<EF>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + BasedVectorSpace<F>,
    Dft: TwoAdicSubgroupDft<F>,
{
    let mat = RowMajorMatrix::new_col(codeword.to_vec());
    let result = dft.coset_idft_algebra_batch(mat, shift);
    result.values
}

/// Commit a natural-order codeword of length `N` as a fiber-organised
/// `(N / 2^log_arity) × 2^log_arity` matrix.
fn commit_as_fiber_matrix<EF: Field, M: Mmcs<EF>>(
    mmcs: &M,
    codeword: &[EF],
    log_arity: usize,
) -> (M::Commitment, M::ProverData<RowMajorMatrix<EF>>) {
    let arity = 1 << log_arity;
    let new_height = codeword.len() / arity;
    let mut matrix = vec![EF::ZERO; codeword.len()];
    matrix
        .par_chunks_mut(arity)
        .enumerate()
        .for_each(|(j, row)| {
            for (k, slot) in row.iter_mut().enumerate() {
                *slot = codeword[j + k * new_height];
            }
        });
    mmcs.commit_matrix(RowMajorMatrix::new(matrix, arity))
}

/// Opens `indices` against the current commitment's fiber matrix as one shared, pruned
/// multi-opening proof.
fn open_fiber_rows<EF: Field, M: Mmcs<EF>>(
    mmcs: &M,
    indices: &[usize],
    prover_data: &M::ProverData<RowMajorMatrix<EF>>,
) -> StirQueryOpenings<EF, M> {
    let (values, opening_proof) = mmcs.open_multi_batch(indices, prover_data);
    let row_evals = values
        .into_iter()
        .map(|mut per_matrix| {
            assert_eq!(
                per_matrix.len(),
                1,
                "STIR commits exactly one codeword matrix"
            );
            per_matrix.swap_remove(0)
        })
        .collect();
    StirQueryOpenings {
        row_evals,
        opening_proof,
    }
}
