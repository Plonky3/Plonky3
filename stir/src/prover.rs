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
    compute_shake_polynomial, eval_poly_parallel, fold_codeword, interpolate_poly,
    next_domain_shift, vanishing_poly_from_roots,
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
        let rc = &config.round_configs[round];
        let log_arity = rc.log_folding_factor;
        let arity = 1 << log_arity;

        let fold_log_domain = current_log_domain - log_arity;

        let fold_shift = current_shift.exp_power_of_2(log_arity);
        let next_log_domain = current_log_domain - 1;
        let next_shift = next_domain_shift(current_shift, log_arity);

        // Step 1: fold. Derive gamma after folding PoW.
        let folding_pow_witness = challenger.grind(rc.folding_pow_bits);
        let gamma: EF = challenger.sample_algebra_element();

        // `fold_codeword` interpolates at subgroup coordinates `g^{·}`. The codeword
        // lives on the coset `current_shift · <g>`, so passing `gamma / current_shift`
        // yields exactly Construction 4.5's coset fold at challenge `gamma`.
        let fold_beta = gamma * EF::from(current_shift.inverse());
        let folded_codeword = fold_codeword::<F, EF>(
            &current_oracle_codeword,
            fold_beta,
            log_arity,
            current_log_domain,
        );
        let fold_coeffs = coeffs_from_codeword(dft, &folded_codeword, fold_shift);

        let next_commit_codeword =
            codeword_from_coeffs(dft, fold_coeffs.clone(), next_shift, next_log_domain);
        let (new_commit, new_data) = commit_as_fiber_matrix(
            &config.mmcs,
            &next_commit_codeword,
            config.log_folding_factor,
        );
        challenger.observe(new_commit.clone());

        // Step 2: OOD sampling.
        // OOD points must be outside the current and next witness domains AND outside the
        // fold-query domain. Excluding the fold domain prevents an honest-prover failure
        // where an OOD point coincides with a sampled query point and the interpolation in
        // step 4 hits duplicate roots.
        let current_domain_size = 1usize << current_log_domain;
        let next_domain_size = 1usize << next_log_domain;
        let fold_domain_size = 1usize << fold_log_domain;
        let mut ood_points = Vec::with_capacity(rc.num_ood_samples);
        while ood_points.len() < rc.num_ood_samples {
            let z: EF = challenger.sample_algebra_element();
            let z_norm_cur = z * EF::from(current_shift).inverse();
            let outside_current = z_norm_cur.exp_power_of_2(current_log_domain) != EF::ONE
                || current_domain_size == 1;
            let z_norm_next = z * EF::from(next_shift).inverse();
            let outside_next =
                z_norm_next.exp_power_of_2(next_log_domain) != EF::ONE || next_domain_size == 1;
            let z_norm_fold = z * EF::from(fold_shift).inverse();
            let outside_fold =
                z_norm_fold.exp_power_of_2(fold_log_domain) != EF::ONE || fold_domain_size == 1;
            // Deduplicate OOD points.
            let not_dup = ood_points.iter().all(|&existing| existing != z);
            if outside_current && outside_next && outside_fold && not_dup {
                ood_points.push(z);
            }
        }

        // `fold_coeffs` is padded to the next round's full domain size, but the folded
        // polynomial's true degree is bounded by the round's degree schedule (a fixed
        // factor smaller); evaluating only the non-trivially-zero prefix cuts Horner's
        // work by that same factor.
        let folded_degree_bound = 1usize << (rc.log_degree - log_arity);
        let truncated_fold_coeffs = &fold_coeffs[..folded_degree_bound.min(fold_coeffs.len())];

        let ood_answers: Vec<EF> = ood_points
            .iter()
            .map(|&z| eval_poly_parallel(truncated_fold_coeffs, z))
            .collect();
        challenger.observe_algebra_slice(&ood_answers);

        // Step 3: query-phase PoW. It protects the immediately following combination challenge
        // and query indices. It does not strengthen the earlier OOD samples or the later shake
        // challenge, which is separated from this grind by prover-controlled messages.
        let pow_witness = challenger.grind(rc.pow_bits);

        // Step 4: Query sampling.
        let fold_gen = F::two_adic_generator(fold_log_domain);

        let mut query_indices = Vec::with_capacity(rc.num_queries);
        let mut query_points = Vec::with_capacity(rc.num_queries);
        let mut query_answers = Vec::with_capacity(rc.num_queries);

        let mut seen_query_indices: alloc::collections::BTreeSet<usize> =
            alloc::collections::BTreeSet::new();

        let r_comb: EF = challenger.sample_algebra_element();

        for _ in 0..rc.num_queries {
            // `RESAMPLE = true`: the challenger loops on field-side rejection internally,
            // so this `expect` is unreachable for every challenger in this workspace.
            // Unbiased sampling is required because `sample_bits` carries a per-draw modular
            // bias of `2^fold_log_domain / |F|`, which is non-negligible over 31-bit fields.
            let j = challenger
                .sample_uniform_bits::<true>(fold_log_domain)
                .expect("RESAMPLE = true: rejection loops internally, never errors");
            let fold_point = EF::from(fold_shift) * EF::from(fold_gen.exp_u64(j as u64));

            query_indices.push(j);

            if seen_query_indices.insert(j) {
                query_points.push(fold_point);
                query_answers.push(folded_codeword[j]);
            }
        }

        // One shared, pruned multi-opening proof for every query drawn this round. Absent when
        // this round reads an external initial oracle, whose fibers the verifier rebuilds.
        let query_openings = current_commit_data
            .as_ref()
            .map(|data| open_fiber_rows(&config.mmcs, &query_indices, data));
        debug_assert!(
            query_openings
                .iter()
                .all(|o| o.row_evals.iter().all(|row| row.len() == arity))
        );

        // Collect first-round query indices for the PCS binding check.
        if round == 0 {
            first_round_query_indices = seen_query_indices.iter().copied().collect();
        }

        // Step 4: Answer polynomial, shake polynomial, and shake-check challenge.
        let all_points: Vec<EF> = ood_points
            .iter()
            .chain(query_points.iter())
            .copied()
            .collect();
        let all_values: Vec<EF> = ood_answers
            .iter()
            .chain(query_answers.iter())
            .copied()
            .collect();

        let ans_poly = interpolate_poly(&all_points, &all_values);
        let shake_poly = compute_shake_polynomial(&ans_poly, &all_points);
        // Bind ans_poly into the transcript BEFORE rho is sampled — otherwise a malicious prover
        // could fit Ans to satisfy the shake identity at a known rho.
        challenger.observe_algebra_slice(&ans_poly);
        challenger.observe_algebra_slice(&shake_poly);

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
        let num_answers = all_points.len();
        let next_domain_size = 1usize << next_log_domain;

        let vanishing_coeffs = vanishing_poly_from_roots(&all_points);
        // Ans interpolates `num_answers` points and the vanishing polynomial has exactly
        // `num_answers + 1` coefficients.
        let log_answer_len = log2_ceil_usize(num_answers + 1).min(next_log_domain);
        let (ans_evals, vanishing_evals) = eval_low_degree_pair_on_coset(
            dft,
            &ans_poly,
            &vanishing_coeffs,
            next_shift,
            next_log_domain,
            log_answer_len,
        );

        // x_j = next_shift * g^j; step_j = r_comb * x_j = step_start * g^j.
        let g_next = F::two_adic_generator(next_log_domain);
        let g_powers: Vec<F> = g_next.powers().collect_n(next_domain_size);
        let g_powers_hi: Vec<F> = g_next
            .exp_u64((num_answers + 1) as u64)
            .powers()
            .collect_n(next_domain_size);
        let step_start = r_comb * EF::from(next_shift);
        let step_start_hi = step_start.exp_u64((num_answers + 1) as u64);

        let combined_denoms: Vec<EF> = (0..next_domain_size)
            .into_par_iter()
            .map(|j| vanishing_evals[j] * (EF::ONE - step_start * EF::from(g_powers[j])))
            .collect();
        let combined_inverses = batch_multiplicative_inverse(&combined_denoms);

        let next_oracle_codeword: Vec<EF> = (0..next_domain_size)
            .into_par_iter()
            .map(|j| {
                let degree_correction_numerator =
                    EF::ONE - step_start_hi * EF::from(g_powers_hi[j]);
                (next_commit_codeword[j] - ans_evals[j])
                    * combined_inverses[j]
                    * degree_correction_numerator
            })
            .collect();

        round_proofs.push(StirRoundProof {
            commitment: new_commit,
            folding_pow_witness,
            ood_answers,
            pow_witness,
            ans_polynomial: ans_poly,
            shake_polynomial: shake_poly,
            query_openings,
        });

        current_oracle_codeword = next_oracle_codeword;
        current_commit_data = Some(new_data);
        current_shift = next_shift;
        current_log_domain = next_log_domain;
    }

    // Final round: fold the last committed codeword and send the resulting polynomial.
    let final_log_arity = config.log_folding_factor;
    let final_arity = 1usize << final_log_arity;
    let final_new_log_domain = current_log_domain - final_log_arity;
    let final_new_shift = current_shift.exp_power_of_2(final_log_arity);

    let final_folding_pow_witness = challenger.grind(config.final_folding_pow_bits);
    let final_gamma: EF = challenger.sample_algebra_element();

    // See the round-fold note: `gamma / current_shift` over subgroup coordinates is
    // the paper's coset fold at challenge `gamma`.
    let final_fold_beta = final_gamma * EF::from(current_shift.inverse());
    let final_codeword = fold_codeword::<F, EF>(
        &current_oracle_codeword,
        final_fold_beta,
        final_log_arity,
        current_log_domain,
    );
    // The final polynomial has only `final_len` coefficients, far fewer than
    // `final_codeword`'s full domain size. Rather than run a full-size iDFT and discard
    // the (necessarily zero) high coefficients, gather a `final_len`-sized coset — every
    // `stride`-th natural-order point, which is exactly the subgroup coset of that size —
    // and run the small iDFT directly on it.
    let final_len = config.final_poly_len();
    let stride = final_codeword.len() / final_len;
    let final_poly_evals: Vec<EF> = (0..final_len).map(|i| final_codeword[i * stride]).collect();
    let final_poly = coeffs_from_codeword(dft, &final_poly_evals, final_new_shift);

    challenger.observe_algebra_slice(&final_poly);
    let final_pow_witness = challenger.grind(config.final_pow_bits);

    let mut final_query_indices = Vec::with_capacity(config.final_queries);
    let mut final_seen: alloc::collections::BTreeSet<usize> = alloc::collections::BTreeSet::new();
    for _ in 0..config.final_queries {
        let j = challenger
            .sample_uniform_bits::<true>(final_new_log_domain)
            .expect("RESAMPLE = true: rejection loops internally, never errors");
        final_seen.insert(j);
        final_query_indices.push(j);
    }

    let final_query_openings = current_commit_data
        .as_ref()
        .map(|data| open_fiber_rows(&config.mmcs, &final_query_indices, data));
    debug_assert!(
        final_query_openings
            .iter()
            .all(|o| o.row_evals.iter().all(|row| row.len() == final_arity))
    );

    // When there are no intermediate rounds the final queries target the
    // initial codeword.  Expose them for PCS input binding.
    if num_rounds == 0 {
        first_round_query_indices = final_seen.into_iter().collect();
    }

    let proof = StirProof {
        initial_commitment: initial_commit,
        round_proofs,
        final_polynomial: final_poly,
        final_folding_pow_witness,
        final_pow_witness,
        final_query_openings,
    };
    (proof, first_round_query_indices)
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
/// stripped: callers downstream (`add_polys`, `multiply_polys`, `codeword_from_coeffs`)
/// either handle variable-length inputs or explicitly resize, and a content-dependent
/// length here would make the contract brittle against future refactors.
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
