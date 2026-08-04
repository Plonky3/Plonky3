//! STIR verifier implementation (Construction 5.2).

use alloc::vec::Vec;

use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{BasedVectorSpace, ExtensionField, TwoAdicField, batch_multiplicative_inverse};
use p3_matrix::Dimensions;
use thiserror::Error;

use crate::config::StirConfig;
use crate::proof::StirProof;
use crate::utils::{
    check_shake_consistency, eval_degree_correction, eval_poly, eval_vanishing_at_roots,
    fold_fiber, next_domain_shift,
};

#[derive(Clone)]
struct VirtualRoundContext<EF> {
    ans_poly: Vec<EF>,
    all_points: Vec<EF>,
    r_comb: EF,
}

fn materialize_virtual_fiber<F, EF>(
    row_evals: &[EF],
    row_index: usize,
    row_height: usize,
    current_log_domain: usize,
    current_shift: F,
    prev_ctx: Option<&VirtualRoundContext<EF>>,
) -> Option<Vec<EF>>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField + BasedVectorSpace<F>,
{
    let Some(ctx) = prev_ctx else {
        return Some(row_evals.to_vec());
    };

    let domain_gen = F::two_adic_generator(current_log_domain);
    let points: Vec<EF> = (0..row_evals.len())
        .map(|col| {
            let natural_index = row_index + col * row_height;
            EF::from(current_shift) * EF::from(domain_gen.exp_u64(natural_index as u64))
        })
        .collect();
    let vanishing_values: Vec<EF> = points
        .iter()
        .map(|&x| eval_vanishing_at_roots(&ctx.all_points, x))
        .collect();
    if vanishing_values.contains(&EF::ZERO) {
        return None;
    }
    let vanishing_inverses = batch_multiplicative_inverse(&vanishing_values);

    Some(
        row_evals
            .iter()
            .zip(points)
            .zip(vanishing_inverses)
            .map(|((&g_value, x), vanishing_inverse)| {
                let quotient = (g_value - eval_poly(&ctx.ans_poly, x)) * vanishing_inverse;
                eval_degree_correction(quotient, x, ctx.r_comb, ctx.all_points.len())
            })
            .collect(),
    )
}

/// Wraps each opened row as a single-matrix batch for [`Mmcs::verify_multi_batch`].
///
/// `verify_multi_batch` takes `opened_values[query][matrix]` to support batches spanning
/// several committed matrices at once; STIR only ever commits one matrix per round, so every
/// inner `Vec` here always holds exactly one row slice.
fn single_matrix_opened_values<EF>(row_evals: &[Vec<EF>]) -> Vec<Vec<&[EF]>> {
    row_evals
        .iter()
        .map(|row| alloc::vec![row.as_slice()])
        .collect()
}

/// Errors returned by [`verify_stir`].
#[derive(Debug, Error, PartialEq, Eq)]
pub enum StirError<MmcsError, InputError = ()> {
    /// A proof-of-work witness failed verification.
    #[error("Invalid proof-of-work witness in round {round}")]
    InvalidPowWitness { round: usize },

    /// A Merkle multi-opening proof failed for a round's queries.
    #[error("Invalid MMCS opening proof in round {round}")]
    InvalidMmcsProof {
        round: usize,
        #[source]
        source: MmcsError,
    },

    /// The shake polynomial identity failed at the random evaluation point.
    #[error("Shake polynomial consistency check failed in round {round}")]
    InvalidShakeConsistency { round: usize },

    /// A virtual-oracle evaluation landed in the prior round's challenge set.
    #[error("Invalid virtual-oracle query in round {round}, query {query}")]
    InvalidRoundConsistency { round: usize, query: usize },

    /// The final polynomial does not evaluate consistently with the last committed codeword.
    #[error("Final polynomial evaluation mismatch")]
    FinalPolyMismatch,

    /// The proof has the wrong number of rounds, queries, or OOD answers.
    #[error("Invalid proof shape")]
    InvalidProofShape,

    /// The PCS layer's cross-commitment accumulated reduced opening did not match the STIR
    /// first-round fiber evaluations. Indicates the input commitments and the STIR oracle are
    /// not consistent (e.g. a claimed evaluation was tampered with).
    #[error("PCS reduced-opening binding mismatch at query {query}, column {column}")]
    PcsBindingMismatch { query: usize, column: usize },

    /// An error propagated from the input polynomial commitment scheme.
    #[error("Input error")]
    InputError(InputError),
}

impl<E1, IE1> StirError<E1, IE1> {
    /// Map the `InputError` variant to a different type.
    pub fn map_input_err<IE2>(self, f: impl FnOnce(IE1) -> IE2) -> StirError<E1, IE2> {
        match self {
            Self::InvalidPowWitness { round } => StirError::InvalidPowWitness { round },
            Self::InvalidMmcsProof { round, source } => {
                StirError::InvalidMmcsProof { round, source }
            }
            Self::InvalidShakeConsistency { round } => StirError::InvalidShakeConsistency { round },
            Self::InvalidRoundConsistency { round, query } => {
                StirError::InvalidRoundConsistency { round, query }
            }
            Self::FinalPolyMismatch => StirError::FinalPolyMismatch,
            Self::InvalidProofShape => StirError::InvalidProofShape,
            Self::PcsBindingMismatch { query, column } => {
                StirError::PcsBindingMismatch { query, column }
            }
            Self::InputError(e) => StirError::InputError(f(e)),
        }
    }
}

/// Side outputs of a successful STIR verification.
///
/// The PCS layer needs to bind the input commitment to the STIR proof at the same fold-domain
/// positions the verifier sampled in the first round (or the final round, when
/// `num_rounds == 0`). Rather than have the PCS replay the Fiat-Shamir transcript by hand —
/// fragile against any future change to `verify_stir`'s transcript order — `verify_stir`
/// returns this view directly. `first_round_fiber_evals[i]` corresponds to
/// `first_round_indices[i]`, and the indices are sorted in ascending order.
#[derive(Debug, Clone)]
pub struct StirVerifyOutputs<EF> {
    /// Sorted (ascending) unique fold-domain query indices from the first round (or the final
    /// round when `num_rounds == 0`). Length ≤ `num_queries` (or `final_queries`) due to
    /// deduplication.
    pub first_round_indices: Vec<usize>,
    /// Row evaluations for each unique query, aligned with `first_round_indices`.
    pub first_round_fiber_evals: Vec<Vec<EF>>,
}

/// Verify a STIR proof (Construction 5.2).
pub fn verify_stir<F, EF, M, Challenger>(
    config: &StirConfig<F, EF, M, Challenger>,
    proof: &StirProof<EF, M, Challenger::Witness>,
    challenger: &mut Challenger,
) -> Result<StirVerifyOutputs<EF>, StirError<M::Error>>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField + BasedVectorSpace<F>,
    M: Mmcs<EF>,
    Challenger: FieldChallenger<F>
        + CanObserve<M::Commitment>
        + GrindingChallenger<Witness = F>
        + CanSampleUniformBits<F>,
{
    let num_rounds = config.num_rounds();

    if proof.round_proofs.len() != num_rounds {
        return Err(StirError::InvalidProofShape);
    }

    challenger.observe(proof.initial_commitment.clone());

    // Initial domain shift is always F::GENERATOR; round_configs[0].domain_shift mirrors it
    // when num_rounds > 0.
    let mut current_shift = F::GENERATOR;
    let mut current_log_domain = config.log_starting_domain_size();
    let mut prev_ctx: Option<VirtualRoundContext<EF>> = None;

    // Round-0 query view, recorded once and returned for the PCS input-binding step.
    // Pairs are inserted in challenger-sample (insertion) order; sorted by index at the end.
    let mut first_round_pairs: Vec<(usize, Vec<EF>)> = Vec::new();

    let round_commitment = |r: usize| -> &M::Commitment {
        if r == 0 {
            &proof.initial_commitment
        } else {
            &proof.round_proofs[r - 1].commitment
        }
    };

    for (round, rp) in proof.round_proofs.iter().enumerate() {
        let rc = &config.round_configs[round];
        let log_arity = rc.log_folding_factor;
        let arity = 1 << log_arity;

        let fold_log_domain = current_log_domain - log_arity;
        let fold_height = 1usize << fold_log_domain;

        let fold_shift = current_shift.exp_power_of_2(log_arity);
        let next_log_domain = current_log_domain - 1;
        let next_shift = next_domain_shift(current_shift, log_arity);

        // Step 1: folding PoW, folding challenge gamma, and folded-oracle commitment.
        if !challenger.check_witness(rc.folding_pow_bits, rp.folding_pow_witness) {
            return Err(StirError::InvalidPowWitness { round });
        }

        let gamma: EF = challenger.sample_algebra_element();
        challenger.observe(rp.commitment.clone());

        // Mirror the prover: fold at coset coordinates via `gamma / current_shift`
        // (`fold_fiber` interpolates at subgroup coordinates).
        let fold_beta = gamma * EF::from(current_shift.inverse());

        // Step 2: OOD sampling and answer observation.
        if rp.ood_answers.len() != rc.num_ood_samples {
            return Err(StirError::InvalidProofShape);
        }

        let current_domain_size = 1usize << current_log_domain;
        let next_domain_size = 1usize << next_log_domain;
        let fold_domain_size = 1usize << fold_log_domain;
        let mut ood_points: Vec<EF> = Vec::with_capacity(rc.num_ood_samples);
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
            let not_dup = ood_points.iter().all(|&existing| existing != z);
            if outside_current && outside_next && outside_fold && not_dup {
                ood_points.push(z);
            }
        }

        challenger.observe_algebra_slice(&rp.ood_answers);

        // Step 3: query-phase PoW. It protects only the immediately following combination
        // challenge and query indices; configuration soundness gives no PoW credit to the
        // earlier OOD samples or the later shake challenge.
        if !challenger.check_witness(rc.pow_bits, rp.pow_witness) {
            return Err(StirError::InvalidPowWitness { round });
        }

        // Step 4: combination challenge, query sampling, and fiber verification.

        if rp.query_openings.row_evals.len() != rc.num_queries {
            return Err(StirError::InvalidProofShape);
        }

        let fold_gen = F::two_adic_generator(fold_log_domain);
        let cur_commit = round_commitment(round);

        let cur_dimensions = alloc::vec![Dimensions {
            height: fold_height,
            width: arity
        }];

        let r_comb: EF = challenger.sample_algebra_element();

        // Step 4a: sample every query index first, in draw order, mirroring the prover's
        // unbiased-sampling policy so the Fiat-Shamir transcript stays in sync. Merkle
        // verification is deferred to a single shared multi-opening check below.
        let mut query_indices: Vec<usize> = Vec::with_capacity(rc.num_queries);
        for _ in 0..rc.num_queries {
            let j = challenger
                .sample_uniform_bits::<true>(fold_log_domain)
                .expect("RESAMPLE = true: rejection loops internally, never errors");
            query_indices.push(j);
        }

        if rp
            .query_openings
            .row_evals
            .iter()
            .any(|row| row.len() != arity)
        {
            return Err(StirError::InvalidProofShape);
        }

        // Step 4b: one shared, pruned Merkle multi-opening proof authenticates every query
        // row against the current commitment at once.
        let opened_values = single_matrix_opened_values(&rp.query_openings.row_evals);
        config
            .mmcs
            .verify_multi_batch(
                cur_commit,
                &cur_dimensions,
                &query_indices,
                &opened_values,
                &rp.query_openings.opening_proof,
            )
            .map_err(|source| StirError::InvalidMmcsProof { round, source })?;

        // Step 4c: per-query virtual-oracle materialization and folding.
        let mut query_points: Vec<EF> = Vec::with_capacity(rc.num_queries);
        let mut query_answers: Vec<EF> = Vec::with_capacity(rc.num_queries);

        let mut seen_query_indices: alloc::collections::BTreeSet<usize> =
            alloc::collections::BTreeSet::new();

        for (q, (&j, row_evals)) in query_indices
            .iter()
            .zip(&rp.query_openings.row_evals)
            .enumerate()
        {
            let fold_point = EF::from(fold_shift) * EF::from(fold_gen.exp_u64(j as u64));

            let current_fiber = materialize_virtual_fiber::<F, EF>(
                row_evals,
                j,
                fold_height,
                current_log_domain,
                current_shift,
                prev_ctx.as_ref(),
            )
            .ok_or(StirError::InvalidRoundConsistency { round, query: q })?;

            let fold_val =
                fold_fiber::<F, EF>(&current_fiber, j, fold_log_domain, log_arity, fold_beta);

            if seen_query_indices.insert(j) {
                query_points.push(fold_point);
                query_answers.push(fold_val);
                if round == 0 {
                    first_round_pairs.push((j, row_evals.clone()));
                }
            }
        }

        // Step 4: ans + shake polynomial observation and consistency check.
        let all_points: Vec<EF> = ood_points
            .iter()
            .chain(query_points.iter())
            .copied()
            .collect();
        let all_values: Vec<EF> = rp
            .ood_answers
            .iter()
            .chain(query_answers.iter())
            .copied()
            .collect();

        // Ans interpolates |all_points| values, so its degree is `< all_points.len()`. The
        // prover may have stripped trailing zeros, so accept any length up to that bound; reject
        // anything larger as malformed. Shake has degree `< all_points.len() - 1`.
        let max_ans_len = all_points.len();
        if rp.ans_polynomial.len() > max_ans_len
            || rp.shake_polynomial.len() > max_ans_len.saturating_sub(1)
        {
            return Err(StirError::InvalidProofShape);
        }

        // Bind ans_poly into the transcript BEFORE rho. The shake identity is a one-point check;
        // observing both polys first means the prover commits to Ans before learning rho.
        challenger.observe_algebra_slice(&rp.ans_polynomial);
        challenger.observe_algebra_slice(&rp.shake_polynomial);

        let rho: EF = challenger.sample_algebra_element();

        if !check_shake_consistency(
            &rp.ans_polynomial,
            &rp.shake_polynomial,
            &all_points,
            &all_values,
            rho,
        ) {
            return Err(StirError::InvalidShakeConsistency { round });
        }

        prev_ctx = Some(VirtualRoundContext {
            ans_poly: rp.ans_polynomial.clone(),
            all_points,
            r_comb,
        });

        current_shift = next_shift;
        current_log_domain = next_log_domain;
    }

    // Final round: verify the final fold against the last virtual oracle.
    let final_log_arity = config.log_folding_factor;
    let final_arity = 1usize << final_log_arity;
    let final_new_log_domain = current_log_domain - final_log_arity;
    let final_new_height = 1usize << final_new_log_domain;
    let final_new_shift = current_shift.exp_power_of_2(final_log_arity);

    if !challenger.check_witness(
        config.final_folding_pow_bits,
        proof.final_folding_pow_witness,
    ) {
        return Err(StirError::InvalidPowWitness { round: num_rounds });
    }

    let final_gamma: EF = challenger.sample_algebra_element();
    // See the round-fold note: coset fold at `final_gamma` via `final_gamma / current_shift`.
    let final_fold_beta = final_gamma * EF::from(current_shift.inverse());

    let expected_final_len = config.final_poly_len();
    if proof.final_polynomial.len() != expected_final_len {
        return Err(StirError::InvalidProofShape);
    }

    challenger.observe_algebra_slice(&proof.final_polynomial);

    if !challenger.check_witness(config.final_pow_bits, proof.final_pow_witness) {
        return Err(StirError::InvalidPowWitness { round: num_rounds });
    }

    if proof.final_query_openings.row_evals.len() != config.final_queries {
        return Err(StirError::InvalidProofShape);
    }

    let last_commit = if num_rounds > 0 {
        &proof.round_proofs[num_rounds - 1].commitment
    } else {
        &proof.initial_commitment
    };

    let final_dimensions = alloc::vec![Dimensions {
        height: final_new_height,
        width: final_arity,
    }];
    let final_gen = F::two_adic_generator(final_new_log_domain);

    // Sample every final-round query index first, deferring Merkle verification to a single
    // shared multi-opening check below.
    let mut final_indices: Vec<usize> = Vec::with_capacity(config.final_queries);
    for _ in 0..config.final_queries {
        let j = challenger
            .sample_uniform_bits::<true>(final_new_log_domain)
            .expect("RESAMPLE = true: rejection loops internally, never errors");
        final_indices.push(j);
    }

    if proof
        .final_query_openings
        .row_evals
        .iter()
        .any(|row| row.len() != final_arity)
    {
        return Err(StirError::InvalidProofShape);
    }

    let opened_values = single_matrix_opened_values(&proof.final_query_openings.row_evals);
    config
        .mmcs
        .verify_multi_batch(
            last_commit,
            &final_dimensions,
            &final_indices,
            &opened_values,
            &proof.final_query_openings.opening_proof,
        )
        .map_err(|source| StirError::InvalidMmcsProof {
            round: num_rounds,
            source,
        })?;

    // When num_rounds == 0 the final queries also serve as the PCS first-round binding.
    // Track them with the same dedup-on-first-occurrence rule as the intermediate-round path.
    let mut final_seen: alloc::collections::BTreeSet<usize> = alloc::collections::BTreeSet::new();

    for (q, (&j, row_evals)) in final_indices
        .iter()
        .zip(&proof.final_query_openings.row_evals)
        .enumerate()
    {
        let current_fiber = materialize_virtual_fiber::<F, EF>(
            row_evals,
            j,
            final_new_height,
            current_log_domain,
            current_shift,
            prev_ctx.as_ref(),
        )
        .ok_or(StirError::InvalidRoundConsistency {
            round: num_rounds,
            query: q,
        })?;

        let fold_val = fold_fiber::<F, EF>(
            &current_fiber,
            j,
            final_new_log_domain,
            final_log_arity,
            final_fold_beta,
        );

        let x_j = EF::from(final_new_shift) * EF::from(final_gen.exp_u64(j as u64));

        let expected = eval_poly(&proof.final_polynomial, x_j);
        if fold_val != expected {
            return Err(StirError::FinalPolyMismatch);
        }

        if num_rounds == 0 && final_seen.insert(j) {
            first_round_pairs.push((j, row_evals.clone()));
        }
    }

    // Sort by index (ascending) so the PCS layer's output ordering is deterministic and
    // matches the prover-side `first_round_query_indices` which is also sorted.
    first_round_pairs.sort_by_key(|(j, _)| *j);
    let (first_round_indices, first_round_fiber_evals): (Vec<_>, Vec<_>) =
        first_round_pairs.into_iter().unzip();

    Ok(StirVerifyOutputs {
        first_round_indices,
        first_round_fiber_evals,
    })
}
