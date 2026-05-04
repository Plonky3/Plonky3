//! STIR verifier implementation (Construction 5.2).

use alloc::vec::Vec;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{BatchOpeningRef, Mmcs};
use p3_field::{BasedVectorSpace, ExtensionField, TwoAdicField};
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
    row_evals
        .iter()
        .enumerate()
        .map(|(col, &g_value)| {
            let natural_index = row_index + col * row_height;
            let x = EF::from(current_shift) * EF::from(domain_gen.exp_u64(natural_index as u64));
            let vanishing = eval_vanishing_at_roots(&ctx.all_points, x);
            if vanishing == EF::ZERO {
                return None;
            }
            let quotient = (g_value - eval_poly(&ctx.ans_poly, x)) * vanishing.inverse();
            Some(eval_degree_correction(
                quotient,
                x,
                ctx.r_comb,
                ctx.all_points.len(),
            ))
        })
        .collect()
}

/// Errors returned by [`verify_stir`].
#[derive(Debug, Error, PartialEq, Eq)]
pub enum StirError<MmcsError, InputError = ()> {
    /// A proof-of-work witness failed verification.
    #[error("Invalid proof-of-work witness in round {round}")]
    InvalidPowWitness { round: usize },

    /// A Merkle opening proof failed.
    #[error("Invalid MMCS opening proof in round {round}, query {query}")]
    InvalidMmcsProof {
        round: usize,
        query: usize,
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

    /// An error propagated from the input polynomial commitment scheme.
    #[error("Input error")]
    InputError(InputError),
}

impl<E1, IE1> StirError<E1, IE1> {
    /// Map the `InputError` variant to a different type.
    pub fn map_input_err<IE2>(self, f: impl FnOnce(IE1) -> IE2) -> StirError<E1, IE2> {
        match self {
            Self::InvalidPowWitness { round } => StirError::InvalidPowWitness { round },
            Self::InvalidMmcsProof {
                round,
                query,
                source,
            } => StirError::InvalidMmcsProof {
                round,
                query,
                source,
            },
            Self::InvalidShakeConsistency { round } => StirError::InvalidShakeConsistency { round },
            Self::InvalidRoundConsistency { round, query } => {
                StirError::InvalidRoundConsistency { round, query }
            }
            Self::FinalPolyMismatch => StirError::FinalPolyMismatch,
            Self::InvalidProofShape => StirError::InvalidProofShape,
            Self::InputError(e) => StirError::InputError(f(e)),
        }
    }
}

/// Verify a STIR proof (Construction 5.2).
pub fn verify_stir<F, EF, M, Challenger>(
    config: &StirConfig<F, EF, M, Challenger>,
    proof: &StirProof<EF, M, Challenger::Witness>,
    challenger: &mut Challenger,
) -> Result<(), StirError<M::Error>>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField + BasedVectorSpace<F>,
    M: Mmcs<EF>,
    Challenger: FieldChallenger<F> + CanObserve<M::Commitment> + GrindingChallenger<Witness = F>,
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

        // Step 2: OOD sampling and answer observation.
        if rp.ood_answers.len() != rc.num_ood_samples {
            return Err(StirError::InvalidProofShape);
        }

        let current_domain_size = 1usize << current_log_domain;
        let next_domain_size = 1usize << next_log_domain;
        let mut ood_points: Vec<EF> = Vec::with_capacity(rc.num_ood_samples);
        while ood_points.len() < rc.num_ood_samples {
            let z: EF = challenger.sample_algebra_element();
            let z_norm_cur = z * EF::from(current_shift).inverse();
            let outside_current = z_norm_cur.exp_power_of_2(current_log_domain) != EF::ONE
                || current_domain_size == 1;
            let z_norm_next = z * EF::from(next_shift).inverse();
            let outside_next =
                z_norm_next.exp_power_of_2(next_log_domain) != EF::ONE || next_domain_size == 1;
            let not_dup = ood_points.iter().all(|&existing| existing != z);
            if outside_current && outside_next && not_dup {
                ood_points.push(z);
            }
        }

        challenger.observe_algebra_slice(&rp.ood_answers);

        // Step 3: query PoW, combination challenge, query sampling, and fiber verification.
        if !challenger.check_witness(rc.pow_bits, rp.pow_witness) {
            return Err(StirError::InvalidPowWitness { round });
        }

        if rp.query_proofs.len() != rc.num_queries {
            return Err(StirError::InvalidProofShape);
        }

        let fold_gen = F::two_adic_generator(fold_log_domain);
        let cur_commit = round_commitment(round);

        let cur_dimensions = alloc::vec![Dimensions {
            height: fold_height,
            width: arity
        }];

        let mut query_points: Vec<EF> = Vec::with_capacity(rc.num_queries);
        let mut query_answers: Vec<EF> = Vec::with_capacity(rc.num_queries);

        let mut seen_query_indices: alloc::collections::BTreeSet<usize> =
            alloc::collections::BTreeSet::new();

        let r_comb: EF = challenger.sample_algebra_element();

        for (q, qp) in rp.query_proofs.iter().enumerate() {
            let j = challenger.sample_bits(fold_log_domain);
            if qp.row_evals.len() != arity {
                return Err(StirError::InvalidProofShape);
            }

            let fold_point = EF::from(fold_shift) * EF::from(fold_gen.exp_u64(j as u64));

            // Verify the Merkle opening of the current commitment at row j.
            let opened_values: alloc::vec::Vec<alloc::vec::Vec<EF>> =
                alloc::vec![qp.row_evals.clone()];
            let batch_opening = BatchOpeningRef::new(&opened_values, &qp.opening_proof);
            config
                .mmcs
                .verify_batch(cur_commit, &cur_dimensions, j, batch_opening)
                .map_err(|source| StirError::InvalidMmcsProof {
                    round,
                    query: q,
                    source,
                })?;

            let current_fiber = materialize_virtual_fiber::<F, EF>(
                &qp.row_evals,
                j,
                fold_height,
                current_log_domain,
                current_shift,
                prev_ctx.as_ref(),
            )
            .ok_or(StirError::InvalidRoundConsistency { round, query: q })?;

            let fold_val =
                fold_fiber::<F, EF>(&current_fiber, j, fold_log_domain, log_arity, gamma);

            if seen_query_indices.insert(j) {
                query_points.push(fold_point);
                query_answers.push(fold_val);
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

    let expected_final_len = config.final_poly_len();
    if proof.final_polynomial.len() != expected_final_len {
        return Err(StirError::InvalidProofShape);
    }

    challenger.observe_algebra_slice(&proof.final_polynomial);

    if !challenger.check_witness(config.final_pow_bits, proof.final_pow_witness) {
        return Err(StirError::InvalidPowWitness { round: num_rounds });
    }

    if proof.final_query_proofs.len() != config.final_queries {
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

    for (q, fqp) in proof.final_query_proofs.iter().enumerate() {
        let j = challenger.sample_bits(final_new_log_domain);

        if fqp.row_evals.len() != final_arity {
            return Err(StirError::InvalidProofShape);
        }

        let opened_values: alloc::vec::Vec<alloc::vec::Vec<EF>> =
            alloc::vec![fqp.row_evals.clone()];
        let batch_opening = BatchOpeningRef::new(&opened_values, &fqp.opening_proof);
        config
            .mmcs
            .verify_batch(last_commit, &final_dimensions, j, batch_opening)
            .map_err(|source| StirError::InvalidMmcsProof {
                round: num_rounds,
                query: q,
                source,
            })?;

        let current_fiber = materialize_virtual_fiber::<F, EF>(
            &fqp.row_evals,
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
            final_gamma,
        );

        let x_j = EF::from(final_new_shift) * EF::from(final_gen.exp_u64(j as u64));

        let expected = eval_poly(&proof.final_polynomial, x_j);
        if fold_val != expected {
            return Err(StirError::FinalPolyMismatch);
        }
    }

    Ok(())
}
