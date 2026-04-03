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
    add_polys, check_shake_consistency, eval_degree_correction, eval_poly, fold_fiber,
    fold_index_to_next_natural_index, interpolate_poly, quotient_by_roots, scale_poly,
};

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

    /// A Merkle opening into the next-round commitment failed.
    #[error("Invalid next-round MMCS opening proof in round {round}, query {query}")]
    InvalidNextMmcsProof {
        round: usize,
        query: usize,
        #[source]
        source: MmcsError,
    },

    /// The shake polynomial identity failed at the random evaluation point.
    #[error("Shake polynomial consistency check failed in round {round}")]
    InvalidShakeConsistency { round: usize },

    /// A claimed OOD answer is inconsistent with the folded polynomial.
    #[error("Invalid OOD answer in round {round}, sample {sample}")]
    InvalidOodAnswer { round: usize, sample: usize },

    /// The folded polynomial is inconsistent with the opened current-round fiber.
    #[error("Invalid fold consistency in round {round}, query {query}")]
    InvalidFoldConsistency { round: usize, query: usize },

    /// The committed next-round oracle is inconsistent with the current round's quotient state.
    #[error("Invalid round transition in round {round}, query {query}")]
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
            Self::InvalidNextMmcsProof {
                round,
                query,
                source,
            } => StirError::InvalidNextMmcsProof {
                round,
                query,
                source,
            },
            Self::InvalidShakeConsistency { round } => StirError::InvalidShakeConsistency { round },
            Self::InvalidOodAnswer { round, sample } => {
                StirError::InvalidOodAnswer { round, sample }
            }
            Self::InvalidFoldConsistency { round, query } => {
                StirError::InvalidFoldConsistency { round, query }
            }
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

    let mut current_shift = if num_rounds > 0 {
        config.round_configs[0].domain_shift
    } else {
        F::GENERATOR
    };
    let mut current_log_domain = config.log_starting_domain_size();

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

        let lde_log_domain = current_log_domain - 1;
        let fold_shift = current_shift.exp_power_of_2(log_arity);
        let lde_shift = fold_shift;

        // Step 1 & 2: folding PoW and folding challenge gamma.
        if !challenger.check_witness(rc.folding_pow_bits, rp.folding_pow_witness) {
            return Err(StirError::InvalidPowWitness { round });
        }

        let gamma: EF = challenger.sample_algebra_element();
        let max_fold_len = 1usize << (rc.log_degree - rc.log_folding_factor);
        if rp.fold_polynomial.is_empty() || rp.fold_polynomial.len() > max_fold_len {
            return Err(StirError::InvalidProofShape);
        }
        challenger.observe_algebra_slice(&rp.fold_polynomial);

        // Step 3 & 4: OOD sampling and answer observation.
        if rp.ood_answers.len() != rc.num_ood_samples {
            return Err(StirError::InvalidProofShape);
        }

        let current_domain_size = 1usize << current_log_domain;
        let lde_size = 1usize << lde_log_domain;
        let mut ood_points: Vec<EF> = Vec::with_capacity(rc.num_ood_samples);
        while ood_points.len() < rc.num_ood_samples {
            let z: EF = challenger.sample_algebra_element();
            let z_norm_cur = z * EF::from(current_shift).inverse();
            let outside_current = z_norm_cur.exp_power_of_2(current_log_domain) != EF::ONE
                || current_domain_size == 1;
            let z_norm_lde = z * EF::from(lde_shift).inverse();
            let outside_lde = z_norm_lde.exp_power_of_2(lde_log_domain) != EF::ONE || lde_size == 1;
            let not_dup = ood_points.iter().all(|&existing| existing != z);
            if outside_current && outside_lde && not_dup {
                ood_points.push(z);
            }
        }

        for (sample, (&z, &answer)) in ood_points.iter().zip(rp.ood_answers.iter()).enumerate() {
            if eval_poly(&rp.fold_polynomial, z) != answer {
                return Err(StirError::InvalidOodAnswer { round, sample });
            }
        }

        challenger.observe_algebra_slice(&rp.ood_answers);

        // Step 5 & 6: query PoW, query sampling, and fiber verification.
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

        let mut query_indices: Vec<usize> = Vec::with_capacity(rc.num_queries);
        let mut query_points: Vec<EF> = Vec::with_capacity(rc.num_queries);
        let mut query_answers: Vec<EF> = Vec::with_capacity(rc.num_queries);

        let mut seen_query_indices: alloc::collections::BTreeSet<usize> =
            alloc::collections::BTreeSet::new();

        let r_comb: EF = challenger.sample_algebra_element();

        for (q, qp) in rp.query_proofs.iter().enumerate() {
            let j = challenger.sample_bits(fold_log_domain);
            query_indices.push(j);

            let fold_point = EF::from(fold_shift) * EF::from(fold_gen.exp_u64(j as u64));

            if qp.fiber_evals.len() != arity {
                return Err(StirError::InvalidProofShape);
            }

            // Verify the Merkle opening of f_i at row j.
            let opened_values: alloc::vec::Vec<alloc::vec::Vec<EF>> =
                alloc::vec![qp.fiber_evals.clone()];
            let batch_opening = BatchOpeningRef::new(&opened_values, &qp.opening_proof);
            config
                .mmcs
                .verify_batch(cur_commit, &cur_dimensions, j, batch_opening)
                .map_err(|source| StirError::InvalidMmcsProof {
                    round,
                    query: q,
                    source,
                })?;

            // Compute g_i(fold_point) by Lagrange interpolation of the fiber.
            let fold_val =
                fold_fiber::<F, EF>(&qp.fiber_evals, j, fold_log_domain, log_arity, gamma);

            if fold_val != eval_poly(&rp.fold_polynomial, fold_point) {
                return Err(StirError::InvalidFoldConsistency { round, query: q });
            }

            if seen_query_indices.insert(j) {
                query_points.push(fold_point);
                query_answers.push(fold_val);
            }
        }

        // Step 7 & 8: shake polynomial observation and consistency check.
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

        challenger.observe_algebra_slice(&rp.shake_polynomial);

        let rho: EF = challenger.sample_algebra_element();

        let ans_poly = interpolate_poly(&all_points, &all_values);

        if !check_shake_consistency(
            &ans_poly,
            &rp.shake_polynomial,
            &all_points,
            &all_values,
            rho,
        ) {
            return Err(StirError::InvalidShakeConsistency { round });
        }

        // Step 9: observe the next-round commitment (f_{i+1}).
        // The commitment is placed at the END of the round so that the
        // next-round gamma is derived after f_{i+1} is fixed.
        let num_answers = all_points.len();
        let numerator = add_polys(
            &rp.fold_polynomial,
            &scale_poly(&ans_poly, EF::ZERO - EF::ONE),
        );
        let quotient = quotient_by_roots(&numerator, &all_points);

        challenger.observe(rp.commitment.clone());

        if rp.next_query_proofs.len() != rc.num_queries {
            return Err(StirError::InvalidProofShape);
        }

        let next_log_arity = if round + 1 < num_rounds {
            config.round_configs[round + 1].log_folding_factor
        } else {
            config.log_folding_factor
        };
        let next_arity = 1usize << next_log_arity;
        let next_height = 1usize << (lde_log_domain - next_log_arity);
        let next_dimensions = alloc::vec![Dimensions {
            height: next_height,
            width: next_arity,
        }];

        for (q, (&j, next_qp)) in query_indices
            .iter()
            .zip(rp.next_query_proofs.iter())
            .enumerate()
        {
            if next_qp.row_evals.len() != next_arity {
                return Err(StirError::InvalidProofShape);
            }

            let natural_index = fold_index_to_next_natural_index(j, log_arity);
            let row = natural_index % next_height;
            let col = natural_index / next_height;

            let opened_values: alloc::vec::Vec<alloc::vec::Vec<EF>> =
                alloc::vec![next_qp.row_evals.clone()];
            let batch_opening = BatchOpeningRef::new(&opened_values, &next_qp.opening_proof);
            config
                .mmcs
                .verify_batch(&rp.commitment, &next_dimensions, row, batch_opening)
                .map_err(|source| StirError::InvalidNextMmcsProof {
                    round,
                    query: q,
                    source,
                })?;

            let x = EF::from(fold_shift) * EF::from(fold_gen.exp_u64(j as u64));
            let expected = eval_degree_correction(eval_poly(&quotient, x), x, r_comb, num_answers);
            if next_qp.row_evals[col] != expected {
                return Err(StirError::InvalidRoundConsistency { round, query: q });
            }
        }

        current_shift = lde_shift;
        current_log_domain = lde_log_domain;
    }

    // Final round: verify the final fold against the last committed codeword.
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

        if fqp.fiber_evals.len() != final_arity {
            return Err(StirError::InvalidProofShape);
        }

        let opened_values: alloc::vec::Vec<alloc::vec::Vec<EF>> =
            alloc::vec![fqp.fiber_evals.clone()];
        let batch_opening = BatchOpeningRef::new(&opened_values, &fqp.opening_proof);
        config
            .mmcs
            .verify_batch(last_commit, &final_dimensions, j, batch_opening)
            .map_err(|source| StirError::InvalidMmcsProof {
                round: num_rounds,
                query: q,
                source,
            })?;

        let fold_val = fold_fiber::<F, EF>(
            &fqp.fiber_evals,
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
