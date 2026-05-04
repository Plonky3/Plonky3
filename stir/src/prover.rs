//! STIR prover implementation (Construction 5.2).
//!
//! Codewords are stored in natural order: `codeword[j] = f(shift * g^j)` where
//! `g = two_adic_generator(log_domain_size)` and `shift` is the domain's coset shift.
//! Before committing, the codeword is arranged as a `(new_height × arity)` matrix where
//! row `j` contains the fiber, allowing a single MMCS opening to reveal the entire fiber.

use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{BasedVectorSpace, ExtensionField, Field, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use tracing::instrument;

use crate::config::StirConfig;
use crate::proof::{StirFinalQueryProof, StirProof, StirQueryProof, StirRoundProof};
use crate::utils::{
    add_polys, compute_shake_polynomial, degree_correct, eval_poly, fold_codeword,
    interpolate_poly, next_domain_shift, quotient_by_roots, scale_poly,
};

/// Prove that a polynomial (given in coefficient form over `EF`) has low degree,
/// using the STIR proximity testing protocol.
///
/// The initial codeword commitment is observed in the challenger internally; callers must
/// NOT pre-commit the initial codeword.
#[instrument(skip_all)]
pub fn prove_stir<F, EF, Dft, M, Challenger>(
    config: &StirConfig<F, EF, M, Challenger>,
    poly_coeffs: Vec<EF>,
    dft: &Dft,
    challenger: &mut Challenger,
) -> StirProof<EF, M, Challenger::Witness>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField + BasedVectorSpace<F>,
    Dft: TwoAdicSubgroupDft<F>,
    M: Mmcs<EF>,
    Challenger: FieldChallenger<F> + CanObserve<M::Commitment> + GrindingChallenger<Witness = F>,
{
    let num_rounds = config.num_rounds();

    // Commit to the initial codeword on L_0.
    let initial_shift = if num_rounds > 0 {
        config.round_configs[0].domain_shift
    } else {
        F::GENERATOR
    };

    let log_initial_domain = config.log_starting_domain_size();
    let initial_domain_size = 1 << log_initial_domain;

    let mut coeffs = poly_coeffs;
    coeffs.resize(initial_domain_size, EF::ZERO);
    let initial_codeword = codeword_from_coeffs(dft, coeffs, initial_shift, log_initial_domain);

    let mut current_oracle_codeword = initial_codeword.clone();
    let mut current_commit_codeword = initial_codeword;
    let mut current_shift = initial_shift;
    let mut current_log_domain = log_initial_domain;

    let log_arity0 = if num_rounds > 0 {
        config.round_configs[0].log_folding_factor
    } else {
        config.log_folding_factor
    };
    let (initial_commit, initial_data) =
        commit_as_fiber_matrix(&config.mmcs, &current_commit_codeword, log_arity0);
    challenger.observe(initial_commit.clone());

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
        let fold_height = 1 << fold_log_domain;

        let fold_shift = current_shift.exp_power_of_2(log_arity);
        let next_log_domain = current_log_domain - 1;
        let next_shift = next_domain_shift(current_shift, log_arity);

        // Step 1: fold. Derive gamma after folding PoW.
        let folding_pow_witness = challenger.grind(rc.folding_pow_bits);
        let gamma: EF = challenger.sample_algebra_element();

        let folded_codeword = fold_codeword::<F, EF>(
            &current_oracle_codeword,
            gamma,
            log_arity,
            current_log_domain,
            current_shift,
        );
        let fold_coeffs = coeffs_from_codeword(dft, &folded_codeword, fold_shift);

        let next_log_arity = if round + 1 < num_rounds {
            config.round_configs[round + 1].log_folding_factor
        } else {
            config.log_folding_factor
        };
        let next_commit_codeword =
            codeword_from_coeffs(dft, fold_coeffs.clone(), next_shift, next_log_domain);
        let (new_commit, new_data) =
            commit_as_fiber_matrix(&config.mmcs, &next_commit_codeword, next_log_arity);
        challenger.observe(new_commit.clone());

        // Step 2: OOD sampling.
        // OOD points must be outside both the current witness domain and the next witness domain.
        let current_domain_size = 1usize << current_log_domain;
        let next_domain_size = 1usize << next_log_domain;
        let mut ood_points = Vec::with_capacity(rc.num_ood_samples);
        while ood_points.len() < rc.num_ood_samples {
            let z: EF = challenger.sample_algebra_element();
            let z_norm_cur = z * EF::from(current_shift).inverse();
            let outside_current = z_norm_cur.exp_power_of_2(current_log_domain) != EF::ONE
                || current_domain_size == 1;
            let z_norm_next = z * EF::from(next_shift).inverse();
            let outside_next =
                z_norm_next.exp_power_of_2(next_log_domain) != EF::ONE || next_domain_size == 1;
            // Deduplicate OOD points.
            let not_dup = ood_points.iter().all(|&existing| existing != z);
            if outside_current && outside_next && not_dup {
                ood_points.push(z);
            }
        }

        let ood_answers: Vec<EF> = ood_points
            .iter()
            .map(|&z| eval_poly(&fold_coeffs, z))
            .collect();
        challenger.observe_algebra_slice(&ood_answers);

        // Step 3: Query phase PoW and query sampling.
        let pow_witness = challenger.grind(rc.pow_bits);

        let fold_gen = F::two_adic_generator(fold_log_domain);

        let mut query_proofs = Vec::with_capacity(rc.num_queries);
        let mut query_points = Vec::with_capacity(rc.num_queries);
        let mut query_answers = Vec::with_capacity(rc.num_queries);

        let mut seen_query_indices: alloc::collections::BTreeSet<usize> =
            alloc::collections::BTreeSet::new();

        let r_comb: EF = challenger.sample_algebra_element();

        let current_opening_data = &current_commit_data;

        for _ in 0..rc.num_queries {
            let j = challenger.sample_bits(fold_log_domain);
            let fold_point = EF::from(fold_shift) * EF::from(fold_gen.exp_u64(j as u64));

            let opening = config.mmcs.open_batch(j, current_opening_data);
            let row_evals = (0..arity)
                .map(|k| current_commit_codeword[j + k * fold_height])
                .collect();

            query_proofs.push(StirQueryProof {
                row_evals,
                opening_proof: opening.opening_proof,
            });

            if seen_query_indices.insert(j) {
                query_points.push(fold_point);
                query_answers.push(folded_codeword[j]);
            }
        }

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

        // Step 5: Construction 5.2 — compute the next virtual witness polynomial
        // f_{i+1} = DegCor((g_i − Ans_i) / Z_{G_i}).
        let num_answers = all_points.len();
        let numerator = add_polys(&fold_coeffs, &scale_poly(&ans_poly, EF::ZERO - EF::ONE));
        let quotient = quotient_by_roots(&numerator, &all_points);
        let f_next_coeffs = degree_correct(&quotient, r_comb, num_answers);
        let next_oracle_codeword =
            codeword_from_coeffs(dft, f_next_coeffs, next_shift, next_log_domain);

        round_proofs.push(StirRoundProof {
            commitment: new_commit,
            folding_pow_witness,
            ood_answers,
            pow_witness,
            ans_polynomial: ans_poly,
            shake_polynomial: shake_poly,
            query_proofs,
        });

        current_oracle_codeword = next_oracle_codeword;
        current_commit_codeword = next_commit_codeword;
        current_commit_data = new_data;
        current_shift = next_shift;
        current_log_domain = next_log_domain;
    }

    // Final round: fold the last committed codeword and send the resulting polynomial.
    let final_log_arity = config.log_folding_factor;
    let final_arity = 1usize << final_log_arity;
    let final_new_log_domain = current_log_domain - final_log_arity;
    let final_new_height = 1usize << final_new_log_domain;
    let final_new_shift = current_shift.exp_power_of_2(final_log_arity);

    let final_folding_pow_witness = challenger.grind(config.final_folding_pow_bits);
    let final_gamma: EF = challenger.sample_algebra_element();

    let final_codeword = fold_codeword::<F, EF>(
        &current_oracle_codeword,
        final_gamma,
        final_log_arity,
        current_log_domain,
        current_shift,
    );
    let final_new_coeffs = coeffs_from_codeword(dft, &final_codeword, final_new_shift);
    let final_len = config.final_poly_len();
    let mut final_poly = final_new_coeffs;
    final_poly.resize(final_len, EF::ZERO);

    challenger.observe_algebra_slice(&final_poly);
    let final_pow_witness = challenger.grind(config.final_pow_bits);

    let mut final_query_proofs = Vec::with_capacity(config.final_queries);
    let mut final_seen: alloc::collections::BTreeSet<usize> = alloc::collections::BTreeSet::new();
    for _ in 0..config.final_queries {
        let j = challenger.sample_bits(final_new_log_domain);
        final_seen.insert(j);
        let opening = config.mmcs.open_batch(j, &current_commit_data);
        let row_evals = (0..final_arity)
            .map(|k| current_commit_codeword[j + k * final_new_height])
            .collect();
        final_query_proofs.push(StirFinalQueryProof {
            row_evals,
            opening_proof: opening.opening_proof,
        });
    }

    // When there are no intermediate rounds the final queries target the
    // initial codeword.  Expose them for PCS input binding.
    if num_rounds == 0 {
        first_round_query_indices = final_seen.into_iter().collect();
    }

    StirProof {
        initial_commitment: initial_commit,
        round_proofs,
        final_polynomial: final_poly,
        final_folding_pow_witness,
        final_pow_witness,
        final_query_proofs,
        first_round_query_indices,
    }
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
pub fn coeffs_from_codeword<F, EF, Dft>(dft: &Dft, codeword: &[EF], shift: F) -> Vec<EF>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + BasedVectorSpace<F>,
    Dft: TwoAdicSubgroupDft<F>,
{
    let mat = RowMajorMatrix::new_col(codeword.to_vec());
    let result = dft.coset_idft_algebra_batch(mat, shift);
    let mut coeffs = result.values;
    while coeffs.last() == Some(&EF::ZERO) && coeffs.len() > 1 {
        coeffs.pop();
    }
    coeffs
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
    for j in 0..new_height {
        for k in 0..arity {
            matrix[j * arity + k] = codeword[j + k * new_height];
        }
    }
    mmcs.commit_matrix(RowMajorMatrix::new(matrix, arity))
}
