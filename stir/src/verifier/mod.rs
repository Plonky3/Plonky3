use alloc::vec;
use alloc::vec::Vec;

use error::{FullRoundVerificationError, VerificationError};
use itertools::{iterate, Itertools};
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{BatchOpeningRef, Mmcs};
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{batch_multiplicative_inverse, eval_poly, ExtensionField, TwoAdicField};
use p3_matrix::Dimensions;

use crate::config::{observe_public_parameters, RoundConfig};
use crate::proof::RoundProof;
use crate::utils::{fold_evaluations_at_small_domain, observe_ext_slice_with_size};
use crate::{Messages, StirConfig, StirProof, POW_BITS_WARNING};

mod error;

#[cfg(test)]
mod tests;

// The virtual function
//   DegCor(Quot(f, interpolating_polynomial), quotient_set)
// in the notation of the paper, where `f` is the underlying function. In the
// case of interest to STIR,
// - f is g_i
// - interpolating_polynomial is Ans_i
// - quotient_set is \mathcal{G}_i
struct VirtualFunction<F: TwoAdicField> {
    // The degree-correction randomness r^comb_i
    comb_randomness: F,
    // The coefficients of the Ans polynomial interpolating the purported values
    // of f at certain queried points
    interpolating_polynomial: Vec<F>,
    // The quotient set \mathcal{G}_i containing the queried points
    quotient_set: Vec<F>,
}

// Oracle allowing the verifier to compute values of f_i, either directly for
// the initial codeword using a virtual function
enum Oracle<F: TwoAdicField> {
    // Transparent oracle: it takes the same value as the underlying function
    Transparent,
    // Virtual oracle: it takes the value of a virtual function over some
    // underlying function
    Virtual(VirtualFunction<F>),
}

impl<F: TwoAdicField> Oracle<F> {
    // Compute the value of the oracle at the point x given the value f(x) of
    // the underlying function.
    //
    // Panics if the point x is one of the points in the quosient set defining
    // the oracle.
    fn evaluate(
        &self,
        // The point at which the oracle is evaluated
        x: F,
        // The value of the underlying function f at the point x
        f_x: F,
        // Since each call to evaluate involves dividing by a denominator, we
        // can batch-invert many of them outside and feed them to the function
        quotient_denom_inverse_hint: Option<F>,
        // A similar phenomenon occurs for the degree-correction factor
        deg_cor_hint: Option<(F, F)>,
    ) -> F {
        match self {
            // In this case, the oracle contains the values of f_0 = g_0
            Oracle::Transparent => f_x,

            // In this case, we need to evaluate the quotient and
            // degree-correction factor
            Oracle::Virtual(virtual_function) => {
                assert!(
                    virtual_function.quotient_set.iter().all(|&q| q != x),
                    "The virtual function is undefined at points in its quotient set"
                );

                // Computing the quotient (Quot in the article)
                let quotient_num = f_x - eval_poly(&virtual_function.interpolating_polynomial, x);

                let quotient_denom_inverse = quotient_denom_inverse_hint.unwrap_or_else(|| {
                    virtual_function
                        .quotient_set
                        .iter()
                        .map(|q| x - *q)
                        .product::<F>()
                        .inverse()
                });

                let quotient_evalution = quotient_num * quotient_denom_inverse;

                // Computing the degree-correction factor (1 - rx)^(e + 1)/(1 - rx)
                let (rx, denom_inverse) = deg_cor_hint.unwrap_or_else(|| {
                    let rx = x * virtual_function.comb_randomness;
                    let denom_inverse = if rx == F::ONE {
                        F::ONE
                    } else {
                        (F::ONE - rx).inverse()
                    };
                    (rx, denom_inverse)
                });

                let num_terms = virtual_function.quotient_set.len();

                let scale_factor = if rx != F::ONE {
                    (F::ONE - rx.exp_u64((num_terms + 1) as u64)) * denom_inverse
                } else {
                    F::from_usize(num_terms + 1)
                };

                // Putting the quotient and degree correction together into the
                // desired virtual-function evaluation
                quotient_evalution * scale_factor
            }
        }
    }
}

/// Auxiliary verifier information which is passed to and received from round
/// verification
pub struct VerificationState<F: TwoAdicField, M: Mmcs<F>> {
    // The indices are given in the following frame of reference: Self is
    // produced inside verify_round for round i (for 1 <= i <= M). The final
    // round does not call verify_round. The first verification state, with
    // index 0, is constructed directly by the verifier.

    // Oracle used to compute the value of the virtual function f_i
    oracle: Oracle<F>,

    // Domain L_i
    domain: TwoAdicMultiplicativeCoset<F>,

    // Folding randomness r_i to be used in the *next* round
    folding_randomness: F,

    // Round index i
    round: usize,

    // Root of the Merkle tree comitting to g_i (or f_0 in the case i = 0)
    root: M::Commitment,
}

/// Verifies the proof that the committed codeword satisfies the low-degreeness
/// bound specified in the configuration.
///
/// # Parameters
///
/// - `config`: The full STIR configuration, including the degree bound.
/// - `commitment`: The commitment to the codeword encoding the polynomial of interest.
/// - `proof`: The proof to verify, which includes the committed .
/// - `challenger`: The challenger to use for the proof verification.
///
/// # Returns
pub fn verify<F, EF, M, C>(
    config: &StirConfig<F, EF, M>,
    commitment: M::Commitment,
    proof: StirProof<EF, M, C::Witness>,
    challenger: &mut C,
) -> Result<(), VerificationError>
where
    F: TwoAdicField,
    EF: TwoAdicField + ExtensionField<F>,
    M: Mmcs<EF>,
    C: FieldChallenger<F> + GrindingChallenger + CanObserve<M::Commitment>,
{
    // Inform the verifier if the configuration requires a proof of work from
    // the prover larger than the POW_BITS_WARNING constant. This is only logged
    // if the tracing module has been init()ialised.
    if config
        .pow_bits_all_rounds()
        .iter()
        .any(|&x| x > POW_BITS_WARNING)
    {
        tracing::warn!(
            "The configuration requires the prover to compute a proof of work of more than {} bits",
            POW_BITS_WARNING
        );
    }

    // Observe the public parameters
    observe_public_parameters(config.parameters(), challenger);

    let StirProof {
        round_proofs,
        starting_folding_pow_witness,
        final_polynomial,
        final_pow_witness,
        final_round_queries,
    } = proof;

    // Degree check on p = g_{M + 1}
    if final_polynomial.len() > 1 << config.log_stopping_degree() {
        return Err(VerificationError::FinalPolynomialDegree);
    }

    // Observe the commitment
    challenger.observe(F::from_u8(Messages::Commitment as u8));
    challenger.observe(commitment.clone());

    // Check the initial proof of work
    if !challenger.check_witness(
        config.starting_folding_pow_bits(),
        starting_folding_pow_witness,
    ) {
        return Err(VerificationError::InitialProofOfWork);
    }

    // Sample the folding randomness r_0
    challenger.observe(F::from_u8(Messages::FoldingRandomness as u8));
    let folding_randomness = challenger.sample_algebra_element();

    let log_size = config.log_starting_degree() + config.log_starting_inv_rate();

    // Cf. the commit method in prover/mod.rs for an explanation on the chosen
    // domain sequence L_0, L_1, ...
    let domain =
        TwoAdicMultiplicativeCoset::new(EF::two_adic_generator(log_size), log_size).unwrap();

    // Preparing the initial verification state manually
    let mut verification_state = VerificationState {
        oracle: Oracle::Transparent,
        domain,
        folding_randomness,
        round: 0,
        root: commitment,
    };

    // ====================== Verification of full rounds ======================
    for (i, round_proof) in round_proofs.into_iter().enumerate() {
        verification_state = verify_round(config, verification_state, round_proof, challenger)
            .map_err(|e| VerificationError::Round(i + 1, e))?;
    }

    let VerificationState {
        oracle: final_oracle,
        domain: final_domain,
        folding_randomness: final_folding_randomness,
        root: g_m_root,
        ..
    } = verification_state;

    // ==================== Verification of the final round ====================

    // log2(k_M)
    let log_last_folding_factor = config.log_last_folding_factor();

    // Logarithm of |(L_M)^{k_M}|
    let log_final_query_domain_size = final_domain.log_size() - log_last_folding_factor;

    // Observe the final polynomial
    challenger.observe(F::from_u8(Messages::FinalPolynomial as u8));
    observe_ext_slice_with_size(challenger, &final_polynomial);

    // Sample the final queried indices
    challenger.observe(F::from_u8(Messages::FinalQueryIndices as u8));
    let final_queried_indices: Vec<usize> = (0..config.final_num_queries())
        .map(|_| challenger.sample_bits(log_final_query_domain_size))
        .unique()
        .collect();

    // Verifying paths of the evaluations of g_M at the k_M-th roots of the
    // final queried points
    for (&i, (leaf, proof)) in final_queried_indices
        .iter()
        .unique()
        .zip(final_round_queries.iter())
    {
        let leaf_vec = vec![leaf.clone()];
        let batch_proof = BatchOpeningRef::new(&leaf_vec, proof);

        if config
            .mmcs_config()
            .verify_batch(
                &g_m_root,
                &[Dimensions {
                    width: 1 << log_last_folding_factor,
                    height: 1 << (final_domain.log_size() - log_last_folding_factor),
                }],
                i,
                batch_proof,
            )
            .is_err()
        {
            return Err(VerificationError::FinalQueryPath);
        }
    }

    // Recover the evaluations of g_M needed to compute the values of f_M the
    // k_M-th roots of the final queried points
    let g_m_evals = final_round_queries
        .into_iter()
        .map(|(eval_batch, _)| eval_batch)
        .collect_vec();

    // Compute the values of f_M at the relevant points given the evaluations of
    // g_M
    let f_m_evals = compute_f_oracle_from_g(
        &final_oracle,
        g_m_evals,
        &final_queried_indices,
        &final_domain,
        log_last_folding_factor,
    );

    // The j-th element of this vector is a distinguished the k_M-th root of the
    // j-th queried point r^shift_{M, j}
    let final_queried_point_roots = final_queried_indices
        .iter()
        .map(|&i| final_domain.element(i))
        .collect_vec();

    // Fold the evaluations of f_M at the roots of the queried points
    let p_evals = compute_folded_evaluations(
        f_m_evals,
        &final_queried_point_roots,
        log_last_folding_factor,
        final_folding_randomness,
        final_domain
            .subgroup_generator()
            .exp_power_of_2(log_final_query_domain_size),
    );

    // Match the evaluations of the final polynomial p = g_{M + 1} sent by the
    // prover against the expected ones computed above
    if !p_evals
        .into_iter()
        .zip(final_queried_point_roots)
        .all(|(eval, root)| {
            eval_poly(
                &final_polynomial,
                root.exp_power_of_2(log_last_folding_factor),
            ) == eval
        })
    {
        return Err(VerificationError::FinalPolynomialEvaluations);
    }

    // Check the final proof of work
    if !challenger.check_witness(config.final_pow_bits(), final_pow_witness) {
        return Err(VerificationError::FinalProofOfWork);
    }

    Ok(())
}

// Verifies the proof of a single full round i = 1, ..., M of STIR
fn verify_round<F, EF, M, C>(
    // The full STIR configuration from which the round-specific configuration
    // is extracted
    config: &StirConfig<F, EF, M>,
    // The verification state produced by the previous full round (or the
    // initial one computed manually)
    verification_state: VerificationState<EF, M>,
    // The proof for the current round
    round_proof: RoundProof<EF, M, C::Witness>,
    // Challenger for the transcript
    challenger: &mut C,
) -> Result<VerificationState<EF, M>, FullRoundVerificationError>
where
    F: TwoAdicField,
    EF: TwoAdicField + ExtensionField<F>,
    M: Mmcs<EF>,
    C: FieldChallenger<F> + GrindingChallenger + CanObserve<M::Commitment>,
{
    let round = verification_state.round + 1;

    // De-structure the round-specific configuration, verification state and
    // round proof
    let RoundConfig {
        log_folding_factor,
        pow_bits,
        num_queries,
        num_ood_samples,
        log_evaluation_domain_size,
        log_inv_rate,
        ..
    } = config.round_config(round).clone();

    let VerificationState {
        oracle,
        domain,
        folding_randomness,
        root: prev_root,
        ..
    } = verification_state;

    let RoundProof {
        g_root,
        betas,
        ans_polynomial,
        query_proofs,
        shake_polynomial,
        pow_witness,
    } = round_proof;

    // Observe the commitment to g_{i - 1}
    challenger.observe(F::from_u8(Messages::RoundCommitment as u8));
    challenger.observe(g_root.clone());

    // Sampling the out-of-domain points
    let mut ood_samples = Vec::new();

    challenger.observe(F::from_u8(Messages::OodSamples as u8));
    while ood_samples.len() < num_ood_samples {
        let el: EF = challenger.sample_algebra_element();

        // Rejection sampling: these points have to be outside L_{i - 1}
        if !domain.contains(el) {
            ood_samples.push(el);
        }
    }

    // Observe the betas, i. e. the replies to the out-of-domain queries
    challenger.observe(F::from_u8(Messages::Betas as u8));
    betas
        .iter()
        .for_each(|&beta| challenger.observe_algebra_element(beta));

    // Sample the degree-correction randomness
    challenger.observe(F::from_u8(Messages::CombRandomness as u8));
    let comb_randomness = challenger.sample_algebra_element();

    // Sample the folding randomness for the next round
    challenger.observe(F::from_u8(Messages::FoldingRandomness as u8));
    let new_folding_randomness = challenger.sample_algebra_element();

    // Sample queried indices of elements in L_{i - 1}^k_{i-1}
    let log_query_domain_size = domain.log_size() - log_folding_factor;

    challenger.observe(F::from_u8(Messages::QueryIndices as u8));
    let queried_indices: Vec<usize> = (0..num_queries)
        .map(|_| challenger.sample_bits(log_query_domain_size))
        .unique()
        .collect();

    // Check the proof of work for this round
    if !challenger.check_witness(pow_bits, pow_witness) {
        return Err(FullRoundVerificationError::ProofOfWork);
    }

    // Observe the Ans and shake polynomials
    challenger.observe(F::from_u8(Messages::AnsPolynomial as u8));
    observe_ext_slice_with_size(challenger, &ans_polynomial);

    challenger.observe(F::from_u8(Messages::ShakePolynomial as u8));
    observe_ext_slice_with_size(challenger, &shake_polynomial);

    // Sample the shake randomness
    challenger.observe(F::from_u8(Messages::ShakeRandomness as u8));
    let shake_randomness: EF = challenger.sample_algebra_element();

    // Verify the Merkle proofs of the evaluations of g_{i - 1}
    for (&i, (leaf, proof)) in queried_indices.iter().unique().zip(query_proofs.iter()) {
        let leaf_vec = vec![leaf.clone()];
        let batch_proof = BatchOpeningRef::new(&leaf_vec, proof);

        if config
            .mmcs_config()
            .verify_batch(
                &prev_root,
                &[Dimensions {
                    width: 1 << log_folding_factor,
                    height: 1 << (domain.log_size() - log_folding_factor),
                }],
                i,
                batch_proof,
            )
            .is_err()
        {
            return Err(FullRoundVerificationError::QueryPath);
        }
    }

    // The j-th element of this vector is the list of evaluations of g_{i - 1}
    // at the k_{i - 1}-th roots of the j-th sampled point r^shift_{i, j}.
    // These give rise to the values of f_{i - 1} at the same points, which got
    // folded into g_i(r^shift_{i, j}).
    let previous_g_values = query_proofs.into_iter().map(|(leaf, _)| leaf).collect_vec();

    // Compute the values of f_{i - 1} from those of g_{i - 1}
    let previous_f_values = compute_f_oracle_from_g(
        &oracle,
        previous_g_values,
        &queried_indices,
        &domain,
        log_folding_factor,
    );

    // For each r^shift_{i, j} in L_{i - 1}^{k_{i - 1}}, we compute one
    // distinguished k_{i - 1}-th root in L_{i - 1}
    let queried_point_roots = queried_indices
        .iter()
        .map(|&i| domain.element(i))
        .collect_vec();

    // We can now fold the evaluations of f_{i - 1} using the oracle
    let folded_evals = compute_folded_evaluations(
        previous_f_values,
        &queried_point_roots,
        log_folding_factor,
        folding_randomness,
        domain
            .subgroup_generator()
            .exp_power_of_2(log_query_domain_size),
    );

    let folded_answers = queried_point_roots
        .into_iter()
        .zip(folded_evals)
        .map(|(root, eval)| (root.exp_power_of_2(log_folding_factor), eval));

    // The quotient definining the function
    let quotient_answers: Vec<_> = ood_samples
        .into_iter()
        .zip(betas)
        .chain(folded_answers)
        .collect();

    // Check that Ans interpolates the expected values using the shake polynomial
    if ans_polynomial.len() > quotient_answers.len() {
        return Err(FullRoundVerificationError::AnsPolynomialDegree);
    }

    let quotient_set = quotient_answers.iter().map(|(x, _)| *x).collect_vec();

    // This is the degree bound (plus 1) for g_i:
    let degree_bound = 1 << (log_evaluation_domain_size - log_inv_rate);
    if quotient_set.len() >= degree_bound {
        // In this case, the interpolator ans_i coincides with g_i, causing f_i
        // to be 0. A potential cause is the combination of a small field,
        // low-degree initial polynomial and large number of security bits
        // required. This does not make the protocol vulnerable, but perhaps
        // less efficient than it could be. In the future, early termination can
        // be designed and implemented for this case, but this is unexplored as
        // of yet.
        tracing::info!("Warning: quotient polynomial is zero in round {}", round);
    }

    // Check that the Ans polynomial interpolates the expected values using
    // the shake polynomial
    if !verify_evaluations(
        &ans_polynomial,
        &shake_polynomial,
        shake_randomness,
        quotient_answers,
    ) {
        return Err(FullRoundVerificationError::AnsPolynomialEvaluations);
    }

    // Produce the new verification state
    Ok(VerificationState {
        oracle: Oracle::Virtual(VirtualFunction {
            comb_randomness,
            interpolating_polynomial: ans_polynomial,
            quotient_set,
        }),
        domain: domain.shrink_coset(1).unwrap(), // Can never panic due to parameter set-up
        folding_randomness: new_folding_randomness,
        round,
        root: g_root,
    })
}

// Compute the values of the oracle f_i given its underlying function g_i
// (or f_i itself in the case of a transparent oracle)
fn compute_f_oracle_from_g<F: TwoAdicField>(
    // Oracle relating f_i to its underlying function g_i
    oracle: &Oracle<F>,
    // Evaluations of g_i at the points of interest
    g_eval_batches: Vec<Vec<F>>,
    // The indices of the queried elements of L_i^{k_i}
    queried_indices: &[usize],
    // The domain L_i
    domain: &TwoAdicMultiplicativeCoset<F>,
    // The log of the folding factor k_i
    log_folding_factor: usize,
) -> Vec<Vec<F>> {
    // 1. Compute the set of k_i-th roots of r^shift_{i, j} for each sampled
    // point r^shift_{i, j} from L_i^k_i. This is simply the coset
    //   (s_j) * {1, c, ..., c^{k_i - 1}},
    // where s_j is any particular k_i-th root of the point r^shift_{i, j} and
    // c is a primitive k_i-th root of unity of the field.

    // This is the size k_i of each coset of roots
    let folding_factor = 1 << log_folding_factor;

    // This is the generator c of (the subgroup defining) each coset
    let log_scaling_factor = domain.log_size() - log_folding_factor;
    let generator = domain
        .subgroup_generator()
        .exp_power_of_2(log_scaling_factor);

    // The j-th element of this vector is the set of k_i-th roots of r^shift_{i, j}
    let queried_point_preimages = queried_indices
        .iter()
        .map(|index| {
            iterate(domain.element(*index), |&x| x * generator)
                .take(folding_factor)
                .collect_vec()
        })
        .collect_vec();

    // 2. Compute the values of f_i at the each element of each coset using the
    // values of g_i therein

    // Each call to the oracle involves a division by an evaluation of the
    // vanishing polynomial at the query set. We can batch-invert those
    // denominators outside the oracle-evaluation function in order to reduce the
    // number of costly calls to invert()
    let denom_inv_hints = match oracle {
        Oracle::Transparent => vec![vec![None; folding_factor]; queried_point_preimages.len()],
        Oracle::Virtual(virtual_function) => {
            let flat_denoms: Vec<F> = queried_point_preimages
                .iter()
                .flat_map(|points| {
                    // Computing the denominator, i. e. the value of the vanishing
                    // polynomial of the quotient set at the point
                    points
                        .iter()
                        .map(|point| {
                            virtual_function
                                .quotient_set
                                .iter()
                                .map(|q| *point - *q)
                                .product()
                        })
                        .collect_vec()
                })
                .collect_vec();

            // Batch-inverting the denominators
            let flat_denom_invs = batch_multiplicative_inverse(&flat_denoms)
                .into_iter()
                .map(|x| Some(x))
                .collect_vec();

            flat_denom_invs
                .chunks_exact(folding_factor)
                .map(|chunk| chunk.to_vec())
                .collect_vec()
        }
    };

    // A similar phenomenon occurs with the degree-correction factor
    // (1 - rx)^(e - 1)/(1 - rx) in the notation of the article (sec. 2.3): we
    // can batch-invert the denominators to save on invert() calls. Since this
    // already necessitates rx, we also store the latter and pass it to the
    // oracle-evaluation function. Note that this batch inversion could be lumped
    // together with the one above for maximum savings at the cost of code
    // clarity.
    let deg_cor_hints = match oracle {
        Oracle::Transparent => vec![vec![None; folding_factor]; queried_point_preimages.len()],
        Oracle::Virtual(virtual_function) => {
            let (flat_rx_s, flat_denoms): (Vec<F>, Vec<F>) = queried_point_preimages
                .iter()
                .flat_map(|points| {
                    points.iter().map(|point| {
                        let rx = *point * virtual_function.comb_randomness;

                        if rx == F::ONE {
                            (F::ONE, F::ONE)
                        } else {
                            (rx, (F::ONE - rx))
                        }
                    })
                })
                .unzip();

            let flat_denom_invs = batch_multiplicative_inverse(&flat_denoms);

            flat_rx_s
                .into_iter()
                .zip(flat_denom_invs)
                .map(|(rx, denom_inv)| Some((rx, denom_inv)))
                .collect_vec()
                .chunks_exact(folding_factor)
                .map(|chunk| chunk.to_vec())
                .collect_vec()
        }
    };

    // Compute the values of f_i at the each element of each coset using the
    // precomputed hints
    queried_point_preimages
        .into_iter()
        .zip(g_eval_batches)
        .zip(denom_inv_hints)
        .zip(deg_cor_hints)
        .map(
            |(((points, g_eval_batch), denom_inverse_hints), deg_cor_hints)| {
                points
                    .into_iter()
                    .zip(g_eval_batch)
                    .zip(denom_inverse_hints)
                    .zip(deg_cor_hints)
                    .map(|(((point, g_eval), denom_inverse_hint), deg_cor_hint)| {
                        oracle.evaluate(point, g_eval, denom_inverse_hint, deg_cor_hint)
                    })
                    .collect_vec()
            },
        )
        .collect_vec()
}

// Given the evaluations of a polynomial at the set of k-th roots of a point,
// compute the evaluation of the k-ary folding of that polynomial at the point.
fn compute_folded_evaluations<F: TwoAdicField>(
    // The j-th element of this list is the list of evaluations of the original
    // polynomial at the k-th roots of the j-th point of interest
    unfolded_evaluations: Vec<Vec<F>>,
    // The j-th element of this list is a k-th root of the j-th point of
    // interest
    point_roots: &[F],
    // The log2 of the folding factor k (i. e. of the arity)
    log_folding_factor: usize,
    // The folding coefficient
    c: F,
    // A primitive k-th root of unity (passed for efficiency reasons)
    omega: F,
) -> Vec<F> {
    // We need the inverses of each root received, so we batch-invert them to
    // save on calls to invert()
    let point_roots_invs = batch_multiplicative_inverse(point_roots);

    // This is called once per round and could be further amortised by passing
    // it as a hint, at the cost of a less clean interface of verify_round()
    let two_inv = F::TWO.inverse();

    let omega_inv = omega.inverse();

    // For each list of evaluations, we call the method fold_evaluations() once
    unfolded_evaluations
        .into_iter()
        .zip(point_roots_invs)
        .map(|(evals, point_root_inv)| {
            fold_evaluations_at_small_domain(
                evals,
                point_root_inv,
                log_folding_factor,
                (omega, omega_inv),
                c,
                two_inv,
            )
        })
        .collect()
}

// Verify that f takes the values y_1, ..., y_n at x_1, ..., x_n (resp.)
// using the auxiliary shake polynomial
//     q(x) = (f(x) - y_1) / (x - x_1) + ... + (f(x) - y_n) / (x - x_n).
// The above equation is checked at a uniformly sampled random point r.
fn verify_evaluations<F: TwoAdicField>(
    // Polynomial whose evaluations are being checked
    f: &[F],
    // Shake polynomial q
    shake_polynomial: &[F],
    // Random field element where the equation is checked
    r: F,
    // Vector of point-evaluation pairs (x_i, y_i)
    points: Vec<(F, F)>,
) -> bool {
    let f_eval = eval_poly(f, r);
    let shake_eval = eval_poly(shake_polynomial, r);

    let (xs, ys): (Vec<F>, Vec<F>) = points.into_iter().unzip();

    // Batch-invert all the denominators for efficiency
    let denominators = batch_multiplicative_inverse(&xs.into_iter().map(|x| r - x).collect_vec());

    shake_eval
        == ys
            .into_iter()
            .zip(denominators)
            .map(|(y, denom)| (f_eval - y) * denom)
            .sum()
}
