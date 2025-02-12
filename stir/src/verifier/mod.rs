use alloc::vec;
use alloc::vec::Vec;
use error::{FullRoundVerificationError, VerificationError};
use p3_coset::TwoAdicCoset;
use p3_matrix::Dimensions;

use itertools::{iterate, Itertools};
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{batch_multiplicative_inverse, ExtensionField, Field, TwoAdicField};
use p3_poly::Polynomial;

use crate::utils::observe_ext_slice_with_size;
use crate::{
    config::{observe_public_parameters, RoundConfig},
    proof::RoundProof,
    utils::fold_evaluations,
    StirConfig, StirProof,
};
use crate::{Messages, POW_BITS_WARNING};

mod error;

#[cfg(test)]
mod tests;

// The virtual function
//     `DegCor(Quotient(f, interpolating_polynomial), quotient_set)`
// in the notation of the paper, where `f` is the underlying function.
struct VirtualFunction<F: TwoAdicField> {
    // In in the case of interest,
    // - f is g_i
    // - interpolating_polynomial is Ans_i
    // - quotient_set is G_i
    comb_randomness: F,
    interpolating_polynomial: Polynomial<F>,
    quotient_set: Vec<F>,
}

// NP TODO rethink generality of description

// Oracle allowing the verifier to compute values of f_i, either directly for
// the original codeword, or using the combination (i. e. degree-correction)
// and folding randomness and the values of g_i.
enum Oracle<F: TwoAdicField> {
    // Transparent oracle: it takes the same value as the underlying function
    Transparent,
    // Virtual oracle: it takes the value of the virtual function over the underlying function
    Virtual(VirtualFunction<F>),
}

impl<F: TwoAdicField> Oracle<F> {
    // Compute the value v(x) of the oracle v at the point x given the value f(x) of the underlying function
    fn evaluate(
        &self,
        x: F,
        f_x: F,
        // Since each call to `evaluate` involves dividing by a denominator, we
        // can batch-invert many of them outside and feed them to the function
        quotient_denom_inverse_hint: Option<F>,
        // A similar phenomenon occurs for the
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
                let quotient_num = f_x - virtual_function.interpolating_polynomial.evaluate(&x);

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
                    F::from_canonical_usize(num_terms + 1)
                };

                // Putting the quotient and degree correction together into the
                // desired virtual-function evaluation
                quotient_evalution * scale_factor
            }
        }
    }
}

/// Input to and output of round verification
pub struct VerificationState<F: TwoAdicField, M: Mmcs<F>> {
    // The indices are given in the following frame of reference: Self is
    // produced inside verify_round for round i (in {1, ..., num_rounds}). The
    // final round ("consistency with the final polynomial"), with index
    // num_rounds + 1, does not produce a StirWitness.

    // Oracle used to compute the value of the virtual function f_i
    oracle: Oracle<F>,

    // NP TODO maybe move to the config or somewehre else (this is proof-independent)

    // Domain L_i
    domain: TwoAdicCoset<F>,

    // Folding randomness r_i to be used in the next round
    folding_randomness: F,

    // Index i in the main loop - starts at 0 with the verification state computed before
    // the first round (round 1)
    round: usize,

    // Root of the Merkle tree comitting to g_i
    root: M::Commitment,
}

pub fn verify<F, EF, M, C>(
    config: &StirConfig<EF, M>,
    proof: StirProof<EF, M, C::Witness>,
    challenger: &mut C,
) -> Result<(), VerificationError>
where
    F: Field,
    EF: TwoAdicField + ExtensionField<F>,
    M: Mmcs<EF>,
    C: FieldChallenger<F> + GrindingChallenger + CanObserve<M::Commitment>,
{
    if config
        .pow_bits_all_rounds()
        .iter()
        .any(|&x| x > POW_BITS_WARNING)
    {
        tracing::warn!(
            "The configuration requires the verifier to compute a proof of work of more than {} bits",
            POW_BITS_WARNING
        );
    }

    observe_public_parameters(config.parameters(), challenger);

    let StirProof {
        commitment,
        round_proofs,
        final_polynomial,
        pow_witness,
        final_round_queries,
    } = proof;

    if final_polynomial
        .degree()
        .is_some_and(|d| d + 1 > 1 << config.log_stopping_degree())
    {
        return Err(VerificationError::FinalPolynomialDegree);
    }

    // Observe the commitment
    challenger.observe(F::from_canonical_u8(Messages::Commitment as u8));
    challenger.observe(commitment.clone());

    challenger.observe(F::from_canonical_u8(Messages::FoldingRandomness as u8));
    let folding_randomness = challenger.sample_ext_element();

    let log_size = config.log_starting_degree() + config.log_starting_inv_rate();

    // Cf. prover/mod.rs for an explanation on the chosen sequence of domain
    // sizes
    let domain = TwoAdicCoset::new(EF::two_adic_generator(log_size), log_size);

    let mut verification_state = VerificationState {
        oracle: Oracle::Transparent,
        domain,
        folding_randomness,
        round: 0,
        root: commitment,
    };

    // Verifying each round
    for (i, round_proof) in round_proofs.into_iter().enumerate() {
        verification_state = verify_round(config, verification_state, round_proof, challenger)
            .map_err(|e| VerificationError::Round(i, e))?;
    }

    let VerificationState {
        oracle: final_oracle,
        domain: mut final_domain,
        folding_randomness: final_folding_randomness,
        root: g_m_root,
        ..
    } = verification_state;

    // Step 2: Consistency with final polynomial

    // log(k_M)
    let log_last_folding_factor = config.log_last_folding_factor();

    // Logarithm of |(L_M)^k_M|
    let log_final_query_domain_size = final_domain.log_size() - log_last_folding_factor;

    // Absorb the final polynomial
    challenger.observe(F::from_canonical_u8(Messages::FinalPolynomial as u8));
    observe_ext_slice_with_size(challenger, final_polynomial.coeffs());

    // Squeeze the final indices
    challenger.observe(F::from_canonical_u8(Messages::FinalQueryIndices as u8));
    let final_queried_indices: Vec<usize> = (0..config.final_num_queries())
        .map(|_| challenger.sample_bits(log_final_query_domain_size))
        .unique()
        .collect();

    // Verifying paths of those evaluations of g_M
    for (&i, (leaf, proof)) in final_queried_indices
        .iter()
        .unique()
        .zip(final_round_queries.iter())
    {
        if config
            .mmcs_config()
            .verify_batch(
                &g_m_root,
                &[Dimensions {
                    width: 1 << log_last_folding_factor,
                    height: 1 << (final_domain.log_size() - log_last_folding_factor),
                }],
                i as usize,
                &vec![leaf.clone()],
                &proof,
            )
            .is_err()
        {
            return Err(VerificationError::FinalQueryPath);
        }
    }

    // Recover the evaluations of g_M needed to compute the values of f_M at
    // points which are relevant to evaluate p(r_i) = Fold(f_M, ...)(r_i), where
    // r_i runs over the final queried indices
    let g_m_evals = final_round_queries
        .into_iter()
        .map(|(eval_batch, _)| eval_batch)
        .collect_vec();

    // Compute the values of f_M at the relevant points given the evaluations of g_M
    let f_m_evals = compute_f_oracle_from_g(
        &final_oracle,
        g_m_evals,
        &final_queried_indices,
        &mut final_domain,
        log_last_folding_factor,
    );

    let final_queried_point_roots = final_queried_indices
        .iter()
        .map(|&i| final_domain.element(i))
        .collect_vec();

    // Compute the evaluations of p (which one could call g_{M + 1}) given those of f_M
    let p_evals = compute_folded_evaluations(
        f_m_evals,
        // These are some k_M-th roots of the queried points
        &final_queried_point_roots,
        log_last_folding_factor,
        final_folding_randomness,
        final_domain
            .generator()
            .exp_power_of_2(log_final_query_domain_size),
    );

    if !p_evals
        .into_iter()
        .zip(final_queried_point_roots)
        .all(|(eval, root)| {
            final_polynomial.evaluate(&root.exp_power_of_2(log_last_folding_factor)) == eval
        })
    {
        return Err(VerificationError::FinalPolynomialEvaluations);
    }

    if !challenger.check_witness(config.final_pow_bits(), pow_witness) {
        return Err(VerificationError::FinalProofOfWork);
    }

    Ok(())
}

fn verify_round<F, EF, M, C>(
    config: &StirConfig<EF, M>,
    verification_state: VerificationState<EF, M>,
    round_proof: RoundProof<EF, M, C::Witness>,
    challenger: &mut C,
) -> Result<VerificationState<EF, M>, FullRoundVerificationError>
where
    F: Field,
    EF: TwoAdicField + ExtensionField<F>,
    M: Mmcs<EF>,
    C: FieldChallenger<F> + GrindingChallenger + CanObserve<M::Commitment>,
{
    let round = verification_state.round + 1;

    // De-structure the round-specific configuration and the verification state
    let RoundConfig {
        log_folding_factor,
        pow_bits,
        num_queries,
        num_ood_samples,
        ..
    } = config.round_config(round).clone();

    let VerificationState {
        oracle,
        mut domain,
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

    // Update the transcript with the root of the Merkle tree

    // Observe the commitment
    challenger.observe(F::from_canonical_u8(Messages::RoundCommitment as u8));
    challenger.observe(g_root.clone());

    // Rejection sampling on the out of domain samples
    let mut ood_samples = Vec::new();

    challenger.observe(F::from_canonical_u8(Messages::OodSamples as u8));
    while ood_samples.len() < num_ood_samples {
        let el: EF = challenger.sample_ext_element();
        if !domain.contains(el) {
            ood_samples.push(el);
        }
    }

    // Observe the betas
    challenger.observe(F::from_canonical_u8(Messages::Betas as u8));
    betas
        .iter()
        .for_each(|&beta| challenger.observe_ext_element(beta));

    // Sample ramdomness used for degree correction
    challenger.observe(F::from_canonical_u8(Messages::CombRandomness as u8));
    let comb_randomness = challenger.sample_ext_element();

    // Sample folding randomness for the next round
    challenger.observe(F::from_canonical_u8(Messages::FoldingRandomness as u8));
    let new_folding_randomness = challenger.sample_ext_element();

    // Sample queried indices of elements in L_{i - 1}^k_{i-1}
    let log_query_domain_size = domain.log_size() - log_folding_factor;

    challenger.observe(F::from_canonical_u8(Messages::QueryIndices as u8));
    let queried_indices: Vec<usize> = (0..num_queries)
        .map(|_| challenger.sample_bits(log_query_domain_size))
        .unique()
        .collect();

    // Verify proof of work
    if !challenger.check_witness(pow_bits, pow_witness) {
        return Err(FullRoundVerificationError::ProofOfWork);
    }

    // Update the transcript with the coefficients of the answer and shake polynomials
    challenger.observe(F::from_canonical_u8(Messages::AnsPolynomial as u8));
    observe_ext_slice_with_size(challenger, ans_polynomial.coeffs());

    challenger.observe(F::from_canonical_u8(Messages::ShakePolynomial as u8));
    observe_ext_slice_with_size(challenger, shake_polynomial.coeffs());

    challenger.observe(F::from_canonical_u8(Messages::ShakeRandomness as u8));
    let shake_randomness: EF = challenger.sample_ext_element();

    // Verify Merkle paths
    for (&i, (leaf, proof)) in queried_indices.iter().unique().zip(query_proofs.iter()) {
        if config
            .mmcs_config()
            .verify_batch(
                &prev_root,
                &[Dimensions {
                    width: 1 << log_folding_factor,
                    height: 1 << (domain.log_size() - log_folding_factor),
                }],
                i as usize,
                &vec![leaf.clone()],
                proof,
            )
            .is_err()
        {
            return Err(FullRoundVerificationError::QueryPath);
        }
    }

    // The j-th element of this vector is the list of values of g_{i - 1} which
    // result in the list of values of f_{i - 1} (by virtue of f_{i - 1} being
    // a virtual function reliant on g_{i - 1}) which get folded into
    // g_i(r_{i, j}^shift)
    let previous_g_values = query_proofs.into_iter().map(|(leaf, _)| leaf).collect_vec();

    // Compute the values of f_{i - 1} from those of g_{i - 1}
    let previous_f_values = compute_f_oracle_from_g(
        &oracle,
        previous_g_values,
        &queried_indices,
        &mut domain,
        log_folding_factor,
    );

    let queried_point_roots = queried_indices
        .iter()
        .map(|&i| domain.element(i))
        .collect_vec();

    // Now, for each of the selected random points, we need to compute the folding of the
    // previous oracle
    let folded_evals = compute_folded_evaluations(
        previous_f_values,
        // These are some k_{i - 1}-th roots of queried points
        &queried_point_roots,
        log_folding_factor,
        folding_randomness,
        domain.generator().exp_power_of_2(log_query_domain_size),
    );

    let folded_answers = queried_point_roots
        .into_iter()
        .zip(folded_evals)
        .map(|(root, eval)| (root.exp_power_of_2(log_folding_factor), eval));

    // The quotient definining the function
    // NP Ans_i in Verifier/Main loop/(b)
    let quotient_answers: Vec<_> = ood_samples
        .into_iter()
        .zip(betas)
        .map(|(alpha, beta)| (alpha, beta))
        .chain(folded_answers)
        .collect();

    // Check that Ans interpolates the expected values using the shake polynomial
    if ans_polynomial
        .degree()
        .is_some_and(|d| d >= quotient_answers.len())
    {
        return Err(FullRoundVerificationError::AnsPolynomialDegree);
    }

    let quotient_set = quotient_answers.iter().map(|(x, _)| *x).collect_vec();

    if !verify_evaluations(
        &ans_polynomial,
        &shake_polynomial,
        shake_randomness,
        quotient_answers,
    ) {
        return Err(FullRoundVerificationError::AnsPolynomialEvaluations);
    }

    Ok(VerificationState {
        oracle: Oracle::Virtual(VirtualFunction {
            comb_randomness,
            interpolating_polynomial: ans_polynomial,
            quotient_set,
        }),
        domain: domain.shrink_subgroup(1),
        folding_randomness: new_folding_randomness,
        round: round,
        root: g_root,
    })
}

fn compute_f_oracle_from_g<F: TwoAdicField>(
    // Oracle relating f_i to its underlying function g_i
    oracle: &Oracle<F>,
    // Evaluations of g_i at the lists of points relevant to each queried point
    g_eval_batches: Vec<Vec<F>>,
    // The queried indices of L_i^{k_i}
    queried_indices: &[usize],
    // The domain L_i
    domain: &mut TwoAdicCoset<F>,
    // The log of the folding factor k_i
    log_folding_factor: usize,
) -> Vec<Vec<F>> {
    // 1. Computing the set S_j of k_i-th roots of r_j for each sampled point
    // r_j from L_i^k_i. S_j is the coset (s_j) * {1, c, ..., c^{k_i - 1}},
    // where s_j is any one k_i-th root and c is the generator of the kernel of
    // k: L_i ->> L_i^{k_i}, i. e. c = g_i^(|L_i| / k_i) (where g_i is the
    // chosen generator of L_i)

    // This is the length of each coset S_j
    let folding_factor = 1 << log_folding_factor;

    // This is the generator c of (the subgroup defining) each coset S_j
    let log_scaling_factor = domain.log_size() - log_folding_factor;
    let generator = domain.generator().exp_power_of_2(log_scaling_factor);

    // The j-th element of this vector is S_i, the set of k_i-th roots of r_j
    let queried_point_preimages = queried_indices
        .into_iter()
        .map(|index| {
            iterate(domain.element(*index), |&x| x * generator)
                .take(folding_factor)
                .collect_vec()
        })
        .collect_vec();

    // 2. Compute the values of f_i at the each element of each S_j using the
    // values of g_i therein

    // Each call to the oracle involves a division by an evaluation of the
    // vanishing polynomial at the query set. We can batch-invert those
    // denominators outside to reduce the number of costly calls to invert()
    let denom_inv_hints = match oracle {
        Oracle::Transparent => vec![vec![None; folding_factor]; queried_point_preimages.len()],
        Oracle::Virtual(virtual_function) => {
            let flat_denoms: Vec<F> = queried_point_preimages
                .iter()
                .flat_map(|points| {
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
    // (1 - rx)^(e - 1)/(1 - rx) in the notation of the paper (sec. 2.3): we can
    // batch-invert the denominators to save on invert() calls. Since this
    // already necessitates rx, we also store the latter and pass it to the
    // oracle-computing function. Note that this batch inversion could be lumped
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

    // Compute the values of f_i at the each element of each S_j using the
    // precomputed hints
    queried_point_preimages
        .into_iter()
        .zip(g_eval_batches.into_iter())
        .zip(denom_inv_hints.into_iter())
        .zip(deg_cor_hints.into_iter())
        .map(
            |(((points, g_eval_batch), denom_inverse_hints), deg_cor_hints)| {
                points
                    .into_iter()
                    .zip(g_eval_batch.into_iter())
                    .zip(denom_inverse_hints.into_iter())
                    .zip(deg_cor_hints.into_iter())
                    .map(|(((point, g_eval), denom_inverse_hint), deg_cor_hint)| {
                        oracle.evaluate(point, g_eval, denom_inverse_hint, deg_cor_hint)
                    })
                    .collect_vec()
            },
        )
        .collect_vec()
}

// Let p_1, ..., p_n be a list of points. For each p_i, given the evaluations of
// a polynomial h at the set of points
//   Y_i = {y in F: y^(arity) = p_i^(arity)},
// compute Fold(h, arity, c)(p_i^(arity)).
//
// Parameters
// - unfolded_evaluation_lists: The i-th element is the list of evaluations of h at Y_i
// - point_roots: The list of p_i's
// - log_arity: The folding arity is 2 raised to this value
// - c: The folding coefficient
// - omega: The generator of the subgroup of arity-th roots of unity in F
fn compute_folded_evaluations<F: TwoAdicField>(
    unfolded_evaluation_lists: Vec<Vec<F>>,
    point_roots: &[F],
    log_arity: usize,
    c: F,
    omega: F,
) -> Vec<F> {
    let point_roots_invs = batch_multiplicative_inverse(&point_roots);

    // This is called once per round and could be further amortised by passing
    // it as a hint, at the cost of a less clean interface of verify_round()
    let two_inv = F::TWO.inverse();

    let omega_inv = omega.inverse();

    unfolded_evaluation_lists
        .into_iter()
        .zip(point_roots.iter())
        .zip(point_roots_invs.into_iter())
        .map(|((evals, &point_root), point_root_inv)| {
            fold_evaluations(
                evals,
                point_root,
                log_arity,
                omega,
                c,
                Some(omega_inv),
                Some(point_root_inv),
                Some(two_inv),
            )
        })
        .collect()
}

// Verify that f takes the values y_1, ..., y_n at x_1, ..., x_n (resp.)
// using the technique of "shake polynomials". The prover has sent the verifier
// an auxiliary shake polynomial q defined (in the honest case) as
//     q(x) = (f(x) - y_1) / (x - x_1) + ... + (f(x) - y_n) / (x - x_n)
// The verifier then has sampled a random field element r and now checks the
// above equation there
fn verify_evaluations<F: TwoAdicField>(
    // Polynomial whose evaluations are being checked
    f: &Polynomial<F>,
    // Shale polynomial q
    shake_polynomial: &Polynomial<F>,
    // Random field element where the equation is checked
    r: F,
    // Vector of (x_i, y_i)
    points: Vec<(F, F)>,
) -> bool {
    let f_eval = f.evaluate(&r);
    let shake_eval = shake_polynomial.evaluate(&r);

    let (xs, ys): (Vec<F>, Vec<F>) = points.into_iter().unzip();

    let denominators = batch_multiplicative_inverse(&xs.into_iter().map(|x| r - x).collect_vec());

    shake_eval
        == ys
            .into_iter()
            .zip(denominators)
            .map(|(y, denom)| (f_eval - y) * denom)
            .sum()
}
