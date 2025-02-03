use alloc::vec;
use alloc::vec::Vec;
use error::{FullRoundVerificationError, VerificationError};
use p3_matrix::Dimensions;

use itertools::iterate;
use itertools::Itertools;
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{batch_multiplicative_inverse, ExtensionField, Field, TwoAdicField};

use crate::config::RoundConfig;
use crate::coset::Radix2Coset;
use crate::polynomial::Polynomial;
use crate::proof::RoundProof;
use crate::utils::fold_evaluations;
use crate::{StirConfig, StirProof};

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
        // NP TODO optimise
        // common_factors_inverse: F,
        // denom_hint: F,
        // ans_eval: F,
    ) -> F {
        match self {
            // In this case, the oracle contains the values of f_0 = g_0
            Oracle::Transparent => f_x,

            // In this case, we need to apply degree correction and the quotient
            Oracle::Virtual(virtual_function) => {
                assert!(
                    virtual_function.quotient_set.iter().all(|&q| q != x),
                    "The virtual function is undefined at points in its quotient set"
                );

                // NP TODO optimise?
                let quotient_num = f_x - virtual_function.interpolating_polynomial.evaluate(&x);
                let quotient_denom: F = virtual_function
                    .quotient_set
                    .iter()
                    .map(|q| x - *q)
                    .product();

                let quotient_evalution = quotient_num * quotient_denom.inverse();

                let num_terms = virtual_function.quotient_set.len();
                let common_factor = x * virtual_function.comb_randomness;

                let scale_factor = if common_factor != F::ONE {
                    (F::ONE - common_factor.exp_u64((num_terms + 1) as u64))
                        * (F::ONE - common_factor).inverse()
                } else {
                    F::from_canonical_usize(num_terms + 1)
                };

                quotient_evalution * scale_factor
            }
        }
    }
}

/// Input to verify_round
pub struct VerificationState<F: TwoAdicField, M: Mmcs<F>> {
    // The indices are given in the following frame of reference: Self is
    // produced inside verify_round for round i (in {1, ..., num_rounds}). The
    // final round ("consistency with the final polynomial"), with index
    // num_rounds + 1, does not produce a StirWitness.

    // Oracle used to compute the value of the virtual function f_i
    oracle: Oracle<F>,

    // NP TODO maybe move to the config or somewehre else (this is proof-independent)

    // Domain L_i
    domain: Radix2Coset<F>,

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
    let StirProof {
        commitment,
        round_proofs,
        final_polynomial,
        pow_witness,
        final_round_queries,
    } = proof;

    if final_polynomial.degree() + 1 > 1 << config.log_stopping_degree() {
        return Err(VerificationError::FinalPolynomialDegree);
    }

    // NP TODO verify merkle paths (inside main loop instead of separately PLUS final round)
    challenger.observe(commitment.clone());
    let folding_randomness = challenger.sample_ext_element();

    let log_size = config.log_starting_degree() + config.log_starting_inv_rate();

    // Cf. prover/mod.rs for an explanation on the chosen sequence of domain
    // sizes
    let domain = Radix2Coset::new(EF::two_adic_generator(log_size), log_size);

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
        domain: final_domain,
        folding_randomness: final_folding_randomness,
        root: g_m_root,
        ..
    } = verification_state;

    // Step 2: Consistency with final polynomial

    // log(k_M)
    let log_last_folding_factor = config.log_last_folding_factor();

    // Logarithm of |(L_M)^k_M|
    let log_final_query_domain_size = final_domain.log_size() - log_last_folding_factor;

    let final_queried_indices: Vec<u64> = (0..config.final_num_queries())
        .map(|_| challenger.sample_bits(log_final_query_domain_size) as u64)
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
        &final_domain,
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

    // NP TODO ask Giacomo why no shake polynomial in the last round
    if !p_evals
        .into_iter()
        .zip(final_queried_point_roots)
        .all(|(eval, root)| {
            final_polynomial.evaluate(&root.exp_power_of_2(log_last_folding_factor)) == eval
        })
    {
        return Err(VerificationError::FinalPolynomialEvaluations);
    }

    // NP TODO verify pow_witness
    if !challenger.check_witness(config.final_pow_bits().ceil() as usize, pow_witness) {
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
    // De-structure the round-specific configuration and the verification state
    let RoundConfig {
        log_folding_factor,
        pow_bits,
        num_queries,
        num_ood_samples,
        ..
    } = config.round_config(verification_state.round).clone();

    let VerificationState {
        oracle,
        domain,
        folding_randomness,
        round,
        root: prev_root,
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
    challenger.observe(g_root.clone());

    // Rejection sampling on the out of domain samples
    let mut ood_samples = Vec::new();

    while ood_samples.len() < num_ood_samples {
        let el: EF = challenger.sample_ext_element();
        if !domain.contains(el) {
            ood_samples.push(el);
        }
    }

    // Observe the betas
    betas
        .iter()
        .for_each(|&beta| challenger.observe_ext_element(beta));

    // Sample ramdomness used for degree correction
    let comb_randomness = challenger.sample_ext_element();

    // Sample folding randomness for the next round
    let new_folding_randomness = challenger.sample_ext_element();

    // Sample queried indices of elements in L_{i - 1}^k_{i-1}
    let log_query_domain_size = domain.log_size() - log_folding_factor;

    let queried_indices: Vec<u64> = (0..num_queries)
        .map(|_| challenger.sample_bits(log_query_domain_size) as u64)
        .unique()
        .collect();

    // Verify proof of work
    if !challenger.check_witness(pow_bits.ceil() as usize, pow_witness) {
        return Err(FullRoundVerificationError::ProofOfWork);
    }

    // Update the transcript with the coefficients of the answer and shake polynomials
    ans_polynomial
        .coeffs()
        .iter()
        .for_each(|&c| challenger.observe_ext_element(c));
    shake_polynomial
        .coeffs()
        .iter()
        .for_each(|&c| challenger.observe_ext_element(c));

    let shake_randomness: EF = challenger.sample_ext_element();

    // Verify Merkle paths
    for (&i, (leaf, proof)) in queried_indices.iter().unique().zip(query_proofs.iter()) {
        if config
            .mmcs_config()
            .verify_batch(
                &prev_root,
                // NP TODO verify this is correct
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
        &domain,
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
    if ans_polynomial.degree() >= quotient_answers.len() {
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
        round: round + 1,
        root: g_root,
    })
}

fn compute_f_oracle_from_g<F: TwoAdicField>(
    // Oracle relating f_i to its underlying function g_i
    oracle: &Oracle<F>,
    // Evaluations of g_i at the lists of points relevant to each queried point
    g_eval_batches: Vec<Vec<F>>,
    // The queried indices of L_i^{k_i}
    queried_indices: &[u64],
    // The domain L_i
    domain: &Radix2Coset<F>,
    // The log of the folding factor k_i
    log_folding_factor: usize,
) -> Vec<Vec<F>> {
    // 1. Computing the coset of points of L_i relevant to each queried point of L_i^k_i

    // This is the length of each coset
    let folding_factor = 1 << log_folding_factor;

    // This is the generator of each coset
    let log_scaling_factor = domain.log_size() - log_folding_factor;
    let generator = domain.generator().exp_power_of_2(log_scaling_factor);

    // The j-th element of this vector is the set of preimages of points of the
    // j-th element of L_i under the map x \mapsto x^{k_i}
    let queried_point_preimages = queried_indices
        .into_iter()
        .map(|index| {
            iterate(domain.element(*index), |&x| x * generator)
                .take(folding_factor)
                .collect_vec()
        })
        .collect_vec();

    // NP TODO this can maybe be optimized

    // Compute the values of f_i at the points using those of g_i
    queried_point_preimages
        .into_iter()
        .zip(g_eval_batches.into_iter())
        .map(|(points, g_eval_batch)| {
            points
                .into_iter()
                .zip(g_eval_batch.into_iter())
                .map(|(point, g_eval)| oracle.evaluate(point, g_eval))
                .collect_vec()
        })
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
    unfolded_evaluation_lists
        .into_iter()
        .zip(point_roots.iter())
        .map(|(evals, point_root)| fold_evaluations(evals, *point_root, log_arity, omega, c))
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
