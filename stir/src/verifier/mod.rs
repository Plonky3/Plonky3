use alloc::vec;
use alloc::vec::Vec;
use core::convert::TryInto;
use core::iter;
use p3_matrix::Dimensions;

use itertools::iterate;
use itertools::Itertools;
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::TwoAdicField;
use p3_matrix::dense::RowMajorMatrix;

use crate::config::RoundConfig;
use crate::coset::Radix2Coset;
use crate::polynomial::Polynomial;
use crate::proof::RoundProof;
use crate::utils::{fold_polynomial, multiply_by_power_polynomial};
use crate::{StirConfig, StirProof};

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
pub struct VerificationState<F: TwoAdicField> {
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
}

pub fn verify<F, M, C>(
    config: &StirConfig<F, M>,
    commitment: M::Commitment,
    proof: StirProof<F, M, C::Witness>,
    challenger: &mut C,
) -> bool
where
    F: TwoAdicField,
    M: Mmcs<F>,
    C: FieldChallenger<F> + GrindingChallenger + CanObserve<M::Commitment>,
{
    let StirProof {
        round_proofs,
        final_polynomial,
        pow_witness,
        final_round_queries,
    } = proof;

    // NP TODO return meaningful verification error

    if final_polynomial.degree() >= 1 << config.log_stopping_degree() {
        return false;
    }

    // NP TODO verify merkle paths (inside main loop instead of separately PLUS final round)
    challenger.observe(commitment);
    let folding_randomness = challenger.sample_ext_element();

    let log_size = config.log_starting_degree() + config.log_starting_inv_rate();

    // Cf. prover/mod.rs for an explanation on the chosen sequence of domain
    // sizes
    let domain = Radix2Coset::new(F::two_adic_generator(log_size), log_size);

    let mut verification_state = VerificationState {
        oracle: Oracle::Transparent,
        domain,
        folding_randomness,
        round: 0,
    };

    for round_proof in round_proofs {
        verification_state =
            if let Some(vs) = verify_round(config, verification_state, round_proof, challenger) {
                vs
            } else {
                return false;
            };
    }

    let VerificationState {
        oracle: final_oracle,
        domain: final_domain,
        folding_randomness: final_folding_randomness,
        ..
    } = verification_state;

    // Step 2: Consistency with final polynomial

    // Logarithm of |(L_M)^k_M|
    let final_log_size = verification_state.domain.log_size() - config.log_last_folding_factor();

    let final_queried_indices: Vec<u64> = (0..config.final_num_queries())
        .map(|_| challenger.sample_bits(final_log_size) as u64)
        .unique()
        .collect();

    // Recover the evaluations of g_M needed to compute the values of f_M at
    // points which are relevant to evaluate p(r_i) = Fold(f_M, ...)(r_i), where
    // r_i runs over the final queried indices
    let g_m_evals = proof
        .final_round_queries
        .into_iter()
        .map(|(eval_batch, _)| eval_batch)
        .collect_vec();

    // The evaluations of g_M were computed and committed to in the last
    // main-loop round (round M)
    let g_m_eval_root = round_proofs.last().unwrap().g_root;

    // Verifying paths of those evaluations of g_M
    for (&i, (leaf, proof)) in final_queried_indices
        .iter()
        .unique()
        .zip(final_round_queries)
    {
        if config
            .mmcs_config()
            .verify_batch(
                &g_m_eval_root,
                // NP TODO verify this is correct
                &[Dimensions {
                    width: 1 << config.log_last_folding_factor(),
                    height: 1 << (domain.log_size() - config.log_last_folding_factor()),
                }],
                i as usize,
                &vec![leaf],
                &proof,
            )
            .is_err()
        {
            return false;
        }
    }

    // Compute the values of f_M at the relevant points given the evaluations of g_M
    let f_m_evals = compute_f_oracle_from_g(
        &final_oracle,
        g_m_evals,
        final_queried_indices,
        &final_domain,
        config.log_last_folding_factor(),
    );

    // Compute the evaluations of p (which one could call g_{M + 1}) given those of f_M
    let p_evals = compute_folded_evaluations(
        &verification_state,
        final_randomness_indexes,
        final_oracle_answers,
    );

    if !folded_answers
        .into_iter()
        .all(|(point, value)| proof.final_polynomial.evaluate(&point) == value)
    {
        return false;
    }

    if !challenger.check_witness(config.final_pow_bits().ceil() as usize, pow_witness) {
        return false;
    }

    return true;
}

fn verify_round<F, M, C>(
    config: &StirConfig<F, M>,
    verification_state: VerificationState<F>,
    round_proof: RoundProof<F, M, C::Witness>,
    challenger: &mut C,
) -> Option<VerificationState<F>>
where
    F: TwoAdicField,
    M: Mmcs<F>,
    C: FieldChallenger<F> + GrindingChallenger + CanObserve<M::Commitment>,
{
    // De-structure the round-specific configuration and the verification state
    let RoundConfig {
        log_folding_factor,
        log_next_folding_factor,
        log_evaluation_domain_size,
        pow_bits,
        num_queries,
        num_ood_samples,
        log_inv_rate,
    } = config.round_config(verification_state.round).clone();

    let VerificationState {
        oracle,
        domain,
        folding_randomness,
        round,
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
        let el: F = challenger.sample_ext_element();
        if !domain.contains(el) {
            ood_samples.push(el);
        }
    }

    // Observe the betas
    challenger.observe_slice(&betas);

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
        return None;
    }

    // Update the transcript with the coefficients of the answer and shake polynomials
    challenger.observe_slice(ans_polynomial.coeffs());
    challenger.observe_slice(shake_polynomial.coeffs());

    let shake_randomness: F = challenger.sample_ext_element();

    // Verify Merkle paths
    for (&i, (leaf, proof)) in queried_indices.iter().unique().zip(query_proofs) {
        if config
            .mmcs_config()
            .verify_batch(
                &g_root,
                // NP TODO verify this is correct
                &[Dimensions {
                    width: 1 << log_folding_factor,
                    height: 1 << (domain.log_size() - log_folding_factor),
                }],
                i as usize,
                &vec![leaf],
                &proof,
            )
            .is_err()
        {
            return None;
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
        queried_indices,
        &domain,
        log_folding_factor,
    );

    // Now, for each of the selected random points, we need to compute the folding of the
    // previous oracle
    let folded_answers =
        compute_folded_evaluations(&verification_state, queried_indices, previous_f_values);

    // The quotient definining the function
    // NP Ans_i in Verifier/Main loop/(b)
    let quotient_answers: Vec<_> = ood_samples
        .into_iter()
        .zip(&round_proof.betas)
        .map(|(alpha, beta)| (alpha, *beta))
        .chain(folded_answers)
        .collect();

    // NP TODO replace by shake-poly machinery

    // Check that Ans interpolates the expected values
    if ans_polynomial.degree() >= quotient_answers.len() {
        return None;
    }

    if quotient_answers
        .iter()
        .any(|(point, &eval)| ans_polynomial.evaluate(point) != eval)
    {
        return None;
    }

    let quotient_set = quotient_answers.into_iter().map(|(x, _)| x).collect_vec();

    // NP TODO degree-test ans poly and shake_poly

    Some(VerificationState {
        oracle: Oracle::Virtual(VirtualFunction {
            comb_randomness,
            interpolating_polynomial: ans_polynomial,
            quotient_set,
        }),
        domain: domain.shrink_subgroup(1),
        folding_randomness: new_folding_randomness,
        round: round + 1,
    })
}

fn compute_f_oracle_from_g<F: TwoAdicField>(
    // Oracle relating f_i to its underlying function g_i
    oracle: &Oracle<F>,
    // Evaluations of g_i at the lists of points relevant to each queried point
    g_eval_batches: Vec<Vec<F>>,
    // The queried indices of L_i^{k_i}
    queried_indices: Vec<u64>,
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
        .map(|point| {
            iterate(domain.element(point), |&x| x * generator)
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

fn compute_folded_evaluations<F: TwoAdicField>(
    oracle: &Oracle<F>,
    previous_f_values: Vec<Vec<F>>,
    verification_state: VerificationState<F>,
    queried_indices: Vec<u64>,
) -> Vec<F> {
    todo!()
}
