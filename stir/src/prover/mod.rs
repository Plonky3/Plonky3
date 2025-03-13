use alloc::vec;
use alloc::vec::Vec;

use itertools::Itertools;
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_coset::TwoAdicCoset;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_poly::Polynomial;

use crate::config::{observe_public_parameters, RoundConfig};
use crate::proof::RoundProof;
use crate::utils::{fold_polynomial, multiply_by_power_polynomial, observe_ext_slice_with_size};
use crate::{Messages, StirConfig, StirProof, POW_BITS_WARNING};

#[cfg(test)]
mod tests;

/// Prover witness for the STIR protocol produced by the [`commit`] method.
pub struct StirWitness<F: TwoAdicField, M: Mmcs<F>> {
    // Domain L_0
    pub(crate) domain: TwoAdicCoset<F>,

    // Polynomial f_0 which was committed to
    pub(crate) polynomial: Polynomial<F>,

    // Merkle tree whose leaves are the stacked evaluations of f_0. Its root is
    // the commitment shared with the verifier.
    pub(crate) merkle_tree: M::ProverData<RowMajorMatrix<F>>,
}

// STIR witness enriched with additional information (round number and folding
// randomness) received and produced by the method prove_round
pub(crate) struct StirRoundWitness<F: TwoAdicField, M: Mmcs<F>> {
    // The indices are given in the following frame of reference: Self is
    // produced inside prove_round for round i (in {1, ..., M}). The final
    // round, with index M + 1, does not produce a StirRoundWitness.

    // Domain L_i. The chosen sequence of domains L_0, L_1, ... is documented
    // at the start of the method commit.
    pub(crate) domain: TwoAdicCoset<F>,

    // Polynomial f_i
    pub(crate) polynomial: Polynomial<F>,

    // Merkle tree whose leaves are the stacked evaluations of g_i
    pub(crate) merkle_tree: M::ProverData<RowMajorMatrix<F>>,

    // Round number i
    pub(crate) round: usize,

    // Folding randomness r_i to be used *in the next round* (g_{i + 1} will be
    // the folding of f_i with randomness r_i)
    pub(crate) folding_randomness: F,
}

/// Commit to the initial polynomial `f_0` whose low-degreeness is being
/// asserted. Returns the witness for the prover and the commitment to the
/// evaluations to be shared with the verifier.
///
/// # Parameters
///
/// - `config`: Full STIR configuration
/// - `polynomial`: Initial polynomial `f_0`
///
/// # Panics
///
/// Panics if the degree of `polynomial` is too large (the configuration supports
/// degree at most `2^{config.log_starting_degree()} - 1`).
pub fn commit<F, M>(
    config: &StirConfig<M>,
    polynomial: Polynomial<F>,
) -> (StirWitness<F, M>, M::Commitment)
where
    F: TwoAdicField,
    M: Mmcs<F>,
{
    assert!(
        polynomial
            .degree()
            .is_none_or(|d| d < (1 << config.log_starting_degree())),
        "The degree of the polynomial ({}) is too large: the configuration \
        only supports polynomials of degree up to 2^{} - 1 = {}",
        polynomial.degree().unwrap(),
        config.log_starting_degree(),
        (1 << config.log_starting_degree()) - 1
    );

    let log_size = config.log_starting_degree() + config.log_starting_inv_rate();

    // Initial domain L_0. The chosen sequence of domains is:
    //   - L_0 = w * <w> = <w>
    //   - L_1 = w * <w^2>
    //   - L_2 = w * <w^4> ...
    //   - L_i = w * <w^{2^i}> This guarantees that, for all i >= 0, (L_i)^{k_i}
    // is disjoint from L_{i + 1} (where k_i is the folding factor of the i-th
    // round), as required for the optimisation mentioned in the article (i. e.
    // avoiding the use of the Fill polynomials).
    //
    // N.B.: Defining L_0 with shift w or 1 is equivalent mathematically, but
    // the former allows one to always use the method shrink_subgroup in the
    // following rounds. This shift does not cause significant extra work in
    // coset.evaluate as it is treated as a special case therein.
    let mut domain = TwoAdicCoset::new(F::two_adic_generator(log_size), log_size);

    // Committing to the evaluations of f_0 over L_0.
    let evals = domain.evaluate_polynomial(polynomial.coeffs().to_vec());

    // The stacking width is
    //   k_0 = 2^{log_size - config.log_starting_folding_factor},
    // which facilitates opening values so that the prover can verify the first
    // folding
    let stacked_evals = RowMajorMatrix::new(
        evals,
        1 << (log_size - config.log_starting_folding_factor()),
    )
    .transpose();

    let (commitment, merkle_tree) = config.mmcs_config().commit_matrix(stacked_evals.clone());

    (
        StirWitness {
            domain,
            polynomial,
            merkle_tree,
        },
        commitment,
    )
}

/// Prove that the committed polynomial satisfies the low-degreeness bound
/// specified in the configuration.
///
/// # Parameters
///
/// - `config`: Full STIR configuration, including the degree bound
/// - `witness`: Witness for the prover containing the polynomial and MMCS
///   prover data
/// - `commitment`: Commitment to the evaluations of the polynomial over L_0
/// - `challenger`: Challenger which produces the transcript of the
///   Fiat-Shamired interaction
pub fn prove<F, EF, M, C>(
    config: &StirConfig<M>,
    witness: StirWitness<EF, M>,
    commitment: M::Commitment,
    challenger: &mut C,
) -> StirProof<EF, M, C::Witness>
where
    F: Field,
    EF: TwoAdicField + ExtensionField<F>,
    M: Mmcs<EF>,
    C: FieldChallenger<F> + GrindingChallenger + CanObserve<M::Commitment>,
{
    // Inform the prover if the configuration requires a proof of work larger
    // than the POW_BITS_WARNING constant. This is only logged if the tracing
    // module has been init()ialised.
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

    // Observe the commitment
    challenger.observe(F::from_u8(Messages::Commitment as u8));
    challenger.observe(commitment.clone());

    // Sample the folding randomness r_0
    challenger.observe(F::from_u8(Messages::FoldingRandomness as u8));
    let folding_randomness: EF = challenger.sample_algebra_element();

    // Enriching the initial witness into a full round witness that prove_round
    // can receive.
    let mut witness = StirRoundWitness {
        domain: witness.domain,
        polynomial: witness.polynomial,
        merkle_tree: witness.merkle_tree,
        round: 0,
        folding_randomness,
    };

    // Prove each full round i = 1, ..., M of the protocol
    let mut round_proofs = vec![];
    for _ in 1..=config.num_rounds() - 1 {
        let (new_witness, round_proof) = prove_round(config, witness, challenger);

        witness = new_witness;
        round_proofs.push(round_proof);
    }

    // Final round i = M + 1
    let log_last_folding_factor = config.log_last_folding_factor();

    // Computing the final polynomial p in the notation of the article (which we
    // also refer to as g_{M + 1})
    let final_polynomial = fold_polynomial(
        &witness.polynomial,
        witness.folding_randomness,
        log_last_folding_factor,
    );

    let final_queries = config.final_num_queries();

    // Logarithm of |(L_M)^(k_M)|
    let log_query_domain_size = witness.domain.log_size() - log_last_folding_factor;

    // Observe the final polynomial g_{M + 1}
    challenger.observe(F::from_u8(Messages::FinalPolynomial as u8));
    observe_ext_slice_with_size(challenger, final_polynomial.coeffs());

    // Sample the indices to query verify the folding of f_M into g_{M + 1} at
    challenger.observe(F::from_u8(Messages::FinalQueryIndices as u8));
    let queried_indices: Vec<u64> = (0..final_queries)
        .map(|_| challenger.sample_bits(log_query_domain_size) as u64)
        .unique()
        .collect();

    // Opening the cosets of evaluations of g_M at each k_M-th root of the
    // points queried
    let queries_to_final: Vec<(Vec<EF>, M::Proof)> = queried_indices
        .into_iter()
        .map(|index| {
            config
                .mmcs_config()
                .open_batch(index as usize, &witness.merkle_tree)
        })
        .map(|(mut k, v)| (k.remove(0), v))
        .collect();

    // Compute the proof-of-work for the final round
    let pow_witness = challenger.grind(config.final_pow_bits());

    StirProof {
        round_proofs,
        final_polynomial,
        pow_witness,
        final_round_queries: queries_to_final,
    }
}

/// Prove a single full round, taking in a witness for the previous round and
/// returning a witness for the new one as well as the round proof.
pub(crate) fn prove_round<F, EF, M, C>(
    // Full STIR configuration from which the round-specific configuration is
    // extracted
    config: &StirConfig<M>,
    // Witness for the previous round (referring to f_{i - 1} if this is round i)
    witness: StirRoundWitness<EF, M>,
    // FS challenger
    challenger: &mut C,
) -> (StirRoundWitness<EF, M>, RoundProof<EF, M, C::Witness>)
where
    F: Field,
    EF: TwoAdicField + ExtensionField<F>,
    M: Mmcs<EF>,
    C: FieldChallenger<F> + GrindingChallenger + CanObserve<M::Commitment>,
{
    let round = witness.round + 1;

    // De-structure the round-specific configuration and the witness
    let RoundConfig {
        log_folding_factor,
        log_next_folding_factor,
        pow_bits,
        num_queries,
        num_ood_samples,
        ..
    } = config.round_config(round).clone();

    let StirRoundWitness {
        domain,
        polynomial,
        merkle_tree,
        folding_randomness,
        ..
    } = witness;

    // ================================ Folding ================================

    // Obtain g_i as the folding of f_{i - 1}
    let folded_polynomial = fold_polynomial(&polynomial, folding_randomness, log_folding_factor);

    // Compute the i-th domain L_i = w * <w^{2^i}> = w * (w^{-1} * L_{i - 1})^2
    let mut new_domain = domain.shrink_subgroup(1);

    // Evaluate g_i over L_i
    let folded_evals = new_domain.evaluate_polynomial(folded_polynomial.coeffs().to_vec());

    // Stack the evaluations, commit to them (in preparation for
    // next-round-folding verification, and therefore with width equal to the
    // folding factor of the next round) and then observe the commitment
    let new_stacked_evals = RowMajorMatrix::new(
        folded_evals,
        1 << (new_domain.log_size() - log_next_folding_factor),
    )
    .transpose();

    let (new_commitment, new_merkle_tree) = config
        .mmcs_config()
        .commit_matrix(new_stacked_evals.clone());

    // Observe the commitment
    challenger.observe(F::from_u8(Messages::RoundCommitment as u8));
    challenger.observe(new_commitment.clone());

    // ======================== Out-of-domain sampling ========================

    let mut ood_samples = Vec::new();

    challenger.observe(F::from_u8(Messages::OodSamples as u8));
    while ood_samples.len() < num_ood_samples {
        let el: EF = challenger.sample_algebra_element();
        if !new_domain.contains(el) {
            ood_samples.push(el);
        }
    }

    // Evaluate the polynomial at the out-of-domain sampled points
    let betas: Vec<EF> = ood_samples
        .iter()
        .map(|x| folded_polynomial.evaluate(x))
        .collect();

    // Observe the evaluations
    challenger.observe(F::from_u8(Messages::Betas as u8));
    betas
        .iter()
        .for_each(|&beta| challenger.observe_algebra_element(beta));

    // ========================== Sampling randomness ==========================

    // Sample ramdomness for degree correction
    challenger.observe(F::from_u8(Messages::CombRandomness as u8));
    let comb_randomness = challenger.sample_algebra_element();

    // Sample folding randomness for the next round
    challenger.observe(F::from_u8(Messages::FoldingRandomness as u8));
    let new_folding_randomness = challenger.sample_algebra_element();

    // Sample queried indices of elements in L_{i - 1}^k_{i - 1}
    let log_query_domain_size = domain.log_size() - log_folding_factor;

    challenger.observe(F::from_u8(Messages::QueryIndices as u8));
    let queried_indices: Vec<usize> = (0..num_queries)
        .map(|_| challenger.sample_bits(log_query_domain_size))
        .unique()
        .collect();

    // Compute the proof-of-work for the current round
    let pow_witness = challenger.grind(pow_bits);

    // ======================= Open queried evaluations =======================

    // Open the Merkle paths for the queried indices
    let query_proofs: Vec<(Vec<EF>, M::Proof)> = queried_indices
        .clone()
        .into_iter()
        .map(|index| config.mmcs_config().open_batch(index, &merkle_tree))
        .map(|(mut k, v)| (k.remove(0), v))
        .collect();

    // ============= Computing the Quot, Ans and shake polynomials =============

    // Compute the domain L_{i - 1}^{k_{i - 1}}
    let mut domain_k = domain.shrink_coset(log_folding_factor);

    // Get the domain elements at the queried indices (i.e r^shift_i in the paper)
    let stir_randomness: Vec<EF> = queried_indices
        .iter()
        .map(|index| domain_k.element(*index))
        .collect();

    // Evaluate the polynomial at those points
    let stir_randomness_evals: Vec<EF> = stir_randomness
        .iter()
        .map(|x| folded_polynomial.evaluate(x))
        .collect();

    // stir_answers has been dedup-ed but beta_answers has not yet:
    let stir_answers = stir_randomness.into_iter().zip(stir_randomness_evals);
    let beta_answers = ood_samples.into_iter().zip(betas.clone()).unique();
    let quotient_answers = beta_answers.chain(stir_answers).collect_vec();

    // Compute the quotient set, \mathcal{G}_i in the notation of the article
    let quotient_set = quotient_answers.iter().map(|(x, _)| *x).collect_vec();
    let quotient_set_size = quotient_set.len();

    // Compute the Ans polynomial and add it to the transcript
    let ans_polynomial = Polynomial::<EF>::lagrange_interpolation(quotient_answers.clone());
    challenger.observe(F::from_u8(Messages::AnsPolynomial as u8));
    observe_ext_slice_with_size(challenger, ans_polynomial.coeffs());

    // Compute the shake polynomial and add it to the transcript
    let shake_polynomial = compute_shake_polynomial(&ans_polynomial, quotient_answers.into_iter());
    challenger.observe(F::from_u8(Messages::ShakePolynomial as u8));
    observe_ext_slice_with_size(challenger, shake_polynomial.coeffs());

    // Shake randomness: this is only used by the verifier, but it doesn't need
    // to be kept private. Therefore, the verifier can sample it from the
    // challenger, in which case the prover must follow suit to keep the
    // challengers in sync.
    challenger.observe(F::from_u8(Messages::ShakeRandomness as u8));
    let _shake_randomness: EF = challenger.sample_algebra_element();

    // Compute the Quot polynomial
    let vanishing_polynomial = Polynomial::vanishing_polynomial(quotient_set);
    let quotient_polynomial = &(&folded_polynomial - &ans_polynomial) / &vanishing_polynomial;

    // Correct the degree by multiplying by the scaling polynomial,
    //   1 + rx + r^2 x^2 + ... + r^n x^n
    // with n = |quotient_set|
    let witness_polynomial =
        multiply_by_power_polynomial(&quotient_polynomial, comb_randomness, quotient_set_size);

    if quotient_polynomial.is_zero() {
        // This happens when the quotient set has deg(g_i) + 1 elements or more,
        // in which case the interpolator ans_i coincides with g_i, causing f_i
        // to be 0. A potential cause is the combination of a small field,
        // low-degree initial polynomial and large number of security bits
        // required. This does not make the protocol vulnerable, but perhaps
        // less efficient than it could be. In the future, early termination can
        // be designed and implemented for this case, but this is unexplored as
        // of yet.
        tracing::info!("The quotient polynomial is zero in round {}", round);
    }

    (
        StirRoundWitness {
            domain: new_domain,
            polynomial: witness_polynomial,
            merkle_tree: new_merkle_tree,
            folding_randomness: new_folding_randomness,
            round,
        },
        RoundProof {
            g_root: new_commitment,
            betas,
            ans_polynomial,
            query_proofs,
            shake_polynomial,
            pow_witness,
        },
    )
}

// Compute the shake polynomial which allows the verifier to evaluate the Ans
// polynomial at all points which it purportedly interpolates.
fn compute_shake_polynomial<F: TwoAdicField>(
    ans_polynomial: &Polynomial<F>,
    quotient_answers: impl Iterator<Item = (F, F)>,
) -> Polynomial<F> {
    // The shake polynomial is defined as:
    //   sum_{y in quotient_answers} (ans_polynomial - y) / (x - y)
    let mut shake_polynomial = Polynomial::zero();
    for (x, y) in quotient_answers {
        let numerator = ans_polynomial - &y;
        let denominator = Polynomial::vanishing_linear_polynomial(x);
        shake_polynomial = &shake_polynomial + &(&numerator / &denominator);
    }
    shake_polynomial
}
