use alloc::vec;
use alloc::vec::Vec;

use itertools::Itertools;
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;

use crate::{
    config::{observe_public_parameters, RoundConfig},
    proof::RoundProof,
    utils::{fold_polynomial, multiply_by_power_polynomial, observe_ext_slice_with_size},
    Messages, StirConfig, StirProof,
};
use p3_coset::TwoAdicCoset;
use p3_poly::Polynomial;

#[cfg(test)]
mod tests;

/// Witness for the STIR protocol produced by the `commit` method.
pub struct StirWitness<F: TwoAdicField, M: Mmcs<F>> {
    // Domain L_0
    pub(crate) domain: TwoAdicCoset<F>,

    // Polynomial f_0 which was committed to
    pub(crate) polynomial: Polynomial<F>,

    // Merkle tree whose leaves are stacked_evals
    pub(crate) merkle_tree: M::ProverData<RowMajorMatrix<F>>,
}

// Witness enriched with additional information (round number and folding
// randomness) received and produced by the method prove_round
struct StirRoundWitness<F: TwoAdicField, M: Mmcs<F>> {
    // The indices are given in the following frame of reference: Self is
    // produced inside prove_round for round i (in {1, ..., M}).
    // final round, with index M + 1, does not produce a StirRoundWitness.

    // Domain L_i
    pub(crate) domain: TwoAdicCoset<F>,

    // Polynomial f_i
    pub(crate) polynomial: Polynomial<F>,

    // Merkle tree whose leaves are stacked_evals
    pub(crate) merkle_tree: M::ProverData<RowMajorMatrix<F>>,

    // Round number i
    pub(crate) round: usize,

    // Folding randomness r_i to be used in the next round
    pub(crate) folding_randomness: F,
}

pub fn commit<F, M>(
    config: &StirConfig<F, M>,
    // afsdfsfdsdfsfssd
    polynomial: Polynomial<F>,
) -> (StirWitness<F, M>, M::Commitment)
where
    F: TwoAdicField,
    M: Mmcs<F>,
{
    assert!(polynomial
        .degree()
        .is_none_or(|d| d < 1 << (config.log_starting_degree() + config.log_starting_inv_rate())));

    let log_size = config.log_starting_degree() + config.log_starting_inv_rate();

    // Initial domain L_0. The chosen sequence of domains is:
    //   - L_0 = w * <w> = <w>
    //   - L_1 = w * <w^2>
    //   - L_2 = w * <w^4>
    //     ...
    //   - L_i = w * <w^{2^i}>
    // This guarantees that, for all i >= 0, (L_i)^{2^l_i} doesn't intersect
    // L_{i + 1} (where l_i > 0 is the log_folding_factor of the i-th round),
    // as required for the optimisation mentioned in the article (i. e. avoiding
    // the use of the Fill polynomials).
    //
    // N.B.: Defining L_0 with shift w or 1 is equivalent mathematically, but
    // the former allows one to always use the method shrink_subgroup in the
    // following rounds.
    let domain = TwoAdicCoset::new(F::two_adic_generator(log_size), log_size);

    // NP TODO check if this is cheaper because of the unnecessary shift
    let evals = domain.evaluate_polynomial(&polynomial);

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

pub fn prove<F, EF, M, C>(
    config: &StirConfig<EF, M>,
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
    // Observe public parameters
    observe_public_parameters(config.parameters(), challenger);

    // Observe the commitment
    challenger.observe(F::from_canonical_u8(Messages::Commitment as u8));
    challenger.observe(commitment.clone());

    // Sample the folding randomness
    challenger.observe(F::from_canonical_u8(Messages::FoldingRandomness as u8));
    let folding_randomness = challenger.sample_ext_element();

    let mut witness = StirRoundWitness {
        domain: witness.domain,
        polynomial: witness.polynomial,
        merkle_tree: witness.merkle_tree,
        round: 0,
        folding_randomness,
    };

    let mut round_proofs = vec![];
    for _ in 0..config.num_rounds() - 1 {
        let (new_witness, round_proof) = prove_round(config, witness, challenger);
        witness = new_witness;
        round_proofs.push(round_proof);
    }

    // Final round
    let log_last_folding_factor = config.log_last_folding_factor();

    // p in the article, which could also be understood as g_{M + 1}
    let final_polynomial = fold_polynomial(
        &witness.polynomial,
        witness.folding_randomness,
        log_last_folding_factor,
    );

    let final_queries = config.final_num_queries();

    // Logarithm of |(L_M)^(k_M)|
    let log_query_domain_size = witness.domain.log_size() - log_last_folding_factor;

    // Absorb the final polynomial
    challenger.observe(F::from_canonical_u8(Messages::FinalPolynomial as u8));
    observe_ext_slice_with_size(challenger, final_polynomial.coeffs());

    // Sample the queried indices
    challenger.observe(F::from_canonical_u8(Messages::FinalQueryIndices as u8));
    let queried_indices: Vec<u64> = (0..final_queries)
        .map(|_| challenger.sample_bits(log_query_domain_size) as u64)
        .unique()
        .collect();

    let queries_to_final: Vec<(Vec<EF>, M::Proof)> = queried_indices
        .into_iter()
        .map(|index| {
            config
                .mmcs_config()
                .open_batch(index as usize, &witness.merkle_tree)
        })
        .map(|(mut k, v)| (k.remove(0), v))
        .collect();

    // NP TODO: Is this correct? Can we just take the ceil?
    let pow_witness = challenger.grind(config.final_pow_bits().ceil() as usize);

    StirProof {
        commitment,
        round_proofs,
        final_polynomial,
        pow_witness,
        final_round_queries: queries_to_final,
    }
}

fn prove_round<F, EF, M, C>(
    config: &StirConfig<EF, M>,
    witness: StirRoundWitness<EF, M>,
    challenger: &mut C,
) -> (StirRoundWitness<EF, M>, RoundProof<EF, M, C::Witness>)
where
    F: Field,
    EF: TwoAdicField + ExtensionField<F>,
    M: Mmcs<EF>,
    C: FieldChallenger<F> + GrindingChallenger + CanObserve<M::Commitment>,
{
    // De-structure the round-specific configuration and the witness
    let RoundConfig {
        log_folding_factor,
        log_next_folding_factor,
        // NP TODO why is this not used?
        log_evaluation_domain_size,
        pow_bits,
        num_queries,
        num_ood_samples,
        // NP TODO why is this not used?
        log_inv_rate,
    } = config.round_config(witness.round).clone();

    let StirRoundWitness {
        domain,
        polynomial,
        merkle_tree,
        round,
        folding_randomness,
    } = witness;

    // ========= FOLDING =========

    // Fold the polynomial and the evaluations
    let folded_polynomial = fold_polynomial(&polynomial, folding_randomness, log_folding_factor);

    // Compute the i-th domain L_i = w * <w^{2^i}>
    let new_domain = domain.shrink_subgroup(1);

    // NP TODO can this be done more efficiently using stacked_evals? If not,
    // remove stacked_evals from the witness?
    let folded_evals = new_domain.evaluate_polynomial(&folded_polynomial);

    // Stack the new folded evaluations, commit and observe the commitment (in
    // preparation for next-round folding verification and hence with the
    // folding factor of the next round)
    let new_stacked_evals = RowMajorMatrix::new(
        folded_evals,
        1 << (new_domain.log_size() - log_next_folding_factor),
    )
    .transpose();

    let (new_commitment, new_merkle_tree) = config
        .mmcs_config()
        .commit_matrix(new_stacked_evals.clone());

    // Absorb the commitment
    challenger.observe(F::from_canonical_u8(Messages::RoundCommitment as u8));
    challenger.observe(new_commitment.clone());

    // ========= OOD SAMPLING =========

    let mut ood_samples = Vec::new();

    challenger.observe(F::from_canonical_u8(Messages::OodSamples as u8));
    while ood_samples.len() < num_ood_samples {
        let el: EF = challenger.sample_ext_element();
        if !new_domain.contains(el) {
            ood_samples.push(el);
        }
    }

    // Evaluate the polynomial at the OOD samples
    let betas: Vec<EF> = ood_samples
        .iter()
        .map(|x| folded_polynomial.evaluate(x))
        .collect();

    // Observe the betas
    challenger.observe(F::from_canonical_u8(Messages::Betas as u8));
    betas
        .iter()
        .for_each(|&beta| challenger.observe_ext_element(beta));

    // ========= STIR MESSAGE =========

    // Sample ramdomness for degree correction
    challenger.observe(F::from_canonical_u8(Messages::CombRandomness as u8));
    let comb_randomness = challenger.sample_ext_element();

    // Sample folding randomness for the next round
    challenger.observe(F::from_canonical_u8(Messages::FoldingRandomness as u8));
    let new_folding_randomness = challenger.sample_ext_element();

    // Sample queried indices of elements in L_{i - 1}^k_{i - 1}
    let log_query_domain_size = domain.log_size() - log_folding_factor;

    challenger.observe(F::from_canonical_u8(Messages::QueryIndices as u8));
    let queried_indices: Vec<usize> = (0..num_queries)
        .map(|_| challenger.sample_bits(log_query_domain_size))
        .unique()
        .collect();

    // Proof-of-work witness
    // NP TODO: Is this correct? Can we just take the ceil?
    // NP TODO unsafe cast to usize
    let pow_witness = challenger.grind(pow_bits.ceil() as usize);

    // ========= QUERY PROOFS =========

    // Open the Merkle paths for the queried indices
    let query_proofs: Vec<(Vec<EF>, M::Proof)> = queried_indices
        .clone()
        .into_iter()
        .map(|index| {
            config
                .mmcs_config()
                .open_batch(index as usize, &merkle_tree)
        })
        .map(|(mut k, v)| (k.remove(0), v))
        .collect();

    // ========= POLY QUOTIENT =========

    // Compute the domain L_{i-1}^k = w^k * <w^{2^{i-1} * k}>
    let mut domain_k = domain.shrink_coset(log_folding_factor);

    // Get the elements in L^k corresponding to the queried indices
    // (i.e r^{shift}_i in the paper)
    // Evaluate the polynomial at the queried indices
    let stir_randomness: Vec<EF> = queried_indices
        .iter()
        .map(|index| domain_k.element(*index))
        .collect();

    let stir_randomness_evals: Vec<EF> = stir_randomness
        .iter()
        .map(|x| folded_polynomial.evaluate(x))
        .collect();

    // Stir answers has (implicitly) been dedup-ed, whereas beta_answers has not yet
    let stir_answers = stir_randomness.into_iter().zip(stir_randomness_evals);
    let beta_answers = ood_samples
        .into_iter()
        .zip(betas.clone())
        .into_iter()
        .unique();
    let quotient_answers = beta_answers.chain(stir_answers).collect_vec();

    // Compute the quotient set, i.e \mathcal{G}_i in the paper
    let quotient_set = quotient_answers.iter().map(|(x, _)| *x).collect_vec();
    let quotient_set_size = quotient_set.len();

    // NP TODO if quotient_set_size is > deg + 1, terminate early or at least handle accordingly - otherwise panics can happen

    // Compute the answer polynomial and add it to the transcript
    let ans_polynomial = Polynomial::<EF>::lagrange_interpolation(quotient_answers.clone());
    challenger.observe(F::from_canonical_u8(Messages::AnsPolynomial as u8));
    observe_ext_slice_with_size(challenger, ans_polynomial.coeffs());

    // Compute the shake polynomial and add it to the transcript
    let shake_polynomial = compute_shake_polynomial(&ans_polynomial, quotient_answers.into_iter());
    challenger.observe(F::from_canonical_u8(Messages::ShakePolynomial as u8));
    observe_ext_slice_with_size(challenger, shake_polynomial.coeffs());

    // Shake randomness This is only used by the verifier, but it doesn't need
    // to be kept private. Therefore, the verifier can squeeze it from the
    // sponge, in which case the prover must follow suit to keep the sponges
    // in sync.
    challenger.observe(F::from_canonical_u8(Messages::ShakeRandomness as u8));
    let _shake_randomness: EF = challenger.sample_ext_element();

    // Compute the quotient polynomial
    let vanishing_polynomial = Polynomial::vanishing_polynomial(quotient_set);
    let quotient_polynomial = &(&folded_polynomial - &ans_polynomial) / &vanishing_polynomial;

    // Degree-correct by multiplying by the scaling polynomial, 1 + rx + r^2 x^2 + ... + r^n x^n with n = |quotient_set|
    let witness_polynomial =
        multiply_by_power_polynomial(&quotient_polynomial, comb_randomness, quotient_set_size);

    // NP TODO remove/fix
    if quotient_polynomial.is_zero() {
        // This happens when the quotient set has deg(g_i) + 1 elements or more,
        // in which case the interpolator ans_i coincides with g_i, causing
        // f_i to be 0. A potential cause is the combination of a small field,
        // low-degree initial polynomial and large number of security bits
        // required. This does not make the protocol vulnerable, but perhaps
        // less efficient than it could be.
        dbg!("Warning: quotient polynomial is zero");
    } else {
        assert_eq!(witness_polynomial.degree(), folded_polynomial.degree());
    }

    (
        StirRoundWitness {
            domain: new_domain,
            polynomial: witness_polynomial,
            merkle_tree: new_merkle_tree,
            folding_randomness: new_folding_randomness,
            round: round + 1,
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

fn compute_shake_polynomial<F: TwoAdicField>(
    ans_polynomial: &Polynomial<F>,
    quotient_answers: impl Iterator<Item = (F, F)>,
) -> Polynomial<F> {
    let mut shake_polynomial = Polynomial::zero();
    for (x, y) in quotient_answers {
        let numerator = ans_polynomial - &y;
        let denominator = Polynomial::vanishing_linear_polynomial(x);
        shake_polynomial = &shake_polynomial + &(&numerator / &denominator);
    }
    shake_polynomial
}

// NP TODO when evaluating in the original domain w * <w>, detect this and evaluate over <w>, then cyclically shift (make sure this is correct)
