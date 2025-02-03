use alloc::vec;
use alloc::vec::Vec;
use core::convert::TryInto;
use core::iter;
use std::collections::HashSet;

use itertools::Itertools;
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{FieldAlgebra, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;

use crate::config::RoundConfig;
use crate::coset::Radix2Coset;
use crate::polynomial::Polynomial;
use crate::proof::RoundProof;
use crate::utils::{fold_evaluations, fold_polynomial, multiply_by_power_polynomial};
use crate::{StirConfig, StirProof};

#[cfg(test)]
mod tests;

pub struct StirWitness<F: TwoAdicField, M: Mmcs<F>> {
    // The indices are given in the following frame of reference: Self is
    // produced inside prove_round for round i (in {1, ..., num_rounds}). The
    // final round, with index num_rounds + 1, does not produce a StirWitness.

    // Domain L_i
    pub(crate) domain: Radix2Coset<F>,

    // Polynomial f_i
    pub(crate) polynomial: Polynomial<F>,

    // Stacked evaluations of g_i = Fold(f_{i - 1}, ...)
    // merkle_tree above is a commitment to this
    pub(crate) stacked_evals: RowMajorMatrix<F>,

    // Merkle tree whose leaves are stacked_evals
    pub(crate) merkle_tree: M::ProverData<RowMajorMatrix<F>>,

    // Round number i
    pub(crate) round: usize,

    // Folding randomness r_i to be used in the next round
    pub(crate) folding_randomness: F,
    // Exceptionally, the first witness (passed to prove_round for round 1) is to be understood as follows:
    // - domain: L_0 = w * <w> = <w>
    // - polynomial: f_0
    // - stacked_evals: stacked evals of f_0
    // - merkle_tree = commitment to staked_evals
    // - round = 0
    // NP TODO remove comment in brackets if this changes to an option
    // - folding_randomness: r_0 (set to 1 in commit, then overwritten)
}

// NP TODO maybe have this and prove() receive &polynomial instead
pub fn commit<F, M>(
    config: &StirConfig<F, M>,
    polynomial: Polynomial<F>,
) -> (StirWitness<F, M>, M::Commitment)
where
    F: TwoAdicField,
    M: Mmcs<F>,
{
    let log_size = config.log_starting_degree() + config.log_starting_inv_rate();

    // Initial domain L_0. The chosen sequence of domains is:
    // - L_0 = w * <w> = <w>
    // - L_1 = w * <w^2>
    // - L_2 = w * <w^4>
    // ...
    // - L_i = w * <w^{2^i}>
    // This guarantees that, for all i >= 0, (L_i)^{2^l_i} doesn't intersect
    // L_{i + 1} (where l_i > 0 is the log_folding_factor of the i-th round),
    // as required for the optimisation mentioned in the paper (avoiding the use
    // of the Fill polynomials).
    // Defining L_0 with shift w or 1 is equivalent mathematically, but the
    // former allows one to always use shrink_subgroup in the next rounds.
    let domain = Radix2Coset::new(F::two_adic_generator(log_size), log_size);

    let evals = domain.evaluate_polynomial(&polynomial);

    // NP TODO create function to collate which only moves stuff around in memory once
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
            stacked_evals,
            round: 0,
            // NP TODO handle more elegantly? Use Option<F>
            folding_randomness: F::ONE,
        },
        commitment,
    )
}

// NP TODO pub fn prove_on_evals
// NP TODO commit_and_prove
pub fn prove<F, M, C>(
    config: &StirConfig<F, M>,
    polynomial: Polynomial<F>,
    challenger: &mut C,
) -> StirProof<F, M, C::Witness>
where
    F: TwoAdicField,
    M: Mmcs<F>,
    C: FieldChallenger<F> + GrindingChallenger + CanObserve<M::Commitment>,
{
    assert!(
        polynomial.degree() - 1
            <= 1 << (config.log_starting_degree() + config.log_starting_inv_rate())
    );

    // NP TODO: Should the prover call commit like in Plonky3's FRI?
    // or should be called separately like in Giacomo's code?
    let (mut witness, commitment) = commit(config, polynomial);

    // Observe the commitment
    challenger.observe(commitment.clone());
    let folding_randomness = challenger.sample_ext_element();

    // NP TODO: Handle more elegantly?
    witness.folding_randomness = folding_randomness;

    let mut round_proofs = vec![];
    for _ in 0..config.num_rounds() - 1 {
        let (new_witness, round_proof) = prove_round(config, witness, challenger);
        witness = new_witness;
        round_proofs.push(round_proof);
    }

    // Final round
    let log_last_folding_factor = config.log_last_folding_factor();

    // p in the article
    let final_polynomial = fold_polynomial(
        &witness.polynomial,
        witness.folding_randomness,
        log_last_folding_factor,
    );

    let final_queries = config.final_num_queries();

    // Logarithm of |(L_M)^(k_M)|
    let log_query_domain_size = witness.domain.log_size() - log_last_folding_factor;

    let queried_indices: Vec<u64> = (0..final_queries)
        .map(|_| challenger.sample_bits(log_query_domain_size) as u64)
        .unique()
        .collect();

    let queries_to_final: Vec<(Vec<F>, M::Proof)> = queried_indices
        .into_iter()
        .map(|index| {
            config
                .mmcs_config()
                .open_batch(index as usize, &witness.merkle_tree)
        })
        .map(|(mut k, v)| (k.remove(0), v))
        .collect();

    println!("Grinding {} bits", config.final_pow_bits().ceil() as usize);

    // NP TODO: Is this correct? Can we just take the ceil?
    // NP TODO reintroduce
    // let pow_witness = challenger.grind(config.final_pow_bits().ceil() as usize);
    let pow_witness = C::Witness::ONE;

    StirProof {
        commitment,
        round_proofs,
        final_polynomial,
        pow_witness,
        final_round_queries: queries_to_final,
    }
}

fn prove_round<F, M, C>(
    config: &StirConfig<F, M>,
    witness: StirWitness<F, M>,
    challenger: &mut C,
) -> (StirWitness<F, M>, RoundProof<F, M, C::Witness>)
where
    F: TwoAdicField,
    M: Mmcs<F>,
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

    let StirWitness {
        domain,
        polynomial,
        merkle_tree,
        stacked_evals,
        round,
        folding_randomness,
    } = witness;

    // ========= FOLDING =========

    // NP TODO ask This folding factor uses the folding factor for this round.
    // The stacking a few lines below ("new_stacked_evals =
    // RowMajorMatrix::new(folded_evals, 1 << log_folding_factor)") uses the
    // folding factor of the next round. Correct? Giacomo's code is not very
    // well suited for this since only one folding factor is passed

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
    // NP TODO create function to collate which only moves stuff around in memory once
    let new_stacked_evals = RowMajorMatrix::new(
        folded_evals,
        1 << (new_domain.log_size() - log_next_folding_factor),
    )
    .transpose();

    let (new_commitment, new_merkle_tree) = config
        .mmcs_config()
        .commit_matrix(new_stacked_evals.clone());

    challenger.observe(new_commitment.clone());

    // ========= OOD SAMPLING =========

    // NP TODO: Sample from the extension field like in FRI

    let mut ood_samples = Vec::new();

    while ood_samples.len() < num_ood_samples {
        let el: F = challenger.sample_ext_element();
        if !new_domain.contains(el) {
            ood_samples.push(el);
        }
    }

    // Evaluate the polynomial at the OOD samples
    let betas: Vec<F> = ood_samples
        .iter()
        .map(|x| folded_polynomial.evaluate(x))
        .collect();

    // Observe the betas
    challenger.observe_slice(&betas);

    // ========= STIR MESSAGE =========

    // Sample ramdomness for degree correction
    let comb_randomness = challenger.sample_ext_element();

    // Sample folding randomness for the next round
    let new_folding_randomness = challenger.sample_ext_element();

    // Sample queried indices of elements in L_{i - 1}^k_{i - 1}
    let log_query_domain_size = domain.log_size() - log_folding_factor;

    let queried_indices: Vec<u64> = (0..num_queries)
        .map(|_| challenger.sample_bits(log_query_domain_size) as u64)
        .unique()
        .collect();

    // Proof of work witness
    // NP TODO: Is this correct? Can we just take the ceil?
    // NP TODO unsafe cast to usize
    println!("Grinding {} bits", pow_bits.ceil() as usize);
    // NP TODO reintroduce
    //let pow_witness = challenger.grind(pow_bits.ceil() as usize);
    let pow_witness = C::Witness::ONE;

    // ========= QUERY PROOFS =========

    // Open the Merkle paths for the queried indices
    let query_proofs: Vec<(Vec<F>, M::Proof)> = queried_indices
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

    // NP TODO revise FS in general

    // NP TODO ask Giacomo: is this division (prover step 5) computed before or
    // after the verifier queries f_{i - 1} (verifier step 1)? The protocol is
    // interactive but the order of the interaction is not shown in the paper,
    // yet it is important for FS

    // Compute the domain L_{i-1}^k = w^k * <w^{2^{i-1} * k}>
    let domain_k = domain.shrink_coset(log_folding_factor);

    // Get the elements in L^k corresponding to the queried indices
    // (i.e r^{shift}_i in the paper)
    // Evaluate the polynomial at the queried indices
    let stir_randomness: Vec<F> = queried_indices
        .iter()
        .map(|index| domain_k.element(*index))
        .collect();

    let stir_randomness_evals: Vec<F> = stir_randomness
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

    // Compute the answer polynomial and add it to the transcript
    let ans_polynomial = Polynomial::<F>::lagrange_interpolation(quotient_answers.clone());
    challenger.observe_slice(ans_polynomial.coeffs());

    // Compute the shake polynomial and add it to the transcript
    let shake_polynomial = compute_shake_polynomial(&ans_polynomial, quotient_answers.into_iter());
    challenger.observe_slice(shake_polynomial.coeffs());

    // Shake randomness This is only used by the verifier, but it doesn't need
    // to be kept private. Therefore, the verifier can squeeze it from the
    // sponge, in which case the prover must follow suit to keep the sponges
    // in sync.
    let _shake_randomness: F = challenger.sample_ext_element();

    // Compute the quotient polynomial
    let vanishing_polynomial = Polynomial::vanishing_polynomial(quotient_set);
    let quotient_polynomial = &(&folded_polynomial - &ans_polynomial) / &vanishing_polynomial;

    // Degree-correct by multiplying by the scaling polynomial, 1 + rx + r^2 x^2 + ... + r^n x^n with n = |quotient_set|
    let witness_polynomial =
        multiply_by_power_polynomial(&quotient_polynomial, comb_randomness, quotient_set_size);

    // NP TODO remove/fix
    if quotient_polynomial.is_zero() {
        dbg!("Warning: quotient polynomial is zero. Reconsider your parameters");
    } else {
        assert_eq!(witness_polynomial.degree(), folded_polynomial.degree());
    }

    (
        StirWitness {
            domain: new_domain,
            polynomial: witness_polynomial,
            merkle_tree: new_merkle_tree,
            stacked_evals: new_stacked_evals,
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

// NP TODO domain separation in sponge

// NP TODO when evaluating in the original domain w * <w>, detect this and evaluate over <w>, then cyclically shift (make sure this is correct)
