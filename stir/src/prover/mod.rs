use alloc::vec;
use alloc::vec::Vec;
use core::convert::TryInto;
use core::iter;

use itertools::Itertools;
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::TwoAdicField;
use p3_matrix::dense::RowMajorMatrix;

use crate::config::RoundConfig;
use crate::coset::Radix2Coset;
use crate::polynomial::Polynomial;
use crate::proof::RoundProof;
use crate::utils::fold_polynomial;
use crate::{StirConfig, StirProof};

#[cfg(test)]
mod tests;

pub struct StirWitness<F: TwoAdicField, M: Mmcs<F>> {
    pub(crate) domain: Radix2Coset<F>,
    pub(crate) polynomial: Polynomial<F>,
    pub(crate) merkle_tree: M::ProverData<RowMajorMatrix<F>>,
    pub(crate) stacked_evals: RowMajorMatrix<F>,
    pub(crate) round: usize,
    pub(crate) folding_randomness: F,
}

pub fn commit<F, M>(
    config: &StirConfig<M>,
    polynomial: Polynomial<F>,
) -> (StirWitness<F, M>, M::Commitment)
where
    F: TwoAdicField,
    M: Mmcs<F>,
{
    let domain = Radix2Coset::new_from_degree_and_rate(
        config.log_starting_degree(),
        config.log_starting_inv_rate(),
    );

    let evals = domain.evaluate_polynomial(&polynomial);
    let stacked_evals = RowMajorMatrix::new(evals, 1 << config.log_starting_folding_factor());
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
pub fn prove<F, M, Challenger>(
    config: &StirConfig<M>,
    polynomial: Polynomial<F>,
    challenger: &mut Challenger,
) -> StirProof<F, M, Challenger::Witness>
where
    F: TwoAdicField,
    M: Mmcs<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger + CanObserve<M::Commitment>,
{
    assert!(
        polynomial.degree() - 1
            <= 1 << (config.log_starting_degree() + config.log_starting_inv_rate())
    );

    // NP TODO: Should the prover call commit like in Plonky3's FRI?
    // or should be called separately like in Giacomo's code?
    let (mut witness, commitment) = commit(config, polynomial);

    // Observe the commitment
    challenger.observe(commitment);
    let folding_randomness = challenger.sample_ext_element();

    // NP TODO: Handle more elegantly?
    witness.folding_randomness = folding_randomness;

    let mut round_proofs = vec![];
    for _ in 0..config.num_rounds() {
        let (new_witness, round_proof) = prove_round(config, witness, challenger);
        witness = new_witness;
        round_proofs.push(round_proof);
    }

    let log_last_folding_factor = config.log_folding_factors().last().unwrap();

    let final_polynomial = fold_polynomial(
        &witness.polynomial,
        witness.folding_randomness,
        1 << log_last_folding_factor,
    );

    let final_queries = config.final_num_queries();

    let scaling_factor = 1 << (witness.domain.log_size() - log_last_folding_factor);

    // NP TODO: Unsafe cast to u64
    // NP TODO: No index deduplication
    let queried_indices: Vec<u64> = (0..final_queries)
        .map(|_| challenger.sample_bits(scaling_factor).try_into().unwrap())
        .collect();

    let queries_to_final: Vec<(Vec<Vec<F>>, M::Proof)> = queried_indices
        .iter()
        .map(|index| {
            config
                .mmcs_config()
                .open_batch(*index as usize, &witness.merkle_tree)
        })
        .collect();

    // NP TODO: Is this correct? Can we just take the ceil?
    let pow_witness = challenger.grind(config.final_pow_bits().ceil() as usize);

    StirProof {
        round_proofs,
        final_polynomial,
        pow_witness,
        queries_to_final,
    }
}

fn prove_round<F, M, Challenger>(
    config: &StirConfig<M>,
    witness: StirWitness<F, M>,
    challenger: &mut Challenger,
) -> (StirWitness<F, M>, RoundProof<F, M, Challenger::Witness>)
where
    F: TwoAdicField,
    M: Mmcs<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger + CanObserve<M::Commitment>,
{
    // De-structure the round-specific configuration and the witness
    let RoundConfig {
        log_folding_factor,
        log_next_folding_factor,
        // NP TODO why is this not used?
        log_evaluation_domain_size,
        pow_bits,
        num_queries,
        ood_samples,
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

    // NP Remove
    assert!(log_evaluation_domain_size == domain.log_size());

    // ========= FOLDING =========

    // NP TODO ask This folding factor uses the folding factor for this round.
    // The stacking a few lines below ("new_stacked_evals =
    // RowMajorMatrix::new(folded_evals, 1 << log_folding_factor)") uses the
    // folding factor of the next round. Correct? Giacomo's code is not very
    // well suited for this since only one folding factor is passed

    // NP TODO remove
    println!("PROVE_ROUND REACHES 0");

    // Fold the polynomial and the evaluations
    let folded_polynomial =
        fold_polynomial(&polynomial, folding_randomness, 1 << log_folding_factor);

    // NP TODO remove
    println!("PROVE_ROUND REACHES 1");

    // Compute L' = omega * <omega^2>
    // Shrink the evaluation domain by a factor of 2 (log_scale_factor = 1)
    // NP TODO: Does this make sense? It's equivalent to Giacomo's code but note that root_of_unity is never updated
    // So this is not really omega * <omega^2> but initial_omega * <omega^2>

    // NP TODO remove
    // domain_0 = shift * <omega>                               // o = shift                // shift * <omega>
    // domain_1 = [omega * (shift^2)] * <omega^2>               // o = omega * shift^2      // omega * shift^2 * <omega^2>
    // domain_2 = [omega * (omega^2 * shift^4)] * <omega^4>     // o = omega^3 * shift^4    // omega^3 * shift^4 * <omega^4>
    // domain_3 = [omega * (omega^6 * shift^8)] * <omega^8>     // o = omega^7 * shift^8    // omega^7 * shift^8 * <omega^8>

    // shift = 1
    // domain_1 = [omega * (shift^2)] * <omega^2>               //  omega * <omega^2>
    // domain_2 = [omega * (omega^2 * shift^4)] * <omega^4>     //  omega^3 * <omega^4>

    // NP TODO maybe keep root of unity separate
    let new_domain = domain.shrink_coset(1).shift_by_root_of_unity();

    // NP TODO can this be done more efficiently using stacked_evals? If not,
    // remove stacked_evals from the witness?
    let folded_evals = new_domain.evaluate_polynomial(&folded_polynomial);

    // Stack the new folded evaluations, commit and observe the commitment (in
    // preparation for next-round folding verification and hence with the
    // folding factor of the next round)
    let new_stacked_evals = RowMajorMatrix::new(folded_evals, 1 << log_next_folding_factor);
    let (new_commitment, new_merkle_tree) = config
        .mmcs_config()
        .commit_matrix(new_stacked_evals.clone());

    challenger.observe(new_commitment.clone());

    // NP TODO remove
    println!("PROVE_ROUND REACHES 2");

    // ========= OOD SAMPLING =========

    // NP TODO: Sample from the extension field like in FRI

    // NP TODO Ask THESE ARE NOT OUT OF THE DOMAIN!
    let ood_samples: Vec<F> = (0..ood_samples)
        .map(|_| challenger.sample_ext_element())
        .collect();

    // Evaluate the polynomial at the OOD samples
    let betas: Vec<F> = ood_samples
        .iter()
        .map(|x| folded_polynomial.evaluate(x))
        .collect();

    // Observe the betas
    challenger.observe_slice(&betas);

    // NP TODO remove
    println!("PROVE_ROUND REACHES 3");

    // ========= STIR MESSAGE =========

    // Sample ramdomness for degree correction
    let comb_randomness = challenger.sample_ext_element();

    // NP TODO remove
    println!("PROVE_ROUND REACHES 4");

    // Sample folding randomness for the next round
    let new_folding_randomness = challenger.sample_ext_element();

    // NP TODO remove
    println!("PROVE_ROUND REACHES 5");

    // Sample queried indices of elements in L^k
    let log_scaling_factor = domain.log_size() - log_folding_factor;

    // NP TODO remove
    println!("PROVE_ROUND REACHES 6");

    // Sample queried indices of elements in L^k_{i-1}
    // NP TODO: Currently no index deduplication
    // NP TODO: Unsafe cast to u64, need u64 here because domain.element() requires u64
    let queried_indices: Vec<u64> = (0..num_queries)
        .map(|_| {
            challenger
                .sample_bits(log_scaling_factor)
                .try_into()
                .unwrap()
        })
        .collect();

    // NP TODO remove
    println!("PROVE_ROUND REACHES 7");

    // Proof of work witness
    // NP TODO: Is this correct? Can we just take the ceil?
    // NP TODO unsafe cast to usize
    let pow_witness = challenger.grind(pow_bits.ceil() as usize);

    // NP TODO remove
    println!("PROVE_ROUND REACHES 8");

    // Shake randomness
    let _shake_randomnes: F = challenger.sample_ext_element();

    // NP TODO remove
    println!("PROVE_ROUND REACHES 9");

    // ========= QUERY PROOFS =========

    // Open the Merkle paths for the queried indices
    let query_proofs: Vec<(Vec<Vec<F>>, M::Proof)> = queried_indices
        .iter()
        .map(|index| {
            config
                .mmcs_config()
                .open_batch(*index as usize, &merkle_tree)
        })
        .collect();

    // ========= POLY QUOTIENT =========

    // NP TODO revise FS in general

    // NP TODO ask Giacomo: is this division (prover step 5) computed before or
    // after the verifier queries f_{i - 1} (verifier step 1)? The protocol is
    // interactive but the order of the interaction is not shown in the paper,
    // yet it is important for FS

    // Compute the domain L^k = shift^k * <omega^k>
    // NP TODO ask Giacomo: should this also scale the shift?
    let domain_k = domain.shrink_coset(log_folding_factor);

    // NP TODO remove
    println!("PROVE_ROUND REACHES 10");

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

    // NP TODO remove
    println!("PROVE_ROUND REACHES 11");

    // Compute the quotient set, i.e \mathcal{G}_i in the paper
    let quotient_set = ood_samples
        .iter()
        .chain(stir_randomness.iter())
        .cloned()
        .collect_vec();

    // NP TODO remove
    println!("PROVE_ROUND REACHES 12");

    // Compute the quotient set evaluations
    let beta_answers = ood_samples.into_iter().zip(betas.clone());
    let stir_answers = stir_randomness.into_iter().zip(stir_randomness_evals);
    let quotient_answers = beta_answers.chain(stir_answers);

    // NP TODO remove
    println!("PROVE_ROUND REACHES 13");

    // Compute the answer polynomial
    let ans_polynomial =
        Polynomial::<F>::lagrange_interpolation(quotient_answers.clone().collect_vec());

    // NP TODO remove
    println!("PROVE_ROUND REACHES 14");

    // Compute the shake polynomial
    let shake_polynomial = compute_shake_polynomial(&ans_polynomial, quotient_answers);

    // NP TODO remove
    println!("PROVE_ROUND REACHES 15");

    // Compute the quotient polynomial
    // NP TODO: Remove the clone
    let vanishing_polynomial = Polynomial::vanishing_polynomial(quotient_set.clone());

    // NP TODO remove
    println!("PROVE_ROUND REACHES 16");

    let quotient_polynomial = &(&folded_polynomial - &ans_polynomial) / &vanishing_polynomial;

    // NP TODO remove
    println!("PROVE_ROUND REACHES 17");

    // Compute the scaling polynomial, 1 + rx + r^2 x^2 + ... + r^n x^n with n = |quotient_set|
    // NP TODO: From the call with Giacomo, it seems that this computation might be wrong
    // NP TODO: Don't use std
    let scaling_polynomial = Polynomial::from_coeffs(
        iter::successors(Some(F::ONE), |&prev| Some(prev * comb_randomness))
            .take(quotient_set.len() + 1)
            .collect_vec(),
    );

    // NP TODO remove
    println!("PROVE_ROUND REACHES 18");

    let witness_polynomial = &quotient_polynomial * &scaling_polynomial;

    // NP TODO remove
    println!("PROVE_ROUND REACHES 19");

    // NP TODO remove
    assert_eq!(witness_polynomial.degree(), folded_polynomial.degree());

    // NP TODO remove
    println!("PROVE_ROUND REACHES 20");

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
        let denominator = Polynomial::monomial(-x);
        shake_polynomial = &shake_polynomial + &(&numerator / &denominator);
    }
    shake_polynomial
}
