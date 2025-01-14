use alloc::vec;
use alloc::vec::Vec;
use core::convert::TryInto;
use core::iter;

use itertools::{izip, Itertools};
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_strict_usize;
use tracing::{info_span, instrument};

use crate::config::RoundConfig;
use crate::coset::Radix2Coset;
use crate::polynomial::Polynomial;
use crate::proof::RoundProof;
use crate::{StirConfig, StirParameters, StirProof};

pub struct StirWitness<F: TwoAdicField, M: Mmcs<F>> {
    pub(crate) domain: Radix2Coset<F>,
    pub(crate) polynomial: Polynomial<F>,
    pub(crate) merkle_tree: M::ProverData<RowMajorMatrix<F>>,
    pub(crate) stacked_evals: RowMajorMatrix<F>,
    pub(crate) round: usize,
    pub(crate) folding_randomness: F,
}

pub fn fold_polynomial<F: TwoAdicField>(
    polynomial: Polynomial<F>,
    folding_randomness: F,
    log_folding_factor: usize,
) -> Polynomial<F> {
    todo!()
}

pub fn fold_evals<F: TwoAdicField>(
    evals: RowMajorMatrix<F>,
    folding_randomness: F,
    log_folding_factor: usize,
) -> Vec<F> {
    todo!()
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
            folding_randomness: F::one(),
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

    // NP TODO final round
    todo!()
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

    // Fold the polynomial and the evaluations
    let folded_polynomial =
        fold_polynomial(polynomial, folding_randomness, 1 << log_folding_factor);

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

    let new_domain = domain.shrink_coset(1).shift_by_root_of_unity();
    let folded_evals = new_domain.evaluate_polynomial(&folded_polynomial);

    // Stack the new folded evaluations, commit and observe the commitment
    let new_stacked_evals = RowMajorMatrix::new(folded_evals, 1 << log_folding_factor);
    let (new_commitment, new_merkle_tree) = config
        .mmcs_config()
        .commit_matrix(new_stacked_evals.clone());
    challenger.observe(new_commitment.clone());

    // ========= OOD SAMPLING =========

    // NP TODO: Sample from the extension field like in FRI
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

    // ========= STIR MESSAGE =========

    // Sample ramdomness for degree correction
    let comb_randomness = challenger.sample_ext_element();

    // Sample randomness for the next folding
    let new_folding_randomness = challenger.sample_ext_element();

    // Sample queried indices of elements in L^k
    let log_scaling_factor = domain.log_size() - log_folding_factor;

    // Sample queried indices of elements in L^k
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

    // Proof of work witness
    // NP TODO: Is this correct? Can we just take the ceil?
    // NP TODO unsafe cast to usize
    let pow_witness = challenger.grind(pow_bits.ceil() as usize);

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

    // Compute the domain L^k = shift * <omega^k>
    let domain_k = domain.shrink_subgroup(log_folding_factor);

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

    // Compute the quotient set, i.e \mathcal{G}_i in the paper
    let quotient_set = ood_samples
        .iter()
        .chain(stir_randomness.iter())
        .cloned()
        .collect_vec();

    // Compute the quotient set evaluations
    let beta_answers = ood_samples.into_iter().zip(betas.clone());
    let stir_answers = stir_randomness.into_iter().zip(stir_randomness_evals);
    let quotient_answers = beta_answers.chain(stir_answers);

    // Compute the answer polynomial
    let ans_polynomial = Polynomial::<F>::naive_interpolate(quotient_answers.clone().collect_vec());

    // Compute the shake polynomial
    let mut shake_polynomial = Polynomial::zero();
    for (x, y) in quotient_answers {
        let numerator = &ans_polynomial - &y;
        let denominator = Polynomial::monomial(-x);
        shake_polynomial = &shake_polynomial + &(&numerator / &denominator);
    }

    // Compute the quotient polynomial
    // NP TODO: Remove the clone
    let vanishing_polynomial = Polynomial::vanishing_polynomial(quotient_set.clone());
    let quotient_polynomial = &(&folded_polynomial + &ans_polynomial) / &vanishing_polynomial;

    // Compute the scaling polynomial, 1 + rx + r^2 x^2 + ... + r^n x^n with n = |quotient_set|
    // NP TODO: From the call with Giacomo, it seems that this computation might be wrong
    // NP TODO: Don't use std
    let scaling_polynomial = Polynomial::from_coeffs(
        iter::successors(Some(F::one()), |&prev| Some(prev * comb_randomness))
            .take(quotient_set.len() + 1)
            .collect_vec(),
    );

    let witness_polynomial = &quotient_polynomial * &scaling_polynomial;

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
