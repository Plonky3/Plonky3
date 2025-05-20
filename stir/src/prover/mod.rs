use alloc::vec;
use alloc::vec::Vec;

use itertools::{izip, Itertools};
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{
    batch_multiplicative_inverse, eval_packed_ext_poly, ExtensionField, Field,
    PackedFieldExtension, PackedValue, TwoAdicField,
};
use p3_matrix::dense::RowMajorMatrix;

use crate::config::{observe_public_parameters, RoundConfig};
use crate::proof::RoundProof;
use crate::utils::{
    add_polys, divide_by_vanishing_linear_polynomial, domain_dft, domain_idft,
    fold_evaluations_at_domain, lagrange_interpolation, observe_ext_slice_with_size,
    power_polynomial, vanishing_polynomial,
};
use crate::{Messages, StirConfig, StirProof, POW_BITS_WARNING};

#[cfg(test)]
mod tests;

/// Prover witness for the STIR protocol produced by the [`commit`] method.
pub struct StirWitness<F: TwoAdicField, M: Mmcs<F>> {
    // Domain L_0
    pub(crate) domain: TwoAdicMultiplicativeCoset<F>,

    // Evaluations of f_0 over K_0

    // NP TODO clarify in note that this memory cannot be saved due to the FFT
    // (even though the information in here is already contained in the Merkle
    // tree below)
    pub(crate) evals_k: Vec<F>,

    // Merkle tree whose leaves are the stacked evaluations of f_0. Its root is
    // the commitment shared with the verifier.
    pub(crate) merkle_tree: M::ProverData<RowMajorMatrix<F>>,
}

// STIR witness enriched with additional information (round number and folding
// randomness) received and produced by the method prove_round
pub(crate) struct StirRoundWitness<F: TwoAdicField, M: Mmcs<F>> {
    // The indices are given in the following frame of reference: Self is
    // produced inside prove_round for round i (in {1, ..., M}). The final
    // round, with index M + 1, does not produce a StirRoundWitness. The first
    // StirRoundWitness, produced inside prove() and passed to the first call to
    // prove_round(), should be understood as having i = 0.

    // Domain L_i. The chosen sequence of domains L_0, L_1, ... is documented
    // at the start of the method commit.
    pub(crate) domain_l: TwoAdicMultiplicativeCoset<F>,

    // Domain K_i
    pub(crate) domain_k: TwoAdicMultiplicativeCoset<F>,

    // Evaluations of f_i over K_i

    // NP TODO same comment as in the evals_k_0 field of StirWitness
    pub(crate) evals_k: Vec<F>,

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
pub fn commit_polynomial<F, M>(
    config: &StirConfig<F, M>,
    polynomial: Vec<F>,
) -> (StirWitness<F, M>, M::Commitment)
where
    F: TwoAdicField,
    M: Mmcs<F>,
{
    assert!(
        polynomial.is_empty() || polynomial.len() <= (1 << config.log_starting_degree()),
        "The degree of the polynomial ({}) is too large: the configuration \
        only supports polynomials of degree up to 2^{} - 1 = {}",
        polynomial.len() - 1,
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
    let domain =
        TwoAdicMultiplicativeCoset::new(F::two_adic_generator(log_size), log_size).unwrap();

    // Committing to the evaluations of f_0 over L_0.
    let evals = domain_dft(domain, polynomial, &config.dft);

    commit_evals(config, evals)
}

pub fn commit_evals<F, M>(
    config: &StirConfig<F, M>,
    evals: Vec<F>,
) -> (StirWitness<F, M>, M::Commitment)
where
    F: TwoAdicField,
    M: Mmcs<F>,
{
    let log_size = config.log_starting_degree() + config.log_starting_inv_rate();

    let domain =
        TwoAdicMultiplicativeCoset::new(F::two_adic_generator(log_size), log_size).unwrap();

    let evals_k = evals
        .iter()
        .step_by(1 << config.log_starting_inv_rate())
        .copied()
        .collect();

    // The stacking width is
    //   k_0 = 2^{log_size - config.log_starting_folding_factor},
    // which facilitates opening values so that the prover can verify the first
    // folding
    let stacked_evals = RowMajorMatrix::new(
        evals,
        1 << (log_size - config.log_starting_folding_factor()),
    )
    .transpose();

    let (commitment, merkle_tree) = config.mmcs_config().commit_matrix(stacked_evals);

    // NP TODO maybe remove domain and/or evals from here and compute them inside prove()
    (
        StirWitness {
            domain,
            evals_k,
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

    // Initial proof of work
    let starting_folding_pow_witness = challenger.grind(config.starting_folding_pow_bits());

    // Sample the folding randomness r_0
    challenger.observe(F::from_u8(Messages::FoldingRandomness as u8));
    let folding_randomness: EF = challenger.sample_algebra_element();

    // NP TODO handle this symmetrically to StirWitness.domain
    let domain_k_0 = witness
        .domain
        .shrink_coset(config.log_starting_inv_rate())
        .unwrap();

    // Enriching the initial witness into a full round witness that prove_round
    // can receive.
    let mut witness = StirRoundWitness {
        domain_l: witness.domain,
        domain_k: domain_k_0,
        evals_k: witness.evals_k,
        merkle_tree: witness.merkle_tree,
        round: 0,
        folding_randomness,
    };

    // Prove each full round i = 1, ..., M of the protocol
    let mut round_proofs = Vec::with_capacity(config.num_rounds() - 1);

    for _ in 1..=config.num_rounds() - 1 {
        // NP TODO two PoW per round
        let (new_witness, round_proof) = prove_round(config, witness, challenger);

        witness = new_witness;
        round_proofs.push(round_proof);
    }

    // Final round i = M + 1
    let log_last_folding_factor = config.log_last_folding_factor();

    // Folding the evaluations of f_M at k_M in order to obtain those of the
    // final polynomial p (i. e. g_{M + 1})
    let final_polynomial_evals = fold_evaluations_at_domain(
        witness.evals_k,
        witness.domain_k,
        log_last_folding_factor,
        witness.folding_randomness,
    );

    let final_domain_pow = witness
        .domain_k
        .exp_power_of_2(log_last_folding_factor)
        .unwrap();

    // Interpolating g_{M + 1}
    let final_polynomial = domain_idft(final_polynomial_evals, final_domain_pow, &config.dft);

    let final_queries = config.final_num_queries();

    // Logarithm of |(L_M)^(k_M)|
    let log_query_domain_size = witness.domain_l.log_size() - log_last_folding_factor;

    // Observe the final polynomial g_{M + 1}
    challenger.observe(F::from_u8(Messages::FinalPolynomial as u8));
    observe_ext_slice_with_size(challenger, &final_polynomial);

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
    let final_pow_witness = challenger.grind(config.final_pow_bits());

    StirProof {
        round_proofs,
        final_polynomial,
        starting_folding_pow_witness,
        final_pow_witness,
        final_round_queries: queries_to_final,
    }
}

/// Prove a single full round, taking in a witness for the previous round and
/// returning a witness for the new one as well as the round proof.
pub(crate) fn prove_round<F, EF, M, C>(
    // Full STIR configuration from which the round-specific configuration is
    // extracted
    config: &StirConfig<EF, M>,
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
        domain_l,
        domain_k,
        evals_k,
        merkle_tree,
        folding_randomness,
        ..
    } = witness;

    // ================================ Folding ================================

    // Obtain g_i as the folding of f_{i - 1}

    // NP TODO
    assert_eq!(domain_k.size(), evals_k.len(), "round: {}", round);

    let folded_evals_k =
        fold_evaluations_at_domain(evals_k, domain_k, log_folding_factor, folding_randomness);
    let domain_k_pow = domain_k.exp_power_of_2(log_folding_factor).unwrap();
    let folded_polynomial = domain_idft(folded_evals_k.clone(), domain_k_pow, &config.dft);

    // TODO: Remove this assert and just pad folded_polynomial by zeroes to make it true.
    // For now this assert seems to hold true for the tests I am running.
    assert!(folded_polynomial.len() % F::Packing::WIDTH == 0);
    let packed_folded_polynomial = folded_polynomial
        .chunks_exact(F::Packing::WIDTH)
        .map(|chunck| EF::ExtensionPacking::from_ext_slice(chunck))
        .collect::<Vec<_>>(); // TODO: Ideally this transformation should be a method in ExtensionPacking.

    // Compute the i-th domain L_i = w * <w^{2^i}> = w * (w^{-1} * L_{i - 1})^2
    // and its subdomain K_{i - 1}
    let new_domain_l = domain_l.shrink_coset(1).unwrap(); // Can never panic due to parameter set-up
    let new_domain_k = domain_k.shrink_coset(log_folding_factor).unwrap(); // Idem

    // Evaluate g_i over L_i
    let folded_evals_l = domain_dft(new_domain_l, folded_polynomial.clone(), &config.dft);

    // Collecting the evaluations of g_i over K_i for later use
    let g_i_evals_k = folded_evals_l
        .iter()
        .step_by(1 << (new_domain_l.log_size() - new_domain_k.log_size()))
        .copied()
        .collect_vec();

    // Stack the evaluations, commit to them (in preparation for
    // next-round-folding verification, and therefore with width equal to the
    // folding factor of the next round) and then observe the commitment
    let new_stacked_evals = RowMajorMatrix::new(
        folded_evals_l,
        1 << (new_domain_l.log_size() - log_next_folding_factor),
    )
    .transpose();

    let (new_commitment, new_merkle_tree) = config.mmcs_config().commit_matrix(new_stacked_evals);

    // Observe the commitment
    challenger.observe(F::from_u8(Messages::RoundCommitment as u8));
    challenger.observe(new_commitment.clone());

    // ======================== Out-of-domain sampling ========================

    let mut ood_samples = Vec::new();

    challenger.observe(F::from_u8(Messages::OodSamples as u8));
    while ood_samples.len() < num_ood_samples {
        let el: EF = challenger.sample_algebra_element();
        if !new_domain_l.contains(el) {
            ood_samples.push(el);
        }
    }

    // Evaluate the polynomial at the out-of-domain sampled points
    let betas: Vec<EF> = ood_samples
        .iter()
        .map(|&x| eval_packed_ext_poly(&packed_folded_polynomial, x))
        .collect();

    // Observe the evaluations
    challenger.observe(F::from_u8(Messages::Betas as u8));
    betas
        .iter()
        .for_each(|&beta| challenger.observe_algebra_element(beta));

    // ========================== Sampling randomness ==========================

    // Sample randomness for degree correction
    challenger.observe(F::from_u8(Messages::CombRandomness as u8));
    let comb_randomness = challenger.sample_algebra_element();

    // Sample folding randomness for the next round
    challenger.observe(F::from_u8(Messages::FoldingRandomness as u8));
    let new_folding_randomness = challenger.sample_algebra_element();

    // Sample queried indices of elements in L_{i - 1}^k_{i - 1}
    let log_query_domain_size = domain_l.log_size() - log_folding_factor;

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
    let domain_pow2_k = domain_l.exp_power_of_2(log_folding_factor).unwrap(); // Can never panic due to parameter set-up

    // Get the domain elements at the queried indices (i.e r^shift_i in the paper)
    let stir_randomness: Vec<EF> = queried_indices
        .iter()
        .map(|&index| domain_pow2_k.element(index))
        .collect();

    // Evaluate the polynomial at those points
    let stir_randomness_evals: Vec<EF> = stir_randomness
        .iter()
        .map(|&x| eval_packed_ext_poly(&packed_folded_polynomial, x))
        .collect();

    // stir_answers has been dedup-ed but beta_answers has not yet:
    let stir_answers = stir_randomness.into_iter().zip(stir_randomness_evals);
    let beta_answers = ood_samples.into_iter().zip(betas.clone()).unique();
    let quotient_answers = beta_answers.chain(stir_answers).collect_vec();

    // Compute the quotient set, \mathcal{G}_i in the notation of the article
    let quotient_set = quotient_answers.iter().map(|(x, _)| *x).collect_vec();
    let quotient_set_size = quotient_set.len();

    // Compute the Ans polynomial and add it to the transcript
    let ans_polynomial = lagrange_interpolation(quotient_answers.clone());
    challenger.observe(F::from_u8(Messages::AnsPolynomial as u8));
    observe_ext_slice_with_size(challenger, &ans_polynomial);

    // Compute the shake polynomial and add it to the transcript
    let shake_polynomial = compute_shake_polynomial(&ans_polynomial, &quotient_set);
    challenger.observe(F::from_u8(Messages::ShakePolynomial as u8));
    observe_ext_slice_with_size(challenger, &shake_polynomial);

    // Shake randomness: this is only used by the verifier, but it doesn't need
    // to be kept private. Therefore, the verifier can sample it from the
    // challenger, in which case the prover must follow suit to keep the
    // challengers in sync.
    challenger.observe(F::from_u8(Messages::ShakeRandomness as u8));
    let _shake_randomness: EF = challenger.sample_algebra_element();

    // Compute the evaluations of the ans, vanishing and power polynomials over K_i
    if quotient_set_size > new_domain_k.size() {
        // NP TODO make sure this should never happen
        panic!("Early termination configuration failed");
    }

    let mut power_polynomial = power_polynomial(comb_randomness, quotient_set_size);
    let mut vanishing_polynomial = vanishing_polynomial(quotient_set);

    // NP TODO this could be done in one go if dft_batch had a coset version.
    // NB: If done, make sure to pad the vectors with zeros to the size
    // new_domain_k.size(); currently this is handled by the convenience
    // function domain_dft from utils

    let mut resized_ans_polynomial = ans_polynomial.clone();

    // NP TODO remove calls to clone() once Polynomial is gone and we are
    // working with coeff vecs directly
    power_polynomial.resize(new_domain_k.size(), EF::ZERO);
    resized_ans_polynomial.resize(new_domain_k.size(), EF::ZERO);
    vanishing_polynomial.resize(new_domain_k.size(), EF::ZERO);

    let flat_coeffs = [
        power_polynomial,
        resized_ans_polynomial,
        vanishing_polynomial,
    ]
    .concat();

    let coeff_matrix = RowMajorMatrix::new(flat_coeffs, new_domain_k.size()).transpose();
    let eval_matrix = config.dft.coset_dft_batch(coeff_matrix, domain_k.shift());

    // Batch-invert all vanishing-polynomial evaluations denominators. Note that
    // vanishing polynomial cannot vanish at any queried point: K_i is contained
    // in L_i and the queried points come from either EF \ L_i (in the case of
    // ood) or L_{i - 1}^{k_{i - 1}} (in the case of stir randomness), both of
    // which are disjoint from L_i.
    let vanishing_evals = eval_matrix.row_slices().map(|r| r[2]).collect_vec();
    let vanishing_evals_inv = batch_multiplicative_inverse(&vanishing_evals);

    // Computing the evaluations of f_i over K_i
    let evals_k = izip!(g_i_evals_k, vanishing_evals_inv, eval_matrix.row_slices())
        .map(|(g, v_inv, row)| row[0] * (g - row[1]) * v_inv)
        .collect();

    (
        StirRoundWitness {
            domain_l: new_domain_l,
            domain_k: new_domain_k,
            evals_k,
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
fn compute_shake_polynomial<F: TwoAdicField>(ans_polynomial: &[F], points: &[F]) -> Vec<F> {
    // The shake polynomial is defined as:
    //   sum_{y in quotient_answers} (ans_polynomial(x) - ans_polynomial(y)) / (x - y)
    let mut shake_polynomial = vec![];

    for p in points {
        let (quotient, _) = divide_by_vanishing_linear_polynomial(ans_polynomial, *p);
        shake_polynomial = add_polys(&shake_polynomial, &quotient);
    }

    shake_polynomial
}
