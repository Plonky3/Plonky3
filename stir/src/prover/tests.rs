use alloc::vec;
use alloc::vec::Vec;

use itertools::Itertools;
use p3_challenger::MockChallenger;
use p3_commit::Mmcs;
use p3_coset::TwoAdicCoset;
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_poly::test_utils::rand_poly;
use p3_poly::Polynomial;
use rand::{rng, Rng};

use super::{prove_round, RoundConfig};
use crate::proof::RoundProof;
use crate::prover::{commit, prove, StirRoundWitness};
use crate::test_utils::*;
use crate::utils::fold_polynomial;
use crate::SecurityAssumption;

// Auxiliary test function which checks that prove_round transforms the round
// polynomial f_i into the expected polynomial f_{i + 1} and produces the right
// round-proof data (shake polynomial, ans polynomial, etc.). It accepts two
// parameters:
//  - repeat_queries: whether to (fake-) sample some elements of L_i^{k_i} more
//    than once
//  - degree_slack: difference between the degree of the starting polynomial f_0
//    and the maximum degree ensured by the LDT, i. e. 2^log_starting_degree - 1
fn test_prove_round_aux(repeat_queries: bool, degree_slack: usize) {
    let mut rng = rng();

    let config = test_bb_stir_config(
        BB_EXT_SEC_LEVEL,
        SecurityAssumption::CapacityBound,
        20,
        2,
        3,
        4,
    );

    // ============================== Committing ==============================

    // Starting polynomial. We allow it to have lower degree than the maximum
    // bound proved by the LDT, i. e. 2^log_starting_degree - 1
    let degree = (1 << config.log_starting_degree()) - 1 - degree_slack;
    let f_0 = rand_poly(degree);

    let original_domain = TwoAdicCoset::new(
        BBExt::ONE,
        config.log_starting_degree() + config.log_starting_inv_rate(),
    );

    let mut original_domain = original_domain.set_shift(original_domain.generator());

    let original_evals = original_domain.evaluate_polynomial(f_0.coeffs().to_vec());

    let stacked_original_evals =
        RowMajorMatrix::new(original_evals, 1 << config.log_starting_folding_factor());

    let dimensions = stacked_original_evals.dimensions();

    let (root, merkle_tree) = config
        .mmcs_config()
        .commit_matrix(stacked_original_evals.clone());

    let r_0: BBExt = rng.random();

    let witness = StirRoundWitness {
        domain: original_domain.clone(),
        polynomial: f_0.clone(),
        merkle_tree,
        round: 0,
        folding_randomness: r_0,
    };

    // ======================= Preparing fake randomness =======================

    let round = 1;
    let round_config = config.round_config(round);

    let RoundConfig {
        log_folding_factor,
        log_inv_rate,
        num_ood_samples,
        num_queries,
        ..
    } = round_config.clone();

    // Prepare the field randomness produced by the mock challenger
    let r_1: BBExt = rng.random();

    // Out-of-domain randomness
    let ood_randomness: Vec<BBExt> = (0..num_ood_samples).map(|_| rng.random()).collect();

    // Degree-correction randomness
    let comb_randomness = rng.random();

    // Shake randomness (which is squeezed but not used by the prover)
    let shake_randomness = rng.random();

    let mut field_replies = ood_randomness.clone();
    field_replies.push(comb_randomness);
    field_replies.push(r_1);
    field_replies.push(shake_randomness);

    // Random queried indices (in the form of bits, not field elements)
    let log_size_second_codeword = config.log_starting_degree() - log_folding_factor + log_inv_rate;

    let mut bit_replies = (0..num_queries)
        .map(|_| rng.random_range(0..usize::MAX))
        .map(|i: usize| i % (1 << log_size_second_codeword))
        .collect::<Vec<_>>();

    // We incorporate this possibility to test the case where some elements of
    // L_i^{k_i} are sampled more than once, in which case the prover and
    // verifier should remove the duplicate queries and work with an Ans
    // polynomial of consequently lower degree (which also affets degree
    // correction)
    if repeat_queries {
        for _ in 0..num_queries / 4 {
            let (i, j) = (
                rng.random_range(0..num_queries),
                rng.random_range(0..num_queries),
            );
            bit_replies[i] = bit_replies[j];
        }
    }

    // Preloading fake randomness
    let mut challenger = MockChallenger::new(field_replies, bit_replies.clone());

    // ====================== prove_round for round i = 1 ======================

    let (witness, round_proof) = prove_round(&config, witness, &mut challenger);

    // ============================ Witness checks ============================

    let StirRoundWitness {
        domain,
        polynomial: f_1,
        folding_randomness,
        round,
        ..
    } = witness;

    let expected_domain = original_domain.shrink_subgroup(1);

    let expected_round = 1;

    // Domain testing
    assert_eq!(domain, expected_domain);
    assert_eq!(r_1, folding_randomness);

    // Round-number testing
    assert_eq!(round, expected_round);

    // Computing the expected polynomial f_1 by hand
    let g_1 = fold_polynomial(&f_0, r_0, log_folding_factor);

    let mut original_domain_pow_k = original_domain.shrink_coset(log_folding_factor);
    let stir_randomness = bit_replies
        .iter()
        .map(|&i| original_domain_pow_k.element(i));

    let quotient_set = stir_randomness
        .chain(ood_randomness.into_iter())
        .unique()
        .collect_vec();

    let quotient_set_points = quotient_set
        .iter()
        .map(|x| (*x, g_1.evaluate(x)))
        .collect_vec();

    let expected_ans_polynomial = Polynomial::lagrange_interpolation(quotient_set_points.clone());

    let quotient_polynomial = &(&g_1 - &expected_ans_polynomial)
        / &Polynomial::vanishing_polynomial(quotient_set.clone());

    let expected_f_1 =
        &Polynomial::power_polynomial(comb_randomness, quotient_set.len()) * &quotient_polynomial;

    // Main check of this entire function
    assert_eq!(f_1, expected_f_1);

    // =================== Ans- and shake-polynomial checks ===================

    let RoundProof {
        query_proofs,
        ans_polynomial,
        shake_polynomial,
        ..
    } = round_proof;

    for (&i, (leaf, proof)) in bit_replies.iter().unique().zip(query_proofs) {
        config
            .mmcs_config()
            .verify_batch(&root, &[dimensions], i, &vec![leaf], &proof)
            .unwrap();
    }

    let expected_shake_polynomial = quotient_set_points
        .into_iter()
        .map(|(x, y)| {
            let (quotient, _) = (&ans_polynomial - &Polynomial::constant(y))
                .divide_by_vanishing_linear_polynomial(x);
            quotient
        })
        .fold(Polynomial::zero(), |sum, next_poly| &sum + &next_poly);

    assert_eq!(ans_polynomial, expected_ans_polynomial);
    assert_eq!(shake_polynomial, expected_shake_polynomial);
}

#[test]
// Checks that prove_round produces the expected witness and round proof, most
// importantly the correct witness, Ans and shake polynomials
fn test_prove_round_no_repeat() {
    test_prove_round_aux(false, 0);
}

#[test]
// Checks the same as test_prove_round_no_repeat, but includes duplicates in the
// in-domain queried points
fn test_prove_round_repeat() {
    test_prove_round_aux(true, 0);
}

#[test]
// Checks the same as test_prove_round_no_repeat, but operates on a polynomial
// with degree lower than the maximum allowed by the configuration
fn test_prove_round_degree_slack() {
    test_prove_round_aux(false, 10);
}

#[test]
// Checks that prove runs from beginning to end and performs a degree check on
// the final polynomial p = g_{num_rounds} (where num_rounds = M + 1 in the
// notation of the article)
fn test_prove() {
    let config = test_bb_stir_config(
        BB_EXT_SEC_LEVEL,
        SecurityAssumption::CapacityBound,
        14,
        1,
        4,
        3,
    );

    let polynomial = rand_poly((1 << config.log_starting_degree()) - 1);

    let (witness, commitment) = commit(&config, polynomial);

    let mut challenger = test_bb_challenger();

    let proof = prove(&config, witness, commitment, &mut challenger);

    // Final-degree testing
    assert_eq!(config.log_stopping_degree(), 2);
    assert!(proof.final_polynomial.degree().is_none_or(|d| d < 1 << 2));
}

#[test]
// Checks that the final polynomial p = g_3 is the expected one in three-round
// STIR
fn test_prove_final_polynomial() {
    let mut rng = rand::rng();

    let log_starting_degree = 20;
    let log_folding_factor = 4;

    // We use a config that will result in the following polynomials and degree
    // bounds:
    //  - f_0:      2^20 - 1
    //  - f_1, g_1: 2^16 - 1
    //  - f_2, g_2: 2^12 - 1
    //  - g_3:      2^8 - 1
    let config = test_bb_stir_config(
        BB_EXT_SEC_LEVEL,
        SecurityAssumption::CapacityBound,
        log_starting_degree,
        2,
        log_folding_factor,
        3,
    );

    let log_initial_codeword_size = log_starting_degree + config.log_starting_inv_rate();

    // ======================= Preparing fake randomness =======================
    let mut field_replies = Vec::new();
    let mut bit_replies = Vec::new();

    let mut round_ood_replies = Vec::new();
    let mut round_comb_replies = Vec::new();
    let mut round_r_replies = Vec::new();
    let mut round_shake_replies = Vec::new();
    let mut round_bit_replies = Vec::new();

    let r_0 = rng.random();
    field_replies.push(r_0);
    round_r_replies.push(r_0);

    for round in 1..=2 {
        let round_config = config.round_config(round);

        let RoundConfig {
            log_folding_factor,
            num_ood_samples,
            num_queries,
            ..
        } = round_config.clone();

        // Out of domain randomness
        let ood_randomness: Vec<BBExt> = (0..num_ood_samples).map(|_| rng.random()).collect();

        // Comb randomness
        let comb_randomness = rng.random();

        // Folding randomness
        let r: BBExt = rng.random();

        // Shake randomness (which is squeezed but not used by the prover)
        let shake_randomness = rng.random();

        field_replies.extend(ood_randomness.clone());
        field_replies.push(comb_randomness);
        field_replies.push(r);
        field_replies.push(shake_randomness);

        round_ood_replies.push(ood_randomness);
        round_comb_replies.push(comb_randomness);
        round_r_replies.push(r);
        round_shake_replies.push(shake_randomness);

        // Random queried indices (in the form of bits, not field elements)

        // This is the log2 of |L_{i - 1}^{k_{i - 1}}|
        let log_prev_domain_size = log_initial_codeword_size - (round - 1) - log_folding_factor;

        let new_bit_replies = (0..num_queries)
            .map(|_| rng.random_range(0..usize::MAX))
            .map(|i: usize| i % (1 << log_prev_domain_size))
            .collect::<Vec<_>>();

        bit_replies.extend(new_bit_replies.clone());
        round_bit_replies.push(new_bit_replies);
    }

    // Prepare the final-round fake randomness (irrelevant to the final
    // polynomial p = g_3, but sampled from the challenger by the prover)

    // log2 of |L_2^{k_2}|
    let log_final_domain_size = log_initial_codeword_size - 4 - log_folding_factor;

    let final_bit_replies = (0..config.final_num_queries())
        .map(|_| rng.random_range(0..usize::MAX))
        .map(|i: usize| i % (1 << log_final_domain_size))
        .collect::<Vec<_>>();

    bit_replies.extend(final_bit_replies.clone());

    // Preloading fake randomness
    let mut challenger = MockChallenger::new(field_replies, bit_replies);

    // ================================ Proving ================================
    let mut polynomial = rand_poly((1 << config.log_starting_degree()) - 1);

    let (witness, commitment) = commit(&config, polynomial.clone());

    let generator = witness.domain.generator();

    let proof = prove(&config, witness, commitment, &mut challenger);

    // ================ Computing expected final polynomial g_3 ===============

    // Computing f_1 and f_2 manually
    for round in 1..=2 {
        let g_i = fold_polynomial(&polynomial, round_r_replies[round - 1], log_folding_factor);

        // Computing the domain L_{i - 1}^{k_{i - 1}}
        let mut domain_pow_k =
            TwoAdicCoset::new(generator, log_initial_codeword_size - (round - 1))
                .shrink_coset(log_folding_factor);

        let stir_randomness = round_bit_replies[round - 1]
            .iter()
            .map(|&i| domain_pow_k.element(i));

        let quotient_set = stir_randomness
            .chain(round_ood_replies[round - 1].clone().into_iter())
            .unique()
            .collect_vec();

        let quotient_set_points = quotient_set
            .iter()
            .map(|x| (*x, g_i.evaluate(x)))
            .collect_vec();

        let ans_polynomial = Polynomial::lagrange_interpolation(quotient_set_points.clone());

        let quotient_polynomial =
            &(&g_i - &ans_polynomial) / &Polynomial::vanishing_polynomial(quotient_set.clone());

        let comb_randomness = round_comb_replies[round - 1];

        // New round polynomial f_i
        polynomial = &Polynomial::power_polynomial(comb_randomness, quotient_set.len())
            * &quotient_polynomial;
    }

    let f_2 = polynomial.clone();

    // Computing the expected final polynomial p = g_3
    let expected_final_polynomial = fold_polynomial(&f_2, round_r_replies[2], 4);

    assert_eq!(proof.final_polynomial, expected_final_polynomial);
}

#[test]
#[should_panic(expected = "The degree of the polynomial (16384) is too large")]
// Checks that the commit method panics if the polynomial is larger than allowed
// by the configuration
fn test_incorrect_polynomial() {
    let config = test_bb_stir_config(
        BB_EXT_SEC_LEVEL,
        SecurityAssumption::CapacityBound,
        14,
        1,
        4,
        3,
    );

    let polynomial = rand_poly(1 << config.log_starting_degree());

    commit(&config, polynomial);
}
