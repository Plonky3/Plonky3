use crate::{
    proof::RoundProof,
    prover::{commit, prove, StirRoundWitness},
    test_utils::*,
    utils::fold_polynomial,
};
use itertools::Itertools;
use p3_challenger::MockChallenger;
use p3_commit::Mmcs;
use p3_coset::TwoAdicCoset;
use p3_field::FieldAlgebra;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_poly::{test_utils::rand_poly, Polynomial};
use rand::Rng;

use super::{prove_round, RoundConfig};

// Checks that prove_round transforms the round polynomial f_i into the expected
// polynomial f_{i + 1}.
//  - repeat_queries: whether to (fake-) sample some elements of L_i^{k_i} more
//    than once
//  - degree_slack: difference between the degree of the starting polynomial f_0
//    and the maximum degree ensured by the LDT, i. e. 2^log_starting_degree - 1
fn test_prove_round_aux(repeat_queries: bool, degree_slack: usize) {
    let mut rng = rand::thread_rng();

    let config = test_bb_stir_config(20, 2, 3, 4);

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
    // Folding randomness for rounds 1 and 2
    let r_0: BBExt = rng.gen();
    let r_1: BBExt = rng.gen();

    // Out of domain randomness
    let ood_randomness: Vec<BBExt> = (0..num_ood_samples).map(|_| rng.gen()).collect();

    // Comb randomness
    let comb_randomness = rng.gen();

    // Shake randomness (which is squeezed but not used by the prover)
    let shake_randomness = rng.gen();

    let mut field_replies = ood_randomness.clone();
    field_replies.push(comb_randomness);
    field_replies.push(r_1);
    field_replies.push(shake_randomness);

    // Queried-index randomness (in the form of bits, not field elements)
    let log_size_second_codeword = config.log_starting_degree() - log_folding_factor + log_inv_rate;

    let mut bit_replies = (0..num_queries)
        .map(|_| rng.gen())
        .map(|i: usize| i % (1 << log_size_second_codeword))
        .collect::<Vec<_>>();

    // We incorporate this possibility to test the case where some elements of
    // L_i^{k_i} are sampled more than once, in which case the prover and
    // verifier should remove the duplicate queries and work with an Ans
    // polynomial of consequently lower degree
    if repeat_queries {
        for _ in 0..num_queries / 4 {
            let (i, j) = (rng.gen_range(0..num_queries), rng.gen_range(0..num_queries));
            bit_replies[i] = bit_replies[j];
        }
    }

    // Preloading fake randomness
    let mut challenger = MockChallenger::new(field_replies, bit_replies.clone());

    // Starting polynomial. We allow it to have lower degree than the maximum
    // bound proved by the LDT, i. e. 2^log_starting_degree - 1
    let degree = (1 << config.log_starting_degree()) - 1 - degree_slack;
    let f_0 = rand_poly(degree);

    let original_domain = TwoAdicCoset::new(
        BBExt::ONE,
        config.log_starting_degree() + config.log_starting_inv_rate(),
    );

    let original_domain = original_domain.set_shift(original_domain.generator());

    let original_evals = original_domain.evaluate_polynomial(&f_0);

    let stacked_original_evals =
        RowMajorMatrix::new(original_evals, 1 << config.log_starting_folding_factor());

    let dimensions = stacked_original_evals.dimensions();

    let (root, merkle_tree) = config
        .mmcs_config()
        .commit_matrix(stacked_original_evals.clone());

    let witness = StirRoundWitness {
        domain: original_domain.clone(),
        polynomial: f_0.clone(),
        merkle_tree,
        round,
        folding_randomness: r_0,
    };

    let (witness, round_proof) = prove_round(&config, witness, &mut challenger);

    // =============== Witness Checks ===============

    let StirRoundWitness {
        domain,
        polynomial: f_1,
        folding_randomness,
        round,
        ..
    } = witness;

    let expected_domain = original_domain.shrink_subgroup(1);

    let expected_round = 2;

    // Domain testing
    assert_eq!(domain, expected_domain);
    assert_eq!(r_1, folding_randomness);

    // Round-number testing
    assert_eq!(round, expected_round);

    // Polynomial testing
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

    assert_eq!(f_1, expected_f_1);

    // ================= Round Proof Checks ==================

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
fn test_prove_round_no_repeat() {
    test_prove_round_aux(false, 0);
}

#[test]
fn test_prove_round_repeat() {
    test_prove_round_aux(true, 0);
}

#[test]
fn test_prove_round_degree_slack() {
    test_prove_round_aux(false, 10);
}

#[test]
// Checks that prove runs from beginning to end and performs a degree check on
// the final polynomial p = g_{num_rounds} (where num_rounds = M + 1 in the
// notation of the article)
fn test_prove() {
    let config = test_bb_stir_config(14, 1, 4, 3);

    let polynomial = rand_poly((1 << config.log_starting_degree()) - 1);

    let (witness, commitment) = commit(&config, polynomial);

    let mut challenger = test_bb_challenger();

    let proof = prove(&config, witness, commitment, &mut challenger);

    // Final-degree testing
    assert_eq!(config.log_stopping_degree(), 2);
    assert!(proof.final_polynomial.degree().is_none_or(|d| d < 1 << 2));
}

#[test]
// Checks that the final polynomial f_3 is the expected one after three rounds
fn test_prove_final_polynomial() {
    let mut rng = rand::thread_rng();

    // We use a config that will result in the following polynomials and degree
    // bounds:
    //  - f_0:      2^20 - 1
    //  - f_1, g_1: 2^16 - 1
    //  - f_2, g_2: 2^12 - 1
    //  - g_3:      2^8 - 1
    let config = test_bb_stir_config(20, 2, 4, 3);

    // =============== Preparing fake randomness ===============
    let mut field_replies = Vec::new();
    let mut bit_replies = Vec::new();

    let mut round_ood_replies = Vec::new();
    let mut round_comb_replies = Vec::new();
    let mut round_r_replies = Vec::new();
    let mut round_shake_replies = Vec::new();
    let mut round_bit_replies = Vec::new();

    // TODO remove
    for round in 1..=2 {
        println!("round: {}", round);
        let round_config = config.round_config(round);
        println!("done");

        let RoundConfig {
            log_folding_factor,
            log_inv_rate,
            num_ood_samples,
            num_queries,
            ..
        } = round_config.clone();

        // Folding randomness for rounds 1 and 2
        let r: BBExt = rng.gen();

        // Out of domain randomness
        let ood_randomness: Vec<BBExt> = (0..num_ood_samples).map(|_| rng.gen()).collect();

        // Comb randomness
        let comb_randomness = rng.gen();

        // Shake randomness (which is squeezed but not used by the prover)
        let shake_randomness = rng.gen();

        field_replies.extend(ood_randomness.clone());
        field_replies.push(comb_randomness);
        field_replies.push(r);
        field_replies.push(shake_randomness);

        round_ood_replies.push(ood_randomness);
        round_comb_replies.push(comb_randomness);
        round_r_replies.push(r);
        round_shake_replies.push(shake_randomness);

        // Queried-index randomness (in the form of bits, not field elements)
        let log_size_second_codeword =
            config.log_starting_degree() - log_folding_factor + log_inv_rate;

        let new_bit_replies = (0..num_queries)
            .map(|_| rng.gen())
            .map(|i: usize| i % (1 << log_size_second_codeword))
            .collect::<Vec<_>>();

        bit_replies.extend(new_bit_replies.clone());
        round_bit_replies.push(new_bit_replies);
    }

    let r_2 = rng.gen();
    field_replies.push(r_2);
    round_r_replies.push(r_2);

    // Preloading fake randomness
    let mut challenger = MockChallenger::new(field_replies, bit_replies);

    // =============== Proving ===============

    // NP TODO remove
    println!("GETS HERE 0");

    let mut polynomial = rand_poly((1 << config.log_starting_degree()) - 1);

    // NP TODO remove
    println!("GETS HERE 1");

    let (witness, commitment) = commit(&config, polynomial.clone());

    // NP TODO remove
    println!("GETS HERE 2");

    let generator = witness.domain.generator();

    let proof = prove(&config, witness, commitment, &mut challenger);

    // NP TODO remove
    println!("GETS HERE 3");

    // =========== Computing expected final polynomial g_3 ==========
    for i in 0..=1 {
        let g_i = fold_polynomial(&polynomial, round_r_replies[i], 4);

        // Computing the domain L_{i - 1}^{k_{i - 1}}
        let mut domain_pow_k = TwoAdicCoset::new(generator, 20 - 2 * i).shrink_coset(4);

        let stir_randomness = round_bit_replies[i]
            .iter()
            .map(|&i| domain_pow_k.element(i));

        let quotient_set = stir_randomness
            .chain(round_ood_replies[i].clone().into_iter())
            .unique()
            .collect_vec();

        let quotient_set_points = quotient_set
            .iter()
            .map(|x| (*x, g_i.evaluate(x)))
            .collect_vec();

        let expected_ans_polynomial =
            Polynomial::lagrange_interpolation(quotient_set_points.clone());

        let quotient_polynomial = &(&g_i - &expected_ans_polynomial)
            / &Polynomial::vanishing_polynomial(quotient_set.clone());

        let comb_randomness = round_comb_replies[i];

        // New round polynomial f_i
        polynomial = &Polynomial::power_polynomial(comb_randomness, quotient_set.len())
            * &quotient_polynomial;
    }

    let f_2 = polynomial.clone();

    // Final round
    let expected_final_polynomial = fold_polynomial(&f_2, round_r_replies[3], 4);

    assert_eq!(proof.final_polynomial, expected_final_polynomial);
}

// NP TODO failed config creation where the num of rounds is too large for the starting degree and folding factor (maybe in /config)
