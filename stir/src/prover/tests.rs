use crate::{
    coset::Radix2Coset,
    polynomial::{rand_poly, Polynomial},
    proof::RoundProof,
    prover::prove,
    test_utils::*,
    utils::{field_element_from_isize, fold_polynomial},
    StirProof,
};
use itertools::Itertools;
use p3_baby_bear::BabyBear;
use p3_challenger::MockChallenger;
use p3_commit::Mmcs;
use p3_field::FieldAlgebra;
use p3_matrix::{dense::RowMajorMatrix, Dimensions, Matrix};
use rand::Rng;

use super::{prove_round, RoundConfig, StirWitness};

type BB = BabyBear;

// NP TODO either remove the manual values or use them by reducing the soundness
#[test]
fn test_prove_round_zero() {
    let config = test_stir_config(3, 1, 1, 2);

    let round = 0;

    let round_config = config.round_config(round);

    let RoundConfig {
        num_ood_samples,
        num_queries,
        log_folding_factor,
        ..
    } = round_config.clone();

    let field_replies = [
        // ood_samples
        (0..num_ood_samples)
            .map(|x| BB::from_canonical_usize(3) * BB::from_canonical_usize(x))
            .collect_vec(),
        vec![
            // comb_randomness
            BB::ONE,
            // folding_randomness
            BB::ONE,
            // shake_randomness (unused)
            BB::ONE,
        ],
    ]
    .concat();

    // indices

    let log_size_second_codeword = config.log_starting_degree() + config.log_starting_inv_rate()
        - config.log_starting_folding_factor();

    let bit_replies = (0..num_queries)
        .map(|i| i % (1 << log_size_second_codeword))
        .collect::<Vec<_>>();

    let mut challenger = MockChallenger::new(field_replies, bit_replies.clone());

    // Starting polynomial: -2 + 17x + 42x^2 + 3x^3 - x^4 - x^5 + 4x^6 + 5x^7
    let coeffs: Vec<BB> = vec![-2, 17, 42, 3, -1, -1, 4, 5]
        .into_iter()
        .map(field_element_from_isize)
        .collect_vec();

    let f = Polynomial::from_coeffs(coeffs);

    let original_domain = Radix2Coset::new(
        BB::ONE,
        config.log_starting_degree() + config.log_starting_inv_rate(),
    );

    let original_domain = original_domain.set_shift(original_domain.generator());

    let original_evals = original_domain.evaluate_polynomial(&f);

    let stacked_original_evals =
        RowMajorMatrix::new(original_evals, 1 << config.log_starting_folding_factor());

    let (root, merkle_tree) = config
        .mmcs_config()
        .commit_matrix(stacked_original_evals.clone());

    let witness = StirWitness {
        domain: original_domain.clone(),
        polynomial: f,
        merkle_tree,
        stacked_evals: stacked_original_evals,
        round,
        folding_randomness: BB::from_canonical_usize(2),
    };

    let (witness, round_proof) = prove_round(&config, witness, &mut challenger);

    // =============== Witness Checks ===============
    let expected_domain = original_domain.shrink_subgroup(1);

    let expected_round = 1;
    let expected_folding_randomness = BB::ONE;

    let StirWitness {
        domain,
        polynomial,
        folding_randomness,
        round,
        ..
    } = witness;

    // Domain testing
    assert_eq!(domain, expected_domain);
    assert_eq!(folding_randomness, expected_folding_randomness);

    // Round-number testing
    assert_eq!(round, expected_round);

    // Polynomial testing In this case, the security level means the
    // interpolator has the same degree as the folded polynomial
    assert!(polynomial.is_zero());

    // Polynomial-evaluation testing
    assert!(domain.iter().all(|x| polynomial.evaluate(&x) == BB::ZERO));

    // ============== Round Proof Checks ===============

    let RoundProof { query_proofs, .. } = round_proof;

    for (&i, (leaf, proof)) in bit_replies.iter().unique().zip(query_proofs) {
        config
            .mmcs_config()
            .verify_batch(
                &root,
                &[Dimensions {
                    width: 1 << log_folding_factor,
                    height: 1 << (original_domain.log_size() - log_folding_factor),
                }],
                i,
                &vec![leaf],
                &proof,
            )
            .unwrap();
    }
}

#[test]
fn test_prove_round_large() {
    let mut rng = rand::thread_rng();

    let config = test_stir_config(10, 3, 2, 2);

    let round = 0;

    let round_config = config.round_config(round);

    let RoundConfig {
        log_folding_factor,
        log_inv_rate,
        num_ood_samples,
        num_queries,
        ..
    } = round_config.clone();

    let r_0: BB = rng.gen(); // Initial folding randomness

    // Field randomness produced by the sponge
    let r_1: BB = rng.gen(); // Folding randomness for round 2 (which never happens)
    let ood_randomness: Vec<BB> = (0..num_ood_samples).map(|_| rng.gen()).collect();
    let comb_randomness = rng.gen();
    let _shake_randomness = rng.gen();

    let mut field_replies = ood_randomness.clone();
    field_replies.push(comb_randomness);
    field_replies.push(r_1);
    field_replies.push(_shake_randomness);

    // Index randomness
    let log_size_second_codeword = config.log_starting_degree() - log_folding_factor + log_inv_rate;

    let bit_replies = (0..num_queries)
        .map(|_| rng.gen())
        .map(|i: usize| i % (1 << log_size_second_codeword))
        .collect::<Vec<_>>();

    // Preloading fake randomness
    let mut challenger = MockChallenger::new(field_replies, bit_replies.clone());

    // Starting polynomial
    let f_0 = rand_poly((1 << config.log_starting_degree()) - 1);

    let original_domain = Radix2Coset::new(
        BB::ONE,
        config.log_starting_degree() + config.log_starting_inv_rate(),
    );

    let original_domain = original_domain.set_shift(original_domain.generator());

    let original_evals = original_domain.evaluate_polynomial(&f_0);

    // NP TODO remove
    // for (i, e) in original_evals.iter().enumerate() {
    //     println!("{}: {}", i, e);
    // }
    // let new_original_domain = original_domain.set_shift(original_domain.generator());
    // let new_original_evals = new_original_domain.evaluate_polynomial(&f_0);
    // assert_eq!(original_evals, new_original_evals);

    let stacked_original_evals =
        RowMajorMatrix::new(original_evals, 1 << config.log_starting_folding_factor());

    let dimensions = stacked_original_evals.dimensions();

    let (root, merkle_tree) = config
        .mmcs_config()
        .commit_matrix(stacked_original_evals.clone());

    let witness = StirWitness {
        domain: original_domain.clone(),
        polynomial: f_0.clone(),
        merkle_tree,
        stacked_evals: stacked_original_evals,
        round,
        folding_randomness: r_0,
    };

    let (witness, round_proof) = prove_round(&config, witness, &mut challenger);

    // =============== Witness Checks ===============

    let StirWitness {
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

    // Polynomial testing
    let g_1 = fold_polynomial(&f_0, r_0, log_folding_factor);

    // NP TODO repeat points
    let original_domain_pow_k = original_domain.shrink_coset(log_folding_factor);
    let stir_randomness = bit_replies
        .iter()
        .map(|&i| original_domain_pow_k.element(i as u64));

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
            let (quotient, _) = Polynomial::divide_by_vanishing_linear_polynomial(
                &(&ans_polynomial - &Polynomial::constant(y)),
                x,
            );
            quotient
        })
        .fold(Polynomial::zero(), |sum, next_poly| &sum + &next_poly);

    assert_eq!(ans_polynomial, expected_ans_polynomial);
    assert_eq!(shake_polynomial, expected_shake_polynomial);
}

#[test]
fn test_prove() {
    // Note this performs no checks against the expected final polynomial - it
    // is only meant to check prove() runs from beginning to end.

    let config = test_stir_config(14, 1, 4, 3);

    let polynomial = rand_poly((1 << config.log_starting_degree()) - 1);

    let mut challenger = test_challenger();

    let proof = prove(&config, polynomial, &mut challenger);

    // Final-degree testing
    assert_eq!(config.log_stopping_degree(), 2);
    assert!(proof.final_polynomial.degree() < 1 << 2);
}

// NP TODO test two subsequent rounds by hand

// NP TODO discuss with Giacomo Every round needs two: this round's, to know how
// to fold; and next round's, to know how to stack the evaluations of the final
// polynomial f' produced by this round

// NP TODO polynomial with degree strictly lower than the bound

// NP TODO failed config creation where the num of rounds is too large for the starting degree and folding factor (maybe in /config)
