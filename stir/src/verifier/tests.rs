use crate::{
    prover::{commit, prove},
    test_utils::*,
    utils::fold_polynomial,
    verifier::{compute_folded_evaluations, verify},
};
use itertools::Itertools;
use p3_coset::TwoAdicCoset;
use p3_field::FieldAlgebra;
use p3_poly::test_utils::rand_poly;
use rand::thread_rng;
use rand::Rng;

#[test]
fn test_compute_folded_evals() {
    let log_arity = 11;

    let poly_degree = 42;
    let polynomial = rand_poly(poly_degree);

    // TODO change thread_rngs to test RNGs which are deterministic
    let mut rng = thread_rng();

    let root: BBExt = rng.gen();
    let c: BBExt = rng.gen();

    let domain = TwoAdicCoset::new(root, log_arity);

    let evaluations = vec![domain.iter().map(|x| polynomial.evaluate(&x)).collect_vec()];

    let folded_eval =
        compute_folded_evaluations(evaluations, &[root], log_arity, c, domain.generator())
            .pop()
            .unwrap();

    let expected_folded_eval =
        fold_polynomial(&polynomial, c, log_arity).evaluate(&root.exp_power_of_2(log_arity));

    assert_eq!(folded_eval, expected_folded_eval);
}

#[test]
fn test_verify() {
    // NP TODO make bigger after more efficient FFT is introduced
    let config = test_bb_stir_config(10, 1, 4, 2);

    let polynomial = rand_poly((1 << config.log_starting_degree()) - 1);

    let mut prover_challenger = test_bb_challenger();
    let mut verifier_challenger = prover_challenger.clone();

    let (witness, commitment) = commit(&config, polynomial);

    let proof = prove(&config, witness, commitment, &mut prover_challenger);
    verify(&config, proof, &mut verifier_challenger).unwrap();
}

#[test]
fn test_verify_variable_folding_factor() {
    // NP TODO make bigger after more efficient FFT is introduced
    let config = test_stir_config_folding_factors(10, 1, vec![4, 5]);

    let polynomial = rand_poly((1 << config.log_starting_degree()) - 1);

    let mut prover_challenger = test_bb_challenger();
    let mut verifier_challenger = prover_challenger.clone();

    let (witness, commitment) = commit(&config, polynomial);

    // NP TODO remove
    println!("Proving");

    let proof = prove(&config, witness, commitment, &mut prover_challenger);

    // NP TODO remove
    println!("Verifying");

    verify(&config, proof, &mut verifier_challenger).unwrap();
}

// NP TODO failing tests

// NP TODO test that the sponge is consistent at the end
