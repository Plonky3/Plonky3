use crate::{
    coset::Radix2Coset,
    polynomial::rand_poly,
    prover::prove,
    test_utils::{test_challenger, test_stir_config, test_stir_config_folding_factors},
    utils::fold_polynomial,
    verifier::{compute_folded_evaluations, verify},
};
use itertools::Itertools;
use p3_baby_bear::BabyBear;
use p3_field::FieldAlgebra;
use rand::thread_rng;
use rand::Rng;

type BB = BabyBear;

#[test]
fn test_compute_folded_evals() {
    let log_arity = 11;

    let poly_degree = 42;
    let polynomial = rand_poly(poly_degree);

    // TODO change thread_rngs to test RNGs which are deterministic
    let mut rng = thread_rng();

    let root: BB = rng.gen();
    let c: BB = rng.gen();

    let domain = Radix2Coset::new(root, log_arity);

    let evaluations = vec![domain.iter().map(|x| polynomial.evaluate(&x)).collect_vec()];

    let folded_eval =
        compute_folded_evaluations(evaluations, &[root], log_arity, c, domain.generator())
            .pop()
            .unwrap();

    let expected_folded_eval =
        fold_polynomial(&polynomial, c, log_arity).evaluate(&root.exp_power_of_2(log_arity));

    assert_eq!(folded_eval, expected_folded_eval,);
}

#[test]
fn test_verify() {
    let config = test_stir_config(14, 1, 4, 3);

    let polynomial = rand_poly((1 << config.log_starting_degree()) - 1);

    let mut prover_challenger = test_challenger();
    let mut verifier_challenger = prover_challenger.clone();

    let proof = prove(&config, polynomial, &mut prover_challenger);
    assert!(verify(&config, proof, &mut verifier_challenger));
}

#[test]
fn test_verify_variable_folding_factor() {
    let config = test_stir_config_folding_factors(14, 1, vec![5, 4, 3]);

    let polynomial = rand_poly((1 << config.log_starting_degree()) - 1);

    let mut prover_challenger = test_challenger();
    let mut verifier_challenger = prover_challenger.clone();

    let proof = prove(&config, polynomial, &mut prover_challenger);
    assert!(verify(&config, proof, &mut verifier_challenger));
}
