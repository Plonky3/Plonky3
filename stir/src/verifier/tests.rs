use crate::{
    prover::{commit, prove},
    test_utils::*,
    utils::fold_polynomial,
    verifier::{
        compute_folded_evaluations,
        error::{FullRoundVerificationError, VerificationError},
        verify,
    },
    StirConfig, StirProof,
};
use itertools::Itertools;
use p3_coset::TwoAdicCoset;
use p3_field::FieldAlgebra;
use p3_poly::test_utils::rand_poly;
use rand::{thread_rng, Rng};
use std::{fs, fs::File, path::Path};

type Proof = StirProof<BBExt, BBExtMMCS, BB>;

fn init_file() -> File {
    let path = Path::new("test_data/proof.json");
    let prefix = path.parent().unwrap();
    fs::create_dir_all(prefix).unwrap();
    File::create(path).unwrap()
}

fn generate_proof_with_config(config: &StirConfig<BBExt, BBExtMMCS>) -> Proof {
    let polynomial = rand_poly((1 << config.log_starting_degree()) - 1);
    let (witness, commitment) = commit(&config, polynomial);
    let proof = prove(&config, witness, commitment, &mut test_bb_challenger());

    // Serialize the proof to a file
    serde_json::to_writer(init_file(), &proof).unwrap();

    proof
}

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
    let config = test_bb_stir_config(20, 2, 4, 3);
    let proof = generate_proof_with_config(&config);

    verify(&config, proof, &mut test_bb_challenger()).unwrap();
}

#[test]
fn test_verify_variable_folding_factor() {
    // NP TODO make bigger after more efficient FFT is introduced
    let config = test_stir_config_folding_factors(14, 1, vec![4, 3, 5]);
    let proof = generate_proof_with_config(&config);

    verify(&config, proof, &mut test_bb_challenger()).unwrap();
}

macro_rules! check_failing_case {
    ($config:expr, $modify_proof:expr, $expected_error:expr) => {{
        let mut verifier_challenger = test_bb_challenger();
        let mut invalid_proof =
            serde_json::from_reader(File::open("test_data/proof.json").unwrap()).unwrap();
        $modify_proof(&mut invalid_proof);
        assert_eq!(
            verify(&$config, invalid_proof, &mut verifier_challenger),
            Err($expected_error)
        );
    }};
}

#[test]
fn test_verify_failing_cases() {
    let mut rng = thread_rng();
    let config = test_bb_stir_config(20, 2, 4, 3);
    let proof = generate_proof_with_config(&config);

    // ---------------- ROUND PROOF OF WORK -----------------

    check_failing_case!(
        config,
        |invalid_proof: &mut Proof| {
            invalid_proof.round_proofs[0].pow_witness = rng.gen();
        },
        VerificationError::Round(0, FullRoundVerificationError::ProofOfWork)
    );

    // ---------------- ROUND QUERY PATH -------------------

    check_failing_case!(
        config,
        |invalid_proof: &mut Proof| {
            let query_proof = proof.round_proofs[0].query_proofs[0].clone();
            let mut invalid_leaf = query_proof.0.clone();
            invalid_leaf[0] = rng.gen();
            let invalid_query_proof = (invalid_leaf, query_proof.1.clone());
            invalid_proof.round_proofs[0].query_proofs[0] = invalid_query_proof;
        },
        VerificationError::Round(0, FullRoundVerificationError::QueryPath)
    );

    // ------------ ANS POLYNOMIAL DEGREE --------------

    check_failing_case!(
        config,
        |invalid_proof: &mut Proof| {
            let original_degree = invalid_proof.round_proofs[0]
                .ans_polynomial
                .degree()
                .unwrap();
            invalid_proof.round_proofs[0].ans_polynomial = rand_poly(original_degree + 1);
        },
        VerificationError::Round(0, FullRoundVerificationError::AnsPolynomialDegree)
    );

    // ---------- ANS POLYNOMIAL EVALUATIONS -----------

    check_failing_case!(
        config,
        |invalid_proof: &mut Proof| {
            let original_degree = invalid_proof.round_proofs[0]
                .ans_polynomial
                .degree()
                .unwrap();
            invalid_proof.round_proofs[0].ans_polynomial = rand_poly(original_degree);
        },
        VerificationError::Round(0, FullRoundVerificationError::AnsPolynomialEvaluations)
    );

    // ----------- FINAL POLYNOMIAL DEGREE -------------

    check_failing_case!(
        config,
        |invalid_proof: &mut Proof| {
            let original_degree = invalid_proof.final_polynomial.degree().unwrap();
            invalid_proof.final_polynomial = rand_poly(original_degree + 1);
        },
        VerificationError::FinalPolynomialDegree
    );

    // --------- FINAL POLYNOMIAL EVALUATIONS -----------

    // TODO NP: To trigger this error, consider replacing the final polynomial
    // with a random one before the challenger can observe it (before verifier.rs: 215)
    // This will allow the path verification to succeed, but the code must fail during the
    // final polynomial evaluation check.

    // --------------- FINAL QUERY PATH -----------------

    check_failing_case!(
        config,
        |invalid_proof: &mut Proof| {
            let query_proof = proof.final_round_queries[0].clone();
            let mut invalid_leaf = query_proof.0.clone();
            invalid_leaf[0] = rng.gen();
            let invalid_query_proof = (invalid_leaf, query_proof.1.clone());
            invalid_proof.final_round_queries[0] = invalid_query_proof;
        },
        VerificationError::FinalQueryPath
    );

    // -------------- FINAL PROOF OF WORK ----------------

    check_failing_case!(
        config,
        |invalid_proof: &mut Proof| {
            invalid_proof.pow_witness = rng.gen();
        },
        VerificationError::FinalProofOfWork
    );
}
// NP TODO test that the sponge is consistent at the end

// Benchmarks with and without the hints
