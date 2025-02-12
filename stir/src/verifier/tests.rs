use core::iter::Iterator;
use std::fs;
use std::fs::File;
use std::path::Path;

use itertools::Itertools;
use p3_challenger::{CanObserve, CanSampleBits, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_coset::TwoAdicCoset;
use p3_field::FieldAlgebra;
use p3_poly::test_utils::rand_poly;
use rand::{thread_rng, Rng};

use crate::config::observe_public_parameters;
use crate::prover::{commit, prove, prove_round, StirRoundWitness};
use crate::test_utils::*;
use crate::utils::{fold_polynomial, observe_ext_slice_with_size};
use crate::verifier::error::{FullRoundVerificationError, VerificationError};
use crate::verifier::{compute_folded_evaluations, verify};
use crate::{Messages, StirConfig, StirProof};

type Proof = StirProof<BBExt, BBExtMMCS, BB>;

fn init_file() -> File {
    let path = Path::new("test_data/proof.json");
    let prefix = path.parent().unwrap();
    fs::create_dir_all(prefix).unwrap();
    File::create(path).unwrap()
}

fn generate_proof_with_config(
    config: &StirConfig<BBExt, BBExtMMCS>,
    challenger: &mut BBChallenger,
) -> Proof {
    let polynomial = rand_poly((1 << config.log_starting_degree()) - 1);
    let (witness, commitment) = commit(&config, polynomial);
    let proof = prove(&config, witness, commitment, challenger);

    // Serialize the proof to a file
    serde_json::to_writer(init_file(), &proof).unwrap();

    proof
}

pub fn test_verify_with_config(config: &StirConfig<BBExt, BBExtMMCS>) {
    let (mut prover_challenger, mut verifier_challenger) =
        (test_bb_challenger(), test_bb_challenger());

    let proof = generate_proof_with_config(config, &mut prover_challenger);
    verify(config, proof, &mut verifier_challenger).unwrap();

    // Check that the sponge is consistent at the end
    assert_eq!(
        prover_challenger.sample_ext_element::<BBExt>(),
        verifier_challenger.sample_ext_element::<BBExt>()
    );
}

macro_rules! test_verify_failing_case {
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
    test_verify_with_config(&config);
}

#[test]
fn test_verify_variable_folding_factor() {
    // NP TODO make bigger after more efficient FFT is introduced
    let config = test_stir_config_folding_factors(14, 1, vec![4, 3, 5]);
    test_verify_with_config(&config);
}

#[test]
fn test_verify_failing_cases() {
    let mut rng = thread_rng();
    let config = test_bb_stir_config(20, 2, 4, 3);
    let proof = generate_proof_with_config(&config, &mut test_bb_challenger());

    // ---------------- ROUND PROOF OF WORK -----------------

    test_verify_failing_case!(
        config,
        |invalid_proof: &mut Proof| {
            invalid_proof.round_proofs[0].pow_witness = rng.gen();
        },
        VerificationError::Round(0, FullRoundVerificationError::ProofOfWork)
    );

    // ---------------- ROUND QUERY PATH -------------------

    test_verify_failing_case!(
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

    test_verify_failing_case!(
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

    test_verify_failing_case!(
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

    test_verify_failing_case!(
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

    let tampered_proof = tamper_with_final_polynomial(&config);
    assert_eq!(
        verify(&config, tampered_proof, &mut test_bb_challenger()),
        Err(VerificationError::FinalPolynomialEvaluations)
    );

    // --------------- FINAL QUERY PATH -----------------

    test_verify_failing_case!(
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

    test_verify_failing_case!(
        config,
        |invalid_proof: &mut Proof| {
            invalid_proof.pow_witness = rng.gen();
        },
        VerificationError::FinalProofOfWork
    );
}

fn tamper_with_final_polynomial(config: &StirConfig<BBExt, BBExtMMCS>) -> Proof {
    let mut challenger = test_bb_challenger();
    let polynomial = rand_poly((1 << config.log_starting_degree()) - 1);
    let (witness, commitment) = commit(&config, polynomial);

    // Observe public parameters
    observe_public_parameters(config.parameters(), &mut challenger);

    // Observe the commitment
    challenger.observe(BB::from_canonical_u8(Messages::Commitment as u8));
    challenger.observe(commitment.clone());

    // Sample the folding randomness
    challenger.observe(BB::from_canonical_u8(Messages::FoldingRandomness as u8));
    let folding_randomness = challenger.sample_ext_element();

    let mut witness = StirRoundWitness {
        domain: witness.domain,
        polynomial: witness.polynomial,
        merkle_tree: witness.merkle_tree,
        round: 0,
        folding_randomness,
    };

    let mut round_proofs = vec![];
    for _ in 0..config.num_rounds() - 1 {
        let (new_witness, round_proof) = prove_round(config, witness, &mut challenger);
        witness = new_witness;
        round_proofs.push(round_proof);
    }

    // Final round
    let log_last_folding_factor = config.log_last_folding_factor();

    // Tamper with the final polynomial
    let final_polynomial = rand_poly(
        witness.polynomial.degree().unwrap() / 2_usize.pow(log_last_folding_factor as u32) as usize,
    );

    let final_queries = config.final_num_queries();

    // Logarithm of |(L_M)^(k_M)|
    let log_query_domain_size = witness.domain.log_size() - log_last_folding_factor;

    // Absorb the final polynomial
    challenger.observe(BB::from_canonical_u8(Messages::FinalPolynomial as u8));
    observe_ext_slice_with_size(&mut challenger, final_polynomial.coeffs());

    // Sample the queried indices
    challenger.observe(BB::from_canonical_u8(Messages::FinalQueryIndices as u8));
    let queried_indices: Vec<u64> = (0..final_queries)
        .map(|_| challenger.sample_bits(log_query_domain_size) as u64)
        .unique()
        .collect();

    let queries_to_final: Vec<(Vec<BBExt>, _)> = queried_indices
        .into_iter()
        .map(|index| {
            config
                .mmcs_config()
                .open_batch(index as usize, &witness.merkle_tree)
        })
        .map(|(mut k, v)| (k.remove(0), v))
        .collect();

    let pow_witness = challenger.grind(config.final_pow_bits());

    StirProof {
        commitment,
        round_proofs,
        final_polynomial,
        pow_witness,
        final_round_queries: queries_to_final,
    }
}

// Benchmarks with and without the hints
