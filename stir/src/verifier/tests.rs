use alloc::vec;
use alloc::vec::Vec;
use core::iter::Iterator;

use itertools::Itertools;
use p3_challenger::{CanObserve, CanSampleBits, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_coset::TwoAdicCoset;
use p3_field::PrimeCharacteristicRing;
use p3_poly::test_utils::rand_poly;
use p3_symmetric::Hash;
use rand::{rng, Rng};

use crate::config::observe_public_parameters;
use crate::prover::{commit, prove, prove_round, StirRoundWitness};
use crate::test_utils::*;
use crate::utils::{fold_polynomial, observe_ext_slice_with_size};
use crate::verifier::error::{FullRoundVerificationError, VerificationError};
use crate::verifier::{compute_folded_evaluations, verify};
use crate::{Messages, SecurityAssumption, StirConfig, StirProof};

type BBProof = StirProof<BBExt, BBExtMMCS, BB>;
type GLProof = StirProof<GLExt, GLExtMMCS, GL>;

// This macro creates a function that commits to a random polynomial and
// produces a STIR proof for it given a configuration
macro_rules! impl_generate_proof_with_config {
    (
        // Name of the function to create
        $name:ident,
        // MMCS
        $ext_mmcs:ty,
        // Type of the proof
        $proof_type:ty,
        // Type of the commitment
        $commitment_type:ty,
        // Type of the challenger
        $challenger:ty
    ) => {
        pub fn $name(
            config: &StirConfig<$ext_mmcs>,
            challenger: &mut $challenger,
        ) -> ($proof_type, $commitment_type) {
            let polynomial = rand_poly((1 << config.log_starting_degree()) - 1);
            let (witness, commitment) = commit(&config, polynomial);
            (
                prove(&config, witness, commitment, challenger),
                commitment.clone(),
            )
        }
    };
}

// This macro creates a function that tests the verification of a STIR proof
// given a configuration. It also checks that the prover and verifier
// challengers are synchronised at the end of the proving/verification process.
macro_rules! impl_test_verify_with_config {
    (
        // Name of the function to create
        $name:ident,
        // Field over which STIR takes place
        $ext:ty,
        // MMCS
        $ext_mmcs:ty,
        // Challenger function
        $challenger_fn:ident,
        // Name of the function which generates the proof
        $proof_fn:ident
    ) => {
        pub fn $name(config: &StirConfig<$ext_mmcs>) {
            let (mut prover_challenger, mut verifier_challenger) =
                ($challenger_fn(), $challenger_fn());

            let (proof, commitment) = $proof_fn(config, &mut prover_challenger);
            verify(config, commitment, proof, &mut verifier_challenger).unwrap();

            // Check that the sponge is consistent at the end
            assert_eq!(
                prover_challenger.sample_algebra_element::<$ext>(),
                verifier_challenger.sample_algebra_element::<$ext>()
            );
        }
    };
}

// Create the function generate_bb_proof_with_config
impl_generate_proof_with_config!(
    generate_bb_proof_with_config,
    BBExtMMCS,
    BBProof,
    Hash<BB, BB, 8>,
    BBChallenger
);

// Create the function generate_gl_proof_with_config
impl_generate_proof_with_config!(
    generate_gl_proof_with_config,
    GLExtMMCS,
    GLProof,
    Hash<GL, GL, 4>,
    GLChallenger
);

// Create the function test_bb_verify_with_config
impl_test_verify_with_config!(
    test_bb_verify_with_config,
    BBExt,
    BBExtMMCS,
    test_bb_challenger,
    generate_bb_proof_with_config
);

// Create the function test_gl_verify_with_config
impl_test_verify_with_config!(
    test_gl_verify_with_config,
    GLExt,
    GLExtMMCS,
    test_gl_challenger,
    generate_gl_proof_with_config
);

// Auxiliary function to trigger a tricky verification error which mimics the
// honest proving procedure but modifies the final polynomial near the end.
fn tamper_with_final_polynomial(config: &StirConfig<BBExtMMCS>) -> (BBProof, Hash<BB, BB, 8>) {
    // ========================== Honest proving =============================

    // This is documented in prover.rs
    let mut challenger = test_bb_challenger();
    let polynomial = rand_poly((1 << config.log_starting_degree()) - 1);
    let (witness, commitment) = commit(&config, polynomial);

    observe_public_parameters(config.parameters(), &mut challenger);

    // Observe the commitment
    challenger.observe(BB::from_u8(Messages::Commitment as u8));
    challenger.observe(commitment.clone());

    // Sample the folding randomness
    challenger.observe(BB::from_u8(Messages::FoldingRandomness as u8));
    let folding_randomness = challenger.sample_algebra_element();

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

    let log_last_folding_factor = config.log_last_folding_factor();

    // ===================== Dishonest final polynomial ========================

    let final_polynomial = rand_poly(
        witness.polynomial.degree().unwrap() / 2_usize.pow(log_last_folding_factor as u32) as usize,
    );

    // ===================== Continuing honest proving ========================
    let final_queries = config.final_num_queries();

    let log_query_domain_size = witness.domain.log_size() - log_last_folding_factor;

    // Absorb the final polynomial
    challenger.observe(BB::from_u8(Messages::FinalPolynomial as u8));
    observe_ext_slice_with_size(&mut challenger, final_polynomial.coeffs());

    // Sample the queried indices
    challenger.observe(BB::from_u8(Messages::FinalQueryIndices as u8));
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

    (
        StirProof {
            round_proofs,
            final_polynomial,
            pow_witness,
            final_round_queries: queries_to_final,
        },
        commitment,
    )
}

#[test]
// Check that compute_folded_evaluations returns the correct result by comparing
// it with the result of evaluating the output of fold_polynomial
fn test_compute_folded_evals() {
    let log_arity = 11;

    let poly_degree = 42;
    let polynomial = rand_poly(poly_degree);

    let mut rng = rng();

    let root: BBExt = rng.random();
    let c: BBExt = rng.random();

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
// Check that verification of a honest proof over the quintic extension of
// BabyBear with fixed folding factor 2^4 works
fn test_bb_verify() {
    let config = test_bb_stir_config(
        BB_EXT_SEC_LEVEL,
        SecurityAssumption::CapacityBound,
        20,
        2,
        4,
        3,
    );
    test_bb_verify_with_config(&config);
}

#[test]
// Check that verification of a honest proof over the quintic extension of
// BabyBear with variable folding factor works
fn test_bb_verify_variable_folding_factor() {
    let config = test_bb_stir_config_folding_factors(
        BB_EXT_SEC_LEVEL,
        SecurityAssumption::CapacityBound,
        20,
        1,
        vec![4, 3, 5],
    );
    test_bb_verify_with_config(&config);
}

#[test]
// Check that verification of a honest proof over the quintic extension of
// BabyBear with variable folding factor works when the unconditionally true
// security configuration JohnsonBound is used. This requires lowering the bits
// of security to 100 in order to get prohibitively high proof-of-work bit
// numbers.
fn test_bb_verify_variable_folding_factor_unconditional() {
    let config = test_bb_stir_config_folding_factors(
        BB_EXT_SEC_LEVEL_LOWER,
        SecurityAssumption::JohnsonBound,
        16,
        1,
        vec![4, 3, 5],
    );
    test_bb_verify_with_config(&config);
}

#[test]
// Check that verification of a honest proof over the quartic extension of
// Goldilocks with fixed folding factor 2^4 works. This requires lowering the
// bits of security to 80 in order to get prohibitively high proof-of-work bit
// numbers.
fn test_gl_verify() {
    let config = test_gl_stir_config(
        GL_EXT_SEC_LEVEL,
        SecurityAssumption::JohnsonBound,
        16,
        1,
        4,
        3,
    );
    test_gl_verify_with_config(&config);
}

#[test]
// Check that verification of a honest proof over the quartic extension of
// Goldilocks with variable folding factor works. As above, we need to lower
// the bits of security to 80.
fn test_gl_verify_variable_folding_factor() {
    let config = test_gl_stir_config_folding_factors(
        GL_EXT_SEC_LEVEL,
        SecurityAssumption::JohnsonBound,
        15,
        2,
        vec![2, 3, 2, 1, 2],
    );
    test_gl_verify_with_config(&config);
}

#[test]
// Check that the warning "The quotient polynomial is zero" is logged correctly
// (cf. prover.rs or verifier.rs for more details)
fn test_verify_zero() {
    tracing_subscriber::fmt::init();

    // Since we deliberately use a small polynomial to trigger the cancellation
    // of g_i, we drastically reduce the security level to avoid getting the
    // unrelated warning "The configuration requires the prover to compute a proof
    // of work of more than 25 bits"
    let security_level = 60;

    let config = test_bb_stir_config(
        security_level,
        SecurityAssumption::CapacityBound,
        8,
        2,
        2,
        3,
    );
    test_bb_verify_with_config(&config);
}

#[test]
// Check that proofs can be serialized and deserialized, then verified correctly
fn test_serialize_deserialize_proof() {
    let config = test_bb_stir_config(
        BB_EXT_SEC_LEVEL,
        SecurityAssumption::CapacityBound,
        20,
        2,
        4,
        3,
    );
    let (proof, commitment) = generate_bb_proof_with_config(&config, &mut test_bb_challenger());

    let serialized_proof = serde_json::to_string(&proof).unwrap();
    let deserialized_proof: BBProof = serde_json::from_str(&serialized_proof).unwrap();

    assert!(verify(
        &config,
        commitment,
        deserialized_proof,
        &mut test_bb_challenger()
    )
    .is_ok());
}

#[test]
// Check that each possible VerificationError is triggered correctly by
// producing various dishonest proofs
fn test_verify_failing_cases() {
    let mut rng = rng();
    let config = test_bb_stir_config(
        BB_EXT_SEC_LEVEL,
        SecurityAssumption::CapacityBound,
        20,
        2,
        4,
        3,
    );

    // This is the honest proof we will tamper with
    let (proof, commitment) = generate_bb_proof_with_config(&config, &mut test_bb_challenger());

    // ============================== ProofOfWork ==============================

    let mut invalid_proof = proof.clone();
    invalid_proof.round_proofs[0].pow_witness = rng.random();

    assert_eq!(
        verify(
            &config,
            commitment,
            invalid_proof,
            &mut test_bb_challenger()
        ),
        Err(VerificationError::Round(
            1,
            FullRoundVerificationError::ProofOfWork
        ))
    );

    // ============================== QueryPath ===============================

    let mut invalid_proof = proof.clone();
    let query_proof = proof.round_proofs[0].query_proofs[0].clone();
    let mut invalid_leaf = query_proof.0.clone();
    invalid_leaf[0] = rng.random();
    let invalid_query_proof = (invalid_leaf, query_proof.1.clone());
    invalid_proof.round_proofs[0].query_proofs[0] = invalid_query_proof;

    assert_eq!(
        verify(
            &config,
            commitment,
            invalid_proof,
            &mut test_bb_challenger()
        ),
        Err(VerificationError::Round(
            1,
            FullRoundVerificationError::QueryPath
        ))
    );

    // ========================== AnsPolynomialDegree ==========================

    let mut invalid_proof = proof.clone();
    let original_degree = invalid_proof.round_proofs[0]
        .ans_polynomial
        .degree()
        .unwrap();
    invalid_proof.round_proofs[0].ans_polynomial = rand_poly(original_degree + 1);

    assert_eq!(
        verify(
            &config,
            commitment,
            invalid_proof,
            &mut test_bb_challenger()
        ),
        Err(VerificationError::Round(
            1,
            FullRoundVerificationError::AnsPolynomialDegree
        ))
    );

    // ======================= AnsPolynomialEvaluations =======================

    let mut invalid_proof = proof.clone();
    let original_degree = invalid_proof.round_proofs[0]
        .ans_polynomial
        .degree()
        .unwrap();
    invalid_proof.round_proofs[0].ans_polynomial = rand_poly(original_degree);

    assert_eq!(
        verify(
            &config,
            commitment,
            invalid_proof,
            &mut test_bb_challenger()
        ),
        Err(VerificationError::Round(
            1,
            FullRoundVerificationError::AnsPolynomialEvaluations
        ))
    );

    // ========================= FinalPolynomialDegree =========================

    let mut invalid_proof = proof.clone();
    let original_degree = invalid_proof.final_polynomial.degree().unwrap();
    invalid_proof.final_polynomial = rand_poly(original_degree + 1);

    assert_eq!(
        verify(
            &config,
            commitment,
            invalid_proof,
            &mut test_bb_challenger()
        ),
        Err(VerificationError::FinalPolynomialDegree)
    );

    // ====================== FinalPolynomialEvaluations ======================

    // This case is substantially more difficult to trigger, as in the protocol,
    // the final polynomial is observed by the challenger *before* the queried
    // indices are sampled. If we modify the final polynomial after proof
    // generation, we'll always trigger a `FinalQueryPath` error instead of
    // `FinalPolynomialEvaluations`. This is because the queried indices (and
    // thus, the queried paths) are sampled by the verifier after the original
    // (untampered) polynomial has been observed by the challenger, but the
    // original prover sampled them after having its challenger observe the
    // honest final polynomial. To properly test this error case, we need to
    // tamper with the final polynomial during the proof-generation process
    // itself, before the indices are sampled. The
    // tamper_with_final_polynomial() function does this by injecting a random
    // polynomial at the correct point in the protocol flow.

    let (tampered_proof, _commitment) = tamper_with_final_polynomial(&config);
    assert_eq!(
        verify(
            &config,
            _commitment,
            tampered_proof,
            &mut test_bb_challenger()
        ),
        Err(VerificationError::FinalPolynomialEvaluations)
    );

    // ============================ FinalQueryPath ============================

    let mut invalid_proof = proof.clone();
    let query_proof = proof.final_round_queries[0].clone();
    let mut invalid_leaf = query_proof.0.clone();
    invalid_leaf[0] = rng.random();
    let invalid_query_proof = (invalid_leaf, query_proof.1.clone());
    invalid_proof.final_round_queries[0] = invalid_query_proof;

    assert_eq!(
        verify(
            &config,
            commitment,
            invalid_proof,
            &mut test_bb_challenger()
        ),
        Err(VerificationError::FinalQueryPath)
    );

    // =========================== FinalProofOfWork ===========================

    let mut invalid_proof = proof.clone();
    invalid_proof.pow_witness = rng.random();

    assert_eq!(
        verify(
            &config,
            commitment,
            invalid_proof,
            &mut test_bb_challenger()
        ),
        Err(VerificationError::FinalProofOfWork)
    );
}
