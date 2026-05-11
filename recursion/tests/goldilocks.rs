//! Integration tests for recursive STARK verification over the Goldilocks field.
//!
//! Goldilocks uses a degree-2 extension (D=2), Poseidon2 width-8, and 4-element
//! digests — all distinct from the BabyBear/KoalaBear D=4, width-16, 8-element
//! configurations tested elsewhere.

mod common;

use p3_circuit::CircuitBuilder;
use p3_circuit::ops::{GoldilocksD2Width8, generate_poseidon2_trace, generate_recompose_trace};
use p3_circuit::test_utils::{FibonacciAir, generate_trace_rows};
use p3_fri::FriParameters;
use p3_goldilocks::Poseidon2Goldilocks;
use p3_matrix::Matrix;
use p3_recursion::pcs::fri::{FriVerifierParams, InputProofTargets, MerkleCapTargets, RecValMmcs};
use p3_recursion::pcs::set_fri_mmcs_private_data;
use p3_recursion::public_inputs::StarkVerifierInputsBuilder;
use p3_recursion::{Poseidon2Config, VerificationError, verify_p3_uni_proof_circuit};
use p3_test_utils::goldilocks_params::*;
use p3_uni_stark::{
    prove, prove_with_preprocessed, setup_preprocessed, verify, verify_with_preprocessed,
};
use p3_util::log2_ceil_usize;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::common::{InnerFriGeneric, MulAir};

type InnerFri = InnerFriGeneric<MyConfig, MyHash, MyCompress, DIGEST_ELEMS>;

fn default_goldilocks_poseidon2_8() -> Poseidon2Goldilocks<8> {
    let mut rng = SmallRng::seed_from_u64(1);
    Poseidon2Goldilocks::<8>::new_from_rng_128(&mut rng)
}

fn make_config() -> (MyConfig, Perm, FriVerifierParams) {
    let perm = default_goldilocks_poseidon2_8();
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = MyMmcs::new(hash, compress, 0);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let fri_params = FriParameters::new_testing(challenge_mmcs, 0);

    // Enable MMCS verification
    let fri_verifier_params = FriVerifierParams::with_mmcs(
        fri_params.log_blowup,
        fri_params.log_final_poly_len,
        fri_params.commit_proof_of_work_bits,
        fri_params.query_proof_of_work_bits,
        Poseidon2Config::GoldilocksD2Width8,
    );

    let pcs = MyPcs::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::new(perm.clone());
    let config = MyConfig::new(pcs, challenger);
    (config, perm, fri_verifier_params)
}

/// Verifies a Fibonacci proof recursively over Goldilocks, exercising the
/// full MMCS Merkle-path check inside the verifier circuit.
#[test]
fn test_goldilocks_fibonacci_verifier() -> Result<(), VerificationError> {
    let n = 1 << 3;
    let x = 21u64;

    let (config, perm, fri_verifier_params) = make_config();

    let trace = generate_trace_rows::<F>(0, 1, n);
    let pis = vec![F::ZERO, F::ONE, F::from_u64(x)];
    let air = FibonacciAir {};
    let proof = prove(&config, &air, trace, &pis);
    assert!(verify(&config, &air, &proof, &pis).is_ok());

    let mut circuit_builder = CircuitBuilder::new();
    circuit_builder.enable_poseidon2_perm_width_8::<GoldilocksD2Width8, _>(
        generate_poseidon2_trace::<Challenge, GoldilocksD2Width8>,
        perm,
    );
    circuit_builder.enable_recompose::<F>(generate_recompose_trace::<F, Challenge>);

    // Allocate all targets
    let verifier_inputs = StarkVerifierInputsBuilder::<
        MyConfig,
        MerkleCapTargets<F, DIGEST_ELEMS>,
        InnerFri,
    >::allocate(&mut circuit_builder, &proof, None, pis.len());

    // Add the verification circuit to the builder.
    let mmcs_op_ids = verify_p3_uni_proof_circuit::<
        FibonacciAir,
        MyConfig,
        MerkleCapTargets<F, DIGEST_ELEMS>,
        InputProofTargets<F, Challenge, RecValMmcs<F, DIGEST_ELEMS, MyHash, MyCompress>>,
        InnerFri,
        _,
        WIDTH,
        RATE,
    >(
        &config,
        &air,
        &mut circuit_builder,
        &verifier_inputs.proof_targets,
        &verifier_inputs.air_public_targets,
        &None,
        &fri_verifier_params,
        Poseidon2Config::GoldilocksD2Width8,
    )?;

    // Build the circuit.
    let circuit = circuit_builder.build()?;

    let mut runner = circuit.runner();

    // Pack values using the same builder
    let (public_inputs, private_inputs) = verifier_inputs.pack_values(&pis, &proof, &None);

    runner
        .set_public_inputs(&public_inputs)
        .map_err(VerificationError::Circuit)?;
    runner
        .set_private_inputs(&private_inputs)
        .map_err(VerificationError::Circuit)?;

    // Set MMCS private data from the FRI proof
    set_fri_mmcs_private_data::<
        F,
        Challenge,
        ChallengeMmcs,
        MyMmcs,
        MyHash,
        MyCompress,
        DIGEST_ELEMS,
    >(&mut runner, &mmcs_op_ids, &proof.opening_proof)
    .map_err(|e| VerificationError::InvalidProofShape(e.to_string()))?;

    runner.run().map_err(VerificationError::Circuit)?;

    Ok(())
}

/// Verifies a MulAir proof with preprocessed columns recursively over Goldilocks,
/// confirming that the preprocessed commitment target is handled correctly.
#[test]
fn test_goldilocks_mul_verifier_with_preprocessed() -> Result<(), VerificationError> {
    let n = 1 << 3;

    let (_config, perm, _) = make_config();

    // Use FriVerifierParams::from to skip MMCS verification (arithmetic only),
    // matching the pattern in mul_air.rs.
    let (config2, _, fri_verifier_params) = {
        let perm2 = default_goldilocks_poseidon2_8();
        let hash2 = MyHash::new(perm2.clone());
        let compress2 = MyCompress::new(perm2.clone());
        let val_mmcs2 = MyMmcs::new(hash2, compress2, 0);
        let challenge_mmcs2 = ChallengeMmcs::new(val_mmcs2.clone());
        let fri_params2 = FriParameters::new_testing(challenge_mmcs2, 0);
        let fri_verifier_params = FriVerifierParams::from(&fri_params2);
        let pcs2 = MyPcs::new(Dft::default(), val_mmcs2, fri_params2);
        let challenger2 = Challenger::new(perm2.clone());
        (MyConfig::new(pcs2, challenger2), perm2, fri_verifier_params)
    };

    let air = MulAir { degree: 2, rows: n };
    let (trace, _) = air.random_valid_trace::<F>(true);

    // Setup preprocessed data
    let (preprocessed_prover_data, preprocessed_vk) =
        setup_preprocessed(&config2, &air, log2_ceil_usize(trace.height())).unzip();

    // Generate and verify proof
    let proof = prove_with_preprocessed(
        &config2,
        &air,
        trace,
        &[],
        preprocessed_prover_data.as_ref(),
    );
    assert!(
        verify_with_preprocessed(&config2, &air, &proof, &[], preprocessed_vk.as_ref()).is_ok()
    );

    let mut circuit_builder = CircuitBuilder::new();
    circuit_builder.enable_poseidon2_perm_width_8::<GoldilocksD2Width8, _>(
        generate_poseidon2_trace::<Challenge, GoldilocksD2Width8>,
        perm,
    );
    circuit_builder.enable_recompose::<F>(generate_recompose_trace::<F, Challenge>);

    // Allocate all targets
    let verifier_inputs = StarkVerifierInputsBuilder::<
        MyConfig,
        MerkleCapTargets<F, DIGEST_ELEMS>,
        InnerFri,
    >::allocate(
        &mut circuit_builder,
        &proof,
        preprocessed_vk.as_ref().map(|vk| &vk.commitment),
        0,
    );

    // Add the verification circuit to the builder
    verify_p3_uni_proof_circuit::<_, _, _, _, _, _, WIDTH, RATE>(
        &config2,
        &air,
        &mut circuit_builder,
        &verifier_inputs.proof_targets,
        &verifier_inputs.air_public_targets,
        &verifier_inputs.preprocessed_commit,
        &fri_verifier_params,
        Poseidon2Config::GoldilocksD2Width8,
    )?;

    // Build the circuit
    let circuit = circuit_builder.build()?;

    let mut runner = circuit.runner();

    // Pack values using the same builder
    let (public_inputs, private_inputs) =
        verifier_inputs.pack_values(&[], &proof, &preprocessed_vk.map(|vk| vk.commitment));

    runner
        .set_public_inputs(&public_inputs)
        .map_err(VerificationError::Circuit)?;
    runner
        .set_private_inputs(&private_inputs)
        .map_err(VerificationError::Circuit)?;

    runner.run().map_err(VerificationError::Circuit)?;

    Ok(())
}
