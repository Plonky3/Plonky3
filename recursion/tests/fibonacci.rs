mod common;

use p3_baby_bear::default_babybear_poseidon2_16;
use p3_circuit::CircuitBuilder;
use p3_circuit::ops::{generate_poseidon2_trace, generate_recompose_trace};
use p3_circuit::test_utils::{FibonacciAir, generate_trace_rows};
use p3_field::PrimeCharacteristicRing;
use p3_fri::FriParameters;
use p3_poseidon2_circuit_air::BabyBearD4Width16;
use p3_recursion::pcs::fri::{FriVerifierParams, InputProofTargets, MerkleCapTargets, RecValMmcs};
use p3_recursion::pcs::set_fri_mmcs_private_data;
use p3_recursion::public_inputs::StarkVerifierInputsBuilder;
use p3_recursion::{Poseidon2Config, VerificationError, verify_p3_uni_proof_circuit};
use p3_test_utils::baby_bear_params::*;
use p3_uni_stark::{prove, verify};
use tracing_forest::ForestLayer;
use tracing_forest::util::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

use crate::common::InnerFriGeneric;

type InnerFri = InnerFriGeneric<MyConfig, MyHash, MyCompress, DIGEST_ELEMS>;

fn init_logger() {
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();
}

struct FibonacciTestSetup {
    config: MyConfig,
    perm: Perm,
    fri_verifier_params: FriVerifierParams,
    proof: p3_uni_stark::Proof<MyConfig>,
    pis: Vec<F>,
    air: FibonacciAir,
}

fn build_fibonacci_test_setup() -> FibonacciTestSetup {
    let n = 1 << 3;
    let x = 21;

    let perm = default_babybear_poseidon2_16();
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = MyMmcs::new(hash, compress, 0);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let trace = generate_trace_rows::<F>(0, 1, n);
    let log_final_poly_len = 0;
    let fri_params = FriParameters::new_testing(challenge_mmcs, log_final_poly_len);

    // Enable MMCS verification
    let fri_verifier_params = FriVerifierParams::with_mmcs(
        fri_params.log_blowup,
        fri_params.log_final_poly_len,
        fri_params.commit_proof_of_work_bits,
        fri_params.query_proof_of_work_bits,
        Poseidon2Config::BabyBearD4Width16,
    );

    let pcs = MyPcs::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::new(perm.clone());
    let config = MyConfig::new(pcs, challenger);
    let pis = vec![F::ZERO, F::ONE, F::from_u64(x)];
    let air = FibonacciAir {};
    let proof = prove(&config, &air, trace, &pis);

    FibonacciTestSetup {
        config,
        perm,
        fri_verifier_params,
        proof,
        pis,
        air,
    }
}

fn run_recursive_verifier(
    setup: &FibonacciTestSetup,
    proof: &p3_uni_stark::Proof<MyConfig>,
    pis: &[F],
) -> Result<(), VerificationError> {
    let mut circuit_builder = CircuitBuilder::new();
    circuit_builder.enable_poseidon2_perm::<BabyBearD4Width16, _>(
        generate_poseidon2_trace::<Challenge, BabyBearD4Width16>,
        setup.perm.clone(),
    );
    circuit_builder.enable_recompose::<F>(generate_recompose_trace::<F, Challenge>);

    // Allocate all targets
    let verifier_inputs = StarkVerifierInputsBuilder::<
        MyConfig,
        MerkleCapTargets<F, DIGEST_ELEMS>,
        InnerFri,
    >::allocate(&mut circuit_builder, proof, None, pis.len());

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
        &setup.config,
        &setup.air,
        &mut circuit_builder,
        &verifier_inputs.proof_targets,
        &verifier_inputs.air_public_targets,
        &None,
        &setup.fri_verifier_params,
        Poseidon2Config::BabyBearD4Width16,
    )?;

    // Build the circuit.
    let circuit = circuit_builder.build()?;

    let mut runner = circuit.runner();

    // Pack values using the same builder
    let (public_inputs, private_inputs) = verifier_inputs.pack_values(pis, proof, &None);
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

#[test]
fn test_fibonacci_verifier() -> Result<(), VerificationError> {
    init_logger();
    let setup = build_fibonacci_test_setup();
    assert!(verify(&setup.config, &setup.air, &setup.proof, &setup.pis).is_ok());
    run_recursive_verifier(&setup, &setup.proof, &setup.pis)
}

/// A tampered trace commitment causes the Fiat-Shamir transcript to diverge from the
/// values used during proving, so the OOD evaluation check fails as a `WitnessConflict`.
#[test]
#[should_panic(expected = "WitnessConflict")]
fn test_tampered_trace_commitment() {
    let mut setup = build_fibonacci_test_setup();

    // The cap at height 0 contains a single digest; corrupt its first word.
    let mut roots = setup.proof.commitments.trace.into_roots();
    roots[0][0] += F::ONE;
    setup.proof.commitments.trace = roots.into();

    run_recursive_verifier(&setup, &setup.proof, &setup.pis).unwrap();
}

/// Flipping a coefficient in the FRI final polynomial breaks the low-degree test,
/// causing a WitnessConflict when the verifier circuit checks the folding equations.
#[test]
#[should_panic(expected = "WitnessConflict")]
fn test_tampered_fri_final_poly() {
    let mut setup = build_fibonacci_test_setup();

    setup.proof.opening_proof.final_poly[0] += Challenge::ONE;

    run_recursive_verifier(&setup, &setup.proof, &setup.pis).unwrap();
}

/// Feeding wrong public inputs to the verifier circuit means the constraint
/// enforcing the Fibonacci output value is not satisfied, yielding a `WitnessConflict`.
#[test]
#[should_panic(expected = "WitnessConflict")]
fn test_wrong_public_inputs() {
    let setup = build_fibonacci_test_setup();

    let mut wrong_pis = setup.pis.clone();
    // Corrupt the claimed output value.
    wrong_pis[2] += F::ONE;

    run_recursive_verifier(&setup, &setup.proof, &wrong_pis).unwrap();
}

/// Modifying an OOD trace evaluation changes the quotient-consistency check
/// inside the verifier circuit, which results in a `WitnessConflict` at run time.
#[test]
#[should_panic(expected = "WitnessConflict")]
fn test_tampered_ood_evaluation() {
    let mut setup = build_fibonacci_test_setup();

    setup.proof.opened_values.trace_local[0] += Challenge::ONE;

    run_recursive_verifier(&setup, &setup.proof, &setup.pis).unwrap();
}

/// A proof with fewer query rounds than the circuit expects causes
/// `set_fri_mmcs_private_data` to report a shape mismatch, returned as
/// VerificationError::InvalidProofShape.
#[test]
fn test_truncated_fri_proof() {
    let setup = build_fibonacci_test_setup();

    assert!(
        !setup.proof.opening_proof.query_proofs.is_empty(),
        "need at least one query round to truncate"
    );

    // Build the circuit against the valid proof so op_ids match the full shape.
    let mut circuit_builder = CircuitBuilder::new();
    circuit_builder.enable_poseidon2_perm::<BabyBearD4Width16, _>(
        generate_poseidon2_trace::<Challenge, BabyBearD4Width16>,
        setup.perm.clone(),
    );
    circuit_builder.enable_recompose::<F>(generate_recompose_trace::<F, Challenge>);
    // Allocate all targets
    let verifier_inputs = StarkVerifierInputsBuilder::<
        MyConfig,
        MerkleCapTargets<F, DIGEST_ELEMS>,
        InnerFri,
    >::allocate(&mut circuit_builder, &setup.proof, None, setup.pis.len());
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
        &setup.config,
        &setup.air,
        &mut circuit_builder,
        &verifier_inputs.proof_targets,
        &verifier_inputs.air_public_targets,
        &None,
        &setup.fri_verifier_params,
        Poseidon2Config::BabyBearD4Width16,
    )
    .unwrap();
    // Build the circuit.
    let circuit = circuit_builder.build().unwrap();
    let mut runner = circuit.runner();
    // Pack values using the same builder
    let (public_inputs, private_inputs) =
        verifier_inputs.pack_values(&setup.pis, &setup.proof, &None);

    runner.set_public_inputs(&public_inputs).unwrap();
    runner.set_private_inputs(&private_inputs).unwrap();

    // Now supply a truncated FRI proof — this gives fewer siblings than op_ids expects.
    let mut truncated_opening_proof = setup.proof.opening_proof.clone();
    truncated_opening_proof.query_proofs.pop();

    let result = set_fri_mmcs_private_data::<
        F,
        Challenge,
        ChallengeMmcs,
        MyMmcs,
        MyHash,
        MyCompress,
        DIGEST_ELEMS,
    >(&mut runner, &mmcs_op_ids, &truncated_opening_proof)
    .map_err(|e| VerificationError::InvalidProofShape(e.to_string()));

    assert!(
        matches!(result, Err(VerificationError::InvalidProofShape(_))),
        "expected InvalidProofShape for a truncated FRI proof, got: {result:?}",
    );
}
