mod common;

use p3_circuit::CircuitBuilder;
use p3_circuit::ops::{
    BabyBearD1Width16, Poseidon2CircuitRow, generate_poseidon2_trace, generate_recompose_trace,
};
use p3_fri::FriParameters;
use p3_poseidon2::ExternalLayerConstants;
use p3_poseidon2_air::RoundConstants;
use p3_poseidon2_circuit_air::{
    Poseidon2CircuitAirBabyBearD4Width16, extract_preprocessed_from_operations,
};
use p3_recursion::pcs::fri::{
    FriProofTargets, FriVerifierParams, InputProofTargets, MerkleCapTargets, RecExtensionValMmcs,
    RecValMmcs, Witness,
};
use p3_recursion::public_inputs::StarkVerifierInputsBuilder;
use p3_recursion::{Poseidon2Config, VerificationError, verify_p3_uni_proof_circuit};
use p3_uni_stark::{
    StarkGenericConfig, prove_with_preprocessed, setup_preprocessed, verify_with_preprocessed,
};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use tracing_forest::ForestLayer;
use tracing_forest::util::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

/// Initializes a global logger with default parameters.
fn init_logger() {
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();
}

use p3_test_utils::baby_bear_params::*;

// Use base field challenges for this test to keep proof size manageable.
// The common module uses extension field challenges (D=4), which would
// create 4x more observations and circuit operations.
type Challenge = F;
type ChallengeMmcs = ExtensionMmcs<F, Challenge, MyMmcs>;
type MyPcs = TwoAdicFriPcs<F, Dft, MyMmcs, ChallengeMmcs>;
type MyConfig = StarkConfig<MyPcs, Challenge, Challenger>;

#[test]
fn test_poseidon2_perm_verifier() -> Result<(), VerificationError> {
    init_logger();

    let mut rng = SmallRng::seed_from_u64(1);
    let beginning_full_constants = rng.random();
    let partial_constants = rng.random();
    let ending_full_constants = rng.random();
    let constants = RoundConstants::new(
        beginning_full_constants,
        partial_constants,
        ending_full_constants,
    );
    let perm = Poseidon2BabyBear::<16>::new(
        ExternalLayerConstants::new(
            beginning_full_constants.to_vec(),
            ending_full_constants.to_vec(),
        ),
        partial_constants.to_vec(),
    );

    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = MyMmcs::new(hash, compress, 0);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    // Keep a small final poly length; with enough rows we still get FRI fold phases.
    let log_final_poly_len = 0;
    let fri_params = FriParameters::new_testing(challenge_mmcs, log_final_poly_len);
    let fri_verifier_params = FriVerifierParams::from(&fri_params);
    let _log_height_max = fri_params.log_final_poly_len + fri_params.log_blowup;
    let _pow_bits = fri_params.query_proof_of_work_bits;
    let pcs = MyPcs::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::new(perm.clone());
    let config = MyConfig::new(pcs, challenger);

    // Build a trace with enough rows to satisfy FRI height constraints.
    let n_rows: usize = 32;
    let ops: Vec<_> = (0..n_rows)
        .map(|row| {
            let input_values: Vec<F> = (0..16_u32)
                .map(|i| F::from_u32(i + 5 + row as u32))
                .collect();
            Poseidon2CircuitRow {
                new_start: true,
                merkle_path: false,
                mmcs_bit: false,
                mmcs_index_sum: F::ZERO,
                input_values,
                in_ctl: vec![false; 4],
                input_indices: vec![0; 4],
                out_ctl: vec![false; 2],
                output_indices: vec![0; 2],
                mmcs_index_sum_idx: 0,
                mmcs_ctl_enabled: false,
            }
        })
        .collect();

    let preprocessed = extract_preprocessed_from_operations::<4, 2, F, F>(&ops, 4, 4);
    let air = Poseidon2CircuitAirBabyBearD4Width16::new_with_preprocessed(
        constants.clone(),
        preprocessed,
    );

    let (prover_data, verifier_data) = setup_preprocessed(&config, &air, 5).unwrap();

    let trace = air.generate_trace_rows(&ops, &constants, 0);

    let public_inputs: Vec<F> = vec![];
    let proof = prove_with_preprocessed(&config, &air, trace, &public_inputs, Some(&prover_data));
    assert!(
        verify_with_preprocessed(&config, &air, &proof, &public_inputs, Some(&verifier_data))
            .is_ok()
    );

    type InnerFri = FriProofTargets<
        p3_uni_stark::Val<MyConfig>,
        <MyConfig as StarkGenericConfig>::Challenge,
        RecExtensionValMmcs<
            p3_uni_stark::Val<MyConfig>,
            <MyConfig as StarkGenericConfig>::Challenge,
            DIGEST_ELEMS,
            RecValMmcs<p3_uni_stark::Val<MyConfig>, DIGEST_ELEMS, MyHash, MyCompress>,
        >,
        InputProofTargets<
            p3_uni_stark::Val<MyConfig>,
            <MyConfig as StarkGenericConfig>::Challenge,
            RecValMmcs<p3_uni_stark::Val<MyConfig>, DIGEST_ELEMS, MyHash, MyCompress>,
        >,
        Witness<p3_uni_stark::Val<MyConfig>>,
    >;

    let mut circuit_builder = CircuitBuilder::new();
    // Use the same permutation as the prover to ensure Fiat-Shamir challengers match.
    // D=1 (base field challenges) uses the base variant which operates on 16 elements directly.
    circuit_builder.enable_poseidon2_perm_base::<BabyBearD1Width16, _>(
        generate_poseidon2_trace::<Challenge, BabyBearD1Width16>,
        perm,
    );
    circuit_builder.enable_recompose::<F>(generate_recompose_trace::<F, Challenge>);
    let verifier_inputs = StarkVerifierInputsBuilder::<
        MyConfig,
        MerkleCapTargets<F, DIGEST_ELEMS>,
        InnerFri,
    >::allocate(
        &mut circuit_builder,
        &proof,
        Some(&verifier_data.commitment),
        public_inputs.len(),
    );

    verify_p3_uni_proof_circuit::<
        Poseidon2CircuitAirBabyBearD4Width16,
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
        &verifier_inputs.preprocessed_commit,
        &fri_verifier_params,
        Poseidon2Config::BabyBearD1Width16,
    )?;

    let circuit = circuit_builder.build()?;
    let mut runner = circuit.runner();

    let (public_inputs, private_inputs) =
        verifier_inputs.pack_values(&public_inputs, &proof, &Some(verifier_data.commitment));

    runner
        .set_public_inputs(&public_inputs)
        .map_err(VerificationError::Circuit)?;
    runner
        .set_private_inputs(&private_inputs)
        .map_err(VerificationError::Circuit)?;
    let _ = runner.run().map_err(VerificationError::Circuit)?;

    Ok(())
}
