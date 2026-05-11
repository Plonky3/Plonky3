//! Koala quintic Fibonacci: prove inner circuit, native-verify, run the recursive verifier circuit
//! with in-circuit FRI MMCS (`with_mmcs` + `set_fri_mmcs_private_data`), then outer-prove the
//! verifier circuit traces with D=5 and native-verify the resulting proof.

mod common;

use p3_batch_stark::ProverData;
use p3_circuit::CircuitBuilder;
use p3_circuit::ops::{
    KoalaBearD1Width16, Poseidon2Config, generate_poseidon2_trace, generate_recompose_trace,
};
use p3_circuit_prover::batch_stark_prover::{
    poseidon2_air_builders_d5, poseidon2_table_provers_d5, recompose_air_builders,
    recompose_table_provers,
};
use p3_circuit_prover::common::{NpoPreprocessor, get_airs_and_degrees_with_prep};
use p3_circuit_prover::{
    BatchStarkProver, CircuitProverData, ConstraintProfile, Poseidon2Preprocessor,
    RecomposePreprocessor, TablePacking,
};
use p3_fri::FriParameters;
use p3_lookup::logup::LogUpGadget;
use p3_recursion::pcs::fri::{FriVerifierParams, InputProofTargets, MerkleCapTargets, RecValMmcs};
use p3_recursion::pcs::set_fri_mmcs_private_data;
use p3_recursion::verifier::verify_p3_batch_proof_circuit;
use p3_test_utils::koala_bear_quintic_params::*;
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

fn fibonacci_challenge(n: usize) -> Challenge {
    if n == 0 {
        return Challenge::ZERO;
    }
    if n == 1 {
        return Challenge::ONE;
    }
    let mut a = F::ZERO;
    let mut b = F::ONE;
    for _ in 2..=n {
        let next = a + b;
        a = b;
        b = next;
    }

    b.into()
}

fn make_test_config() -> MyConfig {
    let perm = default_koalabear_poseidon2_16();
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = MyMmcs::new(hash, compress, 0);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let fri_params = FriParameters::new_testing(challenge_mmcs, 0);
    let pcs = MyPcs::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::new(perm);
    MyConfig::new(pcs, challenger)
}

#[test]
fn test_fibonacci_batch_verifier_quintic_koala() {
    init_logger();

    let n: usize = 48;

    let mut builder = CircuitBuilder::<Challenge>::new();
    let expected_result = builder.public_input();
    let mut a = builder.define_const(Challenge::ZERO);
    let mut b = builder.define_const(Challenge::ONE);
    for _ in 2..=n {
        let next = builder.add(a, b);
        a = b;
        b = next;
    }
    builder.connect(b, expected_result);

    let table_packing = TablePacking::new(2, 4);

    let config_proving = make_test_config();

    let circuit = builder.build().unwrap();
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<MyConfig, _, 5>(
            &circuit,
            &table_packing,
            &[],
            &[],
            ConstraintProfile::Standard,
        )
        .unwrap();
    let (mut airs, degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();
    let mut runner = circuit.runner();
    let expected_fib = fibonacci_challenge(n);
    runner.set_public_inputs(&[expected_fib]).unwrap();
    let traces = runner.run().unwrap();

    let prover_data = ProverData::from_airs_and_degrees(&config_proving, &mut airs, &degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);
    let prover = BatchStarkProver::new(config_proving).with_table_packing(table_packing);
    let lookup_gadget = LogUpGadget::new();
    let batch_stark_proof = prover
        .prove_all_tables(&traces, &circuit_prover_data)
        .unwrap();
    prover.verify_all_tables(&batch_stark_proof).unwrap();

    // `prove_all_tables` may reduce lanes for dummy ALU/public tables; `stark_common` always
    // matches the committed AIR layout, so recursive verification consumes it directly.
    let common = &batch_stark_proof.stark_common;

    let dft2 = Dft::default();
    let perm2 = default_koalabear_poseidon2_16();
    let hash2 = MyHash::new(perm2.clone());
    let compress2 = MyCompress::new(perm2.clone());
    let val_mmcs2 = MyMmcs::new(hash2, compress2, 0);
    let challenge_mmcs2 = ChallengeMmcs::new(val_mmcs2.clone());
    let fri_params2 = FriParameters::new_testing(challenge_mmcs2, 0);
    let fri_verifier_params = FriVerifierParams::with_mmcs(
        fri_params2.log_blowup,
        fri_params2.log_final_poly_len,
        fri_params2.commit_proof_of_work_bits,
        fri_params2.query_proof_of_work_bits,
        Poseidon2Config::KoalaBearD1Width16,
    );
    let pcs_verif = MyPcs::new(dft2, val_mmcs2, fri_params2);
    let challenger_verif = Challenger::new(perm2);
    let config = MyConfig::new(pcs_verif, challenger_verif);

    let batch_proof = &batch_stark_proof.proof;
    const TRACE_D: usize = 5;

    let num_tables = common
        .preprocessed
        .as_ref()
        .map(|g| g.instances.len())
        .unwrap_or(0);
    let pis: Vec<Vec<F>> = vec![vec![]; num_tables];

    let mut circuit_builder = CircuitBuilder::<Challenge>::new();
    let lift = LiftKoalaPermForQuintic::new(default_koalabear_poseidon2_16());
    circuit_builder.enable_poseidon2_perm_base::<KoalaBearD1Width16, _>(
        generate_poseidon2_trace::<Challenge, KoalaBearD1Width16>,
        lift,
    );
    circuit_builder.enable_recompose::<F>(generate_recompose_trace::<F, Challenge>);
    circuit_builder.set_recompose_coeff_ctl_for_decompose_links(true);

    let (verifier_inputs, mmcs_op_ids) = verify_p3_batch_proof_circuit::<
        MyConfig,
        MerkleCapTargets<F, DIGEST_ELEMS>,
        InputProofTargets<F, Challenge, RecValMmcs<F, DIGEST_ELEMS, MyHash, MyCompress>>,
        InnerFri,
        LogUpGadget,
        _,
        WIDTH,
        RATE,
        TRACE_D,
    >(
        &config,
        &mut circuit_builder,
        &batch_stark_proof,
        &fri_verifier_params,
        common,
        &lookup_gadget,
        Poseidon2Config::KoalaBearD1Width16,
        &{
            let mut tp = poseidon2_table_provers_d5(Poseidon2Config::KoalaBearD1Width16);
            tp.extend(recompose_table_provers::<MyConfig, 5>(1, true));
            tp
        },
    )
    .unwrap();

    let verification_circuit = circuit_builder.build().unwrap();
    let expected_public_input_len = verification_circuit.public_flat_len;
    let (public_inputs, private_inputs) = verifier_inputs.pack_values(&pis, batch_proof, common);
    assert_eq!(public_inputs.len(), expected_public_input_len);
    assert!(!public_inputs.is_empty());

    let verification_table_packing = TablePacking::new(1, 8);
    let npo_prep: Vec<Box<dyn NpoPreprocessor<F>>> = vec![
        Box::new(Poseidon2Preprocessor),
        Box::new(RecomposePreprocessor::new(true)),
    ];
    let mut air_builders = poseidon2_air_builders_d5::<MyConfig>();
    air_builders.extend(recompose_air_builders::<MyConfig, 5>(1, true));
    let (verification_airs_degrees, verification_primitive, verification_npo) =
        get_airs_and_degrees_with_prep::<MyConfig, _, 5>(
            &verification_circuit,
            &verification_table_packing,
            &npo_prep,
            &air_builders,
            ConstraintProfile::Standard,
        )
        .unwrap();
    let (mut verification_airs, verification_degrees): (Vec<_>, Vec<usize>) =
        verification_airs_degrees.into_iter().unzip();

    let mut runner = verification_circuit.runner();
    runner.set_public_inputs(&public_inputs).unwrap();
    runner.set_private_inputs(&private_inputs).unwrap();
    if !mmcs_op_ids.is_empty() {
        set_fri_mmcs_private_data::<
            F,
            Challenge,
            ChallengeMmcs,
            MyMmcs,
            MyHash,
            MyCompress,
            DIGEST_ELEMS,
        >(
            &mut runner,
            &mmcs_op_ids,
            &batch_stark_proof.proof.opening_proof,
        )
        .unwrap();
    }
    let verification_traces = runner.run().expect("outer circuit trace generation failed");
    assert!(
        verification_traces.witness_trace.num_rows() > 0,
        "verifier circuit should produce witness trace"
    );

    let config3 = make_test_config();

    let verification_prover_data =
        ProverData::from_airs_and_degrees(&config3, &mut verification_airs, &verification_degrees);
    let verification_circuit_prover_data = CircuitProverData::new(
        verification_prover_data,
        verification_primitive,
        verification_npo,
    );

    let mut verification_prover =
        BatchStarkProver::new(config3).with_table_packing(verification_table_packing);
    verification_prover.register_poseidon2_table::<5>(Poseidon2Config::KoalaBearD1Width16);
    verification_prover.register_recompose_table::<5>(true);

    let verification_proof = verification_prover
        .prove_all_tables(&verification_traces, &verification_circuit_prover_data)
        .expect("Failed to prove verification circuit");

    verification_prover
        .verify_all_tables(&verification_proof)
        .expect("Failed to verify proof of verification circuit");
}
