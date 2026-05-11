//! Integration test: 2-to-1 aggregation of two ZK batch-STARK proofs.
//!
//! Both inner proofs are produced with `HidingFriPcs`, which randomises the
//! committed polynomials so the prover cannot learn anything about the verifier's
//! query positions.  Their recursive verification circuits are then composed into
//! a single aggregation circuit and the aggregated proof is verified.

mod common;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_batch_stark::{
    BatchProof, CommonData, ProverData, StarkInstance, prove_batch, verify_batch,
};
use p3_circuit::ops::{Poseidon2Config, generate_poseidon2_trace, generate_recompose_trace};
use p3_circuit::{CircuitBuilder, NonPrimitiveOpId};
use p3_circuit_prover::batch_stark_prover::{poseidon2_air_builders, recompose_air_builders};
use p3_circuit_prover::common::{NpoPreprocessor, get_airs_and_degrees_with_prep};
use p3_circuit_prover::{
    BatchStarkProver, CircuitProverData, ConstraintProfile, Poseidon2Preprocessor,
    RecomposePreprocessor, TablePacking,
};
use p3_field::Field;
use p3_fri::{FriParameters, HidingFriPcs, TwoAdicFriPcs};
use p3_lookup::logup::LogUpGadget;
use p3_lookup::{Lookup, LookupAir};
use p3_matrix::dense::RowMajorMatrix;
use p3_poseidon2_circuit_air::KoalaBearD4Width16;
use p3_recursion::pcs::fri::{
    FriVerifierParams, HidingFriProofTargets, InputProofTargets, MerkleCapTargets,
    RecExtensionValMmcs, RecValMmcs, Witness,
};
use p3_recursion::pcs::set_fri_mmcs_private_data;
use p3_recursion::{BatchStarkVerifierInputsBuilder, VerificationError, verify_batch_circuit};
use p3_test_utils::koala_bear_params::*;
use rand::SeedableRng;
use rand::rngs::SmallRng;

// Non-ZK config used for the outer aggregated proof of both verification circuits.
type MyConfig = StarkConfig<TwoAdicFriPcs<F, Dft, MyMmcs, ChallengeMmcs>, Challenge, Challenger>;

type MyPcsZk = HidingFriPcs<F, Dft, MyMmcs, ChallengeMmcs, SmallRng>;
type MyConfigZk = StarkConfig<MyPcsZk, Challenge, Challenger>;
type InnerFriZk = HidingFriProofTargets<
    F,
    Challenge,
    RecExtensionValMmcs<
        F,
        Challenge,
        DIGEST_ELEMS,
        RecValMmcs<F, DIGEST_ELEMS, MyHash, MyCompress>,
    >,
    InputProofTargets<F, Challenge, RecValMmcs<F, DIGEST_ELEMS, MyHash, MyCompress>>,
    Witness<F>,
>;

// Simple addition AIR: enforces row[0] + row[1] == row[2].
#[derive(Clone, Copy)]
struct AddAir;

impl<Val: Field> BaseAir<Val> for AddAir {
    fn width(&self) -> usize {
        3
    }
}

impl<AB: AirBuilder> Air<AB> for AddAir
where
    AB::F: Field,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.current_slice();
        builder.assert_zero(row[0] + row[1] - row[2]);
    }
}

impl<F: Field> LookupAir<F> for AddAir {
    fn add_lookup_columns(&mut self) -> Vec<usize> {
        vec![0]
    }

    fn get_lookups(&mut self) -> Vec<Lookup<F>> {
        vec![]
    }
}

fn generate_add_trace<Val: Field>(rows: usize, offset: usize) -> RowMajorMatrix<Val> {
    let width = 3;
    let mut values = Val::zero_vec(rows * width);
    for row in 0..rows {
        let idx = row * width;
        let a = Val::from_usize(row + offset);
        let b = Val::from_usize(row + offset + 1);
        values[idx] = a;
        values[idx + 1] = b;
        values[idx + 2] = a + b;
    }
    RowMajorMatrix::new(values, width)
}

fn make_zk_config(seed: u64) -> MyConfigZk {
    let perm = default_koalabear_poseidon2_16();
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm);
    let val_mmcs = MyMmcs::new(hash, compress, 0);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let fri_params = FriParameters::new_testing(challenge_mmcs, 0);
    let pcs = MyPcsZk::new(
        Dft::default(),
        val_mmcs,
        fri_params,
        2,
        SmallRng::seed_from_u64(seed),
    );
    MyConfigZk::new(pcs, Challenger::new(default_koalabear_poseidon2_16()))
}

fn make_non_zk_config() -> MyConfig {
    let perm = default_koalabear_poseidon2_16();
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = MyMmcs::new(hash, compress, 0);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let fri_params = FriParameters::new_testing(challenge_mmcs, 0);
    let pcs = TwoAdicFriPcs::new(Dft::default(), val_mmcs, fri_params);
    MyConfig::new(pcs, Challenger::new(perm))
}

struct ZkProofData {
    proof: BatchProof<MyConfigZk>,
    prover_data: ProverData<MyConfigZk>,
}

fn prove_zk_add_air(config: &MyConfigZk, trace: &RowMajorMatrix<F>) -> ZkProofData {
    let air = AddAir;
    let instance = StarkInstance {
        air: &air,
        trace,
        public_values: vec![],
        lookups: Vec::new(),
    };
    let instances = vec![instance];
    let prover_data = ProverData::from_instances(config, &instances);
    let common = &prover_data.common;
    let proof = prove_batch(config, &instances, &prover_data);
    verify_batch(config, &[air], &proof, &[vec![]], common).expect("inner ZK verify failed");
    ZkProofData { proof, prover_data }
}

type VerifInputBuilder =
    BatchStarkVerifierInputsBuilder<MyConfigZk, MerkleCapTargets<F, DIGEST_ELEMS>, InnerFriZk>;

/// Adds the recursive verification circuit for one ZK batch proof into
/// `circuit_builder` and returns the verifier inputs builder plus the
/// op-ids needed to set MMCS private data later.
fn add_zk_batch_verifier_to_circuit(
    config: &MyConfigZk,
    proof: &BatchProof<MyConfigZk>,
    common: &CommonData<MyConfigZk>,
    circuit_builder: &mut CircuitBuilder<Challenge>,
) -> Result<(VerifInputBuilder, Vec<NonPrimitiveOpId>), VerificationError> {
    let fri_verifier_params = {
        let hash = MyHash::new(default_koalabear_poseidon2_16());
        let compress = MyCompress::new(default_koalabear_poseidon2_16());
        let val_mmcs = MyMmcs::new(hash, compress, 0);
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs);
        let fri_params = FriParameters::new_testing(challenge_mmcs, 0);
        FriVerifierParams::from(&fri_params)
    };

    let air_public_counts = vec![0usize; proof.opened_values.instances.len()];
    let verifier_inputs = BatchStarkVerifierInputsBuilder::<
        MyConfigZk,
        MerkleCapTargets<F, DIGEST_ELEMS>,
        InnerFriZk,
    >::allocate(circuit_builder, proof, common, &air_public_counts);

    let lookup_gadget = LogUpGadget::new();
    let mmcs_op_ids = verify_batch_circuit::<_, _, _, _, _, _, _, WIDTH, RATE>(
        config,
        &[AddAir],
        circuit_builder,
        &verifier_inputs.proof_targets,
        &verifier_inputs.air_public_targets,
        &fri_verifier_params,
        &verifier_inputs.common_data,
        &lookup_gadget,
        Poseidon2Config::KoalaBearD4Width16,
    )?;

    Ok((verifier_inputs, mmcs_op_ids))
}

/// Proves a ZK batch proof of an `AddAir` trace, then proves a second ZK batch
/// proof of another `AddAir` trace, and finally aggregates both recursive
/// verification circuits into a single non-ZK proof.
#[test]
fn test_zk_aggregation() -> Result<(), VerificationError> {
    // --- Step 1: Prove two independent ZK batch proofs ---
    let config_zk_left = make_zk_config(1);
    let trace_left = generate_add_trace::<F>(1 << 6, 0);
    let left_data = prove_zk_add_air(&config_zk_left, &trace_left);
    let common_left = &left_data.prover_data.common;

    let config_zk_right = make_zk_config(2);
    let trace_right = generate_add_trace::<F>(1 << 6, 100);
    let right_data = prove_zk_add_air(&config_zk_right, &trace_right);
    let common_right = &right_data.prover_data.common;

    // --- Step 2: Build the aggregation circuit ---
    // Both ZK verifiers share one CircuitBuilder; running it produces traces
    // for all tables of both verifiers simultaneously.
    let perm = default_koalabear_poseidon2_16();
    let mut circuit_builder = CircuitBuilder::new();
    circuit_builder.enable_poseidon2_perm::<KoalaBearD4Width16, _>(
        generate_poseidon2_trace::<Challenge, KoalaBearD4Width16>,
        perm,
    );
    circuit_builder.enable_recompose::<F>(generate_recompose_trace::<F, Challenge>);

    let (left_verifier_inputs, left_op_ids) = add_zk_batch_verifier_to_circuit(
        &config_zk_left,
        &left_data.proof,
        common_left,
        &mut circuit_builder,
    )?;

    let (right_verifier_inputs, right_op_ids) = add_zk_batch_verifier_to_circuit(
        &config_zk_right,
        &right_data.proof,
        common_right,
        &mut circuit_builder,
    )?;

    // --- Step 3: Run the aggregation circuit ---
    let aggregation_circuit = circuit_builder.build().unwrap();

    let (mut public_inputs, mut private_inputs) =
        left_verifier_inputs.pack_values(&[vec![]], &left_data.proof, common_left);
    let (right_public_inputs, right_private_inputs) =
        right_verifier_inputs.pack_values(&[vec![]], &right_data.proof, common_right);
    public_inputs.extend(right_public_inputs);
    private_inputs.extend(right_private_inputs);

    let mut runner = aggregation_circuit.runner();
    runner
        .set_public_inputs(&public_inputs)
        .map_err(VerificationError::Circuit)?;
    runner
        .set_private_inputs(&private_inputs)
        .map_err(VerificationError::Circuit)?;

    // HidingFriPcs proof is (random_opened_values, inner_fri_proof); pass the inner part.
    if !left_op_ids.is_empty() {
        set_fri_mmcs_private_data::<
            F,
            Challenge,
            ChallengeMmcs,
            MyMmcs,
            MyHash,
            MyCompress,
            DIGEST_ELEMS,
        >(&mut runner, &left_op_ids, &left_data.proof.opening_proof.1)
        .map_err(|e| VerificationError::InvalidProofShape(e.to_string()))?;
    }

    if !right_op_ids.is_empty() {
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
            &right_op_ids,
            &right_data.proof.opening_proof.1,
        )
        .map_err(|e| VerificationError::InvalidProofShape(e.to_string()))?;
    }

    let aggregation_traces = runner.run().map_err(VerificationError::Circuit)?;

    // --- Step 4: Prove the aggregation circuit itself (non-ZK outer proof) ---
    let config_outer = make_non_zk_config();

    let poseidon2_config = Poseidon2Config::KoalaBearD4Width16;
    let table_packing = TablePacking::new(1, 8);
    let npo_prep: Vec<Box<dyn NpoPreprocessor<F>>> = vec![
        Box::new(Poseidon2Preprocessor),
        Box::new(RecomposePreprocessor::default()),
    ];
    let mut air_builders = poseidon2_air_builders::<_, 4>();
    air_builders.extend(recompose_air_builders(1, false));
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<MyConfig, _, 4>(
            &aggregation_circuit,
            &table_packing,
            &npo_prep,
            &air_builders,
            ConstraintProfile::Standard,
        )
        .unwrap();
    let (mut airs, degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();

    let prover_data = ProverData::from_airs_and_degrees(&config_outer, &mut airs, &degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);

    let mut prover = BatchStarkProver::new(config_outer).with_table_packing(table_packing);
    prover.register_poseidon2_table::<4>(poseidon2_config);
    prover.register_recompose_table::<4>(false);

    let aggregated_proof = prover
        .prove_all_tables(&aggregation_traces, &circuit_prover_data)
        .expect("failed to prove aggregation circuit");

    prover
        .verify_all_tables(&aggregated_proof)
        .expect("failed to verify aggregated proof");

    Ok(())
}
