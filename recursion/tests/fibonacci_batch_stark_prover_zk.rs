mod common;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_batch_stark::{ProverData, StarkInstance, prove_batch, verify_batch};
use p3_circuit::CircuitBuilder;
use p3_circuit::ops::{generate_poseidon2_trace, generate_recompose_trace};
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
use p3_recursion::{
    BatchStarkVerifierInputsBuilder, Poseidon2Config, VerificationError, verify_batch_circuit,
};
use p3_test_utils::koala_bear_params::*;
use rand::SeedableRng;
use rand::rngs::SmallRng;

// Non-ZK config used for the outer recursive proof of the verification circuit.
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

fn generate_add_trace<Val: Field>(rows: usize) -> RowMajorMatrix<Val> {
    let width = 3;
    let mut values = Val::zero_vec(rows * width);
    for row in 0..rows {
        let idx = row * width;
        let a = Val::from_usize(row);
        let b = Val::from_usize(row + 1);
        values[idx] = a;
        values[idx + 1] = b;
        values[idx + 2] = a + b;
    }
    RowMajorMatrix::new(values, width)
}

/// Full end-to-end ZK recursive verification test.
///
/// This test proves an AddAir statement with HidingFriPcs (ZK), then builds and
/// runs a recursive verification circuit for that ZK proof, and finally proves
/// the verification circuit itself with another BatchStarkProver.
#[test]
fn test_batch_verifier_zk_hiding_fri() -> Result<(), VerificationError> {
    let air = AddAir;
    let trace = generate_add_trace::<F>(1 << 6);
    let pvs = vec![vec![]];

    // --- Step 1: Prove the AddAir with HidingFriPcs ---
    let perm = default_koalabear_poseidon2_16();
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = MyMmcs::new(hash, compress, 0);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let fri_params = FriParameters::new_testing(challenge_mmcs, 0);
    let pcs_proving = MyPcsZk::new(dft, val_mmcs, fri_params, 2, SmallRng::seed_from_u64(1));
    let challenger_proving = Challenger::new(perm);
    let config_proving = MyConfigZk::new(pcs_proving, challenger_proving);

    let instance = StarkInstance {
        air: &air,
        trace: &trace,
        public_values: pvs[0].clone(),
        lookups: Vec::new(),
    };
    let instances = vec![instance];
    let prover_data = ProverData::from_instances(&config_proving, &instances);
    let common = &prover_data.common;
    let batch_stark_proof = prove_batch(&config_proving, &instances, &prover_data);

    verify_batch(&config_proving, &[air], &batch_stark_proof, &pvs, common).unwrap();

    // --- Step 2: Build the recursive verification circuit ---
    let perm2 = default_koalabear_poseidon2_16();
    let hash2 = MyHash::new(perm2.clone());
    let compress2 = MyCompress::new(perm2.clone());
    let val_mmcs2 = MyMmcs::new(hash2, compress2, 0);
    let challenge_mmcs2 = ChallengeMmcs::new(val_mmcs2.clone());
    let dft2 = Dft::default();
    let fri_params2 = FriParameters::new_testing(challenge_mmcs2, 0);
    let fri_verifier_params = FriVerifierParams::from(&fri_params2);
    let pcs_verif = MyPcsZk::new(dft2, val_mmcs2, fri_params2, 2, SmallRng::seed_from_u64(2));
    let challenger_verif = Challenger::new(perm2.clone());
    let config = MyConfigZk::new(pcs_verif, challenger_verif);

    let mut circuit_builder = CircuitBuilder::new();
    circuit_builder.enable_poseidon2_perm::<KoalaBearD4Width16, _>(
        generate_poseidon2_trace::<Challenge, KoalaBearD4Width16>,
        perm2,
    );
    circuit_builder.enable_recompose::<F>(generate_recompose_trace::<F, Challenge>);

    let lookup_gadget = LogUpGadget::new();
    let air_public_counts = vec![0usize; batch_stark_proof.opened_values.instances.len()];
    let verifier_inputs = BatchStarkVerifierInputsBuilder::<
        MyConfigZk,
        MerkleCapTargets<F, DIGEST_ELEMS>,
        InnerFriZk,
    >::allocate(
        &mut circuit_builder,
        &batch_stark_proof,
        common,
        &air_public_counts,
    );
    let mmcs_op_ids = verify_batch_circuit::<_, _, _, _, _, _, _, WIDTH, RATE>(
        &config,
        &[air],
        &mut circuit_builder,
        &verifier_inputs.proof_targets,
        &verifier_inputs.air_public_targets,
        &fri_verifier_params,
        &verifier_inputs.common_data,
        &lookup_gadget,
        Poseidon2Config::KoalaBearD4Width16,
    )?;

    let verification_circuit = circuit_builder.build().unwrap();
    let (public_inputs, private_inputs) =
        verifier_inputs.pack_values(&pvs, &batch_stark_proof, common);
    assert_eq!(public_inputs.len(), verification_circuit.public_flat_len);

    // --- Step 3: Run the verification circuit ---
    let mut verification_runner = verification_circuit.runner();
    verification_runner
        .set_public_inputs(&public_inputs)
        .unwrap();
    verification_runner
        .set_private_inputs(&private_inputs)
        .unwrap();

    // HidingFriPcs proof is (random_opened_values, inner_fri_proof); pass the inner part.
    // With test FRI params (num_queries = 0), there may be no Merkle openings.
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
            &mut verification_runner,
            &mmcs_op_ids,
            &batch_stark_proof.opening_proof.1,
        )
        .expect("Failed to set MMCS private data for ZK proof");
    }

    let verification_traces = verification_runner.run().unwrap();

    // --- Step 4: Prove the verification circuit itself ---
    // Use non-ZK PCS for the outer recursive proof; ZK is only required for the inner proof.
    let perm3 = default_koalabear_poseidon2_16();
    let hash3 = MyHash::new(perm3.clone());
    let compress3 = MyCompress::new(perm3.clone());
    let val_mmcs3 = MyMmcs::new(hash3, compress3, 0);
    let challenge_mmcs3 = ChallengeMmcs::new(val_mmcs3.clone());
    let dft3 = Dft::default();
    let fri_params3 = FriParameters::new_testing(challenge_mmcs3, 0);
    let pcs3 = TwoAdicFriPcs::new(dft3, val_mmcs3, fri_params3);
    let challenger3 = Challenger::new(perm3);
    let config3 = MyConfig::new(pcs3, challenger3);

    let verification_table_packing = TablePacking::new(1, 8);
    let poseidon2_config = Poseidon2Config::KoalaBearD4Width16;
    let npo_prep: Vec<Box<dyn NpoPreprocessor<F>>> = vec![
        Box::new(Poseidon2Preprocessor),
        Box::new(RecomposePreprocessor::default()),
    ];
    let mut air_builders = poseidon2_air_builders::<_, 4>();
    air_builders.extend(recompose_air_builders(1, false));
    let (
        verification_airs_degrees,
        verification_primitive_columns,
        verification_non_primitive_columns,
    ) = get_airs_and_degrees_with_prep::<MyConfig, _, 4>(
        &verification_circuit,
        &verification_table_packing,
        &npo_prep,
        &air_builders,
        ConstraintProfile::Standard,
    )
    .unwrap();
    let (mut verification_airs, verification_degrees): (Vec<_>, Vec<usize>) =
        verification_airs_degrees.into_iter().unzip();

    let verification_prover_data =
        ProverData::from_airs_and_degrees(&config3, &mut verification_airs, &verification_degrees);
    let verification_circuit_prover_data = CircuitProverData::new(
        verification_prover_data,
        verification_primitive_columns,
        verification_non_primitive_columns,
    );

    let mut verification_prover =
        BatchStarkProver::new(config3).with_table_packing(verification_table_packing);
    verification_prover.register_poseidon2_table::<4>(poseidon2_config);
    verification_prover.register_recompose_table::<4>(false);

    let verification_proof = verification_prover
        .prove_all_tables(&verification_traces, &verification_circuit_prover_data)
        .expect("Failed to prove ZK verification circuit");

    verification_prover
        .verify_all_tables(&verification_proof)
        .expect("Failed to verify proof of ZK verification circuit");

    Ok(())
}
