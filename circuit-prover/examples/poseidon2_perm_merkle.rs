use std::error::Error;

use p3_batch_stark::ProverData;
use p3_circuit::ops::{
    NpoPrivateData, NpoTypeId, Poseidon2Config, Poseidon2PermPrivateData, generate_poseidon2_trace,
    generate_recompose_trace,
};
use p3_circuit::{CircuitBuilder, ExprId};
use p3_circuit_prover::batch_stark_prover::{poseidon2_air_builders, recompose_air_builders};
use p3_circuit_prover::common::{NpoPreprocessor, get_airs_and_degrees_with_prep};
use p3_circuit_prover::config::KoalaBearConfig;
use p3_circuit_prover::{
    BatchStarkProver, CircuitProverData, ConstraintProfile, Poseidon2Preprocessor,
    RecomposePreprocessor, TablePacking, config,
};
use p3_field::extension::BinomialExtensionField;
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_koala_bear::{KoalaBear, default_koalabear_poseidon2_16};
use p3_poseidon2_circuit_air::KoalaBearD4Width16;
use p3_symmetric::Permutation;
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

type Base = KoalaBear;
type Ext4 = BinomialExtensionField<Base, 4>;

const LIMB_SIZE: usize = 4;
const WIDTH: usize = 16;

fn main() -> Result<(), Box<dyn Error>> {
    init_logger();

    // Three-row Merkle path example (2 levels):
    // Row 0: permutation input is leaf || sibling0. merkle_path = true, new_start = true, mmcs_bit = 0
    // Row 1: merkle_path = true, new_start = false, mmcs_bit = 1 (previous hash becomes right child),
    //        input limbs 2-3 get prev row's output limbs 0-1; input limbs 0-1 take sibling1 as private inputs.
    // Row 2: merkle_path = true, new_start = false, mmcs_bit = 0 (previous hash becomes left child),
    //        input limbs 0-1 get prev row's output limbs 0-1; input limbs 2-3 take sibling2 as private inputs.
    //
    // Tree shape (limb ranges = base-field coeff slices of Ext4):
    //          root (row2 out)
    //         /                 \
    //   row2 left (row1 out)   sibling2 [25..32]
    //      /          \
    // sibling1 [17..24]  row0 out
    //                     /     \
    //               leaf [1..8]  sibling0 [9..16]
    //
    // We expose final digest limbs 0-1 as public inputs and the mmcs_index_sum (should be binary 010 = 2).

    let perm = default_koalabear_poseidon2_16();

    // Build leaf and siblings as extension limbs.
    let leaf_limb0 = Ext4::from_basis_coefficients_slice(&[
        Base::from_u64(1),
        Base::from_u64(2),
        Base::from_u64(3),
        Base::from_u64(4),
    ])
    .expect("extension from coeffs");
    let leaf_limb1 = Ext4::from_basis_coefficients_slice(&[
        Base::from_u64(5),
        Base::from_u64(6),
        Base::from_u64(7),
        Base::from_u64(8),
    ])
    .expect("extension from coeffs");
    let sibling0_limb2 = Ext4::from_basis_coefficients_slice(&[
        Base::from_u64(9),
        Base::from_u64(10),
        Base::from_u64(11),
        Base::from_u64(12),
    ])
    .expect("extension from coeffs");
    let sibling0_limb3 = Ext4::from_basis_coefficients_slice(&[
        Base::from_u64(13),
        Base::from_u64(14),
        Base::from_u64(15),
        Base::from_u64(16),
    ])
    .expect("extension from coeffs");

    let sibling1_limb2 = Ext4::from_basis_coefficients_slice(&[
        Base::from_u64(17),
        Base::from_u64(18),
        Base::from_u64(19),
        Base::from_u64(20),
    ])
    .expect("extension from coeffs");
    let sibling1_limb3 = Ext4::from_basis_coefficients_slice(&[
        Base::from_u64(21),
        Base::from_u64(22),
        Base::from_u64(23),
        Base::from_u64(24),
    ])
    .expect("extension from coeffs");
    let sibling2_limb2 = Ext4::from_basis_coefficients_slice(&[
        Base::from_u64(25),
        Base::from_u64(26),
        Base::from_u64(27),
        Base::from_u64(28),
    ])
    .expect("extension from coeffs");
    let sibling2_limb3 = Ext4::from_basis_coefficients_slice(&[
        Base::from_u64(29),
        Base::from_u64(30),
        Base::from_u64(31),
        Base::from_u64(32),
    ])
    .expect("extension from coeffs");

    // Native row 0 permutation: hash(leaf limbs, sibling0 limbs)
    let row0_state = [leaf_limb0, leaf_limb1, sibling0_limb2, sibling0_limb3];
    let row0_state_base = flatten_ext_limbs(&row0_state);
    let row0_out_base = perm.permute(row0_state_base);

    // Row 1 chaining: mmcs_bit = 1, so previous hash becomes right child.
    // Previous digest (out[0..1]) chains into limbs 2-3; sibling1 provides limbs 0-1.
    let mut row1_state_base = [Base::ZERO; WIDTH];
    // limbs 0-1 from sibling1
    let sibling1_flat: [Base; 2 * LIMB_SIZE] = flatten_ext_limbs(&[sibling1_limb2, sibling1_limb3]);
    row1_state_base[0..2 * LIMB_SIZE].copy_from_slice(&sibling1_flat);
    // limbs 2-3 from row0 output limbs 0-1
    row1_state_base[2 * LIMB_SIZE..4 * LIMB_SIZE].copy_from_slice(&row0_out_base[0..2 * LIMB_SIZE]);

    let row1_out_base = perm.permute(row1_state_base);

    // Row 2 chaining: mmcs_bit = 0, so previous hash becomes left child (limbs 0-1 get prev_out[0..2])
    // limbs 2-3 from sibling2
    let mut row2_state_base = [Base::ZERO; WIDTH];
    row2_state_base[0..2 * LIMB_SIZE].copy_from_slice(&row1_out_base[0..2 * LIMB_SIZE]);
    let sibling2_flat: [Base; 2 * LIMB_SIZE] = flatten_ext_limbs(&[sibling2_limb2, sibling2_limb3]);
    row2_state_base[2 * LIMB_SIZE..4 * LIMB_SIZE].copy_from_slice(&sibling2_flat);

    let row2_out_base = perm.permute(row2_state_base);
    let row2_out_limbs = collect_ext_limbs(&row2_out_base);

    // mmcs_index_sum should be 2 (bits: row1=1, row2=0)
    let mmcs_index_sum_row2 = Base::from_u64(2);

    // Build circuit
    let mut builder = CircuitBuilder::<Ext4>::new();
    builder.enable_poseidon2_perm::<KoalaBearD4Width16, _>(
        generate_poseidon2_trace::<Ext4, KoalaBearD4Width16>,
        perm,
    );
    builder.enable_recompose::<Base>(generate_recompose_trace::<Base, Ext4>);

    // Row 0: expose all inputs
    let mmcs_bit_row0 = builder.alloc_const(Ext4::from_prime_subfield(Base::ZERO), "mmcs_bit_row0");
    let inputs_row0: [ExprId; 4] = [
        builder.alloc_const(row0_state[0], "leaf0"),
        builder.alloc_const(row0_state[1], "leaf1"),
        builder.alloc_const(row0_state[2], "sibling0_2"),
        builder.alloc_const(row0_state[3], "sibling0_3"),
    ];

    let (_row0_op_id, _row0_outputs) =
        builder.add_poseidon2_perm(&p3_circuit::ops::Poseidon2PermCall {
            config: Poseidon2Config::KoalaBearD4Width16,
            new_start: true,
            merkle_path: true,
            mmcs_bit: Some(mmcs_bit_row0),
            inputs: inputs_row0.iter().map(|&x| Some(x)).collect(),
            out_ctl: vec![false, false],
            return_all_outputs: false,
            mmcs_index_sum: None,
        })?;

    let sibling1_inputs: Vec<Option<ExprId>> = vec![None; 4];
    let out0 = builder.public_input();
    let out1 = builder.public_input();
    let mmcs_idx_sum_expr = builder.public_input();

    let mmcs_bit_row1 = builder.alloc_const(Ext4::from_prime_subfield(Base::ONE), "mmcs_bit_row1");
    let (row1_op_id, _row1_outputs) =
        builder.add_poseidon2_perm(&p3_circuit::ops::Poseidon2PermCall {
            config: Poseidon2Config::KoalaBearD4Width16,
            new_start: false,
            merkle_path: true,
            mmcs_bit: Some(mmcs_bit_row1),
            inputs: sibling1_inputs,
            out_ctl: vec![false, false],
            return_all_outputs: false,
            mmcs_index_sum: None,
        })?;

    let mmcs_bit_row2 = builder.alloc_const(Ext4::from_prime_subfield(Base::ZERO), "mmcs_bit_row2");
    let sibling2_inputs: Vec<Option<ExprId>> = vec![None; 4];
    let (row2_op_id, row2_outputs) =
        builder.add_poseidon2_perm(&p3_circuit::ops::Poseidon2PermCall {
            config: Poseidon2Config::KoalaBearD4Width16,
            new_start: false,
            merkle_path: true,
            mmcs_bit: Some(mmcs_bit_row2),
            inputs: sibling2_inputs,
            out_ctl: vec![true, true],
            return_all_outputs: false,
            mmcs_index_sum: Some(mmcs_idx_sum_expr),
        })?;
    let row2_out0 = row2_outputs[0].ok_or("missing row2 out0")?;
    let row2_out1 = row2_outputs[1].ok_or("missing row2 out1")?;
    builder.connect(row2_out0, out0);
    builder.connect(row2_out1, out1);

    let circuit = builder.build()?;
    let table_packing = TablePacking::new(4, 4);
    let poseidon2_config = Poseidon2Config::KoalaBearD4Width16;
    let stark_config = config::koala_bear().build();
    let npo_prep: Vec<Box<dyn NpoPreprocessor<Base>>> = vec![
        Box::new(Poseidon2Preprocessor),
        Box::new(RecomposePreprocessor::default()),
    ];
    let mut air_builders = poseidon2_air_builders::<_, 4>();
    air_builders.extend(recompose_air_builders(1, false));
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<KoalaBearConfig, _, 4>(
            &circuit,
            &table_packing,
            &npo_prep,
            &air_builders,
            ConstraintProfile::Standard,
        )?;
    let (mut airs, degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();

    let mut runner = circuit.runner();
    runner.set_public_inputs(&[
        row2_out_limbs[0],
        row2_out_limbs[1],
        Ext4::from_prime_subfield(mmcs_index_sum_row2),
    ])?;

    // Set private inputs for Row 1
    // Row 1: mmcs_bit = 1 (Right Child). Previous digest chains from previous row.
    // For Merkle mode, provide the sibling (2 limbs). Internal logic handles placement.
    runner.set_private_data(
        row1_op_id,
        NpoPrivateData::new(Poseidon2PermPrivateData {
            sibling: vec![sibling1_limb2, sibling1_limb3],
        }),
    )?;

    // Set private inputs for Row 2
    // Row 2: mmcs_bit = 0 (Left Child). Previous digest chains from previous row.
    // For Merkle mode, provide the sibling (2 limbs). Internal logic handles placement.
    runner.set_private_data(
        row2_op_id,
        NpoPrivateData::new(Poseidon2PermPrivateData {
            sibling: vec![sibling2_limb2, sibling2_limb3],
        }),
    )?;

    let traces = runner.run()?;

    // Check Poseidon2 trace rows and mmcs_index_sum exposure
    let poseidon2_trace = traces
        .non_primitive_trace::<p3_circuit::ops::Poseidon2Trace<Base>>(&NpoTypeId::poseidon2_perm(
            Poseidon2Config::KoalaBearD4Width16,
        ))
        .expect("poseidon2 trace missing");
    assert_eq!(poseidon2_trace.total_rows(), 3, "expected three perm rows");

    let prover_data = ProverData::from_airs_and_degrees(&stark_config, &mut airs, &degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);

    // Lookups order being CONST, PUBLIC, ALU, DYNAMIC.
    assert!(
        !circuit_prover_data.common_data().lookups[3].is_empty(),
        "Poseidon2 table should have lookups"
    );

    let mut prover = BatchStarkProver::new(stark_config).with_table_packing(table_packing);
    prover.register_poseidon2_table::<4>(poseidon2_config);
    prover.register_recompose_table::<4>(false);

    let proof = prover.prove_all_tables(&traces, &circuit_prover_data)?;
    prover.verify_all_tables(&proof)?;

    Ok(())
}

fn flatten_ext_limbs<const N: usize, const M: usize>(limbs: &[Ext4; N]) -> [Base; M] {
    let mut out = [Base::ZERO; M];
    for (i, limb) in limbs.iter().enumerate() {
        let coeffs = limb.as_basis_coefficients_slice();
        let start = i * LIMB_SIZE;
        let end = (start + LIMB_SIZE).min(M);
        out[start..end].copy_from_slice(&coeffs[0..(end - start)]);
    }
    out
}

fn collect_ext_limbs(state: &[Base; WIDTH]) -> [Ext4; 4] {
    let mut limbs = [Ext4::ZERO; 4];
    for i in 0..4 {
        let chunk = &state[i * LIMB_SIZE..(i + 1) * LIMB_SIZE];
        limbs[i] = Ext4::from_basis_coefficients_slice(chunk).unwrap();
    }
    limbs
}
