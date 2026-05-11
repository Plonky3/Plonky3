use p3_baby_bear::BabyBear;
use p3_circuit::builder::CircuitBuilder;
use p3_circuit::ops::poseidon2_perm::{GoldilocksD2Width8, Poseidon2PermCallBase};
use p3_circuit::ops::{
    KoalaBearD1Width16, Poseidon2Config, generate_poseidon2_trace, generate_recompose_trace,
};
use p3_field::PrimeCharacteristicRing;
use p3_field::extension::QuinticTrinomialExtensionField;
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_koala_bear::{KoalaBear, default_koalabear_poseidon2_16};
use p3_symmetric::{CryptographicHasher, PaddingFreeSponge, Permutation};
use p3_test_utils::LiftPermToQuintic;

use super::*;
use crate::ConstraintProfile;
use crate::batch_stark_prover::{
    BABY_BEAR_MODULUS, KOALA_BEAR_MODULUS, Poseidon2Preprocessor, poseidon2_air_builders,
    poseidon2_air_builders_d5, poseidon2_table_provers_d5, recompose_air_builders,
};
use crate::common::{NpoPreprocessor, get_airs_and_degrees_with_prep};
use crate::config::{self, BabyBearConfig, GoldilocksConfig, KoalaBearConfig};

#[test]
fn test_babybear_batch_stark_base_field() {
    let mut builder = CircuitBuilder::<BabyBear>::new();

    // x + 5*2 - 3 + (-1) == expected
    let x = builder.public_input();
    let expected = builder.public_input();
    let c5 = builder.define_const(BabyBear::from_u64(5));
    let c2 = builder.define_const(BabyBear::from_u64(2));
    let c3 = builder.define_const(BabyBear::from_u64(3));
    let neg_one = builder.define_const(BabyBear::NEG_ONE);

    let mul_result = builder.mul(c5, c2); // 10
    let add_result = builder.add(x, mul_result); // x + 10
    let sub_result = builder.sub(add_result, c3); // x + 7
    let final_result = builder.add(sub_result, neg_one); // x + 6

    let diff = builder.sub(final_result, expected);
    builder.assert_zero(diff);

    let circuit = builder.build().unwrap();
    let cfg = config::baby_bear().build();
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<BabyBearConfig, _, 1>(
            &circuit,
            &TablePacking::default(),
            &[],
            &[],
            ConstraintProfile::Standard,
        )
        .unwrap();
    let (mut airs, log_degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();
    let prover_data = ProverData::from_airs_and_degrees(&cfg, &mut airs, &log_degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);

    let mut runner = circuit.runner();

    let x_val = BabyBear::from_u64(7);
    let expected_val = BabyBear::from_u64(13); // 7 + 10 - 3 - 1 = 13
    runner.set_public_inputs(&[x_val, expected_val]).unwrap();
    let traces = runner.run().unwrap();

    let prover = BatchStarkProver::new(cfg);

    let proof = prover
        .prove_all_tables(&traces, &circuit_prover_data)
        .unwrap();
    assert_eq!(proof.ext_degree, 1);
    assert!(proof.w_binomial.is_none());

    assert!(prover.verify_all_tables(&proof).is_ok());
}

#[test]
fn test_table_lookups() {
    let mut builder = CircuitBuilder::<BabyBear>::new();
    let cfg = config::baby_bear().build();

    // x + 5*2 - 3 + (-1) == expected
    let x = builder.public_input();
    let expected = builder.public_input();
    let c5 = builder.define_const(BabyBear::from_u64(5));
    let c2 = builder.define_const(BabyBear::from_u64(2));
    let c3 = builder.define_const(BabyBear::from_u64(3));
    let neg_one = builder.define_const(BabyBear::NEG_ONE);

    let mul_result = builder.mul(c5, c2); // 10
    let add_result = builder.add(x, mul_result); // x + 10
    let sub_result = builder.sub(add_result, c3); // x + 7
    let final_result = builder.add(sub_result, neg_one); // x + 6

    let diff = builder.sub(final_result, expected);
    builder.assert_zero(diff);

    let circuit = builder.build().unwrap();
    let default_packing = TablePacking::default();
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<BabyBearConfig, _, 1>(
            &circuit,
            &default_packing,
            &[],
            &[],
            ConstraintProfile::Standard,
        )
        .unwrap();
    let (mut airs, log_degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();

    let mut runner = circuit.runner();

    let x_val = BabyBear::from_u64(7);
    let expected_val = BabyBear::from_u64(13); // 7 + 10 - 3 - 1 = 13
    runner.set_public_inputs(&[x_val, expected_val]).unwrap();
    let traces = runner.run().unwrap();
    let prover_data = ProverData::from_airs_and_degrees(&cfg, &mut airs, &log_degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);

    let prover = BatchStarkProver::new(cfg);

    let proof = prover
        .prove_all_tables(&traces, &circuit_prover_data)
        .unwrap();
    assert_eq!(proof.ext_degree, 1);
    assert!(proof.w_binomial.is_none());

    assert!(prover.verify_all_tables(&proof).is_ok());

    // Check that the generated lookups are correct and consistent across tables.
    for air in airs.iter_mut() {
        let lookups = LookupAir::get_lookups(air);

        match air {
            CircuitTableAir::Const(_) => {
                assert_eq!(lookups.len(), 1, "Const table should have one lookup");
            }
            CircuitTableAir::Public(_) => {
                assert_eq!(lookups.len(), 1, "Public table should have one lookup");
            }
            CircuitTableAir::Alu(_) => {
                // ALU table sends 4 lookups per lane + 2 extra for double-step Horner a1/c1
                let expected_num_lookups = default_packing.alu_lanes() * 4
                    + 2 * (default_packing.horner_packed_steps() - 1);
                assert_eq!(
                    lookups.len(),
                    expected_num_lookups,
                    "ALU table should have {} lookups, found {}",
                    expected_num_lookups,
                    lookups.len()
                );
            }
            CircuitTableAir::Dynamic(_dynamic_air) => {
                assert!(
                    lookups.is_empty(),
                    "There is no dynamic table in this test, so no lookups expected"
                );
            }
        }
    }
}

#[test]
fn test_extension_field_batch_stark() {
    const D: usize = 4;
    type Ext4 = BinomialExtensionField<BabyBear, D>;
    let cfg = config::baby_bear().build();

    let mut builder = CircuitBuilder::<Ext4>::new();
    let x = builder.public_input();
    let y = builder.public_input();
    let z = builder.public_input();
    let expected = builder.public_input();
    let xy = builder.mul(x, y);
    let res = builder.add(xy, z);
    let diff = builder.sub(res, expected);
    builder.assert_zero(diff);

    let circuit = builder.build().unwrap();
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<BabyBearConfig, _, D>(
            &circuit,
            &TablePacking::default(),
            &[],
            &[],
            ConstraintProfile::Standard,
        )
        .unwrap();
    let (mut airs, degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();

    let mut runner = circuit.runner();
    let xv = Ext4::from_basis_coefficients_slice(&[
        BabyBear::from_u64(2),
        BabyBear::from_u64(3),
        BabyBear::from_u64(5),
        BabyBear::from_u64(7),
    ])
    .unwrap();
    let yv = Ext4::from_basis_coefficients_slice(&[
        BabyBear::from_u64(11),
        BabyBear::from_u64(13),
        BabyBear::from_u64(17),
        BabyBear::from_u64(19),
    ])
    .unwrap();
    let zv = Ext4::from_basis_coefficients_slice(&[
        BabyBear::from_u64(23),
        BabyBear::from_u64(29),
        BabyBear::from_u64(31),
        BabyBear::from_u64(37),
    ])
    .unwrap();
    let expected_v = xv * yv + zv;
    runner.set_public_inputs(&[xv, yv, zv, expected_v]).unwrap();
    let traces = runner.run().unwrap();

    let prover_data = ProverData::from_airs_and_degrees(&cfg, &mut airs, &degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);
    let prover = BatchStarkProver::new(cfg);

    let proof = prover
        .prove_all_tables(&traces, &circuit_prover_data)
        .unwrap();
    assert_eq!(proof.ext_degree, 4);
    // Ensure W was captured
    let expected_w = <Ext4 as ExtractBinomialW<BabyBear>>::extract_w().unwrap();
    assert_eq!(proof.w_binomial, Some(expected_w));
    prover.verify_all_tables(&proof).unwrap();
}

#[test]
fn test_extension_field_table_lookups() {
    const D: usize = 4;
    type Ext4 = BinomialExtensionField<BabyBear, D>;
    let cfg = config::baby_bear().build();

    let mut builder = CircuitBuilder::<Ext4>::new();
    let x = builder.public_input();
    let y = builder.public_input();
    let z = builder.public_input();
    let expected = builder.public_input();
    let xy = builder.mul(x, y);
    let res = builder.add(xy, z);
    let diff = builder.sub(res, expected);
    builder.assert_zero(diff);

    let circuit = builder.build().unwrap();
    let default_packing = TablePacking::default();
    let mut air_builders_ext4 = poseidon2_air_builders::<BabyBearConfig, 4>();
    air_builders_ext4.extend(recompose_air_builders::<BabyBearConfig, 4>(1, false));
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<BabyBearConfig, _, D>(
            &circuit,
            &default_packing,
            &[],
            &air_builders_ext4,
            ConstraintProfile::Standard,
        )
        .unwrap();
    let (mut airs, log_degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();

    let mut runner = circuit.runner();

    let xv = Ext4::from_basis_coefficients_slice(&[
        BabyBear::from_u64(2),
        BabyBear::from_u64(3),
        BabyBear::from_u64(5),
        BabyBear::from_u64(7),
    ])
    .unwrap();
    let yv = Ext4::from_basis_coefficients_slice(&[
        BabyBear::from_u64(11),
        BabyBear::from_u64(13),
        BabyBear::from_u64(17),
        BabyBear::from_u64(19),
    ])
    .unwrap();
    let zv = Ext4::from_basis_coefficients_slice(&[
        BabyBear::from_u64(23),
        BabyBear::from_u64(29),
        BabyBear::from_u64(31),
        BabyBear::from_u64(37),
    ])
    .unwrap();
    let expected_v = xv * yv + zv;
    runner.set_public_inputs(&[xv, yv, zv, expected_v]).unwrap();
    let traces = runner.run().unwrap();

    let prover_data = ProverData::from_airs_and_degrees(&cfg, &mut airs, &log_degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);

    let prover = BatchStarkProver::new(cfg);

    let proof = prover
        .prove_all_tables(&traces, &circuit_prover_data)
        .unwrap();
    assert_eq!(proof.ext_degree, 4);
    // Ensure W was captured
    let expected_w = <Ext4 as ExtractBinomialW<BabyBear>>::extract_w().unwrap();
    assert_eq!(proof.w_binomial, Some(expected_w));

    assert!(prover.verify_all_tables(&proof).is_ok());

    // Check that the generated lookups are correct and consistent across tables.
    for air in airs.iter_mut() {
        let lookups = LookupAir::get_lookups(air);

        match air {
            CircuitTableAir::Const(_) => {
                assert_eq!(lookups.len(), 1, "Const table should have one lookup");
            }
            CircuitTableAir::Public(_) => {
                assert_eq!(lookups.len(), 1, "Public table should have one lookup");
            }
            CircuitTableAir::Alu(_) => {
                // ALU table sends 4 lookups per lane + 2 extra for double-step Horner a1/c1
                let expected_num_lookups = default_packing.alu_lanes() * 4
                    + 2 * (default_packing.horner_packed_steps() - 1);
                assert_eq!(
                    lookups.len(),
                    expected_num_lookups,
                    "ALU table should have {} lookups, found {}",
                    expected_num_lookups,
                    lookups.len()
                );
            }
            CircuitTableAir::Dynamic(_dynamic_air) => {
                assert!(
                    lookups.is_empty(),
                    "There is no dynamic table in this test, so no lookups expected"
                );
            }
        }
    }
}

#[test]
fn test_koalabear_batch_stark_base_field() {
    let mut builder = CircuitBuilder::<KoalaBear>::new();
    let cfg = config::koala_bear().build();

    // a * b + 100 - (-1) == expected
    let a = builder.public_input();
    let b = builder.public_input();
    let expected = builder.public_input();
    let c = builder.define_const(KoalaBear::from_u64(100));
    let d = builder.define_const(KoalaBear::NEG_ONE);

    let ab = builder.mul(a, b);
    let add = builder.add(ab, c);
    let final_res = builder.sub(add, d);
    let diff = builder.sub(final_res, expected);
    builder.assert_zero(diff);

    let circuit = builder.build().unwrap();
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<KoalaBearConfig, _, 1>(
            &circuit,
            &TablePacking::default(),
            &[],
            &[],
            ConstraintProfile::Standard,
        )
        .unwrap();
    let (mut airs, degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();
    let mut runner = circuit.runner();

    let a_val = KoalaBear::from_u64(42);
    let b_val = KoalaBear::from_u64(13);
    let expected_val = KoalaBear::from_u64(647); // 42*13 + 100 - (-1)
    runner
        .set_public_inputs(&[a_val, b_val, expected_val])
        .unwrap();
    let traces = runner.run().unwrap();

    let prover_data = ProverData::from_airs_and_degrees(&cfg, &mut airs, &degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);
    let prover = BatchStarkProver::new(cfg);

    let proof = prover
        .prove_all_tables(&traces, &circuit_prover_data)
        .unwrap();
    assert_eq!(proof.ext_degree, 1);
    assert!(proof.w_binomial.is_none());
    prover.verify_all_tables(&proof).unwrap();
}

#[test]
fn test_koalabear_batch_stark_extension_field_d8() {
    const D: usize = 8;
    type KBExtField = BinomialExtensionField<KoalaBear, D>;
    let mut builder = CircuitBuilder::<KBExtField>::new();
    let cfg = config::koala_bear().build();

    // x * y * z == expected
    let x = builder.public_input();
    let y = builder.public_input();
    let expected = builder.public_input();
    let z = builder.define_const(
        KBExtField::from_basis_coefficients_slice(&[
            KoalaBear::from_u64(1),
            KoalaBear::NEG_ONE,
            KoalaBear::from_u64(2),
            KoalaBear::from_u64(3),
            KoalaBear::from_u64(4),
            KoalaBear::from_u64(5),
            KoalaBear::from_u64(6),
            KoalaBear::from_u64(7),
        ])
        .unwrap(),
    );

    let xy = builder.mul(x, y);
    let xyz = builder.mul(xy, z);
    let diff = builder.sub(xyz, expected);
    builder.assert_zero(diff);

    let circuit = builder.build().unwrap();
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<KoalaBearConfig, _, D>(
            &circuit,
            &TablePacking::default(),
            &[],
            &[],
            ConstraintProfile::Standard,
        )
        .unwrap();
    let (mut airs, degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();
    let mut runner = circuit.runner();

    let x_val = KBExtField::from_basis_coefficients_slice(&[
        KoalaBear::from_u64(4),
        KoalaBear::from_u64(6),
        KoalaBear::from_u64(8),
        KoalaBear::from_u64(10),
        KoalaBear::from_u64(12),
        KoalaBear::from_u64(14),
        KoalaBear::from_u64(16),
        KoalaBear::from_u64(18),
    ])
    .unwrap();
    let y_val = KBExtField::from_basis_coefficients_slice(&[
        KoalaBear::from_u64(12),
        KoalaBear::from_u64(14),
        KoalaBear::from_u64(16),
        KoalaBear::from_u64(18),
        KoalaBear::from_u64(20),
        KoalaBear::from_u64(22),
        KoalaBear::from_u64(24),
        KoalaBear::from_u64(26),
    ])
    .unwrap();
    let z_val = KBExtField::from_basis_coefficients_slice(&[
        KoalaBear::from_u64(1),
        KoalaBear::NEG_ONE,
        KoalaBear::from_u64(2),
        KoalaBear::from_u64(3),
        KoalaBear::from_u64(4),
        KoalaBear::from_u64(5),
        KoalaBear::from_u64(6),
        KoalaBear::from_u64(7),
    ])
    .unwrap();

    let expected_val = x_val * y_val * z_val;
    runner
        .set_public_inputs(&[x_val, y_val, expected_val])
        .unwrap();
    let traces = runner.run().unwrap();

    let prover_data = ProverData::from_airs_and_degrees(&cfg, &mut airs, &degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);
    let prover = BatchStarkProver::new(cfg);

    let proof = prover
        .prove_all_tables(&traces, &circuit_prover_data)
        .unwrap();
    assert_eq!(proof.ext_degree, 8);
    let expected_w = <KBExtField as ExtractBinomialW<KoalaBear>>::extract_w().unwrap();
    assert_eq!(proof.w_binomial, Some(expected_w));
    prover.verify_all_tables(&proof).unwrap();
}

#[test]
fn test_goldilocks_batch_stark_binomial_ext2() {
    const D: usize = 2;
    type Ext2 = BinomialExtensionField<Goldilocks, D>;
    let mut builder = CircuitBuilder::<Ext2>::new();
    let cfg = config::goldilocks().build();

    // x * y + z == expected
    let x = builder.public_input();
    let y = builder.public_input();
    let z = builder.public_input();
    let expected = builder.public_input();

    let xy = builder.mul(x, y);
    let res = builder.add(xy, z);
    let diff = builder.sub(res, expected);
    builder.assert_zero(diff);

    let circuit = builder.build().unwrap();
    let mut air_builders_ext2 = poseidon2_air_builders::<GoldilocksConfig, 2>();
    air_builders_ext2.extend(recompose_air_builders::<GoldilocksConfig, 2>(1, false));
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<GoldilocksConfig, _, D>(
            &circuit,
            &TablePacking::default(),
            &[],
            &air_builders_ext2,
            ConstraintProfile::Standard,
        )
        .unwrap();
    let (mut airs, degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();
    let mut runner = circuit.runner();

    let x_val =
        Ext2::from_basis_coefficients_slice(&[Goldilocks::from_u64(3), Goldilocks::NEG_ONE])
            .unwrap();
    let y_val =
        Ext2::from_basis_coefficients_slice(&[Goldilocks::from_u64(7), Goldilocks::from_u64(11)])
            .unwrap();
    let z_val =
        Ext2::from_basis_coefficients_slice(&[Goldilocks::from_u64(13), Goldilocks::from_u64(17)])
            .unwrap();
    let expected_val = x_val * y_val + z_val;

    runner
        .set_public_inputs(&[x_val, y_val, z_val, expected_val])
        .unwrap();
    let traces = runner.run().unwrap();

    let prover_data = ProverData::from_airs_and_degrees(&cfg, &mut airs, &degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);
    let prover = BatchStarkProver::new(cfg);

    let proof = prover
        .prove_all_tables(&traces, &circuit_prover_data)
        .unwrap();
    assert_eq!(proof.ext_degree, 2);
    let expected_w = <Ext2 as ExtractBinomialW<Goldilocks>>::extract_w().unwrap();
    assert_eq!(proof.w_binomial, Some(expected_w));
    prover.verify_all_tables(&proof).unwrap();
}

#[test]
fn test_goldilocks_poseidon2_circuit_build_and_run() {
    const D: usize = 2;
    type Ext2 = BinomialExtensionField<Goldilocks, D>;
    let mut rng = <rand::rngs::SmallRng as rand::SeedableRng>::seed_from_u64(0);
    let perm = Poseidon2Goldilocks::<8>::new_from_rng_128(&mut rng);
    let perm_for_hash = perm.clone();
    let mut builder = CircuitBuilder::<Ext2>::new();
    builder.enable_poseidon2_perm_width_8::<GoldilocksD2Width8, _>(
        generate_poseidon2_trace::<Ext2, GoldilocksD2Width8>,
        perm,
    );
    builder.enable_recompose::<Goldilocks>(generate_recompose_trace::<Goldilocks, Ext2>);
    let poseidon2_config = Poseidon2Config::GoldilocksD2Width8;
    let inputs = [builder.public_input(), builder.public_input()];
    let hash_outputs = builder
        .add_hash_slice(&poseidon2_config, &inputs, true)
        .unwrap();
    let expected0 = builder.public_input();
    let expected1 = builder.public_input();
    let sub0 = builder.sub(hash_outputs[0], expected0);
    builder.assert_zero(sub0);
    let sub1 = builder.sub(hash_outputs[1], expected1);
    builder.assert_zero(sub1);
    let circuit = builder.build().unwrap();
    let mut runner = circuit.runner();
    let in0 =
        Ext2::from_basis_coefficients_slice(&[Goldilocks::from_u64(1), Goldilocks::ZERO]).unwrap();
    let in1 =
        Ext2::from_basis_coefficients_slice(&[Goldilocks::from_u64(2), Goldilocks::ZERO]).unwrap();
    let hasher = PaddingFreeSponge::<Poseidon2Goldilocks<8>, 8, 4, 4>::new(perm_for_hash);
    let base_inputs = [
        Goldilocks::from_u64(1),
        Goldilocks::ZERO,
        Goldilocks::from_u64(2),
        Goldilocks::ZERO,
    ];
    let expected_hash = hasher.hash_iter(base_inputs);
    let out0 = Ext2::from_basis_coefficients_slice(&expected_hash[0..2]).unwrap();
    let out1 = Ext2::from_basis_coefficients_slice(&expected_hash[2..4]).unwrap();
    runner.set_public_inputs(&[in0, in1, out0, out1]).unwrap();
    let _traces = runner.run().unwrap();
}

#[test]
fn test_koalabear_modulus_constant() {
    // Verify KOALA_BEAR_MODULUS matches the actual KoalaBear field modulus.
    // The modulus p satisfies: from_u64(p) == 0 in the field.
    assert_eq!(
        KoalaBear::from_u64(KOALA_BEAR_MODULUS),
        KoalaBear::ZERO,
        "KOALA_BEAR_MODULUS (0x{:x}) does not match KoalaBear's actual modulus",
        KOALA_BEAR_MODULUS
    );

    // Verify the exact hex value (2130706433 = 0x7f000001).
    assert_eq!(KOALA_BEAR_MODULUS, 0x7f000001);
    assert_eq!(KOALA_BEAR_MODULUS, 2130706433);

    // Verify arithmetic at the modulus boundary with hardcoded expected values.
    // (p - 1) + 2 = 1 in the field
    let p_minus_1 = KoalaBear::from_u64(KOALA_BEAR_MODULUS - 1);
    assert_eq!(p_minus_1, KoalaBear::NEG_ONE);
    assert_eq!(p_minus_1 + KoalaBear::TWO, KoalaBear::ONE);

    // (p - 1) * (p - 1) = 1 in the field (since (-1) * (-1) = 1)
    assert_eq!(p_minus_1 * p_minus_1, KoalaBear::ONE);

    // Verify from_u64(p + 1) == 1
    assert_eq!(KoalaBear::from_u64(KOALA_BEAR_MODULUS + 1), KoalaBear::ONE);
}

#[test]
fn test_babybear_modulus_constant() {
    // Verify BABY_BEAR_MODULUS matches the actual BabyBear field modulus.
    assert_eq!(
        BabyBear::from_u64(BABY_BEAR_MODULUS),
        BabyBear::ZERO,
        "BABY_BEAR_MODULUS (0x{:x}) does not match BabyBear's actual modulus",
        BABY_BEAR_MODULUS
    );

    // Verify the exact hex value (2013265921 = 0x78000001).
    assert_eq!(BABY_BEAR_MODULUS, 0x78000001);
    assert_eq!(BABY_BEAR_MODULUS, 2013265921);

    // Verify arithmetic at the modulus boundary.
    let p_minus_1 = BabyBear::from_u64(BABY_BEAR_MODULUS - 1);
    assert_eq!(p_minus_1, BabyBear::NEG_ONE);
    assert_eq!(p_minus_1 + BabyBear::TWO, BabyBear::ONE);
    assert_eq!(BabyBear::from_u64(BABY_BEAR_MODULUS + 1), BabyBear::ONE);
}

#[test]
fn test_mul_only_circuit_padding() {
    // Circuit with only mul operations; ALU table still needs correct padding/lanes handling.
    let mut builder = CircuitBuilder::<BabyBear>::new();
    let cfg = config::baby_bear().build();

    let x = builder.public_input();
    let y = builder.public_input();

    // Only multiplication, no addition
    builder.mul(x, y);

    let circuit = builder.build().unwrap();
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<BabyBearConfig, _, 1>(
            &circuit,
            &TablePacking::default(),
            &[],
            &[],
            ConstraintProfile::Standard,
        )
        .unwrap();
    let (mut airs, degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();
    let mut runner = circuit.runner();

    let x_val = BabyBear::from_u64(7);
    let y_val = BabyBear::from_u64(11);
    runner.set_public_inputs(&[x_val, y_val]).unwrap();
    let traces = runner.run().unwrap();

    let prover_data = ProverData::from_airs_and_degrees(&cfg, &mut airs, &degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);

    let prover = BatchStarkProver::new(cfg);

    let proof = prover
        .prove_all_tables(&traces, &circuit_prover_data)
        .unwrap();
    prover.verify_all_tables(&proof).unwrap();
}

#[test]
fn test_add_only_circuit_padding() {
    // Circuit with only add operations; ALU table still needs correct padding/lanes handling.
    let mut builder = CircuitBuilder::<BabyBear>::new();
    let cfg = config::baby_bear().build();

    let x = builder.public_input();
    let y = builder.public_input();
    let expected = builder.public_input();

    // Only addition, no multiplication
    let sum = builder.add(x, y);
    let diff = builder.sub(sum, expected);
    builder.assert_zero(diff);

    let circuit = builder.build().unwrap();
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<BabyBearConfig, _, 1>(
            &circuit,
            &TablePacking::default(),
            &[],
            &[],
            ConstraintProfile::Standard,
        )
        .unwrap();
    let (mut airs, degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();
    let mut runner = circuit.runner();

    let x_val = BabyBear::from_u64(42);
    let y_val = BabyBear::from_u64(13);
    let expected_val = x_val + y_val;
    runner
        .set_public_inputs(&[x_val, y_val, expected_val])
        .unwrap();
    let traces = runner.run().unwrap();

    let prover_data = ProverData::from_airs_and_degrees(&cfg, &mut airs, &degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);

    let prover = BatchStarkProver::new(cfg);

    let proof = prover
        .prove_all_tables(&traces, &circuit_prover_data)
        .unwrap();
    prover.verify_all_tables(&proof).unwrap();
}

fn koala_ef5_lift(b: KoalaBear) -> QuinticTrinomialExtensionField<KoalaBear> {
    QuinticTrinomialExtensionField::<KoalaBear>::from_basis_coefficients_slice(&[
        b,
        KoalaBear::ZERO,
        KoalaBear::ZERO,
        KoalaBear::ZERO,
        KoalaBear::ZERO,
    ])
    .expect("basis slice")
}

#[test]
fn test_koalabear_quintic_trinomial_batch_stark_with_poseidon_d1() {
    const D: usize = 5;
    type EF5 = QuinticTrinomialExtensionField<KoalaBear>;

    // Must match KoalaBearD1Width16::round_constants() in poseidon2-circuit-air (not RNG-derived).
    let inner_perm = default_koalabear_poseidon2_16();
    let mut sponge0 = [KoalaBear::ZERO; 16];
    sponge0[0] = KoalaBear::from_u64(11);
    sponge0[1] = KoalaBear::from_u64(13);
    let sponge_out = inner_perm.permute(sponge0);
    let lift_perm = LiftPermToQuintic::new(inner_perm);

    let in0 = koala_ef5_lift(KoalaBear::from_u64(11));
    let in1 = koala_ef5_lift(KoalaBear::from_u64(13));
    let exp0 = koala_ef5_lift(sponge_out[0]);
    let exp1 = koala_ef5_lift(sponge_out[1]);

    let mut builder = CircuitBuilder::<EF5>::new();
    builder.enable_poseidon2_perm_base::<KoalaBearD1Width16, _>(
        generate_poseidon2_trace::<EF5, KoalaBearD1Width16>,
        lift_perm,
    );

    let in_a = builder.public_input();
    let in_b = builder.public_input();
    let mut perm_inputs: [Option<_>; 16] = [None; 16];
    perm_inputs[0] = Some(in_a);
    perm_inputs[1] = Some(in_b);
    let (_pid, hash_outputs) = builder
        .add_poseidon2_perm_base(&Poseidon2PermCallBase {
            config: Poseidon2Config::KoalaBearD1Width16,
            new_start: true,
            inputs: perm_inputs,
            // Only CTL-expose rate limbs that are wired into the rest of the circuit; unused
            // exposed outputs would leave WitnessChecks Receive contributions unmatched.
            out_ctl: [true; 8],
            return_all_outputs: false,
        })
        .unwrap();
    let e0 = builder.public_input();
    let e1 = builder.public_input();
    let h0_diff = builder.sub(hash_outputs[0].unwrap(), e0);
    let h1_diff = builder.sub(hash_outputs[1].unwrap(), e1);
    builder.assert_zero(h0_diff);
    builder.assert_zero(h1_diff);

    let circuit = builder.build().unwrap();
    let cfg = config::koala_bear().build();

    let npo_prep: Vec<Box<dyn NpoPreprocessor<KoalaBear>>> = vec![Box::new(Poseidon2Preprocessor)];
    let air_builders = poseidon2_air_builders_d5::<KoalaBearConfig>();
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<KoalaBearConfig, _, D>(
            &circuit,
            &TablePacking::default(),
            &npo_prep,
            &air_builders,
            ConstraintProfile::Standard,
        )
        .unwrap();
    let (mut airs, degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();
    let mut runner = circuit.runner();

    runner.set_public_inputs(&[in0, in1, exp0, exp1]).unwrap();
    let traces = runner.run().unwrap();

    let prover_data = ProverData::from_airs_and_degrees(&cfg, &mut airs, &degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);

    let mut prover = BatchStarkProver::new(cfg);
    for p in poseidon2_table_provers_d5(Poseidon2Config::KoalaBearD1Width16) {
        prover.register_table_prover(p);
    }

    let proof = prover
        .prove_all_tables(&traces, &circuit_prover_data)
        .unwrap();
    assert_eq!(proof.ext_degree, D);
    assert!(proof.w_binomial.is_none());
    assert!(proof.alu_quintic_trinomial);
    prover.verify_all_tables(&proof).unwrap();
}

/// Two D=1 Poseidon rows in an EF5 circuit: the second row uses `new_start=false` so the full
/// 16-wide state chains through the compact D=1 preprocessed layout (sponge selectors, not Merkle).
#[test]
fn test_koalabear_quintic_trinomial_batch_stark_poseidon_d1_sponge_chain() {
    const D: usize = 5;
    type EF5 = QuinticTrinomialExtensionField<KoalaBear>;

    let inner_perm = default_koalabear_poseidon2_16();
    let mut sponge0 = [KoalaBear::ZERO; 16];
    sponge0[0] = KoalaBear::from_u64(11);
    sponge0[1] = KoalaBear::from_u64(13);
    let sponge_out0 = inner_perm.permute(sponge0);
    let sponge_out1 = inner_perm.permute(sponge_out0);
    let lift_perm = LiftPermToQuintic::new(inner_perm);

    let in0 = koala_ef5_lift(KoalaBear::from_u64(11));
    let in1 = koala_ef5_lift(KoalaBear::from_u64(13));
    let exp0 = koala_ef5_lift(sponge_out1[0]);
    let exp1 = koala_ef5_lift(sponge_out1[1]);

    let mut builder = CircuitBuilder::<EF5>::new();
    builder.enable_poseidon2_perm_base::<KoalaBearD1Width16, _>(
        generate_poseidon2_trace::<EF5, KoalaBearD1Width16>,
        lift_perm,
    );

    let in_a = builder.public_input();
    let in_b = builder.public_input();
    let mut perm0_inputs: [Option<_>; 16] = [None; 16];
    perm0_inputs[0] = Some(in_a);
    perm0_inputs[1] = Some(in_b);
    let (_pid0, _hash0) = builder
        .add_poseidon2_perm_base(&Poseidon2PermCallBase {
            config: Poseidon2Config::KoalaBearD1Width16,
            new_start: true,
            inputs: perm0_inputs,
            out_ctl: [false; 8],
            return_all_outputs: false,
        })
        .unwrap();

    let perm1_inputs: [Option<_>; 16] = [None; 16];
    let (_pid1, hash1_outputs) = builder
        .add_poseidon2_perm_base(&Poseidon2PermCallBase {
            config: Poseidon2Config::KoalaBearD1Width16,
            new_start: false,
            inputs: perm1_inputs,
            out_ctl: [true; 8],
            return_all_outputs: false,
        })
        .unwrap();
    let e0 = builder.public_input();
    let e1 = builder.public_input();
    let h0_diff = builder.sub(hash1_outputs[0].unwrap(), e0);
    let h1_diff = builder.sub(hash1_outputs[1].unwrap(), e1);
    builder.assert_zero(h0_diff);
    builder.assert_zero(h1_diff);

    let circuit = builder.build().unwrap();
    let cfg = config::koala_bear().build();

    let npo_prep: Vec<Box<dyn NpoPreprocessor<KoalaBear>>> = vec![Box::new(Poseidon2Preprocessor)];
    let air_builders = poseidon2_air_builders_d5::<KoalaBearConfig>();
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<KoalaBearConfig, _, D>(
            &circuit,
            &TablePacking::default(),
            &npo_prep,
            &air_builders,
            ConstraintProfile::Standard,
        )
        .unwrap();
    let (mut airs, degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();
    let mut runner = circuit.runner();

    runner.set_public_inputs(&[in0, in1, exp0, exp1]).unwrap();
    let traces = runner.run().unwrap();

    let prover_data = ProverData::from_airs_and_degrees(&cfg, &mut airs, &degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);

    let mut prover = BatchStarkProver::new(cfg);
    for p in poseidon2_table_provers_d5(Poseidon2Config::KoalaBearD1Width16) {
        prover.register_table_prover(p);
    }

    let proof = prover
        .prove_all_tables(&traces, &circuit_prover_data)
        .unwrap();
    assert_eq!(proof.ext_degree, D);
    assert!(proof.w_binomial.is_none());
    assert!(proof.alu_quintic_trinomial);
    prover.verify_all_tables(&proof).unwrap();
}

#[test]
fn test_stark_serialization_round_trip() {
    let mut builder = CircuitBuilder::<BabyBear>::new();

    let x = builder.public_input();
    let expected = builder.public_input();
    let c5 = builder.define_const(BabyBear::from_u64(5));
    let c2 = builder.define_const(BabyBear::from_u64(2));
    let mul_result = builder.mul(c5, c2);
    let add_result = builder.add(x, mul_result);
    let diff = builder.sub(add_result, expected);
    builder.assert_zero(diff);

    let circuit = builder.build().unwrap();
    let cfg = config::baby_bear().build();
    let (airs_degrees, primitive_columns, non_primitive_columns) =
        get_airs_and_degrees_with_prep::<BabyBearConfig, _, 1>(
            &circuit,
            &TablePacking::default(),
            &[],
            &[],
            ConstraintProfile::Standard,
        )
        .unwrap();
    let (mut airs, log_degrees): (Vec<_>, Vec<usize>) = airs_degrees.into_iter().unzip();
    let prover_data = ProverData::from_airs_and_degrees(&cfg, &mut airs, &log_degrees);
    let circuit_prover_data =
        CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);

    let mut runner = circuit.runner();
    let x_val = BabyBear::from_u64(7);
    let expected_val = BabyBear::from_u64(17); // 7 + 5*2 = 17
    runner.set_public_inputs(&[x_val, expected_val]).unwrap();
    let traces = runner.run().unwrap();

    let prover = BatchStarkProver::new(cfg);
    let proof = prover
        .prove_all_tables(&traces, &circuit_prover_data)
        .unwrap();

    let original_preprocessed = proof
        .stark_common
        .preprocessed
        .as_ref()
        .expect("preprocessed binding must be present");
    let original_matrix_to_instance = original_preprocessed.matrix_to_instance.clone();
    let original_instances_len = original_preprocessed.instances.len();

    let bytes = postcard::to_allocvec(&proof).expect("serialize proof");
    let deserialized: BatchStarkProof<BabyBearConfig> =
        postcard::from_bytes(&bytes).expect("deserialize proof");

    let restored_preprocessed = deserialized
        .stark_common
        .preprocessed
        .as_ref()
        .expect("preprocessed binding must survive (de)serialization");
    assert_eq!(
        restored_preprocessed.matrix_to_instance,
        original_matrix_to_instance
    );
    assert_eq!(
        restored_preprocessed.instances.len(),
        original_instances_len
    );

    // Verification must succeed against the deserialized proof, relying only on the
    // proof's own `stark_common` for the preprocessed binding.
    prover
        .verify_all_tables(&deserialized)
        .expect("verification uses proof.stark_common");
}
