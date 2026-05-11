//! 2-to-1 proof aggregation example (binary tree).
//!
//! Builds a full binary aggregation tree from distinct base proofs:
//! 1. **Leaves**: `2^(N+1)` dummy circuits (each a single distinct constant),
//!    each proved independently with batch STARK.
//! 2. **Levels 1..N+1**: Pairwise 2-to-1 aggregation up the tree until a
//!    single root proof remains.
//!
//! `N` is the `--num-recursive-layers` argument (default 1).
//!
//! ## What this proves
//!
//! The root proof attests that every base proof in the tree is valid.  All
//! base proofs are genuinely distinct (different constant values) so the
//! circuit optimizer cannot collapse the two verifications inside an
//! aggregation node.
//!
//! ## Usage
//!
//! ```bash
//! # 4 base proofs, 2 aggregation levels (default)
//! cargo run --release --example recursive_aggregation -- --field koala-bear
//!
//! # KoalaBear with quintic challenge extension (D = 5)
//! cargo run --release --example recursive_aggregation -- --field koala-bear --quintic
//!
//! # 8 base proofs, 3 aggregation levels, custom FRI parameters
//! cargo run --release --example recursive_aggregation -- \
//!     --field koala-bear \
//!     --num-recursive-layers 2 \
//!     --log-blowup 3 \
//!     --max-log-arity 4 \
//!     --log-final-poly-len 5 \
//!     --query-pow-bits 16
//! ```

#[macro_use]
mod common;
use common::*;
use p3_batch_stark::ProverData;

#[derive(Parser, Debug)]
#[command(version, about = "2-to-1 proof aggregation example")]
struct Args {
    /// Tree depth (total base proofs = 2^(tree_depth)).  (1 = single pair, 2 = 4 leaves, …)
    #[arg(
        long,
        default_value_t = 1,
        help = "Tree depth (total base proofs = 2^(tree_depth))"
    )]
    num_recursive_layers: usize,

    #[arg(short, long, ignore_case = true, value_enum, default_value_t = FieldOption::KoalaBear)]
    pub field: FieldOption,

    /// Use quintic (D = 5) challenge extension (KoalaBear only; incompatible with baby-bear / goldilocks).
    #[arg(long, default_value_t = false)]
    pub quintic: bool,

    #[arg(
        long,
        default_value_t = 2,
        help = "Logarithmic blowup factor for the LDE"
    )]
    pub log_blowup: usize,

    #[arg(
        long,
        default_value_t = 2,
        help = "Maximum arity allowed during FRI folding phases"
    )]
    pub max_log_arity: usize,

    #[arg(long, default_value_t = 0, help = "Height of the Merkle cap to open")]
    pub cap_height: usize,

    #[arg(
        long,
        default_value_t = 6,
        help = "Log size of final polynomial after FRI folding"
    )]
    pub log_final_poly_len: usize,

    #[arg(
        long,
        default_value_t = 0,
        help = "PoW grinding bits during FRI commit phase"
    )]
    pub commit_pow_bits: usize,

    #[arg(
        long,
        default_value_t = 15,
        help = "PoW grinding bits during FRI query phase"
    )]
    pub query_pow_bits: usize,

    #[arg(
        long,
        default_value_t = 1,
        help = "Number of public lanes for the table packing in recursive layers"
    )]
    pub public_lanes: usize,

    #[arg(
        long,
        default_value_t = 3,
        help = "Number of ALU lanes for the table packing in recursive layers"
    )]
    pub alu_lanes: usize,

    /// Pack this many consecutive HornerAcc steps (same `b`) per ALU row on lane 0 (must be >= 2).
    #[arg(
        long,
        default_value_t = 4,
        help = "Pack this many consecutive HornerAcc steps (same `b`) per ALU row on lane 0 (must be >= 2)"
    )]
    pub horner_packed_steps: usize,

    #[arg(
        long,
        default_value_t = 1,
        help = "Number of recompose lanes for the table packing in recursive layers"
    )]
    pub recompose_lanes: usize,

    // TODO: Update once https://github.com/Plonky3/Plonky3/pull/1329 lands
    #[arg(
        long,
        default_value_t = 124,
        help = "Targeted security level (conjectured)"
    )]
    pub security_level: usize,

    #[arg(long, default_value_t = false, help = "Enable ZK mode (HidingFriPcs)")]
    pub zk: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Disable recompose NPO (use only Poseidon2 perm)"
    )]
    pub disable_recompose_npo: bool,
}

impl Args {
    pub const fn to_fri_params(&self) -> FriParams {
        FriParams {
            log_blowup: self.log_blowup,
            max_log_arity: self.max_log_arity,
            cap_height: self.cap_height,
            log_final_poly_len: self.log_final_poly_len,
            commit_pow_bits: self.commit_pow_bits,
            query_pow_bits: self.query_pow_bits,
        }
    }

    pub fn table_packing(&self) -> TablePacking {
        TablePacking::new(self.public_lanes, self.alu_lanes)
            .with_horner_pack_k(self.horner_packed_steps)
            .with_npo_lanes(NpoTypeId::recompose(), self.recompose_lanes)
    }
}

fn main() {
    init_logger();

    let args = Args::parse();
    let fri_params = args.to_fri_params();
    let table_packing = args.table_packing();

    assert!(args.num_recursive_layers >= 1);

    assert_quintic_field(args.field, args.quintic);

    info!(
        "2-to-1 aggregation with field {:?}, quintic {}, {} aggregation recursive layers",
        args.field, args.quintic, args.num_recursive_layers
    );

    match args.field {
        FieldOption::KoalaBear if args.quintic => koala_bear_quintic::run(
            args.num_recursive_layers,
            &fri_params,
            &table_packing,
            args.security_level,
            args.zk,
            args.disable_recompose_npo,
        ),
        FieldOption::KoalaBear => koala_bear::run(
            args.num_recursive_layers,
            &fri_params,
            &table_packing,
            args.security_level,
            args.zk,
            args.disable_recompose_npo,
        ),
        FieldOption::BabyBear => baby_bear::run(
            args.num_recursive_layers,
            &fri_params,
            &table_packing,
            args.security_level,
            args.zk,
            args.disable_recompose_npo,
        ),
        FieldOption::Goldilocks => goldilocks::run(
            args.num_recursive_layers,
            &fri_params,
            &table_packing,
            args.security_level,
            args.zk,
            args.disable_recompose_npo,
        ),
    }
}

/// KoalaBear quintic extension (`D = 5`) variant of [`define_field_module`] for aggregation.
macro_rules! define_field_module_aggregation_quintic {
    (
        $mod_name:ident,
        $field:ty,
        $perm:ty,
        $default_perm:path,
        $poseidon2_config:expr,
        $poseidon2_circuit_config:ty,
        $width:expr,
        $rate:expr,
        $digest_elems:expr,
        $backend_width:expr,
        $backend_rate:expr
    ) => {
        mod $mod_name {
            use p3_batch_stark::ProverData;

            use super::*;

            define_quintic_poseidon_perm_lift_and_types!(
                $field,
                $perm,
                $default_perm,
                $poseidon2_config,
                $poseidon2_circuit_config,
                $width,
                $rate,
                $digest_elems,
                $backend_width,
                $backend_rate
            );

            fn prove_dummy_circuit(
                constant_value: u32,
                config: &ConfigWithFriParams,
                table_packing: &TablePacking,
            ) -> RecursionOutput<ConfigWithFriParams> {
                let mut builder = CircuitBuilder::new();
                let c = builder.alloc_const(F::from_u32(constant_value), "dummy_const");
                let expected = builder.alloc_public_input("expected");
                builder.connect(c, expected);
                let circuit = builder.build().unwrap();
                let (airs_degrees, primitive_columns, non_primitive_columns) =
                    get_airs_and_degrees_with_prep::<ConfigWithFriParams, F, 1>(
                        &circuit,
                        &table_packing,
                        &[],
                        &[],
                        ConstraintProfile::Standard,
                    )
                    .unwrap();
                let (mut airs, degrees): (Vec<_>, Vec<_>) = airs_degrees.into_iter().unzip();
                let mut runner = circuit.runner();
                runner
                    .set_public_inputs(&[F::from_u32(constant_value)])
                    .unwrap();
                let traces = runner.run().unwrap();
                let ext_degrees: Vec<usize> =
                    degrees.iter().map(|&d| d + config.is_zk()).collect();
                let prover_data =
                    ProverData::from_airs_and_degrees(config, &mut airs, &ext_degrees);
                let circuit_prover_data = CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);
                let prover =
                    BatchStarkProver::new(config.clone()).with_table_packing(table_packing.clone());
                let proof = prover
                    .prove_all_tables(&traces, &circuit_prover_data)
                    .expect("Failed to prove dummy circuit");
                report_proof_size(&proof);
                prover
                    .verify_all_tables(&proof)
                    .expect("Failed to verify dummy proof");
                RecursionOutput(proof, Rc::new(circuit_prover_data))
            }

            pub fn run(
                num_recursive_layers: usize,
                fri_params: &FriParams,
                table_packing: &TablePacking,
                security_level: usize,
                zk: bool,
                disable_recompose_npo: bool,
            ) {
                if zk {
                    tracing::warn!(
                        "--zk is not yet supported for KoalaBear quintic in recursive_aggregation; \
                         using non-ZK config for all layers."
                    );
                }

                let base_table_packing = TablePacking::new(1, 1)
                    .with_fri_params(fri_params.log_final_poly_len, fri_params.log_blowup);
                let backend = FriRecursionBackend::<$backend_width, $backend_rate>::new_d5(
                    $poseidon2_config,
                );

                let tree_depth = num_recursive_layers;
                let num_leaves = 1usize << tree_depth;
                info!("Binary aggregation tree: {num_leaves} base proofs, {tree_depth} levels");

                macro_rules! run_aggregation {
                    ($cfg_type:ident, $config_base:expr, $config_agg:expr, $prove_base_fn:ident) => {{
                        let config_base: $cfg_type = $config_base;
                        let mut proofs: Vec<RecursionOutput<$cfg_type>> = (0..num_leaves)
                            .map(|i| {
                                let val = (i + 1) as u32;
                                info!("Base proof {i} (const = {val})");
                                $prove_base_fn(val, &config_base, &base_table_packing)
                            })
                            .collect();

                        let mut prep_cache: Option<AggregationPrepCache<$cfg_type>> = None;
                        let mut level = 0u32;
                        while proofs.len() > 1 {
                            level += 1;
                            let pairs = proofs.len() / 2;
                            info!(
                                "Aggregation level {level}: {} proofs -> {pairs}",
                                proofs.len()
                            );

                            let agg_params = ProveNextLayerParams {
                                table_packing: if level == 1 {
                                    TablePacking::new(2, 2)
                                } else {
                                    table_packing.clone()
                                }
                                .with_fri_params(
                                    fri_params.log_final_poly_len,
                                    fri_params.log_blowup,
                                ),
                                constraint_profile: ConstraintProfile::Standard,
                            };
                            let agg_config: $cfg_type = $config_agg(level as u64);

                            let mut next_level = Vec::with_capacity(pairs);
                            for pair_idx in 0..pairs {
                                let li = pair_idx * 2;
                                let left = proofs[li].into_recursion_input::<BatchOnly>();
                                let right = proofs[li + 1].into_recursion_input::<BatchOnly>();

                                let out = build_and_prove_aggregation_layer::<$cfg_type, _, _, _, D>(
                                    &left, &right, &agg_config, &backend, &agg_params,
                                    Some(&mut prep_cache),
                                )
                                .unwrap_or_else(|e| {
                                    panic!("Failed at level {level}, pair {pair_idx}: {e:?}")
                                });

                                report_proof_size(&out.0);
                                let mut verifier = BatchStarkProver::new(agg_config.clone())
                                    .with_table_packing(agg_params.table_packing.clone());
                                verifier.register_poseidon2_table::<D>($poseidon2_config);
                                if !disable_recompose_npo {
                                    verifier.register_recompose_table::<D>($poseidon2_config.d() != D);
                                }
                                verifier
                                    .verify_all_tables(&out.0)
                                    .unwrap_or_else(|e| {
                                        panic!("Verification failed at level {level}, pair {pair_idx}: {e:?}")
                                    });
                                next_level.push(out);
                            }
                            proofs = next_level;
                        }
                    }};
                }

                run_aggregation!(
                    ConfigWithFriParams,
                    config_with_fri_params(fri_params, security_level, true),
                    |_lvl| config_with_fri_params(
                        fri_params,
                        security_level,
                        disable_recompose_npo,
                    ),
                    prove_dummy_circuit
                );

                info!("All levels verified successfully");
            }
        }
    };
}

macro_rules! define_field_module {
    (
        $mod_name:ident,
        $field:ty,
        $perm:ty,
        $default_perm:path,
        $poseidon2_config:expr,
        $poseidon2_circuit_config:ty,
        $d:expr,
        $width:expr,
        $rate:expr,
        $digest_elems:expr,
        $enable_poseidon2_fn:ident,
        $default_perm_circuit:path,
        $backend_width:expr,
        $backend_rate:expr
    ) => {
        mod $mod_name {
            use super::*;

            define_field_module_types!(
                $field,
                $perm,
                $default_perm,
                $poseidon2_config,
                $poseidon2_circuit_config,
                $d,
                $width,
                $rate,
                $digest_elems,
                $enable_poseidon2_fn,
                $default_perm_circuit,
                $backend_width,
                $backend_rate,
                enable_recompose
            );

            /// Build a dummy circuit with a single constant and prove it (non-ZK).
            fn prove_dummy_circuit(
                constant_value: u32,
                config: &ConfigWithFriParams,
                table_packing: &TablePacking,
            ) -> RecursionOutput<ConfigWithFriParams> {
                let mut builder = CircuitBuilder::new();
                let c = builder.alloc_const(F::from_u32(constant_value), "dummy_const");
                let expected = builder.alloc_public_input("expected");
                builder.connect(c, expected);
                let circuit = builder.build().unwrap();
                let (airs_degrees, primitive_columns, non_primitive_columns) =
                    get_airs_and_degrees_with_prep::<ConfigWithFriParams, F, 1>(
                        &circuit,
                        &table_packing,
                        &[],
                        &[],
                        ConstraintProfile::Standard,
                    )
                    .unwrap();
                let (mut airs, degrees): (Vec<_>, Vec<_>) = airs_degrees.into_iter().unzip();
                let mut runner = circuit.runner();
                runner
                    .set_public_inputs(&[F::from_u32(constant_value)])
                    .unwrap();
                let traces = runner.run().unwrap();
                let ext_degrees: Vec<usize> =
                    degrees.iter().map(|&d| d + config.is_zk()).collect();
                let prover_data =
                    ProverData::from_airs_and_degrees(config, &mut airs, &ext_degrees);
                let circuit_prover_data = CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);
                let prover =
                    BatchStarkProver::new(config.clone()).with_table_packing(table_packing.clone());
                let proof = prover
                    .prove_all_tables(&traces, &circuit_prover_data)
                    .expect("Failed to prove dummy circuit");
                report_proof_size(&proof);
                prover
                    .verify_all_tables(&proof)
                    .expect("Failed to verify dummy proof");
                RecursionOutput(proof, Rc::new(circuit_prover_data))
            }

            /// Build a dummy circuit with a single constant and prove it (ZK).
            fn prove_dummy_circuit_zk(
                constant_value: u32,
                config: &ConfigWithFriParamsZk,
                table_packing: &TablePacking,
            ) -> RecursionOutput<ConfigWithFriParamsZk> {
                let mut builder = CircuitBuilder::new();
                let c = builder.alloc_const(F::from_u32(constant_value), "dummy_const");
                let expected = builder.alloc_public_input("expected");
                builder.connect(c, expected);
                let circuit = builder.build().unwrap();
                let (airs_degrees, primitive_columns, non_primitive_columns) =
                    get_airs_and_degrees_with_prep::<ConfigWithFriParamsZk, F, 1>(
                        &circuit,
                        &table_packing,
                        &[],
                        &[],
                        ConstraintProfile::Standard,
                    )
                    .unwrap();
                let (mut airs, degrees): (Vec<_>, Vec<_>) = airs_degrees.into_iter().unzip();
                let mut runner = circuit.runner();
                runner
                    .set_public_inputs(&[F::from_u32(constant_value)])
                    .unwrap();
                let traces = runner.run().unwrap();
                let ext_degrees: Vec<usize> =
                    degrees.iter().map(|&d| d + config.is_zk()).collect();
                let prover_data =
                    ProverData::from_airs_and_degrees(config, &mut airs, &ext_degrees);
                let circuit_prover_data = CircuitProverData::new(prover_data, primitive_columns, non_primitive_columns);
                let prover =
                    BatchStarkProver::new(config.clone()).with_table_packing(table_packing.clone());
                let proof = prover
                    .prove_all_tables(&traces, &circuit_prover_data)
                    .expect("Failed to prove dummy circuit (ZK)");
                report_proof_size(&proof);
                prover
                    .verify_all_tables(&proof)
                    .expect("Failed to verify dummy proof (ZK)");
                RecursionOutput(proof, Rc::new(circuit_prover_data))
            }

            pub fn run(
                num_recursive_layers: usize,
                fri_params: &FriParams,
                table_packing: &TablePacking,
                security_level: usize,
                zk: bool,
                disable_recompose_npo: bool,
            ) {
                let base_table_packing = TablePacking::new(1, 1)
                    .with_fri_params(fri_params.log_final_poly_len, fri_params.log_blowup);
                let backend = FriRecursionBackend::<$backend_width, $backend_rate>::new(
                    $poseidon2_config,
                )
                .for_extension_degree::<$d>();

                let tree_depth = num_recursive_layers;
                let num_leaves = 1usize << tree_depth;
                info!("Binary aggregation tree: {num_leaves} base proofs, {tree_depth} levels");

                macro_rules! run_aggregation {
                    ($cfg_type:ident, $config_base:expr, $config_agg:expr, $prove_base_fn:ident) => {{
                        let config_base: $cfg_type = $config_base;
                        let mut proofs: Vec<RecursionOutput<$cfg_type>> = (0..num_leaves)
                            .map(|i| {
                                let val = (i + 1) as u32;
                                info!("Base proof {i} (const = {val})");
                                $prove_base_fn(val, &config_base, &base_table_packing)
                            })
                            .collect();

                        let mut prep_cache: Option<AggregationPrepCache<$cfg_type>> = None;
                        let mut level = 0u32;
                        while proofs.len() > 1 {
                            level += 1;
                            let pairs = proofs.len() / 2;
                            info!(
                                "Aggregation level {level}: {} proofs -> {pairs}",
                                proofs.len()
                            );

                            let agg_params = ProveNextLayerParams {
                                table_packing: if level == 1 {
                                    TablePacking::new(2, 2)
                                } else {
                                    table_packing.clone()
                                }
                                .with_fri_params(
                                    fri_params.log_final_poly_len,
                                    fri_params.log_blowup,
                                ),
                                constraint_profile: ConstraintProfile::Standard,
                            };
                            let agg_config: $cfg_type = $config_agg(level as u64);

                            let mut next_level = Vec::with_capacity(pairs);
                            for pair_idx in 0..pairs {
                                let li = pair_idx * 2;
                                let left = proofs[li].into_recursion_input::<BatchOnly>();
                                let right = proofs[li + 1].into_recursion_input::<BatchOnly>();

                                let out = build_and_prove_aggregation_layer::<$cfg_type, _, _, _, D>(
                                    &left, &right, &agg_config, &backend, &agg_params,
                                    Some(&mut prep_cache),
                                )
                                .unwrap_or_else(|e| {
                                    panic!("Failed at level {level}, pair {pair_idx}: {e:?}")
                                });

                                report_proof_size(&out.0);
                                let mut verifier = BatchStarkProver::new(agg_config.clone())
                                    .with_table_packing(agg_params.table_packing.clone());
                                verifier.register_poseidon2_table::<$d>($poseidon2_config);
                                if !disable_recompose_npo {
                                    verifier.register_recompose_table::<$d>($poseidon2_config.d() != $d);
                                }
                                verifier
                                    .verify_all_tables(&out.0)
                                    .unwrap_or_else(|e| {
                                        panic!("Verification failed at level {level}, pair {pair_idx}: {e:?}")
                                    });
                                next_level.push(out);
                            }
                            proofs = next_level;
                        }
                    }};
                }

                if zk {
                    run_aggregation!(
                        ConfigWithFriParamsZk,
                        config_with_fri_params_zk(fri_params, security_level, true, 0),
                        |lvl| config_with_fri_params_zk(
                            fri_params,
                            security_level,
                            disable_recompose_npo,
                            lvl,
                        ),
                        prove_dummy_circuit_zk
                    );
                } else {
                    run_aggregation!(
                        ConfigWithFriParams,
                        config_with_fri_params(fri_params, security_level, true),
                        |_lvl| config_with_fri_params(
                            fri_params,
                            security_level,
                            disable_recompose_npo,
                        ),
                        prove_dummy_circuit
                    );
                }

                info!("All levels verified successfully");
            }
        }
    };
}

define_field_module_aggregation_quintic!(
    koala_bear_quintic,
    p3_koala_bear::KoalaBear,
    p3_koala_bear::Poseidon2KoalaBear<16>,
    p3_koala_bear::default_koalabear_poseidon2_16,
    Poseidon2Config::KoalaBearD1Width16,
    p3_poseidon2_circuit_air::KoalaBearD1Width16,
    16,
    8,
    8,
    16,
    8
);

define_field_module!(
    koala_bear,
    p3_koala_bear::KoalaBear,
    p3_koala_bear::Poseidon2KoalaBear<16>,
    p3_koala_bear::default_koalabear_poseidon2_16,
    Poseidon2Config::KoalaBearD4Width16,
    p3_poseidon2_circuit_air::KoalaBearD4Width16,
    4,
    16,
    8,
    8,
    enable_poseidon2_perm,
    p3_koala_bear::default_koalabear_poseidon2_16,
    16,
    8
);

define_field_module!(
    baby_bear,
    p3_baby_bear::BabyBear,
    p3_baby_bear::Poseidon2BabyBear<16>,
    p3_baby_bear::default_babybear_poseidon2_16,
    Poseidon2Config::BabyBearD4Width16,
    p3_poseidon2_circuit_air::BabyBearD4Width16,
    4,
    16,
    8,
    8,
    enable_poseidon2_perm,
    p3_baby_bear::default_babybear_poseidon2_16,
    16,
    8
);

define_field_module!(
    goldilocks,
    p3_goldilocks::Goldilocks,
    p3_goldilocks::Poseidon2Goldilocks<8>,
    default_goldilocks_poseidon2_8,
    Poseidon2Config::GoldilocksD2Width8,
    p3_circuit::ops::GoldilocksD2Width8,
    2,
    8,
    4,
    4,
    enable_poseidon2_perm_width_8,
    default_goldilocks_poseidon2_8,
    8,
    4
);
