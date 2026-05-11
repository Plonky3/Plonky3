//! Common code for all recursive examples.

#![allow(unused_imports)]

pub use std::rc::Rc;
pub use std::sync::Arc;

pub use clap::{Args as ClapArgs, Parser, ValueEnum};
pub use p3_challenger::DuplexChallenger;
pub use p3_circuit::ops::{NpoTypeId, generate_poseidon2_trace, generate_recompose_trace};
pub use p3_circuit::{CircuitBuilder, CircuitRunner, NonPrimitiveOpId};
pub use p3_circuit_prover::batch_stark_prover::poseidon2_air_builders;
pub use p3_circuit_prover::common::{NpoPreprocessor, get_airs_and_degrees_with_prep};
pub use p3_circuit_prover::{
    BatchStarkProver, CircuitProverData, ConstraintProfile, Poseidon2Preprocessor, TablePacking,
};
pub use p3_commit::{ExtensionMmcs, Pcs};
pub use p3_dft::Radix2DitParallel;
pub use p3_field::extension::{BinomialExtensionField, QuinticTrinomialExtensionField};
pub use p3_field::{ExtensionField, Field, PrimeCharacteristicRing};
pub use p3_fri::{FriParameters, HidingFriPcs, TwoAdicFriPcs};
pub use p3_lookup::logup::LogUpGadget;
pub use p3_matrix::Matrix;
pub use p3_matrix::dense::RowMajorMatrix;
pub use p3_merkle_tree::MerkleTreeMmcs;
pub use p3_recursion::pcs::{
    HidingFriProofTargets, InputProofTargets, MerkleCapTargets, RecValMmcs,
    set_fri_mmcs_private_data, set_hiding_fri_mmcs_private_data,
};
pub use p3_recursion::traits::{RecursiveAir, RecursivePcs};
pub use p3_recursion::verifier::VerificationError;
pub use p3_recursion::{
    AggregationPrepCache, BatchOnly, BatchStarkVerifierInputsBuilder, FriRecursionBackend,
    FriRecursionBackendD5, FriRecursionConfig, FriVerifierParams, NextLayerPrepCache,
    Poseidon2Config, ProveNextLayerParams, RecursionInput, RecursionOutput,
    build_and_prove_aggregation_layer, build_and_prove_next_layer, build_next_layer_circuit,
    build_next_layer_prep, prove_next_layer, verify_batch_circuit,
};
pub use p3_symmetric::{PaddingFreeSponge, Permutation, TruncatedPermutation};
pub use p3_uni_stark::{StarkConfig, StarkGenericConfig, Val};
pub use rand::SeedableRng;
pub use rand::rngs::SmallRng;
pub use serde::Serialize;
pub use tracing::info;
pub use tracing_forest::ForestLayer;
pub use tracing_forest::util::LevelFilter;
pub use tracing_subscriber::layer::SubscriberExt;
pub use tracing_subscriber::util::SubscriberInitExt;
pub use tracing_subscriber::{EnvFilter, Registry};

pub fn init_logger() {
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    let _ = Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .try_init();
}

#[derive(Debug, Clone, Copy)]
pub struct FriParams {
    pub log_blowup: usize,
    pub max_log_arity: usize,
    pub cap_height: usize,
    pub log_final_poly_len: usize,
    pub commit_pow_bits: usize,
    pub query_pow_bits: usize,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum FieldOption {
    KoalaBear,
    BabyBear,
    Goldilocks,
}

/// Panics when `--quintic` is set with a base field that does not support the quintic challenge extension.
#[inline]
pub fn assert_quintic_field(field: FieldOption, quintic: bool) {
    if quintic && !matches!(field, FieldOption::KoalaBear) {
        panic!(
            "--quintic is only supported with --field koala-bear (got {:?})",
            field
        );
    }
}

pub fn default_goldilocks_poseidon2_8() -> p3_goldilocks::Poseidon2Goldilocks<8> {
    use rand::SeedableRng;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(1);
    p3_goldilocks::Poseidon2Goldilocks::<8>::new_from_rng_128(&mut rng)
}

/// Report the size of the serialized proof.
#[inline]
pub fn report_proof_size<S: Serialize>(proof: &S) {
    let proof_bytes = postcard::to_allocvec(proof).expect("Failed to serialize proof");
    println!("Proof size: {} bytes", proof_bytes.len());
}

/// Expands to all shared field-specific types and helper functions used by every
/// recursive example, **without** a surrounding `mod` block.
///
/// Each example's `define_field_module!` wraps a call to this macro inside its
/// own `mod $mod_name { use super::*; ... }` block and then appends the
/// example-specific `run` / `run_zk` functions.
///
/// Defines (inline, no module wrapper):
/// - Type aliases: `F`, `D`, `Challenge`, `Dft`, `Perm`, `MyHash`, `MyCompress`,
///   `MyMmcs`, `ChallengeMmcs`, `Challenger`, `MyPcs`, `MyConfig`, `InnerFri`,
///   `ConfigWithFriParams`
/// - Functions: `create_config`, `create_fri_verifier_params`, `config_with_fri_params`
/// - Trait impls: `Deref`, `StarkGenericConfig`, `FriRecursionConfig` for `ConfigWithFriParams`
///
/// Use `D` as the extension degree for `register_poseidon2_table::<D>`, `register_recompose_table::<D>`,
/// `poseidon2_air_builders::<_, D>()`, and `FriRecursionBackend::for_extension_degree::<D>(...)`.
#[macro_export]
macro_rules! define_field_module_types {
    (
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
        $backend_rate:expr,
        $enable_recompose_fn:ident
    ) => {
        pub type F = $field;
        pub const D: usize = $d;
        const WIDTH: usize = $width;
        const RATE: usize = $rate;
        const DIGEST_ELEMS: usize = $digest_elems;

        type Challenge = BinomialExtensionField<F, D>;
        type Dft = Radix2DitParallel<F>;
        type Perm = $perm;
        type MyHash = PaddingFreeSponge<Perm, WIDTH, RATE, DIGEST_ELEMS>;
        type MyCompress = TruncatedPermutation<Perm, 2, DIGEST_ELEMS, WIDTH>;
        type MyMmcs = MerkleTreeMmcs<
            <F as Field>::Packing,
            <F as Field>::Packing,
            MyHash,
            MyCompress,
            2,
            DIGEST_ELEMS,
        >;
        type ChallengeMmcs = ExtensionMmcs<F, Challenge, MyMmcs>;
        type Challenger = DuplexChallenger<F, Perm, WIDTH, RATE>;
        type MyPcs = TwoAdicFriPcs<F, Dft, MyMmcs, ChallengeMmcs>;
        type MyConfig = StarkConfig<MyPcs, Challenge, Challenger>;

        type InnerFri = p3_recursion::pcs::FriProofTargets<
            F,
            Challenge,
            p3_recursion::pcs::RecExtensionValMmcs<
                F,
                Challenge,
                DIGEST_ELEMS,
                RecValMmcs<F, DIGEST_ELEMS, MyHash, MyCompress>,
            >,
            InputProofTargets<F, Challenge, RecValMmcs<F, DIGEST_ELEMS, MyHash, MyCompress>>,
            p3_recursion::pcs::Witness<F>,
        >;

        #[allow(dead_code)]
        type MyPcsZk = HidingFriPcs<F, Dft, MyMmcs, ChallengeMmcs, SmallRng>;
        #[allow(dead_code)]
        type MyConfigZk = StarkConfig<MyPcsZk, Challenge, Challenger>;

        #[allow(dead_code)]
        type InnerFriZk = HidingFriProofTargets<
            F,
            Challenge,
            p3_recursion::pcs::RecExtensionValMmcs<
                F,
                Challenge,
                DIGEST_ELEMS,
                RecValMmcs<F, DIGEST_ELEMS, MyHash, MyCompress>,
            >,
            InputProofTargets<F, Challenge, RecValMmcs<F, DIGEST_ELEMS, MyHash, MyCompress>>,
            p3_recursion::pcs::Witness<F>,
        >;

        #[derive(Clone)]
        struct ConfigWithFriParams {
            config: Arc<MyConfig>,
            fri_verifier_params: FriVerifierParams,
            disable_recompose_npo: bool,
        }

        #[allow(dead_code)]
        #[derive(Clone)]
        struct ConfigWithFriParamsZk {
            config: Arc<MyConfigZk>,
            fri_verifier_params: FriVerifierParams,
            disable_recompose_npo: bool,
        }

        impl core::ops::Deref for ConfigWithFriParams {
            type Target = MyConfig;
            fn deref(&self) -> &MyConfig {
                &self.config
            }
        }

        impl core::ops::Deref for ConfigWithFriParamsZk {
            type Target = MyConfigZk;
            fn deref(&self) -> &MyConfigZk {
                &self.config
            }
        }

        impl StarkGenericConfig for ConfigWithFriParams {
            type Challenge = Challenge;
            type Challenger = Challenger;
            type Pcs = MyPcs;
            fn pcs(&self) -> &MyPcs {
                self.config.pcs()
            }
            fn initialise_challenger(&self) -> Challenger {
                self.config.initialise_challenger()
            }
        }

        impl StarkGenericConfig for ConfigWithFriParamsZk {
            type Challenge = Challenge;
            type Challenger = Challenger;
            type Pcs = MyPcsZk;
            fn pcs(&self) -> &MyPcsZk {
                self.config.pcs()
            }
            fn initialise_challenger(&self) -> Challenger {
                self.config.initialise_challenger()
            }
        }

        impl FriRecursionConfig for ConfigWithFriParams
        where
            MyPcs: RecursivePcs<
                    ConfigWithFriParams,
                    InputProofTargets<
                        F,
                        Challenge,
                        RecValMmcs<F, DIGEST_ELEMS, MyHash, MyCompress>,
                    >,
                    InnerFri,
                    MerkleCapTargets<F, DIGEST_ELEMS>,
                    <MyPcs as Pcs<Challenge, Challenger>>::Domain,
                >,
        {
            type Commitment = MerkleCapTargets<F, DIGEST_ELEMS>;
            type InputProof =
                InputProofTargets<F, Challenge, RecValMmcs<F, DIGEST_ELEMS, MyHash, MyCompress>>;
            type OpeningProof = InnerFri;
            type RawOpeningProof = <MyPcs as Pcs<Challenge, Challenger>>::Proof;
            const DIGEST_ELEMS: usize = $digest_elems;

            fn with_fri_opening_proof<'a, A, R>(
                prev: &RecursionInput<'a, Self, A>,
                f: impl FnOnce(&Self::RawOpeningProof) -> R,
            ) -> R
            where
                A: RecursiveAir<Val<Self>, Self::Challenge, LogUpGadget>,
            {
                match prev {
                    RecursionInput::UniStark { proof, .. } => f(&proof.opening_proof),
                    RecursionInput::BatchStark { proof, .. } => f(&proof.proof.opening_proof),
                }
            }

            fn prepare_circuit_for_verification(
                &self,
                circuit: &mut CircuitBuilder<Challenge>,
            ) -> Result<(), VerificationError> {
                let perm = $default_perm_circuit();
                circuit.$enable_poseidon2_fn::<$poseidon2_circuit_config, _>(
                    generate_poseidon2_trace::<Challenge, $poseidon2_circuit_config>,
                    perm,
                );
                if self.disable_recompose_npo {
                    circuit.noop_enable_recompose::<F>(generate_recompose_trace::<F, Challenge>);
                } else {
                    circuit.$enable_recompose_fn::<F>(generate_recompose_trace::<F, Challenge>);
                }
                if <$poseidon2_circuit_config as p3_circuit::ops::Poseidon2Params>::D == 1
                    && <Challenge as ::p3_field::BasedVectorSpace<F>>::DIMENSION > 1
                {
                    circuit.set_recompose_coeff_ctl_for_decompose_links(true);
                }
                Ok(())
            }

            fn pcs_verifier_params(
                &self,
            ) -> &<MyPcs as RecursivePcs<
                ConfigWithFriParams,
                InputProofTargets<F, Challenge, RecValMmcs<F, DIGEST_ELEMS, MyHash, MyCompress>>,
                InnerFri,
                MerkleCapTargets<F, DIGEST_ELEMS>,
                <MyPcs as Pcs<Challenge, Challenger>>::Domain,
            >>::VerifierParams {
                &self.fri_verifier_params
            }

            fn set_fri_private_data(
                runner: &mut CircuitRunner<'_, Challenge>,
                op_ids: &[NonPrimitiveOpId],
                opening_proof: &Self::RawOpeningProof,
            ) -> Result<(), &'static str> {
                set_fri_mmcs_private_data::<
                    F,
                    Challenge,
                    ChallengeMmcs,
                    MyMmcs,
                    MyHash,
                    MyCompress,
                    DIGEST_ELEMS,
                >(runner, op_ids, opening_proof)
            }
        }

        impl FriRecursionConfig for ConfigWithFriParamsZk
        where
            MyPcsZk: RecursivePcs<
                    ConfigWithFriParamsZk,
                    InputProofTargets<
                        F,
                        Challenge,
                        RecValMmcs<F, DIGEST_ELEMS, MyHash, MyCompress>,
                    >,
                    InnerFriZk,
                    MerkleCapTargets<F, DIGEST_ELEMS>,
                    <MyPcsZk as Pcs<Challenge, Challenger>>::Domain,
                >,
        {
            type Commitment = MerkleCapTargets<F, DIGEST_ELEMS>;
            type InputProof =
                InputProofTargets<F, Challenge, RecValMmcs<F, DIGEST_ELEMS, MyHash, MyCompress>>;
            type OpeningProof = InnerFriZk;
            type RawOpeningProof = <MyPcsZk as Pcs<Challenge, Challenger>>::Proof;
            const DIGEST_ELEMS: usize = $digest_elems;

            fn with_fri_opening_proof<'a, A, R>(
                prev: &RecursionInput<'a, Self, A>,
                f: impl FnOnce(&Self::RawOpeningProof) -> R,
            ) -> R
            where
                A: RecursiveAir<Val<Self>, Self::Challenge, LogUpGadget>,
            {
                match prev {
                    RecursionInput::UniStark { proof, .. } => f(&proof.opening_proof),
                    RecursionInput::BatchStark { proof, .. } => f(&proof.proof.opening_proof),
                }
            }

            fn prepare_circuit_for_verification(
                &self,
                circuit: &mut CircuitBuilder<Challenge>,
            ) -> Result<(), VerificationError> {
                let perm = $default_perm_circuit();
                circuit.$enable_poseidon2_fn::<$poseidon2_circuit_config, _>(
                    generate_poseidon2_trace::<Challenge, $poseidon2_circuit_config>,
                    perm,
                );
                if self.disable_recompose_npo {
                    circuit.noop_enable_recompose::<F>(generate_recompose_trace::<F, Challenge>);
                } else {
                    circuit.$enable_recompose_fn::<F>(generate_recompose_trace::<F, Challenge>);
                }
                if <$poseidon2_circuit_config as p3_circuit::ops::Poseidon2Params>::D == 1
                    && <Challenge as ::p3_field::BasedVectorSpace<F>>::DIMENSION > 1
                {
                    circuit.set_recompose_coeff_ctl_for_decompose_links(true);
                }
                Ok(())
            }

            fn pcs_verifier_params(
                &self,
            ) -> &<MyPcsZk as RecursivePcs<
                ConfigWithFriParamsZk,
                InputProofTargets<F, Challenge, RecValMmcs<F, DIGEST_ELEMS, MyHash, MyCompress>>,
                InnerFriZk,
                MerkleCapTargets<F, DIGEST_ELEMS>,
                <MyPcsZk as Pcs<Challenge, Challenger>>::Domain,
            >>::VerifierParams {
                &self.fri_verifier_params
            }

            fn set_fri_private_data(
                runner: &mut CircuitRunner<'_, Challenge>,
                op_ids: &[NonPrimitiveOpId],
                opening_proof: &Self::RawOpeningProof,
            ) -> Result<(), &'static str> {
                set_hiding_fri_mmcs_private_data::<
                    F,
                    Challenge,
                    ChallengeMmcs,
                    MyMmcs,
                    MyHash,
                    MyCompress,
                    DIGEST_ELEMS,
                >(runner, op_ids, opening_proof)
            }
        }

        fn create_config(fp: &FriParams, security_level: usize) -> MyConfig {
            let perm = $default_perm();
            let hash = MyHash::new(perm.clone());
            let compress = MyCompress::new(perm.clone());
            let val_mmcs = MyMmcs::new(hash, compress, fp.cap_height);
            let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
            let dft = Dft::default();

            let num_queries = (security_level - fp.query_pow_bits) / fp.log_blowup;

            let fri_params = FriParameters {
                max_log_arity: fp.max_log_arity,
                log_blowup: fp.log_blowup,
                log_final_poly_len: fp.log_final_poly_len,
                num_queries,
                commit_proof_of_work_bits: fp.commit_pow_bits,
                query_proof_of_work_bits: fp.query_pow_bits,
                mmcs: challenge_mmcs,
            };
            let pcs = MyPcs::new(dft, val_mmcs, fri_params);
            let challenger = Challenger::new(perm);
            MyConfig::new(pcs, challenger)
        }

        const fn create_fri_verifier_params(fp: &FriParams) -> FriVerifierParams {
            FriVerifierParams::with_mmcs(
                fp.log_blowup,
                fp.log_final_poly_len,
                fp.commit_pow_bits,
                fp.query_pow_bits,
                $poseidon2_config,
            )
        }

        fn config_with_fri_params(
            fp: &FriParams,
            security_level: usize,
            disable_recompose_npo: bool,
        ) -> ConfigWithFriParams {
            ConfigWithFriParams {
                config: Arc::new(create_config(fp, security_level)),
                fri_verifier_params: create_fri_verifier_params(fp),
                disable_recompose_npo,
            }
        }

        #[allow(dead_code)]
        fn create_config_zk(fp: &FriParams, security_level: usize, rng_seed: u64) -> MyConfigZk {
            let perm = $default_perm();
            let hash = MyHash::new(perm.clone());
            let compress = MyCompress::new(perm.clone());
            let val_mmcs = MyMmcs::new(hash, compress, fp.cap_height);
            let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
            let dft = Dft::default();

            let num_queries = (security_level - fp.query_pow_bits) / fp.log_blowup;

            let fri_params = FriParameters {
                max_log_arity: fp.max_log_arity,
                log_blowup: fp.log_blowup,
                log_final_poly_len: fp.log_final_poly_len,
                num_queries,
                commit_proof_of_work_bits: fp.commit_pow_bits,
                query_proof_of_work_bits: fp.query_pow_bits,
                mmcs: challenge_mmcs,
            };
            let pcs = MyPcsZk::new(
                dft,
                val_mmcs,
                fri_params,
                2,
                SmallRng::seed_from_u64(rng_seed),
            );
            let challenger = Challenger::new(perm);
            MyConfigZk::new(pcs, challenger)
        }

        #[allow(dead_code)]
        fn config_with_fri_params_zk(
            fp: &FriParams,
            security_level: usize,
            disable_recompose_npo: bool,
            rng_seed: u64,
        ) -> ConfigWithFriParamsZk {
            ConfigWithFriParamsZk {
                config: Arc::new(create_config_zk(fp, security_level, rng_seed)),
                fri_verifier_params: create_fri_verifier_params(fp),
                disable_recompose_npo,
            }
        }
    };
}

/// Variant of [`define_field_module_types`] for KoalaBear quintic extension (D=5).
///
/// Key differences from the standard macro:
/// - `Challenge = QuinticTrinomialExtensionField<F>` instead of `BinomialExtensionField<F, D>`.
/// - `prepare_circuit_for_verification` uses `enable_poseidon2_perm_base` with a
///   `$perm_circuit_constructor` closure that must produce a
///   `impl Permutation<[Challenge; WIDTH]>`.
///
/// Recursive examples usually call `define_quintic_poseidon_perm_lift_and_types!` instead, which
/// wires in [`p3_test_utils::LiftPermToQuintic`] and then invokes this macro.
#[macro_export]
macro_rules! define_field_module_types_quintic {
    (
        $field:ty,
        $perm:ty,
        $default_perm:path,
        $poseidon2_config:expr,
        $poseidon2_circuit_config:ty,
        $width:expr,
        $rate:expr,
        $digest_elems:expr,
        $perm_circuit_constructor:expr,
        $backend_width:expr,
        $backend_rate:expr
    ) => {
        pub type F = $field;
        pub const D: usize = 5;
        const WIDTH: usize = $width;
        const RATE: usize = $rate;
        const DIGEST_ELEMS: usize = $digest_elems;

        type Challenge = QuinticTrinomialExtensionField<F>;
        type Dft = Radix2DitParallel<F>;
        type Perm = $perm;
        type MyHash = PaddingFreeSponge<Perm, WIDTH, RATE, DIGEST_ELEMS>;
        type MyCompress = TruncatedPermutation<Perm, 2, DIGEST_ELEMS, WIDTH>;
        type MyMmcs = MerkleTreeMmcs<
            <F as Field>::Packing,
            <F as Field>::Packing,
            MyHash,
            MyCompress,
            2,
            DIGEST_ELEMS,
        >;
        type ChallengeMmcs = ExtensionMmcs<F, Challenge, MyMmcs>;
        type Challenger = DuplexChallenger<F, Perm, WIDTH, RATE>;
        type MyPcs = TwoAdicFriPcs<F, Dft, MyMmcs, ChallengeMmcs>;
        type MyConfig = StarkConfig<MyPcs, Challenge, Challenger>;

        type InnerFri = p3_recursion::pcs::FriProofTargets<
            F,
            Challenge,
            p3_recursion::pcs::RecExtensionValMmcs<
                F,
                Challenge,
                DIGEST_ELEMS,
                RecValMmcs<F, DIGEST_ELEMS, MyHash, MyCompress>,
            >,
            InputProofTargets<F, Challenge, RecValMmcs<F, DIGEST_ELEMS, MyHash, MyCompress>>,
            p3_recursion::pcs::Witness<F>,
        >;

        #[derive(Clone)]
        struct ConfigWithFriParams {
            config: Arc<MyConfig>,
            fri_verifier_params: FriVerifierParams,
            disable_recompose_npo: bool,
        }

        impl core::ops::Deref for ConfigWithFriParams {
            type Target = MyConfig;
            fn deref(&self) -> &MyConfig {
                &self.config
            }
        }

        impl StarkGenericConfig for ConfigWithFriParams {
            type Challenge = Challenge;
            type Challenger = Challenger;
            type Pcs = MyPcs;
            fn pcs(&self) -> &MyPcs {
                self.config.pcs()
            }
            fn initialise_challenger(&self) -> Challenger {
                self.config.initialise_challenger()
            }
        }

        impl FriRecursionConfig for ConfigWithFriParams
        where
            MyPcs: RecursivePcs<
                    ConfigWithFriParams,
                    InputProofTargets<
                        F,
                        Challenge,
                        RecValMmcs<F, DIGEST_ELEMS, MyHash, MyCompress>,
                    >,
                    InnerFri,
                    MerkleCapTargets<F, DIGEST_ELEMS>,
                    <MyPcs as Pcs<Challenge, Challenger>>::Domain,
                >,
        {
            type Commitment = MerkleCapTargets<F, DIGEST_ELEMS>;
            type InputProof =
                InputProofTargets<F, Challenge, RecValMmcs<F, DIGEST_ELEMS, MyHash, MyCompress>>;
            type OpeningProof = InnerFri;
            type RawOpeningProof = <MyPcs as Pcs<Challenge, Challenger>>::Proof;
            const DIGEST_ELEMS: usize = $digest_elems;

            fn with_fri_opening_proof<'a, A, R>(
                prev: &RecursionInput<'a, Self, A>,
                f: impl FnOnce(&Self::RawOpeningProof) -> R,
            ) -> R
            where
                A: RecursiveAir<Val<Self>, Self::Challenge, LogUpGadget>,
            {
                match prev {
                    RecursionInput::UniStark { proof, .. } => f(&proof.opening_proof),
                    RecursionInput::BatchStark { proof, .. } => f(&proof.proof.opening_proof),
                }
            }

            fn prepare_circuit_for_verification(
                &self,
                circuit: &mut CircuitBuilder<Challenge>,
            ) -> Result<(), VerificationError> {
                let perm = ($perm_circuit_constructor)();
                circuit.enable_poseidon2_perm_base::<$poseidon2_circuit_config, _>(
                    generate_poseidon2_trace::<Challenge, $poseidon2_circuit_config>,
                    perm,
                );
                if self.disable_recompose_npo {
                    circuit.noop_enable_recompose::<F>(generate_recompose_trace::<F, Challenge>);
                } else {
                    circuit.enable_recompose::<F>(generate_recompose_trace::<F, Challenge>);
                }
                if <$poseidon2_circuit_config as p3_circuit::ops::Poseidon2Params>::D == 1
                    && <Challenge as ::p3_field::BasedVectorSpace<F>>::DIMENSION > 1
                {
                    circuit.set_recompose_coeff_ctl_for_decompose_links(true);
                }
                Ok(())
            }

            fn pcs_verifier_params(
                &self,
            ) -> &<MyPcs as RecursivePcs<
                ConfigWithFriParams,
                InputProofTargets<F, Challenge, RecValMmcs<F, DIGEST_ELEMS, MyHash, MyCompress>>,
                InnerFri,
                MerkleCapTargets<F, DIGEST_ELEMS>,
                <MyPcs as Pcs<Challenge, Challenger>>::Domain,
            >>::VerifierParams {
                &self.fri_verifier_params
            }

            fn set_fri_private_data(
                runner: &mut CircuitRunner<'_, Challenge>,
                op_ids: &[NonPrimitiveOpId],
                opening_proof: &Self::RawOpeningProof,
            ) -> Result<(), &'static str> {
                set_fri_mmcs_private_data::<
                    F,
                    Challenge,
                    ChallengeMmcs,
                    MyMmcs,
                    MyHash,
                    MyCompress,
                    DIGEST_ELEMS,
                >(runner, op_ids, opening_proof)
            }
        }

        fn create_config(fp: &FriParams, security_level: usize) -> MyConfig {
            let perm = $default_perm();
            let hash = MyHash::new(perm.clone());
            let compress = MyCompress::new(perm.clone());
            let val_mmcs = MyMmcs::new(hash, compress, fp.cap_height);
            let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
            let dft = Dft::default();

            let num_queries = (security_level - fp.query_pow_bits) / fp.log_blowup;

            let fri_params = FriParameters {
                max_log_arity: fp.max_log_arity,
                log_blowup: fp.log_blowup,
                log_final_poly_len: fp.log_final_poly_len,
                num_queries,
                commit_proof_of_work_bits: fp.commit_pow_bits,
                query_proof_of_work_bits: fp.query_pow_bits,
                mmcs: challenge_mmcs,
            };
            let pcs = MyPcs::new(dft, val_mmcs, fri_params);
            let challenger = Challenger::new(perm);
            MyConfig::new(pcs, challenger)
        }

        const fn create_fri_verifier_params(fp: &FriParams) -> FriVerifierParams {
            FriVerifierParams::with_mmcs(
                fp.log_blowup,
                fp.log_final_poly_len,
                fp.commit_pow_bits,
                fp.query_pow_bits,
                $poseidon2_config,
            )
        }

        fn config_with_fri_params(
            fp: &FriParams,
            security_level: usize,
            disable_recompose_npo: bool,
        ) -> ConfigWithFriParams {
            ConfigWithFriParams {
                config: Arc::new(create_config(fp, security_level)),
                fri_verifier_params: create_fri_verifier_params(fp),
                disable_recompose_npo,
            }
        }
    };
}

/// Expands [`define_field_module_types_quintic!`] with a circuit permutation constructor that uses
/// [`p3_test_utils::LiftPermToQuintic`].
#[macro_export]
macro_rules! define_quintic_poseidon_perm_lift_and_types {
    (
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
        define_field_module_types_quintic!(
            $field,
            $perm,
            $default_perm,
            $poseidon2_config,
            $poseidon2_circuit_config,
            $width,
            $rate,
            $digest_elems,
            || ::p3_test_utils::LiftPermToQuintic::<$field, $perm, $width>::new($default_perm()),
            $backend_width,
            $backend_rate
        );
    };
}
