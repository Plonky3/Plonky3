//! Plonky3 circuit prover (PoC): generic over base field and permutation.
//!
//! Generics glossary used across this crate:
//! - `F`: Prover/verifier base field (BabyBear/KoalaBear/Goldilocks). PCS and FFTs operate over `F`.
//! - `P`: Cryptographic permutation over `F` used by hash/compress and the challenger.
//! - `EF`: Element field in circuit traces. Either `F` (base) or `BinomialExtensionField<F, D>`.
//! - `D`: Element-field extension degree. Must equal `EF::DIMENSION`. AIRs are parameterized as `<F, D>`.
//! - `CD`: FRI challenge field degree, independent of `D`.
//!
//! - Build a field-specific config via `config::{babybear_config, koalabear_config, goldilocks_config}`.
//! - Create a `BatchStarkProver` from that config.
//! - Generate traces from a `p3_circuit::Circuit` runner and prove/verify.
//!
//! Example (BabyBear):
//!
//! ```ignore
//! use p3_baby_bear::BabyBear;
//! use p3_circuit::builder::CircuitBuilder;
//! use p3_circuit_prover::config::babybear_config::build_standard_config_babybear;
//! use p3_circuit_prover::BatchStarkProver;
//!
//! let mut builder = CircuitBuilder::<BabyBear>::new();
//! let x = builder.public_input();
//! let y = builder.public_input();
//! let z = builder.add(x, y);
//! builder.assert_zero(builder.sub(z, builder.define_const(BabyBear::from_u64(3))));
//! let circuit = builder.build();
//! let mut runner = circuit.runner();
//! runner.set_public_inputs(&[BabyBear::from_u64(1), BabyBear::from_u64(2)]).unwrap();
//! let traces = runner.run().unwrap();
//! let cfg = build_standard_config_babybear();
//! let prover = BatchStarkProver::new(cfg);
//! let proof = prover.prove_all_tables(&traces).unwrap();
//! prover.verify_all_tables(&proof).unwrap();
//! ```
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod air;
pub mod batch_stark_prover;
pub mod common;
pub mod config;
pub mod constraint_profile;
pub mod field_params;
pub mod whir_native;
pub mod whir_native_sumcheck;

// Re-export main API
pub use batch_stark_prover::*;
pub use constraint_profile::ConstraintProfile;
pub use whir_native::*;
pub use whir_native_sumcheck::*;
